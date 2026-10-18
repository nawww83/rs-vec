#Requires -Version 5.1
<#
.SYNOPSIS
    Собирает и запускает независимый тестовый набор для rs_vec.hpp.

.DESCRIPTION
    Компилятор ищется в PATH, затем в типовых местах установки Qt Tools /
    MSVC Build Tools / LLVM. Собранные файлы и логи кладутся в <repo>/build
    (каталог в .gitignore). Дополнительно собирается и запускается пример из
    README, чтобы документация не могла разойтись с реальным API.

.EXAMPLE
    .\run_tests.ps1

.EXAMPLE
    .\run_tests.ps1 -Compiler clang -Sanitize ubsan
#>
[CmdletBinding()]
param(
    [ValidateSet('auto', 'gcc', 'clang', 'msvc', 'clang-cl')]
    [string]$Compiler = 'auto',

    [ValidateSet('none', 'asan', 'ubsan')]
    [string]$Sanitize = 'none',

    # Явный путь к компилятору, если автопоиск не подходит.
    [string]$CompilerPath = ''
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$buildDir = Join-Path $repoRoot 'build'
$testSrc = Join-Path $PSScriptRoot 'rs_tests.cpp'
$readmeSrc = Join-Path $PSScriptRoot 'readme_example.cpp'

if (-not (Test-Path $buildDir)) { New-Item -ItemType Directory -Path $buildDir | Out-Null }

function Find-FirstExisting([string[]]$Paths) {
    foreach ($p in $Paths) {
        if ($p -and (Test-Path -Path $p -PathType Leaf)) { return (Get-Item -Path $p).FullName }
    }
    return $null
}

function Get-CommandSource([string]$Name) {
    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    return $null
}

function Resolve-Compiler([string]$Kind) {
    switch ($Kind) {
        'gcc' {
            return Find-FirstExisting @(
                (Get-CommandSource 'g++'),
                'C:\Qt\Tools\mingw1120_64\bin\g++.exe',
                'C:\Qt\Tools\mingw810_64\bin\g++.exe',
                'C:\msys64\mingw64\bin\g++.exe',
                'C:\MinGW\bin\g++.exe'
            )
        }
        'clang' {
            return Find-FirstExisting @(
                (Get-CommandSource 'clang++'),
                'C:\Qt\Tools\llvm-mingw1706_64\bin\clang++.exe',
                'C:\Program Files\LLVM\bin\clang++.exe'
            )
        }
        'msvc' {
            return Find-FirstExisting @(
                (Get-CommandSource 'cl'),
                "$env:ProgramFiles\Microsoft Visual Studio\*\*\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe",
                "${env:ProgramFiles(x86)}\Microsoft Visual Studio\*\*\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe"
            )
        }
        'clang-cl' {
            return Find-FirstExisting @(
                (Get-CommandSource 'clang-cl'),
                'C:\Program Files\LLVM\bin\clang-cl.exe',
                'C:\Qt\Tools\llvm-mingw1706_64\bin\clang-cl.exe'
            )
        }
    }
    return $null
}

function Resolve-VcVars() {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path -LiteralPath $vswhere) {
        $root = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
        if ($root) {
            $bat = Join-Path $root 'VC\Auxiliary\Build\vcvars64.bat'
            if (Test-Path -LiteralPath $bat) { return $bat }
        }
    }
    $found = Get-ChildItem -Path "${env:ProgramFiles(x86)}\Microsoft Visual Studio",
        "$env:ProgramFiles\Microsoft Visual Studio" -Filter 'vcvars64.bat' -Recurse -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($found) { return $found.FullName }
    return $null
}

# --------------------------------------------------------------------------
# Выбор компилятора
# --------------------------------------------------------------------------
$order = if ($Compiler -eq 'auto') { @('gcc', 'clang', 'msvc') } else { @($Compiler) }
$kind = $null
$exe = $null
foreach ($candidate in $order) {
    $resolved = Resolve-Compiler $candidate
    if ($resolved) { $exe = $resolved; $kind = $candidate; break }
}
if ($CompilerPath) { $exe = $CompilerPath; $kind = if ($kind) { $kind } else { 'custom' } }
if (-not $exe) {
    Write-Error "Не найден компилятор ($($order -join ', ')). Укажите -CompilerPath."
}

if ($Sanitize -ne 'none' -and $kind -eq 'msvc') {
    Write-Error "Sanitizer'ы доступны только для gcc/clang/clang-cl, а не для cl.exe. Используйте -Compiler clang-cl."
}

$tag = "${kind}_${Sanitize}"
$exePath = Join-Path $buildDir "rs_tests_$tag.exe"
$logPath = Join-Path $buildDir "run_tests_$tag.log"
$buildLog = Join-Path $buildDir "build_tests_$tag.log"
$objDir = Join-Path $buildDir "obj_$tag"

Write-Host "Компилятор : $kind ($exe)"
Write-Host "Sanitizer  : $Sanitize"

# --------------------------------------------------------------------------
# Флаги
# --------------------------------------------------------------------------
$sanFlags = switch ($Sanitize) {
    'asan' { @('-fsanitize=address', '-fno-omit-frame-pointer') }
    'ubsan' { @('-fsanitize=undefined', '-fno-sanitize-recover=all') }
    default { @() }
}
# clang-cl принимает те же -f флаги, но O-флаги у него в стиле MSVC.
$sanFlagsCl = switch ($Sanitize) {
    'asan' { @('/fsanitize=address') }
    'ubsan' { @('-fsanitize=undefined', '-fno-sanitize-recover=all') }
    default { @() }
}
$optFlags = if ($Sanitize -eq 'none') { @('-O3') } else { @('-O1', '-g') }
# -I<repo> моделирует установленный заголовок: в README пример пишет
# #include "rs_vec.hpp" без путей.
$gnuFlags = @('-mavx2', '-std=c++20', '-Wall', '-Wextra', '-Wpedantic') + $optFlags +
    $sanFlags + @('-I', $repoRoot)

$msvcAbi = $kind -in @('msvc', 'clang-cl')

function Invoke-Compile([string]$Source, [string]$Output, [string]$BuildLog) {
    if ($msvcAbi) {
        if (-not (Test-Path $objDir)) { New-Item -ItemType Directory -Path $objDir | Out-Null }
        $bat = Join-Path $buildDir "compile_$(Split-Path -Leaf $Output).bat"
        $vcvars = Resolve-VcVars
        if (-not $vcvars) { Write-Error 'Не найден vcvars64.bat для MSVC.' }
        $opt = if ($Sanitize -eq 'none') { '/O2' } else { '/O1' }
        $flags = if ($kind -eq 'clang-cl') { $sanFlagsCl } else { @() }
        $cmd = @('cl', '/nologo', $opt, '/std:c++20', '/arch:AVX2', '/W4', '/EHsc', '/utf-8')
        if ($kind -eq 'clang-cl') { $cmd = @("`"$exe`"", '/nologo', $opt, '/std:c++20', '/arch:AVX2', '/W4', '/EHsc', '/utf-8') }
        $cmd += $flags
        $cmd += @("/I`"$repoRoot`"", "/Fe:`"$Output`"", "/Fo:`"$objDir\\`"", "`"$Source`"")
        @(
            '@echo off',
            "call `"$vcvars`" >nul 2>&1",
            "cd /d `"$repoRoot`"",
            ($cmd -join ' ')
        ) | Set-Content -Path $bat -Encoding ASCII
        & cmd.exe /c "`"$bat`" > `"$BuildLog`" 2>&1"
        return $LASTEXITCODE
    }

    $env:PATH = "$(Split-Path -Parent $exe);$env:PATH"
    $argv = @($gnuFlags) + @($Source, '-o', $Output)
    $quoted = ($argv | ForEach-Object { '"' + $_ + '"' }) -join ' '
    & cmd.exe /c "`"$exe`" $quoted > `"$BuildLog`" 2>&1"
    return $LASTEXITCODE
}

# --------------------------------------------------------------------------
# Основной тестовый набор
# --------------------------------------------------------------------------
Write-Host ''
Write-Host '--- сборка тестового набора --------------------------------'
$buildExit = Invoke-Compile -Source $testSrc -Output $exePath -BuildLog $buildLog
Get-Content -LiteralPath $buildLog -ErrorAction SilentlyContinue | Where-Object { $_ -ne '' }
if ($buildExit -ne 0) {
    Write-Error "Сборка не удалась (exit $buildExit). См. $buildLog"
}

if ($Sanitize -ne 'none') {
    # Для MSVC-ABI нужен каталог с clang_rt.*-x86_64.dll
    $rt = Get-ChildItem 'C:\Program Files\LLVM\lib\clang' -Recurse -Filter 'clang_rt.*-x86_64.dll' -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if (-not $rt) {
        $llvmRoot = Split-Path -Parent (Split-Path -Parent $exe)
        $rt = Get-ChildItem (Join-Path $llvmRoot 'lib\clang') -Recurse -Filter 'clang_rt.*-x86_64.dll' -ErrorAction SilentlyContinue |
            Select-Object -First 1
    }
    if ($rt) { $env:PATH = "$($rt.DirectoryName);$env:PATH" }
}

Write-Host ''
Write-Host '--- запуск тестов ------------------------------------------'
& $exePath 2>&1 | Tee-Object -FilePath $logPath
$runExit = $LASTEXITCODE

# --------------------------------------------------------------------------
# Пример из README
# --------------------------------------------------------------------------
Write-Host ''
Write-Host '--- пример из README ---------------------------------------'
$readmeExe = Join-Path $buildDir "readme_example_$tag.exe"
$readmeLog = Join-Path $buildDir "build_readme_example_$tag.log"
$readmeExit = Invoke-Compile -Source $readmeSrc -Output $readmeExe -BuildLog $readmeLog
Get-Content -LiteralPath $readmeLog -ErrorAction SilentlyContinue | Where-Object { $_ -ne '' }
if ($readmeExit -ne 0) {
    Write-Host "Сборка примера из README не удалась (exit $readmeExit)."
    $runExit = 1
} else {
    & $readmeExe
    if ($LASTEXITCODE -ne 0) { $runExit = 1 }
}

Write-Host ''
if ($runExit -eq 0) {
    Write-Host "РЕЗУЛЬТАТ: успешно ($tag)"
} else {
    Write-Host "РЕЗУЛЬТАТ: ЕСТЬ ПАДЕНИЯ ($tag) - см. $logPath"
}
exit $runExit
