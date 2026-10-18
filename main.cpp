// Бенчмарк и демонстрация библиотеки rs_vec.hpp.
//
// Сборка:
//   g++ -O3 -mavx2 -std=c++20 main.cpp -o rs_benchmark
#include "rs_vec.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

void run_test_t3() {
    std::cout << "\n=== Тестирование и бенчмарк RSStreamCodec (t = 3) ===\n";

    constexpr size_t K_CHUNK = 223;
    constexpr int T_ERRORS = 3;
    rs_gf256::RSStreamCodec<T_ERRORS, K_CHUNK> codec_t3;

    // Генерируем массив объемом ~10 МБ для точного замера пропускной способности
    constexpr size_t TEST_SIZE = 10 * 1024 * 1024; // 10 485 760 байт
    std::vector<uint8_t> original_data(TEST_SIZE);
    std::mt19937 rng(777);
    for (size_t i = 0; i < TEST_SIZE; ++i) {
        original_data[i] = static_cast<uint8_t>(rng());
    }

    std::cout << "Исходный объем: " << (double(TEST_SIZE) / (1024.0 * 1024.0)) << " МБ (" 
              << TEST_SIZE << " байт)\n";

    // -------------------------------------------------------------
    // 1. Измерение скорости кодирования
    // -------------------------------------------------------------
    auto enc_start = std::chrono::high_resolution_clock::now();
    std::vector<uint8_t> encoded_stream = codec_t3.encode(original_data);
    auto enc_end = std::chrono::high_resolution_clock::now();

    double enc_time_sec = std::chrono::duration<double>(enc_end - enc_start).count();
    double enc_mb = (double(TEST_SIZE) / (1024.0 * 1024.0)) / enc_time_sec;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "[КОДЕР] Время: " << (enc_time_sec * 1000.0) << " мс | Скорость: " 
              << enc_mb << " МБ/с (" << (enc_mb / 1024.0) << " ГБ/с)\n";
    std::cout << "Размер закодированного потока: " 
              << (double(encoded_stream.size()) / (1024.0 * 1024.0)) << " МБ\n";

    // -------------------------------------------------------------
    // 2. Внесение ошибок (10% пачек повреждено + пакетный сбой)
    // -------------------------------------------------------------
    constexpr size_t TOTAL_CODEWORD_LEN = K_CHUNK + 2 * T_ERRORS; // 223 + 6 = 229
    constexpr size_t BATCH_BYTES = TOTAL_CODEWORD_LEN * 32;
    size_t total_batches = encoded_stream.size() / BATCH_BYTES;
    size_t damaged_lanes = 0;

    for (size_t b = 0; b < total_batches; ++b) {
        if (rng() % 10 == 0) { // Портим каждую 10-ю пачку
            uint8_t* batch_ptr = encoded_stream.data() + b * BATCH_BYTES;
            for (int lane = 0; lane < 32; ++lane) {
                damaged_lanes++;
                int errs = (rng() % T_ERRORS) + 1; // 1..3 ошибок на дорожку
                for (int e = 0; e < errs; ++e) {
                    size_t step = rng() % TOTAL_CODEWORD_LEN;
                    batch_ptr[step * 32 + lane] ^= static_cast<uint8_t>((rng() % 255) + 1);
                }
            }
        }
    }

    // Длинный пакетный сбой: 90 байт подряд (~3 байта на дорожку)
    if (encoded_stream.size() > 5000) {
        size_t burst_pos = 4096;
        for (size_t i = 0; i < 90; ++i) {
            encoded_stream[burst_pos + i] ^= 0x7E;
        }
    }
    std::cout << "Внесены повреждения: сбои добавлены в ~10% блоков (" 
              << damaged_lanes << " кодовых слов) + 90 байт burst подряд.\n";

    // -------------------------------------------------------------
    // 3. Измерение скорости декодирования с коррекцией ошибок
    // -------------------------------------------------------------
    auto dec_start = std::chrono::high_resolution_clock::now();
    std::vector<uint8_t> restored_data;
    bool success = codec_t3.decode(encoded_stream, restored_data);
    auto dec_end = std::chrono::high_resolution_clock::now();

    double dec_time_sec = std::chrono::duration<double>(dec_end - dec_start).count();
    double dec_mb = (double(encoded_stream.size()) / (1024.0 * 1024.0)) / dec_time_sec;

    std::cout << "[ДЕКОДЕР] Время: " << (dec_time_sec * 1000.0) << " мс | Скорость: " 
              << dec_mb << " МБ/с (" << (dec_mb / 1024.0) << " ГБ/с)\n";

    // -------------------------------------------------------------
    // 4. Проверка совпадения данных
    // -------------------------------------------------------------
    if (success && restored_data == original_data) {
        std::cout << "[STATUS] [SUCCESS] Все данные при t = 3 полностью восстановлены!\n";
    } else {
        std::cout << "[STATUS] [FAIL] Ошибка восстановления данных при t = 3!\n";
        std::exit(1);
    }
}

// ---------------------------------------------------------------------------
// Демонстрация границы радиуса t и пользы внешнего CRC.
//
// Повреждения сверх ёмкости (t + 1 ошибка в дорожке) иногда приводят к тому,
// что принятое слово оказывается ближе к ДРУГОМУ кодовому слову. Это
// фундаментальное свойство любого корректирующего кода; детектировать такую
// подмену позволяет только внешний контроль целостности.
// ---------------------------------------------------------------------------
void run_crc_demo() {
    constexpr int T = 3;
    constexpr size_t K = 223;
    constexpr size_t TOTAL_CODEWORD_LEN = K + 2 * T;
    constexpr size_t CW = TOTAL_CODEWORD_LEN * 32;
    constexpr int TRIALS = 200;

    rs_gf256::RSStreamCodec<T, K, false> plain;
    rs_gf256::RSStreamCodec<T, K, true> with_crc;

    std::mt19937 rng(2024);
    size_t plain_silent = 0, crc_silent = 0, crc_detected = 0;

    std::cout << "\n=== Граница радиуса t и внешний CRC (" << TRIALS
              << " опытов, t + 1 = " << (T + 1) << " ошибок в одной дорожке) ===\n";

    for (int it = 0; it < TRIALS; ++it) {
        const size_t size = 5000 + rng() % 20000;
        std::vector<uint8_t> data(size);
        for (auto& b : data) b = static_cast<uint8_t>(rng());

        auto enc_plain = plain.encode(data);
        auto enc_crc = with_crc.encode(data);

        const size_t b = rng() % (enc_plain.size() / CW);
        const int lane = static_cast<int>(rng() % 32);
        uint8_t* bp = enc_plain.data() + b * CW;
        uint8_t* bc = enc_crc.data() + b * CW;
        for (int e = 0; e < T + 1; ++e) {
            const size_t pos = (rng() % TOTAL_CODEWORD_LEN) * 32 + lane;
            const uint8_t v = static_cast<uint8_t>((rng() % 255) + 1);
            bp[pos] ^= v;
            bc[pos] ^= v;
        }

        std::vector<uint8_t> out;
        if (plain.decode(enc_plain, out) && out != data) plain_silent++;

        out.clear();
        if (with_crc.decode(enc_crc, out)) {
            if (out != data) crc_silent++;
        } else {
            crc_detected++;
        }
    }

    std::cout << "Без CRC: decode() вернул true, но данные испорчены: " << plain_silent
              << " / " << TRIALS << "\n";
    std::cout << "С CRC:   decode() вернул false (подмена обнаружена): " << crc_detected
              << " / " << TRIALS << ", тихих подмен: " << crc_silent << "\n";
}

int main() {
    // 1. Прогон теста для t = 5
    std::cout << "=== Тестирование обертки RSStreamCodec (t = 5) ===\n";
    rs_gf256::RSStreamCodec<5, 223> codec_t5;
    constexpr size_t TEST_SIZE_5 = 1234567;
    std::vector<uint8_t> data_t5(TEST_SIZE_5);
    std::mt19937 rng(42);
    for (auto& b : data_t5) b = static_cast<uint8_t>(rng());

    std::vector<uint8_t> enc_t5 = codec_t5.encode(data_t5);

    constexpr size_t TOTAL_CW_5 = 223 + 10;
    for (size_t b = 0; b < enc_t5.size() / (TOTAL_CW_5 * 32); ++b) {
        if (rng() % 10 == 0) {
            uint8_t* batch = enc_t5.data() + b * TOTAL_CW_5 * 32;
            for (int lane = 0; lane < 32; ++lane) {
                int errs = (rng() % 5) + 1;
                for (int e = 0; e < errs; ++e) {
                    size_t step = rng() % TOTAL_CW_5;
                    batch[step * 32 + lane] ^= static_cast<uint8_t>((rng() % 255) + 1);
                }
            }
        }
    }

    std::vector<uint8_t> dec_t5;
    bool ok_t5 = codec_t5.decode(enc_t5, dec_t5);
    if (ok_t5 && dec_t5 == data_t5) {
        std::cout << "[SUCCESS] Данные при t = 5 полностью восстановлены!\n";
    } else {
        std::cout << "[FAIL] Ошибка при t = 5!\n";
        return 1;
    }

    // 2. Прогон теста для t = 3
    run_test_t3();

    // 3. Демонстрация границы радиуса t и внешнего CRC
    run_crc_demo();

    return 0;
}
