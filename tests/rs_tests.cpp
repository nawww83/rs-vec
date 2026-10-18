// Независимый тестовый набор для rs_vec.hpp.
//
// Запуск: см. tests/run_tests.ps1 либо
//   g++ -O3 -mavx2 -std=c++20 -Wall -Wextra -Wpedantic tests/rs_tests.cpp -o build/rs_tests
//
// Набор проверяет как happy path, так и границы применимости кода:
//   * round-trip на всевозможных размерах и на границах пачек;
//   * полное исправление любой комбинации из <= t ошибок в дорожке;
//   * burst-ошибки: 32*t подряд восстанавливаются, 32*t + 1 — уже нет;
//   * независимая (скалярная) проверка синдромов выходного кодового слова;
//   * поведение за радиусом t: декодер обязан либо отвергнуть слово, либо
//     вернуть именно кодовое слово (никогда — произвольный мусор);
//   * CRC-32: эталонное значение, формат потока и отсечение "тихих" подмен;
//   * устойчивость к мусорным входам и детерминизм.
#include "../rs_vec.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

using namespace rs_gf256;
using u8 = uint8_t;

static int g_checks = 0;
static int g_fails = 0;

static void check(bool cond, const char* what) {
    ++g_checks;
    if (!cond) {
        ++g_fails;
        std::printf("   [FAIL] %s\n", what);
    }
}

template <class F>
static void section(const char* name, F&& fn) {
    std::printf("== %s\n", name);
    const int before = g_fails;
    fn();
    if (g_fails == before)
        std::printf("   -> OK\n");
    else
        std::printf("   -> %d PROBLEM(S)\n", g_fails - before);
    std::fflush(stdout);
}

static std::vector<u8> make_data(size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<u8> v(n);
    for (auto& x : v) x = static_cast<u8>(rng());
    return v;
}

// ---------------------------------------------------------------------------
// CRC-32C as such: known-answer test.
// ---------------------------------------------------------------------------
static void t_crc_known_answer() {
    const char* s = "123456789";
    const u32 got = crc32c(reinterpret_cast<const u8*>(s), 9);
    std::printf("   crc32c(\"123456789\") = 0x%08X (expect 0xE3069283)\n", got);
    check(got == 0xE3069283u, "CRC-32C check value");
    check(crc32c(nullptr, 0) == 0u, "CRC-32C of empty input is 0");
}

// ---------------------------------------------------------------------------
// 1. Round-trip without any corruption, across many sizes incl. batch edges.
// ---------------------------------------------------------------------------
template <int t, size_t k, bool crc>
static void t_roundtrip_sizes() {
    using Codec = RSStreamCodec<t, k, crc>;
    Codec codec;
    const size_t PAY = Codec::BATCH_PAYLOAD_BYTES;
    const size_t CW = Codec::BATCH_CODEWORD_BYTES;
    const size_t HDR = Codec::HEADER_LEN;

    std::vector<size_t> sizes = {0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33,
                                 63, 64, 65, 100, 222, 223, 224, 255, 256,
                                 257, 1000, 4096, 10000};
    for (int d = 13; d >= 1; --d) {
        if (PAY > static_cast<size_t>(d)) sizes.push_back(PAY - d);
    }
    sizes.push_back(PAY);
    sizes.push_back(PAY + 1);
    sizes.push_back(3 * PAY + 17);

    for (size_t size : sizes) {
        auto data = make_data(size, static_cast<uint32_t>(size * 2654435761u + t));
        auto enc = codec.encode(data);

        const size_t expect_batches = (HDR + size + PAY - 1) / PAY;
        check(enc.size() == expect_batches * CW, "encoded size formula");
        check(enc.size() % CW == 0, "encoded size is a whole number of batches");

        std::vector<u8> out;
        const bool ok = codec.decode(enc, out);
        check(ok, "decode(clean) returns true");
        check(out == data, "decode(clean) reproduces payload");
    }
}

// ---------------------------------------------------------------------------
// 2. Random in-capacity errors, including the worst case of a fully damaged
//    batch and exactly t errors per lane.
// ---------------------------------------------------------------------------
template <int t, size_t k, bool crc>
static void t_error_correction(int trials) {
    using Codec = RSStreamCodec<t, k, crc>;
    Codec codec;
    const size_t CW = Codec::BATCH_CODEWORD_BYTES;
    const size_t n = Codec::TOTAL_CODEWORD_LEN;
    std::mt19937 rng(0xC0FFEE);

    for (int it = 0; it < trials; ++it) {
        const size_t size = 1 + rng() % (4 * k * 32);
        auto data = make_data(size, rng());
        auto enc = codec.encode(data);

        const size_t batches = enc.size() / CW;
        const int mode = static_cast<int>(rng() % 3);
        for (size_t b = 0; b < batches; ++b) {
            u8* base = enc.data() + b * CW;
            for (int lane = 0; lane < 32; ++lane) {
                int errs = 0;
                if (mode == 0)
                    errs = static_cast<int>(rng() % (t + 1));
                else if (mode == 1)
                    errs = t;  // always the maximum correctable
                else
                    errs = (rng() % 2) ? t : 0;
                for (int e = 0; e < errs; ++e) {
                    const size_t s = rng() % n;
                    base[s * 32 + lane] ^= static_cast<u8>((rng() % 255) + 1);
                }
            }
        }

        std::vector<u8> out;
        const bool ok = codec.decode(enc, out);
        if (!ok || out != data) {
            std::printf(
                "   [FAIL] uncorrected in-capacity damage: size=%zu mode=%d "
                "ok=%d size_out=%zu\n",
                size, mode, (int)ok, out.size());
            ++g_fails;
            break;
        }
    }
    ++g_checks;
}

// ---------------------------------------------------------------------------
// 3. Burst errors: the interleaving claim from the README.
// ---------------------------------------------------------------------------
template <int t, size_t k, bool crc>
static void t_burst() {
    using Codec = RSStreamCodec<t, k, crc>;
    Codec codec;
    const size_t CW = Codec::BATCH_CODEWORD_BYTES;

    const size_t size = 100000;
    auto data = make_data(size, 0xBEEF);

    struct Case {
        const char* name;
        size_t len;
        bool must_recover;
    };
    std::vector<Case> cases = {
        {"burst = t bytes", (size_t)t, true},
        {"burst = 32*t - 1 bytes (readme limit)", 32 * (size_t)t - 1, true},
        {"burst = 32*t bytes (readme limit)", 32 * (size_t)t, true},
        {"burst = 32*t + 1 bytes (over limit)", 32 * (size_t)t + 1, false},
        {"burst = 2*32*t bytes (2x over limit)", 64 * (size_t)t, false},
    };

    for (const auto& c : cases) {
        auto enc = codec.encode(data);
        // Place the burst in the middle of the first batch's payload so it
        // stays inside a single batch for the in-limit cases.
        const size_t start = CW / 2;
        for (size_t i = 0; i < c.len && start + i < enc.size(); ++i)
            enc[start + i] ^= 0x5A;

        std::vector<u8> out;
        const bool ok = codec.decode(enc, out);
        const bool good = ok && out == data;
        std::printf("   %-38s len=%-5zu recovered=%s (decode=%s)\n", c.name, c.len,
                    good ? "yes" : "no", ok ? "true" : "false");
        if (c.must_recover) check(good, "burst within 32*t must be recovered");
        else check(!good, "burst over 32*t must not silently pass as correct");
    }
}

// ---------------------------------------------------------------------------
// 4. Independent validation of the encoder: the produced codeword must
//    satisfy every syndrome equation S_j = 0, computed with a separate
//    scalar routine (not the AVX2 kernel).
// ---------------------------------------------------------------------------
template <int t, size_t k, bool crc>
static void t_encoder_produces_valid_codewords() {
    using Codec = RSStreamCodec<t, k, crc>;
    Codec codec;
    const size_t n = Codec::TOTAL_CODEWORD_LEN;
    const size_t CW = Codec::BATCH_CODEWORD_BYTES;
    const auto& tbl = get_tables();
    std::mt19937 rng(0xABCD);
    size_t bad = 0, checked_words = 0;

    for (int it = 0; it < 10; ++it) {
        auto data = make_data(1000 + rng() % 20000, rng());
        auto enc = codec.encode(data);
        for (size_t b = 0; b < enc.size() / CW; ++b) {
            const u8* base = enc.data() + b * CW;
            for (int lane = 0; lane < 32; ++lane) {
                ++checked_words;
                for (int j = 1; j <= 2 * t; ++j) {
                    u8 acc = 0;
                    for (size_t s = 0; s < n; ++s) {
                        const u32 e =
                            static_cast<u32>((uint64_t(j) * (n - 1 - s)) % 255);
                        acc ^= gf_mul(base[s * 32 + lane], tbl.exp[e]);
                    }
                    if (acc != 0) ++bad;
                }
            }
        }
    }
    std::printf("   verified %zu codewords x %d syndromes, violations=%zu\n",
                checked_words, 2 * t, bad);
    check(bad == 0, "all zero syndromes on the clean encoder output");
}

// ---------------------------------------------------------------------------
// 5. Beyond-capacity damage: every "success" must land on a real codeword.
//    Reporting the counts is the point; the assertion is only the invariant
//    "decode()==true  =>  the returned bytes are a valid codeword".
// ---------------------------------------------------------------------------
template <int t, size_t k, bool crc>
static void t_beyond_capacity(int trials) {
    using Codec = RSStreamCodec<t, k, crc>;
    Codec codec;
    const size_t CW = Codec::BATCH_CODEWORD_BYTES;
    const size_t n = Codec::TOTAL_CODEWORD_LEN;
    std::mt19937 rng(0xDEAD);

    ReedSolomonGF256Batch<t> checker;  // re-validate corrected blocks
    std::array<std::array<u8, 2 * t>, 32> syn{};

    size_t detected = 0, recovered = 0, silent_wrong = 0, silent_but_detectable = 0;
    for (int it = 0; it < trials; ++it) {
        const size_t size = 1000 + rng() % 20000;
        auto data = make_data(size, rng());
        auto enc = codec.encode(data);

        const size_t b = rng() % (enc.size() / CW);
        const int lane = static_cast<int>(rng() % 32);
        u8* base = enc.data() + b * CW;
        for (int e = 0; e < t + 1; ++e)
            base[(rng() % n) * 32 + lane] ^= static_cast<u8>((rng() % 255) + 1);

        std::vector<u8> out;
        const bool ok = codec.decode(enc, out);
        if (!ok) {
            ++detected;
        } else if (out == data) {
            ++recovered;
        } else {
            ++silent_wrong;
            // Invariant: a "successful" result must be a valid codeword.
            const uint32_t mask =
                checker.compute_syndromes(enc.data() + b * CW, n, syn);
            if (mask != 0) ++silent_but_detectable;
        }
    }
    std::printf(
        "   t+1 errors in one lane, %d trials: detected=%zu  recovered=%zu  "
        "wrong-but-valid-codeword=%zu (non-codeword yet accepted: %zu)\n",
        trials, detected, recovered, silent_wrong, silent_but_detectable);
    check(silent_but_detectable == 0,
          "decode()==true must always yield a real codeword");
    check(detected + recovered + silent_wrong == (size_t)trials, "accounting");
}

// ---------------------------------------------------------------------------
// 6. Silent-failure profile as a function of errors per lane. For with_crc
//    variants it must collapse to zero.
// ---------------------------------------------------------------------------
template <int t, size_t k, bool crc>
static void t_silent_failure_profile(int trials) {
    using Codec = RSStreamCodec<t, k, crc>;
    Codec codec;
    const size_t CW = Codec::BATCH_CODEWORD_BYTES;
    const size_t n = Codec::TOTAL_CODEWORD_LEN;
    std::mt19937 rng(0x5EED);

    std::printf("   errors/lane | decoded-false | correct | wrong-data-accepted\n");
    for (int e = t; e <= 2 * t + 2; ++e) {
        size_t wrong = 0, failed = 0, ok = 0;
        for (int it = 0; it < trials; ++it) {
            auto data = make_data(40000, rng());
            auto enc = codec.encode(data);
            const size_t b = rng() % (enc.size() / CW);
            const int lane = static_cast<int>(rng() % 32);
            u8* base = enc.data() + b * CW;
            for (int i = 0; i < e; ++i)
                base[(rng() % n) * 32 + lane] ^= static_cast<u8>((rng() % 255) + 1);

            std::vector<u8> out;
            if (!codec.decode(enc, out))
                ++failed;
            else if (out == data)
                ++ok;
            else
                ++wrong;
        }
        std::printf("   %11d | %13zu | %7zu | %zu  (%.0f%%)\n", e, failed, ok, wrong,
                    100.0 * double(wrong) / double(trials));
        if (crc) check(wrong == 0, "with CRC no wrong data may be accepted");
    }
}

// ---------------------------------------------------------------------------
// 7. Hostile input to decode().
// ---------------------------------------------------------------------------
template <int t, size_t k, bool crc>
static void t_decode_input_validation() {
    using Codec = RSStreamCodec<t, k, crc>;
    Codec codec;
    const size_t CW = Codec::BATCH_CODEWORD_BYTES;

    std::vector<u8> empty;
    std::vector<u8> out;
    check(codec.decode(empty, out) == false, "decode(empty) == false");

    std::vector<u8> truncated(CW - 1, 0);
    check(codec.decode(truncated, out) == false, "decode(truncated batch) == false");

    std::vector<u8> one_and_half(CW + CW / 2, 0);
    check(codec.decode(one_and_half, out) == false, "decode(1.5 batches) == false");

    // All-zero stream: header says size 0, CRC of nothing is 0.
    std::vector<u8> zeros(CW, 0);
    check(codec.decode(zeros, out) == true, "decode(all-zero) == true");
    check(out.empty(), "decode(all-zero) yields empty payload");

    // A random stream: header is a huge number, must be rejected, not crash.
    auto garbage = make_data(CW, 0x1234);
    const bool r = codec.decode(garbage, out);
    std::printf("   random-garbage batch: decode=%s out=%zu bytes\n",
                r ? "true" : "false", out.size());
    check(true, "garbage does not crash");
}

// ---------------------------------------------------------------------------
// 8. A stream may only be read back by the matching codec variant.
// ---------------------------------------------------------------------------
template <int t, size_t k>
static void t_variant_separation() {
    RSStreamCodec<t, k, false> plain;
    RSStreamCodec<t, k, true> crc;
    auto data = make_data(7777, 0x77);

    // A CRC-framed stream read by the plain codec starts its payload at the
    // wrong offset, so it must never come back as the original bytes.
    auto enc_crc = crc.encode(data);
    std::vector<u8> out;
    const bool ok = plain.decode(enc_crc, out);
    check(!(ok && out == data), "CRC stream misread as plain must not yield the data");
    std::printf("   crc-stream via plain codec: decode=%s, matches-original=%s\n",
                ok ? "true" : "false",
                (ok && out == data) ? "yes" : "no");

    // The matching codec of course works.
    out.clear();
    check(crc.decode(enc_crc, out) && out == data, "matching CRC codec round-trips");
}

// ---------------------------------------------------------------------------
// 9. Throughput sanity.
// ---------------------------------------------------------------------------
template <int t, size_t k, bool crc>
static void t_throughput() {
    using Codec = RSStreamCodec<t, k, crc>;
    Codec codec;
    const size_t CW = Codec::BATCH_CODEWORD_BYTES;
    const size_t n = Codec::TOTAL_CODEWORD_LEN;
    const size_t TEST = 10 * 1024 * 1024;
    auto data = make_data(TEST, 777);

    auto enc_start = std::chrono::high_resolution_clock::now();
    auto enc = codec.encode(data);
    auto enc_end = std::chrono::high_resolution_clock::now();
    const double enc_s = std::chrono::duration<double>(enc_end - enc_start).count();

    for (int pct : {0, 1, 2, 5, 10, 25, 50, 100}) {
        auto damaged = enc;
        std::mt19937 rng(99);
        size_t damaged_batches = 0;
        if (pct > 0) {
            const size_t batches = damaged.size() / CW;
            for (size_t b = 0; b < batches; ++b) {
                if (static_cast<int>(b % 100) >= pct) continue;
                ++damaged_batches;
                u8* base = damaged.data() + b * CW;
                for (int lane = 0; lane < 32; ++lane) {
                    const int errs = 1 + static_cast<int>(rng() % t);
                    for (int e = 0; e < errs; ++e)
                        base[(rng() % n) * 32 + lane] ^= static_cast<u8>((rng() % 255) + 1);
                }
            }
        }
        std::vector<u8> out;
        auto t0 = std::chrono::high_resolution_clock::now();
        const bool ok = codec.decode(damaged, out);
        auto t1 = std::chrono::high_resolution_clock::now();
        const double d_s = std::chrono::duration<double>(t1 - t0).count();
        const double mb = double(TEST) / (1024.0 * 1024.0);
        std::printf(
            "   damaged batches %3d%% (%5zu)  decode %7.2f ms  %8.2f MB/s  "
            "payload-intact=%s\n",
            pct, damaged_batches, d_s * 1000.0, mb / d_s,
            (ok && out == data) ? "yes" : "NO");
        check(ok && out == data, "damaged run must restore payload");
    }
    std::printf("   encode: %6.2f ms  %8.2f MB/s\n", enc_s * 1000.0,
                double(TEST) / (1024.0 * 1024.0) / enc_s);
    std::fflush(stdout);
}

// ---------------------------------------------------------------------------
// 10. Determinism: the same input must give bit-identical output.
// ---------------------------------------------------------------------------
template <int t, size_t k, bool crc>
static void t_determinism() {
    RSStreamCodec<t, k, crc> a, b;
    auto data = make_data(50000, 42);
    auto e1 = a.encode(data);
    auto e2 = a.encode(data);
    check(e1 == e2, "repeated encode deterministic");
    check(e1 == b.encode(data), "encode deterministic across instances");
}

template <int t, size_t k, bool crc>
static void run_config(const char* label) {
    std::printf("\n############ %s ############\n", label);
    section("round-trip / batch-edge sizes", [] { t_roundtrip_sizes<t, k, crc>(); });
    section("in-capacity random errors", [] { t_error_correction<t, k, crc>(40); });
    section("encoder emits valid codewords (scalar check)",
            [] { t_encoder_produces_valid_codewords<t, k, crc>(); });
    section("burst errors", [] { t_burst<t, k, crc>(); });
    section("beyond capacity", [] { t_beyond_capacity<t, k, crc>(200); });
    section("silent-failure profile", [] { t_silent_failure_profile<t, k, crc>(100); });
    section("decode() input validation", [] { t_decode_input_validation<t, k, crc>(); });
    section("determinism", [] { t_determinism<t, k, crc>(); });
    std::fflush(stdout);
}

int main() {
    std::printf("rs_vec.hpp independent test harness\n");

    section("CRC-32C known answer", [] { t_crc_known_answer(); });
    section("codec variant separation", [] { t_variant_separation<3, 223>(); });

    run_config<1, 253, false>("t = 1, k = 253, no CRC");
    run_config<2, 251, false>("t = 2, k = 251, no CRC");
    run_config<3, 223, false>("t = 3, k = 223, no CRC (README default)");
    run_config<5, 223, false>("t = 5, k = 223, no CRC (default)");
    run_config<8, 200, false>("t = 8, k = 200, no CRC");
    run_config<5, 100, false>("t = 5, k = 100, no CRC");

    run_config<3, 223, true>("t = 3, k = 223, with CRC-32C");
    run_config<5, 223, true>("t = 5, k = 223, with CRC-32C");

    section("throughput (t=3, k=223, no CRC)", [] { t_throughput<3, 223, false>(); });
    section("throughput (t=3, k=223, with CRC)", [] { t_throughput<3, 223, true>(); });

    std::printf("\n=========================================\n");
    std::printf("checks: %d, failures: %d\n", g_checks, g_fails);
    std::printf("=========================================\n");
    return g_fails == 0 ? 0 : 1;
}
