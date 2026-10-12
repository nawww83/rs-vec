#include <immintrin.h>

#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

namespace rs_gf256 {
using u8 = uint8_t;
using u32 = uint32_t;
using u64 = uint64_t;

constexpr u32 PRIM_POLY = 0x11D;

struct GF256Tables {
    std::array<u8, 256> exp{};
    std::array<u8, 256> log{};

    constexpr GF256Tables() {
        u32 val = 1;
        for (int i = 0; i < 255; ++i) {
            exp[i] = static_cast<u8>(val);
            log[val] = static_cast<u8>(i);
            val <<= 1;
            if (val & 0x100) val ^= PRIM_POLY;
        }
        log[0] = 0;
    }
};

inline const GF256Tables& get_tables() {
    static const GF256Tables tables;
    return tables;
}

inline u8 gf_add(u8 a, u8 b) { return a ^ b; }
inline u8 gf_sub(u8 a, u8 b) { return a ^ b; }

inline u8 gf_mul(u8 a, u8 b) {
    if (a == 0 || b == 0) return 0;
    const auto& tbl = get_tables();
    return tbl.exp[(tbl.log[a] + tbl.log[b]) % 255];
}

inline u8 gf_inv(u8 a) {
    assert(a != 0);
    const auto& tbl = get_tables();
    return tbl.exp[(255 - tbl.log[a]) % 255];
}

struct VectorMultiplier {
    __m256i low_tbl;
    __m256i high_tbl;

    void init(u8 factor) {
        alignas(32) std::array<u8, 32> l{};
        alignas(32) std::array<u8, 32> h{};
        for (int i = 0; i < 16; ++i) {
            u8 val_low = gf_mul(static_cast<u8>(i), factor);
            u8 val_high = gf_mul(static_cast<u8>(i << 4), factor);
            l[i] = val_low;
            l[i + 16] = val_low;
            h[i] = val_high;
            h[i + 16] = val_high;
        }
        low_tbl = _mm256_load_si256(reinterpret_cast<const __m256i*>(l.data()));
        high_tbl =
            _mm256_load_si256(reinterpret_cast<const __m256i*>(h.data()));
    }

    inline __m256i multiply(__m256i v) const {
        const __m256i mask4 = _mm256_set1_epi8(0x0F);
        __m256i low_nibbles = _mm256_and_si256(v, mask4);
        __m256i high_nibbles = _mm256_and_si256(_mm256_srli_epi16(v, 4), mask4);

        __m256i r_low = _mm256_shuffle_epi8(low_tbl, low_nibbles);
        __m256i r_high = _mm256_shuffle_epi8(high_tbl, high_nibbles);

        return _mm256_xor_si256(r_low, r_high);
    }
};

template <int t>
class ReedSolomonGF256Batch {
   public:
    static constexpr int num_syndromes = 2 * t;
    static constexpr int BATCH_SIZE = 32;

   private:
    std::array<u8, num_syndromes> roots;
    std::array<VectorMultiplier, num_syndromes> root_multipliers;
    std::vector<u8> G;
    std::array<VectorMultiplier, num_syndromes> g_multipliers;

   public:
    ReedSolomonGF256Batch() {
        u8 alpha = 2;
        u8 cur = alpha;
        for (int i = 0; i < num_syndromes; ++i) {
            roots[i] = cur;
            root_multipliers[i].init(cur);
            cur = gf_mul(cur, alpha);
        }

        G.assign(num_syndromes + 1, 0);
        G[0] = 1;
        for (int i = 0; i < num_syndromes; ++i) {
            u8 root = roots[i];
            for (int j = i + 1; j > 0; --j) {
                G[j] = gf_add(G[j - 1], gf_mul(G[j], root));
            }
            G[0] = gf_mul(G[0], root);
        }

        for (int i = 0; i < num_syndromes; ++i) {
            g_multipliers[i].init(G[i]);
        }
    }

    void encode_stream(const u8* data_slices, size_t k_len,
                       u8* out_codeword_slices) const {
        alignas(32) __m256i rem[num_syndromes];
        for (int i = 0; i < num_syndromes; ++i) {
            rem[i] = _mm256_setzero_si256();
        }

        for (size_t step = 0; step < k_len; ++step) {
            __m256i in_vec =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                    data_slices + step * BATCH_SIZE));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(out_codeword_slices +
                                                           step * BATCH_SIZE),
                                in_vec);

            __m256i feedback = _mm256_xor_si256(in_vec, rem[num_syndromes - 1]);
            for (int j = num_syndromes - 1; j > 0; --j) {
                __m256i term = g_multipliers[j].multiply(feedback);
                rem[j] = _mm256_xor_si256(rem[j - 1], term);
            }
            rem[0] = g_multipliers[0].multiply(feedback);
        }

        for (int i = num_syndromes - 1; i >= 0; --i) {
            size_t out_idx = k_len + (num_syndromes - 1 - i);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(
                                    out_codeword_slices + out_idx * BATCH_SIZE),
                                rem[i]);
        }
    }

    uint32_t compute_syndromes(const u8* codeword_slices, size_t n,
                               std::array<std::array<u8, num_syndromes>,
                                          BATCH_SIZE>& out_syndromes) const {
        alignas(32) __m256i syn_lanes[num_syndromes];
        for (int i = 0; i < num_syndromes; ++i) {
            syn_lanes[i] = _mm256_setzero_si256();
        }

        for (size_t step = 0; step < n; ++step) {
            __m256i in_vec =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                    codeword_slices + step * BATCH_SIZE));
            for (int i = 0; i < num_syndromes; ++i) {
                __m256i mult = root_multipliers[i].multiply(syn_lanes[i]);
                syn_lanes[i] = _mm256_xor_si256(mult, in_vec);
            }
        }

        alignas(32) std::array<u8, BATCH_SIZE> buf{};
        uint32_t error_mask = 0;
        for (int i = 0; i < num_syndromes; ++i) {
            _mm256_store_si256(reinterpret_cast<__m256i*>(buf.data()),
                               syn_lanes[i]);
            for (int b = 0; b < BATCH_SIZE; ++b) {
                out_syndromes[b][i] = buf[b];
                if (buf[b] != 0) {
                    error_mask |= (1u << b);
                }
            }
        }

        return error_mask;
    }

    bool correct_single_block(u8* codeword_slices, size_t n, int b,
                              const std::array<u8, num_syndromes>& S) const {
        const auto& tbl = get_tables();
        std::vector<u8> Lambda(t + 1, static_cast<u8>(0));
        std::vector<u8> B(t + 1, static_cast<u8>(0));
        Lambda[0] = 1;
        B[0] = 1;
        int L = 0, m = 1;
        u8 b_val = 1;

        for (int r = 0; r < num_syndromes; ++r) {
            u8 d = S[r];
            for (int i = 1; i <= L; ++i) {
                d = gf_add(d, gf_mul(Lambda[i], S[r - i]));
            }

            if (d == 0) {
                m++;
            } else {
                std::vector<u8> T = Lambda;
                u8 scale = gf_mul(d, gf_inv(b_val));
                for (int i = 0; i <= t; ++i) {
                    if (i + m <= t) {
                        Lambda[i + m] =
                            gf_add(Lambda[i + m], gf_mul(scale, B[i]));
                    }
                }
                if (2 * L <= r) {
                    L = r + 1 - L;
                    B = T;
                    b_val = d;
                    m = 1;
                } else {
                    m++;
                }
            }
        }

        if (L > t || L == 0) return false;

        std::vector<u8> Omega(L, static_cast<u8>(0));
        for (int i = 0; i < L; ++i) {
            u8 term = 0;
            for (int j = 0; j <= i; ++j) {
                term = gf_add(term, gf_mul(Lambda[j], S[i - j]));
            }
            Omega[i] = term;
        }

        int errors_found = 0;
        for (size_t step = 0; step < n; ++step) {
            int p = static_cast<int>(n - 1 - step);
            int exp = (255 - (p % 255)) % 255;
            u8 X_inv = tbl.exp[exp];

            u8 sum = 0;
            u8 cur = 1;
            for (int i = 0; i <= L; ++i) {
                sum = gf_add(sum, gf_mul(Lambda[i], cur));
                cur = gf_mul(cur, X_inv);
            }

            if (sum == 0) {
                u8 num = 0;
                cur = 1;
                for (int i = 0; i < L; ++i) {
                    num = gf_add(num, gf_mul(Omega[i], cur));
                    cur = gf_mul(cur, X_inv);
                }

                u8 den = 0;
                cur = 1;
                for (int i = 1; i <= L; i += 2) {
                    den = gf_add(den, gf_mul(Lambda[i], cur));
                    cur = gf_mul(cur, gf_mul(X_inv, X_inv));
                }

                if (den == 0) return false;

                u8 error_val = gf_mul(num, gf_inv(den));
                codeword_slices[step * BATCH_SIZE + b] =
                    gf_add(codeword_slices[step * BATCH_SIZE + b], error_val);
                errors_found++;

                if (errors_found == L) break;
            }
        }

        return (errors_found == L);
    }

    bool decode_stream(u8* codeword_slices, size_t n) const {
        std::array<std::array<u8, num_syndromes>, BATCH_SIZE> syndromes;
        uint32_t error_mask = compute_syndromes(codeword_slices, n, syndromes);

        if (error_mask == 0) return true;  // Без ошибок — мгновенный выход

        bool all_recovered = true;
        for (int b = 0; b < BATCH_SIZE; ++b) {
            if ((error_mask & (1u << b)) != 0) {
                if (!correct_single_block(codeword_slices, n, b,
                                          syndromes[b])) {
                    all_recovered = false;
                }
            }
        }

        return all_recovered;
    }
};

template <int t = 5, size_t k_chunk = 223>
class RSStreamCodec {
    static_assert(k_chunk + 2 * t <= 255,
                  "Суммарный размер блока не может превышать 255!");

    static constexpr size_t BATCH_SIZE = 32;
    static constexpr size_t PARITY_LEN = 2 * t;
    static constexpr size_t TOTAL_CODEWORD_LEN = k_chunk + PARITY_LEN;
    static constexpr size_t BATCH_PAYLOAD_BYTES = k_chunk * BATCH_SIZE;
    static constexpr size_t BATCH_CODEWORD_BYTES =
        TOTAL_CODEWORD_LEN * BATCH_SIZE;

    ReedSolomonGF256Batch<t> engine;

   public:
    RSStreamCodec() = default;

    std::vector<u8> encode(const std::vector<u8>& input) const {
        return encode(input.data(), input.size());
    }

    std::vector<u8> encode(const u8* data, size_t size) const {
        const u64 original_size = static_cast<u64>(size);
        const size_t total_payload = sizeof(u64) + size;

        size_t num_batches = (total_payload + BATCH_PAYLOAD_BYTES - 1) / BATCH_PAYLOAD_BYTES;
        if (num_batches == 0) num_batches = 1;

        // Сразу выделяем итоговый буфер без промежуточных векторов!
        std::vector<u8> encoded(num_batches * BATCH_CODEWORD_BYTES, static_cast<u8>(0));

        // Используем маленький стековый буфер пачки (L1-кэш!)
        alignas(32) std::array<u8, BATCH_PAYLOAD_BYTES> batch_buf{};

        size_t bytes_left = size;
        const u8* cur_src = data;
        bool header_written = false;

        for (size_t b = 0; b < num_batches; ++b) {
            batch_buf.fill(0);
            size_t offset = 0;

            if (!header_written) {
                std::memcpy(batch_buf.data(), &original_size, sizeof(u64));
                offset += sizeof(u64);
                header_written = true;
            }

            size_t to_copy = std::min(bytes_left, BATCH_PAYLOAD_BYTES - offset);
            if (to_copy > 0) {
                std::memcpy(batch_buf.data() + offset, cur_src, to_copy);
                cur_src += to_copy;
                bytes_left -= to_copy;
            }

            u8* out_ptr = encoded.data() + b * BATCH_CODEWORD_BYTES;
            engine.encode_stream(batch_buf.data(), k_chunk, out_ptr);
        }

        return encoded;
    }

    bool decode(std::vector<u8>& encoded, std::vector<u8>& output) const {
        if (encoded.empty() || (encoded.size() % BATCH_CODEWORD_BYTES != 0)) {
            return false;
        }

        const size_t num_batches = encoded.size() / BATCH_CODEWORD_BYTES;

        // 1. Декодируем и исправляем ошибки на месте (in-place)
        for (size_t b = 0; b < num_batches; ++b) {
            u8* batch_ptr = encoded.data() + b * BATCH_CODEWORD_BYTES;
            if (!engine.decode_stream(batch_ptr, TOTAL_CODEWORD_LEN)) {
                return false;
            }
        }

        // 2. Считываем заголовок с оригинальной длиной из начала первого батча
        u64 original_size = 0;
        std::memcpy(&original_size, encoded.data(), sizeof(u64));

        const size_t max_possible = num_batches * BATCH_PAYLOAD_BYTES - sizeof(u64);
        if (original_size > max_possible) {
            return false;
        }

        // 3. Выделяем память ровно один раз и пишем напрямую из батчей без decoded_payload!
        output.resize(original_size);
        if (original_size == 0) return true;

        size_t written = 0;
        bool header_skipped = false;

        for (size_t b = 0; b < num_batches && written < original_size; ++b) {
            const u8* src = encoded.data() + b * BATCH_CODEWORD_BYTES;
            size_t src_offset = 0;

            if (!header_skipped) {
                src_offset = sizeof(u64);
                header_skipped = true;
            }

            size_t available_payload = BATCH_PAYLOAD_BYTES - src_offset;
            size_t to_write = std::min(available_payload, original_size - written);

            if (to_write > 0) {
                std::memcpy(output.data() + written, src + src_offset, to_write);
                written += to_write;
            }
        }

        return true;
    }
};
}  // namespace rs_gf256


#include <chrono>
#include <iomanip>

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

    return 0;
}
