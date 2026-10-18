// rs_vec.hpp - AVX2 interleaved Reed-Solomon codec over GF(2^8).
//
// Header-only C++20 library for streaming encoding/decoding of Reed-Solomon
// codes with 32-way SIMD interleaving (structure-of-arrays) on AVX2.
//
// Field: GF(2^8), primitive polynomial x^8 + x^4 + x^3 + x^2 + 1 (0x11D),
// primitive root alpha = 2. Generator roots are alpha^1 .. alpha^(2t).
#pragma once

#include <immintrin.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
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

// ---------------------------------------------------------------------------
// CRC-32C (Castagnoli, reflected poly 0x82F63B78).
// Used by RSStreamCodec<t, k, with_crc = true> as an outer integrity check.
//
// Реализация использует аппаратную инструкцию CRC32 (SSE4.2), которая на
// целевом железе (AVX2 => SSE4.2) доступна всегда; табличный фолбэк даёт
// побайтово идентичный результат, просто медленнее.
// ---------------------------------------------------------------------------
struct CRC32CTables {
    std::array<u32, 256> table{};

    constexpr CRC32CTables() {
        for (u32 i = 0; i < 256; ++i) {
            u32 c = i;
            for (int bit = 0; bit < 8; ++bit) {
                c = (c & 1u) ? (0x82F63B78u ^ (c >> 1)) : (c >> 1);
            }
            table[i] = c;
        }
    }
};

inline const CRC32CTables& get_crc32c_tables() {
    static const CRC32CTables tables;
    return tables;
}

inline u32 crc32c(const u8* data, size_t size) {
#if defined(__SSE4_2__)
    // Целевая архитектура little-endian, поэтому порядок байт совпадает с
    // порядком обхода потока инструкцией _mm_crc32_u64.
    u32 crc = 0xFFFFFFFFu;
    size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        u64 chunk = 0;
        std::memcpy(&chunk, data + i, sizeof(u64));
        crc = static_cast<u32>(_mm_crc32_u64(crc, chunk));
    }
    for (; i < size; ++i) {
        crc = static_cast<u32>(_mm_crc32_u8(crc, data[i]));
    }
    return crc ^ 0xFFFFFFFFu;
#else
    const auto& tbl = get_crc32c_tables();
    u32 crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < size; ++i) {
        crc = (crc >> 8) ^ tbl.table[(crc ^ data[i]) & 0xFFu];
    }
    return crc ^ 0xFFFFFFFFu;
#endif
}

// ---------------------------------------------------------------------------
// VectorMultiplier: SIMD multiply-by-constant over 32 GF(2^8) lanes using
// nibble lookup tables (VPSSHUFB).
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// RSStreamCodec: payload framing on top of the 32-way interleaved engine.
//
//   with_crc = false : [u64 original_size][data][zero padding]
//   with_crc = true  : [u64 original_size][u32 crc32(data)][data][zero padding]
//
// The header itself lives inside the RS-protected payload, so it is repaired
// together with the data.
// ---------------------------------------------------------------------------
template <int t = 5, size_t k_chunk = 223, bool with_crc = false>
class RSStreamCodec {
    static_assert(k_chunk + 2 * t <= 255,
                  "Суммарный размер блока не может превышать 255!");
    static_assert(t >= 1, "Необходимо исправлять хотя бы одну ошибку.");

   public:
    static constexpr size_t BATCH_SIZE = 32;
    static constexpr size_t PARITY_LEN = 2 * t;
    static constexpr size_t TOTAL_CODEWORD_LEN = k_chunk + PARITY_LEN;
    static constexpr size_t BATCH_PAYLOAD_BYTES = k_chunk * BATCH_SIZE;
    static constexpr size_t BATCH_CODEWORD_BYTES =
        TOTAL_CODEWORD_LEN * BATCH_SIZE;
    static constexpr size_t HEADER_LEN = sizeof(u64) + (with_crc ? sizeof(u32) : 0);
    static constexpr bool uses_crc = with_crc;

    static_assert(HEADER_LEN <= BATCH_PAYLOAD_BYTES,
                  "Заголовок не помещается в полезную нагрузку пачки.");

   private:
    ReedSolomonGF256Batch<t> engine;

   public:
    RSStreamCodec() = default;

    std::vector<u8> encode(const std::vector<u8>& input) const {
        return encode(input.data(), input.size());
    }

    std::vector<u8> encode(const u8* data, size_t size) const {
        const u64 original_size = static_cast<u64>(size);
        const size_t total_payload = HEADER_LEN + size;

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
                if (with_crc) {
                    const u32 crc = crc32c(data, size);
                    std::memcpy(batch_buf.data() + offset, &crc, sizeof(u32));
                    offset += sizeof(u32);
                }
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

    // Декодирование с исправлением ошибок.
    //
    // Гарантии и границы применимости:
    //   * Любая комбинация не более чем t ошибочных байт в каждой дорожке
    //     (т.е. в каждом из 32 независимых кодовых слов) исправляется
    //     полностью и достоверно. Это классическое bounded-distance (ML)
    //     декодирование: если в шаре радиуса t вокруг принятого слова есть
    //     кодовое слово, оно единственное, и возвращается именно оно.
    //   * При числе повреждений больше t достоверность невозможна в принципе,
    //     и это свойство любого корректирующего кода: пространство синдромов
    //     (256^(2t)) почти целиком покрыто классами ошибок веса <= t, поэтому
    //     заметная доля слов вне радиуса декодирования лежит в шаре ДРУГОГО
    //     кодового слова. Любой декодер с той же избыточностью вернёт это
    //     другое слово (измеренная доля совпадает с теоретическим отношением
    //     объёма шаров радиуса t к 256^(2t): ~99% для t = 1, ~12% для t = 3).
    //   * Следствие: true от decode() НЕ является гарантией целостности при
    //     повреждениях сверх t. Для недоверенных каналов используйте
    //     RSStreamCodec<t, k, true> (CRC-32 внутри RS-защиты) либо свой
    //     внешний контроль целостности: тогда необнаружимая подмена остаётся
    //     только при совпадении CRC-32, вероятность ~2^-32.
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

        u32 expected_crc = 0;
        if (with_crc) {
            std::memcpy(&expected_crc, encoded.data() + sizeof(u64), sizeof(u32));
        }

        const size_t max_possible = num_batches * BATCH_PAYLOAD_BYTES - HEADER_LEN;
        if (original_size > max_possible) {
            return false;
        }

        // 3. Выделяем память ровно один раз и пишем напрямую из батчей без decoded_payload!
        output.resize(original_size);
        if (original_size == 0) {
            if (with_crc && crc32c(nullptr, 0) != expected_crc) {
                return false;
            }
            return true;
        }

        size_t written = 0;
        bool header_skipped = false;

        for (size_t b = 0; b < num_batches && written < original_size; ++b) {
            const u8* src = encoded.data() + b * BATCH_CODEWORD_BYTES;
            size_t src_offset = 0;

            if (!header_skipped) {
                src_offset = HEADER_LEN;
                header_skipped = true;
            }

            size_t available_payload = BATCH_PAYLOAD_BYTES - src_offset;
            size_t to_write = std::min(available_payload, original_size - written);

            if (to_write > 0) {
                std::memcpy(output.data() + written, src + src_offset, to_write);
                written += to_write;
            }
        }

        // 4. Внешний контроль целостности: отсекает "тихое" попадание в другое
        //    кодовое слово при повреждениях сверх t.
        if (with_crc && crc32c(output.data(), output.size()) != expected_crc) {
            output.clear();
            return false;
        }

        return true;
    }
};
}  // namespace rs_gf256
