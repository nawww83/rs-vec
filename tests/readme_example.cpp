// Компиляционная проверка примера из README.md ("Пример использования").
// Собирается и запускается скриптом tests/run_tests.ps1, чтобы пример в
// документации не мог разойтись с реальным API.
#include "rs_vec.hpp"

#include <iostream>
#include <vector>

int main() {
    // Создаем кодек: t = 3 (исправление до 3 ошибок на блок), k = 223 байта полезных данных
    rs_gf256::RSStreamCodec<3, 223> codec;

    // 1. Исходные данные произвольной длины
    std::vector<uint8_t> original_data = { 'H', 'e', 'l', 'l', 'o', ',', ' ', 'A', 'V', 'X', '2', '!' };

    // 2. Кодирование
    std::vector<uint8_t> encoded_stream = codec.encode(original_data);

    // 3. Симуляция повреждений (порча байтов)
    encoded_stream[0] ^= 0x55;
    encoded_stream[1] ^= 0xAA;
    encoded_stream[2] ^= 0x3C;

    // 4. Декодирование и восстановление
    std::vector<uint8_t> restored_data;
    bool success = codec.decode(encoded_stream, restored_data);

    if (success && restored_data == original_data) {
        std::cout << "Данные успешно восстановлены: ";
        for (char c : restored_data) std::cout << c;
        std::cout << std::endl;
    } else {
        std::cerr << "Сбой восстановления данных!" << std::endl;
        return 1;
    }

    // 5. Вариант с внешним контролем целостности (CRC-32 внутри RS-защиты).
    rs_gf256::RSStreamCodec<3, 223, true> codec_crc;
    auto stream_crc = codec_crc.encode(original_data);
    stream_crc[0] ^= 0x55;
    std::vector<uint8_t> restored_crc;
    if (!codec_crc.decode(stream_crc, restored_crc) || restored_crc != original_data) {
        std::cerr << "Сбой восстановления данных (CRC-вариант)!" << std::endl;
        return 1;
    }
    std::cout << "CRC-вариант также восстановил данные корректно." << std::endl;

    return 0;
}
