#include "model.h"

#include <cstdint>
#include <iomanip>
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: midas_model_probe model.midas\n";
        return 2;
    }
    try {
        midas_native::ModelFile model(
            argv[1], MIDAS_MODEL_V21_SMALL_256);
        if (!model.derivation().present) {
            std::cerr << "model has no derivation metadata\n";
            return 3;
        }
        std::cout << "tensors=" << model.tensor_count() << "\nsha256=";
        for (std::uint8_t byte :
             model.derivation().canonical_sha256) {
            std::cout << std::hex << std::setw(2)
                      << std::setfill('0')
                      << static_cast<unsigned>(byte);
        }
        std::cout << "\nconverter="
                  << model.derivation().converter << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
