#include "transformer_translation_viz.hpp"
#include <iostream>
#include <fstream>
#include <filesystem>

/**
 * @file main.cpp
 * @brief Main entry point for Transformer Translation Visualization Pipeline
 * 
 * This file demonstrates the complete C++ implementation of a transformer-based
 * neural machine translation system with integrated attention visualization.
 * 
 * @author Karan Pathania
 * @date 2026
 * @version 1.0
 */

namespace fs = std::filesystem;

/**
 * Creates output directory if it doesn't exist
 * @param dir_path Path to the output directory
 */
void create_output_directory(const std::string& dir_path) {
    try {
        if (!fs::exists(dir_path)) {
            fs::create_directories(dir_path);
            std::cout << "Created output directory: " << dir_path << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "Error creating output directory: " << e.what() << "\n";
    }
}

/**
 * Main function: Orchestrates the translation pipeline
 */
int main(int argc, char* argv[]) {
    try {
        std::cout << "============================================================\n";
        std::cout << "   Transformer Translation & Visualization Pipeline (C++)\n";
        std::cout << "============================================================\n\n";
        
        // Create output directory
        std::string output_dir = "./output";
        create_output_directory(output_dir);
        
        // Initialize tokenizer
        auto tokenizer = std::make_shared<TransformerViz::SimpleTokenizer>();
        std::cout << "✓ Tokenizer initialized. Vocabulary size: " 
                  << tokenizer->get_vocab_size() << "\n\n";
        
        // Initialize model
        auto model = std::make_shared<TransformerViz::SimpleTransformerModel>(tokenizer);
        std::cout << "✓ Model initialized.\n";
        std::cout << "  - Model dimension (d_model): " << model->get_d_model() << "\n";
        std::cout << "  - Vocabulary size: " << model->get_vocab_size() << "\n";
        std::cout << "  - Decoder start token ID: " << model->get_decoder_start_token_id() << "\n";
        std::cout << "  - EOS token ID: " << model->get_eos_token_id() << "\n\n";
        
        // Create translation pipeline
        TransformerViz::TranslationPipeline pipeline(
            tokenizer, 
            model,
            TransformerViz::TOP_K_DEFAULT,
            TransformerViz::MAX_DECODE_STEPS_DEFAULT
        );
        
        // Get input
        std::string input_sentence;
        if (argc > 1) {
            // Use command-line argument if provided
            input_sentence = argv[1];
        } else {
            // Read from user input
            std::cout << "Enter a sentence to translate: ";
            std::getline(std::cin, input_sentence);
        }
        
        if (input_sentence.empty()) {
            input_sentence = "hello world python code transformer";
            std::cout << "Using default input: \"" << input_sentence << "\"\n\n";
        }
        
        // Run translation pipeline
        pipeline.run(input_sentence, output_dir);
        
        std::cout << "\n============================================================\n";
        std::cout << "✓ Pipeline completed successfully!\n";
        std::cout << "  Output files saved to: " << output_dir << "\n";
        std::cout << "============================================================\n\n";
        
        return EXIT_SUCCESS;
        
    } catch (const std::exception& e) {
        std::cerr << "\n Error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
