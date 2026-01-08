#ifndef TRANSFORMER_TRANSLATION_VISUALIZATION_HPP
#define TRANSFORMER_TRANSLATION_VISUALIZATION_HPP

#include <vector>
#include <string>
#include <map>
#include <memory>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <sstream>

/**
 * @namespace TransformerViz
 * @brief Namespace containing transformer translation and visualization components
 */
namespace TransformerViz {

    // ============================================================================
    // CONFIGURATION CONSTANTS
    // ============================================================================
    
    constexpr int TOP_K_DEFAULT = 20;
    constexpr int MAX_DECODE_STEPS_DEFAULT = 20;
    constexpr const char* PROBABILITY_COLORMAP = "viridis";
    constexpr const char* ATTENTION_COLORMAP = "YlGnBu";
    constexpr float SOFTMAX_TEMPERATURE = 1.0f;
    constexpr int MAX_ANNOTATION_CELLS = 25;

    // ============================================================================
    // TENSOR DATA STRUCTURES
    // ============================================================================

    /**
     * @class Tensor1D
     * @brief Represents a 1D tensor (vector) for numerical computations
     */
    class Tensor1D {
    private:
        std::vector<float> data;
        size_t size_;

    public:
        Tensor1D() : size_(0) {}
        explicit Tensor1D(size_t size) : data(size, 0.0f), size_(size) {}
        Tensor1D(const std::vector<float>& values) : data(values), size_(values.size()) {}

        float& operator[](size_t idx) { return data[idx]; }
        const float& operator[](size_t idx) const { return data[idx]; }
        
        size_t size() const { return size_; }
        size_t dim() const { return 1; }
        
        std::vector<float>& get_data() { return data; }
        const std::vector<float>& get_data() const { return data; }
        
        void fill(float value) { std::fill(data.begin(), data.end(), value); }
        float sum() const {
            float result = 0.0f;
            for (float v : data) result += v;
            return result;
        }
    };

    /**
     * @class Tensor2D
     * @brief Represents a 2D tensor (matrix) for attention and embedding operations
     */
    class Tensor2D {
    private:
        std::vector<std::vector<float>> data;
        size_t rows_;
        size_t cols_;

    public:
        Tensor2D() : rows_(0), cols_(0) {}
        
        Tensor2D(size_t rows, size_t cols) 
            : data(rows, std::vector<float>(cols, 0.0f)), rows_(rows), cols_(cols) {}
        
        Tensor2D(const std::vector<std::vector<float>>& values) 
            : data(values), rows_(values.size()), cols_(values.empty() ? 0 : values[0].size()) {}

        std::vector<float>& operator[](size_t row) { return data[row]; }
        const std::vector<float>& operator[](size_t row) const { return data[row]; }
        
        float& at(size_t row, size_t col) { return data[row][col]; }
        const float& at(size_t row, size_t col) const { return data[row][col]; }

        size_t rows() const { return rows_; }
        size_t cols() const { return cols_; }
        size_t dim() const { return 2; }
        
        void fill(float value) {
            for (auto& row : data) {
                std::fill(row.begin(), row.end(), value);
            }
        }

        std::vector<std::vector<float>>& get_data() { return data; }
        const std::vector<std::vector<float>>& get_data() const { return data; }
    };

    // ============================================================================
    // MATHEMATICAL OPERATIONS
    // ============================================================================

    /**
     * @class MathOps
     * @brief Collection of mathematical operations for tensor computations
     */
    class MathOps {
    public:
        /**
         * Computes softmax normalization along a dimension
         * @param tensor Input tensor
         * @param temperature Softmax temperature parameter
         * @return Normalized tensor with softmax applied
         */
        static Tensor1D softmax(const Tensor1D& tensor, float temperature = 1.0f) {
            Tensor1D result(tensor.size());
            float max_val = *std::max_element(tensor.get_data().begin(), tensor.get_data().end());
            float sum = 0.0f;

            for (size_t i = 0; i < tensor.size(); ++i) {
                result[i] = std::exp((tensor.get_data()[i] - max_val) / temperature);
                sum += result[i];
            }

            for (size_t i = 0; i < tensor.size(); ++i) {
                result[i] /= sum;
            }

            return result;
        }

        /**
         * Computes top-k values and indices
         * @param tensor Input tensor
         * @param k Number of top elements to retrieve
         * @return Pair of (values, indices) for top-k elements
         */
        static std::pair<std::vector<float>, std::vector<int>> 
        topk(const Tensor1D& tensor, int k) {
            std::vector<std::pair<float, int>> pairs;
            k = std::min(k, static_cast<int>(tensor.size()));

            for (size_t i = 0; i < tensor.size(); ++i) {
                pairs.emplace_back(tensor.get_data()[i], i);
            }

            std::partial_sort(
                pairs.begin(), 
                pairs.begin() + k, 
                pairs.end(),
                [](const auto& a, const auto& b) { return a.first > b.first; }
            );

            std::vector<float> values;
            std::vector<int> indices;
            for (int i = 0; i < k; ++i) {
                values.push_back(pairs[i].first);
                indices.push_back(pairs[i].second);
            }

            return {values, indices};
        }

        /**
         * Computes sinusoidal positional encodings
         * @param seq_len Sequence length
         * @param d_model Model dimension
         * @return 2D tensor of positional encodings
         */
        static Tensor2D compute_positional_encoding(int seq_len, int d_model) {
            Tensor2D pe(seq_len, d_model);
            constexpr float div_constant = 10000.0f;

            for (int pos = 0; pos < seq_len; ++pos) {
                for (int i = 0; i < d_model; ++i) {
                    float div_term = std::pow(div_constant, (2.0f * (i / 2)) / d_model);
                    
                    if (i % 2 == 0) {
                        pe.at(pos, i) = std::sin(pos / div_term);
                    } else {
                        pe.at(pos, i) = std::cos(pos / div_term);
                    }
                }
            }

            return pe;
        }

        /**
         * Computes scaled dot-product attention
         * @param query Query matrix
         * @param key Key matrix
         * @param value Value matrix
         * @return Attention output matrix
         */
        static Tensor2D scaled_dot_product_attention(
            const Tensor2D& query, 
            const Tensor2D& key, 
            const Tensor2D& value) {
            
            size_t seq_len = query.rows();
            size_t d_k = key.cols();
            
            // Compute attention scores: Q * K^T / sqrt(d_k)
            Tensor2D scores(seq_len, seq_len);
            float scale = 1.0f / std::sqrt(static_cast<float>(d_k));

            for (size_t i = 0; i < seq_len; ++i) {
                for (size_t j = 0; j < seq_len; ++j) {
                    float dot_product = 0.0f;
                    for (size_t k = 0; k < d_k; ++k) {
                        dot_product += query.at(i, k) * key.at(j, k);
                    }
                    scores.at(i, j) = dot_product * scale;
                }
            }

            // Apply softmax to get attention weights
            Tensor2D attention_weights(seq_len, seq_len);
            for (size_t i = 0; i < seq_len; ++i) {
                Tensor1D row(seq_len);
                for (size_t j = 0; j < seq_len; ++j) {
                    row[j] = scores.at(i, j);
                }
                Tensor1D softmax_row = softmax(row);
                for (size_t j = 0; j < seq_len; ++j) {
                    attention_weights.at(i, j) = softmax_row[j];
                }
            }

            // Multiply by values: Attention * V
            Tensor2D output(seq_len, value.cols());
            for (size_t i = 0; i < seq_len; ++i) {
                for (size_t j = 0; j < value.cols(); ++j) {
                    float sum = 0.0f;
                    for (size_t k = 0; k < seq_len; ++k) {
                        sum += attention_weights.at(i, k) * value.at(k, j);
                    }
                    output.at(i, j) = sum;
                }
            }

            return output;
        }
    };

    // ============================================================================
    // VISUALIZATION FUNCTIONS
    // ============================================================================

    /**
     * @class Visualization
     * @brief Handles all visualization outputs and formatting
     */
    class Visualization {
    public:
        /**
         * Prints tensor values in readable format
         * @param tensor Input tensor to print
         * @param title Title for the output
         * @param limit Maximum number of elements to display
         */
        static void print_vectors(const Tensor1D& tensor, const std::string& title, int limit = 5) {
            std::cout << "\n--- " << title << " ---\n";
            std::cout << "Shape: [" << tensor.size() << "]\n";
            
            int display_limit = std::min(limit, static_cast<int>(tensor.size()));
            std::cout << "Vector: [";
            
            for (int i = 0; i < display_limit; ++i) {
                std::cout << std::fixed << std::setprecision(4) << tensor.get_data()[i];
                if (i < display_limit - 1) std::cout << ", ";
            }
            
            if (display_limit < tensor.size()) std::cout << ", ...";
            std::cout << "]\n";
        }

        /**
         * Prints 2D tensor (matrix) values
         * @param tensor Input 2D tensor to print
         * @param title Title for the output
         * @param limit Maximum number of rows and columns to display
         */
        static void print_vectors(const Tensor2D& tensor, const std::string& title, int limit = 5) {
            std::cout << "\n--- " << title << " ---\n";
            std::cout << "Shape: [" << tensor.rows() << ", " << tensor.cols() << "]\n";
            
            int row_limit = std::min(limit, static_cast<int>(tensor.rows()));
            int col_limit = std::min(limit, static_cast<int>(tensor.cols()));
            
            for (int i = 0; i < row_limit; ++i) {
                std::cout << "Row " << i << ": [";
                for (int j = 0; j < col_limit; ++j) {
                    std::cout << std::fixed << std::setprecision(4) << tensor.at(i, j);
                    if (j < col_limit - 1) std::cout << ", ";
                }
                if (col_limit < tensor.cols()) std::cout << ", ...";
                std::cout << "]\n";
            }
            
            if (row_limit < tensor.rows()) std::cout << "...\n";
        }

        /**
         * Generates CSV file for probability visualization
         * @param probabilities Probability distribution tensor
         * @param id_to_token Mapping from token IDs to token strings
         * @param output_file Output file path
         * @param top_k Number of top elements to include
         */
        static void save_probabilities_csv(
            const Tensor1D& probabilities,
            const std::map<int, std::string>& id_to_token,
            const std::string& output_file,
            int top_k = TOP_K_DEFAULT) {
            
            auto [values, indices] = MathOps::topk(probabilities, top_k);
            
            std::ofstream file(output_file);
            file << "Token,Probability\n";
            
            for (size_t i = 0; i < values.size(); ++i) {
                int token_id = indices[i];
                std::string token = id_to_token.at(token_id);
                file << token << "," << std::fixed << std::setprecision(6) << values[i] << "\n";
            }
            
            file.close();
            std::cout << "Saved probability distribution to: " << output_file << "\n";
        }

        /**
         * Generates CSV file for attention heatmap visualization
         * @param attention_matrix Attention weights matrix
         * @param x_labels Source token labels (keys)
         * @param y_labels Target token labels (queries)
         * @param output_file Output file path
         */
        static void save_attention_csv(
            const Tensor2D& attention_matrix,
            const std::vector<std::string>& x_labels,
            const std::vector<std::string>& y_labels,
            const std::string& output_file) {
            
            std::ofstream file(output_file);
            
            // Header
            file << "Query";
            for (const auto& label : x_labels) {
                file << "," << label;
            }
            file << "\n";
            
            // Data rows
            for (size_t i = 0; i < attention_matrix.rows(); ++i) {
                file << y_labels[std::min(i, y_labels.size() - 1)];
                for (size_t j = 0; j < attention_matrix.cols(); ++j) {
                    file << "," << std::fixed << std::setprecision(6) << attention_matrix.at(i, j);
                }
                file << "\n";
            }
            
            file.close();
            std::cout << "Saved attention matrix to: " << output_file << "\n";
        }

        /**
         * Generates HTML visualization for attention heatmap
         * @param attention_matrix Attention weights matrix
         * @param x_labels Source token labels
         * @param y_labels Target token labels
         * @param title Title for the visualization
         * @param output_file Output HTML file path
         */
        static void generate_attention_html(
            const Tensor2D& attention_matrix,
            const std::vector<std::string>& x_labels,
            const std::vector<std::string>& y_labels,
            const std::string& title,
            const std::string& output_file) {
            
            std::ofstream file(output_file);
            
            file << R"(<!DOCTYPE html>
<html>
<head>
    <title>)" << title << R"(</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        table { border-collapse: collapse; margin-top: 20px; }
        td, th { border: 1px solid #ddd; padding: 8px; text-align: center; }
        th { background-color: #4CAF50; color: white; }
        .value { font-weight: bold; }
    </style>
</head>
<body>
    <h1>)" << title << R"(</h1>
    <table>
        <tr>
            <th>Query/Key</th>)";
            
            for (const auto& label : x_labels) {
                file << "<th>" << label << "</th>";
            }
            file << "</tr>\n";
            
            for (size_t i = 0; i < attention_matrix.rows(); ++i) {
                file << "<tr><th>" << y_labels[std::min(i, y_labels.size() - 1)] << "</th>";
                
                for (size_t j = 0; j < attention_matrix.cols(); ++j) {
                    float value = attention_matrix.at(i, j);
                    int intensity = static_cast<int>(value * 255);
                    file << "<td style=\"background-color: rgb(" 
                         << intensity << ", " << (255 - intensity) << ", 100);\">"
                         << std::fixed << std::setprecision(3) << value << "</td>";
                }
                file << "</tr>\n";
            }
            
            file << "</table>\n</body>\n</html>";
            file.close();
            std::cout << "Saved HTML visualization to: " << output_file << "\n";
        }
    };

    // ============================================================================
    // TOKENIZER INTERFACE
    // ============================================================================

    /**
     * @class Tokenizer
     * @brief Abstract base class for tokenization operations
     */
    class Tokenizer {
    public:
        virtual ~Tokenizer() = default;
        
        virtual std::vector<int> encode(const std::string& text) = 0;
        virtual std::string decode(int token_id) = 0;
        virtual std::vector<std::string> convert_ids_to_tokens(const std::vector<int>& ids) = 0;
        virtual std::map<int, std::string> get_vocab() = 0;
        virtual int get_vocab_size() = 0;
    };

    /**
     * @class SimpleTokenizer
     * @brief Simple word-level tokenizer for demonstration
     */
    class SimpleTokenizer : public Tokenizer {
    private:
        std::map<std::string, int> token_to_id;
        std::map<int, std::string> id_to_token;
        int vocab_size_;

    public:
        SimpleTokenizer() : vocab_size_(0) {
            // Initialize with common tokens and vocabulary
            initialize_vocab();
        }

    private:
        void initialize_vocab() {
            std::vector<std::string> tokens = {
                "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
                "the", "a", "an", "is", "are", "was", "were",
                "hello", "world", "python", "code", "transformer",
                "neural", "network", "machine", "learning"
            };
            
            for (size_t i = 0; i < tokens.size(); ++i) {
                token_to_id[tokens[i]] = i;
                id_to_token[i] = tokens[i];
            }
            vocab_size_ = tokens.size();
        }

    public:
        std::vector<int> encode(const std::string& text) override {
            std::vector<int> ids;
            std::istringstream iss(text);
            std::string word;
            
            while (iss >> word) {
                auto it = token_to_id.find(word);
                ids.push_back(it != token_to_id.end() ? it->second : 1); // 1 is [UNK]
            }
            
            return ids;
        }

        std::string decode(int token_id) override {
            auto it = id_to_token.find(token_id);
            return it != id_to_token.end() ? it->second : "[UNK]";
        }

        std::vector<std::string> convert_ids_to_tokens(const std::vector<int>& ids) override {
            std::vector<std::string> tokens;
            for (int id : ids) {
                tokens.push_back(decode(id));
            }
            return tokens;
        }

        std::map<int, std::string> get_vocab() override {
            return id_to_token;
        }

        int get_vocab_size() override {
            return vocab_size_;
        }
    };

    // ============================================================================
    // TRANSFORMER MODEL INTERFACE
    // ============================================================================

    /**
     * @struct AttentionOutput
     * @brief Structure to hold attention computation outputs
     */
    struct AttentionOutput {
        Tensor2D output;
        Tensor2D cross_attention;
        Tensor2D decoder_attention;
    };

    /**
     * @class TransformerModel
     * @brief Abstract base class for transformer model operations
     */
    class TransformerModel {
    public:
        virtual ~TransformerModel() = default;
        
        virtual AttentionOutput forward(
            const std::vector<int>& encoder_input_ids,
            const std::vector<int>& decoder_input_ids) = 0;
        
        virtual int get_vocab_size() = 0;
        virtual int get_d_model() = 0;
        virtual int get_decoder_start_token_id() = 0;
        virtual int get_eos_token_id() = 0;
    };

    /**
     * @class SimpleTransformerModel
     * @brief Simplified transformer model for demonstration
     */
    class SimpleTransformerModel : public TransformerModel {
    private:
        int vocab_size_;
        int d_model_;
        int decoder_start_token_id_;
        int eos_token_id_;
        std::shared_ptr<Tokenizer> tokenizer_;

    public:
        SimpleTransformerModel(std::shared_ptr<Tokenizer> tokenizer)
            : vocab_size_(tokenizer->get_vocab_size()),
              d_model_(512),
              decoder_start_token_id_(2),  // [CLS]
              eos_token_id_(3),             // [SEP]
              tokenizer_(tokenizer) {}

        AttentionOutput forward(
            const std::vector<int>& encoder_input_ids,
            const std::vector<int>& decoder_input_ids) override {
            
            // Simulate encoder embeddings
            Tensor2D encoder_embeddings(encoder_input_ids.size(), d_model_);
            encoder_embeddings.fill(0.1f);
            
            // Simulate decoder embeddings
            Tensor2D decoder_embeddings(decoder_input_ids.size(), d_model_);
            decoder_embeddings.fill(0.2f);
            
            // Compute positional encodings
            Tensor2D encoder_pe = MathOps::compute_positional_encoding(
                encoder_input_ids.size(), d_model_);
            
            // Simulate cross-attention and decoder self-attention
            AttentionOutput output;
            output.output = encoder_embeddings;
            output.cross_attention = Tensor2D(decoder_input_ids.size(), encoder_input_ids.size());
            output.decoder_attention = Tensor2D(decoder_input_ids.size(), decoder_input_ids.size());
            
            // Fill with pseudo-attention weights
            for (size_t i = 0; i < output.cross_attention.rows(); ++i) {
                for (size_t j = 0; j < output.cross_attention.cols(); ++j) {
                    output.cross_attention.at(i, j) = 1.0f / output.cross_attention.cols();
                }
            }
            
            for (size_t i = 0; i < output.decoder_attention.rows(); ++i) {
                for (size_t j = 0; j < output.decoder_attention.cols(); ++j) {
                    output.decoder_attention.at(i, j) = 1.0f / output.decoder_attention.cols();
                }
            }
            
            return output;
        }

        int get_vocab_size() override { return vocab_size_; }
        int get_d_model() override { return d_model_; }
        int get_decoder_start_token_id() override { return decoder_start_token_id_; }
        int get_eos_token_id() override { return eos_token_id_; }
    };

    // ============================================================================
    // MAIN TRANSLATION PIPELINE
    // ============================================================================

    /**
     * @class TranslationPipeline
     * @brief Main class orchestrating translation and visualization
     */
    class TranslationPipeline {
    private:
        std::shared_ptr<Tokenizer> tokenizer_;
        std::shared_ptr<TransformerModel> model_;
        int top_k_;
        int max_decode_steps_;

    public:
        TranslationPipeline(
            std::shared_ptr<Tokenizer> tokenizer,
            std::shared_ptr<TransformerModel> model,
            int top_k = TOP_K_DEFAULT,
            int max_decode_steps = MAX_DECODE_STEPS_DEFAULT)
            : tokenizer_(tokenizer), model_(model), top_k_(top_k), 
              max_decode_steps_(max_decode_steps) {}

        /**
         * Runs the complete translation pipeline
         * @param input_sentence Input text to translate
         * @param output_dir Directory to save visualizations
         */
        void run(const std::string& input_sentence, const std::string& output_dir = "./output") {
            std::cout << "\n=== Transformer Translation Pipeline ===\n";
            std::cout << "Input Sentence: \"" << input_sentence << "\"\n";
            
            // Tokenize input
            std::vector<int> encoder_input_ids = tokenizer_->encode(input_sentence);
            std::vector<std::string> encoder_tokens = 
                tokenizer_->convert_ids_to_tokens(encoder_input_ids);
            
            std::cout << "Encoded Input IDs: [";
            for (size_t i = 0; i < encoder_input_ids.size(); ++i) {
                std::cout << encoder_input_ids[i];
                if (i < encoder_input_ids.size() - 1) std::cout << ", ";
            }
            std::cout << "]\n";
            std::cout << "Tokens: [";
            for (size_t i = 0; i < encoder_tokens.size(); ++i) {
                std::cout << encoder_tokens[i];
                if (i < encoder_tokens.size() - 1) std::cout << ", ";
            }
            std::cout << "]\n";
            
            // Visualize positional encodings
            Tensor2D positional_encoding = MathOps::compute_positional_encoding(
                encoder_input_ids.size(), model_->get_d_model());
            std::cout << "\nPositional Encoding computed for sequence length: " 
                     << encoder_input_ids.size() << "\n";
            
            // Initialize decoder
            std::vector<int> decoder_input_ids = {model_->get_decoder_start_token_id()};
            std::vector<std::string> decoder_tokens = {tokenizer_->decode(decoder_input_ids[0])};
            
            // Autoregressive decoding
            std::cout << "\n--- Autoregressive Decoding ---\n";
            
            for (int step = 0; step < max_decode_steps_; ++step) {
                std::cout << "\nStep " << (step + 1) << ":\n";
                
                // Forward pass
                AttentionOutput outputs = model_->forward(encoder_input_ids, decoder_input_ids);
                
                // Simulate logits from output embeddings
                Tensor1D logits(model_->get_vocab_size());
                for (int i = 0; i < model_->get_vocab_size(); ++i) {
                    logits[i] = (static_cast<float>(i) / model_->get_vocab_size());
                }
                
                // Apply softmax
                Tensor1D probabilities = MathOps::softmax(logits);
                
                // Get top-k predictions
                auto [top_probs, top_indices] = MathOps::topk(probabilities, top_k_);
                
                // Select next token (argmax for now)
                int next_token_id = top_indices[0];
                std::string predicted_token = tokenizer_->decode(next_token_id);
                
                std::cout << "Predicted Token: '" << predicted_token << "'\n";
                std::cout << "Top " << top_k_ << " Probabilities:\n";
                for (int i = 0; i < std::min(top_k_, 5); ++i) {
                    std::cout << "  " << tokenizer_->decode(top_indices[i]) 
                             << ": " << std::fixed << std::setprecision(4) << top_probs[i] << "\n";
                }
                
                // Save probability distribution
                std::map<int, std::string> id_to_token = tokenizer_->get_vocab();
                std::string prob_file = output_dir + "/step_" + std::to_string(step + 1) + "_probs.csv";
                Visualization::save_probabilities_csv(probabilities, id_to_token, prob_file, top_k_);
                
                // Update decoder
                decoder_input_ids.push_back(next_token_id);
                decoder_tokens.push_back(predicted_token);
                
                // Check for end-of-sequence
                if (next_token_id == model_->get_eos_token_id()) {
                    std::cout << "End-of-sequence token generated. Stopping decoding.\n";
                    break;
                }
            }
            
            // Final translation
            std::cout << "\n--- Final Results ---\n";
            std::cout << "Generated Tokens: [";
            for (size_t i = 0; i < decoder_tokens.size(); ++i) {
                std::cout << decoder_tokens[i];
                if (i < decoder_tokens.size() - 1) std::cout << ", ";
            }
            std::cout << "]\n";
            
            // Save final attention visualizations
            std::string cross_attn_file = output_dir + "/cross_attention.csv";
            std::string decoder_attn_file = output_dir + "/decoder_attention.csv";
            
            AttentionOutput final_outputs = model_->forward(encoder_input_ids, decoder_input_ids);
            
            Visualization::save_attention_csv(
                final_outputs.cross_attention, 
                encoder_tokens, 
                decoder_tokens, 
                cross_attn_file);
            
            Visualization::save_attention_csv(
                final_outputs.decoder_attention, 
                decoder_tokens, 
                decoder_tokens, 
                decoder_attn_file);
            
            // Generate HTML visualizations
            Visualization::generate_attention_html(
                final_outputs.cross_attention,
                encoder_tokens,
                {decoder_tokens.back()},
                "Cross-Attention (Last Token)",
                output_dir + "/cross_attention.html");
            
            Visualization::generate_attention_html(
                final_outputs.decoder_attention,
                decoder_tokens,
                {decoder_tokens.back()},
                "Masked Self-Attention (Last Token)",
                output_dir + "/decoder_attention.html");
        }
    };

} // namespace TransformerViz

#endif // TRANSFORMER_TRANSLATION_VISUALIZATION_HPP
