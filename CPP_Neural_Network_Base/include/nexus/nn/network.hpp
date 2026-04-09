#pragma once
#include "../config.hpp"
#include "../core/arena.hpp"
#include "../core/tensor.hpp"
#include "layers.hpp"
#include "losses.hpp"
#include <vector>
#include <memory>
#include <fstream>
#include <sstream>

namespace nexus {

// NEURAL NETWORK EXECUTION ENGINE (nn/network.cpp + include)

class NeuralNetwork {
private:
    // V2.0 BASELINE TOPOLOGY: Storing layers as unique_ptrs to a base class.
    // Requires heap indirection and forces the CPU to use dynamic dispatch.
    std::vector<std::unique_ptr<Layer>> layers_;
    MemoryArena* arena_ = nullptr;
    
    // Pre-allocated buffers for forward/backward
    std::vector<TensorView> activations_;
    std::vector<TensorView> gradients_;
    
    // Temporary gradient storage for output layer
    TensorView output_grad_;
    
    bool compiled_ = false;

public:
    explicit NeuralNetwork(MemoryArena& arena) : arena_(&arena) {}
    
    template<typename LayerType, typename... Args>
    void add_layer(Args&&... args) {
        if (compiled_) {
            throw std::runtime_error("Cannot add layers after compilation");
        }
        layers_.push_back(std::make_unique<LayerType>(*arena_, std::forward<Args>(args)...));
    }
    
    void compile() {
        if (compiled_) return;
        
        // Pre-allocate all intermediate buffers
        for (usize i = 0; i < layers_.size(); ++i) {
            f32* act_data = arena_->allocate_array<f32>(layers_[i]->output_size());
            TensorView act(act_data, layers_[i]->output_size());
            activations_.push_back(act);
            
            f32* grad_data = arena_->allocate_array<f32>(
                i == 0 ? layers_[0]->input_size() : layers_[i]->input_size()
            );
            TensorView grad(grad_data, 
                i == 0 ? layers_[0]->input_size() : layers_[i]->input_size());
            gradients_.push_back(grad);
        }
        
        // Output gradient buffer
        if (!layers_.empty()) {
            output_grad_.data = arena_->allocate_array<f32>(layers_.back()->output_size());
            output_grad_.ndim = 1;
            output_grad_.shape_[0] = layers_.back()->output_size();
        }
        
        compiled_ = true;
    }
    
    void forward(const TensorView& input, TensorView& output) {
        if (!compiled_) compile();
        
        const TensorView* current_input = &input;
        
        for (usize i = 0; i < layers_.size(); ++i) {
            // VTABLE LOOKUP OCCURS HERE: CPU cannot inline this call.
            layers_[i]->forward(*current_input, activations_[i]);
            current_input = &activations_[i];
        }
        
        // Copy final output
        if (!layers_.empty()) {
            std::memcpy(output.data, current_input->data, 
                       layers_.back()->output_size() * sizeof(f32));
        }
    }
    
    f32 backward(const TensorView& input, const TensorView& target, 
                losses::Loss& loss_fn) {
        if (!compiled_) compile();
        
        // Forward pass to compute activations
        forward(input, activations_.back());
        
        // Compute loss and output gradient
        f32 loss = loss_fn.compute(activations_.back(), target, output_grad_);
        
        // Backpropagate through layers
        TensorView* grad_output = &output_grad_;
        
        for (int i = static_cast<int>(layers_.size()) - 1; i >= 0; --i) {
            const TensorView& layer_input = (i == 0) ? input : activations_[i - 1];
            TensorView& grad_input = gradients_[i];
            
            // VTABLE LOOKUP OCCURS HERE
            layers_[i]->backward(*grad_output, grad_input, layer_input, activations_[i]);
            
            if (i > 0) {
                grad_output = &grad_input;
            }
        }
        
        return loss;
    }
    
    void step(f32 learning_rate) {
        for (auto& layer : layers_) {
            layer->update_weights(learning_rate);
        }
    }
    
    void zero_grad() {
        for (auto& layer : layers_) {
            layer->zero_grad();
        }
    }
    
    void train_step(const TensorView& input, const TensorView& target,
                   losses::Loss& loss_fn, f32 learning_rate) {
        zero_grad();
        f32 loss = backward(input, target, loss_fn);
        step(learning_rate);
    }
    
    void save(const std::string& filename) const {
        std::ofstream file(filename, std::ios::binary);
        if (!file) throw std::runtime_error("Cannot open file for writing");
        
        // Write layer count
        usize num_layers = layers_.size();
        file.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));
        
        // Write each layer's parameters
        for (const auto& layer : layers_) {
            // Write layer type and dimensions
            // (Simplified - would need proper serialization in production)
        }
    }
    
    void load(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) throw std::runtime_error("Cannot open file for reading");
        // Implementation...
    }
    
    std::string summary() const {
        std::ostringstream oss;
        oss << "NeuralNetwork Summary:\n";
        oss << "======================\n";
        
        usize total_params = 0;
        usize total_memory = 0;
        
        for (usize i = 0; i < layers_.size(); ++i) {
            oss << "Layer " << i << ": " << layers_[i]->name() << "\n";
            oss << "  Parameters: " << layers_[i]->parameter_count() << "\n";
            oss << "  Memory: " << layers_[i]->memory_footprint() / 1024 << " KB\n";
            total_params += layers_[i]->parameter_count();
            total_memory += layers_[i]->memory_footprint();
        }
        
        oss << "----------------------\n";
        oss << "Total Parameters: " << total_params << " (" 
            << total_params / 1000000.0f << "M)\n";
        oss << "Total Memory: " << total_memory / (1024 * 1024) << " MB\n";
        
        return oss.str();
    }
};


}
