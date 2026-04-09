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

// STATIC NEURAL NETWORK (No Virtual Dispatch / Zero-Overhead)
// By utilizing C++20 Variadic Templates and std::tuple, the compiler unrolls 
// the network topology directly into the instruction cache. This completely 
// eliminates vtable lookups and branching in the hot path during inference.

namespace nexus {

template<typename... Layers>
class StaticNeuralNetwork {
private:
    std::tuple<Layers...> layers_;
    MemoryArena* arena_ = nullptr;
    
    // Statically sized arrays for activation buffers
    std::array<TensorView, sizeof...(Layers)> pre_acts_;
    std::array<TensorView, sizeof...(Layers)> acts_;
    std::array<TensorView, sizeof...(Layers)> grads_;

public:
    StaticNeuralNetwork(MemoryArena& arena, usize batch_size, Layers&&... layers) 
        : arena_(&arena), layers_(std::forward<Layers>(layers)...) {
        
        // C++20 Template Lambda: Pre-allocate contiguous buffers for the whole network
        [&]<std::size_t... I>(std::index_sequence<I...>) {
            (..., (
                pre_acts_[I] = TensorView(arena.allocate_array<f32>(batch_size * std::get<I>(layers_).out_size()), batch_size, std::get<I>(layers_).out_size()),
                acts_[I]   = TensorView(arena.allocate_array<f32>(batch_size * std::get<I>(layers_).out_size()), batch_size, std::get<I>(layers_).out_size()),
                grads_[I]  = TensorView(arena.allocate_array<f32>(batch_size * std::get<I>(layers_).in_size()), batch_size, std::get<I>(layers_).in_size())
            ));
        }(std::make_index_sequence<sizeof...(Layers)>{});
    }

    // Fully inline, unrolled forward pass.
    // The compiler turns this into one continuous block of assembly.
    void forward(const TensorView& input) {
        const TensorView* current_input = &input;
        [&]<std::size_t... I>(std::index_sequence<I...>) {
            // Fold expression dynamically expands across the tuple
            (..., (
                std::get<I>(layers_).forward(*current_input, pre_acts_[I], acts_[I]),
                current_input = &acts_[I]
            ));
        }(std::make_index_sequence<sizeof...(Layers)>{});
    }

    f32 backward(const TensorView& input, const TensorView& target, losses::Loss& loss_fn) {
        usize batch_size = target.rows();
        usize out_dim = target.cols();
        
        TensorView& final_out = acts_.back();
        TensorView out_grad(arena_->allocate_array<f32>(batch_size * out_dim), batch_size, out_dim);
        
        f32 loss = loss_fn.compute(final_out, target, out_grad);
        const TensorView* current_grad = &out_grad;

        // Compile-time reverse unrolling for Backpropagation
        [&]<std::size_t... I>(std::index_sequence<I...>) {
            constexpr std::size_t N = sizeof...(Layers) - 1;
            (..., (
                std::get<N - I>(layers_).backward(
                    *current_grad, 
                    grads_[N - I], 
                    (N - I == 0) ? input : acts_[N - I - 1], 
                    pre_acts_[N - I]
                ),
                current_grad = &grads_[N - I]
            ));
        }(std::make_index_sequence<sizeof...(Layers)>{});
        
        return loss;
    }

    void step(f32 learning_rate) {
        [&]<std::size_t... I>(std::index_sequence<I...>) {
            (..., (std::get<I>(layers_).update_weights(learning_rate)));
        }(std::make_index_sequence<sizeof...(Layers)>{});
    }

    void zero_grad() {
        [&]<std::size_t... I>(std::index_sequence<I...>) {
            (..., (std::get<I>(layers_).zero_grad()));
        }(std::make_index_sequence<sizeof...(Layers)>{});
    }

    void train_step(const TensorView& input, const TensorView& target, losses::Loss& loss_fn, f32 learning_rate) {
        zero_grad();
        forward(input);
        backward(input, target, loss_fn);
        step(learning_rate);
    }
    
    void save(const std::string& filename) const {
        std::ofstream file(filename, std::ios::binary);
        if (!file) throw std::runtime_error("Cannot open file");
        usize num_layers = sizeof...(Layers);
        file.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));
        // Note: Real serialization logic for weights would follow here
    }
    
    void load(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) throw std::runtime_error("Cannot open file");
    }
    
    std::string summary() const {
        std::ostringstream oss;
        oss << "NeuralNetwork Summary:\n======================\n";
        usize total_params = 0, total_memory = 0;
        
        [&]<std::size_t... I>(std::index_sequence<I...>) {
            (..., (
                oss << "Layer " << I << ": " << std::get<I>(layers_).name() << "\n"
                    << "  Parameters: " << std::get<I>(layers_).parameter_count() << "\n"
                    << "  Memory: " << std::get<I>(layers_).memory_footprint() / 1024 << " KB\n",
                total_params += std::get<I>(layers_).parameter_count(),
                total_memory += std::get<I>(layers_).memory_footprint()
            ));
        }(std::make_index_sequence<sizeof...(Layers)>{});
        
        oss << "----------------------\n"
            << "Total Parameters: " << total_params << " (" << total_params / 1000000.0f << "M)\n"
            << "Total Memory: " << total_memory / (1024 * 1024) << " MB\n";
        return oss.str();
    }
};

}