#pragma once
#include "../config.hpp"
#include "../core/arena.hpp"
#include "../core/tensor.hpp"
#include "../math/kernels.hpp"
#include "../math/gemm.hpp"

namespace nexus {

// NEURAL NETWORK LAYERS (nn/layers.hpp + include)

enum class ActivationType {
    None,
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,
    Swish,
    GELU
};

// V2.0 BASELINE DESIGN: The abstract base class.
// Note: In low-latency systems, virtual functions (vtable lookups) in the hot path 
// can stall the CPU pipeline because the jump address isn't known until runtime.
class Layer {
public:
    virtual ~Layer() = default;
    
    virtual void forward(const TensorView& input, TensorView& output) = 0;
    virtual void backward(const TensorView& grad_output,
                         TensorView& grad_input,
                         const TensorView& input,
                         const TensorView& output) = 0;
    
    virtual void zero_grad() = 0;
    virtual void update_weights(f32 learning_rate) = 0;
    
    virtual usize input_size() const = 0;
    virtual usize output_size() const = 0;
    
    virtual usize parameter_count() const { return 0; }
    virtual usize memory_footprint() const { return 0; }
    
    virtual std::string name() const = 0;
};

class DenseLayer : public Layer {
private:
    // Parameters (owned by arena)
    TensorView weights_;      // [output_size x input_size]
    TensorView bias_;         // [output_size]
    
    // Gradients
    TensorView grad_weights_;
    TensorView grad_bias_;
    
    // Adam optimizer state
    TensorView m_weights_;    // First moment
    TensorView v_weights_;    // Second moment
    TensorView m_bias_;
    TensorView v_bias_;
    
    // Forward pass cache
    mutable TensorView pre_activation_;  // Z = XW + b
    
    usize input_size_;
    usize output_size_;
    ActivationType activation_;
    
    // Adam hyperparameters
    f32 beta1_ = 0.9f;
    f32 beta2_ = 0.999f;
    f32 epsilon_ = 1e-8f;
    f32 weight_decay_ = 0.0f;
    mutable f32 timestep_ = 0.0f;
    
    MemoryArena* arena_ = nullptr;

public:
    DenseLayer(MemoryArena& arena,
               usize in_size,
               usize out_size,
               ActivationType act = ActivationType::ReLU)
        : input_size_(in_size)
        , output_size_(out_size)
        , activation_(act)
        , arena_(&arena) {
        
        // Allocate from arena
        weights_.data = arena.allocate_array<f32>(out_size * in_size);
        weights_.ndim = 2;
        weights_.shape_[0] = out_size;
        weights_.shape_[1] = in_size;
        weights_.strides_[0] = in_size;
        weights_.strides_[1] = 1;
        
        bias_.data = arena.allocate_array<f32>(out_size);
        bias_.ndim = 1;
        bias_.shape_[0] = out_size;
        bias_.strides_[0] = 1;
        
        grad_weights_.data = arena.allocate_array<f32>(out_size * in_size);
        grad_weights_.ndim = 2;
        grad_weights_.shape_[0] = out_size;
        grad_weights_.shape_[1] = in_size;
        
        grad_bias_.data = arena.allocate_array<f32>(out_size);
        grad_bias_.ndim = 1;
        grad_bias_.shape_[0] = out_size;
        
        // Adam state
        m_weights_.data = arena.allocate_array<f32>(out_size * in_size);
        m_weights_.ndim = 2;
        m_weights_.shape_[0] = out_size;
        m_weights_.shape_[1] = in_size;
        
        v_weights_.data = arena.allocate_array<f32>(out_size * in_size);
        v_weights_.ndim = 2;
        v_weights_.shape_[0] = out_size;
        v_weights_.shape_[1] = in_size;
        
        m_bias_.data = arena.allocate_array<f32>(out_size);
        m_bias_.ndim = 1;
        m_bias_.shape_[0] = out_size;
        
        v_bias_.data = arena.allocate_array<f32>(out_size);
        v_bias_.ndim = 1;
        v_bias_.shape_[0] = out_size;
        
        // Pre-activation cache
        pre_activation_.data = arena.allocate_array<f32>(out_size);
        pre_activation_.ndim = 1;
        pre_activation_.shape_[0] = out_size;
        
        initialize_weights();
        zero_grad();
        zero_adam_state();
    }
    
    void initialize_weights() {
        // Kaiming/He initialization for ReLU
        std::random_device rd;
        std::mt19937 gen(rd());
        
        f32 scale = 0.0f;
        switch (activation_) {
            case ActivationType::ReLU:
            case ActivationType::Swish:
                scale = std::sqrt(2.0f / input_size_);
                break;
            case ActivationType::Sigmoid:
            case ActivationType::Tanh:
                scale = std::sqrt(1.0f / input_size_);
                break;
            default:
                scale = std::sqrt(1.0f / input_size_);
        }
        
        std::normal_distribution<f32> dist(0.0f, scale);
        
        for (usize i = 0; i < weights_.size(); ++i) {
            weights_.data[i] = dist(gen);
        }
        
        bias_.zero();
    }
    
    void zero_grad() {
        grad_weights_.zero();
        grad_bias_.zero();
    }
    
    void zero_adam_state() {
        m_weights_.zero();
        v_weights_.zero();
        m_bias_.zero();
        v_bias_.zero();
        timestep_ = 0.0f;
    }
    
    void forward(const TensorView& input, TensorView& output) override {
        // Z = X @ W^T + b
        // Input: [batch x input_size], Weights: [output_size x input_size]
        
        // Matrix-vector multiplication for single sample
        for (usize j = 0; j < output_size_; ++j) {
            f32 sum = bias_.data[j];
            for (usize k = 0; k < input_size_; ++k) {
                sum += input.data[k] * weights_(j, k);
            }
            pre_activation_.data[j] = sum;
        }
        
        // Apply activation
        switch (activation_) {
            case ActivationType::ReLU:
                kernels::vec_relu(pre_activation_.data, output.data, output_size_);
                break;
            case ActivationType::Sigmoid:
                kernels::vec_sigmoid(pre_activation_.data, output.data, output_size_);
                break;
            case ActivationType::Tanh:
                kernels::vec_tanh(pre_activation_.data, output.data, output_size_);
                break;
            case ActivationType::Softmax:
                kernels::softmax(pre_activation_.data, output.data, output_size_);
                break;
            case ActivationType::None:
                std::memcpy(output.data, pre_activation_.data, output_size_ * sizeof(f32));
                break;
            default:
                // Fallback to ReLU
                kernels::vec_relu(pre_activation_.data, output.data, output_size_);
        }
    }
    
    // V2.0 BASELINE IMPLEMENTATION: Scalar Backpropagation.
    // Note: Mathematically correct, but this nested for-loop structure completely 
    // bypasses the AVX-512 GEMM engine, making backprop significantly slower than inference.
    void backward(const TensorView& grad_output,
                  TensorView& grad_input,
                  const TensorView& input,
                  const TensorView& output) override {
        
        // Compute gradient of activation
        alignas(NEXUS_SIMD_ALIGNMENT) f32 grad_pre_act[256];  // Max output size
        
        switch (activation_) {
            case ActivationType::ReLU:
                for (usize j = 0; j < output_size_; ++j) {
                    grad_pre_act[j] = pre_activation_.data[j] > 0.0f ? grad_output.data[j] : 0.0f;
                }
                break;
            case ActivationType::Sigmoid:
                for (usize j = 0; j < output_size_; ++j) {
                    f32 s = output.data[j];
                    grad_pre_act[j] = grad_output.data[j] * s * (1.0f - s);
                }
                break;
            case ActivationType::None:
                std::memcpy(grad_pre_act, grad_output.data, output_size_ * sizeof(f32));
                break;
            default:
                std::memcpy(grad_pre_act, grad_output.data, output_size_ * sizeof(f32));
        }
        
        // Gradients w.r.t. weights and bias (Scalar loops)
        for (usize j = 0; j < output_size_; ++j) {
            f32 go = grad_pre_act[j];
            grad_bias_.data[j] += go;
            
            for (usize k = 0; k < input_size_; ++k) {
                grad_weights_(j, k) += go * input.data[k];
            }
        }
        
        // Gradient w.r.t. input (for backprop) (Scalar loops)
        if (grad_input.valid()) {
            for (usize k = 0; k < input_size_; ++k) {
                f32 sum = 0.0f;
                for (usize j = 0; j < output_size_; ++j) {
                    sum += grad_pre_act[j] * weights_(j, k);
                }
                grad_input.data[k] = sum;
            }
        }
    }
    
    void update_weights(f32 learning_rate) override {
        ++timestep_;
        f32 lr_t = learning_rate * std::sqrt(1.0f - std::pow(beta2_, timestep_)) 
                   / (1.0f - std::pow(beta1_, timestep_));
        
        // Update weights with Adam + weight decay
        for (usize i = 0; i < weights_.size(); ++i) {
            f32 g = grad_weights_.data[i] + weight_decay_ * weights_.data[i];
            m_weights_.data[i] = beta1_ * m_weights_.data[i] + (1.0f - beta1_) * g;
            v_weights_.data[i] = beta2_ * v_weights_.data[i] + (1.0f - beta2_) * g * g;
            weights_.data[i] -= lr_t * m_weights_.data[i] / (std::sqrt(v_weights_.data[i]) + epsilon_);
        }
        
        // Update biases (no weight decay)
        for (usize j = 0; j < output_size_; ++j) {
            f32 g = grad_bias_.data[j];
            m_bias_.data[j] = beta1_ * m_bias_.data[j] + (1.0f - beta1_) * g;
            v_bias_.data[j] = beta2_ * v_bias_.data[j] + (1.0f - beta2_) * g * g;
            bias_.data[j] -= lr_t * m_bias_.data[j] / (std::sqrt(v_bias_.data[j]) + epsilon_);
        }
    }
    
    void zero_grad() override {
        grad_weights_.zero();
        grad_bias_.zero();
    }
    
    usize input_size() const override { return input_size_; }
    usize output_size() const override { return output_size_; }
    
    usize parameter_count() const override {
        return weights_.size() + bias_.size();
    }
    
    usize memory_footprint() const override {
        return (weights_.size() + bias_.size() + 
                grad_weights_.size() + grad_bias_.size() +
                m_weights_.size() + v_weights_.size() +
                m_bias_.size() + v_bias_.size()) * sizeof(f32);
    }
    
    std::string name() const override {
        return "Dense(" + std::to_string(input_size_) + " -> " + 
               std::to_string(output_size_) + ")";
    }
};

}
