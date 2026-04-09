#pragma once
#include "../config.hpp"
#include "../core/arena.hpp"
#include "../core/tensor.hpp"
#include "../math/kernels.hpp"
#include "../math/gemm.hpp"

namespace nexus {

// NEURAL NETWORK LAYERS

enum class ActivationType { None, ReLU, Sigmoid, Tanh, Softmax, Swish, GELU };

class DenseLayer {
private:
    TensorView weights_;      // Stored row-major [output_size x input_size]
    TensorView bias_;         // [output_size]
    
    TensorView grad_weights_;
    TensorView grad_bias_;
    
    TensorView m_weights_;    // Adam First moment for fast SGD convergence
    TensorView v_weights_;    // Adam Second moment
    TensorView m_bias_;
    TensorView v_bias_;
    
    usize input_size_;
    usize output_size_;
    ActivationType activation_;
    
    f32 beta1_ = 0.9f;
    f32 beta2_ = 0.999f;
    f32 epsilon_ = 1e-8f;
    f32 weight_decay_ = 0.0f;
    f32 timestep_ = 0.0f;
    
    MemoryArena* arena_ = nullptr;

public:
    DenseLayer(MemoryArena& arena, usize in_size, usize out_size, ActivationType act = ActivationType::ReLU)
        : input_size_(in_size), output_size_(out_size), activation_(act), arena_(&arena) {
        
        weights_.data = arena.allocate_array<f32>(out_size * in_size);
        weights_.ndim = 2; weights_.shape_[0] = out_size; weights_.shape_[1] = in_size;
        weights_.strides_[0] = in_size; weights_.strides_[1] = 1;
        
        bias_.data = arena.allocate_array<f32>(out_size);
        bias_.ndim = 1; bias_.shape_[0] = out_size; bias_.strides_[0] = 1;
        
        grad_weights_.data = arena.allocate_array<f32>(out_size * in_size);
        grad_weights_.ndim = 2; grad_weights_.shape_[0] = out_size; grad_weights_.shape_[1] = in_size;
        grad_weights_.strides_[0] = in_size; grad_weights_.strides_[1] = 1;
        
        grad_bias_.data = arena.allocate_array<f32>(out_size);
        grad_bias_.ndim = 1; grad_bias_.shape_[0] = out_size;
        
        m_weights_.data = arena.allocate_array<f32>(out_size * in_size);
        m_weights_.ndim = 2; m_weights_.shape_[0] = out_size; m_weights_.shape_[1] = in_size;
        
        v_weights_.data = arena.allocate_array<f32>(out_size * in_size);
        v_weights_.ndim = 2; v_weights_.shape_[0] = out_size; v_weights_.shape_[1] = in_size;
        
        m_bias_.data = arena.allocate_array<f32>(out_size);
        m_bias_.ndim = 1; m_bias_.shape_[0] = out_size;
        
        v_bias_.data = arena.allocate_array<f32>(out_size);
        v_bias_.ndim = 1; v_bias_.shape_[0] = out_size;
        
        initialize_weights();
        zero_grad();
        zero_adam_state();
    }
    
    void initialize_weights() {
        std::mt19937 gen(42); 
        f32 scale = 0.0f;
        switch (activation_) {
            case ActivationType::ReLU:
            case ActivationType::Swish:
                scale = std::sqrt(2.0f / input_size_); break;
            default:
                scale = std::sqrt(1.0f / input_size_);
        }
        
        std::normal_distribution<f32> dist(0.0f, scale);
        for (usize i = 0; i < weights_.size(); ++i) weights_.data[i] = dist(gen);
        bias_.zero();
    }
    
    void zero_grad() {
        grad_weights_.zero();
        grad_bias_.zero();
    }
    
    void zero_adam_state() {
        m_weights_.zero(); v_weights_.zero();
        m_bias_.zero(); v_bias_.zero();
        timestep_ = 0.0f;
    }
    
    void forward(const TensorView& input, TensorView& pre_act, TensorView& output) {
        usize batch_size = input.rows();
        
        // Z = X * W^T
        // To use the AVX GEMM engine, we transpose W so its dimensions map correctly.
        f32* W_T = arena_->allocate_array<f32>(input_size_ * output_size_);
        gemm::transpose_blocked(weights_.data, W_T, output_size_, input_size_);
        
        gemm::gemm_blocked(input.data, W_T, pre_act.data, 
                           batch_size, output_size_, input_size_, 
                           input_size_, output_size_, output_size_);
        
        // Bias addition is vectorized inline
        for (usize i = 0; i < batch_size; ++i) {
            kernels::vec_add(pre_act.data + i * output_size_, bias_.data, pre_act.data + i * output_size_, output_size_);
        }
        
        // Apply SIMD activation functions
        switch (activation_) {
            case ActivationType::ReLU: kernels::vec_relu(pre_act.data, output.data, batch_size * output_size_); break;
            case ActivationType::Sigmoid: kernels::vec_sigmoid(pre_act.data, output.data, batch_size * output_size_); break;
            case ActivationType::Tanh: kernels::vec_tanh(pre_act.data, output.data, batch_size * output_size_); break;
            case ActivationType::Softmax: 
                for (usize i = 0; i < batch_size; ++i) kernels::softmax(pre_act.data + i * output_size_, output.data + i * output_size_, output_size_);
                break;
            case ActivationType::None: std::memcpy(output.data, pre_act.data, batch_size * output_size_ * sizeof(f32)); break;
            default: kernels::vec_relu(pre_act.data, output.data, batch_size * output_size_);
        }
    }
    
    // Fully Vectorized Backpropagation Pipeline
    void backward(const TensorView& grad_output, TensorView& grad_input,
                  const TensorView& input, const TensorView& pre_act) {
        
        usize batch_size = grad_output.rows();
        f32* grad_pre_act = arena_->allocate_array<f32>(batch_size * output_size_);
        
        // 1. Vectorized Activation Derivative Calculation
        if (activation_ == ActivationType::ReLU) {
            #ifdef NEXUS_HAS_AVX512
            __m512 zero = _mm512_setzero_ps();
            for (usize i = 0; i < batch_size * output_size_; i += 16) {
                __m512 z_val = _mm512_loadu_ps(pre_act.data + i);
                __m512 g_out = _mm512_loadu_ps(grad_output.data + i);
                
                // AVX-512 intrinsic to create a bitmask (1 if z_val > 0, else 0)
                __mmask16 mask = _mm512_cmp_ps_mask(z_val, zero, _CMP_GT_OQ);
                
                // Apply the mask to the gradient (equivalent to branchless `if z>0`)
                __m512 g_in = _mm512_maskz_mov_ps(mask, g_out);
                _mm512_storeu_ps(grad_pre_act + i, g_in);
            }
            #else
            for (usize i = 0; i < batch_size * output_size_; ++i) {
                grad_pre_act[i] = pre_act.data[i] > 0.0f ? grad_output.data[i] : 0.0f;
            }
            #endif
        } else {
            // Simplify other derivatives for standard memcpy fallback
            std::memcpy(grad_pre_act, grad_output.data, batch_size * output_size_ * sizeof(f32));
        }

        // 2. Vectorized Weight Gradients: dW = grad_pre_act^T * Input
        // Replaces slow scalar nested loops with massive AVX-512 GEMM throughput.
        f32* gZ_T = arena_->allocate_array<f32>(output_size_ * batch_size);
        gemm::transpose_blocked(grad_pre_act, gZ_T, batch_size, output_size_);
        
        gemm::gemm_blocked(gZ_T, input.data, grad_weights_.data, 
                           output_size_, input_size_, batch_size, 
                           batch_size, input_size_, input_size_);

        // 3. Vectorized Input Gradients: dX = grad_pre_act * Weights
        if (grad_input.valid()) {
            gemm::gemm_blocked(grad_pre_act, weights_.data, grad_input.data,
                               batch_size, input_size_, output_size_,
                               output_size_, input_size_, input_size_);
        }
        
        // 4. Bias Gradients
        for (usize j = 0; j < output_size_; ++j) {
            f32 sum = 0;
            for (usize i = 0; i < batch_size; ++i) sum += grad_pre_act[i * output_size_ + j];
            grad_bias_.data[j] = sum;
        }
    }
    
    // Parameter optimization via Adam (Adaptive Moment Estimation)
    void update_weights(f32 learning_rate) {
        timestep_ += 1.0f;
        f32 lr_t = learning_rate * std::sqrt(1.0f - std::pow(beta2_, timestep_)) 
                   / (1.0f - std::pow(beta1_, timestep_));
        
        for (usize i = 0; i < weights_.size(); ++i) {
            f32 g = grad_weights_.data[i] + weight_decay_ * weights_.data[i];
            m_weights_.data[i] = beta1_ * m_weights_.data[i] + (1.0f - beta1_) * g;
            v_weights_.data[i] = beta2_ * v_weights_.data[i] + (1.0f - beta2_) * g * g;
            weights_.data[i] -= lr_t * m_weights_.data[i] / (std::sqrt(v_weights_.data[i]) + epsilon_);
        }
        
        for (usize j = 0; j < output_size_; ++j) {
            f32 g = grad_bias_.data[j];
            m_bias_.data[j] = beta1_ * m_bias_.data[j] + (1.0f - beta1_) * g;
            v_bias_.data[j] = beta2_ * v_bias_.data[j] + (1.0f - beta2_) * g * g;
            bias_.data[j] -= lr_t * m_bias_.data[j] / (std::sqrt(v_bias_.data[j]) + epsilon_);
        }
    }
    
    usize in_size() const { return input_size_; }
    usize out_size() const { return output_size_; }
    
    usize parameter_count() const { return weights_.size() + bias_.size(); }
    usize memory_footprint() const {
        return (weights_.size() + bias_.size() + grad_weights_.size() + grad_bias_.size() +
                m_weights_.size() + v_weights_.size() + m_bias_.size() + v_bias_.size()) * sizeof(f32);
    }
    std::string name() const { return "Dense(" + std::to_string(input_size_) + "->" + std::to_string(output_size_) + ")"; }
};

}