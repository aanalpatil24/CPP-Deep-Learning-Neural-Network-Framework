#pragma once
#include "../config.hpp"
#include "../core/tensor.hpp"
#include <vector>

namespace nexus {

// LOSS FUNCTIONS (nn/losses.hpp + include)

namespace losses {

struct LossResult {
    f32 loss;
    std::vector<f32> gradient;  // Stored in temporary buffer
};

class Loss {
public:
    virtual ~Loss() = default;
    virtual f32 compute(const TensorView& prediction, 
                       const TensorView& target,
                       TensorView& grad_output) = 0;
    virtual std::string name() const = 0;
};

class MSELoss : public Loss {
public:
    f32 compute(const TensorView& pred, 
                const TensorView& target,
                TensorView& grad) override {
        f32 loss = 0.0f;
        usize n = pred.size();
        
        for (usize i = 0; i < n; ++i) {
            f32 diff = pred.data[i] - target.data[i];
            grad.data[i] = 2.0f * diff / static_cast<f32>(n);
            loss += diff * diff;
        }
        
        return loss / static_cast<f32>(n);
    }
    
    std::string name() const override { return "MSE"; }
};

class CrossEntropyLoss : public Loss {
public:
    f32 compute(const TensorView& pred,
                const TensorView& target,
                TensorView& grad) override {
        // Assumes pred is logits (pre-softmax)
        // Numerically stable softmax
        f32 max_logit = pred.data[0];
        for (usize i = 1; i < pred.size(); ++i) {
            max_logit = std::max(max_logit, pred.data[i]);
        }
        
        f32 sum_exp = 0.0f;
        for (usize i = 0; i < pred.size(); ++i) {
            sum_exp += std::exp(pred.data[i] - max_logit);
        }
        
        f32 loss = 0.0f;
        for (usize i = 0; i < pred.size(); ++i) {
            f32 softmax_i = std::exp(pred.data[i] - max_logit) / sum_exp;
            grad.data[i] = softmax_i - target.data[i];
            if (target.data[i] > 0.5f) {
                loss = -std::log(softmax_i + 1e-8f);
            }
        }
        
        return loss;
    }
    
    std::string name() const override { return "CrossEntropy"; }
};

}

}
