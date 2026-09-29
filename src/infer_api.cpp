#include "infer_api.h"
#include "backend_onnx.h"
#include "backend_libtorch.h"
#include <algorithm>
#include <cctype>
#include <stdexcept>

static std::string lower_copy(std::string s){
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return std::tolower(c); });
  return s;
}

bool InferenceEngine::init(const std::string& b, const std::string& model_path, int threads) {
  std::string lb = lower_copy(b);
  if (lb == "onnx") {
#ifdef HAS_ONNX_BACKEND
    auto* impl = new ONNXBackend();
    impl_ = impl;
    return impl->load(model_path, threads);
#else
    (void)model_path;(void)threads;
    return false;
#endif
  } else if (lb == "libtorch" || lb == "torch") {
#ifdef HAS_TORCH_BACKEND
    auto* impl = new TorchBackend();
    impl_ = impl;
    return impl->load(model_path);
#else
    (void)model_path;(void)threads;
    return false;
#endif
  }
  return false;
}

InferResult InferenceEngine::classify(const unsigned char* rgb, int w, int h, int c, int topk) {
  InferResult R;
  if (!impl_) throw std::runtime_error("Engine not initialized");
#ifdef HAS_ONNX_BACKEND
  {
    auto* impl = (ONNXBackend*)impl_;
    auto r = impl->classify(rgb, w, h, c, topk);
    R.indices = std::move(r.top_indices);
    R.scores  = std::move(r.top_scores);
    return R;
  }
#endif
#ifdef HAS_TORCH_BACKEND
  {
    auto* impl = (TorchBackend*)impl_;
    auto r = impl->classify(rgb, w, h, c, topk);
    R.indices = std::move(r.top_indices);
    R.scores  = std::move(r.top_scores);
    return R;
  }
#endif
  throw std::runtime_error("Selected backend not built");
}
