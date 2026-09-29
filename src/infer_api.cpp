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

InferenceEngine::~InferenceEngine() {
  if (impl_) {
    if (backend_ == Backend::ONNX) {
#ifdef HAS_ONNX_BACKEND
      delete static_cast<ONNXBackend*>(impl_);
#endif
    } else if (backend_ == Backend::TORCH) {
#ifdef HAS_TORCH_BACKEND
      delete static_cast<TorchBackend*>(impl_);
#endif
    }
    impl_ = nullptr;
  }
}

InferenceEngine::InferenceEngine(InferenceEngine&& other) noexcept
    : backend_(other.backend_), impl_(other.impl_) {
  other.impl_ = nullptr;
}

InferenceEngine& InferenceEngine::operator=(InferenceEngine&& other) noexcept {
  if (this != &other) {
    this->~InferenceEngine();
    backend_ = other.backend_;
    impl_ = other.impl_;
    other.impl_ = nullptr;
  }
  return *this;
}

bool InferenceEngine::init(const std::string& b, const std::string& model_path, int threads) {
  std::string lb = lower_copy(b);
  if (lb == "onnx") {
#ifdef HAS_ONNX_BACKEND
    backend_ = Backend::ONNX;
    auto* impl = new ONNXBackend();
    impl_ = impl;
    return impl->load(model_path, threads);
#else
    (void)model_path;(void)threads;
    return false;
#endif
  } else if (lb == "libtorch" || lb == "torch") {
#ifdef HAS_TORCH_BACKEND
    backend_ = Backend::TORCH;
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

  if (backend_ == Backend::ONNX) {
#ifdef HAS_ONNX_BACKEND
    auto* impl = static_cast<ONNXBackend*>(impl_);
    auto r = impl->classify(rgb, w, h, c, topk);
    R.indices = std::move(r.top_indices);
    R.scores  = std::move(r.top_scores);
    return R;
#else
    throw std::runtime_error("ONNX backend not enabled");
#endif
  } else if (backend_ == Backend::TORCH) {
#ifdef HAS_TORCH_BACKEND
    auto* impl = static_cast<TorchBackend*>(impl_);
    auto r = impl->classify(rgb, w, h, c, topk);
    R.indices = std::move(r.top_indices);
    R.scores  = std::move(r.top_scores);
    return R;
#else
    throw std::runtime_error("LibTorch backend not enabled");
#endif
  }
  throw std::runtime_error("Unknown backend type");
}

