#include "backend_onnx.h"
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <vector>
#include <stdexcept>
#include <array>
#include <iostream>

#if defined(__linux__)
  #include <dlfcn.h>
#elif defined(_WIN32)
  #include <windows.h>
#endif

static bool has_cuda_runtime() {
#if defined(ORT_HAS_CUDA)
  #if defined(__linux__)
    void* h = dlopen("libcuda.so.1", RTLD_LAZY);
    if (h) { dlclose(h); return true; }
    return false;
  #elif defined(_WIN32)
    HMODULE h = LoadLibraryA("nvcuda.dll");
    if (h) { FreeLibrary(h); return true; }
    return false;
  #else
    return false;
  #endif
#else
  return false;
#endif
}

static void topk(const float* data, int n, int k, std::vector<int>& idx, std::vector<float>& val) {
  std::vector<int> order(n);
  for (int i = 0; i < n; ++i) order[i] = i;
  std::partial_sort(order.begin(), order.begin() + k, order.end(),
                    [&](int a, int b) { return data[a] > data[b]; });
  idx.assign(order.begin(), order.begin() + k);
  val.resize(k);
  for (int i = 0; i < k; ++i) val[i] = data[idx[i]];
}

bool ONNXBackend::load(const std::string& model_path, int num_threads) {
  // Warn users if linked against GPU ORT but CUDA runtime isn't available
#if defined(ORT_HAS_CUDA)
  if (!has_cuda_runtime()) {
    std::cerr << "[warn] ONNX Runtime GPU library linked but CUDA runtime not detected. "
                 "This build may fail at runtime or fall back to CPU. "
                 "If you intended CPU-only, rebuild with FORCE_CPU_ORT=ON or fetch assets without ORT_USE_CUDA.\n";
  }
#endif

  Ort::Env* env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "highperf-ai-ml-inference");
  env_ = env;

  Ort::SessionOptions opts;
  opts.SetIntraOpNumThreads(num_threads);
  opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

  Ort::Session* sess = nullptr;
  try {
    sess = new Ort::Session(*env, model_path.c_str(), opts);
  } catch (const Ort::Exception& e) {
    delete env;
    env_ = nullptr;
    throw std::runtime_error(std::string("ONNX load failed: ") + e.what());
  }
  session_ = sess;
  memory_info_ = new Ort::MemoryInfo(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
  return true;
}

ONNXPredictResult ONNXBackend::classify(const unsigned char* rgb, int W, int H, int C, int topkN) {
  if (!session_) throw std::runtime_error("Session not loaded");
  if (C != 3) throw std::runtime_error("Expect 3-channel RGB input");

  const int outH = input_h_, outW = input_w_, outC = 3;
  std::vector<unsigned char> resized(outH * outW * outC);
  for (int y = 0; y < outH; ++y) {
    int sy = y * H / outH;
    for (int x = 0; x < outW; ++x) {
      int sx = x * W / outW;
      for (int c = 0; c < 3; ++c) {
        resized[(y * outW + x) * 3 + c] = rgb[(sy * W + sx) * 3 + c];
      }
    }
  }

  const float mean[3] = {0.485f, 0.456f, 0.406f};
  const float stdv[3] = {0.229f, 0.224f, 0.225f};
  std::vector<float> input(1 * 3 * outH * outW);
  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < outH; ++y) {
      for (int x = 0; x < outW; ++x) {
        int dst_idx = c * outH * outW + y * outW + x;
        unsigned char v = resized[(y * outW + x) * 3 + c];
        float fv = (float)v / 255.0f;
        input[dst_idx] = (fv - mean[c]) / stdv[c];
      }
    }
  }

  Ort::Session* sess = (Ort::Session*)session_;
  Ort::AllocatorWithDefaultOptions allocator;

  auto input_name_alloc  = sess->GetInputNameAllocated(0, allocator);
  auto output_name_alloc = sess->GetOutputNameAllocated(0, allocator);
  const char* input_name  = input_name_alloc.get();
  const char* output_name = output_name_alloc.get();

  std::vector<const char*> input_names{input_name};
  std::vector<const char*> output_names{output_name};

  std::array<int64_t, 4> dims{1, 3, outH, outW};
  Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
      *((Ort::MemoryInfo*)memory_info_), input.data(), input.size(), dims.data(), dims.size());

  auto outputTensors = sess->Run(Ort::RunOptions{nullptr}, input_names.data(), &inputTensor, 1, output_names.data(), 1);
  auto& out = outputTensors.front();

  float* scores = out.GetTensorMutableData<float>();
  size_t n = out.GetTensorTypeAndShapeInfo().GetElementCount();

  ONNXPredictResult r;
  topk(scores, (int)n, topkN, r.top_indices, r.top_scores);
  return r;
}
