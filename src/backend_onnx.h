#pragma once
#include <string>
#include <vector>

struct ONNXPredictResult {
  std::vector<int> top_indices;
  std::vector<float> top_scores;
};

class ONNXBackend {
public:
  bool load(const std::string& model_path, int num_threads = 1);
  ONNXPredictResult classify(const unsigned char* rgb_data, int width, int height, int channels, int topk = 5);

private:
  void* env_ = nullptr;         // Ort::Env*
  void* session_ = nullptr;     // Ort::Session*
  void* memory_info_ = nullptr; // Ort::MemoryInfo*
  int input_h_ = 224, input_w_ = 224;
};
