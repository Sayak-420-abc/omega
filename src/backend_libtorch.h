#pragma once
#include <string>
#include <vector>

struct TorchPredictResult {
  std::vector<int> top_indices;
  std::vector<float> top_scores;
};

class TorchBackend {
public:
  bool load(const std::string& model_path);
  TorchPredictResult classify(const unsigned char* rgb_data, int width, int height, int channels, int topk = 5);
private:
  void* module_ = nullptr; // torch::jit::script::Module*
  int input_h_ = 224, input_w_ = 224;
};
