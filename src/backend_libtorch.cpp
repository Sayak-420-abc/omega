#include "backend_libtorch.h"
#ifdef HAS_TORCH_BACKEND
#include <torch/script.h>
#endif
#include <vector>
#include <stdexcept>
#include <algorithm>

static void topk_vec(const float* data, int n, int k, std::vector<int>& idx, std::vector<float>& val) {
  std::vector<int> order(n);
  for (int i=0;i<n;++i) order[i]=i;
  std::partial_sort(order.begin(), order.begin()+k, order.end(),
    [&](int a, int b){ return data[a] > data[b]; });
  idx.assign(order.begin(), order.begin()+k);
  val.resize(k);
  for (int i=0;i<k;++i) val[i]=data[idx[i]];
}

bool TorchBackend::load(const std::string& model_path) {
#ifndef HAS_TORCH_BACKEND
  (void)model_path;
  throw std::runtime_error("LibTorch backend not enabled");
#else
  auto* m = new torch::jit::script::Module(torch::jit::load(model_path));
  m->eval();
  module_ = m;
  return true;
#endif
}

TorchPredictResult TorchBackend::classify(const unsigned char* rgb, int W, int H, int C, int topkN) {
#ifndef HAS_TORCH_BACKEND
  (void)rgb; (void)W; (void)H; (void)C; (void)topkN;
  throw std::runtime_error("LibTorch backend not enabled");
#else
  if (!module_) throw std::runtime_error("Module not loaded");
  if (C != 3) throw std::runtime_error("Expect RGB image");

  const int outH = input_h_, outW = input_w_;
  std::vector<unsigned char> resized(outH*outW*3);
  for (int y=0;y<outH;++y){
    int sy = y * H / outH;
    for (int x=0;x<outW;++x){
      int sx = x * W / outW;
      for (int c=0;c<3;++c){
        resized[(y*outW+x)*3+c] = rgb[(sy*W+sx)*3+c];
      }
    }
  }
  const float mean[3] = {0.485f,0.456f,0.406f};
  const float stdv[3] = {0.229f,0.224f,0.225f};
  std::vector<float> buf(1*3*outH*outW);
  for (int c=0;c<3;++c){
    for (int y=0;y<outH;++y){
      for (int x=0;x<outW;++x){
        float v = resized[(y*outW+x)*3+c]/255.0f;
        buf[c*outH*outW + y*outW + x] = (v-mean[c])/stdv[c];
      }
    }
  }
#ifdef HAS_TORCH_BACKEND
  auto options = torch::TensorOptions().dtype(torch::kFloat32);
  torch::Tensor input = torch::from_blob(buf.data(), {1,3,outH,outW}, options).clone();
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(input);
  torch::Tensor out = ((torch::jit::script::Module*)module_)->forward(inputs).toTensor();
  auto out_contig = out.squeeze().contiguous();
  auto ptr = out_contig.data_ptr<float>();
  int n = out_contig.numel();
  TorchPredictResult r;
  topk_vec(ptr, n, topkN, r.top_indices, r.top_scores);
  return r;
#else
  throw std::runtime_error("LibTorch backend not built");
#endif
#endif
}
