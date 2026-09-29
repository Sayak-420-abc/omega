#include <iostream>
#include <cxxopts.hpp>
#include "image_io.h"
#include "infer_api.h"

#ifdef HAS_REST_API
#include <httplib.h>
#endif

#include <fstream>
#include <sstream>
#include <vector>   // labels

static std::vector<std::string> load_labels(const std::string& path) {
  std::vector<std::string> labels;
  std::ifstream in(path);
  if (!in.good()) return labels;
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty() && (line.back()=='\r' || line.back()=='\n')) line.pop_back();
    labels.push_back(line);
  }
  return labels;
}

static std::string join_json_array(const std::vector<std::string>& arr) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i=0;i<arr.size();++i) {
    std::string s = arr[i];
    std::string esc; esc.reserve(s.size()+4);
    for (char c: s) {
      if (c=='\"') esc += "\\\"";
      else if (c=='\\') esc += "\\\\";
      else esc += c;
    }
    oss << "\"" << esc << "\"";
    if (i+1<arr.size()) oss << ",";
  }
  oss << "]";
  return oss.str();
}

int main(int argc, char** argv) {
  cxxopts::Options options("highperf-ai-ml-inference", "High-performance C++ ML inference (ONNX Runtime / LibTorch)");
  options.add_options()
    ("backend", "Backend: onnx|libtorch", cxxopts::value<std::string>()->default_value("onnx"))
    ("labels",  "Path to ImageNet labels (.txt, 1 per line)", cxxopts::value<std::string>()->default_value("assets/imagenet_labels.txt"))
    ("model",   "Path to model file (.onnx or .pt)", cxxopts::value<std::string>()->default_value("models/squeezenet1.1.onnx"))
    ("input",   "Path to input image (jpg/png)", cxxopts::value<std::string>()->default_value("assets/sample.jpg"))
    ("topk",    "Top-K predictions", cxxopts::value<int>()->default_value("5"))
    ("threads", "Intra-op threads", cxxopts::value<int>()->default_value("1"))
    ("serve",   "Start REST server on given port", cxxopts::value<int>()->implicit_value("8080"))
    ("h,help",  "Show help");

  auto result = options.parse(argc, argv);
  if (result.count("help")) {
    std::cout << options.help() << "\n";
    return 0;
  }

  const std::string backend = result["backend"].as<std::string>();
  const std::string model   = result["model"].as<std::string>();
  const std::string input   = result["input"].as<std::string>();
  const std::string labels_path = result["labels"].as<std::string>();
  int topk = result["topk"].as<int>();
  int threads = result["threads"].as<int>();

  auto labels = load_labels(labels_path);

  InferenceEngine engine;
  if (!engine.init(backend, model, threads)) {
    std::cerr << "Failed to initialize inference engine\n";
    return 1;
  }

  if (result.count("serve")) {
#ifndef HAS_REST_API
    std::cerr << "Built without REST API\n";
    return 1;
#else
    int port = result["serve"].as<int>();
    httplib::Server svr;

    svr.Post("/predict",
      [&](const httplib::Request& req, httplib::Response& res){
        auto it = req.params.find("file");
        if (it == req.params.end()) {
          res.status = 400;
          res.set_content("{\"error\":\"use ?file=path\"}", "application/json");
          return;
        }
        std::string path = it->second;

        std::vector<unsigned char> rgb; int w=0,h=0,c=0;
        if (!load_image_rgb(path, rgb, w, h, c)) {
          res.status = 400;
          res.set_content("{\"error\":\"failed to load image\"}", "application/json");
          return;
        }
        auto r = engine.classify(rgb.data(), w, h, c, 5);

        std::vector<std::string> names;
        for (auto idx : r.indices) {
          if (!labels.empty() && idx >= 0 && (size_t)idx < labels.size()) names.push_back(labels[idx]);
          else names.push_back("class_" + std::to_string(idx));
        }
        std::string json = "{\"top_indices\":[";
        for (size_t i=0;i<r.indices.size();++i){
          json += std::to_string(r.indices[i]);
          if (i+1<r.indices.size()) json += ",";
        }
        json += "],\"top_scores\":[";
        for (size_t i=0;i<r.scores.size();++i){
          json += std::to_string(r.scores[i]);
          if (i+1<r.scores.size()) json += ",";
        }
        json += "],\"top_labels\":" + join_json_array(names) + "}";
        res.set_content(json, "application/json");
      });

    std::cout << "Serving on http://0.0.0.0:" << port << "  (POST /predict?file=assets/sample.jpg)\n";
    svr.listen("0.0.0.0", port);
    return 0;
#endif
  }

  // CLI single-shot
  std::vector<unsigned char> rgb; int w=0,h=0,c=0;
  if (!load_image_rgb(input, rgb, w, h, c)) {
    std::cerr << "Failed to load image: " << input << "\n";
    return 1;
  }
  auto r = engine.classify(rgb.data(), w, h, c, topk);
  std::cout << "Top-" << topk << " indices: ";
  for (auto i : r.indices) std::cout << i << " ";
  std::cout << "\nScores: ";
  for (auto s : r.scores) std::cout << s << " ";
  std::cout << "\n";
  if (!labels.empty()) {
    std::cout << "Labels: ";
    for (auto i : r.indices) {
      if (i >= 0 && (size_t)i < labels.size()) std::cout << labels[i] << " | ";
      else std::cout << "class_" << i << " | ";
    }
    std::cout << "\n";
  }
  return 0;
}
