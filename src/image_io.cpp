#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#include "image_io.h"

bool load_image_rgb(const std::string& path, std::vector<unsigned char>& data, int& w, int& h, int& c) {
  int channels;
  unsigned char* img = stbi_load(path.c_str(), &w, &h, &channels, 3);
  if (!img) return false;
  c = 3;
  data.assign(img, img + (w*h*3));
  stbi_image_free(img);
  return true;
}
