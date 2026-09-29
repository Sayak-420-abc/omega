#pragma once
#include <string>
#include <vector>

bool load_image_rgb(const std::string& path, std::vector<unsigned char>& data, int& w, int& h, int& c);
