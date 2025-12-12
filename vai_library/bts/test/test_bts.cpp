/*
 * Copyright 2022-2023 Advanced Micro Devices Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdio.h>

#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vitis/ai/bts.hpp>
using namespace std;

void write_bin(const int8_t* src, int size, const char* file_path) {
  std::cout << "out path: " << file_path << "\n";
  std::ofstream out(file_path, ios::out | ios::binary);
  out.write((char*)src, sizeof(int8_t) * size);
  out.close();
}

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cout << "usage : " << argv[0] << "<model_name>"
              << " <img_url> [<img_url> ...]" << std::endl;
    return -1;
  }

  auto bts = vitis::ai::BTS::create(argv[1], true);
  int width = bts->getInputWidth();
  int height = bts->getInputHeight();

  for (int i = 2; i < argc; i++) {
    cv::Mat image = cv::imread(argv[i]);
    if (image.empty()) {
      std::cout << "cannot load " << argv[i] << std::endl;
      continue;
    }
    cv::Mat img_resize;

    cv::resize(image, img_resize, cv::Size(width, height), 0, 0,
               cv::INTER_LINEAR);
    // cv::imwrite(std::string("resize_after_") + argv[i], img_resize);

    auto result = bts->run(img_resize);
    LOG(INFO) << "width :" << result.width;
    LOG(INFO) << "height:" << result.height;
    std::ofstream f("out_file.txt", ios::out);
    for (auto i = 0; i < result.height; ++i) {
      for (auto j = 0; j < result.width; ++j) {
        // f << result.depth[i * result.width + j] << " ";
        f << result.depth_mat.at<uint16_t>(i, j) / 1000.f << " ";
      }
      f << std::endl;
    }
  }
  return 0;
}
