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
#pragma once
#include <glog/logging.h>

#include <fstream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace vitis {
namespace ai {
class ImageList {
 public:
  explicit ImageList(const std::string& filename);
  ImageList(const ImageList&) = delete;
  ImageList& operator=(const ImageList& other) = delete;
  virtual ~ImageList();

 public:
  const std::pair<cv::Mat, float> operator[](long i) const;

  bool empty() const { return list_.empty(); }
  size_t size() const { return list_.size(); }

 private:
  std::vector<std::pair<cv::Mat, float>> list_;
};

static std::vector<std::pair<cv::Mat, float>> get_list(
    const std::string& filename) {
  auto ret = std::vector<std::pair<cv::Mat, float>>{};
  std::ifstream fin;
  fin.open(filename, std::ios_base::in);
  if (!fin) {
    std::cout << "Can't open the file " << filename << "\n";
    exit(-1);
  }
  std::string line;
  while (std::getline(fin, line)) {
    if (line.empty()) continue;
    std::istringstream ss(line);
    std::string name = "";  // raw fileneme
    int iso = 0;  // ISO parameter, KSigma transform need this parameter
    ss >> name >> iso;
    cv::Mat img;
    img.create(3000, 4000, CV_16UC1);
    auto mode = std::ios_base::in | std::ios_base::binary;
    CHECK(std::ifstream(name, mode).read((char*)img.data, 24000000).good())
        << "fail to read! filename=" << name;
    ret.push_back(std::make_pair(img, (float)iso));
  }
  fin.close();
  // for (const auto& r : ret) {
  //   LOG(INFO) << "input filename : " << r.first << "   ISO : " << r.second;
  // }
  return ret;
}

inline ImageList::ImageList(const std::string& filename)
    : list_{get_list(filename)} {}

inline ImageList::~ImageList() {}

inline const std::pair<cv::Mat, float> ImageList::operator[](long i) const {
  return list_[i % list_.size()];
}

}  // namespace ai
}  // namespace vitis
