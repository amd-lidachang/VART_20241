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

#include "sesrs_onnx.hpp"

using namespace cv;

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "usage :" << argv[0] << " <model_name>"
              << "  <pic1_url> ... " << std::endl;
    abort();
  }
  auto model_name = std::string(argv[1]);
  auto det = OnnxSesrs::create(model_name);
  std::vector<cv::Mat> imgxs;

  for(int i=2; i<argc; i++) {
    Mat imgx = imread(argv[i]);
    if (imgx.empty()) {
      cerr << "can't load image! " << argv[i] << endl;
      return -1;
    }
    imgxs.emplace_back(imgx);
  }

  auto ret = det->run(imgxs);

  for (int k = 0; k < (int)ret.size(); k++) {
     cv::imwrite("res_"+ std::to_string(k) + ".png", ret[k].mat);
  }

  return 0;
}

