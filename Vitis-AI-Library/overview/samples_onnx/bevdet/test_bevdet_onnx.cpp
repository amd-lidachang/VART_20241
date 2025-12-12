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

#include <glog/logging.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#if _WIN32
#include <codecvt>
#include <locale>
using convert_t = std::codecvt_utf8<wchar_t>;
std::wstring_convert<convert_t, wchar_t> strconverter;
#endif
#include <vitis/ai/profiling.hpp>

#include "./bevdet_onnx.hpp"
#include "process_result.hpp"
using namespace std;
using namespace cv;
void LoadImageNames(std::string const& filename,
                    std::vector<std::string>& images) {
  images.clear();

  /*Check if path is a valid directory path. */
  FILE* fp = fopen(filename.c_str(), "r");
  if (NULL == fp) {
    fprintf(stdout, "open file: %s  error\n", filename.c_str());
    exit(1);
  }

  char buffer[256] = {0};
  while (fgets(buffer, 256, fp) != NULL) {
    int n = strlen(buffer);
    buffer[n - 1] = '\0';
    std::string name = buffer;
    images.push_back(name);
  }

  fclose(fp);
}
int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "usage :" << argv[0] << " <model_name>"
              << " <image_url> [<image_url> ...]" << std::endl;
    abort();
  }
#if _WIN32
  auto model_name = strconverter.from_bytes(std::string(argv[1]));
#else
  auto model_name = std::string(argv[1]);
#endif

  auto runner = BEVDetPtOnnx::create(model_name);

  std::vector<std::string> names;
  LoadImageNames(argv[2], names);
  std::vector<cv::Mat> images;
  for (auto&& i : names) {
    images.emplace_back(cv::imread(i));
  }
  std::vector<std::vector<char>> bins;
  std::vector<std::string> bin_names;
  LoadImageNames(argv[3], bin_names);
  for (auto&& i : bin_names) {
    auto infile = std::ifstream(i, std::ios_base::binary);
    bins.emplace_back(std::vector<char>(std::istreambuf_iterator<char>(infile),
                                        std::istreambuf_iterator<char>()));
  }
  auto res = runner->run(images, bins);
  // res = runner->run(images, bins);
  for (size_t i = 0; i < 32 && i < res.size(); i++) {
    const auto& r = res[i];
    cout << "label: " << r.label << " score: " << r.score
         << " bbox: " << r.bbox[0] << " " << r.bbox[1] << " " << r.bbox[2]
         << " " << r.bbox[3] << " " << r.bbox[4] << " " << r.bbox[5] << " "
         << r.bbox[6] << " " << r.bbox[7] << " " << r.bbox[8] << endl;
  }
  if (res.size()) {
    auto img = draw_bev(res);
    cv::imwrite("bev_out.jpg", img);
  }
  return 0;
}
