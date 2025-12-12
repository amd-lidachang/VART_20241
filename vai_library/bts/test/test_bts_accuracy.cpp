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
  if (argc < 5) {
    std::cout << "usage : " << argv[0] << " <model_name>"
              << " <dataset_path> <filelist> <output_path>" << std::endl;
    return -1;
  }

  auto bts = vitis::ai::BTS::create(argv[1], true);
  int width = bts->getInputWidth();
  int height = bts->getInputHeight();

  auto dataset_path = std::string(argv[2]);
  auto filelist = std::string(argv[3]);
  auto output_path = std::string(argv[4]);
  vector<string> names;
  LoadImageNames(filelist.c_str(), names);

  for (auto l : names) {
    auto full_name = dataset_path + "/" + l;
    cv::Mat image = cv::imread(full_name);
    if (image.empty()) {
      std::cout << "cannot load " << full_name << std::endl;
      continue;
    }
    std::cout << std::endl;
    cv::Mat img_resize;

    cv::resize(image, img_resize, cv::Size(width, height), 0, 0,
               cv::INTER_LINEAR);
    // cv::imwrite(std::string("resize_after_") + argv[i], img_resize);

    auto result = bts->run(img_resize);
    // LOG(INFO) << "result:" << &result;

    // vector<uint16_t> bitmap(result.height * result.width);
    // for (auto i = 0; i < result.height; ++i) {
    //  for (auto j = 0; j < result.width; ++j) {
    //    auto index = i * result.width + j;
    //    bitmap[index] = result.depth[index] * 1000;
    //  }
    //}
    // auto m =
    //    cv::Mat(result.height, result.width, CV_16U, bitmap.data()).clone();
    auto mid_dir_pos = l.find('/');
    if (mid_dir_pos == std::string::npos) {
      LOG(WARNING) << "name :" << l << " error!";
      continue;
    }
    l[mid_dir_pos] = '_';
    l.replace(l.size() - 3, 3, "png");
    auto out_name = output_path + "/" + l;
    // LOG(INFO) << "use depth mat new:";
    LOG(INFO) << "output name :" << out_name;
    // cv::imwrite(out_name, m);
    cv::imwrite(out_name, result.depth_mat);
  }
  return 0;
}
