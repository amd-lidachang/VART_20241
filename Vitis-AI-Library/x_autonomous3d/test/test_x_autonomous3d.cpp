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

#include <sys/stat.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vitis/ai/profiling.hpp>
#include <vitis/ai/x_autonomous3d.hpp>
using namespace std;

static void read_points_file(const std::string& points_file_name,
                             std::vector<float>& points) {
  struct stat file_stat;
  if (stat(points_file_name.c_str(), &file_stat) != 0) {
    std::cerr << "file:" << points_file_name << " state error!" << std::endl;
    exit(-1);
  }
  auto file_size = file_stat.st_size;
  LOG(INFO) << "input file:" << points_file_name << " size:" << file_size;
  // points_info.points.resize(file_size / 4);
  points.resize(file_size / 4);
  // CHECK(std::ifstream(points_file_name).read(reinterpret_cast<char
  // *>(points_info.points.data()), file_size).good());
  CHECK(std::ifstream(points_file_name)
            .read(reinterpret_cast<char*>(points.data()), file_size)
            .good());
}

int main(int argc, char* argv[]) {
  if (argc < 4) {
    std::cout << "usage: " << argv[0] << " <model1> <model2> <input_file>"
              << std::endl;
    exit(0);
  }

  // 1. read float bin
  auto input_file = std::string(argv[3]);
  std::vector<float> input;
  read_points_file(input_file, input);
  // int read_last_dim = 7;
  int last_dim = 6;
  int points_num = input.size() / last_dim;
  // int points_num = input.size() / read_last_dim;
  // std::cout << "points_num: " << points_num << std::endl;
  //
  // std::vector<float> data(last_dim * points_num);
  // for (int i = 0; i < points_num; ++i) {
  //  for (int j = 0; j < last_dim; ++j) {
  //    data[i * last_dim + j] = input[i * read_last_dim + j + 1];
  //  }
  //}

  // std::cout << "input shape: " << last_dim << " * " << points_num
  //          << std::endl;
  std::cout << "input shape: " << last_dim << " * " << points_num << std::endl;

  std::string model_0 = argv[1];
  std::string model_1 = argv[2];
  auto centerpoint = vitis::ai::X_Autonomous3D::create(model_0, model_1);

  // auto result = centerpoint->run(data);
  auto result = centerpoint->run(input);
  cout << "size: " << result.bboxes.size() << std::endl;
  for (auto& i : result.bboxes) {
    cout << "bbox:   ";
    for (auto& j : i.bbox) cout << j << "  ";
    cout << "score:  " << i.score << " "
         << "label:  " << i.label << endl;
  }
  cout << "visible result: " << result.bboxes.size() << std::endl;
  float HALF_PI = 3.1415926 / 2;
  for (auto& i : result.bboxes) {
    cout << "visible bbox:   ";
    auto bbox = i.bbox;
    bbox[3] = i.bbox[4];
    bbox[4] = i.bbox[3];
    bbox[6] = -1.0 * i.bbox[6] - HALF_PI;
    for (auto& j : bbox) cout << j << "  ";
    cout << "score:  " << i.score << " "
         << "label:  " << i.label << endl;
  }
  return 0;
}

