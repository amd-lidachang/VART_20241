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
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>
#include <vitis/ai/x_autonomous3d.hpp>

using namespace vitis::ai;

DEF_ENV_PARAM(SAMPLES_BATCH_NUM, "0");

static void read_points_file(const std::string& points_file_name,
                             std::vector<float>& points) {
  struct stat file_stat;
  if (stat(points_file_name.c_str(), &file_stat) != 0) {
    std::cerr << "file:" << points_file_name << " state error!" << std::endl;
    exit(-1);
  }
  auto file_size = file_stat.st_size;
  points.resize(file_size / 4);
  CHECK(std::ifstream(points_file_name)
            .read(reinterpret_cast<char*>(points.data()), file_size)
            .good());
}

int main(int argc, char* argv[]) {
  if (argc < 4) {
    std::cout << "usage:" << argv[0] << " [model_0] [model_1] [input_file]"
              << std::endl;
    exit(0);
  }

  int input_num = argc - 3;
  if (ENV_PARAM(SAMPLES_BATCH_NUM)) {
    input_num = std::min(ENV_PARAM(SAMPLES_BATCH_NUM), input_num);
    // std::cout << "set batch num :" << input_num << std::endl;
  }

  vector<vector<float>> batch_inputs(input_num);
  for (auto i = 0; i < input_num; ++i) {
    std::string file_name = argv[3 + i];
    read_points_file(file_name, batch_inputs[i]);
  }

  std::string model_0 = argv[1];
  std::string model_1 = argv[2];

  auto x_autonomous3d_detector =
      vitis::ai::X_Autonomous3D::create(model_0, model_1);

  auto batch_ret = x_autonomous3d_detector->run(batch_inputs);
  for (auto b = 0u; b < batch_ret.size(); ++b) {
    auto& ret = batch_ret[b];
    std::cout << "batch : " << b << std::endl;
    for (auto i = 0u; i < ret.bboxes.size(); ++i) {
      std::cout << "label: " << ret.bboxes[i].label;
      std::cout << " bbox: ";
      for (auto& j : ret.bboxes[i].bbox) std::cout << j << "  ";
      std::cout << " score: " << ret.bboxes[i].score << std::endl;
    }
  }
  return 0;
}

