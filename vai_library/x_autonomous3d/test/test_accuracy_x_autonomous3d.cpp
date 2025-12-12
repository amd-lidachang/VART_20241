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
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>
#include <vitis/ai/x_autonomous3d.hpp>

DEF_ENV_PARAM(DEBUG_X_AUTONOMOUS3D_ACC, "0");

using std::string;
using std::vector;
using namespace vitis::ai;

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

void LoadNames(std::string const& filename,
               std::vector<std::string>& input_file_names) {
  input_file_names.clear();

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
    input_file_names.push_back(name);
  }

  fclose(fp);
}

int main(int argc, char* argv[]) {
  if (argc < 6) {
    // std::cout << "usage: " << argv[0] << " <input_file>" << std::endl;
    std::cout
        << "usage:" << argv[0]
        << " [model_0] [model_1] [input_list] [dataset_path] [output_path]"
        << std::endl;
    exit(0);
  }

  std::string model_0 = argv[1];
  std::string model_1 = argv[2];

  std::string input_file_list = argv[3];
  std::string base_path = argv[4];
  std::string output_path = argv[5];

  std::string cmd = "mkdir -p " + output_path;
  if (system(cmd.c_str()) == -1) {
    std::cerr << "command: " << cmd << " error!" << std::endl;
    exit(-1);
  }

  vector<string> names;
  LoadNames(input_file_list.c_str(), names);
  auto x_autonomous3d_detector =
      vitis::ai::X_Autonomous3D::create(model_0, model_1);

  size_t batch = x_autonomous3d_detector->get_input_batch();

  auto group = 0u;
  if (names.size() % batch) {
    group = names.size() / batch + 1;
  } else {
    group = names.size() / batch;
  }

  for (auto i = 0u; i < group; ++i) {
    std::string input_path;
    size_t input_num = names.size() - i * batch;
    if (input_num > batch) {
      input_num = batch;
    }

    vector<vector<float>> batch_inputs(input_num);
    for (auto j = 0u; j < input_num; ++j) {
      std::string points_file_path = base_path + "/" + names[i * batch + j];
      LOG_IF(INFO, ENV_PARAM(DEBUG_X_AUTONOMOUS3D_ACC))
          << "read: " << points_file_path;
      // 1. read float bin
      read_points_file(points_file_path, batch_inputs[j]);
    }

    auto batch_ret = x_autonomous3d_detector->run(batch_inputs);
    for (auto j = 0u; j < input_num; ++j) {
      auto name = names[i * batch + j];
      if (name.find('.') != std::string::npos) {
        name = name.substr(0, name.find('.'));
      }
      std::string output_file_name = output_path + "/" + name + ".txt";
      LOG_IF(INFO, ENV_PARAM(DEBUG_X_AUTONOMOUS3D_ACC))
          << "output: " << output_file_name;
      std::ofstream out(output_file_name);
      auto& ret = batch_ret[j];
      for (auto& r : ret.bboxes) {
        out << r.label << " ";
        for (auto& b : r.bbox) {
          out << b << " ";
        }
        out << r.score << std::endl;
      }
      out.close();
    }
  }

  return 0;
}

