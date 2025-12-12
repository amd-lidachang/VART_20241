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
#include "../src/scatter.hpp"

constexpr int nx = 624;
constexpr int ny = 624;
constexpr int in_channels = 64;

DEF_ENV_PARAM(DEBUG_INPUT_SCALE, "0");

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cout << "usage: " << argv[0] << " <input_float_file> <coors_file>"
              << std::endl;
    exit(0);
  }

  // 1. read int8 bin
  // 1. read float bin
  auto input_file = std::string(argv[1]);
  struct stat file_stat;
  if (stat(input_file.c_str(), &file_stat) != 0) {
    std::cerr << "file:" << input_file << " state error!" << std::endl;
    exit(-1);
  }
  auto file_size = file_stat.st_size;
  auto len = file_size;
  LOG(INFO) << "input file size:" << file_size;

  // auto input = std::vector<int8_t>(len);
  // LOG(INFO) << "input data size: " << input.size();
  // CHECK(std::ifstream(input_file).read(reinterpret_cast<char
  // *>(input.data()), input.size()).good());

  auto input = std::vector<float>(len);
  LOG(INFO) << "input data size: " << input.size();
  CHECK(std::ifstream(input_file)
            .read(reinterpret_cast<char*>(input.data()), input.size())
            .good());
  auto input_int8 = std::vector<int8_t>(len);
  for (int i = 0; i < len; ++i) {
    input_int8[i] = (int)(input[i] * 16);
    if (ENV_PARAM(DEBUG_INPUT_SCALE)) {
      if (i < 100) {
        std::cout << "input float[" << i << "]:" << input[i]
                  << ", int8:" << (int)input_int8[i] << std::endl;
      }
    }
  }

  // 2. read coors file
  auto coors_file = std::string(argv[2]);
  if (stat(coors_file.c_str(), &file_stat) != 0) {
    std::cerr << "file:" << coors_file << " state error!" << std::endl;
    exit(-1);
  }
  file_size = file_stat.st_size;
  len = file_size / 4;
  LOG(INFO) << "coors file size:" << file_size;
  // auto coors = vitis::ai::x_autonomous3d::DataContainer<int>({uint32_t(len /
  // 4), 4}, 0);
  std::vector<int> coors(len, 0);
  LOG(INFO) << "coors data size: " << coors.size();
  CHECK(std::ifstream(coors_file)
            .read(reinterpret_cast<char*>(coors.data()), coors.size() * 4)
            .good());

  std::vector<int8_t> output(nx * ny * in_channels);
  // vitis::ai::x_autonomous3d::scatter(coors, 4, input.data(), 1.0,
  // output.data(), 1.0, in_channels, nx, ny);
  vitis::ai::x_autonomous3d::scatter(coors, 4, input_int8.data(), 1.0,
                                     output.data(), 1.0, in_channels, nx, ny);

  LOG(INFO) << "output data size:" << output.size();

  CHECK(
      std::ofstream("middle_input.bin")
          .write(reinterpret_cast<char*>(input_int8.data()), input_int8.size())
          .good());
  CHECK(std::ofstream("middle_output.bin")
            .write(reinterpret_cast<char*>(output.data()),
                   sizeof(int8_t) * output.size())
            .good());
  std::vector<float> output_float(output.size());  // CHW
  auto x_file = std::string("./middle_x_float.bin");
  CHECK(std::ifstream(x_file)
            .read(reinterpret_cast<char*>(output_float.data()),
                  output_float.size() * 4)
            .good());
  for (auto c = 0u; c < 64; ++c) {
    for (auto h = 0u; h < 624; ++h) {
      for (auto w = 0u; w < 624; ++w) {
        output[h * 624 * 64 + w * 64 + c] =
            output_float[c * 624 * 624 + h * 624 + w] * 16;
      }
    }
  }
  LOG(INFO) << "output float data size:" << output_float.size();
  CHECK(std::ofstream("middle_x_float2int8.bin")
            .write(reinterpret_cast<char*>(output.data()),
                   sizeof(int8_t) * output.size())
            .good());
  return 0;
}

