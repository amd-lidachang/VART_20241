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

#ifndef __UTIL_HPP__
#define __UTIL_HPP__

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <string>
#include <iostream>

namespace cosim {
namespace xv2dpu {
namespace alu {

using namespace std;

class Util {
 private:
  Util() {}
  ~Util() {}

 public:
  template <typename T>
  static void Random(T* data, uint64_t size, T low, T high, uint64_t seed);
  template <typename T>
  static void SaveDec(const string& save_name, const T* data, uint64_t size,
                      int line_size = 16, int mode = std::ios::app);
};

template <typename T>
void Util::Random(T* data, uint64_t size, T low, T high, uint64_t seed) {
  // std::random_device rd;
  std::mt19937 gen(seed);
  std::uniform_int_distribution<T> dis(low, high);
  for (auto i = 0U; i < size; i++) {
    data[i] = (seed == 0) ? 0 : dis(gen);
  }
}

template <typename T>
void Util::SaveDec(const string& save_name, const T* data, uint64_t size,
                   int line_size, int mode) {
  std::ofstream f;
  if (mode == std::ios::trunc) {
    f.open(save_name, std::ios::trunc);
  } else {
    f.open(save_name, std::ios::app);
  }

  int line_num = size / line_size + 1;
  int line_left = size % line_size;

  for (int i = 0; i < line_num; i++) {
    if (i == line_num - 1 && line_left == 0) break;
    for (int j = 0; j < line_size; j++) {
      if (i == line_num - 1 && j > line_left) break;
      if (j != 0 && j % 4 == 0)
        f << " "
          << " "
          << " "
          << " ";
      if (line_size == 1) {
        f << dec << +data[i * line_size + j];
      } else {
        if (sizeof(T) == 1) {
          f << dec << setw(6) << setfill(' ') << +data[i * line_size + j];
        } else if (sizeof(T) == 2) {
          f << dec << setw(9) << setfill(' ') << +data[i * line_size + j];
        } else if (sizeof(T) == 4) {
          f << dec << setw(6) << setfill(' ') << +data[i * line_size + j];
        } else if (sizeof(T) == 8) {
          f << dec << setw(24) << setfill(' ') << +data[i * line_size + j];
        } else {
          assert("Not support type, sizeof(T)");
          abort();
        }
      }
    }
    f << endl;
  }
  f.close();
}

enum class CTRL_PKT_TYPE : int {
  MT_TO_STREAM = 0,
  LAYER_PARAMETER = 1,
  MISC_DMA = 2,
  START_QUEUE = 3,
  STREAM_TO_MT = 4
};

std::string get_ctrlpkt_file_name(CTRL_PKT_TYPE ctrlpkt_type) {
  std::string suffix = ".bin";
  switch (ctrlpkt_type) {
    case CTRL_PKT_TYPE::MT_TO_STREAM: {
      return "";
    }
    case CTRL_PKT_TYPE::LAYER_PARAMETER: {
      return "";
    }
    case CTRL_PKT_TYPE::MISC_DMA: {
      return "";
    }
    case CTRL_PKT_TYPE::START_QUEUE: {
      return "";
    }
    case CTRL_PKT_TYPE::STREAM_TO_MT: {
      return "";
    }
    default: { assert(false && "error ctrlpkt type"); }
  }
  return "";
}

}  // namespace alu
}  // namespace xv2dpu
}  // namespace cosim

#endif /*__UTIL_HPP__*/
