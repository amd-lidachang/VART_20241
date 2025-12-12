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

#include <math.h>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include "cache.h"
#include "def.h"

namespace cosim {
namespace xv2dpu {
namespace load {

using namespace std;

enum DataType { INT8, INT16, INT32, INT64 };

class log {
 public:
  log();
  ~log();
  static void init();
  static int32_t paritycheck(int32_t data);
  static int32_t bd_addr(int32_t bd_num, int32_t wd_num);
  static int32_t dma_ctrl_addr(int32_t dma_ctrl_num);
  static void saveToPktHex(string fileName, std::vector<uint64_t> pkt,
                           int length);
  static void saveToPktint(string fileName, std::vector<uint64_t> pkt,
                           int length);
  static void saveFile(string fileName, char* wbuffer, int length, int size);
  static void saveASCIIFile(string fileName, char* wbuffer, int length,
                            int unit_size);
  static void debug_info(string record_s);
  static void debug_info(string description, void* data, DataType data_type,
                         int shape[2]);
};

}  // namespace load
}  // namespace xv2dpu
}  // namespace cosim
