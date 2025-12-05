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

#include <stdint.h>
#include <stdio.h>
#include <string>

namespace cosim {
namespace xv2dpu {
namespace load {

using namespace std;

class cache {
 public:
  void writeBank(char* data, int32_t len, int32_t address, int32_t bankID);
  void readBank(char* data, int32_t len, int32_t address, int32_t bankID);
  void writeDDR(char* data, int32_t len, int32_t address);
  void readDDR(char* data, int32_t len, int32_t address);

  cache(const char* file1, const char* file2, const char* binFile1,
        const char* binFile2);
  ~cache();
  string getBankBinFile(int bankID);
  void toLittleEndian(char* inBuffer, int32_t inLen,
                      signed short int* outBuffer, int32_t outLen,
                      int32_t dataSize);
  void toCharList(char* outBuffer, int32_t outLen, signed short int* inBuffer,
                  int32_t inLen, int32_t dataSize);
  void clearVectorReg();
  int64_t* vectorReg;

 private:
  const char* bankFile;
  const char* DDRFile;
  const char* bankBinFile;
  const char* DDRBinFile;

  int DDRWidth;
  int bankWidth;

  void toBinForBank();
  void toBinForDDR();
};

}  // namespace load
}  // namespace xv2dpu
}  // namespace cosim
