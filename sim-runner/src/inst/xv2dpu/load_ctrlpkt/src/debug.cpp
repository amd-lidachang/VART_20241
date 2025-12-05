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

#include "debug.h"
#include <iomanip>

namespace cosim {
namespace xv2dpu {
namespace load {

#define DEBUG_FILE "./debug/info.txt"

/********************************
 * log
 *********************************/

log::log() {}

// used for clear the debug file contents
void log::init() {
  ofstream ofile(DEBUG_FILE, ios::out | ios::trunc);
  ofile.close();
}

int32_t log::paritycheck(int32_t data) {
  int parity = 0;
  int num = 0;
  int32_t res = 0;
  int32_t temp = data;
  // cout <<"data"<<hex<<data<<endl;
  while (data) {
    if (data & 0x1) {
      num++;
    }
    data >>= 1;
  }
  if (num % 2 == 0) {
    parity = 1;
  } else {
    parity = 0;
  }
  res = (parity << 31) | temp;
  // cout <<"res"<<hex<<res<<endl;
  return res;
}

int32_t log::bd_addr(int32_t bd_num, int32_t wd_num) {
  int32_t res = 0xa0000 + 0x20 * bd_num + 4 * wd_num;
  return res;
}
int32_t log::dma_ctrl_addr(int32_t dma_ctrl_num) {
  int32_t res = 0xa0600 + dma_ctrl_num * 8;
  return res;
}

void log::saveToPktHex(string fileName, std::vector<uint64_t> pkt, int length) {
  char str[30];
  ofstream ofile(fileName, ios::app);
  if (!ofile.is_open()) {
    cout << "fail to open file " << fileName << endl;
    return;
  }
  // cout << "temp"<<endl;
  for (int i = 0; i < length; i++) {
    // cout << hex<<setw(16)<<setfill('0')<<pkt[i]<<endl;
    if ((i % 2) == 0)
      sprintf(str, "TDATA : %016lx ", pkt.at(i));
    else
      sprintf(str, "TLAST : %016lx ", pkt.at(i));
    ofile << str << endl;
  }
  ofile.close();
}

void log::saveToPktint(string fileName, std::vector<uint64_t> pkt, int length) {
  char str[30];
  ofstream ofile(fileName, ios::app);
  if (!ofile.is_open()) {
    cout << "fail to open file " << fileName << endl;
    return;
  }
  // cout << "temp"<<endl;
  for (int i = 0; i < length; i++) {
    if ((i % 2) == 1) ofile << "TLAST" << endl;
    sprintf(str, "%ld %ld", pkt.at(i) & 0xffffffff, pkt.at(i) >> 32);
    ofile << str << endl;
  }
  ofile.close();
}
void log::saveFile(string fileName, char* wbuffer, int length, int size) {
  char* line;
  char str[10];
  line = new char[size];
  ofstream ofile(fileName, ios::out | ios::trunc);
  if (!ofile.is_open()) {
    cout << "fail to open file " << fileName << endl;
    return;
  }

  for (int i = 0; i < length; i++) {
    line[i % size] = wbuffer[i];
    if ((i + 1) % size == 0) {
      for (int j = size - 1; j >= 0; j--) {
        sprintf(str, "%02x", (unsigned char)(line[j]));
        ofile << str;
      }
      ofile << endl;
    }
  }

  ofile.close();

  delete[] line;
}

void log::saveASCIIFile(string fileName, char* wbuffer, int length,
                        int unit_size) {
  int wdata = 0;
  bool wdata_sign;
  uint64_t peak_value = 1 << (8 * unit_size);

  ofstream ofile(fileName, ios::out | ios::trunc);
  if (!ofile.is_open()) {
    cout << "fail to open file " << fileName << endl;
    return;
  }

  for (int i = 0; i < length; i++) {
    int shift = 8 * (i % unit_size);
    wdata = wdata + (((unsigned char)(wbuffer[i])) << shift);
    wdata_sign = wbuffer[i] >> 7;
    if ((i + 1) % unit_size == 0) {
      if (wdata_sign == 0)
        ofile << wdata << endl;
      else {
        wdata = wdata - peak_value;
        ofile << wdata << endl;
      }
      wdata = 0;
    }
  }
}

void log::debug_info(string record_s) {
  ofstream ofile(DEBUG_FILE, ios::out | ios::app);
  if (!ofile.is_open()) {
    cout << "fail to open file " << DEBUG_FILE << endl;
    return;
  }

  ofile << record_s << endl;

  ofile.close();
}

void log::debug_info(string description, void* data, DataType data_type,
                     int shape[2]) {
  ofstream ofile(DEBUG_FILE, ios::out | ios::app);
  if (!ofile.is_open()) {
    cout << "fail to open file " << DEBUG_FILE << endl;
    return;
  }

  ofile << description << endl;

  for (int i = 0; i < shape[0]; i++) {
    for (int j = 0; j < shape[1]; j++) {
      switch (data_type) {
        case INT8: {
          ofile << dec << setw(6) << ((char*)data)[i * shape[1] + j] << " ";
          break;
        }
        case INT16: {
          ofile << dec << setw(6) << ((int16_t*)data)[i * shape[1] + j] << " ";
          break;
        }
        case INT32: {
          ofile << dec << setw(6) << ((int32_t*)data)[i * shape[1] + j] << " ";
          break;
        }
        case INT64: {
          ofile << dec << setw(6) << ((int64_t*)data)[i * shape[1] + j] << " ";
          break;
        }
        default: {
          ofile << dec << setw(6) << ((int16_t*)data)[i * shape[1] + j] << " ";
          break;
        }
      }
    }
    ofile << endl;
  }

  ofile.close();
}

log::~log() {}

}  // namespace load
}  // namespace xv2dpu
}  // namespace cosim
