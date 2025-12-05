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

#include "cache.h"
#include <stdio.h>
#include <string.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include "def.h"

namespace cosim {
namespace xv2dpu {
namespace load {

using namespace std;

cache::cache(const char* file1, const char* file2, const char* binFile1,
             const char* binFile2)
    : bankFile(file1),
      DDRFile(file2),
      bankBinFile(binFile1),
      DDRBinFile(binFile2) {
  // toBinForBank();
  // toBinForDDR();
  // vectorReg = new int64_t[COLUMNS*NAX_CHANNELS];
  // clearVectorReg();
}

cache::~cache() { delete[] vectorReg; }

void cache::clearVectorReg() {
  memset(vectorReg, 0, COLUMNS * NAX_CHANNELS * sizeof(int64_t));
}

void cache::toBinForDDR() {
  char* buffer;
  string line;
  int32_t rdAddr;
  char rdValueStr[200];

  ifstream ifile(DDRFile, ios::in);
  if (!ifile.is_open()) {
    cout << "fail to open file " << DDRFile << endl;
    return;
  }

  ofstream ofile(DDRBinFile, ios::out | ios::binary | ios::trunc);
  if (!ofile.is_open()) {
    cout << "fail to open file " << DDRBinFile << endl;
    return;
  }

  buffer = new char[AXI_BW];
  while (getline(ifile, line)) {
    int strPos;
    sscanf(line.c_str(), "%x : %s\n", &rdAddr, rdValueStr);

    strPos = strlen(rdValueStr) - 2;
    for (int i = 0; i < AXI_BW; i++) {
      char data{};

      if (rdValueStr[strPos] <= '9' && rdValueStr[strPos] >= '0') {
        data = rdValueStr[strPos] - '0';
      } else if (rdValueStr[strPos] <= 'F' && rdValueStr[strPos] >= 'A') {
        data = rdValueStr[strPos] - 'A' + 10;
      } else if (rdValueStr[strPos] <= 'f' && rdValueStr[strPos] >= 'a') {
        data = rdValueStr[strPos] - 'a' + 10;
      }

      if (rdValueStr[strPos + 1] <= '9' && rdValueStr[strPos + 1] >= '0') {
        data = data * 16 + rdValueStr[strPos + 1] - '0';
      } else if (rdValueStr[strPos + 1] <= 'F' &&
                 rdValueStr[strPos + 1] >= 'A') {
        data = data * 16 + rdValueStr[strPos + 1] - 'A' + 10;
      } else if (rdValueStr[strPos + 1] <= 'f' &&
                 rdValueStr[strPos + 1] >= 'a') {
        data = data * 16 + rdValueStr[strPos + 1] - 'a' + 10;
      }

      buffer[i] = data;
      strPos -= 2;
    }
    ofile.seekp(rdAddr, ios::beg);
    ofile.write(buffer, AXI_BW);
  }

  ifile.close();
  ofile.close();

  delete[] buffer;
}

string cache::getBankBinFile(int bankID) {
  string eachBankBinFile = "";

  eachBankBinFile.append(bankBinFile);

  switch (bankID) {
    case 0: {
      eachBankBinFile.append("_weight");
      break;
    }
    case 1: {
      eachBankBinFile.append("_vector");
      break;
    }
    case 2: {
      eachBankBinFile.append("_bias");
      break;
    }
    case 3: {
      eachBankBinFile.append("_shared0");
      break;
    }
    case 4: {
      eachBankBinFile.append("_shared1");
      break;
    }
    case 5: {
      eachBankBinFile.append("_shared2");
      break;
    }
    case 6: {
      eachBankBinFile.append("_shared3");
      break;
    }
    case 7: {
      eachBankBinFile.append("_shared4");
      break;
    }
    case 8: {
      eachBankBinFile.append("_shared5");
      break;
    }
    case 9: {
      eachBankBinFile.append("_shared6");
      break;
    }
    case 10: {
      eachBankBinFile.append("_shared7");
      break;
    }
  }  // switch

  return eachBankBinFile;
}

void cache::toBinForBank() {
  string eachBankBinFile;
  int WEIGHT_GROUP_BYTES = WEIGHT_GROUP_BW * WEIGHT_GROUP_DEPTH;
  int VECTOR_BANK_BYTES = VECTOR_BANK_BW * VECTOR_BANK_DEPTH;
  int BIAS_BANK_BYTES = BIAS_BANK_BW * BIAS_BANK_DEPTH;
  int SHARED_BANK_BYTES = SHARED_BANK_BW * SHARED_BANK_DEPTH;

  int BytesInBank[11] = {
      WEIGHT_GROUP_BYTES, VECTOR_BANK_BYTES, BIAS_BANK_BYTES,
      SHARED_BANK_BYTES,  SHARED_BANK_BYTES, SHARED_BANK_BYTES,
      SHARED_BANK_BYTES,  SHARED_BANK_BYTES, SHARED_BANK_BYTES,
      SHARED_BANK_BYTES,  SHARED_BANK_BYTES};
  char* buffer;
  string line;
  int32_t bankID, preBankID;
  int32_t bankRdAddr, rdAddr;
  char rdValueStr[5000];

  ifstream ifile(bankFile, ios::in);
  if (!ifile.is_open()) {
    cout << "fail to open file " << bankFile << endl;
    return;
  }

  preBankID = 0;
  rdAddr = 0;
  buffer = new char[BytesInBank[0]];
  while (getline(ifile, line)) {
    sscanf(line.c_str(), "%d-%d : %s\n", &bankID, &bankRdAddr, rdValueStr);

    if (preBankID != bankID) {
      eachBankBinFile = getBankBinFile(preBankID);
      ofstream ofile(eachBankBinFile, ios::out | ios::binary | ios::trunc);
      if (!ofile.is_open()) {
        cout << "fail to open file " << eachBankBinFile << endl;
        return;
      }
      ofile.seekp(0, ios::beg);
      ofile.write(buffer, BytesInBank[preBankID]);
      ofile.close();

      delete[] buffer;
      buffer = new char[BytesInBank[bankID]];
      rdAddr = 0;
    }

    int strPos = strlen(rdValueStr) - 2;
    for (; strPos + 1 >= 0; strPos -= 2) {
      char data{};

      if (rdValueStr[strPos] <= '9' && rdValueStr[strPos] >= '0') {
        data = rdValueStr[strPos] - '0';
      } else if (rdValueStr[strPos] <= 'F' && rdValueStr[strPos] >= 'A') {
        data = rdValueStr[strPos] - 'A' + 10;
      } else if (rdValueStr[strPos] <= 'f' && rdValueStr[strPos] >= 'a') {
        data = rdValueStr[strPos] - 'a' + 10;
      }

      if (rdValueStr[strPos + 1] <= '9' && rdValueStr[strPos + 1] >= '0') {
        data = data * 16 + rdValueStr[strPos + 1] - '0';
      } else if (rdValueStr[strPos + 1] <= 'F' &&
                 rdValueStr[strPos + 1] >= 'A') {
        data = data * 16 + rdValueStr[strPos + 1] - 'A' + 10;
      } else if (rdValueStr[strPos + 1] <= 'f' &&
                 rdValueStr[strPos + 1] >= 'a') {
        data = data * 16 + rdValueStr[strPos + 1] - 'a' + 10;
      }

      buffer[rdAddr++] = data;

      if (rdAddr > BytesInBank[bankID]) {
        cout << "error: rdAddr in " << bankID
             << " exceeds range for bank_bin file" << endl;
        return;
      }
    }

    preBankID = bankID;

  }  // while

  eachBankBinFile = getBankBinFile(bankID);
  ofstream ofile(eachBankBinFile, ios::out | ios::binary | ios::trunc);
  if (!ofile.is_open()) {
    cout << "fail to open file " << eachBankBinFile << endl;
    return;
  }

  ofile.seekp(0, ios::beg);
  ofile.write(buffer, BytesInBank[bankID]);
  ofile.close();

  ifile.close();
  delete[] buffer;
}

void cache::writeBank(char* data, int32_t len, int32_t address,
                      int32_t bankID) {
  string eachBankBinFile;

  eachBankBinFile = getBankBinFile(bankID);

  fstream ofile(eachBankBinFile, ios::out | ios::binary | ios::in);
  if (!ofile.is_open()) {
    cout << "fail to open file " << eachBankBinFile << endl;
    return;
  }

  ofile.seekp(address, ios::beg);
  ofile.write(data, len);

  ofile.close();
}

void cache::readBank(char* data, int32_t len, int32_t address, int32_t bankID) {
  string eachBankBinFile;

  eachBankBinFile = getBankBinFile(bankID);
  ifstream ifile(eachBankBinFile, ios::in | ios::binary);
  if (!ifile.is_open()) {
    cout << "fail to open file " << eachBankBinFile << endl;
    return;
  }
  ifile.seekg(address, ios::beg);
  ifile.read(data, len);

  ifile.close();
}

void cache::writeDDR(char* data, int32_t len, int32_t address) {
  fstream ofile(DDRBinFile, ios::out | ios::binary | ios::in);
  if (!ofile.is_open()) {
    cout << "fail to open file " << DDRBinFile << endl;
    return;
  }

  ofile.seekp(address, ios::beg);
  ofile.write(data, len);

  ofile.close();
}

void cache::readDDR(char* data, int32_t len, int32_t address) {
  ifstream ifile(DDRBinFile, ios::in | ios::binary);
  if (!ifile.is_open()) {
    cout << "fail to open file " << DDRBinFile << endl;
    return;
  }

  ifile.seekg(address, ios::beg);
  ifile.read(data, len);
  ifile.close();
}

void cache::toLittleEndian(char* inBuffer, int32_t inLen,
                           signed short int* outBuffer, int32_t outLen,
                           int32_t dataSize) {
  uint16_t data;

  for (int i = 0; i < outLen; i++) {
    data = 0;
    for (int j = 0; j < dataSize; j++) {
      data = data + ((unsigned char)(inBuffer[dataSize * i + j]) << 8 * j);
    }
    outBuffer[i] = (signed short int)(data);
  }
}

void cache::toCharList(char* outBuffer, int32_t outLen,
                       signed short int* inBuffer, int32_t inLen,
                       int32_t dataSize) {
  char data;
  uint16_t temp;

  int outIndex = 0;
  for (int i = 0; i < inLen; i++) {
    temp = inBuffer[i];
    for (int j = 0; j < dataSize; j++) {
      data = temp % 256;
      outBuffer[outIndex++] = data;
      temp = temp >> 8;
    }
  }
}

}  // namespace load
}  // namespace xv2dpu
}  // namespace cosim
