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

#include <chrono>

#include "Calc.hpp"
#include "DumpOfstream.hpp"
#include "common.hpp"

#define __TIC__(tag)                                                           \
  auto __##tag##_start_time = std::chrono::steady_clock::now()

#define __TOC__(tag)                                                           \
  auto __##tag##_end_time = std::chrono::steady_clock::now();                  \
  UNI_LOG_INFO << #tag << " : "                                                \
               << std::chrono::duration_cast<std::chrono::milliseconds>(       \
                      __##tag##_end_time - __##tag##_start_time)               \
                      .count()                                                 \
               << "ms"

enum SaveFmt {
  DATA_FMT_MIN = 0,
  DATA_FMT_BIN = DATA_FMT_MIN,
  DATA_FMT_DEC,                         // 1
  DATA_FMT_HEX_CONT_SMALLEND,           // 2
  DATA_FMT_HEX_CONT_BIGEND,             // 3
  DATA_FMT_HEX_CONT_SMALLEND_BANKADDR,  // 4
  DATA_FMT_HEX_CONT_BIGEND_BANKADDR,    // 5
  DATA_FMT_HEX_CONT_SMALLEND_DDRADDR,   // 6
  DATA_FMT_HEX_CONT_BIGEND_DDRADDR,     // 7
  DATA_FMT_MAX,
};

enum SaveMode {
  SM_MIN = 0,
  SM_TRUNC = SM_MIN,
  SM_APPEND,
  SM_MAX,
};

namespace Util {

// log folder and file operation
template <typename T>
void ChkOpen(T& f, const string& fname);

void DelFileOrDir(const string& name);
void ChkFile(const string& name, bool del_exists = false);
void ChkFolder(const string& name, bool del_exists = false);
uint64_t GetFileSize(const string& fname);
uint64_t GetFileLine(const string& fname);

// load funcs: only SaveDec using templates
template <typename T>
uint64_t LoadData(int data_fmt, const string& fname, T* data, uint64_t size,
                  int line_size = 16);
template <typename T>
uint64_t LoadDec(const string& load_name, T* data, uint64_t size);

uint64_t LoadBin(const string& load_name, char* data, uint64_t size);
uint64_t LoadHexContSmallEnd(const string& load_name, char* data, uint64_t size,
                             int line_size = 16);
uint64_t LoadHexContBigEnd(const string& load_name, char* data, uint64_t size,
                           int line_size = 16);
uint64_t LoadHexContSmallEndBankAddr(const string& load_name, char* data,
                                     uint64_t size, int line_size = 16);
uint64_t LoadHexContBigEndBankAddr(const string& load_name, char* data,
                                   uint64_t size, int line_size = 16);
uint64_t LoadHexContSmallEndDDRAddr(const string& load_name, char* data,
                                    uint64_t size, int line_size = 16);
uint64_t LoadHexContBigEndDDRAddr(const string& load_name, char* data,
                                  uint64_t size, int line_size = 16);

// save funcs: only SaveDec using templates
template <typename T>
void SaveData(int data_fmt, const string& save_name, const T* data,
              uint64_t size, int line_size = 16, int mode = SM_TRUNC);
template <typename T>
void SaveDec(const string& save_name, const T* data, uint64_t size,
             int line_size = 16, int mode = SM_TRUNC);
template <typename T>
void SaveDec(std::ofstream& f, const T* data, uint64_t size, int line_size);
void SaveBin(const string& save_name, const char* data, uint64_t size,
             int mode = SM_TRUNC);
void SaveHexContSmallEnd(const string& save_name, const char* data,
                         uint64_t size, int line_size = 16,
                         int mode = SM_TRUNC);
void SaveHexContBigEnd(const string& save_name, const char* data, uint64_t size,
                       int line_size = 16, int mode = SM_TRUNC);
void SaveHexContSmallEndBankAddr(const string& save_name, const char* data,
                                 uint64_t size, int line_size = 16,
                                 int bank_height = 2048, int bank_id = 0,
                                 int bank_addr = 0, int mode = SM_TRUNC);
void SaveHexContBigEndBankAddr(const string& save_name, const char* data,
                               uint64_t size, int line_size = 16,
                               int bank_height = 2048, int bank_id = 0,
                               int bank_addr = 0, int mode = SM_TRUNC);
void SaveHexContSmallEndDDRAddr(const string& save_name, const char* data,
                                uint64_t size, int line_size = 16,
                                uint64_t addr_offset = 0, int reg_id = 0,
                                int mode = SM_TRUNC);
void SaveHexContBigEndDDRAddr(const string& save_name, const char* data,
                              uint64_t size, int line_size = 16,
                              uint64_t addr_offset = 0, int reg_id = 0,
                              int mode = SM_TRUNC);

// misc
template <typename T>
T Random(T low, T high);
template <typename T>
void Random(T* data, uint64_t size, T low, T high, uint64_t seed);

bool Str2Bool(const string& str);
string Bool2Str(bool flag);

int64_t Str2S64(const string& str);
uint64_t Str2U64(const string& str);

uint32_t FillMaskU32(int start, int num);
uint64_t FillMaskU64(int start, int num);

uint64_t AlignByN(uint64_t data, uint64_t align);

string GetFileNameSuffix(int fmt);

template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

// handle string
string Trim(const string& str);
vector<string> Split(const string& str, const string& delim = " ,:;\t",
                     bool trim_flag = true);
string SplitFirst(const string& str, const string& delim = " ,:;\t",
                  bool trim_flag = true);

// used to calculate time
enum TimeType {
  TIME_S,
  TIME_MS,
  TIME_US,
};
void TimeBegin(struct timeval& begin);
unsigned long TimeEnd(const struct timeval& begin, int type = TIME_MS);

};  // namespace Util

template <typename T>
void Util::ChkOpen(T& f, const string& fname) {
  struct stat buf;

  if (lstat(fname.c_str(), &buf) == -1) {
    UNI_LOG_FATAL(SIM_FILE_OPEN_FAILED)
        << fname << " stat error: " << strerror(errno) << endl;
  }

  // handle common file
  if (!S_ISREG(buf.st_mode)) {
    UNI_LOG_FATAL(SIM_FILE_OPEN_FAILED)
        << "Not a normal file: " << fname << endl;
  }

  if (f.fail()) {
    UNI_LOG_FATAL(SIM_FILE_OPEN_FAILED)
        << "open file fail(): " << fname << endl;
  }

  if (f.bad()) {
    UNI_LOG_FATAL(SIM_FILE_OPEN_FAILED) << "open file bad(): " << fname << endl;
  }

  if (f.eof()) {
    UNI_LOG_FATAL(SIM_FILE_OPEN_FAILED) << "open file eof(): " << fname << endl;
  }
}

template <>
inline void Util::ChkOpen<dump::DumpOfstream_>(dump::DumpOfstream_& df,
                                               const string& fname) {
  auto df_context = df.getContext();
  UNI_LOG_CHECK(nullptr != df_context, SIM_PARAMETER_FAILED)
      << "context is nullptr: " << fname << endl;
  if (df_context->isFine) return;
  std::ofstream f;
  f.open(df_context->dumpFilePath_, df_context->openMode_);
  Util::ChkOpen(f, fname);
  f.close();
  df_context->isFine = true;
}

template <typename T>
T Util::Random(T low, T high) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<T> dis(low, high);

  return dis(gen);
}

template <typename T>
void Util::Random(T* data, uint64_t size, T low, T high, uint64_t seed) {
  std::random_device rd;
  std::mt19937 gen(seed);
  std::uniform_int_distribution<T> dis(low, high);

  for (auto i = 0U; i < size; i++) {
    if (seed == 0)
      data[i] = 0;
    else
      data[i] = dis(gen);
  }
}

template <typename T>
uint64_t Util::LoadData(int data_fmt, const string& fname, T* data,
                        uint64_t size, int line_size) {
  // check param validation
  UNI_LOG_CHECK(data != nullptr && size > 0, SIM_PARAMETER_FAILED);

  uint64_t n = size;

  switch (data_fmt) {
    case DATA_FMT_BIN:
      n = LoadBin(fname, reinterpret_cast<char*>(data), size * sizeof(T));
      break;
    case DATA_FMT_DEC:
      n = LoadDec(fname, data, size);
      break;
    case DATA_FMT_HEX_CONT_SMALLEND:
      n = LoadHexContSmallEnd(fname, reinterpret_cast<char*>(data),
                              size * sizeof(T), line_size * sizeof(T));
      break;
    case DATA_FMT_HEX_CONT_BIGEND:
      n = LoadHexContBigEnd(fname, reinterpret_cast<char*>(data),
                            size * sizeof(T), line_size * sizeof(T));
      break;
    case DATA_FMT_HEX_CONT_SMALLEND_BANKADDR:
      n = LoadHexContSmallEndBankAddr(fname, reinterpret_cast<char*>(data),
                                      size * sizeof(T), line_size * sizeof(T));
      break;
    case DATA_FMT_HEX_CONT_BIGEND_BANKADDR:
      n = LoadHexContBigEndBankAddr(fname, reinterpret_cast<char*>(data),
                                    size * sizeof(T), line_size * sizeof(T));
      break;
    case DATA_FMT_HEX_CONT_SMALLEND_DDRADDR:
      n = LoadHexContSmallEndDDRAddr(fname, reinterpret_cast<char*>(data),
                                     size * sizeof(T), line_size * sizeof(T));
      break;
    case DATA_FMT_HEX_CONT_BIGEND_DDRADDR:
      n = LoadHexContBigEndDDRAddr(fname, reinterpret_cast<char*>(data),
                                   size * sizeof(T), line_size * sizeof(T));
      break;
    default:
      UNI_LOG_FATAL(SIM_OUT_OF_RANGE)
          << "Not support init method: " << data_fmt << endl;
  }

  return n;
}

template <typename T>
uint64_t Util::LoadDec(const string& load_name, T* data, uint64_t size) {
  string word;
  std::fstream f(load_name);
  vector<string> ori_data;

  ChkOpen(f, load_name);

  uint64_t num = 0;
  while (f >> word) {
    if (num > size) {
      UNI_LOG_WARNING << "file size exceeds memory size: " << load_name;
      break;
    }

    try {
      data[num++] = stoll(word, nullptr, 10);
    } catch (std::invalid_argument& e) {
      UNI_LOG_FATAL(SIM_PARAMETER_FAILED) << e.what() << " invalid argument!";
    }
  }

  return num;
}

template <typename T>
void Util::SaveData(int data_fmt, const string& save_name, const T* data,
                    uint64_t size, int line_size, int mode) {
  // check param validation
  UNI_LOG_CHECK(data != nullptr && size > 0, SIM_PARAMETER_FAILED);

  switch (data_fmt) {
    case DATA_FMT_BIN:
      SaveBin(save_name, reinterpret_cast<const char*>(data), size * sizeof(T),
              mode);
      break;
    case DATA_FMT_DEC:
      SaveDec(save_name, data, size, line_size, mode);
      break;
    case DATA_FMT_HEX_CONT_SMALLEND:
      SaveHexContSmallEnd(save_name, reinterpret_cast<const char*>(data),
                          size * sizeof(T), line_size * sizeof(T), mode);
      break;
    case DATA_FMT_HEX_CONT_BIGEND:
      SaveHexContBigEnd(save_name, reinterpret_cast<const char*>(data),
                        size * sizeof(T), line_size * sizeof(T), mode);
      break;
    default:
      UNI_LOG_FATAL(SIM_OUT_OF_RANGE)
          << "Not support save data fmt: " << data_fmt << endl;
  }
}

template <typename T>
void Util::SaveDec(const string& save_name, const T* data, uint64_t size,
                   int line_size, int mode) {
  // write into file
  DumpOfstream f;

  if (mode == SM_TRUNC) {
    f.open(save_name, std::ios::trunc);
  } else {
    f.open(save_name, std::ios::app);
  }

  ChkOpen(f, save_name);

  // line related vars
  int line_num = size / line_size + 1;
  int line_left = size % line_size;

  // write line_num lines
  for (int i = 0; i < line_num; i++) {
    // handle last line, it's special
    if (i == line_num - 1 && line_left == 0) break;

    // write line_size data every line
    for (int j = 0; j < line_size; j++) {
      // handle last line, it's special
      if (i == line_num - 1 && j >= line_left) break;

      // every 4 data elements, insert 4 spaces
      if (j != 0 && j % 4 == 0)
        f << " "
          << " "
          << " "
          << " ";

      // save data using related format
      if (line_size == 1) {
        f << dec << +data[i * line_size + j];
      } else {
        if (sizeof(T) == 1) {
          f << dec << setw(6) << setfill(' ') << +data[i * line_size + j];
        } else if (sizeof(T) == 2) {
          f << dec << setw(9) << setfill(' ') << +data[i * line_size + j];
        } else if (sizeof(T) == 4) {
          f << dec << setw(12) << setfill(' ') << +data[i * line_size + j];
        } else if (sizeof(T) == 8) {
          f << dec << setw(24) << setfill(' ') << +data[i * line_size + j];
        } else {
          UNI_LOG_FATAL(SIM_OUT_OF_RANGE)
              << "Not support type, sizeof(T) = " << sizeof(T) << endl;
        }
      }
    }
    f << endl;
  }

  f.close();
}

template <typename T>
void Util::SaveDec(std::ofstream& f, const T* data, uint64_t size,
                   int line_size) {
  int line_num = div_ceil(size, line_size);
  int last_line_size = size % line_size == 0 ? line_size : size % line_size;

  f << dec << setfill(' ');
  int w;
  if (sizeof(T) == 1)
    w = 6;
  else if (sizeof(T) == 2)
    w = 9;
  else if (sizeof(T) == 4)
    w = 12;
  else if (sizeof(T) == 8)
    w = 24;
  else
    UNI_LOG_FATAL(SIM_OUT_OF_RANGE)
        << "Not support type, sizeof(T) = " << sizeof(T) << endl;

  auto ptr = data;
  for (int i = 0; i < line_num; i++) {
    auto elem_num = i == line_num - 1 ? last_line_size : line_size;
    for (int j = 0; j < elem_num; j++) {
      // for each 4 elements, insert 4 spaces
      if (j != 0 && j % 4 == 0) f << "    ";

      f << setw(w) << +*ptr++;
    }
    f << std::endl;
  }
}

#endif /* __UTIL_HPP__*/
