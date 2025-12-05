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
#ifndef __CONVADDRL_HPP__
#define __CONVADDRL_HPP__

#include "pub/InstBase.hpp"

enum class conv_addr_type_t : uint32_t {
  CONV_ADDR_IFM = 0,
  CONV_ADDR_WGT = 1,
  CONV_ADDR_BIAS = 2,
  CONV_ADDR_OFM = 3,
  CONV_ADDR_IFM_ELEW = 4,
};

typedef struct conv_addr_s {
  int32_t type;
  int32_t h_num;
  int32_t mt_addr;
  int32_t invalid;
  int32_t jump;
  int32_t jump_endl;
} conv_addr_t;

template <DPUVersion T>
class ConvAddr : public InstBase {
 public:
  explicit ConvAddr(int inst_type, int instid, vector<string>& inst_str,
                    vector<uint32_t>& inst_val);
  ~ConvAddr();

  static ConvAddr* CUROBJ;

 public:
  // excute load instruction
  void Exec() override final{};
  void GenInstStr(int inst_fmt) override final;

 public:
  static int32_t get_num();
  static int32_t get_direction(int32_t idx);
  static int32_t get_h_num(int32_t idx);
  static int32_t get_mt_addr(int32_t idx);
  static conv_addr_type_t get_type(int32_t idx);
  static int32_t get_invalid(int32_t idx);
  static void clear();
  conv_addr_t &get_convaddr_info(int32_t idx);

 private:
  static int32_t num_;
  static std::vector<conv_addr_t> convaddr_info;

  int32_t local_direction_;
  int32_t local_h_num_;
  int32_t local_mt_addr_;
  int32_t local_invalid_;
  int32_t local_type_;
  int32_t local_last_;
  int32_t local_compression_;
  int32_t local_jump_;
  int32_t local_jump_endl_;
};

template <DPUVersion T>
ConvAddr<T>* ConvAddr<T>::CUROBJ = nullptr;
template <DPUVersion T>
ConvAddr<T>::~ConvAddr() {}

#endif /* __CONVADDRL_HPP__ */
