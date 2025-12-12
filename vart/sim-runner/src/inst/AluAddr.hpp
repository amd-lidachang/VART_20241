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
#ifndef __ALUADDRL_HPP__
#define __ALUADDRL_HPP__

#include "pub/InstBase.hpp"

enum class alu_addr_type_t : int {
  ALU_ADDR_IFM = 0,
  ALU_ADDR_WGT = 1,
  ALU_ADDR_BIAS = 2,
  ALU_ADDR_OFM = 3,
};

typedef struct alu_addr_s {
  int32_t type;
  int32_t last;
  int32_t compression;
  int32_t invalid;
  int32_t h_num;
  int32_t macc_dim;
  int32_t mt_addr;
  int32_t id;
  int32_t jump;
  int32_t jump_endl;
} alu_addr_t;

template <DPUVersion T>
class AluAddr : public InstBase {
 public:
  explicit AluAddr(int inst_type, int instid, vector<string>& inst_str,
                   vector<uint32_t>& inst_val);
  ~AluAddr();

  static AluAddr* CUROBJ;

 public:
  // excute load instruction
  void Exec() override final{};
  void GenInstStr(int inst_fmt) override final;

 public:
  static int32_t get_num();
  static int32_t get_direction(int32_t idx);
  static int32_t get_h_num(int32_t idx);
  static int32_t get_mt_addr(int32_t idx);
  static int32_t get_id(int32_t idx);
  static int32_t get_jump(int32_t idx);
  static int32_t get_jump_endl(int32_t idx);
  static int32_t get_invalid(int32_t idx);
  static int32_t get_last(int32_t idx);
  static int32_t get_compression(int32_t idx);
  static alu_addr_type_t get_type(int32_t idx);
  static alu_addr_t &get_alu_addr_info(int32_t idx);
  static void clear();
  static bool check();

 private:
  static int32_t num_;

  int32_t local_type_;
  int32_t local_h_num_;
  int32_t local_mt_addr_;
  int32_t local_macc_dim_;
  int32_t local_id_;
  int32_t local_jump_;
  int32_t local_jump_endl_;
  int32_t local_invalid_;
  int32_t local_last_;
  int32_t local_compression_;
  static vector<alu_addr_t> aluaddr_info;
};

template <DPUVersion T>
AluAddr<T>* AluAddr<T>::CUROBJ = nullptr;
template <DPUVersion T>
AluAddr<T>::~AluAddr() {}

#endif /* __ALUADDRL_HPP__ */
