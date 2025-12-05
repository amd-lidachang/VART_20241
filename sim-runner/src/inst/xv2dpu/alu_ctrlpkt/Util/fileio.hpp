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

#include <fstream>
#include "log.h"
#include "simCfg.hpp"

namespace cosim {
namespace xv2dpu {
namespace alu {

string fn_img_mt_prefix = "mt_aPr_b0_";
string fn_img_core_prefix = "co_aPr_b0_";
string fn_wgt_mt_prefix = "mt_bPr_b0_";
string fn_wgt_core_prefix = "co_bPr_b0_";
string fn_ofm_mt_prefix = "mt_cPr_b0_";
string fn_ofm_core_prefix = "co_cPr_b0_";
string fn_ctrl_mt_prefix = "mt_ctrl_b0_";
string fn_ctrl_core_prefix = "co_ctrl_b0_";

template <typename T>
int check_read_file(string fn, T* buffer, int n) {
  // param:
  // int n: total number
  ifstream fp;
  fp.open(fn, ios::in);
  if (!fp.is_open()) {
    cout << fn << " file open fail" << endl;
    return -1;
  } else {
    string str;
    int num = 0;
    int i = 0;

    while (getline(fp, str)) {
      istringstream istr(str);
      while (istr >> num) {
        *(buffer + i) = num;
        i++;
      }
    }
    if (i != n) cout << "file number does not match n" << endl;
    fp.close();
  }
  return 0;
}

template <typename T>
int check_write_file(const char* fn, T* buffer, int n, int line_cnt) {
  FILE* fp = fopen(fn, "w");

  assert(n % line_cnt == 0 && "n divide by line_cnt ");

  if (fp == NULL) {
    printf("Failuer opening file %s for writing!!\n", fn);
    return -1;
  } else {
    T* p = (T*)buffer;

    for (int i = 0; i < n; i++) {
      fprintf(fp, "%d ", *(p + i));
      if (i % line_cnt == (line_cnt - 1)) fprintf(fp, "\n");
    }
    fclose(fp);
    return 0;
  }
}

inline string getfname(int g, int l, string type, string data,
                       string func_name = "") {
  // g: group
  // l: parallel
  // func_name:
  // type: only 'mt' or 'co'
  string fn_prefix;
  string fn_premfix;
  if (type == "mt" && data == "img") {
    fn_prefix = fn_img_mt_prefix;
    fn_premfix = "_l";
  }
  if (type == "co" && data == "img") {
    fn_prefix = fn_img_core_prefix;
    fn_premfix = "_c";
  }
  if (type == "mt" && data == "wgt") {
    fn_prefix = fn_wgt_mt_prefix;
    fn_premfix = "_l";
  }
  if (type == "co" && data == "wgt") {
    fn_prefix = fn_wgt_core_prefix;
    fn_premfix = "_c";
  }
  if (type == "mt" && data == "ofm") {
    fn_prefix = fn_ofm_mt_prefix;
    fn_premfix = "_s";
  }
  if (type == "co" && data == "ofm") {
    fn_prefix = fn_ofm_core_prefix;
    fn_premfix = "_c";
  }
  string data_dir = SimCfgProxy::Params().ALU_CTRLPKT_PATH + "/";
  return data_dir + func_name + fn_prefix + "g" + to_string(g) + fn_premfix +
         to_string(l) + ".txt";
}

inline string getCtrlPktfname(int p, string type) {
  string data_dir = SimCfgProxy::Params().ALU_CTRLPKT_PATH + "/";
  string fn_prefix = (type == "co") ? fn_ctrl_core_prefix : fn_ctrl_mt_prefix;
  string fn =
      (type == "mti")
          ? (data_dir + "img" + fn_prefix + "n" + to_string(p) + ".txt")
          : (type == "mtw")
                ? (data_dir + "wgt" + fn_prefix + "n" + to_string(p) + ".txt")
                : (data_dir + fn_prefix + "c" + to_string(p) + ".txt");
  return fn;
}

template <typename T>
void appendImg2IntFile(const string& fname, const T* img, int len, int line) {
  // append file
  assert(line > 1);
  std::ofstream f;
  f.open(fname, std::ios::app);
  int row = len / line;
  row += (len % line == 0) ? 0 : 1;
  for (int i = 0; i < row; i++) {
    if (i == row - 1) f << "TLAST" << endl;
    for (int j = 0; j < line; j++) {
      int idx = i * line + j;
      if (idx < len) {
        if (sizeof(T) == 4)
          // f << (int32_t)img[idx] << " ";
          f << img[idx] << " ";
        else if (sizeof(T) == 1)
          f << +(int8_t)img[idx] << " ";
      }
    }
    f << endl;
  }
  f.close();
}

template <typename Dtype>
void write_to_file(vector<Dtype> in, int g, int l, string type, string data,
                   int len, string func_name = "") {
  uint32_t* uptr;
  int file_size = in.size();
  string fn_img_mt = getfname(g, l, "mt", "img");
  uptr = reinterpret_cast<uint32_t*>(in.data());
  appendImg2IntFile(fn_img_mt, uptr, file_size / sizeof(Dtype), len);
  appendImg2IntFile(fn_img_mt + ".int", in.data(), file_size, len * 4);
}

}  // namespace alu
}  // namespace xv2dpu
}  // namespace cosim
