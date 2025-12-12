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

namespace cosim {
namespace xv2dpu {
namespace alu {

class BufDes {
  // buffer descriptor
 private:
  int length;
  int offset;
  vector<int> stepsize;
  vector<int> wrap;
  vector<vector<int>> padding;

 public:
  BufDes(){};
  BufDes(int length, int offset, vector<int> stepsize, vector<int> wrap,
         vector<vector<int>> padding);
  int get_length() { return length; };
  int get_offset() { return offset; };
  vector<int> get_stepsize() { return stepsize; };
  vector<int> get_wrap() { return wrap; };
  vector<vector<int>> get_padding() { return padding; };
  void set_length(int len) { length = len; };
  void set_offset(int off) { offset = off; };

  void print_stepsize();
  void print_wrap();
  void print_padding();

  template <typename T>
  int write_access(const T* src_in, const T* dst_in);

  template <typename T>
  int read_access(const T* src_in, const T* dst_in);
};

BufDes::BufDes(int length, int offset, vector<int> stepsize, vector<int> wrap,
               vector<vector<int>> padding = {})
    : length(length),
      offset(offset),
      stepsize(stepsize),
      wrap(wrap),
      padding(padding) {
  assert(length > 0 && "length should > 0");
  assert(offset >= 0 && "offset should >= 0");
  assert((stepsize.size() > 0) && (stepsize.size() <= 4) &&
         "stepsize length should > 0");
  assert((stepsize.size() == (wrap.size() + 1) ||
          stepsize.size() == wrap.size()) &&
         "stepsize.size == wrap.size + 1");
  assert(padding.size() <= wrap.size() && "stepsize.size == wrap.size + 1");
  for (auto vec = padding.begin(); vec != padding.end(); vec++) {
    assert((*vec).size() == 2 && "padding params vector size == 2");
  }
  while (this->wrap.size() < 4) {
    this->wrap.push_back(1);
  }
  while (this->stepsize.size() < 4) {
    this->stepsize.push_back(0);
  }
  while (this->padding.size() < 3) {
    this->padding.push_back({0, 0});
  }
}

template <typename T>
int BufDes::write_access(const T* src_in, const T* dst_in) {
  int* src = (int*)src_in;
  int* dst = (int*)dst_in;
  // write, do not support padding
  int res = 0;
  int rof = 0;  //  read_offset
  int start_addr = offset;
  int wof = offset;  //# write_offset
  int size = length;
  if (src == nullptr || dst == nullptr) {
    res = -1;
    return res;
  }

  while (rof < size) {
    for (int i = 0; i < wrap[3]; i++) {
      for (int j = 0; j < wrap[2]; j++) {
        for (int k = 0; k < wrap[1]; k++) {
          for (int l = 0; l < wrap[0]; l++) {
            wof = start_addr + stepsize[0] * l + stepsize[1] * k +
                  stepsize[2] * j + stepsize[3] * i;
            dst[wof] = src[rof];
            rof++;
          }
          if (stepsize[1] == 0) {
            start_addr += stepsize[0];
            break;
          }
        }
        if (stepsize[2] == 0) {
          start_addr += stepsize[1];
          break;
        }
      }
      if (stepsize[3] == 0) {
        start_addr += stepsize[2];
        break;
      } else {
        start_addr += stepsize[3];
      }
    }
  }

  if (rof > size) {
    cout << "wrap and step params not correct" << endl;
    cout << "rof is :" << rof << endl;
    cout << "size is :" << size << endl;
    res = 1;
  } else if (rof == size) {
    // cout << "buffer_des success" << endl;
    res = 0;
  } else {
    cout << "ERROR!!!!!!" << endl;
    res = 2;
  }

  return res;
}

template <typename T>
int BufDes::read_access(const T* src_in, const T* dst_in) {
  int* src = (int*)src_in;
  int* dst = (int*)dst_in;
  // read, support padding
  int res = 0;
  int rof = 0;  //  read_offset
  int start_addr = offset;
  int wof = 0;  //# write_offset
  int size = length;
  if (src == nullptr || dst == nullptr) {
    res = -1;
    return res;
  }

  vector<int> v0 = padding[0];
  vector<int> v1 = padding[1];
  vector<int> v2 = padding[2];
  int hp2 = v2[0];
  int ep2 = v2[1];
  int hp1 = v1[0];
  int ep1 = v1[1];
  int hp0 = v0[0];
  int ep0 = v0[1];

  while (wof < size) {
    for (int i = 0; i < wrap[3]; i++) {
      //#padding head
      for (int h = 0; h < (hp2 * (wrap[1] + hp1 + ep1) * (wrap[0] + hp0 + ep0));
           h++) {
        dst[wof] = 0;
        wof += 1;
      }

      for (int j = 0; j < wrap[2]; j++) {
        //#padding head
        for (int h = 0; h < (hp1 * (wrap[0] + hp0 + ep0)); h++) {
          dst[wof] = 0;
          wof += 1;
        }

        for (int k = 0; k < wrap[1]; k++) {
          //#padding head
          for (int h = 0; h < hp0; h++) {
            dst[wof] = 0;
            wof += 1;
          }

          for (int l = 0; l < wrap[0]; l++) {
            rof = stepsize[0] * l + stepsize[1] * k + stepsize[2] * j +
                  stepsize[3] * i + start_addr;
            dst[wof] = src[rof];
            wof += 1;
          }

          for (int e = 0; e < (ep0); e++) {
            dst[wof] = 0;
            wof += 1;
          }
          if (stepsize[1] == 0) {
            start_addr += stepsize[0];
            break;
          }
        }
        for (int e = 0; e < (ep1 * (wrap[0] + hp0 + ep0)); e++) {
          dst[wof] = 0;
          wof += 1;
        }
        if (stepsize[2] == 0) {
          start_addr += stepsize[1];
          break;
        }
      }
      // padding end
      for (int e = 0; e < (ep2 * (wrap[1] + hp1 + ep1) * (wrap[0] + hp0 + ep0));
           e++) {
        dst[wof] = 0;
        wof += 1;
      }
      if (stepsize[3] == 0) {
        start_addr += stepsize[2];
        break;
      } else {
        start_addr += stepsize[3];
      }
    }
  }
  if (wof > size) {
    res = 1;
    // print('wrap and step params not correct')
  } else {
    // print('buffer_des success')
  }

  return res;
}

void BufDes::print_stepsize() {
  cout << "stepsize: {";
  for (int i = 0; i < int(stepsize.size()); i++) {
    int v = stepsize[i];
    cout << v << ", ";
  }
  cout << "}" << endl;
}

void BufDes::print_wrap() {
  cout << "wrap: {";
  for (int i = 0; i < int(wrap.size()); i++) {
    int v = wrap[i];
    cout << v << ", ";
  }
  cout << "}" << endl;
}

void BufDes::print_padding() {
  cout << "padding: {";
  for (auto v = padding.begin(); v != padding.end(); v++) {
    cout << "{";
    vector<int> vec(*v);
    for (int j = 0; j < int(vec.size()); j++) {
      int k = vec[j];
      cout << k << ", ";
    }
    cout << "},";
  }
  cout << "}" << endl;
}

}  // namespace alu
}  // namespace xv2dpu
}  // namespace cosim
