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

#include "params.hpp"

namespace cosim {
namespace xv2dpu {
namespace alu {

enum PKT_TYPE {
  PKT_TYPE_CTRL = 0,
  PKT_TYPE_DATA = 1,
};

string uint2str(uint32_t* pkts, int len, PKT_TYPE type, int WIDTH);
string uint2hex(uint32_t* tmp, int len, int WIDTH = 4);

inline uint32_t addOddParityBits(uint32_t in) {
  uint32_t odd = 1;
  for (int i = 0; i < 31; i++) {
    odd ^= (in >> i);
  }
  return (in | (odd << 31));
}

//////////////////Packets Header//////////////////
inline uint32_t genStreamPktHeader(int streamid, int pkt_type, int src_row,
                                   int src_col) {
  assert((((streamid >> 5) & 0xffffffff) == 0) &&
         "streamid 5 bits valid");  // 5 bit
  assert((((pkt_type >> 3) & 0xffffffff) == 0) &&
         "pkt_type 3 bits valid");  // 3 bit
  assert((((src_row >> 5) & 0xffffffff) == 0) &&
         "src_row  5 bits valid");  // 5 bit
  assert((((src_col >> 7) & 0xffffffff) == 0) &&
         "src_col  7 bits valid");  // 7 bit
  uint32_t header = 0x00000000;
  header |= streamid;          // bit:4-0
  header |= (pkt_type << 12);  // bit:14-12
  header |= (src_row << 16);   // bit:20-16
  header |= (src_col << 21);   // bit:27:21
  return addOddParityBits(header);
}

//////////////////Ctrl Informance//////////////////
// data_beat:
//  00->1, 01->2, 10->3, 11->4
// option:
//  00: write without return packet
//  01: read with return packet
//  10: write with return packet
//  11: reserved
///////////////////////////////////////////////////
inline uint32_t genCtrlInfo(int address, int data_beat, int option,
                            int id4return) {
  assert((((address >> 20) & 0xffffffff) == 0) && "address   20 bits valid");
  assert((((data_beat >> 3) & 0xffffffff) == 0) && "data_beat  2 bits valid");
  assert((((option >> 2) & 0xffffffff) == 0) && "option     2 bits valid");
  assert((((id4return >> 5) & 0xffffffff) == 0) && "id4return  5 bits valid");
  uint32_t header = 0x00000000;
  header |= address;                  // bit:19-0
  header |= ((data_beat - 1) << 20);  // bit:21-20
  header |= (option << 22);           // bit:23-22
  header |= (id4return << 24);        // bit:28-24
  return addOddParityBits(header);
}

//////////////////core BD config generate//////////////////
inline uint32_t genCOBD0(uint32_t addr, uint32_t length) {
  assert((((addr >> 14) & 0xffffffff) == 0) && "addr   only 14 bits valid");
  assert((((length >> 14) & 0xffffffff) == 0) && "length only 14 bits valid");
  uint32_t res = 0x00000000;
  res |= (addr << 14);  // addr: 27:14
  res |= length;        // len : 13:0
  return res;
}

inline uint32_t genCOBD2(int D1_step, int D0_step) {
  assert((((D1_step >> 13) & 0xffffffff) == 0) && "D1_step only 13 bits valid");
  assert((((D0_step >> 13) & 0xffffffff) == 0) && "D0_step only 13 bits valid");
  uint32_t res = 0x00000000;
  res |= (D1_step << 13);  // D1_step: 25:13
  res |= D0_step;          // D0_step: 12:0
  return res;
}

inline uint32_t genCOBD3(int D1_wrap, int D0_wrap, int D2_stepsize) {
  assert((((D1_wrap >> 8) & 0xffffffff) == 0) &&
         "D1_wrap     only  8 bits valid");
  assert((((D0_wrap >> 8) & 0xffffffff) == 0) &&
         "D0_wrap     only  8 bits valid");
  assert((((D2_stepsize >> 13) & 0xffffffff) == 0) &&
         "D2_stepsize only 13 bits valid");
  uint32_t res = 0x00000000;
  res |= (D1_wrap << 21);  // D1_wrap    : 28:21
  res |= (D0_wrap << 13);  // D0_wrap    : 20:13
  res |= D2_stepsize;      // D2_stepsize: 12:0
  return res;
}

inline uint32_t genCOBD4(int Iter_Current, int Iter_Wrap, int Iter_Step) {
  assert((((Iter_Current >> 6) & 0xffffffff) == 0) &&
         "Iter_Current only  6 bits valid");
  assert((((Iter_Wrap >> 6) & 0xffffffff) == 0) &&
         "Iter_Wrap    only  6 bits valid");
  assert((((Iter_Step >> 13) & 0xffffffff) == 0) &&
         "Iter_Step    only 13 bits valid");
  uint32_t res = 0x00000000;
  res |= (Iter_Current << 19);  // Iter_Current : 24:19
  res |= (Iter_Wrap << 13);     // Iter_Wrap    : 18:13
  res |= Iter_Step;             // Iter_Step    : 12:0
  return res;
}

inline uint32_t genCOBD5(int Lock_Rel_Value, int Lock_Rel_ID,
                         int Lock_Acq_Value, int Lock_Acq_ID,
                         int TLAST_Suppress, int Next_BD, int Use_Next_BD,
                         int Lock_Acq_Enable) {
  int Valid_BD = 1;
  assert((((Valid_BD >> 1) & 0xffffffff) == 0) &&
         "Valid_BD        only  1 bits valid");
  assert((((TLAST_Suppress >> 1) & 0xffffffff) == 0) &&
         "TLAST_Suppress  only  1 bits valid");
  assert((((Next_BD >> 4) & 0xffffffff) == 0) &&
         "Next_BD         only  4 bits valid");
  assert((((Use_Next_BD >> 1) & 0xffffffff) == 0) &&
         "Use_Next_BD     only  1 bits valid");
  assert((((Lock_Acq_Enable >> 1) & 0xffffffff) == 0) &&
         "Lock_Acq_Enable only  1 bits valid");
  assert(((((Lock_Rel_Value >> 7) & 0xffffffff) == 0) ||
          ((Lock_Rel_Value >> 7) == (int)0xffffffff)) &&
         "Lock_Rel_Value  only  7 bits valid");
  assert((((Lock_Rel_ID >> 4) & 0xffffffff) == 0) &&
         "Lock_Rel_ID     only  4 bits valid");
  assert(((((Lock_Acq_Value >> 7) & 0xffffffff) == 0) ||
          ((Lock_Acq_Value >> 7) == (int)0xffffffff)) &&
         "Lock_Acq_Value  only  7 bits valid");
  assert((((Lock_Acq_ID >> 4) & 0xffffffff) == 0) &&
         "Lock_Acq_ID     only  4 bits valid");
  uint32_t res = 0x00000000;
  res |= (TLAST_Suppress << 31);                 // TLAST_Suppress : 31
  res |= (Next_BD << 27);                        // Next_BD        : 30:27
  res |= (Use_Next_BD << 26);                    // Use_Next_BD    : 26
  res |= (Valid_BD << 25);                       // Valid_BD       : 25
  res |= ((Lock_Rel_Value & 0x0000007f) << 18);  // Lock_Rel_Value : 24:18
  res |= (Lock_Rel_ID << 13);                    // Lock_Rel_ID    : 16:13
  res |= (Lock_Acq_Enable << 12);                // Lock_Acq_Enable: 12
  res |= ((Lock_Acq_Value & 0x0000007f) << 5);   // Lock_Acq_Value : 11:5
  res |= Lock_Acq_ID;                            // Lock_Acq_ID    : 3:0
  return res;
}

//////////////////memt BD config generate//////////////////
inline uint32_t genMTBD0(uint32_t length) {
  assert((((length >> 17) & 0xffffffff) == 0) && "length only 17 bits valid");
  uint32_t res = 0x00000000;
  res |= length & 0x0001ffff;  // length : 16:0
  return res;
}

inline uint32_t genMTBD1(int D0_Zero_Before, int address, int next_bd,
                         bool use_next) {
  assert((((D0_Zero_Before >> 6) & 0xffffffff) == 0) &&
         "D0_Zero_Before only  6 bits valid");
  assert((((address >> 19) & 0xffffffff) == 0) &&
         "address        only 19 bits valid");
  assert((((next_bd >> 6) & 0xffffffff) == 0) &&
         "next_bd        only 6  bits valid");
  assert((((use_next >> 1) & 0xffffffff) == 0) &&
         "use_next_bd    only 1  bits valid");
  uint32_t res = 0x00000000;
  res |= (D0_Zero_Before << 26);  // D0_Zero_Before : 31:26
  res |= address;                 // address        : 18:0
  res |= next_bd << 20;           // next bd        : 25:20
  res |= use_next << 19;          // use next bd   : 19
  return res;
}

inline uint32_t genMTBD2(int TLAST_Suppress, int D0_wrap, int D0_stepsize) {
  assert((((TLAST_Suppress >> 1) & 0xffffffff) == 0) &&
         "TLAST_Suppress only  1 bits valid");
  assert((((D0_wrap >> 10) & 0xffffffff) == 0) &&
         "D0_wrap        only 10 bits valid");
  assert((((D0_stepsize >> 17) & 0xffffffff) == 0) &&
         "D0_stepsize    only 17 bits valid");
  uint32_t res = 0x00000000;
  res |= (TLAST_Suppress << 31);          // TLAST_Suppress : 31
  res |= ((D0_wrap & 0x000003ff) << 17);  // D0_wrap        : 26:17
  res |= (D0_stepsize & 0x0001ffff);      // D0_stepsize    : 16:0
  return res;
}

inline uint32_t genMTBD3(int D1_Zero_Before, int D1_wrap, int D1_stepsize) {
  assert((((D1_Zero_Before >> 5) & 0xffffffff) == 0) &&
         "D1_Zero_Before only  5 bits valid");
  assert((((D1_wrap >> 10) & 0xffffffff) == 0) &&
         "D1_wrap        only 10 bits valid");
  assert((((D1_stepsize >> 17) & 0xffffffff) == 0) &&
         "D1_stepsize    only 17 bits valid");
  uint32_t res = 0x00000000;
  res |= (D1_Zero_Before << 27);          // D1_Zero_Before : 31:27
  res |= ((D1_wrap & 0x000003ff) << 17);  // D1_wrap        : 26:17
  res |= (D1_stepsize & 0x0001ffff);      // D1_stepsize    : 16:0
  return res;
}

inline uint32_t genMTBD4(int Enable_Compression, int D2_Zero_Before,
                         int D2_wrap, int D2_stepsize) {
  assert((((Enable_Compression >> 1) & 0xffffffff) == 0) &&
         "Enable_Compression only  1 bits valid");
  assert((((D2_Zero_Before >> 4) & 0xffffffff) == 0) &&
         "D2_Zero_Before     only  4 bits valid");
  assert((((D2_wrap >> 10) & 0xffffffff) == 0) &&
         "D2_wrap            only 10 bits valid");
  assert((((D2_stepsize >> 17) & 0xffffffff) == 0) &&
         "D2_stepsize        only 17 bits valid");
  uint32_t res = 0x00000000;
  res |= (Enable_Compression << 31);      // Enable_Compression : 31
  res |= (D2_Zero_Before << 27);          // D2_Zero_Before     : 30:27
  res |= ((D2_wrap & 0x000003ff) << 17);  // D2_wrap            : 26:17
  res |= (D2_stepsize & 0x0001ffff);      // D2_stepsize        : 16:0
  return res;
}

inline uint32_t genMTBD5(int D2_Zero_After, int D1_Zero_After,
                         int D0_Zero_After, int D3_stepsize) {
  assert((((D2_Zero_After >> 4) & 0xffffffff) == 0) &&
         "D2_Zero_After only  4 bits valid");
  assert((((D1_Zero_After >> 5) & 0xffffffff) == 0) &&
         "D1_Zero_After only  5 bits valid");
  assert((((D0_Zero_After >> 6) & 0xffffffff) == 0) &&
         "D0_Zero_After only  6 bits valid");
  assert((((D3_stepsize >> 17) & 0xffffffff) == 0) &&
         "D3_stepsize   only 17 bits valid");
  uint32_t res = 0x00000000;
  res |= (D2_Zero_After << 28);       // D2_Zero_After : 31:28
  res |= (D1_Zero_After << 23);       // D1_Zero_After : 27:23
  res |= (D0_Zero_After << 17);       // D0_Zero_After : 22:17
  res |= (D3_stepsize & 0x0001ffff);  // D3_stepsize   : 16:0
  return res;
}

inline uint32_t genMTBD6(int Iter_Current, int Iter_Wrap, int Iter_Step) {
  assert((((Iter_Current >> 6) & 0xffffffff) == 0) &&
         "Iter_Current only  6 bits valid");
  assert((((Iter_Wrap >> 6) & 0xffffffff) == 0) &&
         "Iter_Wrap    only  6 bits valid");
  assert((((Iter_Step >> 17) & 0xffffffff) == 0) &&
         "Iter_Step    only 17 bits valid");
  uint32_t res = 0x00000000;
  res |= (Iter_Current << 23);  // Iter_Current : 28:23
  res |= (Iter_Wrap << 17);     // Iter_Wrap    : 22:17
  res |= Iter_Step;             // Iter_Step    : 16:0
  return res;
}

inline uint32_t genMTBD7(int Valid_BD, int Lock_Rel_Value, int Lock_Rel_ID,
                         int Lock_Acq_Enable, int Lock_Acq_Value,
                         int Lock_Acq_ID) {
  assert((((Valid_BD >> 1) & 0xffffffff) == 0) &&
         "Valid_BD        only  1 bits valid");
  assert(((((Lock_Rel_Value >> 7) & 0xffffffff) == 0) ||
          ((Lock_Rel_Value >> 7) == (int)0xffffffff)) &&
         "Lock_Rel_Value  only  7 bits valid");
  assert((((Lock_Rel_ID >> 8) & 0xffffffff) == 0) &&
         "Lock_Rel_ID     only  8 bits valid");
  assert((((Lock_Acq_Enable >> 1) & 0xffffffff) == 0) &&
         "Lock_Acq_Enable only  1 bits valid");
  assert(((((Lock_Acq_Value >> 7) & 0xffffffff) == 0) ||
          ((Lock_Acq_Value >> 7) == (int)0xffffffff)) &&
         "Lock_Acq_Value  only  7 bits valid");
  assert((((Lock_Acq_ID >> 8) & 0xffffffff) == 0) &&
         "Lock_Acq_ID     only  8 bits valid");
  uint32_t res = 0x00000000;
  res |= (Valid_BD << 31);                       // Valid_BD       : 31
  res |= ((Lock_Rel_Value & 0x0000007f) << 24);  // Lock_Rel_Value : 30:24
  res |= (Lock_Rel_ID << 16);                    // Lock_Rel_ID    : 23:16
  res |= (Lock_Acq_Enable << 15);                // Lock_Acq_Enable: 15
  res |= ((Lock_Acq_Value & 0x0000007f) << 8);   // Lock_Acq_Value : 14:8
  res |= Lock_Acq_ID;                            // Lock_Acq_ID    : 7:0
  return res;
}

inline uint32_t genStartQ(int Enable_Token_Issue, int Repeat_Count,
                          int Start_BD_ID) {
  assert((((Enable_Token_Issue >> 1) & 0xffffffff) == 0) &&
         "Enable_Token_Issue only  1 bits valid");
  assert((((Repeat_Count >> 8) & 0xffffffff) == 0) &&
         "Repeat_Count       only  8 bits valid");
  assert((((Start_BD_ID >> 6) & 0xffffffff) == 0) &&
         "Start_BD_ID        only  6 bits valid");
  uint32_t res = 0x00000000;
  res |= (Enable_Token_Issue << 31);  // Enable_Token_Issue : 31
  res |= (Repeat_Count << 16);        // Repeat_Count       : 23:16
  res |= Start_BD_ID;  // Start_BD_ID        : 3:0 in core / 5:0 in memt
  return res;
}

//////////////////set the core layer parameter class//////////////////
class SetParamPkt {
 public:
  explicit SetParamPkt(int ID, int src_row, int src_col, int addr,
                       const int32_t* data, int size)
      : ID_(ID), src_row_(src_row), src_col_(src_col), addr_(addr) {
    assert(size > 0 && size < 5);
    genPackets(data, size);
  }
  string to_string() {
    return uint2str(pkts_.data(), pkts_.size(), PKT_TYPE_CTRL, PKT_WORD_WIDTH);
  }
  string to_hex() { return uint2hex(pkts_.data(), pkts_.size()); }

 private:
  void genPackets(const int32_t* data, int size) {
    pkts_.push_back(genStreamPktHeader(ID_, pkt_type_, src_row_, src_col_));
    pkts_.push_back(genCtrlInfo(addr_, size, option_, ID_));
    uint32_t* uptr =
        const_cast<uint32_t*>(reinterpret_cast<const uint32_t*>(data));
    for (int i = 0; i < size; i++) {
      pkts_.push_back(uptr[i]);
    }
  }

 private:
  int ID_;
  int src_row_;
  int src_col_;
  int pkt_type_ = 0;
  int option_ = 00;
  int addr_;
  vector<uint32_t> pkts_;
};

//////////////////set the core BD class//////////////////
class genCOBDPkt {
 public:
  explicit genCOBDPkt(int ID, int src_row, int src_col, int BD, int hot_code,
                      int baseAddr, int length, int D1_step, int D0_step,
                      int D1_wrap, int D0_wrap, int D2_stepsize,
                      int Lock_ACQ_ID, int Lock_REL_ID, bool BD_casc_en = 0,
                      int idx_casc = 1)
      : ID_(ID),
        src_row_(src_row),
        src_col_(src_col),
        BD_(BD),
        baseAddr_(baseAddr),
        length_(length),
        D1_step_(D1_step),
        D0_step_(D0_step),
        D1_wrap_(D1_wrap),
        D0_wrap_(D0_wrap),
        D2_stepsize_(D2_stepsize),
        Lock_Acq_ID_(Lock_ACQ_ID),
        Lock_Rel_ID_(Lock_REL_ID) {
    size_ = PKT_WORD_WIDTH;
    pkt_type_ = 0;
    option_ = 0;
    BD_0_addr = 0;
    BD_1_addr = 1;
    BD_2_addr = 2;
    BD_3_addr = 3;
    BD_4_addr = 4;
    BD_5_addr = 5;
    genPackets(hot_code, BD_casc_en, idx_casc);
  }
  string to_string() {
    return uint2str(pkts_.data(), pkts_.size(), PKT_TYPE_CTRL, PKT_WORD_WIDTH);
  }
  string to_hex() { return uint2hex(pkts_.data(), pkts_.size()); }
  string ctrPktStr() { return ctrPktStr_; }
  string ctrPktHex() { return ctrPktHex_; }

 private:
  void genPackets(int hot_code, bool BD_casc_en, int idx_casc) {
    if ((hot_code & 0x00000001) != 0) {
      // IMG DMA_BD_0+1
      pkts_.push_back(genStreamPktHeader(ID_, pkt_type_, src_row_, src_col_));
      int BDReg0Addr =
          core_DMA_BD_0_ADDRESS + (core_DMA_ADDR_STEP * BD_) + BD_0_addr * 4;
      pkts_.push_back(genCtrlInfo(BDReg0Addr, size_, option_, ID_));
      pkts_.push_back(genCOBD0(baseAddr_, length_));
      pkts_.push_back(0u);
      ctrPktStr_ +=
          uint2str(pkts_.data(), pkts_.size(), PKT_TYPE_CTRL, PKT_WORD_WIDTH);
      ctrPktHex_ += uint2hex(pkts_.data(), pkts_.size());
      pkts_.pop_back();
      pkts_.pop_back();
      pkts_.pop_back();
      pkts_.pop_back();
    }

    if ((hot_code & 0x00000002) != 0) {
      // IMG DMA_BD_2+3
      pkts_.push_back(genStreamPktHeader(ID_, pkt_type_, src_row_, src_col_));
      int BDReg2Addr =
          core_DMA_BD_0_ADDRESS + (core_DMA_ADDR_STEP * BD_) + BD_2_addr * 4;
      pkts_.push_back(genCtrlInfo(BDReg2Addr, size_, option_, ID_));
      pkts_.push_back(genCOBD2(D1_step_, D0_step_));
      pkts_.push_back(genCOBD3(D1_wrap_, D0_wrap_, D2_stepsize_));
      ctrPktStr_ +=
          uint2str(pkts_.data(), pkts_.size(), PKT_TYPE_CTRL, PKT_WORD_WIDTH);
      ctrPktHex_ += uint2hex(pkts_.data(), pkts_.size());
      pkts_.pop_back();
      pkts_.pop_back();
      pkts_.pop_back();
      pkts_.pop_back();
    }

    if ((hot_code & 0x00000004) != 0) {
      // IMG DMA_BD_4+5
      pkts_.push_back(genStreamPktHeader(ID_, pkt_type_, src_row_, src_col_));
      int BDReg4Addr =
          core_DMA_BD_0_ADDRESS + (core_DMA_ADDR_STEP * BD_) + BD_4_addr * 4;
      pkts_.push_back(genCtrlInfo(BDReg4Addr, size_, option_, ID_));
      pkts_.push_back(0u);
      // if (BD_casc_en) { if (idx_casc==0) { pkts_.push_back( genCOBD5   (1,
      // Lock_Rel_ID_, (-1), Lock_Acq_ID_, 1, BD_+6, 1, 1)); }
      //                  else             { pkts_.push_back( genCOBD5   (1,
      //                  Lock_Rel_ID_, (-1), Lock_Acq_ID_, 0,     0, 0, 0)); }
      //                  }
      if (BD_casc_en) {
        if (idx_casc == 0) {
          pkts_.push_back(
              genCOBD5(0, Lock_Rel_ID_, -1, Lock_Acq_ID_, 1, BD_ + 6, 1, 1));
        } else {
          pkts_.push_back(
              genCOBD5(1, Lock_Rel_ID_, 0, Lock_Acq_ID_, 0, 0, 0, 0));
        }
      } else {
        pkts_.push_back(
            genCOBD5(1, Lock_Rel_ID_, (-1), Lock_Acq_ID_, 0, 0, 0, 1));
      }
      // if (BD_casc_en) { if (idx_casc==0) { pkts_.push_back( genCOBD5   ( 0,
      // Lock_ID_, ((Lock_ID_<4)?0:1), Lock_ID_, 1, BD_+6, 1, 1)); }
      //                  else             { pkts_.push_back( genCOBD5
      //                  (((Lock_ID_<4)?1:-1), Lock_ID_, ((Lock_ID_<4)?0:1),
      //                  Lock_ID_, 0,     0, 0, 0)); } }
      // else { pkts_.push_back( genCOBD5   (((Lock_ID_<4)?1:-1), Lock_ID_,
      // ((Lock_ID_<4)?0:1), Lock_ID_, 0, 0, 0, 1));}
      ctrPktStr_ +=
          uint2str(pkts_.data(), pkts_.size(), PKT_TYPE_CTRL, PKT_WORD_WIDTH);
      ctrPktHex_ += uint2hex(pkts_.data(), pkts_.size());
      pkts_.pop_back();
      pkts_.pop_back();
      pkts_.pop_back();
      pkts_.pop_back();
    }
  }

 private:
  int size_;
  int ID_;
  int src_row_;
  int src_col_;
  int BD_;
  int baseAddr_;
  int length_;
  int D1_step_;
  int D0_step_;
  int D1_wrap_;
  int D0_wrap_;
  int D2_stepsize_;
  int Lock_Acq_ID_;
  int Lock_Rel_ID_;
  int pkt_type_;
  int option_;
  int BD_0_addr;
  int BD_1_addr;
  int BD_2_addr;
  int BD_3_addr;
  int BD_4_addr;
  int BD_5_addr;
  vector<uint32_t> pkts_;
  string ctrPktStr_;
  string ctrPktHex_;
};

//////////////////set the memt BD class//////////////////
class genMTBDPkt {
 public:
  explicit genMTBDPkt(
      int ID, int src_row, int src_col, int BD, int hot_code, int length,
      int D0_Zero_Before, int baseAddr, int D0_wrap, int D0_stepsize,
      int D1_Zero_Before, int D1_wrap, int D1_stepsize, int D2_Zero_Before,
      int D2_wrap, int D2_stepsize, int D2_Zero_After, int D1_Zero_After,
      int D0_Zero_After, int D3_stepsize, int Lock_Rel_Value, int Lock_Rel_ID,
      int Lock_Acq_Enable, int Lock_Acq_Value, int Lock_Acq_ID,
      bool BD_casc_last = 0
      //                      int next_bd = 0, bool use_next_bd = 0
      )
      : ID_(ID),
        src_row_(src_row),
        src_col_(src_col),
        BD_(BD),
        baseAddr_(baseAddr),
        length_(length),
        D0_wrap_(D0_wrap),
        D1_wrap_(D1_wrap),
        D2_wrap_(D2_wrap),
        D0_stepsize_(D0_stepsize),
        D1_stepsize_(D1_stepsize),
        D2_stepsize_(D2_stepsize),
        D3_stepsize_(D3_stepsize),
        D0_Zero_Before_(D0_Zero_Before),
        D1_Zero_Before_(D1_Zero_Before),
        D2_Zero_Before_(D2_Zero_Before),
        D0_Zero_After_(D0_Zero_After),
        D1_Zero_After_(D1_Zero_After),
        D2_Zero_After_(D2_Zero_After),
        Lock_Rel_Value_(Lock_Rel_Value),
        Lock_Rel_ID_(Lock_Rel_ID),
        Lock_Acq_Enable_(Lock_Acq_Enable),
        Lock_Acq_Value_(Lock_Acq_Value),
        Lock_Acq_ID_(Lock_Acq_ID)
  // Next_bd_(next_bd), Use_Next_bd_(use_next_bd)
  {
    genPackets(hot_code, BD_casc_last);
  }
  string to_string() {
    return uint2str(pkts_.data(), pkts_.size(), PKT_TYPE_CTRL, PKT_WORD_WIDTH);
  }
  string to_hex() { return uint2hex(pkts_.data(), pkts_.size()); }
  string ctrPktStr() { return ctrPktStr_; }
  string ctrPktHex() { return ctrPktHex_; }

 private:
  void genPackets(int hot_code, bool BD_casc_last) {
    if ((hot_code & 0x00000001) != 0) {
      // MT DMA_BD_0+1
      pkts_.push_back(genStreamPktHeader(ID_, pkt_type_, src_row_, src_col_));
      int BDReg0Addr =
          memt_DMA_BD_0_ADDRESS + (memt_DMA_ADDR_STEP * BD_) + BD_0_addr * 4;
      pkts_.push_back(genCtrlInfo(BDReg0Addr, size_, option_, ID_));
      pkts_.push_back(genMTBD0(length_));
      // pkts_.push_back( genMTBD1          (D0_Zero_Before_, baseAddr_,
      // Next_bd_, Use_Next_bd_));
      pkts_.push_back(genMTBD1(D0_Zero_Before_, baseAddr_, 0, 0));
      ctrPktStr_ +=
          uint2str(pkts_.data(), pkts_.size(), PKT_TYPE_CTRL, PKT_WORD_WIDTH);
      ctrPktHex_ += uint2hex(pkts_.data(), pkts_.size());
      pkts_.pop_back();
      pkts_.pop_back();
      pkts_.pop_back();
      pkts_.pop_back();
    }

    if ((hot_code & 0x00000002) != 0) {
      // MT DMA_BD_2+3
      pkts_.push_back(genStreamPktHeader(ID_, pkt_type_, src_row_, src_col_));
      int BDReg2Addr =
          memt_DMA_BD_0_ADDRESS + (memt_DMA_ADDR_STEP * BD_) + BD_2_addr * 4;
      pkts_.push_back(genCtrlInfo(BDReg2Addr, size_, option_, ID_));
      pkts_.push_back(genMTBD2(BD_casc_last, D0_wrap_, D0_stepsize_));
      pkts_.push_back(genMTBD3(D1_Zero_Before_, D1_wrap_, D1_stepsize_));
      ctrPktStr_ +=
          uint2str(pkts_.data(), pkts_.size(), PKT_TYPE_CTRL, PKT_WORD_WIDTH);
      ctrPktHex_ += uint2hex(pkts_.data(), pkts_.size());
      pkts_.pop_back();
      pkts_.pop_back();
      pkts_.pop_back();
      pkts_.pop_back();
    }

    if ((hot_code & 0x00000004) != 0) {
      // MT DMA_BD_4+5
      pkts_.push_back(genStreamPktHeader(ID_, pkt_type_, src_row_, src_col_));
      int BDReg4Addr =
          memt_DMA_BD_0_ADDRESS + (memt_DMA_ADDR_STEP * BD_) + BD_4_addr * 4;
      pkts_.push_back(genCtrlInfo(BDReg4Addr, size_, option_, ID_));
      pkts_.push_back(genMTBD4(Enable_Compression_, D2_Zero_Before_, D2_wrap_,
                               D2_stepsize_));
      pkts_.push_back(genMTBD5(D2_Zero_After_, D1_Zero_After_, D0_Zero_After_,
                               D3_stepsize_));
      ctrPktStr_ +=
          uint2str(pkts_.data(), pkts_.size(), PKT_TYPE_CTRL, PKT_WORD_WIDTH);
      ctrPktHex_ += uint2hex(pkts_.data(), pkts_.size());
      pkts_.pop_back();
      pkts_.pop_back();
      pkts_.pop_back();
      pkts_.pop_back();
    }

    if ((hot_code & 0x00000008) != 0) {
      // MT DMA_BD_6+7
      pkts_.push_back(genStreamPktHeader(ID_, pkt_type_, src_row_, src_col_));
      int BDReg6Addr =
          memt_DMA_BD_0_ADDRESS + (memt_DMA_ADDR_STEP * BD_) + BD_6_addr * 4;
      pkts_.push_back(genCtrlInfo(BDReg6Addr, size_, option_, ID_));
      pkts_.push_back(genMTBD6(0, 0, 0));
      pkts_.push_back(genMTBD7(1, Lock_Rel_Value_, Lock_Rel_ID_,
                               Lock_Acq_Enable_, Lock_Acq_Value_,
                               Lock_Acq_ID_));
      ctrPktStr_ +=
          uint2str(pkts_.data(), pkts_.size(), PKT_TYPE_CTRL, PKT_WORD_WIDTH);
      ctrPktHex_ += uint2hex(pkts_.data(), pkts_.size());
      pkts_.pop_back();
      pkts_.pop_back();
      pkts_.pop_back();
      pkts_.pop_back();
    }
  }

 private:
  int size_ = PKT_WORD_WIDTH;
  int ID_;
  int src_row_;
  int src_col_;
  int BD_;
  int baseAddr_;
  int length_;
  int D0_wrap_;
  int D1_wrap_;
  int D2_wrap_;
  int D0_stepsize_;
  int D1_stepsize_;
  int D2_stepsize_;
  int D3_stepsize_;
  int D0_Zero_Before_;
  int D1_Zero_Before_;
  int D2_Zero_Before_;
  int D0_Zero_After_;
  int D1_Zero_After_;
  int D2_Zero_After_;
  int Lock_Rel_Value_;
  int Lock_Rel_ID_;
  int Lock_Acq_Enable_;
  int Lock_Acq_Value_;
  int Lock_Acq_ID_;
  int Enable_Compression_ = 0;
  int pkt_type_ = 0;
  int option_ = 00;
  int BD_0_addr = 0;
  int BD_1_addr = 1;
  int BD_2_addr = 2;
  int BD_3_addr = 3;
  int BD_4_addr = 4;
  int BD_5_addr = 5;
  int BD_6_addr = 6;
  int BD_7_addr = 7;
  int Next_bd_ = 0;
  int Use_Next_bd_ = 0;
  vector<uint32_t> pkts_;
  string ctrPktStr_;
  string ctrPktHex_;
};

//////////////////set the start queue BD class//////////////////
class genBDStartQPkt {
 public:
  explicit genBDStartQPkt(int ID, int src_row, int src_col, int BD, int channel,
                          bool isS2MM, bool isCORE, int repeat = 0)
      : ID_(ID),
        src_row_(src_row),
        src_col_(src_col),
        BD_(BD),
        channel_(channel),
        S2MMorMM2S_(isS2MM),
        isCORE_(isCORE),
        repeat_(repeat) {
    genPackets();
  }
  string to_string() {
    return uint2str(pkts_.data(), pkts_.size(), PKT_TYPE_CTRL, PKT_WORD_WIDTH);
  }
  string to_hex() { return uint2hex(pkts_.data(), pkts_.size()); }

 private:
  void genPackets() {
    DMA_Ctrl_0_ADDRESS =
        isCORE_ ? core_DMA_Ctrl_0_ADDRESS : memt_DMA_Ctrl_0_ADDRESS;
    DMA_Ctrl_ADDR_STEP =
        isCORE_ ? core_DMA_Ctrl_ADDR_STEP : memt_DMA_Ctrl_ADDR_STEP;
    DMA_IO_ADDR_STEP = isCORE_ ? core_DMA_IO_ADDR_STEP : memt_DMA_IO_ADDR_STEP;
    pkt_type_ = 0;
    option_ = 0;
    size_ = PKT_WORD_WIDTH;
    pkts_.push_back(genStreamPktHeader(ID_, pkt_type_, src_row_, src_col_));
    int DMACtrlAddr = DMA_Ctrl_0_ADDRESS + (DMA_Ctrl_ADDR_STEP * channel_) +
                      DMA_IO_ADDR_STEP * S2MMorMM2S_;
    pkts_.push_back(genCtrlInfo(DMACtrlAddr, size_, option_, ID_));
    pkts_.push_back(0x00000000);
    pkts_.push_back(genStartQ(isCORE_ ? 0 : 1, repeat_, BD_));
    // pkts_.push_back( genStartQ         (0, repeat_, BD_));
  }

 private:
  unsigned int DMA_Ctrl_0_ADDRESS;
  unsigned int DMA_Ctrl_ADDR_STEP;
  unsigned int DMA_IO_ADDR_STEP;
  unsigned int size_;
  unsigned int ID_;
  unsigned int src_row_;
  unsigned int src_col_;
  unsigned int BD_;
  unsigned int channel_;
  unsigned int pkt_type_;
  unsigned int option_;
  bool S2MMorMM2S_;
  bool isCORE_;
  unsigned int repeat_;
  vector<uint32_t> pkts_;
};

string uint2str(uint32_t* pkts, int len, PKT_TYPE type, int WIDTH) {
  assert(type == PKT_TYPE_CTRL || type == PKT_TYPE_DATA);
  assert(WIDTH > 0);
  stringstream ss;
  ss.clear();
  ss.str("");
  int row = len / WIDTH;
  row += (len % WIDTH == 0) ? 0 : 1;
  for (int i = 0; i < row; i++) {
    if (i == row - 1) ss << "TLAST" << endl;
    for (int j = 0; j < WIDTH; j++) {
      int idx = i * WIDTH + j;
      if (idx < len) {
        ss << (int32_t)pkts[idx] << " ";
      }
    }
    ss << endl;
  }
  return ss.str();
}

string uint2hex(uint32_t* tmp, int len, int WIDTH) {
  assert(WIDTH > 0);
  stringstream ss;
  ss.clear();
  ss.str("");
  int row = len / WIDTH;
  row += (len % WIDTH == 0) ? 0 : 1;
  for (int i = 0; i < row - 1; i++) {
    int row_base = i * WIDTH;
    for (int j = 0; j < WIDTH; j++) {
      ss << std::hex << std::setw(8) << std::setfill('0')
         << tmp[row_base + 3 - j] << " ";
    }
    ss << "\n";
  }
  int mod = len % WIDTH;
  mod = (mod == 0) ? WIDTH : mod;
  for (int i = 0; i < WIDTH - mod; i++) {
    ss << std::hex << std::setw(8) << std::setfill('0') << 0 << " ";
  }
  for (int i = 0; i < mod; i++) {
    ss << std::hex << std::setw(8) << std::setfill('0') << tmp[len - i - 1]
       << " ";
  }
  ss << "\n";
  return ss.str();
}

}  // namespace alu
}  // namespace xv2dpu
}  // namespace cosim
