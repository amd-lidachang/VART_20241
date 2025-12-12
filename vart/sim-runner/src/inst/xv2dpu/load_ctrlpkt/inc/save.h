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
#include <iomanip>
#include <map>
#include <string>
#include <vector>
#include "cache.h"
#include "debug.h"
#include "def.h"
#include "simCfg.hpp"

namespace cosim {
namespace xv2dpu {
namespace save {

using namespace std;
using namespace cosim::xv2dpu::load;

struct save {
  save(){};
  ~save(){};

  int32_t dependOn, dependBy;
  int32_t const_en;
  int32_t mt_addr;
  int32_t reg_id;
  int32_t const_value;
  int32_t jump_write;
  int32_t jump_read;
  int32_t length;
  int32_t channel;
  int32_t ddr_addr;
  static int32_t ping_pong_flag;

  std::vector<uint64_t> saveGenPacket();
  void savePacketFile();
};

int32_t save::ping_pong_flag = 0;

// word0(header) + word1(control) + word2(data0) + word3(data1)
std::vector<uint64_t> save::saveGenPacket() {
  int32_t bd_ping = 32;
  int32_t bd_pong = 33;
  int32_t dma_chn = 9;
  uint32_t header = 0;
  uint64_t control = 0;
  int32_t bd0 = 0;
  uint64_t bd1 = 0;
  int32_t bd2 = 0;
  int32_t d1_wrap = 0;
  uint64_t bd3 = 0;
  int32_t bd4 = 0;
  uint64_t bd5 = 0;
  int32_t bd6 = 0;
  uint64_t bd7 = 0;
  int32_t dma_ctrl = 0;
  uint64_t startque = 0;
  // int32_t stream_pkt_id = 0;
  int32_t data_beat_addr = 1;  // mean 2 word
  // int32_t split_cnt = 0;
  int32_t channel_align = 0;
  uint64_t temp0 = 0;
  uint64_t temp1 = 0;
  int32_t ping = 0;
  int32_t pong = 1;
  int32_t bd = 0;
  uint32_t PORT_IDX=1;
  // int32_t num = 0;
  std::vector<uint64_t> pkt(20, 0);
  if (ping_pong_flag == 0)
    bd = bd_ping;
  else
    bd = bd_pong;
  // cout <<"ping_pong_flag:"<<ping_pong_flag <<endl;
  // cout <<"bd:"<<bd <<endl;
  if (channel % 32 == 0)
    channel_align = channel / 32;
  else
    channel_align = channel / 32 + 1;
  for (int32_t stream_pk_id = 0; stream_pk_id < 2; stream_pk_id++) {
    // bd6,bd7
    header = log::paritycheck(stream_pk_id);
    control = (log::bd_addr(bd, 6) & 0xfffff) | (data_beat_addr << 20);
    control = log::paritycheck(control);
    bd7 = 0x80000000;  // bd is valid or not
    bd6 = 0;
    temp0 = control << 32 | header;
    temp1 = bd7 << 32 | bd6;
    pkt.at(stream_pk_id * 10 + 0) = temp0;
    pkt.at(stream_pk_id * 10 + 1) = temp1;

    // cout <<hex<<setw(16)<<setfill('0')<<temp0<<endl;
    // cout <<hex<<setw(16)<<setfill('0')<<temp1<<endl;
    // bd4, bd5
    header = log::paritycheck(stream_pk_id);
    control = (log::bd_addr(bd, 4) & 0xfffff) | (data_beat_addr << 20);
    control = log::paritycheck(control);
    bd5 = 0x0;
    bd4 = 0x7;
    temp0 = control << 32 | header;
    temp1 = bd5 << 32 | bd4;
    pkt.at(stream_pk_id * 10 + 2) = temp0;
    pkt.at(stream_pk_id * 10 + 3) = temp1;
    // cout <<hex<<setw(16)<<setfill('0')<<temp0<<endl;
    // cout <<hex<<setw(16)<<setfill('0')<<temp1<<endl;
    // bd2,bd3
    header = log::paritycheck(stream_pk_id);
    control = (log::bd_addr(bd, 2) & 0xfffff) | (data_beat_addr << 20);
    control = log::paritycheck(control);
    if(length>=1024)//d1_wrap 10bit
      d1_wrap = 0;
    else
      d1_wrap = length;
    bd3 = (jump_read * 4 - 1) | (d1_wrap << 17);
    bd2 = channel_align * 4 << 17;
    temp0 = control << 32 | header;
    temp1 = bd3 << 32 | bd2;
    pkt.at(stream_pk_id * 10 + 4) = temp0;
    pkt.at(stream_pk_id * 10 + 5) = temp1;
    // cout <<hex<<setw(16)<<setfill('0')<<temp0<<endl;
    // cout <<hex<<setw(16)<<setfill('0')<<temp1<<endl;
    // bd0,bd1
    header = log::paritycheck(stream_pk_id);
    control = (log::bd_addr(bd, 0) & 0xfffff) | (data_beat_addr << 20);
    control = log::paritycheck(control);
    bd1 = mt_addr * 4;
    bd0 = channel_align * 4 * length;
    temp0 = control << 32 | header;
    temp1 = bd1 << 32 | bd0;
    pkt.at(stream_pk_id * 10 + 6) = temp0;
    pkt.at(stream_pk_id * 10 + 7) = temp1;
    // cout <<hex<<setw(16)<<setfill('0')<<temp0<<endl;
    // cout <<hex<<setw(16)<<setfill('0')<<temp1<<endl;
    // dma ctrl and start queue
    header = log::paritycheck(stream_pk_id);
    control = (log::dma_ctrl_addr(dma_chn) & 0xfffff) | (data_beat_addr << 20);
    control = log::paritycheck(control);
    dma_ctrl = (stream_pk_id + PORT_IDX*2) << 8;
    if (ping_pong_flag == ping)
      startque = bd_ping | (0x1 << 31);
    else
      startque = bd_pong | (0x1 << 31);
    temp0 = control << 32 | header;
    temp1 = startque << 32 | dma_ctrl;
    pkt.at(stream_pk_id * 10 + 8) = temp0;
    pkt.at(stream_pk_id * 10 + 9) = temp1;
    // cout <<hex<<setw(16)<<setfill('0')<<temp0<<endl;
    // cout <<hex<<setw(16)<<setfill('0')<<temp1<<endl;
  }
  if (ping_pong_flag == ping)
    ping_pong_flag = pong;
  else
    ping_pong_flag = ping;

  return pkt;
}
void save::savePacketFile() {
  // string logFile = "./case/log/save/save_pkt.txt";
  string logFile = SimCfgProxy::Params().SAVE_CTRLPKT_PATH + "/save_pkt.txt";
  // string intFile = "./case/save/save_pkt_int.txt";
  std::vector<uint64_t> temp = saveGenPacket();
  log::saveToPktHex(logFile, temp, 20);
  // log::saveToPktint(intFile,temp,20);
}

}  // namespace save
}  // namespace xv2dpu
}  // namespace cosim
