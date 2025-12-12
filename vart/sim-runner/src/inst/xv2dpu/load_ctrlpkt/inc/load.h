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

#ifndef LOAD_H_
#define LOAD_H_

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
namespace load {

using namespace std;

struct load {
  load(){};
  ~load(){};

  int32_t dependOn, dependBy;
  int32_t mt_dst;
  int32_t mt_addr;
  int32_t block_num;
  int32_t broadcast;
  int32_t pad_start;
  int32_t pad_end;
  int32_t pad_idx;
  int32_t reg_id;
  int32_t const_value;
  int32_t jump_read;
  int32_t mode_avg;
  int32_t length;
  int32_t channel;
  int32_t ddr_addr;
  int32_t jump_write;
  int32_t jump_write_endl;
  static int32_t load_ping_pong_flag;
  static int32_t load_wb_ping_pong_flag_l;
  static int32_t load_wb_ping_pong_flag_r;
  static int32_t jump_bn[];

  std::vector<uint64_t> loadGenImgPkt();          // load image
  std::vector<uint64_t> loadGenWBPkt(int wgt_p);  // load weight and Bias
  void savePacketFile();
};

/********************************
 * load
 * ********************************/
int32_t load::load_ping_pong_flag = 0;
int32_t load::load_wb_ping_pong_flag_l = 0;
int32_t load::load_wb_ping_pong_flag_r = 0;
int32_t load::jump_bn[] = {0, 0};

std::vector<uint64_t> load::loadGenImgPkt() {
  int32_t bd_ping = 2;
  int32_t bd_pong = 3;
  int32_t dma_chn = 2;
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
  uint32_t channel_align = 0;
  uint64_t temp0 = 0;
  uint64_t temp1 = 0;
  int32_t ping = 0;
  int32_t pong = 1;
  int32_t bd = 0;
  uint32_t PKT_INST=1;
  // int32_t num = 0;
  std::vector<uint64_t> pkt(20, 0);
  if (load_ping_pong_flag == 0)
    bd = bd_ping;
  else
    bd = bd_pong;
  cout << "load_ping_pong_flag:" << load_ping_pong_flag << endl;
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
    // bd2,bd3
    header = log::paritycheck(stream_pk_id);
    control = (log::bd_addr(bd, 2) & 0xfffff) | (data_beat_addr << 20);
    control = log::paritycheck(control);
    // mt align 4 words
    if(length>=1024)//d1_wrap 10bit
      d1_wrap = 0;
    else
      d1_wrap = length;
    bd3 = ((jump_write << 2) - 1) | (d1_wrap << 17);
    bd2 = channel_align << 2 << 17;
    temp0 = control << 32 | header;
    temp1 = bd3 << 32 | bd2;
    pkt.at(stream_pk_id * 10 + 4) = temp0;
    pkt.at(stream_pk_id * 10 + 5) = temp1;
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
    dma_ctrl = (stream_pk_id+PKT_INST*2) << 8;
    if (load_ping_pong_flag == ping)
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
  if (load_ping_pong_flag == ping)
    load_ping_pong_flag = pong;
  else
    load_ping_pong_flag = ping;

  return pkt;
}

std::vector<uint64_t> load::loadGenWBPkt(int wgt_p) {
  int32_t bd_ping_l1 = 6;
  int32_t bd_pong_l1 = 7;
  int32_t bd_ping_l2 = 32;
  int32_t bd_pong_l2 = 33;
  int32_t dma_chn_l1 = 0;
  int32_t dma_chn_l2 = 1;
  int32_t dma_chn = 0;
  int32_t load_wgt_p = 2;
  int32_t wgt_num_col = 2;

  uint32_t header = 0;
  uint64_t control = 0;
  uint32_t bd0 = 0;
  uint64_t bd1 = 0;
  uint32_t bd2 = 0;
  uint64_t bd3 = 0;
  uint32_t bd4 = 0;
  uint64_t bd5 = 0;
  uint32_t bd6 = 0;
  uint64_t bd7 = 0;
  int32_t dma_ctrl = 0;
  uint64_t startque = 0;
  // int32_t stream_pkt_id = 0;
  int32_t data_beat_addr = 1;  // mean 2 word
  // int32_t split_cnt = 0;
  int32_t channel_align = 0;
  int32_t addr_offset = 131072;
  int32_t vld_nbyte = 0;
  int32_t jump_byte = 0;
  int32_t jump_byte_add = 0;
  int32_t d3_step = 0;
  int32_t d2_wrap = 0;
  int32_t d2_step = 0;
  // int32_t d1_wrap = 0;
  // int32_t d1_step = 0;
  // int32_t d0_wrap = 0;
  // int32_t d0_step = 0;
  uint64_t temp0 = 0;
  uint64_t temp1 = 0;
  int32_t ping = 0;
  int32_t pong = 1;
  int32_t bd = 0;
  //uint32_t PKT_INST=1;
  // int32_t num = 0;
  int32_t jump_align = 0;
  int32_t jump_remain = 0;
  std::vector<uint64_t> pkt(10 * wgt_num_col * load_wgt_p, 0);
  // cout <<"bd:"<<bd <<endl;
  //>>1 line17109
  //load control need recover instr channel value.
  //if(mt_dst==2)
  //  channel = channel*8;
  //else if (mt_dst==3)
  //  channel = channel*2;
  //else if (mt_dst==4)
  //  channel = channel*2;

  if (mt_dst == 1)
    jump_align = (block_num >> 5) >> 1;  // need caution
  else if (mt_dst == 2)
    jump_align = (channel >> 8) >> 1;  // need caution
  else if (mt_dst == 3)
    jump_align = (channel >> 6) >> 1;  // need caution
  else if (mt_dst == 4)
    jump_align = (channel >> 6) >> 1;  // need caution
  if (wgt_p == 0) {
    if (load_wb_ping_pong_flag_l == 0)
      bd = bd_ping_l1;
    else
      bd = bd_pong_l1;
  } else {
    if (load_wb_ping_pong_flag_r == 0)
      bd = bd_ping_l2;
    else
      bd = bd_pong_l2;
  }

  if (mt_dst == 1)
    jump_remain = (((block_num >> 5) & 0x1) > wgt_p) ? 1 : 0;
  else if (mt_dst == 2)
    jump_remain = (((channel >> 8) & 0x1) > wgt_p) ? 1 : 0;
  else if (mt_dst == 3)
    jump_remain = (((channel >> 6) & 0x1) > wgt_p) ? 1 : 0;
  else if (mt_dst == 4)
    jump_remain = (((channel >> 6) & 0x1) > wgt_p) ? 1 : 0;

  jump_bn[wgt_p] = (jump_align + jump_remain);
  if (mt_dst == 1)
    vld_nbyte = (jump_read * length) << 5;
  else if (mt_dst == 2)
    vld_nbyte = 256;
  else if (mt_dst == 3)
    vld_nbyte = (length << 6);  // len_hp
  else if (mt_dst == 4)
    vld_nbyte = 64;

  jump_byte = (vld_nbyte * jump_bn[0]) >> 6;
  if (wgt_p == 0)
    jump_byte_add = 0;
  else
    jump_byte_add = jump_byte;
  // cout<<"vld_nbyte jump_byte jump_bn mt_addr"<<vld_nbyte<<"  "<<jump_byte<<"
  // "<<jump_bn[0]<<" "<<mt_addr<<endl;
  int blk_num = 0;
  if (mt_dst == 1)
    blk_num = (jump_bn[wgt_p] << 5);
  else
    blk_num = block_num;
  // calc channel
  int cfg_channel = 0;
  if (mt_dst == 1)  // wgt
    cfg_channel = channel;
  else if (mt_dst == 2)  // bias
    cfg_channel = (jump_bn[wgt_p] << 8);
  else if (mt_dst == 3)  // misc wgt
    cfg_channel = (jump_bn[wgt_p] << 6);
  else if (mt_dst == 4)  // misc wgt
    cfg_channel = (jump_bn[wgt_p] << 6);
  if (cfg_channel % 32 == 0)
    channel_align = cfg_channel / 32;
  else
    channel_align = cfg_channel / 32 + 1;

  // calc packet data
  for (int32_t pkt_num = 0; pkt_num < wgt_num_col; pkt_num++) {
    int stream_pk_id = pkt_num;
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

    // bd4, bd5
    header = log::paritycheck(stream_pk_id);
    control = (log::bd_addr(bd, 4) & 0xfffff) | (data_beat_addr << 20);
    control = log::paritycheck(control);
    // jump_align = (block_num>>5)>>1;//need caution
    // cout<<"channel_align: "<<cfg_channel<<endl;
    if (mt_dst == 1)  // load wgt
      d2_wrap = channel_align;
    else if (mt_dst == 2)  // load bias
      //d2_wrap = channel_align >> 3;
      d2_wrap = 0;
    else
      //d2_wrap = channel_align >> 1;
      d2_wrap = 0;
    if(mt_dst==1){
      d2_step = (jump_write<<2)-1;
      d3_step =(jump_write_endl<<2)-1;
    }
    else{
      d2_step = (jump_write_endl<<2)-1;
      d3_step = 0;
    }

    bd5 = d3_step; 
    //bd4 = ((jump_write << 2) - 1) | (d2_wrap << 17);
    bd4 = d2_step | (d2_wrap<<17);

    temp0 = control << 32 | header;
    temp1 = bd5 << 32 | bd4;
    pkt.at(stream_pk_id * 10 + 2) = temp0;
    pkt.at(stream_pk_id * 10 + 3) = temp1;
    // bd2,bd3,d1,d0
    header = log::paritycheck(stream_pk_id);
    control = (log::bd_addr(bd, 2) & 0xfffff) | (data_beat_addr << 20);
    control = log::paritycheck(control);
    // mt align 4 words
    if (mt_dst == 1) {  // conv wgt
      bd3 = (64 - 1) | (length << 17);
      bd2 = 64 << 17;  //*256/4
    } else if (mt_dst == 2) {
      //bd3 = (16 - 1) | (length << 17);
      //bd2 = 16 << 17;  //*256/4
      bd3 = ((jump_write<<2) -1) | ((channel_align>>3)<<17);
      bd2 = (length*16) <<17;
    } else {
      //bd3 = (4 - 1) | (length << 17);
      //bd2 = 4 << 17;  //*256/4
      bd3 = ((jump_write<<2) -1) | ((channel_align>>1)<<17);
      bd2 = (length*4) <<17;
    }
    temp0 = control << 32 | header;
    temp1 = bd3 << 32 | bd2;
    pkt.at(stream_pk_id * 10 + 4) = temp0;
    pkt.at(stream_pk_id * 10 + 5) = temp1;
    // bd0,bd1
    header = log::paritycheck(stream_pk_id);
    control = (log::bd_addr(bd, 0) & 0xfffff) | (data_beat_addr << 20);
    control = log::paritycheck(control);
    // jump_align = (block_num>>5)>>1;

    bd1 = ((mt_addr + jump_byte_add) << 2) + addr_offset;
    bd0 = (length * channel_align * blk_num) << 1;
    temp0 = control << 32 | header;
    temp1 = bd1 << 32 | bd0;
    pkt.at(stream_pk_id * 10 + 6) = temp0;
    pkt.at(stream_pk_id * 10 + 7) = temp1;
    // dma ctrl and start queue
    header = log::paritycheck(stream_pk_id);
    dma_chn = (wgt_p == 0) ? dma_chn_l1 : dma_chn_l2;
    control = (log::dma_ctrl_addr(dma_chn) & 0xfffff) | (data_beat_addr << 20);
    control = log::paritycheck(control);
    //dma_ctrl = (stream_pk_id+PKT_INST*2) << 8;
    dma_ctrl = (stream_pk_id) << 8;
    startque = bd | (0x1 << 31);
    temp0 = control << 32 | header;
    temp1 = startque << 32 | dma_ctrl;
    pkt.at(stream_pk_id * 10 + 8) = temp0;
    pkt.at(stream_pk_id * 10 + 9) = temp1;
  }
  if (wgt_p == 0) {
    if (load_wb_ping_pong_flag_l == ping)
      load_wb_ping_pong_flag_l = pong;
    else
      load_wb_ping_pong_flag_l = ping;
  } else {
    if (load_wb_ping_pong_flag_r == ping)
      load_wb_ping_pong_flag_r = pong;
    else
      load_wb_ping_pong_flag_r = ping;
  }

  return pkt;
}
void load::savePacketFile() {
  // string logFile = "./case/log/save/save_pkt.txt";
  string logFile =
      SimCfgProxy::Params().LOAD_CTRLPKT_PATH + "/load_img_pkt.txt";
  // string intFile = "./load_pkt_int.txt";
  string logFile_l =
      SimCfgProxy::Params().LOAD_CTRLPKT_PATH + "/load_pkt_l.txt";
  string logFile_r =
      SimCfgProxy::Params().LOAD_CTRLPKT_PATH + "/load_pkt_r.txt";
  std::vector<uint64_t> temp;
  int array_l = 20;
  if (mt_dst == 0) {  // load image
    temp = loadGenImgPkt();
    log::saveToPktHex(logFile, temp, array_l);
    // log::saveToPktint(intFile,temp,array_l);
  } else if (mt_dst == 1) {  // load conv weight
    if (block_num <= 32) {   // only need 1 load wgt port
      temp = loadGenWBPkt(0);
      log::saveToPktHex(logFile_l, temp, array_l);
    } else {
      temp = loadGenWBPkt(0);
      log::saveToPktHex(logFile_l, temp, array_l);
      temp = loadGenWBPkt(1);
      log::saveToPktHex(logFile_r, temp, array_l);
    }
  } else if (mt_dst == 2) {  // load conv bias
    if (channel <= 256) {    // only need 1 load wgt port
      temp = loadGenWBPkt(0);
      log::saveToPktHex(logFile_l, temp, array_l);
    } else {
      temp = loadGenWBPkt(0);
      log::saveToPktHex(logFile_l, temp, array_l);
      temp = loadGenWBPkt(1);
      log::saveToPktHex(logFile_r, temp, array_l);
    }
  } else if (mt_dst == 3) {  // load misc wgt
    if (channel <= 64) {     // only need 1 load wgt port
      temp = loadGenWBPkt(0);
      log::saveToPktHex(logFile_l, temp, array_l);
    } else {
      temp = loadGenWBPkt(0);
      log::saveToPktHex(logFile_l, temp, array_l);
      temp = loadGenWBPkt(1);
      log::saveToPktHex(logFile_r, temp, array_l);
    }
  } else if (mt_dst == 4) {  // load misc bias
    if (channel <= 64) {     // only need 1 load wgt port
      temp = loadGenWBPkt(0);
      log::saveToPktHex(logFile_l, temp, array_l);
    } else {
      temp = loadGenWBPkt(0);
      log::saveToPktHex(logFile_l, temp, array_l);
      temp = loadGenWBPkt(1);
      log::saveToPktHex(logFile_r, temp, array_l);
    }
  }
}

}  // namespace load
}  // namespace xv2dpu
}  // namespace cosim

#endif
