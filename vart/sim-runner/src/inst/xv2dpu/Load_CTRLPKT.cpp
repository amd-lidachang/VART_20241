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

#include "Load.hpp"
#include "load.h"
#include "simCfg.hpp"
#include "simUtil.hpp"

template <>
void Load<DPUVersion::XV2DPU>::load_gen_ctrlpkg() const {
  if (!SimCfgProxy::Params().CO_SIM_ON) return;

  cosim::xv2dpu::load::load load_ctrlpkt;
  load_ctrlpkt.dependOn = dpdon_.to_ullong();
  load_ctrlpkt.dependBy = dpdby_.to_ullong();
  load_ctrlpkt.mt_dst = mt_dst_;
  load_ctrlpkt.mt_addr = bank_addr_;
  load_ctrlpkt.block_num = block_num_;
  load_ctrlpkt.broadcast = broadcast_;
  load_ctrlpkt.pad_start = pad_start_;
  load_ctrlpkt.pad_end = pad_end_;
  load_ctrlpkt.pad_idx = pad_idx_;
  load_ctrlpkt.reg_id = reg_id_;
  load_ctrlpkt.const_value = const_value_;
  load_ctrlpkt.jump_read = jump_read_;
  load_ctrlpkt.mode_avg = mode_avg_;
  load_ctrlpkt.length = length_;
  if(mt_dst_==2)
      load_ctrlpkt.channel = channel_*8;
  else if(mt_dst_==3)
      load_ctrlpkt.channel = channel_*2;
  else if(mt_dst_==4)
      load_ctrlpkt.channel = channel_*2;
  else
      load_ctrlpkt.channel = channel_;
  load_ctrlpkt.ddr_addr = ddr_addr_;
  load_ctrlpkt.jump_write = jump_write_;
  load_ctrlpkt.jump_write_endl = jump_write_endl_;

  load_ctrlpkt.savePacketFile();
}
