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

#include "Save.hpp"
#include "save.h"
#include "simCfg.hpp"
#include "simUtil.hpp"

template <>
void Save<DPUVersion::XV2DPU>::save_gen_ctrlpkg() const {
  if (!SimCfgProxy::Params().CO_SIM_ON) return;

  cosim::xv2dpu::save::save save_ctrlpkt;
  save_ctrlpkt.dependOn = dpdon_.to_ullong();
  save_ctrlpkt.dependBy = dpdby_.to_ullong();
  save_ctrlpkt.const_en = const_en_;
  save_ctrlpkt.mt_addr = bank_addr_;
  save_ctrlpkt.reg_id = reg_id_;
  save_ctrlpkt.const_value = const_value_;
  save_ctrlpkt.jump_write = jump_write_;
  save_ctrlpkt.jump_read = jump_read_;
  save_ctrlpkt.length = length_;
  save_ctrlpkt.channel = channel_;
  save_ctrlpkt.ddr_addr = ddr_addr_;

  save_ctrlpkt.savePacketFile();
}