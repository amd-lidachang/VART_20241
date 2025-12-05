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

enum class MISC_CASE : int {
  CASE_NONE = -1,
  CASE_MACC = 0,
  CASE_AVGP = 1,
  CASE_MAXP = 2,
  CASE_ELEWMUL = 3,  // elew add/mul
  CASE_ELEWADD = 4,  // elew add/mul
  CASE_ELEWS = 5,    // elew shift
  CASE_TEST = 6      // elew shift
};

constexpr int S2MM = 0;
constexpr int MM2S = 1;
constexpr int PKT_WORD_WIDTH = 2;
constexpr int CORE_DMA_WORD_WIDTH = 1;
constexpr int core_DMA_BD_0_ADDRESS = 0x1D000;
constexpr int core_DMA_ADDR_STEP = 0x20;
constexpr int core_DMA_Ctrl_0_ADDRESS = 0x1DE00;
constexpr int core_DMA_Ctrl_ADDR_STEP = 0x08;
constexpr int core_DMA_IO_ADDR_STEP = 0x10;
constexpr int memt_DMA_BD_0_ADDRESS = 0xA0000;
constexpr int memt_DMA_ADDR_STEP = 0x20;
constexpr int memt_DMA_Ctrl_0_ADDRESS = 0xA0600;
constexpr int memt_DMA_Ctrl_ADDR_STEP = 0x08;
constexpr int memt_DMA_IO_ADDR_STEP = 0x30;

constexpr int ID_CTRL = 0;
// this is lengh of layer parameter
constexpr int LAYER_PARAM_WORD_SIZE = 33;

constexpr int PARAM_BASE_ADDR_PING = 0xFF60;
constexpr int PARAM_BASE_ADDR_PONG = 0xFEC0;
constexpr int IMG_BASE_ADDR_PING = 0x0000;
constexpr int IMG_BASE_ADDR_PONG = 0x8000;
constexpr int WGT_BASE_ADDR_PING = 0x4000;
constexpr int WGT_BASE_ADDR_PONG = 0xC000;
constexpr int OFM_BASE_ADDR_PING = 0x1000;
constexpr int OFM_BASE_ADDR_PONG = 0x9000;

constexpr int BD_IMG_PING = 0;
constexpr int BD_IMG_PONG = 1;
constexpr int BD_WGT_PING = 2;
constexpr int BD_WGT_PONG = 3;
constexpr int BD_OFM_PING = 4;
constexpr int BD_OFM_PONG = 5;

constexpr int IMG_CHANNEL = 0;
constexpr int WGT_CHANNEL = 1;
constexpr int OFM_CHANNEL = 0;

constexpr int MT_SIZE = 0x20000;

constexpr int MT_IMG_CONV_BASE_ADDR_IFM_PING = 0x20000;
constexpr int MT_IMG_CONV_BASE_ADDR_IFM_PONG = 0x20000;
constexpr int MT_IMG_CONV_BASE_ADDR_OFM_PING = 0x21000;
constexpr int MT_IMG_CONV_BASE_ADDR_OFM_PONG = 0x21000;
constexpr int MT_IMG_MISC_BASE_ADDR_IFM_PING = 0x28000;
constexpr int MT_IMG_MISC_BASE_ADDR_IFM_PONG = 0x28000;
constexpr int MT_IMG_MISC_BASE_ADDR_OFM_PING = 0x2C000;
constexpr int MT_IMG_MISC_BASE_ADDR_OFM_PONG = 0x2C000;
constexpr int MT_IMG_LOAD_BASE_ADDR_PING = 0x20000;
constexpr int MT_IMG_LOAD_BASE_ADDR_PONG = 0x20000;
constexpr int MT_IMG_SAVE_BASE_ADDR_PING = 0x21000;
constexpr int MT_IMG_SAVE_BASE_ADDR_PONG = 0x21000;

constexpr int MT_WGT_CONV_BASE_ADDR_PING = 0x20000;
constexpr int MT_WGT_CONV_BASE_ADDR_PONG = 0x20000;
constexpr int MT_WGT_MISC_BASE_ADDR_PING = 0x24000;
constexpr int MT_WGT_MISC_BASE_ADDR_PONG = 0x24000;
constexpr int MT_WGT_LOAD_BASE_ADDR_PING = 0x20000;
constexpr int MT_WGT_LOAD_BASE_ADDR_PONG = 0x20000;

constexpr int MT_IMG_BD_CONV_PING_IFM_H0 = 2;
constexpr int MT_IMG_BD_CONV_PONG_IFM_H0 = 3;
constexpr int MT_IMG_BD_CONV_PING_IFM_H1 = 24;
constexpr int MT_IMG_BD_CONV_PONG_IFM_H1 = 25;
// constexpr int MT_IMG_BD_CONV_PING_OFM_H0 = 26;
// constexpr int MT_IMG_BD_CONV_PONG_OFM_H0 = 27;
// constexpr int MT_IMG_BD_CONV_PING_OFM_H1 = 28;
// constexpr int MT_IMG_BD_CONV_PONG_OFM_H1 = 29;
constexpr int MT_IMG_BD_CONV_PING_OFM_H0 = 24;
constexpr int MT_IMG_BD_CONV_PONG_OFM_H0 = 25;
constexpr int MT_IMG_BD_CONV_PING_OFM_H1 = 26;
constexpr int MT_IMG_BD_CONV_PONG_OFM_H1 = 27;
constexpr int MT_IMG_BD_MISC_PING_IFM = 30;
constexpr int MT_IMG_BD_MISC_PONG_IFM = 31;
constexpr int MT_IMG_BD_MISC_PING_OFM = 2;
constexpr int MT_IMG_BD_MISC_PONG_OFM = 3;
// constexpr int MT_IMG_BD_LOAD_PING = 0;
// constexpr int MT_IMG_BD_LOAD_PONG = 1;
// constexpr int MT_IMG_BD_SAVE_PING = 26;
// constexpr int MT_IMG_BD_SAVE_PONG = 27;
constexpr int MT_IMG_BD_LOAD_PING = 0;
constexpr int MT_IMG_BD_LOAD_PONG = 1;
constexpr int MT_IMG_BD_SAVE_PING = 26;
constexpr int MT_IMG_BD_SAVE_PONG = 27;

constexpr int MT_IMG_CHN_CONV_IFM_H0 = 0;
constexpr int MT_IMG_CHN_CONV_IFM_H1 = 3;
constexpr int MT_IMG_CHN_CONV_OFM_H0 = 1;
constexpr int MT_IMG_CHN_CONV_OFM_H1 = 4;
constexpr int MT_IMG_CHN_MISC_IFM = 1;
constexpr int MT_IMG_CHN_MISC_OFM = 2;
constexpr int MT_IMG_CHN_LOAD = 0;
constexpr int MT_IMG_CHN_SAVE = 1;

constexpr int MT_WGT_BD_CONV_PING = 2;
constexpr int MT_WGT_BD_CONV_PONG = 3;
constexpr int MT_WGT_BD_CONVB_PING = 4;
constexpr int MT_WGT_BD_CONVB_PONG = 5;
constexpr int MT_WGT_BD_MISC_PING = 24;
constexpr int MT_WGT_BD_MISC_PONG = 25;
constexpr int MT_WGT_BD_LOAD_PING_L1 = 0;
constexpr int MT_WGT_BD_LOAD_PONG_L1 = 1;
constexpr int MT_WGT_BD_LOAD_PING_L2 = 26;
constexpr int MT_WGT_BD_LOAD_PONG_L2 = 27;

constexpr int MT_WGT_CHN_CONV = 0;
constexpr int MT_WGT_CHN_MISC = 1;
constexpr int MT_WGT_CHN_LOAD_L1 = 0;
constexpr int MT_WGT_CHN_LOAD_L2 = 1;

// word width
constexpr int IMG_PT_WD = 4;
constexpr int WGT_PT_WD = 4;
constexpr int OFM_PT_WD = 4;

}  // namespace alu
}  // namespace xv2dpu
}  // namespace cosim
