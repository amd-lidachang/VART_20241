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

#include <InstTable.hpp>
#include <Targets.hpp>
#include <set>
#include <vector>

#include "UniLog/UniLog.hpp"
#include "Util.hpp"

enum class StreamDataFormat {
  TXT,
  HEX
};

enum class StreamDataSequence {
  BIGEND,
  SMALLEND
};

enum class SIM_OP_TYPE {
  LOAD_IFM,
  LOAD_CONV_WGT,
  LOAD_CONV_BIAS,
  LOAD_MISC_WGT,
  LOAD_MISC_BIAS,
  SAVE_OFM,

  // CONV
  CONV_IFM,
  CONV_WGTS,
  CONV_BIAS,
  CONV_OFM,

  // ALU
  ALU_IFM,
  ALU_WGTS,
  ALU_BIAS,
  ALU_OFM
};

class SimCfgProxy : public Target<DPUVersion::XV2DPU> {
 public:
  static SimCfgProxy& Params() {
    static SimCfgProxy cfg;
    return cfg;
  }

public:
 const bool CO_SIM_ON{false};
 int LOAD_INST_NO;
 int SAVE_INST_NO;

 // hardware spec
 const int IFM_ROW_MT_NUM_PER_PARALLEL{2};
 const int IFM_COL_MT_NUM_PER_PARALLEL{1};
 const int WGT_ROW_MT_NUM_PER_PARALLEL{2};
 const int WGT_COL_MT_NUM_PER_PARALLEL{2};
 const int OFM_ROW_MT_NUM_PER_PARALLEL{2};
 const int OFM_COL_MT_NUM_PER_PARALLEL{1};

 const int IFM_MT_STREAM_BITWIDTH{128};
 const int WGT_MT_STREAM_BITWIDTH{128};
 const int OFM_MT_STREAM_BITWIDTH{128};

 // LOAD
 const int LOAD_IFM_PARALLEL;
 const int LOAD_WGT_PARALLEL;

 // SAVE
 const int SAVE_OFM_PARALLEL;

 const int BATCH_NUM;
 const std::vector<StreamDataFormat> DUMP_FORMATS;
 const StreamDataSequence DUMP_SEQUENCE;
 const int DUMP_LINE_DATA_NUM{16};
 const std::string DUMP_DIR;
 const std::string ALL_CTRLPKT_PATH;
 const std::string ALU_CTRLPKT_PATH;
 const std::string LOAD_CTRLPKT_PATH;
 const std::string SAVE_CTRLPKT_PATH;

private:
  SimCfgProxy();
};