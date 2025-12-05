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

#include "simCfg.hpp"
#include "simUtil.hpp"
#include "SimCfg.hpp"

SimCfgProxy::SimCfgProxy()
    : CO_SIM_ON(SimCfg::Instance().get_co_sim_on()),
      LOAD_INST_NO(-1),
      SAVE_INST_NO(-1),
      LOAD_IFM_PARALLEL(SimCfg::Instance().get_load_img_parallel()),
      LOAD_WGT_PARALLEL(SimCfg::Instance().get_load_wgt_parallel()),
      SAVE_OFM_PARALLEL(SimCfg::Instance().get_save_parallel()),
      BATCH_NUM(1),
      DUMP_FORMATS({StreamDataFormat::TXT, StreamDataFormat::HEX}),
      DUMP_SEQUENCE(StreamDataSequence::SMALLEND),
      DUMP_DIR(data_path()),
      ALL_CTRLPKT_PATH(data_path("control_golden")),
      ALU_CTRLPKT_PATH(data_path("control_golden/alu")),
      LOAD_CTRLPKT_PATH(data_path("control_golden/load")),
      SAVE_CTRLPKT_PATH(data_path("control_golden/save")) {
  if (CO_SIM_ON) {
    UNI_LOG_CHECK(LOAD_IFM_PARALLEL >= SAVE_OFM_PARALLEL, SIM_CFG_FMT_NO_MATCH);
    DelFileOrDir_check(DUMP_DIR);
    mkdir_check(DUMP_DIR);

    DelFileOrDir_check(ALL_CTRLPKT_PATH);
    mkdir_check(ALL_CTRLPKT_PATH);

    DelFileOrDir_check(ALU_CTRLPKT_PATH);
    mkdir_check(ALU_CTRLPKT_PATH);

    DelFileOrDir_check(LOAD_CTRLPKT_PATH);
    mkdir_check(LOAD_CTRLPKT_PATH);

    DelFileOrDir_check(SAVE_CTRLPKT_PATH);
    mkdir_check(SAVE_CTRLPKT_PATH);
  }
}