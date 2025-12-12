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
#include "simCfg.hpp"
#include "simUtil.hpp"

template <>
void Save<DPUVersion::XV2DPU>::save_ofm_PL() const {
  if (!SimCfgProxy::Params().CO_SIM_ON) return;

  ++SimCfgProxy::Params().SAVE_INST_NO;
  int idx_parallel = 0;
  if (2 == SimCfgProxy::Params().SAVE_OFM_PARALLEL) {
    idx_parallel = SimCfgProxy::Params().SAVE_INST_NO % 2;
  }
  std::string save_dump_path = SimCfgProxy::Params().DUMP_DIR + "/save";
  mkdir_check(save_dump_path);

  auto regSize = DDR::Instance().GetSize(reg_id_);
  auto bank = Buffer<DPU_DATA_TYPE>::Instance().GetBank(bank_id_);
  UNI_LOG_CHECK(bank != nullptr, SIM_OUT_OF_RANGE)
      << "bank_id " << bank_id_ << " out of range!" << endl;
  int bank_depth = bank->GetH();
  int save_ocp =
      ArchCfg::Instance().get_param().save_engine().channel_parallel();
  int mt_row_num = SimCfgProxy::Params().OFM_ROW_MT_NUM_PER_PARALLEL;
  int mt_data_size_per_cycle = SimCfgProxy::Params().OFM_MT_STREAM_BITWIDTH / 8;
  UNI_LOG_CHECK(mt_data_size_per_cycle * mt_row_num == save_ocp,
                SIM_OUT_OF_RANGE);

  for (auto idx_length = 0; idx_length < length_; idx_length++) {
    auto ddr_addr = ddr_addr_ + idx_length * jump_write_;
    auto bank_addr = (bank_addr_ + idx_length * jump_read_) % bank_depth;
    UNI_LOG_CHECK(ddr_addr < regSize, SIM_OUT_OF_RANGE)
        << "ddr addr " << ddr_addr << " beyond reg size " << regSize;

    // to dump
    int ocg = div_ceil(channel_, save_ocp);
    int channel_align = ocg * save_ocp;
    std::vector<DPU_DATA_TYPE> ofm_buffer_PL(channel_align, 0);
    if (const_en_) {
      std::fill_n(ofm_buffer_PL.begin(), channel_, const_value_);
    } else {
      bank->Read(bank_addr, channel_, ofm_buffer_PL.data());
    }
    for (int idx_ocg = 0; idx_ocg < ocg; ++idx_ocg) {
      bool isLast = (idx_length == length_ - 1) && (idx_ocg == ocg - 1);
      for (int idx_row_mt = 0; idx_row_mt < mt_row_num; ++idx_row_mt) {
        int buffer_addr_start =
            idx_ocg * save_ocp + idx_row_mt * mt_data_size_per_cycle;
        int buffer_addr_end = buffer_addr_start + mt_data_size_per_cycle;
        std::vector<DPU_DATA_TYPE> dumpData;
        std::copy(ofm_buffer_PL.begin() + buffer_addr_start,
                  ofm_buffer_PL.begin() + buffer_addr_end,
                  std::back_inserter(dumpData));
        auto dumpFileName = getSaveOfmDumpFileName(0, idx_row_mt, idx_parallel);
        streamDump(dumpData, SimCfgProxy::Params().DUMP_FORMATS,
                   SimCfgProxy::Params().DUMP_SEQUENCE, save_dump_path,
                   dumpFileName, isLast);
      }
    }

    // ddr dump
    for (int idx_channel = 0; idx_channel < channel_; ++idx_channel) {
      DDR::Instance().set_addr_used(reg_id_, ddr_addr + idx_channel);
    }
  }

  // token
  for (int idx_row_mt = 0; idx_row_mt < mt_row_num; ++idx_row_mt) {
    token2file_load_save_PL(SIM_OP_TYPE::SAVE_OFM, 0, idx_row_mt, idx_parallel,
                            idx_parallel);
  }
}