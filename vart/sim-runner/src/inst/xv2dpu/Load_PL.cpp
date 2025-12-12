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

#define ENABLE_VITISAI_LIBBOOST

#include "Load.hpp"
#include "simCfg.hpp"
#include "simUtil.hpp"

template <>
void Load<DPUVersion::XV2DPU>::load_ifm_PL() {
  if (!SimCfgProxy::Params().CO_SIM_ON) return;

  static int LOAD_IFM_INST_NO = -1;
  ++LOAD_IFM_INST_NO;
  ++SimCfgProxy::Params().LOAD_INST_NO;
  int idx_parallel = 0;
  if (2 == SimCfgProxy::Params().LOAD_IFM_PARALLEL) {
    idx_parallel = LOAD_IFM_INST_NO % 2;
  }
  std::string load_dump_path = SimCfgProxy::Params().DUMP_DIR + "/load";
  mkdir_check(load_dump_path);

  UNI_LOG_CHECK(block_num_ == 1, SIM_OUT_OF_RANGE)
      << "Now bank_id = " << bank_id_
      << ". It is not a weights bank, so long load is unsupported. "
         "Block_num shoud be 1.";

  auto bank = Buffer<DPU_DATA_TYPE>::Instance().GetBank(bank_id_);
  UNI_LOG_CHECK(bank != nullptr, SIM_OUT_OF_RANGE)
      << "bank_id " << bank_id_ << " out of range!" << endl;
  int bank_width = bank->GetW();

  string save_name = debug_path_ + "inst_" + to_string(instid_) + "_" +
                     to_string(instid_classify_) + ".tick";

  // align in PL
  auto* buf = reinterpret_cast<DPU_DATA_TYPE*>(
      DDR::Instance().GetAddr(reg_id_, ddr_addr_));
  UNI_LOG_CHECK(static_cast<uint64_t>((length_ - 1) * jump_read_ + channel_) <=
                    DDR::Instance().GetSize(reg_id_),
                SIM_OUT_OF_RANGE);

  // set pad_mask for channel augment
  int ll = length_ * channel_;
  vector<bool> pad_mask(ll, true);
  gen_PadMask(pad_mask, bank_width);

  int load_icp =
      ArchCfg::Instance().get_param().load_engine().channel_parallel();
  auto icg = div_ceil(channel_, load_icp);

  int mt_row_num = SimCfgProxy::Params().IFM_ROW_MT_NUM_PER_PARALLEL;
  int mt_data_size_per_cycle = SimCfgProxy::Params().IFM_MT_STREAM_BITWIDTH / 8;
  UNI_LOG_CHECK(mt_data_size_per_cycle * mt_row_num == load_icp,
                SIM_OUT_OF_RANGE);
  for (int idx_length = 0; idx_length < length_; ++idx_length) {
    int src_addr_length = idx_length * jump_read_;
    // set dirty data = 0
    int channel_align = load_icp * icg;
    std::vector<char> ifm_buffer_PL(channel_align, 0);
    for (auto idx_channel = 0; idx_channel < channel_; idx_channel++) {
      auto pad_mask_idx = idx_length * channel_ + idx_channel;
      if (pad_mask[pad_mask_idx] == false) {
        ifm_buffer_PL[idx_channel] = 0;
        continue;
      } else {
        int ddr_addr = src_addr_length + idx_channel;
        ifm_buffer_PL[idx_channel] = buf[ddr_addr];
      }
    }

    // ddr dump
    for (int idx_channel = 0; idx_channel < channel_; ++idx_channel) {
      DDR::Instance().set_addr_used(reg_id_,
                                    ddr_addr_ + src_addr_length + idx_channel);
    }

    // to dump
    for (int idx_icg = 0; idx_icg < icg; ++idx_icg) {
      bool isLast = (idx_length == length_ - 1) && (idx_icg == icg - 1);

      int buffer_addr_icg = idx_icg * load_icp;
      for (int idx_row_mt = 0; idx_row_mt < mt_row_num; ++idx_row_mt) {
        int buffer_addr_start =
            buffer_addr_icg + idx_row_mt * mt_data_size_per_cycle;
        int buffer_addr_end = buffer_addr_start + mt_data_size_per_cycle;
        std::vector<char> dumpData;
        std::copy(ifm_buffer_PL.begin() + buffer_addr_start,
                  ifm_buffer_PL.begin() + buffer_addr_end,
                  std::back_inserter(dumpData));
        auto dumpFileName = getLoadIfmDumpFileName(0, idx_row_mt, idx_parallel);
        streamDump(dumpData, SimCfgProxy::Params().DUMP_FORMATS,
                   SimCfgProxy::Params().DUMP_SEQUENCE, load_dump_path,
                   dumpFileName, isLast);
      }
    }
  }

  // token
  for (int idx_row_mt = 0; idx_row_mt < mt_row_num; ++idx_row_mt) {
    token2file_load_save_PL(SIM_OP_TYPE::LOAD_IFM, 0, idx_row_mt, idx_parallel,
                            idx_parallel);
  }
}

template <>
void Load<DPUVersion::XV2DPU>::load_conv_wgts_PL() const {
  if (!SimCfgProxy::Params().CO_SIM_ON) return;

  ++SimCfgProxy::Params().LOAD_INST_NO;
  std::string load_dump_path = SimCfgProxy::Params().DUMP_DIR + "/load";
  mkdir_check(load_dump_path);

  int conv_ocp =
      ArchCfg::Instance().get_param().conv_engine().output_channel_parallel();
  int conv_icp =
      ArchCfg::Instance().get_param().conv_engine().input_channel_parallel();
  auto wgt_bank_group_name =
      ArchCfg::Instance().get_param().conv_engine().weight_bank();
  int bank_width =
      Buffer<DPU_DATA_TYPE>::Instance().GetBankW(wgt_bank_group_name);
  int wgt_bank_num =
      Buffer<DPU_DATA_TYPE>::Instance().GetBankNum(wgt_bank_group_name);
  UNI_LOG_CHECK(conv_icp == 32, SIM_OUT_OF_RANGE)
      << "icp is 32. Now icp = " << conv_icp;
  UNI_LOG_CHECK(conv_ocp == 32, SIM_OUT_OF_RANGE)
      << "ocp is 32. Now ocp = " << conv_ocp;
  UNI_LOG_CHECK(bank_width == 16, SIM_OUT_OF_RANGE)
      << "bank_width is 16. Now bank_width = " << bank_width;
  UNI_LOG_CHECK(wgt_bank_num == 4, SIM_OUT_OF_RANGE)
      << "wgt_bank_num is 4. Now wgt_bank_num = " << wgt_bank_num;

  string save_name = debug_path_ + "inst_" + to_string(instid_) + "_" +
                     to_string(instid_classify_) + ".tick";
  auto* buf = reinterpret_cast<DPU_DATA_TYPE*>(
      DDR::Instance().GetAddr(reg_id_, ddr_addr_));

  int icg = div_ceil(channel_, conv_icp);
  int ocg = div_ceil(block_num_, conv_ocp);
  int ddr_block_size = conv_icp * 1 * 1 * conv_ocp;
  int ddr_block_num = length_ * icg * ocg;
  int fetch_cycles_per_block = ddr_block_size / bank_width / wgt_bank_num;
  UNI_LOG_CHECK_EQ(ddr_block_size % (wgt_bank_num * bank_width), 0,
                   SIM_ALIGN_ERROR)
      << UNI_LOG_VALUES(ddr_block_size, wgt_bank_num, bank_width);

  int idx_parallel = 0;
  for (int idx_ocg = 0; idx_ocg < ocg; ++idx_ocg) {
    int first_parallel_block_num = -1;
    if (2 == SimCfgProxy::Params().LOAD_WGT_PARALLEL) {
      int first_parallel_ocg = ocg / 2 + (ocg % 2);
      idx_parallel = (idx_ocg < first_parallel_ocg) ? 0 : 1;
      first_parallel_block_num = length_ * icg * first_parallel_ocg;
    }

    for (int idx_icg = 0; idx_icg < icg; ++idx_icg) {
      for (int idx_length = 0; idx_length < length_; ++idx_length) {
        for (int idx_cycle = 0; idx_cycle < fetch_cycles_per_block;
             ++idx_cycle) {
          for (int idx_bank = 0; idx_bank < wgt_bank_num; ++idx_bank) {
            int idx_block_num =
                idx_ocg * icg * length_ + idx_icg * length_ + idx_length;
            auto ddr_addr_offset = idx_block_num * ddr_block_size +
                                   idx_cycle * bank_width * wgt_bank_num +
                                   idx_bank * bank_width;
            vector<DPU_DATA_TYPE> wgt_buffer(bank_width, 0);
            memcpy(wgt_buffer.data(), buf + ddr_addr_offset, bank_width);

            // ddr dump
            for (int idx_channel = 0; idx_channel < bank_width; ++idx_channel) {
              DDR::Instance().set_addr_used(
                  reg_id_, ddr_addr_ + ddr_addr_offset + idx_channel);
            }

            // to dump
            bool isLast = ((idx_block_num == ddr_block_num - 1) ||
                           (idx_block_num == first_parallel_block_num - 1)) &&
                          (idx_cycle == fetch_cycles_per_block - 1);
            const int mtRowIndex =
                idx_bank % SimCfgProxy::Params().WGT_ROW_MT_NUM_PER_PARALLEL;
            const int mtColumnIndex =
                idx_bank / SimCfgProxy::Params().WGT_COL_MT_NUM_PER_PARALLEL;
            auto dumpFileName = getLoadConvWgtDumpFileName(
                0, mtRowIndex, mtColumnIndex, idx_parallel);
            streamDump(wgt_buffer, SimCfgProxy::Params().DUMP_FORMATS,
                       SimCfgProxy::Params().DUMP_SEQUENCE, load_dump_path,
                       dumpFileName, isLast);
          }
        }
      }
    }
  }

  // token
  const int mt_row_num = SimCfgProxy::Params().WGT_ROW_MT_NUM_PER_PARALLEL;
  const int mt_col_num = SimCfgProxy::Params().WGT_COL_MT_NUM_PER_PARALLEL;
  int real_done_parallel =
      ((2 == SimCfgProxy::Params().LOAD_WGT_PARALLEL) && ocg > 1) ? 2 : 1;
  for (int idx_col_mt = 0; idx_col_mt < mt_col_num; ++idx_col_mt) {
    for (int idx_row_mt = 0; idx_row_mt < mt_row_num; ++idx_row_mt) {
      for (int idx_done_parallel = 0; idx_done_parallel < real_done_parallel;
           ++idx_done_parallel) {
        token2file_load_save_PL(SIM_OP_TYPE::LOAD_CONV_WGT, 0, idx_row_mt,
                                idx_col_mt, idx_done_parallel);
      }
    }
  }
}

template <>
void Load<DPUVersion::XV2DPU>::load_conv_bias_PL() const {
  if (!SimCfgProxy::Params().CO_SIM_ON) return;

  ++SimCfgProxy::Params().LOAD_INST_NO;
  std::string load_dump_path = SimCfgProxy::Params().DUMP_DIR + "/load";
  mkdir_check(load_dump_path);

  auto conv_ocp =
      ArchCfg::Instance().get_param().conv_engine().output_channel_parallel();
  auto bias_bank_group_name =
      ArchCfg::Instance().get_param().conv_engine().bias_bank();
  auto bank_width =
      Buffer<DPU_DATA_TYPE>::Instance().GetBankW(bias_bank_group_name);
  auto bias_bank_num =
      Buffer<DPU_DATA_TYPE>::Instance().GetBankNum(bias_bank_group_name);
  UNI_LOG_CHECK(conv_ocp == 32, SIM_OUT_OF_RANGE)
      << "ocp is 32. Now ocp = " << conv_ocp;
  UNI_LOG_CHECK(bank_width == 16, SIM_OUT_OF_RANGE)
      << "bank_width is 16. Now bank_width = " << bank_width;
  UNI_LOG_CHECK(bias_bank_num == 4, SIM_OUT_OF_RANGE)
      << "bias_bank_num is 4. Now bias_bank_num = " << bias_bank_num;
  UNI_LOG_CHECK(block_num_ == 1, SIM_OUT_OF_RANGE)
      << "block_num is 1. Now block_num = " << block_num_;

  auto* buf = reinterpret_cast<DPU_DATA_TYPE*>(
      DDR::Instance().GetAddr(reg_id_, ddr_addr_));
  string save_name = debug_path_ + "inst_" + to_string(instid_) + "_" +
                     to_string(instid_classify_) + ".tick";

  int output_channel = channel_;
  int ocg = div_ceil(output_channel, conv_ocp);
  int ddr_block_size = 256;
  int fetch_cycles_per_block = ddr_block_size / bank_width / bias_bank_num;
  UNI_LOG_CHECK_EQ(ddr_block_size % (bias_bank_num * bank_width), 0,
                   SIM_ALIGN_ERROR)
      << UNI_LOG_VALUES(ddr_block_size, bias_bank_num, bank_width);

  int idx_parallel = 0;
  for (int idx_ocg = 0; idx_ocg < ocg; ++idx_ocg) {
    int first_parallel_block_num = -1;
    if (2 == SimCfgProxy::Params().LOAD_WGT_PARALLEL) {
      int first_parallel_ocg = ocg / 2 + (ocg % 2);
      idx_parallel = (idx_ocg < first_parallel_ocg) ? 0 : 1;
      first_parallel_block_num = first_parallel_ocg;
    }
    for (int idx_cycle = 0; idx_cycle < fetch_cycles_per_block; ++idx_cycle) {
      for (int idx_bank = 0; idx_bank < bias_bank_num; ++idx_bank) {
        vector<DPU_DATA_TYPE> bias_buffer(bank_width, 0);
        auto ddr_addr_offset = idx_ocg * ddr_block_size +
                               idx_cycle * bank_width * bias_bank_num +
                               idx_bank * bank_width;
        memcpy(bias_buffer.data(), buf + ddr_addr_offset, bank_width);

        // ddr dump
        for (int idx_channel = 0; idx_channel < bank_width; ++idx_channel) {
          DDR::Instance().set_addr_used(
              reg_id_, ddr_addr_ + ddr_addr_offset + idx_channel);
        }

        // to dump
        bool isLast = ((idx_ocg == first_parallel_block_num - 1) ||
                       (idx_ocg == ocg - 1)) &&
                      (idx_cycle == fetch_cycles_per_block - 1);
        const int mtRowIndex =
            idx_bank % SimCfgProxy::Params().WGT_ROW_MT_NUM_PER_PARALLEL;
        const int mtColumnIndex =
            idx_bank / SimCfgProxy::Params().WGT_COL_MT_NUM_PER_PARALLEL;
        auto dumpFileName = getLoadConvWgtDumpFileName(
            0, mtRowIndex, mtColumnIndex, idx_parallel);
        streamDump(bias_buffer, SimCfgProxy::Params().DUMP_FORMATS,
                   SimCfgProxy::Params().DUMP_SEQUENCE, load_dump_path,
                   dumpFileName, isLast);
      }
    }
  }

  // token
  const int mt_row_num = SimCfgProxy::Params().WGT_ROW_MT_NUM_PER_PARALLEL;
  const int mt_col_num = SimCfgProxy::Params().WGT_COL_MT_NUM_PER_PARALLEL;
  int real_done_parallel =
      ((2 == SimCfgProxy::Params().LOAD_WGT_PARALLEL) && ocg > 1) ? 2 : 1;
  for (int idx_col_mt = 0; idx_col_mt < mt_col_num; ++idx_col_mt) {
    for (int idx_row_mt = 0; idx_row_mt < mt_row_num; ++idx_row_mt) {
      for (int idx_done_parallel = 0; idx_done_parallel < real_done_parallel;
           ++idx_done_parallel) {
        token2file_load_save_PL(SIM_OP_TYPE::LOAD_CONV_BIAS, 0, idx_row_mt,
                                idx_col_mt, idx_done_parallel);
      }
    }
  }
}

template <>
void Load<DPUVersion::XV2DPU>::load_misc_wgts_PL() const {
  if (!SimCfgProxy::Params().CO_SIM_ON) return;

  ++SimCfgProxy::Params().LOAD_INST_NO;
  std::string load_dump_path = SimCfgProxy::Params().DUMP_DIR + "/load";
  mkdir_check(load_dump_path);

  auto conv_ocp =
      ArchCfg::Instance().get_param().alu_engine().channel_parallel();
  auto wgt_bank_group_name =
      ArchCfg::Instance().get_param().alu_engine().weight_bank();
  auto bank_width =
      Buffer<DPU_DATA_TYPE>::Instance().GetBankW(wgt_bank_group_name);
  auto wgt_bank_num =
      Buffer<DPU_DATA_TYPE>::Instance().GetBankNum(wgt_bank_group_name);
  UNI_LOG_CHECK(conv_ocp == 32, SIM_OUT_OF_RANGE)
      << "ocp is 32. Now ocp = " << conv_ocp;
  UNI_LOG_CHECK(bank_width == 16, SIM_OUT_OF_RANGE)
      << "bank_width is 16. Now bank_width = " << bank_width;
  UNI_LOG_CHECK(wgt_bank_num == 4, SIM_OUT_OF_RANGE)
      << "wgt_bank_num is 4. Now wgt_bank_num = " << wgt_bank_num;
  UNI_LOG_CHECK(block_num_ == 1, SIM_OUT_OF_RANGE)
      << "block_num is 1. Now block_num = " << block_num_;

  auto* buf = reinterpret_cast<DPU_DATA_TYPE*>(
      DDR::Instance().GetAddr(reg_id_, ddr_addr_));
  string save_name = debug_path_ + "inst_" + to_string(instid_) + "_" +
                     to_string(instid_classify_) + ".tick";

  int output_channel = channel_;
  int ocg = div_ceil(output_channel, conv_ocp);
  int ddr_block_size = 64;

  int idx_parallel = 0;
  for (int idx_ocg = 0; idx_ocg < ocg; ++idx_ocg) {
    int first_parallel_block_num = -1;
    if (2 == SimCfgProxy::Params().LOAD_WGT_PARALLEL) {
      int first_parallel_ocg = ocg / 2 + (ocg % 2);
      idx_parallel = (idx_ocg < first_parallel_ocg) ? 0 : 1;
      first_parallel_block_num = first_parallel_ocg;
    }
    for (int idx_length = 0; idx_length < length_; ++idx_length) {
      for (int idx_bank = 0; idx_bank < wgt_bank_num; ++idx_bank) {
        std::vector<DPU_DATA_TYPE> wgt_buffer(bank_width, 0);
        int idx_block_num = idx_ocg * length_ + idx_length;
        auto ddr_addr_offset =
            idx_block_num * ddr_block_size + idx_bank * bank_width;
        memcpy(wgt_buffer.data(), buf + ddr_addr_offset, bank_width);

        // ddr dump
        for (int idx_channel = 0; idx_channel < bank_width; ++idx_channel) {
          DDR::Instance().set_addr_used(
              reg_id_, ddr_addr_ + ddr_addr_offset + idx_channel);
        }

        // to dump
        bool isLast = ((idx_ocg == first_parallel_block_num - 1) ||
                       (idx_ocg == ocg - 1)) &&
                      (idx_length == length_ - 1);
        const int mtRowIndex =
            idx_bank % SimCfgProxy::Params().WGT_ROW_MT_NUM_PER_PARALLEL;
        const int mtColumnIndex =
            idx_bank / SimCfgProxy::Params().WGT_COL_MT_NUM_PER_PARALLEL;
        auto dumpFileName = getLoadConvWgtDumpFileName(
            0, mtRowIndex, mtColumnIndex, idx_parallel);
        streamDump(wgt_buffer, SimCfgProxy::Params().DUMP_FORMATS,
                   SimCfgProxy::Params().DUMP_SEQUENCE, load_dump_path,
                   dumpFileName, isLast);
      }
    }
  }

  // token
  const int mt_row_num = SimCfgProxy::Params().WGT_ROW_MT_NUM_PER_PARALLEL;
  const int mt_col_num = SimCfgProxy::Params().WGT_COL_MT_NUM_PER_PARALLEL;
  int real_done_parallel =
      ((2 == SimCfgProxy::Params().LOAD_WGT_PARALLEL) && ocg > 1) ? 2 : 1;
  for (int idx_col_mt = 0; idx_col_mt < mt_col_num; ++idx_col_mt) {
    for (int idx_row_mt = 0; idx_row_mt < mt_row_num; ++idx_row_mt) {
      for (int idx_done_parallel = 0; idx_done_parallel < real_done_parallel;
           ++idx_done_parallel) {
        token2file_load_save_PL(SIM_OP_TYPE::LOAD_MISC_WGT, 0, idx_row_mt,
                                idx_col_mt, idx_done_parallel);
      }
    }
  }
}

template <>
void Load<DPUVersion::XV2DPU>::load_misc_bias_PL() const {
  if (!SimCfgProxy::Params().CO_SIM_ON) return;

  ++SimCfgProxy::Params().LOAD_INST_NO;
  int idx_parallel = 0;
  std::string load_dump_path = SimCfgProxy::Params().DUMP_DIR + "/load";
  mkdir_check(load_dump_path);

  auto conv_ocp =
      ArchCfg::Instance().get_param().alu_engine().channel_parallel();
  auto bias_bank_group_name =
      ArchCfg::Instance().get_param().conv_engine().bias_bank();
  auto bank_width =
      Buffer<DPU_DATA_TYPE>::Instance().GetBankW(bias_bank_group_name);
  auto bias_bank_num =
      Buffer<DPU_DATA_TYPE>::Instance().GetBankNum(bias_bank_group_name);
  UNI_LOG_CHECK(conv_ocp == 32, SIM_OUT_OF_RANGE)
      << "ocp is 32. Now ocp = " << conv_ocp;
  UNI_LOG_CHECK(bank_width == 16, SIM_OUT_OF_RANGE)
      << "bank_width is 16. Now bank_width = " << bank_width;
  UNI_LOG_CHECK(bias_bank_num == 4, SIM_OUT_OF_RANGE)
      << "bias_bank_num is 4. Now bias_bank_num = " << bias_bank_num;
  UNI_LOG_CHECK(block_num_ == 1, SIM_OUT_OF_RANGE)
      << "block_num is 1. Now block_num = " << block_num_;
  UNI_LOG_CHECK(length_ == 1, SIM_OUT_OF_RANGE)
      << "length is 1. Now length = " << length_;

  auto* buf = reinterpret_cast<DPU_DATA_TYPE*>(
      DDR::Instance().GetAddr(reg_id_, ddr_addr_));
  string save_name = debug_path_ + "inst_" + to_string(instid_) + "_" +
                     to_string(instid_classify_) + ".tick";

  int output_channel = channel_;
  int ocg = div_ceil(output_channel, conv_ocp);
  int ddr_block_size = 64;

  for (int idx_ocg = 0; idx_ocg < ocg; ++idx_ocg) {
    int first_parallel_block_num = -1;
    if (2 == SimCfgProxy::Params().LOAD_WGT_PARALLEL) {
      int first_parallel_ocg = ocg / 2 + (ocg % 2);
      idx_parallel = (idx_ocg < first_parallel_ocg) ? 0 : 1;
      first_parallel_block_num = first_parallel_ocg;
    }
    for (int idx_bank = 0; idx_bank < bias_bank_num; ++idx_bank) {
      vector<DPU_DATA_TYPE> bias_buffer(bank_width, 0);
      auto ddr_addr_offset = idx_ocg * ddr_block_size + idx_bank * bank_width;
      memcpy(bias_buffer.data(), buf + ddr_addr_offset, bank_width);

      // ddr dump
      for (int idx_channel = 0; idx_channel < bank_width; ++idx_channel) {
        DDR::Instance().set_addr_used(
            reg_id_, ddr_addr_ + ddr_addr_offset + idx_channel);
      }

      // to dump
      bool isLast =
          (idx_ocg == first_parallel_block_num - 1) || (idx_ocg == ocg - 1);
      const int mtRowIndex =
          idx_bank % SimCfgProxy::Params().WGT_ROW_MT_NUM_PER_PARALLEL;
      const int mtColumnIndex =
          idx_bank / SimCfgProxy::Params().WGT_COL_MT_NUM_PER_PARALLEL;
      auto dumpFileName = getLoadConvWgtDumpFileName(
          0, mtRowIndex, mtColumnIndex, idx_parallel);
      streamDump(bias_buffer, SimCfgProxy::Params().DUMP_FORMATS,
                 SimCfgProxy::Params().DUMP_SEQUENCE, load_dump_path,
                 dumpFileName, isLast);
    }
  }

  // token
  const int mt_row_num = SimCfgProxy::Params().WGT_ROW_MT_NUM_PER_PARALLEL;
  const int mt_col_num = SimCfgProxy::Params().WGT_COL_MT_NUM_PER_PARALLEL;
  int real_done_parallel =
      ((2 == SimCfgProxy::Params().LOAD_WGT_PARALLEL) && ocg > 1) ? 2 : 1;
  for (int idx_col_mt = 0; idx_col_mt < mt_col_num; ++idx_col_mt) {
    for (int idx_row_mt = 0; idx_row_mt < mt_row_num; ++idx_row_mt) {
      for (int idx_done_parallel = 0; idx_done_parallel < real_done_parallel;
           ++idx_done_parallel) {
        token2file_load_save_PL(SIM_OP_TYPE::LOAD_MISC_BIAS, 0, idx_row_mt,
                                idx_col_mt, idx_done_parallel);
      }
    }
  }
}