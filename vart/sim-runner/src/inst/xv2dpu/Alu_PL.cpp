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

#include "Alu.hpp"
#include "AluAddr.hpp"
#include "AluInit.hpp"
#include "simCfg.hpp"
#include "simUtil.hpp"

// class SrcHWindowForSingleOutH
// record the input rows in the slide window of output row 'out_h_index'
class SrcHWindowForSingleOutH {
 private:
  static int stride_h;
  static int kernel_h;

 public:
  static void staticInit(int sh, int kh);

 public:
  SrcHWindowForSingleOutH(int out_h_index) {
    start_h_index = out_h_index * SrcHWindowForSingleOutH::stride_h;
    end_h_index = start_h_index + SrcHWindowForSingleOutH::kernel_h - 1;
  }

  SrcHWindowForSingleOutH(int start, int end)
      : start_h_index(start), end_h_index(end) {}

 public:
  int start_h_index;
  int end_h_index;
};

int SrcHWindowForSingleOutH::stride_h = 0;
int SrcHWindowForSingleOutH::kernel_h = 0;

void SrcHWindowForSingleOutH::staticInit(int sh, int kh) {
  SrcHWindowForSingleOutH::stride_h = sh;
  SrcHWindowForSingleOutH::kernel_h = kh;
}

// class SrcHRegionForInAluAddr
// record the input rows corresponding to the input convaddr
class SrcHRegionForInAluAddr {
 private:
  static int pad_top;
  static int pad_bottom;
  static int pad_top_start;
  static int pad_top_end;
  static int pad_bottom_start;
  static int pad_bottom_end;
  static int last_h_index;

 public:
  static void staticInit(int pt, int pb, int num);

 public:
  SrcHRegionForInAluAddr(int h_num) : h_num_(h_num) {
    start_h_index = SrcHRegionForInAluAddr::last_h_index + 1;
    end_h_index = SrcHRegionForInAluAddr::last_h_index + h_num_;
    SrcHRegionForInAluAddr::last_h_index = end_h_index;
  }

  bool isInRegion(const SrcHWindowForSingleOutH& outHregion) const {
    for (int h_index = outHregion.start_h_index;
         h_index <= outHregion.end_h_index; ++h_index) {
      if ((h_index >= start_h_index && h_index <= end_h_index) &&
          !(h_index >= pad_bottom_start && h_index <= pad_bottom_end) &&
          !(h_index >= pad_top_start && h_index <= pad_top_end)) {
        return true;
      }
    }
    return false;
  }

 private:
  int h_num_;
  int start_h_index;
  int end_h_index;
};

int SrcHRegionForInAluAddr::pad_top = 0;
int SrcHRegionForInAluAddr::pad_bottom = 0;
int SrcHRegionForInAluAddr::pad_top_start = 0;
int SrcHRegionForInAluAddr::pad_top_end = 0;
int SrcHRegionForInAluAddr::pad_bottom_start = 0;
int SrcHRegionForInAluAddr::pad_bottom_end = 0;
int SrcHRegionForInAluAddr::last_h_index = -1;

void SrcHRegionForInAluAddr::staticInit(int pt, int pb, int src_h_num) {
  SrcHRegionForInAluAddr::pad_top = pt;
  SrcHRegionForInAluAddr::pad_bottom = pb;
  SrcHRegionForInAluAddr::pad_top_start = 0;
  SrcHRegionForInAluAddr::pad_top_end = pt - 1;
  SrcHRegionForInAluAddr::pad_bottom_start = src_h_num - pb;
  SrcHRegionForInAluAddr::pad_bottom_end = src_h_num - 1;
  SrcHRegionForInAluAddr::last_h_index = -1;
}

// class SrcHRegionForInAluAddr_elew
// for elew, id must be considered for input
class SrcHRegionForInAluAddr_elew {
 private:
  static std::map<int, int> last_h_index_map;  // {id, value}

 public:
  static void staticInit(int id_num);

 public:
  SrcHRegionForInAluAddr_elew(int id, int h_num) : id_(id), h_num_(h_num) {
    start_h_index = SrcHRegionForInAluAddr_elew::last_h_index_map.at(id) + 1;
    end_h_index = SrcHRegionForInAluAddr_elew::last_h_index_map.at(id) + h_num_;
    SrcHRegionForInAluAddr_elew::last_h_index_map.at(id) = end_h_index;
  }

  bool isInRegion(const SrcHWindowForSingleOutH& outHregion) const {
    for (int h_index = outHregion.start_h_index;
         h_index <= outHregion.end_h_index; ++h_index) {
      if (h_index >= start_h_index && h_index <= end_h_index) {
        return true;
      }
    }
    return false;
  }

 private:
  int id_;
  int h_num_;
  int start_h_index;
  int end_h_index;
};

std::map<int, int> SrcHRegionForInAluAddr_elew::last_h_index_map;

void SrcHRegionForInAluAddr_elew::staticInit(int id_num) {
  SrcHRegionForInAluAddr_elew::last_h_index_map.clear();
  for (int id = 0; id < id_num; ++id) {
    SrcHRegionForInAluAddr_elew::last_h_index_map[id] = -1;
  }
}

// class OutHRegionForOutAluAddr
// record the output rows corresponding to the output convaddr
class OutHRegionForOutAluAddr {
 private:
  static int last_h_index;

 public:
  static void staticInit();

 public:
  OutHRegionForOutAluAddr(int h_num) : h_num_(h_num) {
    start_h_index = OutHRegionForOutAluAddr::last_h_index + 1;
    end_h_index = OutHRegionForOutAluAddr::last_h_index + h_num_;
    OutHRegionForOutAluAddr::last_h_index = end_h_index;
  }

  bool isInRegion(int out_h_index) const {
    return (out_h_index >= start_h_index && out_h_index <= end_h_index);
  }

 private:
  int h_num_;
  int start_h_index;
  int end_h_index;
};

int OutHRegionForOutAluAddr::last_h_index = -1;

void OutHRegionForOutAluAddr::staticInit() {
  OutHRegionForOutAluAddr::last_h_index = -1;
}

// class OutHRegionForDmaChannel
// record the involved input rows and output rows for a certain DMA channel
class OutHRegionForDmaChannel {
 public:
  OutHRegionForDmaChannel(int start_out_h, int end_out_h) {
    for (int idx_out_h = start_out_h; idx_out_h <= end_out_h; ++idx_out_h) {
      outHRegions.push_back(idx_out_h);
      outHWindowRegions.push_back(SrcHWindowForSingleOutH{idx_out_h});
    }
  }

  template <typename SrcHRegionForInAluAddr_t>
  bool isNeededByThisChannel_ifm(
      const SrcHRegionForInAluAddr_t& inConvaddr) const {
    // the input rows region transfered via this DMA channel is
    // [min_src_h_index, max_src_h_index]
    int min_src_h_index = std::numeric_limits<int>::max(), max_src_h_index = 0;
    for (auto& window : outHWindowRegions) {
      min_src_h_index = (window.start_h_index < min_src_h_index)
                            ? window.start_h_index
                            : min_src_h_index;
      max_src_h_index = (window.end_h_index > max_src_h_index)
                            ? window.end_h_index
                            : max_src_h_index;
    }
    SrcHWindowForSingleOutH globalWindow(min_src_h_index, max_src_h_index);
    return inConvaddr.isInRegion(globalWindow);
  }

  bool isNeededByThisChannel_ofm(
      const OutHRegionForOutAluAddr& outConvaddr) const {
    for (int idx = 0; idx < int(outHRegions.size()); ++idx) {
      if (outConvaddr.isInRegion(outHRegions.at(idx))) {
        return true;
      }
    }
    return false;
  }

 private:
  std::vector<int> outHRegions;
  std::vector<SrcHWindowForSingleOutH> outHWindowRegions;
};

template <>
void Alu<DPUVersion::XV2DPU>::alu_genToken_PL_elew() const {
  // ensure init static firstly
  int stride_h = 1;
  int kernel_h = 1;
  SrcHWindowForSingleOutH::staticInit(stride_h, kernel_h);
  int id_num = in_h_num_map_.size();
  SrcHRegionForInAluAddr_elew::staticInit(id_num);
  OutHRegionForOutAluAddr::staticInit();

  // divide ofm into two parts for h01 and h23
  // MTs:
  // C0H01(0)    C0H23(2)
  // C1H01(1)    C1H23(3)
  // there is only one channel for each input mt.
  // the same column mts have the same output rows, but with different
  // channle(C0/C1)
  // from the view of interconnection, the output is only send to H23, but the
  // DMA channel of H23 can visit H01
  std::vector<OutHRegionForDmaChannel> dmaChannelOutRegions;
  UNI_LOG_CHECK(2 == pp_alu_, VART_INVALID_VALUE);
  for (int idx_mt_col = 0; idx_mt_col < 2; ++idx_mt_col) {
    for (int idx_mt_row = 0; idx_mt_row < 2; ++idx_mt_row) {
      int start_out_h = idx_mt_col * tile_ohg_;
      int end_out_h = start_out_h + tile_ohg_ - 1;
      dmaChannelOutRegions.push_back(
          OutHRegionForDmaChannel(start_out_h, end_out_h));
    }
  }

  std::map<int, std::vector<SrcHRegionForInAluAddr_elew>> in_aluaddrs_map;
  std::vector<OutHRegionForOutAluAddr> out_aluaddrs;
  for (auto iter = in_h_num_map_.begin(); iter != in_h_num_map_.end(); ++iter) {
    const int id = iter->first;
    const auto in_h_num_list = iter->second;
    for (auto idx = 0U; idx < in_h_num_list.size(); ++idx) {
      const int h_num = in_h_num_list.at(idx);
      in_aluaddrs_map[id].push_back(SrcHRegionForInAluAddr_elew(id, h_num));
    }
  }
  const int outAddrNum = out_h_num_list_.size();
  for (auto idx = 0; idx < outAddrNum; ++idx) {
    const int h_num = out_h_num_list_.at(idx);
    out_aluaddrs.push_back(OutHRegionForOutAluAddr(h_num));
  }

  // IFM c0 c1 -> h01 h23 -> convaddr instr -> id -> ow_iter
  for (int idx_ow_iter = 0; idx_ow_iter < ow_iter_; ++idx_ow_iter) {
    int normal = 1;
    int stride_w_copy = 0;
    for (int idx_sw_copy = 0; idx_sw_copy < normal + stride_w_copy;
         ++idx_sw_copy) {
      for (int elew_id = 0; elew_id < id_num; ++elew_id) {
        auto in_aluaddrs = in_aluaddrs_map.at(elew_id);
        for (int idx_aluaddr = 0; idx_aluaddr < int(in_aluaddrs.size());
             ++idx_aluaddr) {
          const SrcHRegionForInAluAddr_elew& aluAddr =
              in_aluaddrs.at(idx_aluaddr);
          std::array<bool, 4> genTokenMasks;
          for (int idx_mt = 0; idx_mt < 4; ++idx_mt) {
            const OutHRegionForDmaChannel& dmaChannelRegion =
                dmaChannelOutRegions.at(idx_mt);
            // check if the input convaddr offers the input rows needed by the
            // dma channel, the input rows involved with the dma channel is
            // decided by its output rows
            bool doToken =
                dmaChannelRegion
                    .isNeededByThisChannel_ifm<SrcHRegionForInAluAddr_elew>(
                        aluAddr);
            genTokenMasks.at(idx_mt) = doToken;
          }

          const int mt_col_num = 2;
          const int mt_row_num = 2;
          for (int idx_col_mt = 0; idx_col_mt < mt_col_num; ++idx_col_mt) {
            for (int idx_row_mt = 0; idx_row_mt < mt_row_num; ++idx_row_mt) {
              int idx_mt = idx_col_mt * mt_row_num + idx_row_mt;
              if (genTokenMasks.at(idx_mt)) {
                token2file_alu_PL(SIM_OP_TYPE::ALU_IFM, 0, idx_row_mt,
                                  idx_col_mt);
              }
            }
          }
        }
      }
    }
  }

  // OFM c0 c1 -> h01 h23 -> convaddr instr -> ow-iter
  for (int idx_ow_iter = 0; idx_ow_iter < ow_iter_; ++idx_ow_iter) {
    for (int idx_aluaddr = 0; idx_aluaddr < int(out_aluaddrs.size());
         ++idx_aluaddr) {
      const OutHRegionForOutAluAddr& aluAddr = out_aluaddrs.at(idx_aluaddr);
      std::array<bool, 4> genTokenMasks;
      for (int idx_mt = 0; idx_mt < 4; ++idx_mt) {
        const OutHRegionForDmaChannel& dmaChannelRegion =
            dmaChannelOutRegions.at(idx_mt);
        bool doToken = dmaChannelRegion.isNeededByThisChannel_ofm(aluAddr);
        genTokenMasks.at(idx_mt) = doToken;
      }

      const int mt_col_num = 2;
      const int mt_row_num = 2;
      for (int idx_col_mt = 0; idx_col_mt < mt_col_num; ++idx_col_mt) {
        for (int idx_row_mt = 0; idx_row_mt < mt_row_num; ++idx_row_mt) {
          int idx_mt = idx_col_mt * mt_row_num + idx_row_mt;
          const int mtColumnIndex = 1;
          if (genTokenMasks.at(idx_mt)) {
            token2file_alu_PL(SIM_OP_TYPE::ALU_OFM, 0, idx_row_mt,
                              mtColumnIndex);
          }
        }
      }
    }
  }
}

template <>
void Alu<DPUVersion::XV2DPU>::alu_genToken_PL_macc_comp() const {
  // ensure init static firstly
  SrcHWindowForSingleOutH::staticInit(stride_h_, kernel_h_);
  // here using the real pad value, rather than the one converted
  int real_pad_top = pad_top_;
  int real_pad_bottom = kernel_h_ - pad_bottom_ - 1;

  auto in_h_num_list = in_h_num_map_.at(0);
  const int inAddrNum = in_h_num_list.size();
  int src_h_num = 0;
  for (auto h_num : in_h_num_list) src_h_num += h_num;
  SrcHRegionForInAluAddr::staticInit(real_pad_top, real_pad_bottom, src_h_num);
  OutHRegionForOutAluAddr::staticInit();

  // divide ofm into two parts for h01 and h23
  // 0  2
  // 1  3
  std::vector<OutHRegionForDmaChannel> dmaChannelOutRegions;
  UNI_LOG_CHECK(2 == pp_alu_, VART_INVALID_VALUE);
  for (int idx_mt_col = 0; idx_mt_col < 2; ++idx_mt_col) {
    for (int idx_mt_row = 0; idx_mt_row < 2; ++idx_mt_row) {
      int start_out_h = idx_mt_col * tile_ohg_;
      int end_out_h = start_out_h + tile_ohg_ - 1;
      dmaChannelOutRegions.push_back(
          OutHRegionForDmaChannel(start_out_h, end_out_h));
    }
  }

  std::vector<SrcHRegionForInAluAddr> in_aluaddrs;
  std::vector<OutHRegionForOutAluAddr> out_aluaddrs;
  for (auto idx = 0; idx < inAddrNum; ++idx) {
    const int h_num = in_h_num_list.at(idx);
    in_aluaddrs.push_back(SrcHRegionForInAluAddr(h_num));
  }
  const int outAddrNum = out_h_num_list_.size();
  for (auto idx = 0; idx < outAddrNum; ++idx) {
    const int h_num = out_h_num_list_.at(idx);
    out_aluaddrs.push_back(OutHRegionForOutAluAddr(h_num));
  }

  // IFM c0 c1 -> h01 h23 -> convaddr instr -> stride_w copy -> ow_iter
  for (int idx_ow_iter = 0; idx_ow_iter < ow_iter_; ++idx_ow_iter) {
    int normal = 1;
    int stride_w_copy = 1;
    for (int idx_sw_copy = 0; idx_sw_copy < normal + stride_w_copy;
         ++idx_sw_copy) {
      for (int idx_aluaddr = 0; idx_aluaddr < int(in_aluaddrs.size());
           ++idx_aluaddr) {
        const SrcHRegionForInAluAddr& aluAddr = in_aluaddrs.at(idx_aluaddr);
        std::array<bool, 4> genTokenMasks;
        for (int idx_mt = 0; idx_mt < 4; ++idx_mt) {
          const OutHRegionForDmaChannel& dmaChannelRegion =
              dmaChannelOutRegions.at(idx_mt);
          bool doToken =
              dmaChannelRegion
                  .isNeededByThisChannel_ifm<SrcHRegionForInAluAddr>(aluAddr);
          genTokenMasks.at(idx_mt) = doToken;
        }

        const int mt_col_num = 2;
        const int mt_row_num = 2;
        for (int idx_col_mt = 0; idx_col_mt < mt_col_num; ++idx_col_mt) {
          for (int idx_row_mt = 0; idx_row_mt < mt_row_num; ++idx_row_mt) {
            int idx_mt = idx_col_mt * mt_row_num + idx_row_mt;
            if (genTokenMasks.at(idx_mt)) {
              token2file_alu_PL(SIM_OP_TYPE::ALU_IFM, 0, idx_row_mt,
                                idx_col_mt);
            }
          }
        }
      }
    }
  }

  // OFM c0 c1 -> h01 h23 -> convaddr instr -> ow-iter
  for (int idx_ow_iter = 0; idx_ow_iter < ow_iter_; ++idx_ow_iter) {
    for (int idx_aluaddr = 0; idx_aluaddr < int(out_aluaddrs.size());
         ++idx_aluaddr) {
      const OutHRegionForOutAluAddr& aluAddr = out_aluaddrs.at(idx_aluaddr);
      std::array<bool, 4> genTokenMasks;
      for (int idx_mt = 0; idx_mt < 4; ++idx_mt) {
        const OutHRegionForDmaChannel& dmaChannelRegion =
            dmaChannelOutRegions.at(idx_mt);
        bool doToken = dmaChannelRegion.isNeededByThisChannel_ofm(aluAddr);
        genTokenMasks.at(idx_mt) = doToken;
      }

      const int mt_col_num = 2;
      const int mt_row_num = 2;
      for (int idx_col_mt = 0; idx_col_mt < mt_col_num; ++idx_col_mt) {
        for (int idx_row_mt = 0; idx_row_mt < mt_row_num; ++idx_row_mt) {
          int idx_mt = idx_col_mt * mt_row_num + idx_row_mt;
          const int mtColumnIndex = 1;
          if (genTokenMasks.at(idx_mt)) {
            token2file_alu_PL(SIM_OP_TYPE::ALU_OFM, 0, idx_row_mt,
                              mtColumnIndex);
          }
        }
      }
    }
  }

  if (exec_mode_ == Calc::ALU_TYPE_MACC) {
    // oc0 oc1 -> copy -> wgt,bias -> ow_iter
    const int normal_time = 1;
    const int copy_time = 1;
    for (int idx_ow_iter = 0; idx_ow_iter < ow_iter_; ++idx_ow_iter) {
      bool doBiasToken = (b_mode_ != 2);
      int mt_row_num = SimCfgProxy::Params().WGT_ROW_MT_NUM_PER_PARALLEL;
      const int mtColumnIndex = 0;

      // wgt
      // only mt OC0_IC0 and OC1_IC0 are used in weights(mt column 0)
      for (int idx_time = 0; idx_time < normal_time + copy_time; ++idx_time) {
        for (int idx_row_mt = 0; idx_row_mt < mt_row_num; ++idx_row_mt) {
          token2file_alu_PL(SIM_OP_TYPE::ALU_WGTS, 0, idx_row_mt,
                            mtColumnIndex);
        }
      }

      // bias
      if (doBiasToken) {
        for (int idx_time = 0; idx_time < normal_time + copy_time; ++idx_time) {
          for (int idx_row_mt = 0; idx_row_mt < mt_row_num; ++idx_row_mt) {
            token2file_alu_PL(SIM_OP_TYPE::ALU_BIAS, 0, idx_row_mt,
                              mtColumnIndex);
          }
        }
      }
    }
  }
}

template <>
void Alu<DPUVersion::XV2DPU>::alu_genToken_PL() const {
  if (!SimCfgProxy::Params().CO_SIM_ON) return;
  if (exec_mode_ == Calc::ALU_TYPE_MACC || exec_mode_ == Calc::ALU_TYPE_COMP) {
    alu_genToken_PL_macc_comp();
  }

  if (exec_mode_ == Calc::ALU_TYPE_ELEW_ADD ||
      exec_mode_ == Calc::ALU_TYPE_ELEW_MUL) {
    alu_genToken_PL_elew();
  }
}
