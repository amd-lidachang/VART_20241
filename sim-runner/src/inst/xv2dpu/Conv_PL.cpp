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

#include "Conv.hpp"
#include "ConvAddr.hpp"
#include "ConvInit.hpp"
#include "simCfg.hpp"
#include "simUtil.hpp"

// class OutHStrideWindowRegion
// record the input rows in the slide window of output row 'out_h_index'
class OutHStrideWindowRegion {
 private:
  static int stride_h;
  static int kernel_h;

 public:
  static void staticInit(int sh, int kh);

 public:
  OutHStrideWindowRegion(int out_h_index) {
    start_h_index = out_h_index * OutHStrideWindowRegion::stride_h;
    end_h_index = start_h_index + OutHStrideWindowRegion::kernel_h - 1;
  }

  OutHStrideWindowRegion(int start, int end)
      : start_h_index(start), end_h_index(end) {}

 public:
  int start_h_index;
  int end_h_index;
};

int OutHStrideWindowRegion::stride_h = 0;
int OutHStrideWindowRegion::kernel_h = 0;

void OutHStrideWindowRegion::staticInit(int sh, int kh) {
  OutHStrideWindowRegion::stride_h = sh;
  OutHStrideWindowRegion::kernel_h = kh;
}

// class InConvAddrRegion
// record the input rows corresponding to the input convaddr
class InConvAddrRegion {
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
  InConvAddrRegion(int h_num) : h_num_(h_num) {
    start_h_index = InConvAddrRegion::last_h_index + 1;
    end_h_index = InConvAddrRegion::last_h_index + h_num_;
    InConvAddrRegion::last_h_index = end_h_index;
  }

  bool isInRegion(const OutHStrideWindowRegion& outHregion) const {
    for (int h_index = outHregion.start_h_index;
         h_index <= outHregion.end_h_index; ++h_index) {
      // pad_bottom check will be ignored acorrding to
      // https://xilinx.slack.com/archives/D0345PF64HE/p1669191886413169
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

int InConvAddrRegion::pad_top = 0;
int InConvAddrRegion::pad_bottom = 0;
int InConvAddrRegion::pad_top_start = 0;
int InConvAddrRegion::pad_top_end = 0;
int InConvAddrRegion::pad_bottom_start = 0;
int InConvAddrRegion::pad_bottom_end = 0;
int InConvAddrRegion::last_h_index = -1;

void InConvAddrRegion::staticInit(int pt, int pb, int src_h_num) {
  // when pt/pb is zero, the pad_top_end/pad_bottom_end is less than
  // pad_top_start/pad_bottom_start, so the pad_top/pad_bottom check will
  // automatically fail
  InConvAddrRegion::pad_top = pt;
  InConvAddrRegion::pad_bottom = pb;
  InConvAddrRegion::pad_top_start = 0;
  InConvAddrRegion::pad_top_end = pt - 1;
  InConvAddrRegion::pad_bottom_start = src_h_num - pb;
  InConvAddrRegion::pad_bottom_end = src_h_num - 1;
  InConvAddrRegion::last_h_index = -1;
}

// class OutConvAddrRegion
// record the output rows corresponding to the output convaddr
class OutConvAddrRegion {
 private:
  static int last_h_index;

 public:
  static void staticInit();

 public:
  OutConvAddrRegion(int h_num) : h_num_(h_num) {
    start_h_index = OutConvAddrRegion::last_h_index + 1;
    end_h_index = OutConvAddrRegion::last_h_index + h_num_;
    OutConvAddrRegion::last_h_index = end_h_index;
  }

  bool isInRegion(int out_h_index) const {
    return (out_h_index >= start_h_index && out_h_index <= end_h_index);
  }

 private:
  int h_num_;
  int start_h_index;
  int end_h_index;
};

int OutConvAddrRegion::last_h_index = -1;

void OutConvAddrRegion::staticInit() { OutConvAddrRegion::last_h_index = -1; }

// class DmaChanneloutHRegion
// record the involved input rows and output rows for a certain DMA channel
class DmaChanneloutHRegion {
 public:
  DmaChanneloutHRegion(int start_out_h, int end_out_h) {
    for (int idx_out_h = start_out_h; idx_out_h <= end_out_h; ++idx_out_h) {
      outHRegions.push_back(idx_out_h);
      outHWindowRegions.push_back(OutHStrideWindowRegion{idx_out_h});
    }
  }

  bool isNeededByThisChannel_ifm(const InConvAddrRegion& inConvaddr) const {
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
    OutHStrideWindowRegion globalWindow(min_src_h_index, max_src_h_index);
    return inConvaddr.isInRegion(globalWindow);
  }

  bool isNeededByThisChannel_ofm(const OutConvAddrRegion& outConvaddr) const {
    for (int idx = 0; idx < int(outHRegions.size()); ++idx) {
      if (outConvaddr.isInRegion(outHRegions.at(idx))) {
        return true;
      }
    }
    return false;
  }

 private:
  std::vector<int> outHRegions;
  std::vector<OutHStrideWindowRegion> outHWindowRegions;
};

template <>
void Conv<DPUVersion::XV2DPU>::conv_genToken_PL() const {
  if (!SimCfgProxy::Params().CO_SIM_ON) return;

  // ensure init static firstly
  OutHStrideWindowRegion::staticInit(stride_h_, kernel_h_);
  int real_pad_top = pad_top_;
  int real_pad_bottom = kernel_h_ - pad_bottom_ - 1;
  const int inAddrNum = in_h_num_list.size();
  const int outAddrNum = out_h_num_list.size();
  int src_h_num = 0;
  for (auto h_num : in_h_num_list) src_h_num += h_num;
  InConvAddrRegion::staticInit(real_pad_top, real_pad_bottom, src_h_num);
  OutConvAddrRegion::staticInit();

  std::vector<DmaChanneloutHRegion> dmaChannelOutRegions;
  // here '4' indicates there are 4 DMA channels, 2 for each mt column
  // in fact, each mt uses 2 DMA channale for input/output, however, the mts
  // in the same column have the same input/output rows and the same DMA channel
  // IDs, so we only handle 2 channels for one mt column
  // https://confluence.xilinx.com/display/~huangy/xv2dpu+cosim+vart
  for (int idx_mt_H = 0; idx_mt_H < 4; ++idx_mt_H) {
    int start_out_h = idx_mt_H * tile_ohg_;
    int end_out_h = (idx_mt_H + 1) * tile_ohg_ - 1;
    dmaChannelOutRegions.push_back(
        DmaChanneloutHRegion(start_out_h, end_out_h));
  }

  std::vector<InConvAddrRegion> in_convaddrs;
  std::vector<OutConvAddrRegion> out_convaddrs;
  for (auto idx = 0; idx < inAddrNum; ++idx) {
    const int h_num = in_h_num_list.at(idx);
    in_convaddrs.push_back(InConvAddrRegion(h_num));
  }
  for (auto idx = 0; idx < outAddrNum; ++idx) {
    const int h_num = out_h_num_list.at(idx);
    out_convaddrs.push_back(OutConvAddrRegion(h_num));
  }

  // IFM up/dowm mt -> h0 h1 -> if split icg pad -> stride_w copy ->
  // convaddr instr -> ic_iter -> ow_iter
  for (int idx_ow_iter = 0; idx_ow_iter < ow_iter_; ++idx_ow_iter) {
    for (int idx_ic_iter = 0; idx_ic_iter < ic_iter_; ++idx_ic_iter) {
      int normal_out_addr = 1;
      int stride_w_copy = 1;
      for (int idx_sw_copy = 0; idx_sw_copy < normal_out_addr + stride_w_copy;
           ++idx_sw_copy) {
        for (int idx_convaddr = 0; idx_convaddr < int(in_convaddrs.size());
             ++idx_convaddr) {
          const InConvAddrRegion& convAddr = in_convaddrs.at(idx_convaddr);
          std::array<bool, 4> genTokenMasks;
          for (int idx_mt_H = 0; idx_mt_H < 4; ++idx_mt_H) {
            const DmaChanneloutHRegion& dmaChannelRegion =
                dmaChannelOutRegions.at(idx_mt_H);
            // check if the input convaddr offers the input rows needed by the
            // dma channel
            bool doToken = dmaChannelRegion.isNeededByThisChannel_ifm(convAddr);
            genTokenMasks.at(idx_mt_H) = doToken;
          }

          int normal_in_addr = 1;
          int icg_split = 0;
          if ((icg_offset_ != 0) && (idx_ic_iter == ic_iter_ - 1)) {
            icg_split = 1;
          }

          int genTokenTimes = normal_in_addr + icg_split;
          int mt_row_num = SimCfgProxy::Params().IFM_ROW_MT_NUM_PER_PARALLEL;

          for (int idx_genToken = 0; idx_genToken < genTokenTimes;
               ++idx_genToken) {
            for (int idx_mt_H = 0; idx_mt_H < 4; ++idx_mt_H) {
              if (!genTokenMasks.at(idx_mt_H)) continue;

              // idx_mt_H = [0, 1] for mt column 0
              // idx_mt_H = [2, 3] for mt column 1
              // here handle the mt in different rows
              for (int idx_row_mt = 0; idx_row_mt < mt_row_num; ++idx_row_mt) {
                int mtColumnIndex = idx_mt_H / 2;
                token2file_conv_PL(SIM_OP_TYPE::CONV_IFM, 0, idx_row_mt,
                                   mtColumnIndex, idx_mt_H);
              }
            }
          }
        }
      }
    }
  }

  // OFM up/down mt  ->  h0 h1 -> convaddr instr-> ow-iter
  for (int idx_ow_iter = 0; idx_ow_iter < ow_iter_; ++idx_ow_iter) {
    for (int idx_convaddr = 0; idx_convaddr < int(out_convaddrs.size());
         ++idx_convaddr) {
      const OutConvAddrRegion& convAddr = out_convaddrs.at(idx_convaddr);
      std::array<bool, 4> genTokenMasks;
      for (int idx_mt_H = 0; idx_mt_H < 4; ++idx_mt_H) {
        const DmaChanneloutHRegion& dmaChannelRegion =
            dmaChannelOutRegions.at(idx_mt_H);
        // check if the output convaddr offers the output rows needed by the
        // dma channel
        bool doToken = dmaChannelRegion.isNeededByThisChannel_ofm(convAddr);
        genTokenMasks.at(idx_mt_H) = doToken;
      }

      int mt_row_num = SimCfgProxy::Params().IFM_ROW_MT_NUM_PER_PARALLEL;
      for (int idx_mt_H = 0; idx_mt_H < 4; ++idx_mt_H) {
        if (!genTokenMasks.at(idx_mt_H)) continue;

        for (int idx_row_mt = 0; idx_row_mt < mt_row_num; ++idx_row_mt) {
          int mtColumnIndex = idx_mt_H / 2;
          token2file_conv_PL(SIM_OP_TYPE::CONV_OFM, 0, idx_row_mt,
                             mtColumnIndex, idx_mt_H);
        }
      }
    }
  }

  // for weight mts, there is only one DMA channel for each mt.
  // wgt up/down mt -> ic0 ic1 -> ic_iter -> ow_iter
  for (int idx_ow_iter = 0; idx_ow_iter < ow_iter_; ++idx_ow_iter) {
    for (int idx_ic_iter = 0; idx_ic_iter < ic_iter_; ++idx_ic_iter) {
      bool doBiasToken = false;
      if (idx_ic_iter == ic_iter_ - 1) {
        doBiasToken = true;
      }
      int mt_row_num = SimCfgProxy::Params().WGT_ROW_MT_NUM_PER_PARALLEL;
      int mt_col_num = SimCfgProxy::Params().WGT_COL_MT_NUM_PER_PARALLEL;

      for (int idx_col_num = 0; idx_col_num < mt_col_num; ++idx_col_num) {
        // wgt
        for (int idx_row_mt = 0; idx_row_mt < mt_row_num; ++idx_row_mt) {
          token2file_conv_PL(SIM_OP_TYPE::CONV_WGTS, 0, idx_row_mt, idx_col_num,
                             0);
        }
        // bias
        if (doBiasToken) {
          for (int idx_row_mt = 0; idx_row_mt < mt_row_num; ++idx_row_mt) {
            token2file_conv_PL(SIM_OP_TYPE::CONV_BIAS, 0, idx_row_mt,
                               idx_col_num, 0);
          }
        }
      }
    }
  }
}