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
#include "BaseOps.hpp"
#include "BufDes.hpp"
#include "fileio.hpp"

namespace cosim {
namespace xv2dpu {
namespace alu {

/*
 * This is for macc/avgpool/maxpool/elew_shift
 */
template <typename Dtype>
vector<vector<Dtype>> dma_macc_ifm_in_mt(const BaseOps<Dtype>& B,
                                         const AIE2Cfg& A,
                                         const vector<Dtype>& img) {
  uint32_t* uptr;
  vector<vector<Dtype>> res;
  // check if IC can be devided into groups
  assert(A.icg % A.IMG_MT_GRP == 0 && "IC cannot be seperated by MT Group");

  uint32_t img_size = B.ifm_size;
  uint32_t img_grp_size = img_size / A.IMG_MT_GRP;
  uint32_t img_mt_size = img_grp_size / A.IMG_MT_NUM;

  // ifm size for mt group
  uint32_t img_byte_size = B.ifm_size * sizeof(Dtype);
  uint32_t img_mt_grp_byte_size = img_byte_size / A.IMG_MT_GRP;
  uint32_t img_mt_grp_word_size = img_mt_grp_byte_size / 4;

  // ifm size for mt tile
  uint32_t img_mt_byte_size = img_mt_grp_byte_size / A.IMG_MT_NUM;
  uint32_t img_mt_word_size = img_mt_byte_size / 4;

  // a.1 image to memtile
  for (int i = 0; i < A.IMG_MT_GRP; i++) {
    int offset_grp = i * A.ICp;
    vector<Dtype> imgGP(img_mt_size);
    BufDes bd_gp(img_mt_grp_word_size, offset_grp,
                 {
                     1,
                     (int)(A.IMG_MT_GRP * A.ICp * sizeof(Dtype) / 4),
                     int(A.ic * sizeof(Dtype) / 4),
                 },
                 {
                     (int)(A.ICp * sizeof(Dtype) / 4),
                     (int)(A.icg / A.IMG_MT_GRP),
                 },
                 {});
    bd_gp.read_access(img.data(), imgGP.data());
    res.push_back(imgGP);

    for (int j = 0; j < A.IMG_MT_NUM; j++) {
      int offset_mt = j * img_mt_word_size;
      vector<Dtype> imgMT(img_mt_size);
      BufDes bd_mt(img_mt_grp_word_size, offset_mt,
                   {
                       1,
                   },
                   {}, {});
      bd_mt.read_access(imgGP.data(), imgMT.data());

      string fn_img_mt = getfname(0, 0, "mt", "img");
      uptr = reinterpret_cast<uint32_t*>(imgMT.data());
      appendImg2IntFile(fn_img_mt, uptr, img_mt_word_size, IMG_PT_WD);
      appendImg2IntFile(fn_img_mt + ".int", imgMT.data(), img_mt_byte_size,
                        IMG_PT_WD * 4);
    }
  }
  return res;
}

/*
 * This is for macc/avgpool
 */
template <typename Dtype>
vector<vector<Dtype>> dma_macc_wgt_in_mt(const BaseOps<Dtype>& B,
                                         const AIE2Cfg& A,
                                         const vector<Dtype>& weights,
                                         const vector<Dtype>& bias) {
  uint32_t* uptr;
  vector<vector<Dtype>> res;
  // check if IC can be devided into groups

  if (B.wgt_size) {
    int32_t kernel_h = A.kernel_h;
    int32_t kernel_w = A.kernel_w;
    int ic = A.ic;
    int ICp = A.ICp;
    if (A.EXEC_TYPE == MISC_CASE::CASE_AVGP) {
      kernel_h = 1;
      kernel_w = 1;
      ic = ICp;
    }

    uint32_t weights_mt_grp_byte_size = B.wgt_size * sizeof(Dtype);
    uint32_t weights_mt_grp_word_size = weights_mt_grp_byte_size / 4;
    uint32_t bias_mt_grp_byte_size = B.bias_size * sizeof(Dtype);
    uint32_t bias_mt_grp_word_size = bias_mt_grp_byte_size / 4;
    uint32_t wgt_mt_grp_byte_size =
        weights_mt_grp_byte_size + bias_mt_grp_byte_size;
    uint32_t wgt_mt_grp_word_size = wgt_mt_grp_byte_size / 4;

    // weight pre shuffle no pre shuffle in memtile!
    // use 2 bd to read out memtile.
    vector<Dtype> wgtMT_pre(weights_mt_grp_byte_size);
    BufDes bd_wgt_mt_pre1(
        weights_mt_grp_word_size, 0,
        {1, (int)(kernel_w * kernel_h * ICp * sizeof(Dtype) / 4), 4},
        {
            4,
            int(ic / ICp),
        },
        {});
    bd_wgt_mt_pre1.write_access(weights.data(), wgtMT_pre.data());

    vector<Dtype> bias_pre(bias_mt_grp_byte_size);
    BufDes bd_bias_pre1(bias_mt_grp_word_size, 0,
                        {
                            1,
                            4,
                        },
                        {4, ic / 4 / 4}, {});
    // BufDes bd_bias_pre2(bias_mt_grp_word_size , 4, {1, 8,}, {4, ic_/4/4 },
    // {});
    bd_bias_pre1.write_access(bias.data(), bias_pre.data());
    // bd_bias_pre2.write_access(bias.data(), bias_pre.data());

    // here wgtOrigin is origin wgt, in memtile wgt is doubled due to misc
    // design
    vector<Dtype> wgtMT(wgt_mt_grp_byte_size);
    BufDes bd_wgt_mt_write(weights_mt_grp_word_size, 0, {1}, {}, {});
    bd_wgt_mt_write.write_access(wgtMT_pre.data(), wgtMT.data());
    BufDes bd_bias_mt_write(bias_mt_grp_word_size, weights_mt_grp_word_size,
                            {1}, {}, {});
    bd_bias_mt_write.write_access(bias_pre.data(), wgtMT.data());

    res.push_back(wgtMT);

    // b.1 wgt to memtile
    string fn_wgt_mt = getfname(0, 0, "mt", "wgt");
    uptr = reinterpret_cast<uint32_t*>(wgtMT.data());
    appendImg2IntFile(fn_wgt_mt, uptr, wgt_mt_grp_word_size, WGT_PT_WD);
    appendImg2IntFile(fn_wgt_mt + ".int", wgtMT.data(), wgt_mt_grp_byte_size,
                      WGT_PT_WD * 4);
  } else {
    string fn_wgt_mt = getfname(0, 0, "mt", "wgt");
    uptr = nullptr;
    appendImg2IntFile(fn_wgt_mt, uptr, 0, WGT_PT_WD);
    appendImg2IntFile(fn_wgt_mt + ".int", uptr, 0, WGT_PT_WD * 4);
  }

  return res;
}

/*
 * This is for macc/avgpool/maxpool/all_elew
 */
template <typename Dtype>
vector<vector<Dtype>> dma_macc_ofm_in_mt(const BaseOps<Dtype>& B,
                                         const AIE2Cfg& A,
                                         const vector<Dtype>& ofm) {
  vector<vector<int8_t>> res;
  uint32_t* uptr;

  // a.1 image to memtile
  uint32_t img_mt_grp_byte_size = B.ofm_size * sizeof(Dtype);
  uint32_t img_mt_grp_word_size = img_mt_grp_byte_size / 4;
  vector<Dtype> imgMT(img_mt_grp_byte_size);
  BufDes bd_mt_write(img_mt_grp_word_size, 0, {1}, {}, {});
  bd_mt_write.write_access(ofm.data(), imgMT.data());
  string fn_img_mt = getfname(0, 0, "mt", "ofm");
  uptr = reinterpret_cast<uint32_t*>(imgMT.data());
  appendImg2IntFile(fn_img_mt, uptr, img_mt_grp_word_size, IMG_PT_WD);
  appendImg2IntFile(fn_img_mt + ".int", imgMT.data(), img_mt_grp_byte_size,
                    IMG_PT_WD * 4);

  res.push_back(imgMT);

  return res;
}

/*
 * This is for core
 *
 */

template <typename Dtype>
void dma_macc_ifm_in_core(const BaseOps<Dtype>& B, const AIE2Cfg& A,
                          const vector<vector<Dtype>>& vec_img) {
  // a.2 image read out of Memtile to core
  // here it needs 2 bds to transport from memtile to core!!!
  uint32_t* uptr = nullptr;
  int offset = 0;
  int IWiter_offset = (A.owg_tile * A.OWp) * A.stride_w;
  int IHiter_offset = (A.ohg_tile * A.OHp) * A.stride_h;
  int one_line = A.ic * A.src_w;
  int oneIterSize = B.ICin1Iter * B.IWin1Iter * B.IHin1Iter * 2;

  for (int v_idx = 0; v_idx < (int)(vec_img.size()); v_idx++) {
    vector<Dtype> imgMT = vec_img[v_idx];
    for (int c_idx = 0; c_idx < B.ICIter_NUM; c_idx++) {
      for (int h_idx = 0; h_idx < B.OHIter_NUM; h_idx++) {
        for (int w_idx = 0; w_idx < B.OWIter_NUM; w_idx++) {
          offset =
              (w_idx * IWiter_offset * A.ic + h_idx * IHiter_offset * one_line +
               c_idx * A.ICp * A.icg_tile) /
              4;

          // uint32_t img_mt_to_core_byte_size = src_h_ * (src_w_ *2 -
          // stride_w_*2) * ic_ * sizeof(Dtype);
          uint32_t img_mt2core_iter_byte_size = oneIterSize * sizeof(Dtype);
          uint32_t img_mt2core_iter_word_size = img_mt2core_iter_byte_size / 4;
          uint32_t img_mt2core_bd_byte_size = oneIterSize * sizeof(Dtype) / 2;
          uint32_t img_mt2core_bd_word_size = img_mt2core_bd_byte_size / 4;
          vector<Dtype> imgCoreDM(img_mt2core_iter_byte_size);
          vector<Dtype> imgMT_reading_stream(img_mt2core_bd_byte_size);

          BufDes bd_mt_read1(img_mt2core_bd_word_size, offset,
                             {1, (int)(A.ic * sizeof(Dtype) / 4),
                              (int)(one_line * sizeof(Dtype) / 4), 4},
                             {
                                 4,
                                 B.IWin1Iter,
                                 B.IHin1Iter,
                             },
                             {});
          bd_mt_read1.read_access(imgMT.data(), imgMT_reading_stream.data());
          BufDes bd_dm_write1(img_mt2core_bd_word_size, 0,
                              {
                                  1,
                              },
                              {}, {});
          bd_dm_write1.write_access(imgMT_reading_stream.data(),
                                    imgCoreDM.data());

          BufDes bd_mt_read2(img_mt2core_bd_word_size,
                             offset + A.ic * A.stride_w * sizeof(Dtype) / 4,
                             {1, (int)(A.ic * sizeof(Dtype) / 4),
                              (int)(one_line * sizeof(Dtype) / 4), 4},
                             {
                                 4,
                                 B.IWin1Iter,
                                 B.IHin1Iter,
                             },
                             {});
          bd_mt_read2.read_access(imgMT.data(), imgMT_reading_stream.data());
          BufDes bd_dm_write2(img_mt2core_bd_word_size,
                              img_mt2core_bd_word_size,
                              {
                                  1,
                              },
                              {}, {});
          bd_dm_write2.write_access(imgMT_reading_stream.data(),
                                    imgCoreDM.data());

          string fn_img_co = getfname(0, v_idx, "co", "img");
          uptr = reinterpret_cast<uint32_t*>(imgCoreDM.data());
          appendImg2IntFile(fn_img_co, uptr, img_mt2core_iter_word_size,
                            IMG_PT_WD);
          appendImg2IntFile(fn_img_co + ".int", imgCoreDM.data(),
                            img_mt2core_iter_byte_size, IMG_PT_WD * 4);
        }
      }
    }
  }
}

template <typename Dtype>
void dma_macc_wgt_in_core(const BaseOps<Dtype>& B, const AIE2Cfg& A,
                          const vector<vector<Dtype>>& vec_wgt) {
  // b.2 wgt to core
  // same with wgt in memtile
  //
  uint32_t* uptr = nullptr;
  // int offset = 0;
  int32_t kernel_h = A.kernel_h;
  int32_t kernel_w = A.kernel_w;
  if (A.EXEC_TYPE == MISC_CASE::CASE_AVGP) {
    kernel_h = 1;
    kernel_w = 1;
  }

  for (int v_idx = 0; v_idx < int(vec_wgt.size()); v_idx++) {
    vector<Dtype> wgtMT = vec_wgt[v_idx];
    for (int c_idx = 0; c_idx < B.ICIter_NUM; c_idx++) {
      for (int h_idx = 0; h_idx < B.OHIter_NUM; h_idx++) {
        for (int w_idx = 0; w_idx < B.OWIter_NUM; w_idx++) {
          int offset =
              (c_idx * A.ICp * kernel_w * kernel_h) * sizeof(Dtype) / 4;
          int bias_offset = B.weights_mt_grp_byte_size * sizeof(Dtype) / 4 +
                            c_idx * A.ocg_tile * A.OCp / 4;
          if (A.EXEC_TYPE == MISC_CASE::CASE_AVGP) {
            offset = 0;
            bias_offset = B.weights_mt_grp_byte_size * sizeof(Dtype) / 4;
          }

          int weight_wnd_byte_size = B.wgt_wnd_byte_size * sizeof(Dtype);
          int weight_wnd_word_size = weight_wnd_byte_size / 4;
          BufDes bd_wgt_aie_out_read(weight_wnd_word_size, offset,
                                     {
                                         1,
                                     },
                                     {}, {});
          int bias_wnd_byte_size = B.bias_wnd_byte_size * sizeof(Dtype);
          int bias_wnd_word_size = bias_wnd_byte_size / 4;
          BufDes bd_bias_aie_out_read(bias_wnd_word_size, bias_offset,
                                      {
                                          1,
                                      },
                                      {}, {});

          vector<Dtype> weight_mt2core(weight_wnd_byte_size);
          bd_wgt_aie_out_read.read_access(wgtMT.data(), weight_mt2core.data());
          vector<Dtype> bias_mt2core(bias_wnd_byte_size);
          bd_bias_aie_out_read.read_access(wgtMT.data(), bias_mt2core.data());

          vector<Dtype> wgt_mt2core(
              2 * (weight_wnd_byte_size + bias_wnd_byte_size));
          BufDes bd_weight_mt_out_write(weight_wnd_word_size, 0, {1}, {});
          BufDes bd_bias_mt_out_write(bias_wnd_word_size, weight_wnd_word_size,
                                      {1}, {});
          bd_weight_mt_out_write.write_access(weight_mt2core.data(),
                                              wgt_mt2core.data());
          bd_bias_mt_out_write.write_access(bias_mt2core.data(),
                                            wgt_mt2core.data());

          BufDes bd_weight_mt_out_write2(
              weight_wnd_word_size, (weight_wnd_word_size + bias_wnd_word_size),
              {1}, {});
          BufDes bd_bias_mt_out_write2(
              bias_wnd_word_size, weight_wnd_word_size * 2 + bias_wnd_word_size,
              {1}, {});
          bd_weight_mt_out_write2.write_access(weight_mt2core.data(),
                                               wgt_mt2core.data());
          bd_bias_mt_out_write2.write_access(bias_mt2core.data(),
                                             wgt_mt2core.data());

          string fn_wgt_co = getfname(0, v_idx, "co", "wgt");
          uptr = reinterpret_cast<uint32_t*>(wgt_mt2core.data());
          appendImg2IntFile(fn_wgt_co, uptr,
                            2 * (weight_wnd_word_size + bias_wnd_word_size),
                            WGT_PT_WD);
          appendImg2IntFile(fn_wgt_co + ".int", wgt_mt2core.data(),
                            2 * (weight_wnd_byte_size + bias_wnd_byte_size),
                            WGT_PT_WD * 4);
        }
      }
    }
  }
}

template <typename Dtype>
void dma_macc_ofm_in_core(const BaseOps<Dtype>& B, const AIE2Cfg& A,
                          const vector<vector<Dtype>>& vec_ofm) {
  // a.2 ofm read out  core
  int oneIterSize = B.OCin1Iter * B.OWin1iter * B.OHin1iter;
  int one_line = A.ic * A.dst_w;
  uint32_t* uptr;

  // int offset = 0;
  for (int v_idx = 0; v_idx < int(vec_ofm.size()); v_idx++) {
    vector<Dtype> ofmMT = vec_ofm[v_idx];
    for (int c_idx = 0; c_idx < B.ICIter_NUM; c_idx++) {
      for (int h_idx = 0; h_idx < B.OHIter_NUM; h_idx++) {
        for (int w_idx = 0; w_idx < B.OWIter_NUM; w_idx++) {
          int offset = (w_idx * B.OWin1iter * A.oc +
                        h_idx * B.OHin1iter * one_line + c_idx * B.OCin1Iter) *
                       sizeof(Dtype) / 4;

          uint32_t img_core2mt_iter_byte_size = oneIterSize * sizeof(Dtype);
          uint32_t img_core2mt_iter_word_size = img_core2mt_iter_byte_size / 4;

          // c.1 ofm from core to Memtile
          vector<Dtype> ofm_out_core(img_core2mt_iter_byte_size);
          BufDes bd_ofm_aie_out_read(
              img_core2mt_iter_word_size, offset,
              {1, (int)(A.oc * sizeof(Dtype) / 4),
               (int)(one_line * sizeof(Dtype) / 4),
               (int)(A.ICp * sizeof(Dtype) / 4)},
              {(int)(A.ICp * sizeof(Dtype) / 4), B.OWin1iter, B.OHin1iter}, {});
          //{ICp_*sizeof(Dtype)/4, B.OWin1iter, OHin1iter, icg_tile_}, {});
          bd_ofm_aie_out_read.read_access(ofmMT.data(), ofm_out_core.data());

          string fn_ofm_co = getfname(0, v_idx, "co", "ofm");
          uptr = reinterpret_cast<uint32_t*>(ofm_out_core.data());
          appendImg2IntFile(fn_ofm_co, uptr, img_core2mt_iter_word_size,
                            OFM_PT_WD);
          appendImg2IntFile(fn_ofm_co + ".int", ofm_out_core.data(),
                            img_core2mt_iter_byte_size, OFM_PT_WD * 4);
        }
      }
    }
  }
}

}  // namespace alu
}  // namespace xv2dpu
}  // namespace cosim
