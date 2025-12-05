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
#include "AIE2Cfg.hpp"
#include "CoreCtrl.hpp"
#include "Util.hpp"
#include "fileio.hpp"
#include "math_op.hpp"

namespace cosim {
namespace xv2dpu {
namespace alu {

/*
 * Base Operations
 * func name method: (ifm/wgt/ofm)_(core/mt/pt)_func()
 */
template <typename T>
class BaseOps {
 public:
  AIE2Cfg AIE;

  int layer_param[LAYER_PARAM_WORD_SIZE];

  // total size
  int ifm_size;
  int wgt_size;
  int bias_size;
  int ofm_size;

  // iters size
  int IWin1Iter;
  int IHin1Iter;
  int ICin1Iter;
  int OWin1iter;
  int OHin1iter;
  int OCin1Iter;
  int OW_offset;
  int OCG_offset;

  // iter count
  int OHIter_NUM;
  int OWIter_NUM;
  int OCIter_NUM;
  int ICIter_NUM;
  int OCP_MT;

  // window size
  int32_t img_wnd_byte_size;
  int32_t wgt_wnd_byte_size;
  int32_t bias_wnd_byte_size;
  int32_t ofm_wnd_byte_size;

  // memtile group size
  uint32_t ofm_mt_grp_byte_size;
  uint32_t img_mt_grp_byte_size;
  uint32_t weights_mt_grp_byte_size;
  uint32_t bias_mt_grp_byte_size;

 public:
  BaseOps(AIE2Cfg AIE_) : AIE(AIE_) {
    this->ifm_size = AIE.ic * AIE.oc * AIE.src_w * AIE.src_h;
    this->wgt_size = 0;
    this->bias_size = 0;
    this->ofm_size = AIE.ic * AIE.oc * AIE.src_w * AIE.src_h;

    this->IWin1Iter =
        (AIE.owg_tile * AIE.OWp - 2) * AIE.stride_w + AIE.kernel_w;
    this->IHin1Iter =
        (AIE.ohg_tile * AIE.OHp - 1) * AIE.stride_h + AIE.kernel_h;
    this->ICin1Iter = AIE.icg_tile * AIE.ICp;
    this->OCin1Iter = AIE.ocg_tile * AIE.OCp;
    this->OW_offset =
        AIE.dst_w % 8 == 0
            ? 0
            : (8 - AIE.dst_w % 8);  // dst_w should not larger than 8
    this->OCG_offset = 0;
    this->OWin1iter = AIE.OWp * AIE.owg_tile;
    this->OHin1iter = AIE.OHp * AIE.ohg_tile;

    this->img_wnd_byte_size = ICin1Iter * IWin1Iter * IHin1Iter * 2;
    this->wgt_wnd_byte_size = ICin1Iter * AIE.kernel_w * AIE.kernel_h;
    this->bias_wnd_byte_size = ICin1Iter;
    this->ofm_wnd_byte_size = ICin1Iter * OWin1iter * OHin1iter;

    this->img_mt_grp_byte_size = AIE.src_h * AIE.src_w * AIE.ic;
    this->ofm_mt_grp_byte_size =
        AIE.dst_h * AIE.dst_w * AIE.OCp * up_align(AIE.ocg, AIE.IMG_MT_GRP);
    this->weights_mt_grp_byte_size =
        AIE.kernel_h * AIE.kernel_w * AIE.ICp * AIE.icg;
    this->bias_mt_grp_byte_size = AIE.OCp * AIE.ocg;

    cout << AIE.OCp << " " << AIE.WGT_MT_GRP << endl;
    OCP_MT = AIE.OCp * AIE.WGT_MT_GRP;
    OHIter_NUM = up_align(AIE.dst_h, AIE.ohg_tile * AIE.OHP);
    OWIter_NUM = up_align(AIE.dst_w, AIE.owg_tile * AIE.OWP);
    OCIter_NUM = (up_align(AIE.oc, OCP_MT) / AIE.ocg_tile);
    ICIter_NUM = up_align(AIE.ic, AIE.icg_tile * AIE.ICp);

    // cout << "img wnd size: " << img_wnd_byte_size << endl;
    // cout << "wgt_wnd size: " << wgt_wnd_byte_size << endl;
    // cout << "bias_wnd size: " << bias_wnd_byte_size << endl;
    // cout << "ofm_wnd size: " << ofm_wnd_byte_size << endl;
    // cout << "ow offset: " << OW_offset << endl;
  }

  // BaseOps(const BaseOps&){};

  virtual void math_op(const vector<T>& img, const vector<T>& wgt,
                       const vector<T>& bias, vector<T>& rlt) = 0;

  // layer parameters for core.
  // params are set in this->layer_param[]
  virtual void set_layer_params() { return; };

  // ifm in memtile arrangement
  virtual vector<vector<T>> ifm_mt_arrangement(const vector<T>& ifm);

  // wgt and bias in memtile arrangement
  virtual vector<vector<T>> wgt_mt_arrangement(const vector<T>& wgt,
                                               const vector<T>& bias);

  // ofm in memtile arrangement
  virtual vector<vector<T>> ofm_mt_arrangement(const vector<T>& ofm);

  // ifm window size and BD params for core
  virtual string ifm_core_set_window() { return 0; };

  // wgt window size and BD params for core
  virtual string wgt_core_set_window() { return 0; };

  // ofm window size and BD params for core
  virtual string ofm_core_set_window() { return 0; };

  virtual string set_mt2co_ifm(
      int idx_mt_grp, int idx_mt_num, uint32_t img_bytes_size,
      uint32_t mt_img_ifm_addr, int count, int pad_left, int pad_top,
      int pad_right, int pad_bottom, int IWin1Iter, int IHin1Iter, int src_h,
      int src_w, int stride_w, int ICG, int ICG_tile, int Lock_Acq_ID,
      int Lock_Rel_ID, bool BD_casc_fir, bool BD_casc_last, int isPing) {
    return 0;
  };

  // ifm control pkt for memtile
  void ifm_mt_to_core_ctrl_pkt();
  void ifm_pl_to_mt_ctrl_pkt();

  // ofm control pkt for memtile
  void ofm_core_to_mt_ctrl_pkt();
  void ofm_mt_to_pl_ctrl_pkt();

  // wgt control pkt for memtile
  void wgt_mt_to_core_ctrl_pkt();
  void wgt_pl_to_mt_ctrl_pkt();

  void generate_core_ctrl_pkt();
  void generate_mt_ctrl_pkts();

  template <typename Dtype>
  friend vector<vector<Dtype>> dma_macc_ifm_in_mt(const BaseOps<Dtype>& B,
                                                  const AIE2Cfg& A,
                                                  const vector<Dtype>& img);
  template <typename Dtype>
  friend vector<vector<Dtype>> dma_macc_wgt_in_mt(const BaseOps<Dtype>& B,
                                                  const AIE2Cfg& A,
                                                  const vector<Dtype>& weights,
                                                  const vector<Dtype>& bias);
  template <typename Dtype>
  friend vector<vector<Dtype>> dma_macc_ofm_in_mt(const BaseOps<Dtype>& B,
                                                  const AIE2Cfg& A,
                                                  const vector<Dtype>& ofm);
  template <typename Dtype>
  friend void dma_macc_ifm_in_core(const BaseOps<Dtype>& B, const AIE2Cfg& A,
                                   const vector<vector<Dtype>>& vec_img);
  template <typename Dtype>
  friend void dma_macc_wgt_in_core(const BaseOps<Dtype>& B, const AIE2Cfg& A,
                                   const vector<vector<Dtype>>& vec_wgt);
  template <typename Dtype>
  friend void dma_macc_ofm_in_core(const BaseOps<Dtype>& B, const AIE2Cfg& A,
                                   const vector<vector<Dtype>>& vec_ofm);
};

template <typename T>
vector<vector<T>> BaseOps<T>::ifm_mt_arrangement(const vector<T>& ifm) {
  vector<vector<T>> res;
  res = dma_macc_ifm_in_mt(*this, AIE, ifm);
  dma_macc_ifm_in_core(*this, AIE, res);
  return res;
}

template <typename T>
vector<vector<T>> BaseOps<T>::wgt_mt_arrangement(const vector<T>& wgt,
                                                 const vector<T>& bias) {
  vector<vector<T>> res;
  res = dma_macc_wgt_in_mt(*this, AIE, wgt, bias);
  dma_macc_wgt_in_core(*this, AIE, res);
  return res;
}

template <typename T>
vector<vector<T>> BaseOps<T>::ofm_mt_arrangement(const vector<T>& ofm) {
  vector<vector<T>> res;
  res = dma_macc_ofm_in_mt(*this, AIE, ofm);
  dma_macc_ofm_in_core(*this, AIE, res);
  return res;
}

template <typename T>
void BaseOps<T>::generate_core_ctrl_pkt() {
  uint32_t bufping = 1;
  uint32_t ofmPing = 1;
  uint32_t PARAM_BASE_ADDR =
      bufping ? PARAM_BASE_ADDR_PING : PARAM_BASE_ADDR_PONG;

  set_layer_params();

  for (int idx_core_num = 0; idx_core_num < AIE.CORE_NUM; idx_core_num++) {
    string fn_ctrl = getCtrlPktfname(idx_core_num, "co");
    cout << "[INFO]: packing ctrl package: " << fn_ctrl << endl;

    string DM_Param =
        setDMParam(PARAM_BASE_ADDR, layer_param, LAYER_PARAM_WORD_SIZE);
    appendStr2File(fn_ctrl, DM_Param);

    string Core_ifm_BD = ifm_core_set_window();
    string Core_wgt_BD = wgt_core_set_window();
    string Core_ofm_BD = ofm_core_set_window();

    appendStr2File(fn_ctrl, Core_ifm_BD);
    appendStr2File(fn_ctrl, Core_wgt_BD);
    appendStr2File(fn_ctrl, Core_ofm_BD);
  }

  for (int idx_core_num = 0; idx_core_num < AIE.CORE_NUM; idx_core_num++) {
    string fn_ctrl = getCtrlPktfname(idx_core_num, "co");
    for (int idx_oh_iter = 0; idx_oh_iter < OHIter_NUM; idx_oh_iter++) {
      for (int idx_ow_iter = 0; idx_ow_iter < OWIter_NUM; idx_ow_iter++) {
        for (int idx_ic_iter = 0; idx_ic_iter < ICIter_NUM; idx_ic_iter++) {
          // IFM + WGT
          // start S2MM for ifm
          string ctrPktStr = startPkts(bufping, 0);

          if (wgt_wnd_byte_size > 0) {
            // only start wgt S2MM, bias is next to wgt automatically
            ctrPktStr += startPkts(bufping, 1);
          }

          appendStr2File(fn_ctrl, ctrPktStr);
          // OFM  misc takes one core
          string ctrOfmStr = startPkts(ofmPing, 2);
          appendStr2File(fn_ctrl, ctrOfmStr);
          ofmPing = (ofmPing) ? 0 : 1;
          bufping = (bufping) ? 0 : 1;
        }
      }
    }
  }
}

template <typename T>
void BaseOps<T>::generate_mt_ctrl_pkts() {
  ifm_mt_to_core_ctrl_pkt();
  ifm_pl_to_mt_ctrl_pkt();

  ofm_core_to_mt_ctrl_pkt();
  ofm_mt_to_pl_ctrl_pkt();

  wgt_mt_to_core_ctrl_pkt();
  wgt_pl_to_mt_ctrl_pkt();
}

template <typename T>
void BaseOps<T>::ifm_mt_to_core_ctrl_pkt() {
  // e. ctrl package generate for mem tile
  cout << "[INFO]: e. ctrl-packet for MEMT file generating..." << endl;
  int src_h = AIE.src_h;
  int src_w = AIE.src_w;
  // int dst_h = AIE.dst_h;
  // int dst_w = AIE.dst_w;
  int ICp = AIE.ICp;
  int OCp = AIE.OCp;
  int OWp = AIE.OWp;
  int OHp = AIE.OHp;
  int ICG = AIE.icg;
  int ICG_tile = AIE.icg_tile;
  int OCG_tile = AIE.ocg_tile;
  int OWG_tile = AIE.owg_tile;
  int OHG_tile = AIE.ohg_tile;
  // int kernel_w = AIE.kernel_w;
  // int kernel_h = AIE.kernel_h;
  int stride_w = AIE.stride_w;
  int stride_h = AIE.stride_h;

  int pad_left = AIE.pad_left;
  int pad_top = AIE.pad_top;
  int pad_right = AIE.pad_right;
  int pad_bottom = AIE.pad_bottom;
  // int hot_code = 0;
  int count = 0;

  if (OW_offset) {
    ofm_wnd_byte_size = (OWin1iter - OW_offset) * OHin1iter * OCG_tile * OCp;
  }

  // e.1 generate img ctrl-pkts for MEMT to AIE
  for (int idx_img_grp = 0; idx_img_grp < AIE.IMG_MT_GRP; idx_img_grp++) {
    for (int idx_img_num = 0; idx_img_num < AIE.IMG_MT_NUM; idx_img_num++) {
      int isMT2IFMPing = 1;
      string fn_ctrl = getCtrlPktfname(idx_img_num, "mti");
      count = 0;
      for (int idx_oh_iter = 0; idx_oh_iter < OHIter_NUM; idx_oh_iter++) {
        for (int idx_ow_iter = 0; idx_ow_iter < OWIter_NUM; idx_ow_iter++) {
          for (int idx_ic_iter = 0; idx_ic_iter < ICIter_NUM; idx_ic_iter++) {
            cout << "[INFO]: packing image ctrl package for MEMT to AIE "
                    "[idx_ic_iter, idx_ow_iter, idx_oh_iter] = [";
            cout << idx_ic_iter << ", " << idx_ow_iter << ", " << idx_oh_iter
                 << "] : " << fn_ctrl << endl;

            int Lock_Acq_ID = 66;
            int Lock_Rel_ID = 67;
            uint32_t mt_img_ifm_addr =
                idx_ic_iter * ICp * ICG_tile +
                (idx_ow_iter * (OWp * OWG_tile * stride_w) -
                 ((idx_ow_iter == 0) ? 0 : pad_left)) *
                    ICp * ICG +
                (idx_oh_iter * (OHp * OHG_tile * stride_h) -
                 ((idx_oh_iter == 0) ? 0 : pad_top)) *
                    src_w * ICp * ICG;

            string img_bd2core = set_mt2co_ifm(
                idx_img_grp, idx_img_num, img_wnd_byte_size, mt_img_ifm_addr,
                count, (idx_ow_iter == 0 ? pad_left : 0),
                (idx_oh_iter == 0 ? pad_top : 0),
                (idx_ow_iter == (OWIter_NUM - 1) ? pad_right : 0),
                (idx_oh_iter == (OHIter_NUM - 1) ? pad_bottom : 0), IWin1Iter,
                IHin1Iter, src_h, src_w, stride_w, ICG, ICG_tile, Lock_Acq_ID,
                Lock_Rel_ID, count == 0 ? 1 : 0, 1, isMT2IFMPing);

            appendStr2File(fn_ctrl, img_bd2core);
            string img_bd2core_start =
                startMTPkts(isMT2IFMPing, 0, idx_img_grp);

            appendStr2File(fn_ctrl, img_bd2core_start);
            isMT2IFMPing = (isMT2IFMPing + 1) % 2;
            count += 1;
          }
          //}
        }
      }
    }
  }
}

template <typename T>
void BaseOps<T>::ifm_pl_to_mt_ctrl_pkt() {
  int hot_code = 0;
  // e.3 generate img ctrl-pkts for PL to MEMT
  for (int idx_img_grp = 0; idx_img_grp < AIE.IMG_MT_GRP; idx_img_grp++) {
    for (int idx_img_num = 0; idx_img_num < AIE.IMG_MT_NUM; idx_img_num++) {
      int isPL2IFMPing = 1;
      string fn_ctrl = getCtrlPktfname(idx_img_num, "mti");
      cout << "[INFO]: packing image ctrl package for PL to MEMT = " << fn_ctrl
           << endl;
      hot_code = 15;
      int Lock_Acq_ID = 67;
      int Lock_Rel_ID = 66;
      string img_bd2memt = setMTfor1DPL(
          idx_img_grp, idx_img_num, img_mt_grp_byte_size, 0, hot_code,
          Lock_Acq_ID, Lock_Rel_ID, 1, 1, isPL2IFMPing, 0);
      img_bd2memt += startMTPkts(isPL2IFMPing, 6, idx_img_grp);
      appendStr2File(fn_ctrl, img_bd2memt);
      isPL2IFMPing = (isPL2IFMPing + 1) % 2;
    }
  }
}

template <typename T>
void BaseOps<T>::ofm_core_to_mt_ctrl_pkt() {
  // int src_h = AIE.src_h;
  // int src_w = AIE.src_w;
  // int dst_h = AIE.dst_h;
  int dst_w = AIE.dst_w;
  // int ICp = AIE.ICp;
  int OCp = AIE.OCp;
  // int OWp = AIE.OWp;
  int OHp = AIE.OHp;
  // int ICG = AIE.icg;
  int OCG = AIE.ocg;
  // int ICG_tile = AIE.icg_tile;
  int OCG_tile = AIE.ocg_tile;
  // int OWG_tile = AIE.owg_tile;
  int OHG_tile = AIE.ohg_tile;
  // int kernel_w = AIE.kernel_w;
  // int kernel_h = AIE.kernel_h;
  // int stride_w = AIE.stride_w;
  // int stride_h = AIE.stride_h;

  // int pad_left = AIE.pad_left;
  // int pad_top = AIE.pad_top;
  // int pad_right = AIE.pad_right;
  // int pad_bottom = AIE.pad_bottom;
  int hot_code = 0;
  int count = 0;

  // e.2 generate img ctrl-pkts for AIE to MEMT
  for (int idx_img_grp = 0; idx_img_grp < AIE.IMG_MT_GRP; idx_img_grp++) {
    for (int idx_img_num = 0; idx_img_num < AIE.IMG_MT_NUM; idx_img_num++) {
      int isCO2OFMPing = 1;
      string fn_ctrl = getCtrlPktfname(idx_img_num, "mti");
      count = 0;
      for (int idx_oh_iter = 0; idx_oh_iter < OHIter_NUM; idx_oh_iter++) {
        for (int idx_ow_iter = 0; idx_ow_iter < OWIter_NUM; idx_ow_iter++) {
          for (int idx_oc_iter = 0; idx_oc_iter < OCIter_NUM; idx_oc_iter++) {
            cout << "[INFO]: packing image ctrl package for AIE to MEMT "
                    "[idx_oc_iter, idx_ow_iter, idx_oh_iter] = [";
            cout << idx_oc_iter << ", " << idx_ow_iter << ", " << idx_oh_iter
                 << "] : " << fn_ctrl << endl;
            // int Lock_Acq_ID = 67;
            // int Lock_Rel_ID = 66;
            int Lock_Acq_ID = 65;
            int Lock_Rel_ID = 64;
            uint32_t mt_img_ofm_addr =
                idx_oc_iter * OCp * OCG_tile +
                idx_ow_iter * OWin1iter * OCp * OCG +
                idx_oh_iter * OHG_tile * OHp * dst_w * OCp * OCG;
            hot_code = 15;
            int BD_first = 1;
            int BD_last = 1;
            if (OHIter_NUM * OWIter_NUM * OCIter_NUM > 1) {
              if (count == 0) {
                BD_last = 0;
              } else if (count == OHIter_NUM * OWIter_NUM * OCIter_NUM) {
                BD_first = 0;
              } else {
                BD_first = 0;
                BD_last = 0;
              }
            }
            // bool BD_casc_fir, bool BD_casc_last, int isPing)
            string img_bd2memt = setMISCImgCO2MT(
                idx_img_grp, idx_img_num, ofm_wnd_byte_size, mt_img_ofm_addr,
                hot_code, OHin1iter, OWin1iter, OW_offset, OCG, OCG_tile,
                OCG_offset, dst_w, Lock_Acq_ID, Lock_Rel_ID, BD_first, BD_last,
                isCO2OFMPing);
            img_bd2memt += startMTPkts(isCO2OFMPing, 2, idx_img_grp);
            appendStr2File(fn_ctrl, img_bd2memt);
            isCO2OFMPing = (isCO2OFMPing + 1) % 2;
            count += 1;
          }
        }
      }
    }
  }
}

template <typename T>
void BaseOps<T>::ofm_mt_to_pl_ctrl_pkt() {
  int hot_code = 0;
  for (int idx_img_grp = 0; idx_img_grp < AIE.IMG_MT_GRP; idx_img_grp++) {
    for (int idx_img_num = 0; idx_img_num < AIE.IMG_MT_NUM; idx_img_num++) {
      int isOFM2PLPing = 1;
      string fn_ctrl = getCtrlPktfname(idx_img_num, "mti");
      cout << "[INFO]: packing image ctrl package for MEMT to PL = " << fn_ctrl
           << endl;
      hot_code = 15;
      int Lock_Acq_ID = 64;
      int Lock_Rel_ID = 65;
      // int Lock_Acq_ID = 66;
      // int Lock_Rel_ID = 67;
      // string img_bd2pl = setMTfor1DPL(idx_img_grp, idx_img_num,
      // (ofm_save_parallel==1)?ofm_mt_grp_byte_size:ofm_mt_port_byte_size[idx_img_num],
      // 0, hot_code,
      string img_bd2pl = setMTfor1DPL(
          idx_img_grp, idx_img_num, ofm_mt_grp_byte_size, 0, hot_code,
          Lock_Acq_ID, Lock_Rel_ID, 1, 1, isOFM2PLPing, 1);
      img_bd2pl += startMTPkts(isOFM2PLPing, 7, idx_img_grp);
      appendStr2File(fn_ctrl, img_bd2pl);
      isOFM2PLPing = (isOFM2PLPing + 1) % 2;
    }
  }
}

template <typename T>
void BaseOps<T>::wgt_mt_to_core_ctrl_pkt() {
  // int src_h = AIE.src_h;
  // int src_w = AIE.src_w;
  // int dst_h = AIE.dst_h;
  // int dst_w = AIE.dst_w;
  int ICp = AIE.ICp;
  // int OCp = AIE.OCp;
  // int OWp = AIE.OWp;
  // int OHp = AIE.OHp;
  int ICG = AIE.icg;
  // int OCG = AIE.ocg;
  int ICG_tile = AIE.icg_tile;
  int OCG_tile = AIE.ocg_tile;
  // int OWG_tile = AIE.owg_tile;
  // int OHG_tile = AIE.ohg_tile;
  int kernel_w = AIE.kernel_w;
  int kernel_h = AIE.kernel_h;
  // int stride_w = AIE.stride_w;
  // int stride_h = AIE.stride_h;

  // int pad_left = AIE.pad_left;
  // int pad_top = AIE.pad_top;
  // int pad_right = AIE.pad_right;
  // int pad_bottom = AIE.pad_bottom;
  // int hot_code = 0;
  int count = 0;

  if (wgt_wnd_byte_size + bias_wnd_byte_size > 0) {
    // e.5 generate wgt ctrl-pkts for MEMT to AIE
    for (int idx_wgt_grp = 0; idx_wgt_grp < AIE.WGT_MT_GRP; idx_wgt_grp++) {
      for (int idx_wgt_num = 0; idx_wgt_num < AIE.WGT_MT_NUM; idx_wgt_num++) {
        int isMT2WGTPing = 1;
        // int isMT2BIASPing = 1;
        string fn_ctrl = getCtrlPktfname(idx_wgt_num, "mtw");
        count = 0;
        for (int idx_oh_iter = 0; idx_oh_iter < OHIter_NUM; idx_oh_iter++) {
          for (int idx_ow_iter = 0; idx_ow_iter < OWIter_NUM; idx_ow_iter++) {
            for (int idx_ic_iter = 0; idx_ic_iter < ICIter_NUM; idx_ic_iter++) {
              cout << "[INFO]: packing weight ctrl package for MEMT to AIE "
                      "[idx_ic_iter, idx_ow_iter, idx_oh_iter] = [";
              cout << idx_ic_iter << ", " << idx_ow_iter << ", " << idx_oh_iter
                   << "] : " << fn_ctrl << endl;
              int Lock_Acq_ID = 64;
              int Lock_Rel_ID = 65;
              uint32_t mt_wgt_addr =
                  idx_ic_iter * kernel_h * kernel_w * ICp * ICG_tile +
                  idx_wgt_num * kernel_h * kernel_w * ICp * ICG;
              // uint32_t mt_wgt_addr = 0;
              //
              uint32_t mt_bias_addr =
                  weights_mt_grp_byte_size + idx_ic_iter * ICp * ICG_tile;
              if (wgt_wnd_byte_size == ICp && bias_wnd_byte_size == ICp) {
                mt_wgt_addr = 0;
                mt_bias_addr = weights_mt_grp_byte_size;
              }
              // hot_code =
              //     (count == 0) ? 15 : (count == 1) ? 15 : (count == 2) ? 11 : 3;
              string wgt_bd2co = setMiscWgtMT2CO(
                  idx_wgt_grp, idx_wgt_num, wgt_wnd_byte_size, mt_wgt_addr,
                  bias_wnd_byte_size, mt_bias_addr, count, kernel_w * kernel_h,
                  ICG_tile, OCG_tile, ICG, Lock_Acq_ID, Lock_Rel_ID,
                  count == 0 ? 1 : 0,
                  ((idx_wgt_num == AIE.WGT_MT_NUM) &&
                   (idx_ic_iter == ICIter_NUM - 1))
                      ? 1
                      : 0,
                  isMT2WGTPing);
              wgt_bd2co += startMTPkts(isMT2WGTPing, 8, idx_wgt_grp);

              wgt_bd2co += startMTPkts(isMT2WGTPing, 12, idx_wgt_grp);
              appendStr2File(fn_ctrl, wgt_bd2co);
              isMT2WGTPing = (isMT2WGTPing + 1) % 2;
              count++;
            }
            //}
          }
        }
      }
    }
  }
}

template <typename T>
void BaseOps<T>::wgt_pl_to_mt_ctrl_pkt() {
  int hot_code = 0;
  // int count = 0;
  int wgt_mt_grp_byte_size = weights_mt_grp_byte_size + bias_mt_grp_byte_size;
  // e.6 generate wgt ctrl-pkts for PL to MEMT
  if (weights_mt_grp_byte_size + bias_mt_grp_byte_size > 0) {
    for (int idx_wgt_grp = 0; idx_wgt_grp < AIE.WGT_MT_GRP; idx_wgt_grp++) {
      for (int idx_wgt_num = 0; idx_wgt_num < AIE.WGT_MT_NUM; idx_wgt_num++) {
        int isPL2WGTPing = 1;
        string fn_ctrl = getCtrlPktfname(idx_wgt_num, "mtw");
        cout << "[INFO]: packing weight ctrl package for PL to MEMT = "
             << fn_ctrl << endl;
        hot_code = 15;
        int Lock_Acq_ID = 65;
        int Lock_Rel_ID = 64;
        // this line is not right for idx_wgt_num > 1
        assert(idx_wgt_num < 1 && "now this only support wgt_mt_num == 1");
        string wgt_bd2memt =
            setMTfor1DPL(idx_wgt_grp, idx_wgt_num, wgt_mt_grp_byte_size,
                         idx_wgt_num * weights_mt_grp_byte_size, hot_code,
                         Lock_Acq_ID, Lock_Rel_ID, 1, 1, isPL2WGTPing, 2);
        wgt_bd2memt += startMTPkts(isPL2WGTPing, 10, idx_wgt_grp);
        appendStr2File(fn_ctrl, wgt_bd2memt);
        isPL2WGTPing = (isPL2WGTPing + 1) % 2;
      }
    }
  } else {
    string fn_ctrl = getCtrlPktfname(0, "mtw");
    string wgt_bd2memt = "";
    appendStr2File(fn_ctrl, wgt_bd2memt);
  }
}

}  // namespace alu
}  // namespace xv2dpu
}  // namespace cosim
