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
#include "Misc2Cfg.hpp"
#include "OpAvgp.hpp"
#include "OpDwconv.hpp"
#include "OpELEWA.hpp"
#include "OpELEWM.hpp"
#include "OpELEWS.hpp"
#include "OpMaxp.hpp"
#include "params.hpp"
#include "simCfg.hpp"
#include "simUtil.hpp"

using namespace cosim::xv2dpu::alu;

template <>
void Alu<DPUVersion::XV2DPU>::alu_config_mt_transmit_to_stream() const {
  // TODO
}

template <>
void Alu<DPUVersion::XV2DPU>::alu_config_layer_parameter() const {
  // TODO
}

template <>
void Alu<DPUVersion::XV2DPU>::alu_config_DMA_BDs() const {
  // TODO
}

template <>
void Alu<DPUVersion::XV2DPU>::alu_config_start_DMA() const {
  // TODO
}

template <>
void Alu<DPUVersion::XV2DPU>::alu_config_mt_receive_from_stream() const {
  // TODO
}

// template <>
// void Alu<DPUVersion::XV2DPU>::alu_gen_ctrlpkg() const {
//   alu_config_mt_transmit_to_stream();
//   alu_config_layer_parameter();
//   alu_config_DMA_BDs();
//   alu_config_start_DMA();
//   alu_config_mt_receive_from_stream();
// }

template <>
void Alu<DPUVersion::XV2DPU>::alu_gen_ctrlpkg() const {
  if (!SimCfgProxy::Params().CO_SIM_ON) return;

  //  input
  int core_num = 4;
  int src_h = src_h_;
  int src_w = src_w_;
  int ic = ic_;
  int oc = oc_;
  int kernel_h = kernel_h_;
  int kernel_w = kernel_w_;
  int stride_h = stride_h_;
  int stride_w = stride_w_;
  //!!! is real pad value is needed?
  int pad_top = pad_top_;
  int pad_bottom = pad_bottom_;
  int pad_left = pad_left_;
  int pad_right = pad_right_;

  int owg_tile = tile_owg_;
  int ocg_tile = tile_cg_;
  int icg_tile = tile_cg_;
  int ohg_tile = tile_ohg_;

  int SHIFT_CUT = shift_cut_;
  int SHIFT_BIAS = shift_bias_;

  //!!! check needed
  int HSIGMOID_IN = hsigmoid_in_;
  int HSIGMOID_OUT = shift_hsigmoid_;
  int HSWISH_OUT = shift_hswish_;

  int ACT_TYPE = act_type_;
  MISC_CASE EXCE_TYPE = MISC_CASE::CASE_NONE;
  if (exec_mode_ == Calc::ALU_TYPE_MACC) {
    EXCE_TYPE = MISC_CASE::CASE_MACC;
  } else if (exec_mode_ == Calc::ALU_TYPE_COMP) {
    EXCE_TYPE = MISC_CASE::CASE_MAXP;
  } else if (exec_mode_ == Calc::ALU_TYPE_ELEW_ADD ||
             exec_mode_ == Calc::ALU_TYPE_ELEW_MUL) {
    if (1 == num_) {
      EXCE_TYPE = MISC_CASE::CASE_ELEWS;
    } else if (exec_mode_ == Calc::ALU_TYPE_ELEW_ADD) {
      EXCE_TYPE = MISC_CASE::CASE_ELEWADD;
    } else {
      EXCE_TYPE = MISC_CASE::CASE_ELEWMUL;
    }
  } else {
    assert(false && "unsupported alu exec type");
  }

  // configuration
  MISC2Cfg aluCfg(core_num, src_h, src_w, ic, oc, kernel_h, kernel_w, stride_h,
                  stride_w, pad_top, pad_bottom, pad_left, pad_right, owg_tile,
                  ocg_tile, icg_tile, ohg_tile, SHIFT_CUT, SHIFT_BIAS,
                  HSIGMOID_IN, HSIGMOID_OUT, HSWISH_OUT, ACT_TYPE, EXCE_TYPE);
  aluCfg.Check();
  /*
  std::shared_ptr<BaseOps<int8_t>> baseOp = nullptr;
  switch (EXCE_TYPE) {
    case (MISC_CASE::CASE_MACC): {
      baseOp = std::make_shared<OpDwconv>(aluCfg);
    } break;
    case (MISC_CASE::CASE_AVGP): {
      baseOp = std::make_shared<OpAvgp>(aluCfg);
    } break;
    case (MISC_CASE::CASE_MAXP): {
      baseOp = std::make_shared<OpMaxp>(aluCfg);
    } break;
    case (MISC_CASE::CASE_ELEWS): {
      baseOp = std::make_shared<OpELEWS>(aluCfg);
    } break;
    case (MISC_CASE::CASE_ELEWMUL): {
      baseOp = std::make_shared<OpELEWM>(aluCfg);
    } break;
    case (MISC_CASE::CASE_ELEWADD): {
      baseOp = std::make_shared<OpELEWA>(aluCfg);
    } break;
    default: { assert(false && "unsupported alu exec type"); }
  }

  baseOp->generate_core_ctrl_pkt();
  baseOp->generate_mt_ctrl_pkts();
  */
}
