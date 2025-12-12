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
#include "Targets.hpp"
#include "common.hpp"

#define XVDPU_ACQ_WIN 290
#define XVDPU_REG_PREP 16
#define XVDPU_LOAD_INTER 6
#define XVDPU_SAVE_INTER 20
#define XVDPU_INNER_CAL 8
#define XVDPU_CASC_TRANS 24
#define XVDPU_CASC_TRANS_LK 30

class MECfg : public Target<DPUVersion::XVDPU> {
 public:
  enum ET_A : int32_t {
    ET_A_IN2DM = 1,
    ET_A_INaDM2DM = 2,
    ET_A_INaDM2CASC = 3,
    ET_A_IN2CASC = 4,
    ET_A_IN2CASC_IL2 = 5,
    ET_A_IN2CASC_IL1 = 6,
    ET_A_IN2DM_OLNP = 7,
    ET_A_INaDM2DM_OLNP = 8,
    ET_A_INaDM2CASC_OLNP = 9,
  };
  enum ET_B : int32_t {
    ET_B_IN2DM = 1,
    ET_B_INaDM2DM = 2,
    ET_B_INaDMaCASC2OUT = 3,
    ET_B_INaCASC2OUT = 4,
    ET_B_INaCASC2OUT_IL2 = 5,
    ET_B_INaCASC2OUT_IL1 = 6,
    ET_B_IN2DM_OLNP = 7,
    ET_B_INaDM2DM_OLNP = 8,
    ET_B_INaCASC2OUT_OL1 = 9,
    ET_B_INaDMaCASC2OUT_LK = 10,
    ET_B_INaCASC2OUT_LK = 11,
    ET_B_INaCASC2OUT_IL2_LK = 12,
    ET_B_INaCASC2OUT_IL1_LK = 13,
    ET_B_INaCASC2OUT_OL1_LK = 14,
  };

  const static unsigned int S2MM;
  const static unsigned int MM2S;
  const static unsigned int MAX_WORD_PER_PKT;
  const static string fn_img_prefix;
  const static string fn_wgt_prefix;
  const static string fn_rlt_prefix;
  const static string fn_inter_prefix;
  const static string fn_tlast;

 public:
  MECfg() { init(); };
  ~MECfg(){};

  static MECfg& Instance() {
    static MECfg mecfg;
    return mecfg;
  }
  void Touch(){};

  void SetbufPing(int bufPing) { bufPing_ = bufPing; }
  void SetrltPing(int rltPing) { rltPing_ = rltPing; }
  void SetrltPing_B(int rltPing_B) { rltPing_B_ = rltPing_B; }
  void SetuseRltB(int use) { useRltB_ = use; }
  void SetXYRltB(int use) { setXYRltB_ = use; }
  void SetIfmB(int use) { setIfmB_ = use; }
  void Setnum_weights_port(int num_weights_port) {
    num_weights_port_ = num_weights_port;
  }
  void Setimg_port_width(int img_port_width) {
    img_port_width_ = img_port_width;
  };
  void Setweights_port_width(int weights_port_width) {
    weights_port_width_ = weights_port_width;
  };
  void Setrlt_port_width(int rlt_port_width) {
    rlt_port_width_ = rlt_port_width;
  };

 public:
  int bufPing() { return bufPing_; }
  int rltPing() { return rltPing_; }
  int rltPing_B() { return rltPing_B_; }
  int useRltB() { return useRltB_; }
  int isSetXYRltB() { return setXYRltB_; }
  int isSetIfmB() { return setIfmB_; }
  int num_weights_port() { return num_weights_port_; }
  int img_port_width() { return img_port_width_; };
  int weights_port_width() { return weights_port_width_; };
  int rlt_port_width() { return rlt_port_width_; };

  unsigned int ID_CTRL() { return ID_CTRL_; }
  unsigned int ID_DATA() { return ID_DATA_; }

  unsigned int BD_IMG_PING() { return BD_IMG_PING_; }
  unsigned int BD_IMG_PONG() { return BD_IMG_PONG_; }
  unsigned int BD_IMG_PING_B() { return BD_IMG_PING_B_; }
  unsigned int BD_IMG_PONG_B() { return BD_IMG_PONG_B_; }
  unsigned int IMG_CHANNEL() { return IMG_CHANNEL_; }
  unsigned int LOCK_BD_IMG_PING() { return LOCK_BD_IMG_PING_; }
  unsigned int LOCK_BD_IMG_PONG() { return LOCK_BD_IMG_PONG_; }

  unsigned int BD_WGT_PING() { return BD_WGT_PING_; }
  unsigned int BD_WGT_PONG() { return BD_WGT_PONG_; }
  unsigned int WGT_CHANNEL() { return WGT_CHANNEL_; }
  unsigned int LOCK_BD_WGT_PING() { return LOCK_BD_WGT_PING_; }
  unsigned int LOCK_BD_WGT_PONG() { return LOCK_BD_WGT_PONG_; }

  unsigned int BD_RLT_PING() { return BD_RLT_PING_; }
  unsigned int BD_RLT_PONG() { return BD_RLT_PONG_; }
  unsigned int RLT_CHANNEL() { return RLT_CHANNEL_; }
  unsigned int LOCK_BD_RLT_PING() { return LOCK_BD_RLT_PING_; }
  unsigned int LOCK_BD_RLT_PONG() { return LOCK_BD_RLT_PONG_; }
  unsigned int BD_RLT_PING_B() { return BD_RLT_PING_B_; }
  unsigned int BD_RLT_PONG_B() { return BD_RLT_PONG_B_; }
  unsigned int RLT_CHANNEL_B() { return RLT_CHANNEL_B_; }
  unsigned int LOCK_BD_RLT_PING_B() { return LOCK_BD_RLT_PING_B_; }
  unsigned int LOCK_BD_RLT_PONG_B() { return LOCK_BD_RLT_PONG_B_; }

  unsigned int IMG_BASE_ADDR_PING() { return IMG_BASE_ADDR_PING_; }
  unsigned int IMG_BASE_ADDR_PONG() { return IMG_BASE_ADDR_PONG_; }

  bool AorB() { return AorB_; };
  unsigned int ITER_PARAM_WORD_SIZE() { return ITER_PARAM_WORD_SIZE_; }
  unsigned int LAYER_PARAM_WORD_SIZE() { return LAYER_PARAM_WORD_SIZE_; }
  unsigned int MAX_WLUT_WORD_SIZE() { return MAX_WLUT_WORD_SIZE_; }
  unsigned int PARAM_BASE_ADDR_PING() { return PARAM_BASE_ADDR_PING_; }
  unsigned int PARAM_BASE_ADDR_PONG() { return PARAM_BASE_ADDR_PONG_; }
  unsigned int WLUT_BASE_ADDR_PING() { return WLUT_BASE_ADDR_PING_; }
  unsigned int WLUT_BASE_ADDR_PONG() { return WLUT_BASE_ADDR_PONG_; }

 private:
  void init();

 private:
  int bufPing_;
  int useRltB_;
  int setXYRltB_;
  int setIfmB_;
  int rltPing_;
  int rltPing_B_;
  int num_weights_port_;
  int img_port_width_;
  int weights_port_width_;
  int rlt_port_width_;

  // weight packets split
  unsigned int ID_CTRL_;
  unsigned int ID_DATA_;

  // img startBD, channel, lockID
  unsigned int BD_IMG_PING_;
  unsigned int BD_IMG_PONG_;
  unsigned int BD_IMG_PING_B_;
  unsigned int BD_IMG_PONG_B_;
  unsigned int IMG_CHANNEL_;
  unsigned int LOCK_BD_IMG_PING_;
  unsigned int LOCK_BD_IMG_PONG_;

  // weights startBD, channel, lockID
  unsigned int BD_WGT_PING_;
  unsigned int BD_WGT_PONG_;
  unsigned int WGT_CHANNEL_;
  unsigned int LOCK_BD_WGT_PING_;
  unsigned int LOCK_BD_WGT_PONG_;

  // out startBD, channel, lockID
  unsigned int BD_RLT_PING_;
  unsigned int BD_RLT_PONG_;
  unsigned int RLT_CHANNEL_;
  unsigned int LOCK_BD_RLT_PING_;
  unsigned int LOCK_BD_RLT_PONG_;
  unsigned int BD_RLT_PING_B_;
  unsigned int BD_RLT_PONG_B_;
  unsigned int RLT_CHANNEL_B_;
  unsigned int LOCK_BD_RLT_PING_B_;
  unsigned int LOCK_BD_RLT_PONG_B_;
  // img pointer
  unsigned int IMG_BASE_ADDR_PING_;  // 0x4000/4, need 32-bits aligned address
  unsigned int IMG_BASE_ADDR_PONG_;  // 0x6000/4

  bool AorB_;
  // buf size
  unsigned int ITER_PARAM_WORD_SIZE_;
  unsigned int LAYER_PARAM_WORD_SIZE_;
  unsigned int MAX_WLUT_WORD_SIZE_;
  // buf pointer
  unsigned int PARAM_BASE_ADDR_PING_;
  unsigned int PARAM_BASE_ADDR_PONG_;
  unsigned int WLUT_BASE_ADDR_PING_;
  unsigned int WLUT_BASE_ADDR_PONG_;
};
