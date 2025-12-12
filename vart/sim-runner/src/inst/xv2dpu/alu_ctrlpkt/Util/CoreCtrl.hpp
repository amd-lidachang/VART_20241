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

#include "genBD.hpp"
#include "params.hpp"

namespace cosim {
namespace xv2dpu {
namespace alu {

void appendStr2File(const string& filename, const string& data) {
  std::ofstream f_os;
  f_os.open(filename, std::ios::app);
  f_os << data;
  f_os.close();
}

string setDMParam(int addr, int32_t* param, int len) {
  const int addrIncr = PKT_WORD_WIDTH * 0x04;  // 8 Byte
  const int ROW = 0;
  const int COL = 0;
  int clpkts_dm_pnum = (len + PKT_WORD_WIDTH - 1) / PKT_WORD_WIDTH;  // 16
  int last_pkt_word =
      (len % PKT_WORD_WIDTH == 0) ? PKT_WORD_WIDTH : (len % PKT_WORD_WIDTH);
  int last_not_full = (len % PKT_WORD_WIDTH == 0) ? 0 : 1;

  cout << "setDMParam: "
       << "ID_CTRL=" << ID_CTRL << "; clpkts_dm_pnum=" << clpkts_dm_pnum
       << "; last_pkt_word=" << last_pkt_word << ";" << endl;
  string paramStr;

  for (int idx_pnum = 0; idx_pnum < clpkts_dm_pnum - last_not_full;
       idx_pnum++) {
    SetParamPkt paramPkt(ID_CTRL, ROW, COL,
                         addr + idx_pnum * addrIncr,         // addr
                         param + idx_pnum * PKT_WORD_WIDTH,  // data
                         PKT_WORD_WIDTH);                    // size
    paramStr += paramPkt.to_string();
  }

  if (last_not_full == 1) {
    int32_t lastpkt[PKT_WORD_WIDTH];
    for (int i = 0; i < PKT_WORD_WIDTH; i++) {
      if ((((clpkts_dm_pnum - 1) * PKT_WORD_WIDTH) + i) >= len) {
        lastpkt[i] = 0;
      } else {
        lastpkt[i] = *(param + ((clpkts_dm_pnum - 1) * PKT_WORD_WIDTH) + i);
      }
    }
    SetParamPkt paramPkt(ID_CTRL, ROW, COL,
                         addr + (clpkts_dm_pnum - 1) * addrIncr,  // addr
                         lastpkt,                                 // data
                         PKT_WORD_WIDTH);                         // size
    paramStr += paramPkt.to_string();
  }
  return paramStr;
}

string ifm_core_set_window_macc(uint32_t img_wnd_bytes, int IWin1Iter,
                                int IHin1Iter, int ICG_tile) {
  int row = 0, col = 0;
  int BD_IMG;
  int IMG_BASE_ADDR;
  int Lock_ACQ_ID;
  int Lock_REL_ID;

  string ctrPktStr;
  string ctrPktHex;

  for (int isPing = 1; isPing >= 0; isPing--) {
    // IMG DMA_BD
    BD_IMG = (isPing) ? BD_IMG_PING                 // 0
                      : BD_IMG_PONG;                // 1
    IMG_BASE_ADDR = (isPing) ? IMG_BASE_ADDR_PING   // 0x0000
                             : IMG_BASE_ADDR_PONG;  // 0x8000
    // Lock_ID = (isPing) ? 2 : 3;
    Lock_ACQ_ID = 0;
    Lock_REL_ID = 1;
    // Lock_ID = (isPing) ? 0 : 1;
    genCOBDPkt ImgBDPkt(ID_CTRL, row, col, BD_IMG, 7, IMG_BASE_ADDR / 4,
                        (img_wnd_bytes / 2) / 4,
                        (32 * IWin1Iter * IHin1Iter / 4) - 1, 1 - 1,
                        // IWin1Iter*IHin1Iter*ICG_tile, 1-1,
                        ICG_tile, 16 / 4, (32 / 4) - 1,
                        // IWin1Iter*IHin1Iter*ICG_tile, 16/4, (16/4)-1,
                        // 0,0,0,0,0,
                        Lock_ACQ_ID, Lock_REL_ID, 1, 0);
    ctrPktStr += ImgBDPkt.ctrPktStr();
    ctrPktHex += ImgBDPkt.ctrPktHex();

    genCOBDPkt ImgBDcPkt(
        ID_CTRL, row, col, BD_IMG + 6, 7, (IMG_BASE_ADDR + 16) / 4,
        (img_wnd_bytes / 2) / 4, (32 * IWin1Iter * IHin1Iter / 4) - 1, 1 - 1,
        // IWin1Iter*IHin1Iter*ICG_tile, 1-1,
        ICG_tile, 16 / 4, (32 / 4) - 1, Lock_ACQ_ID, Lock_REL_ID, 1, 1);
    ctrPktStr += ImgBDcPkt.ctrPktStr();
    ctrPktHex += ImgBDcPkt.ctrPktHex();
  }

  // appendStr2File("./data/ctrl.txt", ctrPktHex);
  return ctrPktStr;
}

string ifm_core_set_window_elew_single(uint32_t img_wnd_bytes, int IWin1Iter,
                                       int IHin1Iter, int ICG_tile) {
  int row = 0, col = 0;
  int BD_IMG;
  int IMG_BASE_ADDR;
  int Lock_ACQ_ID;
  int Lock_REL_ID;

  string ctrPktStr;
  string ctrPktHex;

  for (int isPing = 1; isPing >= 0; isPing--) {
    // IMG DMA_BD
    BD_IMG = (isPing) ? BD_IMG_PING   // 0
                      : BD_IMG_PONG;  // 1
    // BD_IMG = (isPing) ? 2:3;   //0
    IMG_BASE_ADDR = (isPing) ? (IMG_BASE_ADDR_PING)   // 0x0000
                             : (IMG_BASE_ADDR_PONG);  // 0x8000
    // Lock_ID = (isPing) ? 2 : 3;
    Lock_ACQ_ID = 0;
    Lock_REL_ID = 1;
    // Lock_ID = (isPing) ? 0 : 1;
    genCOBDPkt ImgBDPkt(ID_CTRL, row, col, BD_IMG, 5, IMG_BASE_ADDR / 4,
                        img_wnd_bytes / 4, 0, 0, 0, 0, 0, Lock_ACQ_ID,
                        Lock_REL_ID, 0, 0);
    ctrPktStr += ImgBDPkt.ctrPktStr();
    ctrPktHex += ImgBDPkt.ctrPktHex();
  }

  // appendStr2File("./data/ctrl.txt", ctrPktHex);
  return ctrPktStr;
}

string ifm_core_set_window_elew_double(uint32_t img_wnd_bytes, int IWin1Iter,
                                       int IHin1Iter, int ICG_tile) {
  int row = 0, col = 0;
  int BD_IMG;
  int IMG_BASE_ADDR;
  int WGT_BASE_ADDR;
  int Lock_ACQ_ID;
  int Lock_REL_ID;

  string ctrPktStr;
  string ctrPktHex;

  for (int isPing = 1; isPing >= 0; isPing--) {
    // IMG DMA_BD
    BD_IMG = (isPing) ? BD_IMG_PING                 // 0
                      : BD_IMG_PONG;                // 1
    IMG_BASE_ADDR = (isPing) ? IMG_BASE_ADDR_PING   // 0x0000
                             : IMG_BASE_ADDR_PONG;  // 0x8000
    Lock_ACQ_ID = 0;
    Lock_REL_ID = 1;
    genCOBDPkt ImgBDPkt(ID_CTRL, row, col, BD_IMG, 5, IMG_BASE_ADDR / 4,
                        img_wnd_bytes / 2 / 4, 0, 0, 0, 0, 0, Lock_ACQ_ID,
                        Lock_REL_ID, 1, 0);
    ctrPktStr += ImgBDPkt.ctrPktStr();
    ctrPktHex += ImgBDPkt.ctrPktHex();

    // the second ifm put in WGT BANK
    WGT_BASE_ADDR = (isPing) ? (WGT_BASE_ADDR_PING) :  // 0x4000
                        (WGT_BASE_ADDR_PONG);          // 0xC000
    genCOBDPkt ImgBDcPkt(ID_CTRL, row, col, BD_IMG + 6, 5, (WGT_BASE_ADDR) / 4,
                         (img_wnd_bytes / 2) / 4, 0, 0, 0, 0, 0, Lock_ACQ_ID,
                         Lock_REL_ID, 1, 1);
    ctrPktStr += ImgBDcPkt.ctrPktStr();
    ctrPktHex += ImgBDcPkt.ctrPktHex();
  }

  // appendStr2File("./data/ctrl.txt", ctrPktHex);
  return ctrPktStr;
}

string wgt_core_set_window_macc(uint32_t wgt_wnd_bytes,
                                uint32_t bias_wnd_bytes) {
  int row = 0, col = 0;
  // int BD_IMG;
  int BD_WGT;
  // int BD_OFM;
  // int IMG_BASE_ADDR;
  int WGT_BASE_ADDR;
  // int OFM_BASE_ADDR;
  int Lock_ACQ_ID;
  int Lock_REL_ID;

  string ctrPktStr;
  string ctrPktHex;

  if (wgt_wnd_bytes > 0) {
    for (int isPing = 1; isPing >= 0; isPing--) {
      // WGT DMA_BD
      BD_WGT = (isPing) ? (BD_WGT_PING) :                // 2
                   (BD_WGT_PONG);                        // 3
      WGT_BASE_ADDR = (isPing) ? (WGT_BASE_ADDR_PING) :  // 0x4000
                          (WGT_BASE_ADDR_PONG);          // 0xC000
      // uint32_t WGT_BYTE_LENGTH = wgt_wnd_bytes + bias_wnd_bytes;

      cout << "wgt_wnd bytes: " << wgt_wnd_bytes << endl;
      cout << "bias_wnd bytes: " << bias_wnd_bytes << endl;
      // Lock_ID = (isPing) ? 2 : 3;
      // Lock_ID = (isPing) ? 0 : 1;
      Lock_ACQ_ID = 2;
      Lock_REL_ID = 3;
      // genCOBDPkt WgtBDPkt(ID_CTRL, row, col, BD_WGT, 7,
      //                  WGT_BASE_ADDR/4, WGT_BYTE_LENGTH/4,
      //                  //0,0,0,0,0,
      //                  8-1, 1-1, (WGT_BYTE_LENGTH/4/4/2), 4, 4-1,
      //                  Lock_ID);
      genCOBDPkt WgtBDPkt(ID_CTRL, row, col, BD_WGT, 7, WGT_BASE_ADDR / 4,
                          wgt_wnd_bytes / 4,
                          // 0,0,0,0,0,
                          8 - 1, 1 - 1, (wgt_wnd_bytes / 4 / 4 / 2), 4, 4 - 1,
                          Lock_ACQ_ID, Lock_REL_ID, 1, 0);
      ctrPktStr += WgtBDPkt.ctrPktStr();
      ctrPktHex += WgtBDPkt.ctrPktHex();

      // Lock_ID = (isPing) ? 6 : 7;
      // genCOBDPkt WgtBDPkt(ID_CTRL, row, col, BD_WGT, 7,
      //                  WGT_BASE_ADDR/4, WGT_BYTE_LENGTH/4,
      //                  //0,0,0,0,0,
      //                  8-1, 1-1, (WGT_BYTE_LENGTH/4/4/2), 4, 4-1,
      //                  Lock_ID);
      genCOBDPkt BiasBDPkt(ID_CTRL, row, col, BD_WGT + 6, 7,
                           (WGT_BASE_ADDR + wgt_wnd_bytes) / 4,
                           bias_wnd_bytes / 4,
                           // 0,0,0,0,0,
                           8 - 1, 1 - 1, (bias_wnd_bytes / 4 / 4 / 2), 4, 4 - 1,
                           Lock_ACQ_ID, Lock_REL_ID, 1, 1);
      ctrPktStr += BiasBDPkt.ctrPktStr();
      ctrPktHex += BiasBDPkt.ctrPktHex();
    }
  }

  // appendStr2File("./data/ctrl.txt", ctrPktHex);
  return ctrPktStr;
}

string ofm_core_set_window_macc(uint32_t ofm_wnd_bytes, int OWin1iter,
                                int OHin1iter, int OW_offset, int OCG_tile,
                                int OCG_offset) {
  // This can also be used for avgp/maxp/elewadd/elewmul/elews
  int row = 0, col = 0;
  // int BD_IMG;
  // int BD_WGT;
  int BD_OFM;
  // int IMG_BASE_ADDR;
  // int WGT_BASE_ADDR;
  int OFM_BASE_ADDR;
  int Lock_ACQ_ID;
  int Lock_REL_ID;

  string ctrPktStr;
  string ctrPktHex;

  for (int isPing = 1; isPing >= 0; isPing--) {
    // OFM DMA_BD
    // if (portidx == 1) {
    BD_OFM = (isPing) ? (BD_OFM_PING) :  // 4
                 (BD_OFM_PONG);          // 5
    // OFM_BASE_ADDR = (isPing) ? (OFM_BASE_ADDR_PING): //0x1000
    //                           (OFM_BASE_ADDR_PONG); //0x9000 ofm_wnd_bytes
    OFM_BASE_ADDR = (isPing) ? (0x4000 - ofm_wnd_bytes) :  // 0x1000
                        (0xC000 - ofm_wnd_bytes);  // 0x9000 ofm_wnd_bytes
    // OFM_BASE_ADDR = (isPing) ? (0x2000): //0x1000
    //                           (0x9000); //0x9000 ofm_wnd_bytes
    uint32_t ofm_output_bytes = ofm_wnd_bytes;
    if (OW_offset) {
      ofm_output_bytes = (OWin1iter - OW_offset) * OHin1iter * OCG_tile * 16;
    }
    // Lock_ID = (isPing) ? 4 : 5;
    Lock_ACQ_ID = 5;
    Lock_REL_ID = 4;
    genCOBDPkt OfmBDPkt(ID_CTRL, row, col, BD_OFM, 7, OFM_BASE_ADDR / 4,
                        ofm_output_bytes / 4,

                        (16 * OWin1iter / 4) - 1, 1 - 1, OHin1iter,
                        (16 * (OWin1iter - OW_offset)) / 4,
                        (16 * OWin1iter * OHin1iter) / 4 - 1,
                        // OWin1iter*OHin1iter*16/4-1, 1-1,
                        // ICG_tile, 16/4, 16/4-1,
                        Lock_ACQ_ID, Lock_REL_ID);
    ctrPktStr += OfmBDPkt.ctrPktStr();
    ctrPktHex += OfmBDPkt.ctrPktHex();
  }
  // appendStr2File("./data/ctrl.txt", ctrPktHex);
  return ctrPktStr;
}

string startPkts(int isPing, int type) {
  int row = 0, col = 0;
  int BD_IMG = (isPing) ? BD_IMG_PING   // 0
                        : BD_IMG_PONG;  // 1
  int BD_WGT = (isPing) ? BD_WGT_PING   // 2
                        : BD_WGT_PONG;  // 3
  int BD_OFM = (isPing) ? BD_OFM_PING   // 4
                        : BD_OFM_PONG;  // 5

  string ctrPktStr;
  string ctrPktHex;

  if (type == 0) {
    genBDStartQPkt BDStart(ID_CTRL, row, col, BD_IMG, IMG_CHANNEL, S2MM, 1);
    ctrPktStr += BDStart.to_string();
    ctrPktHex += BDStart.to_hex();
  } else if (type == 1) {
    genBDStartQPkt BDStart(ID_CTRL, row, col, BD_WGT, WGT_CHANNEL, S2MM, 1);
    ctrPktStr += BDStart.to_string();
    ctrPktHex += BDStart.to_hex();
  } else if (type == 2) {
    genBDStartQPkt BDStart(ID_CTRL, row, col, BD_OFM, OFM_CHANNEL, MM2S, 1);
    ctrPktStr += BDStart.to_string();
    ctrPktHex += BDStart.to_hex();
  } else if (type == 3) {
    genBDStartQPkt BDStart(ID_CTRL, row, col, BD_WGT + 6, WGT_CHANNEL, S2MM, 1);
    ctrPktStr += BDStart.to_string();
    ctrPktHex += BDStart.to_hex();
  } else {
    assert(((type >= 0) && (type <= 3)) && "startPkts type valid is wrong");
  }

  // appendStr2File("./data/ctrl.txt", ctrPktHex);
  return ctrPktStr;
}

string startMTPkts(int isPing, int type, int idx_mt_grp) {
  const int MT_WGT_CHN_CONVB = MT_WGT_CHN_CONV;
  int row = 0, col = 0;
  int MT_IMG_BD_CONV_IFM_H0 =
      (isPing) ? MT_IMG_BD_CONV_PING_IFM_H0 : MT_IMG_BD_CONV_PONG_IFM_H0;
  int MT_IMG_BD_CONV_IFM_H1 =
      (isPing) ? MT_IMG_BD_CONV_PING_IFM_H1 : MT_IMG_BD_CONV_PONG_IFM_H1;
  int MT_IMG_BD_CONV_OFM_H0 =
      (isPing) ? MT_IMG_BD_CONV_PING_OFM_H0 : MT_IMG_BD_CONV_PONG_OFM_H0;
  int MT_IMG_BD_CONV_OFM_H1 =
      (isPing) ? MT_IMG_BD_CONV_PING_OFM_H1 : MT_IMG_BD_CONV_PONG_OFM_H1;
  int MT_IMG_BD_MISC_IFM =
      (isPing) ? MT_IMG_BD_MISC_PING_IFM : MT_IMG_BD_MISC_PONG_IFM;
  int MT_IMG_BD_MISC_OFM =
      (isPing) ? MT_IMG_BD_MISC_PING_OFM : MT_IMG_BD_MISC_PONG_OFM;
  int MT_IMG_BD_LOAD = (isPing) ? MT_IMG_BD_LOAD_PING : MT_IMG_BD_LOAD_PONG;
  int MT_IMG_BD_SAVE = (isPing) ? MT_IMG_BD_SAVE_PING : MT_IMG_BD_SAVE_PONG;
  int MT_WGT_BD_CONV = (isPing) ? MT_WGT_BD_CONV_PING : MT_WGT_BD_CONV_PONG;
  int MT_WGT_BD_MISC = (isPing) ? MT_WGT_BD_MISC_PING : MT_WGT_BD_MISC_PONG;
  int MT_WGT_BD_LOAD_L1 =
      (isPing) ? MT_WGT_BD_LOAD_PING_L1 : MT_WGT_BD_LOAD_PONG_L1;
  int MT_WGT_BD_LOAD_L2 =
      (isPing) ? MT_WGT_BD_LOAD_PING_L2 : MT_WGT_BD_LOAD_PONG_L2;
  int MT_WGT_BD_CONVB = (isPing) ? MT_WGT_BD_CONVB_PING : MT_WGT_BD_CONVB_PONG;

  string ctrPktStr;
  string ctrPktHex;

  if (type == 0) {
    genBDStartQPkt BDStart(ID_CTRL, row, col, MT_IMG_BD_CONV_IFM_H0,
                           MT_IMG_CHN_CONV_IFM_H0, MM2S, 0);
    ctrPktStr += BDStart.to_string();
    ctrPktHex += BDStart.to_hex();

    genBDStartQPkt BDStart2(ID_CTRL, row, col, MT_IMG_BD_CONV_IFM_H0 + 6,
                            MT_IMG_CHN_CONV_IFM_H0, MM2S, 0);
    ctrPktStr += BDStart2.to_string();
    ctrPktHex += BDStart2.to_hex();
  }

  else if (type == 1) {
    genBDStartQPkt BDStart(ID_CTRL, row, col, MT_IMG_BD_CONV_IFM_H1,
                           MT_IMG_CHN_CONV_IFM_H1, MM2S, 0);
    ctrPktStr += BDStart.to_string();
    ctrPktHex += BDStart.to_hex();
  } else if (type == 2) {
    genBDStartQPkt BDStart(ID_CTRL, row, col, MT_IMG_BD_CONV_OFM_H0,
                           MT_IMG_CHN_CONV_OFM_H0, S2MM, 0);
    ctrPktStr += BDStart.to_string();
    ctrPktHex += BDStart.to_hex();
  } else if (type == 3) {
    genBDStartQPkt BDStart(ID_CTRL, row, col, MT_IMG_BD_CONV_OFM_H1,
                           MT_IMG_CHN_CONV_OFM_H1, S2MM, 0);
    ctrPktStr += BDStart.to_string();
    ctrPktHex += BDStart.to_hex();
  } else if (type == 4) {
    genBDStartQPkt BDStart(ID_CTRL, row, col, MT_IMG_BD_MISC_IFM,
                           MT_IMG_CHN_MISC_IFM, MM2S, 0);
    ctrPktStr += BDStart.to_string();
    ctrPktHex += BDStart.to_hex();
  } else if (type == 5) {
    genBDStartQPkt BDStart(ID_CTRL, row, col, MT_IMG_BD_MISC_OFM,
                           MT_IMG_CHN_MISC_OFM, S2MM, 0);
    ctrPktStr += BDStart.to_string();
    ctrPktHex += BDStart.to_hex();
  } else if (type == 6) {
    genBDStartQPkt BDStart(ID_CTRL, row, col, MT_IMG_BD_LOAD, MT_IMG_CHN_LOAD,
                           S2MM, 0);
    ctrPktStr += BDStart.to_string();
    ctrPktHex += BDStart.to_hex();
  } else if (type == 7) {
    genBDStartQPkt BDStart(ID_CTRL, row, col, MT_IMG_BD_SAVE, MT_IMG_CHN_SAVE,
                           MM2S, 0);
    ctrPktStr += BDStart.to_string();
    ctrPktHex += BDStart.to_hex();
  } else if (type == 8) {
    genBDStartQPkt BDStart(ID_CTRL, row, col, MT_WGT_BD_CONV, MT_WGT_CHN_CONV,
                           MM2S, 0, 2 - 1);
    ctrPktStr += BDStart.to_string();
    ctrPktHex += BDStart.to_hex();
    // genBDStartQPkt BDStart1(ID_CTRL, row, col, MT_WGT_BD_CONV + 6,
    // MT_WGT_CHN_CONV, MM2S, 0); ctrPktStr += BDStart1.to_string(); ctrPktHex
    // += BDStart1.to_hex(); }
  }

  else if (type == 9) {
    genBDStartQPkt BDStart(ID_CTRL, row, col, MT_WGT_BD_MISC, MT_WGT_CHN_MISC,
                           MM2S, 0);
    ctrPktStr += BDStart.to_string();
    ctrPktHex += BDStart.to_hex();
  } else if (type == 10) {
    genBDStartQPkt BDStart(ID_CTRL, row, col, MT_WGT_BD_LOAD_L1,
                           MT_WGT_CHN_LOAD_L1, S2MM, 0);
    ctrPktStr += BDStart.to_string();
    ctrPktHex += BDStart.to_hex();
  } else if (type == 11) {
    genBDStartQPkt BDStart(ID_CTRL, row, col, MT_WGT_BD_LOAD_L2,
                           MT_WGT_CHN_LOAD_L2, S2MM, 0);
    ctrPktStr += BDStart.to_string();
    ctrPktHex += BDStart.to_hex();
  } else if (type == 12) {
    genBDStartQPkt BDStart0(ID_CTRL, row, col, MT_WGT_BD_CONVB,
                            MT_WGT_CHN_CONVB, MM2S, 0, 2 - 1);
    ctrPktStr += BDStart0.to_string();
    ctrPktHex += BDStart0.to_hex();
  } else {
    assert(((type >= 0) && (type <= 12)) && "startPkts type valid is wrong");
  }

  // appendStr2File("./data/ctrl.txt", ctrPktHex);
  return ctrPktStr;
}

string updateCONVWgtBD(uint32_t wgt_wnd_bytes, int isPing) {
  int row = 0, col = 0;
  int BD_WGT;
  int WGT_BASE_ADDR;
  int Lock_ACQ_ID;
  int Lock_REL_ID;
  string ctrPktStr;
  string ctrPktHex;

  BD_WGT = (isPing) ? BD_WGT_PING :                // 2
               BD_WGT_PONG;                        // 3
  WGT_BASE_ADDR = (isPing) ? WGT_BASE_ADDR_PING :  // 0x4000
                      WGT_BASE_ADDR_PONG;          // 0xC000
  Lock_ACQ_ID = 2;
  Lock_REL_ID = 3;
  genCOBDPkt WgtBDPkt(ID_CTRL, row, col, BD_WGT, 1, WGT_BASE_ADDR / 4,
                      wgt_wnd_bytes / 4, 0, 0, 0, 0, 0, Lock_ACQ_ID,
                      Lock_REL_ID);
  ctrPktStr += WgtBDPkt.ctrPktStr();
  ctrPktHex += WgtBDPkt.ctrPktHex();

  // appendStr2File("./data/ctrl.txt", ctrPktHex);
  return ctrPktStr;
}

string setMACCImgMT2CO(int idx_mt_grp, int idx_mt_num, uint32_t img_bytes_size,
                       uint32_t mt_img_ifm_addr, int count, int pad_left,
                       int pad_top, int pad_right, int pad_bottom,
                       int IWin1Iter, int IHin1Iter, int src_h, int src_w,
                       int stride_w, int ICG, int ICG_tile, int Lock_Acq_ID,
                       int Lock_Rel_ID, bool BD_casc_fir, bool BD_casc_last,
                       int isPing) {
  const int ID_CTRL0 = ID_CTRL + idx_mt_grp;
  int row = 0, col = 0;
  int BD_IMG_IFM;
  int MT_IMG_IFM_BASE_ADDR;
  int hot_code = 15;

  string ctrPktStr;
  string ctrPktHex;

  // CONV IMG MT BD set to core
  BD_IMG_IFM =
      (isPing) ? (MT_IMG_BD_CONV_PING_IFM_H0) : (MT_IMG_BD_CONV_PONG_IFM_H0);
  MT_IMG_IFM_BASE_ADDR = (isPing) ? (MT_IMG_CONV_BASE_ADDR_IFM_PING)
                                  : (MT_IMG_CONV_BASE_ADDR_IFM_PONG);

  int lock_enable = 1;
  if (count > 0) {
    Lock_Rel_ID = 0;
    Lock_Acq_ID = 0;
    lock_enable = 0;
  }
  cout << "====" << IWin1Iter << "," << pad_left << "," << pad_right << ","
       << pad_top << "," << pad_bottom << endl;

  {
    int first_bd_pad_right =
        pad_right - stride_w > 0 ? pad_right - stride_w : 0;
    int secnd_bd_pad_left = pad_left - stride_w > 0 ? pad_left - stride_w : 0;
    genMTBDPkt ImgMT2COPkt0(
        ID_CTRL0, row, col, BD_IMG_IFM, hot_code, img_bytes_size / 4 / 2, 0,
        MT_IMG_IFM_BASE_ADDR + mt_img_ifm_addr / 4 - idx_mt_num * MT_SIZE,
        (16 * ICG_tile) / 4, 1 - 1, pad_left,
        IWin1Iter - pad_left - first_bd_pad_right, (16 * ICG) / 4 - 1, pad_top,
        IHin1Iter - pad_top - pad_bottom, (src_w * 16 * ICG) / 4 - 1,
        // pad_bottom, pad_right, 0, 16/4-1,
        pad_bottom, first_bd_pad_right, 0, 0,
        // isPing?1:-1, Lock_Rel_ID, 1, isPing?1:0, Lock_Acq_ID,
        0, Lock_Rel_ID, lock_enable, isPing ? 1 : 0, Lock_Acq_ID, 1);
    ctrPktStr += ImgMT2COPkt0.ctrPktStr();
    ctrPktHex += ImgMT2COPkt0.ctrPktHex();

    int secnd_bd_offset = stride_w - pad_left > 0 ? stride_w - pad_left : 0;
    genMTBDPkt ImgMT2COPkt1(
        ID_CTRL0, row, col, BD_IMG_IFM + 6, hot_code, img_bytes_size / 4 / 2, 0,
        MT_IMG_IFM_BASE_ADDR + mt_img_ifm_addr / 4 - idx_mt_num * MT_SIZE +
            (secnd_bd_offset) * (ICG * 16 / 4),
        (16 * ICG_tile) / 4, 1 - 1, secnd_bd_pad_left,
        IWin1Iter - pad_right - secnd_bd_pad_left, (16 * ICG) / 4 - 1, pad_top,
        IHin1Iter - pad_top - pad_bottom, (src_w * 16 * ICG) / 4 - 1,
        // pad_bottom, pad_right, 0, 16/4-1,
        pad_bottom, pad_right, 0, 16 / 4 - 1,
        // isPing?1:-1, Lock_Rel_ID, 1, isPing?0:1, Lock_Acq_ID,
        isPing ? 1 : -1, Lock_Rel_ID, 0, 0, Lock_Acq_ID, 1);
    ctrPktStr += ImgMT2COPkt1.ctrPktStr();
    ctrPktHex += ImgMT2COPkt1.ctrPktHex();
    /*
    genMTBDPkt ImgMT2COPkt0(ID_CTRL0, row, col, BD_IMG_IFM, hot_code,
                           img_bytes_size/4/2,
                           0,
    MT_IMG_IFM_BASE_ADDR+mt_img_ifm_addr/4-idx_mt_num*MT_SIZE, 16/4, 1-1,
                           pad_left, IWin1Iter - pad_left - first_bd_pad_right,
    (16*ICG)/4-1, pad_top, IHin1Iter - pad_top - pad_bottom, (src_w*16*ICG)/4-1,
                           //pad_bottom, pad_right, 0, 16/4-1,
                           pad_bottom, first_bd_pad_right, 0, 16/4-1,
                           //isPing?1:-1, Lock_Rel_ID, 1, isPing?1:0,
    Lock_Acq_ID, 0, Lock_Rel_ID, lock_enable , isPing?1:0, Lock_Acq_ID, 1);
    ctrPktStr += ImgMT2COPkt0.ctrPktStr();
    ctrPktHex += ImgMT2COPkt0.ctrPktHex();

    int secnd_bd_offset = stride_w - pad_left > 0? stride_w - pad_left : 0;
    genMTBDPkt ImgMT2COPkt1(ID_CTRL0, row, col, BD_IMG_IFM + 6, hot_code,
                           img_bytes_size/4/2,
                           0,
    MT_IMG_IFM_BASE_ADDR+mt_img_ifm_addr/4-idx_mt_num*MT_SIZE +
    (secnd_bd_offset)*(ICG*16/4), 16/4, 1-1, secnd_bd_pad_left, IWin1Iter -
    pad_right - secnd_bd_pad_left, (16*ICG)/4-1, pad_top, IHin1Iter - pad_top -
    pad_bottom, (src_w*16*ICG)/4-1,
                           //pad_bottom, pad_right, 0, 16/4-1,
                           pad_bottom, pad_right, 0, 16/4-1,
                           //isPing?1:-1, Lock_Rel_ID, 1, isPing?0:1,
    Lock_Acq_ID, isPing?1:-1, Lock_Rel_ID, 0, 0, Lock_Acq_ID, 1); ctrPktStr +=
    ImgMT2COPkt1.ctrPktStr(); ctrPktHex += ImgMT2COPkt1.ctrPktHex();
    */
  }

  // appendStr2File("./data/ctrl.txt", ctrPktHex);
  return ctrPktStr;
}
string setELEWADDImgMT2CO(int idx_mt_grp, int idx_mt_num,
                          uint32_t img_bytes_size, uint32_t mt_img_ifm_addr,
                          int count, int pad_left, int pad_top, int pad_right,
                          int pad_bottom, int IWin1Iter, int IHin1Iter,
                          int src_h, int src_w, int stride_w, int ICG,
                          int Lock_Acq_ID, int Lock_Rel_ID, bool BD_casc_fir,
                          bool BD_casc_last, int isPing) {
  const int ID_CTRL0 = ID_CTRL + idx_mt_grp;
  int row = 0, col = 0;
  int BD_IMG_IFM;
  int MT_IMG_IFM_BASE_ADDR;
  int hot_code = 15;

  string ctrPktStr;
  string ctrPktHex;

  // CONV IMG MT BD set to core
  BD_IMG_IFM =
      (isPing) ? (MT_IMG_BD_CONV_PING_IFM_H0) : (MT_IMG_BD_CONV_PONG_IFM_H0);
  MT_IMG_IFM_BASE_ADDR = (isPing) ? (MT_IMG_CONV_BASE_ADDR_IFM_PING)
                                  : (MT_IMG_CONV_BASE_ADDR_IFM_PONG);

  int lock_enable = 1;
  if (count > 0) {
    Lock_Rel_ID = 0;
    Lock_Acq_ID = 0;
    lock_enable = 0;
  }

  {
    genMTBDPkt ImgMT2COPkt0(
        ID_CTRL0, row, col, BD_IMG_IFM, hot_code, img_bytes_size / 4 / 2, 0,
        MT_IMG_IFM_BASE_ADDR + mt_img_ifm_addr / 4 - idx_mt_num * MT_SIZE,
        16 / 4, 1 - 1, pad_left, IWin1Iter, (16 * ICG) / 4 - 1, pad_top,
        IHin1Iter, (src_w * 16 * ICG) / 4 - 1,
        // pad_bottom, pad_right, 0, 16/4-1,
        pad_bottom, pad_right, 0, 16 / 4 - 1,
        // isPing?1:-1, Lock_Rel_ID, 1, isPing?1:0, Lock_Acq_ID,
        0, Lock_Rel_ID, lock_enable, isPing ? 1 : 0, Lock_Acq_ID, 1);
    ctrPktStr += ImgMT2COPkt0.ctrPktStr();
    ctrPktHex += ImgMT2COPkt0.ctrPktHex();

    genMTBDPkt ImgMT2COPkt1(
        ID_CTRL0, row, col, BD_IMG_IFM + 6, hot_code, img_bytes_size / 4 / 2, 0,
        MT_IMG_IFM_BASE_ADDR + mt_img_ifm_addr / 4 - idx_mt_num * MT_SIZE +
            src_h * src_w * (ICG * 16 / 4),
        16 / 4, 1 - 1, pad_left, IWin1Iter, (16 * ICG) / 4 - 1, pad_top,
        IHin1Iter, (src_w * 16 * ICG) / 4 - 1,
        // pad_bottom, pad_right, 0, 16/4-1,
        pad_bottom, pad_right, 0, 16 / 4 - 1,
        // isPing?1:-1, Lock_Rel_ID, 1, isPing?0:1, Lock_Acq_ID,
        isPing ? 1 : -1, Lock_Rel_ID, 0, 0, Lock_Acq_ID, 1);
    ctrPktStr += ImgMT2COPkt1.ctrPktStr();
    ctrPktHex += ImgMT2COPkt1.ctrPktHex();
  }

  // appendStr2File("./data/ctrl.txt", ctrPktHex);
  return ctrPktStr;
}
string setELEWSHIFTImgMT2CO(int idx_mt_grp, int idx_mt_num,
                            uint32_t img_bytes_size, uint32_t mt_img_ifm_addr,
                            int count, int pad_left, int pad_top, int pad_right,
                            int pad_bottom, int IWin1Iter, int IHin1Iter,
                            int src_h, int src_w, int stride_w, int ICG,
                            int Lock_Acq_ID, int Lock_Rel_ID, bool BD_casc_fir,
                            bool BD_casc_last, int isPing) {
  const int ID_CTRL0 = ID_CTRL + idx_mt_grp;
  int row = 0, col = 0;
  int BD_IMG_IFM;
  int MT_IMG_IFM_BASE_ADDR;
  int hot_code = 15;

  string ctrPktStr;
  string ctrPktHex;

  // CONV IMG MT BD set to core
  BD_IMG_IFM =
      (isPing) ? (MT_IMG_BD_CONV_PING_IFM_H0) : (MT_IMG_BD_CONV_PONG_IFM_H0);
  MT_IMG_IFM_BASE_ADDR = (isPing) ? (MT_IMG_CONV_BASE_ADDR_IFM_PING)
                                  : (MT_IMG_CONV_BASE_ADDR_IFM_PONG);

  // int lock_enable = 1;
  if (count > 0) {
    Lock_Rel_ID = 0;
    Lock_Acq_ID = 0;
    // lock_enable = 0;
  }
  {
    genMTBDPkt ImgMT2COPkt0(
        ID_CTRL0, row, col, BD_IMG_IFM, hot_code, img_bytes_size / 4, 0,
        MT_IMG_IFM_BASE_ADDR + mt_img_ifm_addr / 4 - idx_mt_num * MT_SIZE,
        16 / 4, 1 - 1, pad_left, IWin1Iter, (16 * ICG) / 4 - 1, pad_top,
        IHin1Iter, (src_w * 16 * ICG) / 4 - 1,
        // pad_bottom, pad_right, 0, 16/4-1,
        pad_bottom, pad_right, 0, 16 / 4 - 1, isPing ? 1 : -1, Lock_Rel_ID, 1,
        isPing ? 1 : 0, Lock_Acq_ID,
        // 0, Lock_Rel_ID, lock_enable , isPing?1:0, Lock_Acq_ID,
        1);
    ctrPktStr += ImgMT2COPkt0.ctrPktStr();
    ctrPktHex += ImgMT2COPkt0.ctrPktHex();
  }

  // appendStr2File("./data/ctrl.txt", ctrPktHex);
  return ctrPktStr;
}

string setMTfor1DPL(int idx_mt_grp, int idx_mt_num, uint32_t data_total_size,
                    uint32_t data_addr_offset, int hot_code, int Lock_Acq_ID,
                    int Lock_Rel_ID, bool BD_casc_fir, bool BD_casc_last,
                    int isPing, int type) {
  const int ID_CTRL0 = ID_CTRL + idx_mt_grp;
  int row = 0, col = 0;
  int BD{};
  int MT_BASE_ADDR{};
  int Acq_value{};
  int Rel_value{};

  string ctrPktStr;
  string ctrPktHex;

  // CONV IMG MT BD set PL to memt
  if (type == 0) {
    BD = (isPing) ? (MT_IMG_BD_LOAD_PING) : (MT_IMG_BD_LOAD_PONG);
    MT_BASE_ADDR =
        (isPing) ? (MT_IMG_LOAD_BASE_ADDR_PING) : (MT_IMG_LOAD_BASE_ADDR_PONG);
    Acq_value = isPing ? 0 : 1;
    Rel_value = isPing ? 1 : -1;
  } else if (type == 1) {
    BD = (isPing) ? (MT_IMG_BD_SAVE_PING) : (MT_IMG_BD_SAVE_PONG);
    MT_BASE_ADDR =
        (isPing) ? (MT_IMG_SAVE_BASE_ADDR_PING) : (MT_IMG_SAVE_BASE_ADDR_PONG);
    Acq_value = isPing ? 1 : 0;
    Rel_value = isPing ? 1 : -1;
  } else if (type == 2) {
    BD = (isPing) ? (MT_WGT_BD_LOAD_PING_L1) : (MT_WGT_BD_LOAD_PONG_L1);
    MT_BASE_ADDR =
        (isPing) ? (MT_WGT_LOAD_BASE_ADDR_PING) : (MT_WGT_LOAD_BASE_ADDR_PONG);
    Acq_value = isPing ? 0 : 1;
    Rel_value = isPing ? 1 : -1;
  } else {
    assert(((type >= 0) && (type <= 2)) && "setMTfor1DPL type valid is wrong");
  }

  genMTBDPkt BDforPLPkt(
      ID_CTRL0, row, col, BD, hot_code, data_total_size / 4, 0,
      MT_BASE_ADDR + data_addr_offset / 4 - idx_mt_num * MT_SIZE, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, Rel_value, Lock_Rel_ID, BD_casc_fir ? 1 : 0,
      Acq_value, Lock_Acq_ID, BD_casc_last);
  ctrPktStr += BDforPLPkt.ctrPktStr();
  ctrPktHex += BDforPLPkt.ctrPktHex();

  // appendStr2File("./data/ctrl.txt", ctrPktHex);
  return ctrPktStr;
}

string setMISCImgCO2MT(int idx_mt_grp, int idx_mt_num, uint32_t ofm_byte_size,
                       uint32_t mt_img_ofm_addr, int hot_code, int OHin1iter,
                       int OWin1iter, int OW_offset, int OCG, int OCG_tile,
                       int OCG_offset, int dst_w, int Lock_Acq_ID,
                       int Lock_Rel_ID, bool BD_casc_fir, bool BD_casc_last,
                       int isPing) {
  const int ID_CTRL0 = ID_CTRL + idx_mt_grp;
  int row = 0, col = 0;
  int BD_IMG_OFM;
  int MT_IMG_OFM_BASE_ADDR;

  int rel_value = 1;
  int acq_value = 0;

  string ctrPktStr;
  string ctrPktHex;

  // CONV IMG MT BD set to memt
  BD_IMG_OFM =
      (isPing) ? (MT_IMG_BD_CONV_PING_OFM_H0) : (MT_IMG_BD_CONV_PONG_OFM_H0);
  MT_IMG_OFM_BASE_ADDR = (isPing) ? (MT_IMG_CONV_BASE_ADDR_OFM_PING)
                                  : (MT_IMG_CONV_BASE_ADDR_OFM_PONG);
  if (BD_casc_fir && BD_casc_last) {
    rel_value = 1;
    acq_value = 0;
  } else if (BD_casc_fir) {
    rel_value = 0;
    acq_value = 0;
  } else if (BD_casc_last) {
    rel_value = 1;
    acq_value = 0;
  }

  genMTBDPkt ImgCO2MTPkt(
      ID_CTRL0, row, col, BD_IMG_OFM, hot_code, ofm_byte_size / 4, 0,
      MT_IMG_OFM_BASE_ADDR + mt_img_ofm_addr / 4 - idx_mt_num * MT_SIZE,
      // OCG_tile*16/4, 1-1,
      // 0, OWin1iter, (16*OCG)/4-1,
      // 0, 0,(dst_w*16*OCG)/4-1,
      // 0, 0, 0, 0,
      16 / 4, 1 - 1, 0, OWin1iter - OW_offset, (16 * OCG) / 4 - 1, 0, OHin1iter,
      (16 * OCG * dst_w) / 4 - 1, 0, 0, 0, 16 / 4 - 1, rel_value, Lock_Rel_ID,
      BD_casc_fir ? 1 : 0, acq_value, Lock_Acq_ID, BD_casc_last);

  ctrPktStr += ImgCO2MTPkt.ctrPktStr();
  ctrPktHex += ImgCO2MTPkt.ctrPktHex();

  // appendStr2File("./data/ctrl.txt", ctrPktHex);
  return ctrPktStr;
}

string setMiscWgtMT2CO(int idx_mt_grp, int idx_mt_num, uint32_t wgt_byte_size,
                       uint32_t mt_conv_wgt_addr, uint32_t bias_byte_size,
                       uint32_t mt_conv_bias_addr, int count, int kernel_size,
                       int ICG_tile, int OCG_tile, int ICG, int Lock_Acq_ID,
                       int Lock_Rel_ID, bool BD_casc_fir, bool BD_casc_last,
                       int isPing) {
  const int ID_CTRL0 = ID_CTRL + idx_mt_grp;
  int row = 0, col = 0;
  int BD_CONV_WGT;
  int BD_CONV_BIAS;
  int MT_CONV_WGT_BASE_ADDR;

  string ctrPktStr;
  string ctrPktHex;

  // CONV WGT MT BD set to core
  BD_CONV_WGT = (isPing) ? (MT_WGT_BD_CONV_PING) : (MT_WGT_BD_CONV_PONG);
  BD_CONV_BIAS = (isPing) ? (MT_WGT_BD_CONVB_PING) : (MT_WGT_BD_CONVB_PONG);
  MT_CONV_WGT_BASE_ADDR =
      (isPing) ? (MT_WGT_CONV_BASE_ADDR_PING) : (MT_WGT_CONV_BASE_ADDR_PONG);
  BD_CONV_BIAS = (isPing) ? (MT_WGT_BD_CONVB_PING) : (MT_WGT_BD_CONVB_PONG);

  int lock_enable = 1;
  int lock_rel_value = isPing ? 1 : -1;
  // int lock_rel_value = 0;
  int lock_acq_value = isPing ? 1 : 0;
  if (count >= 1) {
    Lock_Rel_ID = 0;
    Lock_Acq_ID = 0;
    lock_enable = 0;
    lock_rel_value = 0;
    lock_acq_value = 0;
  }

  // notice: here wgt and bias read MT in order so wrap and stepsize params are
  // not needed.
  int hot_code = 9;
  genMTBDPkt WgtMT2COPkt0(
      ID_CTRL0, row, col, BD_CONV_WGT, hot_code, wgt_byte_size / 4, 0,
      MT_CONV_WGT_BASE_ADDR + mt_conv_wgt_addr / 4 - idx_mt_num * MT_SIZE, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, Lock_Rel_ID, lock_enable, lock_acq_value,
      Lock_Acq_ID, 1);

  ctrPktStr += WgtMT2COPkt0.ctrPktStr();
  ctrPktHex += WgtMT2COPkt0.ctrPktHex();

  genMTBDPkt BiasMT2COPkt0(
      ID_CTRL0, row, col, BD_CONV_BIAS, hot_code, bias_byte_size / 4, 0,
      MT_CONV_WGT_BASE_ADDR + mt_conv_bias_addr / 4 - idx_mt_num * MT_SIZE, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      // 0, 0, 0, 0, 0,
      lock_rel_value, Lock_Rel_ID, 0, 0, Lock_Acq_ID, 1);
  ctrPktStr += BiasMT2COPkt0.ctrPktStr();
  ctrPktHex += BiasMT2COPkt0.ctrPktHex();

  // appendStr2File("./data/ctrl.txt", ctrPktHex);
  return ctrPktStr;
}

}  // namespace alu
}  // namespace xv2dpu
}  // namespace cosim
