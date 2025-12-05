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
#include "MEUtil.hpp"

void MEUtil::appendImg2HexFile(const string& fname, const char* data, int size,
                               int line_size) {
  UNI_LOG_CHECK(size > 1, SIM_OUT_OF_RANGE)
      << " Packing data to ME, port with need > 1" << endl;
  std::ofstream f;
  f.open(fname, std::ios::app);

  Util::ChkOpen(f, fname);
  int line_num = size / line_size;
  int line_left = size % line_size;
  line_num += (line_left == 0) ? 0 : 1;

  // write line_num lines
  for (int i = 0; i < line_num - 1; i++) {
    f << "TDATA : ";
    for (int j = line_size - 1; j >= 0; j--) {
      char tmp = data[i * line_size + j];
      uint8_t tmp_u8 = static_cast<uint8_t>(tmp);
      uint32_t tmp_u32 = static_cast<uint32_t>(tmp_u8);
      f << hex << setw(2) << setfill('0') << tmp_u32;
    }
    f << endl;
  }
  f << "TLAST : ";
  // handle last line, it's special
  line_left = (line_left == 0) ? line_size : line_left;
  for (int j = line_size - 1; j >= 0; j--) {
    char tmp = 0;
    tmp = (j >= line_left) ? 0 : data[(line_num - 1) * line_size + j];
    uint8_t tmp_u8 = static_cast<uint8_t>(tmp);
    uint32_t tmp_u32 = static_cast<uint32_t>(tmp_u8);
    f << hex << setw(2) << setfill('0') << tmp_u32;
  }
  f << endl;
  f.close();
}

string MEUtil::sendCtrlPkts(int portidx, int isPing, int isrltPing,
                            uint32_t img_wnd_bytes, uint32_t wgt_wnd_bytes,
                            uint32_t rlt_wnd_bytes) {
  unsigned int row = 0, col = 0;
  unsigned int BD_IMG;
  unsigned int BD_WGT;
  // unsigned int LOCK_BD_IMG;
  unsigned int LOCK_BD_WGT;
  unsigned int IMG_BASE_ADDR;
  bool AorB = MECfg::Instance().AorB();
  unsigned int IMG_CHANNEL = MECfg::Instance().IMG_CHANNEL();
  unsigned int WGT_CHANNEL = MECfg::Instance().WGT_CHANNEL();
  unsigned int S2MM = MECfg::S2MM;

  if (isPing == 1) {
    BD_IMG = MECfg::Instance().BD_IMG_PING();
    BD_WGT = MECfg::Instance().BD_WGT_PING();
    //    LOCK_BD_IMG = MECfg::Instance().LOCK_BD_IMG_PING();
    LOCK_BD_WGT = MECfg::Instance().LOCK_BD_WGT_PING();
    IMG_BASE_ADDR = MECfg::Instance().IMG_BASE_ADDR_PING();
  } else {
    BD_IMG = MECfg::Instance().BD_IMG_PONG();
    BD_WGT = MECfg::Instance().BD_WGT_PONG();
    //  LOCK_BD_IMG = MECfg::Instance().LOCK_BD_IMG_PONG();
    LOCK_BD_WGT = MECfg::Instance().LOCK_BD_WGT_PONG();
    IMG_BASE_ADDR = MECfg::Instance().IMG_BASE_ADDR_PONG();
  }
  // d.3.1 BD Ctrl Packet
  string ctrPktStr;
  string ctrPktHex;
  unsigned int ID_CTRL = MECfg::Instance().ID_CTRL();

  /*
  if((portidx%4)== 1 || (portidx%4)== 2){
     unsigned int BD_RLT;
     //unsigned int LOCK_BD_RLT;
     if(isrltPing == 1){
         BD_RLT = MECfg::Instance().BD_RLT_PING();
         //LOCK_BD_RLT = MECfg::Instance().LOCK_BD_RLT_PING();
     }else{
         BD_RLT = MECfg::Instance().BD_RLT_PONG();
         //LOCK_BD_RLT = MECfg::Instance().LOCK_BD_RLT_PONG();
     }
     //rlt Length: streamid, BDid, length, AorB, enable_pkt
     ChangeBDLengthPkt RltBDLen(ID_CTRL, 0, 0, BD_RLT, rlt_wnd_bytes/4, AorB,
  0); ctrPktStr += RltBDLen.to_string();
  }
  */
  // img Length: streamid, BDid, length, AorB, enable_pkt
  ChangeBDLengthPkt ImgBDLenPkt(ID_CTRL, row, col, BD_IMG, img_wnd_bytes / 4,
                                AorB, 0);
  ctrPktStr += ImgBDLenPkt.to_string();
  ctrPktHex += ImgBDLenPkt.to_hex();

  // wgt base addr: streamid, BDid, base_addr, lock
  ChangeBDBaseAddrPkt WgtBDBaseAddr(ID_CTRL, row, col, BD_WGT,
                                    IMG_BASE_ADDR + img_wnd_bytes / 4 + 1,
                                    LOCK_BD_WGT);  // add 1-word for align
  ctrPktStr += WgtBDBaseAddr.to_string();
  ctrPktHex += WgtBDBaseAddr.to_hex();

  // wgt Length
  ChangeBDLengthPkt WgtBDLenPkt(ID_CTRL, row, col, BD_WGT, wgt_wnd_bytes / 4,
                                AorB, 0);
  ctrPktStr += WgtBDLenPkt.to_string();
  ctrPktHex += WgtBDLenPkt.to_hex();
  // start Queue: streamid, BDid, channel, S2MMorMM2S
  ChangeBDStartQueuePkt ImgBDStart(ID_CTRL, row, col, BD_IMG, IMG_CHANNEL,
                                   S2MM);
  ctrPktStr += ImgBDStart.to_string();
  ctrPktHex += ImgBDStart.to_hex();
  ChangeBDStartQueuePkt WgtBDStart(ID_CTRL, row, col, BD_WGT, WGT_CHANNEL,
                                   S2MM);
  ctrPktStr += WgtBDStart.to_string();
  ctrPktHex += WgtBDStart.to_hex();
  //     appendStr2File("./data/ctrl.txt", ctrPktHex);
  return ctrPktStr;
}

void MEUtil::startRltPkts(int isPing, uint32_t rlt_wnd_bytes,
                          vector<string>& retstr, int useRltB) {
  unsigned int BD_RLT;
  // unsigned int LOCK_BD_RLT;
  unsigned int ID_CTRL = MECfg::Instance().ID_CTRL();
  unsigned int RLT_CHANNEL = MECfg::Instance().RLT_CHANNEL();
  unsigned int MM2S = MECfg::MM2S;
  // unsigned int AorB = MECfg::Instance().AorB();

  if (isPing == 1) {
    BD_RLT = (useRltB == 0) ? MECfg::Instance().BD_RLT_PING()
                            : MECfg::Instance().BD_RLT_PING_B();
    // LOCK_BD_RLT = MECfg::Instance().LOCK_BD_RLT_PING();
  } else {
    BD_RLT = (useRltB == 0) ? MECfg::Instance().BD_RLT_PONG()
                            : MECfg::Instance().BD_RLT_PONG_B();
    // LOCK_BD_RLT = MECfg::Instance().LOCK_BD_RLT_PONG();
  }
  // d.3.1 BD Ctrl Packet
  string ctrPktStr;
  string ctrPktHex;
  // rlt Length: streamid, BDid, length, AorB, enable_pkt
  // ChangeBDLengthPkt RltBDLen(ID_CTRL, 0, 0, BD_RLT, rlt_wnd_bytes/4, AorB,
  // 0); ctrPktStr += RltBDLen.to_string(); ctrPktHex += RltBDLen.to_hex(); start
  // Queue: streamid, BDid, channel, S2MMorMM2S
  ChangeBDStartQueuePkt RltBDStart(ID_CTRL, 0, 0, BD_RLT, RLT_CHANNEL, MM2S);
  ctrPktStr += RltBDStart.to_string();
  ctrPktHex += RltBDStart.to_hex();
  for (auto& s : retstr) {
    s.clear();
  }
  retstr[0] = ctrPktStr;
  retstr[1] = ctrPktHex;
}

void MEUtil::setDataMem(unsigned int addr, int32_t* param, int len,
                        vector<string>& retstr) {
  unsigned int ID_CTRL = MECfg::Instance().ID_CTRL();
  unsigned int ROW = 0;
  unsigned int COL = 0;
  unsigned int BASE_ADDR = addr;
  unsigned int MAX_WORD_PER_PKT = 2;  // MECfg::MAX_WORD_PER_PKT;
  unsigned int addrIncr = MAX_WORD_PER_PKT * 0x04;
  unsigned int pkt_num = len / MAX_WORD_PER_PKT;  // max 4 word
  unsigned int last_pkt_word = len % MAX_WORD_PER_PKT;
  pkt_num = (last_pkt_word == 0) ? pkt_num : (pkt_num + 1);
  last_pkt_word = (last_pkt_word == 0) ? MAX_WORD_PER_PKT : last_pkt_word;
  string paramStr;
  string hexStr;
  for (auto i = 0U; i < pkt_num - 1; i++) {
    SetParamPkt paramPkt(ID_CTRL, ROW, COL, BASE_ADDR + i * addrIncr,
                         param + i * MAX_WORD_PER_PKT, MAX_WORD_PER_PKT);
    paramStr += paramPkt.to_string();
    hexStr += paramPkt.to_hex();
  }
  /*
  SetParamPkt paramPkt(ID_CTRL, ROW,  COL,  BASE_ADDR + (pkt_num-1)*addrIncr,
                                  param + (pkt_num-1)*MAX_WORD_PER_PKT,
  last_pkt_word);
  */
  int32_t lastpkt[2];
  lastpkt[0] = *(param + ((pkt_num - 1) * MAX_WORD_PER_PKT) % len);
  lastpkt[1] = *(param + ((pkt_num - 1) * MAX_WORD_PER_PKT + 1) % len);
  SetParamPkt paramPkt(ID_CTRL, ROW, COL, BASE_ADDR + (pkt_num - 1) * addrIncr,
                       lastpkt, 2);
  paramStr += paramPkt.to_string();
  hexStr += paramPkt.to_hex();
  for (auto& s : retstr) {
    s.clear();
  }
  retstr[0] = paramStr;
  retstr[1] = hexStr;
}

void MEUtil::appendStr2File(const string& filename, const string& data) {
  std::ofstream f;
  f.open(filename, std::ios::app);
  f << data;
  f.close();
}

void MEUtil::setBDAddrLen(int portidx, uint32_t img_wnd_bytes,
                          uint32_t wgt_wnd_bytes, uint32_t rlt_wnd_bytes,
                          uint32_t rlt_wnd_bytes_offset, vector<string>& retstr,
                          int useRltB) {
  unsigned int row = 0, col = 0;
  unsigned int BD_IMG;
  unsigned int BD_WGT;
  unsigned int LOCK_BD_WGT;
  unsigned int IMG_BASE_ADDR;
  bool AorB = MECfg::Instance().AorB();
  unsigned int BD_RLT;
  unsigned int LOCK_BD_RLT;

  string ctrPktStr;
  string ctrPktHex;
  unsigned int ID_CTRL = MECfg::Instance().ID_CTRL();
  // img Length: streamid, BDid, length, AorB, enable_pkt
  for (int isPing = 1; isPing >= 0; isPing--) {
    if (useRltB == 0) {
      BD_IMG = (isPing) ? (MECfg::Instance().BD_IMG_PING())
                        : (MECfg::Instance().BD_IMG_PONG());
    } else {
      BD_IMG = (isPing) ? (MECfg::Instance().BD_IMG_PING_B())
                        : (MECfg::Instance().BD_IMG_PONG_B());
    }
    ChangeBDLengthPkt ImgBDLenPkt(ID_CTRL, row, col, BD_IMG, img_wnd_bytes / 4,
                                  AorB, 0);
    ctrPktStr += ImgBDLenPkt.to_string();
    ctrPktHex += ImgBDLenPkt.to_hex();
  }
  // wgt base addr: streamid, BDid, base_addr, lock
  for (int isPing = 1; isPing >= 0; isPing--) {
    if (isPing == 1) {
      BD_WGT = MECfg::Instance().BD_WGT_PING();
      LOCK_BD_WGT = MECfg::Instance().LOCK_BD_WGT_PING();
      IMG_BASE_ADDR = MECfg::Instance().IMG_BASE_ADDR_PING();
    } else {
      BD_WGT = MECfg::Instance().BD_WGT_PONG();
      LOCK_BD_WGT = MECfg::Instance().LOCK_BD_WGT_PONG();
      IMG_BASE_ADDR = MECfg::Instance().IMG_BASE_ADDR_PONG();
    }
    ChangeBDBaseAddrPkt WgtBDBaseAddr(ID_CTRL, row, col, BD_WGT,
                                      IMG_BASE_ADDR + img_wnd_bytes / 4 + 1,
                                      LOCK_BD_WGT);  // add 1-word for align
    ctrPktStr += WgtBDBaseAddr.to_string();
    ctrPktHex += WgtBDBaseAddr.to_hex();
  }

  // wgt Length
  for (int isPing = 1; isPing >= 0; isPing--) {
    BD_WGT = (isPing) ? (MECfg::Instance().BD_WGT_PING())
                      : (MECfg::Instance().BD_WGT_PONG());
    ChangeBDLengthPkt WgtBDLenPkt(ID_CTRL, row, col, BD_WGT, wgt_wnd_bytes / 4,
                                  AorB, 0);
    ctrPktStr += WgtBDLenPkt.to_string();
    ctrPktHex += WgtBDLenPkt.to_hex();
  }

  // rlt, even core 0 not use ofm, set it for save rtl logic
  //   if(portidx== 1 || portidx== 2){
  // base addr
  for (int isPing = 1; isPing >= 0; isPing--) {
    if (isPing == 1) {
      BD_RLT = (useRltB == 0) ? MECfg::Instance().BD_RLT_PING()
                              : MECfg::Instance().BD_RLT_PING_B();
      LOCK_BD_RLT = (useRltB == 0) ? MECfg::Instance().LOCK_BD_RLT_PING()
                                   : MECfg::Instance().LOCK_BD_RLT_PING_B();
      IMG_BASE_ADDR = MECfg::Instance().IMG_BASE_ADDR_PING();
    } else {
      BD_RLT = (useRltB == 0) ? MECfg::Instance().BD_RLT_PONG()
                              : MECfg::Instance().BD_RLT_PONG_B();
      LOCK_BD_RLT = (useRltB == 0) ? MECfg::Instance().LOCK_BD_RLT_PONG()
                                   : MECfg::Instance().LOCK_BD_RLT_PONG_B();
      IMG_BASE_ADDR = MECfg::Instance().IMG_BASE_ADDR_PONG();
    }
    ChangeBDBaseAddrPkt RltBDBaseAddr(ID_CTRL, 0, 0, BD_RLT,
                                      IMG_BASE_ADDR + rlt_wnd_bytes_offset / 4,
                                      LOCK_BD_RLT, false);
    ctrPktStr += RltBDBaseAddr.to_string();
    ctrPktHex += RltBDBaseAddr.to_hex();
  }
  // length
  for (int isPing = 1; isPing >= 0; isPing--) {
    if (useRltB == 0) {
      BD_RLT = (isPing) ? (MECfg::Instance().BD_RLT_PING())
                        : (MECfg::Instance().BD_RLT_PONG());
    } else {
      BD_RLT = (isPing) ? (MECfg::Instance().BD_RLT_PING_B())
                        : (MECfg::Instance().BD_RLT_PONG_B());
    }
    // rlt Length: streamid, BDid, length, AorB, enable_pkt
    ChangeBDLengthPkt RltBDLen(ID_CTRL, 0, 0, BD_RLT, rlt_wnd_bytes / 4, AorB,
                               0);
    ctrPktStr += RltBDLen.to_string();
    ctrPktHex += RltBDLen.to_hex();
  }
  //    }
  for (auto& s : retstr) {
    s.clear();
  }
  retstr[0] = ctrPktStr;
  retstr[1] = ctrPktHex;
}

void MEUtil::startInputPkts(int isPing, vector<string>& retstr, int useRltB) {
  unsigned int row = 0, col = 0;
  unsigned int BD_IMG;
  unsigned int BD_WGT;
  unsigned int IMG_CHANNEL = MECfg::Instance().IMG_CHANNEL();
  unsigned int WGT_CHANNEL = MECfg::Instance().WGT_CHANNEL();
  unsigned int S2MM = MECfg::S2MM;

  if (isPing == 1) {
    BD_IMG = useRltB == 0 ? MECfg::Instance().BD_IMG_PING()
                          : MECfg::Instance().BD_IMG_PING_B();
    BD_WGT = MECfg::Instance().BD_WGT_PING();
  } else {
    BD_IMG = useRltB == 0 ? MECfg::Instance().BD_IMG_PONG()
                          : MECfg::Instance().BD_IMG_PONG_B();
    BD_WGT = MECfg::Instance().BD_WGT_PONG();
  }
  unsigned int ID_CTRL = MECfg::Instance().ID_CTRL();
  string ctrPktStr;
  string ctrPktHex;
  // start Queue: streamid, BDid, channel, S2MMorMM2S
  ChangeBDStartQueuePkt ImgBDStart(ID_CTRL, row, col, BD_IMG, IMG_CHANNEL,
                                   S2MM);
  ctrPktStr += ImgBDStart.to_string();
  ctrPktHex += ImgBDStart.to_hex();
  ChangeBDStartQueuePkt WgtBDStart(ID_CTRL, row, col, BD_WGT, WGT_CHANNEL,
                                   S2MM);
  ctrPktStr += WgtBDStart.to_string();
  ctrPktHex += WgtBDStart.to_hex();
  for (auto& s : retstr) {
    s.clear();
  }
  retstr[0] = ctrPktStr;
  retstr[1] = ctrPktHex;
}

uint64_t MEUtil::calCycle(const uint32_t out_loop, const uint32_t inner_loop,
                          const uint32_t exec_type) {
  uint64_t calTime =0;
  uint64_t inner_overhead = XVDPU_REG_PREP;
  switch (exec_type) {
    case MECfg::ET_B_IN2DM:
      inner_overhead += XVDPU_SAVE_INTER;
      break;
    case MECfg::ET_B_INaDM2DM:
      inner_overhead += XVDPU_LOAD_INTER + XVDPU_SAVE_INTER;
      break;
    case MECfg::ET_B_INaDMaCASC2OUT:
      inner_overhead += XVDPU_LOAD_INTER + XVDPU_CASC_TRANS;
      break;
    case MECfg::ET_B_INaCASC2OUT:
    case MECfg::ET_B_INaCASC2OUT_IL2:
      inner_overhead += XVDPU_CASC_TRANS;
      break;
    case MECfg::ET_B_INaDMaCASC2OUT_LK:
      inner_overhead += XVDPU_LOAD_INTER + XVDPU_CASC_TRANS_LK;
      break;
    case MECfg::ET_B_INaCASC2OUT_LK:
    case MECfg::ET_B_INaCASC2OUT_IL2_LK:
      inner_overhead += XVDPU_CASC_TRANS_LK;
      break;
    case MECfg::ET_B_IN2DM_OLNP:
    case MECfg::ET_B_INaCASC2OUT_OL1:
    case MECfg::ET_B_INaDM2DM_OLNP:
    case MECfg::ET_B_INaCASC2OUT_OL1_LK:
    case MECfg::ET_B_INaCASC2OUT_IL1:
    case MECfg::ET_B_INaCASC2OUT_IL1_LK:
      inner_overhead += XVDPU_CASC_TRANS + 1;
      break;
    default:
      UNI_LOG_CHECK(0, SIM_OUT_OF_RANGE)
          << "unsupported exec_type " << exec_type << endl;
      abort();
  }
  if(inner_loop >=2 ){
      calTime =
      XVDPU_ACQ_WIN +
      out_loop * ((inner_loop - 2) * XVDPU_INNER_CAL + inner_overhead);
  }
  else{
      calTime =
      XVDPU_ACQ_WIN +
      out_loop * (inner_overhead);
  }

  return calTime;
}

void MEUtil::initRltB(vector<string>& retstr) {
  string ctrPktStr;
  string ctrPktHex;
  unsigned int ID_CTRL = MECfg::Instance().ID_CTRL();
  SetBDXYPkt xypkt6 =
      SetBDXYPkt(ID_CTRL, 0, 0, MECfg::Instance().BD_RLT_PING_B());
  SetBDXYPkt xypkt7 =
      SetBDXYPkt(ID_CTRL, 0, 0, MECfg::Instance().BD_RLT_PONG_B());
  ctrPktStr += xypkt6.to_string();
  ctrPktStr += xypkt7.to_string();
  ctrPktHex += xypkt6.to_hex();
  ctrPktHex += xypkt7.to_hex();
  for (auto& s : retstr) {
    s.clear();
  }
  retstr[0] = ctrPktStr;
  retstr[1] = ctrPktHex;
}
void MEUtil::initIfmB(vector<string>& retstr) {
  string ctrPktStr;
  string ctrPktHex;
  unsigned int ID_CTRL = MECfg::Instance().ID_CTRL();
  unsigned int BD_IMG_B_ping = MECfg::Instance().BD_IMG_PING_B();
  unsigned int BD_IMG_B_pong = MECfg::Instance().BD_IMG_PONG_B();
  ChangeBDBaseAddrPkt IfmBDBaseAddr_Bping(
      ID_CTRL, 0, 0, BD_IMG_B_ping, MECfg::Instance().IMG_BASE_ADDR_PING(),
      MECfg::Instance().LOCK_BD_IMG_PING());
  ChangeBDBaseAddrPkt IfmBDBaseAddr_Bpong(
      ID_CTRL, 0, 0, BD_IMG_B_pong, MECfg::Instance().IMG_BASE_ADDR_PONG(),
      MECfg::Instance().LOCK_BD_IMG_PONG());
  ctrPktStr += IfmBDBaseAddr_Bping.to_string();
  ctrPktStr += IfmBDBaseAddr_Bpong.to_string();
  ctrPktHex += IfmBDBaseAddr_Bping.to_hex();
  ctrPktHex += IfmBDBaseAddr_Bpong.to_hex();
  SetBDXYPkt xypkt8 = SetBDXYPkt(ID_CTRL, 0, 0, BD_IMG_B_ping);
  SetBDXYPkt xypkt9 = SetBDXYPkt(ID_CTRL, 0, 0, BD_IMG_B_pong);
  ctrPktStr += xypkt8.to_string();
  ctrPktStr += xypkt9.to_string();
  ctrPktHex += xypkt8.to_hex();
  ctrPktHex += xypkt9.to_hex();
  for (auto& s : retstr) {
    s.clear();
  }
  retstr[0] = ctrPktStr;
  retstr[1] = ctrPktHex;
}

void MEUtil::appendTlastList(const string& fname,
                             const vector<uint64_t> execlist) {
  UNI_LOG_CHECK(execlist.size() >= 1, SIM_OUT_OF_RANGE);
  std::ofstream f;
  f.open(fname, std::ios::app);
  for (auto i = 0U; i < execlist.size(); i++) {
    f << "TLAST : " << hex << execlist[i] << "\n";
  }
  f.close();
}

const unsigned int MECfg::S2MM = 0;
const unsigned int MECfg::MM2S = 1;
const unsigned int MECfg::MAX_WORD_PER_PKT = 4;
const string MECfg::fn_img_prefix = "aPr_b0_";
const string MECfg::fn_wgt_prefix = "bPr_";
const string MECfg::fn_rlt_prefix = "cPr_b0_";
const string MECfg::fn_inter_prefix = "inter_b0_";
const string MECfg::fn_tlast = "tlast.txt";

void MECfg::init() {
  bufPing_ = 1;
  rltPing_ = 1;
  rltPing_B_ = 1;
  useRltB_ = 1;
  setXYRltB_ = 1;
  setIfmB_ = 1;
  num_weights_port_ = 4;
  img_port_width_ = 2;
  weights_port_width_ = 4;
  rlt_port_width_ = 2;

  ID_CTRL_ = 1;

  // img startBD, channel, lockID
  BD_IMG_PING_ = 0;
  BD_IMG_PONG_ = 1;
  BD_IMG_PING_B_ = 8;
  BD_IMG_PONG_B_ = 9;
  IMG_CHANNEL_ = 0;
  LOCK_BD_IMG_PING_ = 0;
  LOCK_BD_IMG_PONG_ = 1;

  // weights startBD, channel, lockID
  BD_WGT_PING_ = 2;
  BD_WGT_PONG_ = 3;
  WGT_CHANNEL_ = 1;
  LOCK_BD_WGT_PING_ = 2;
  LOCK_BD_WGT_PONG_ = 3;

  // out startBD, channel, lockID
  BD_RLT_PING_ = 4;
  BD_RLT_PONG_ = 5;
  RLT_CHANNEL_ = 0;
  LOCK_BD_RLT_PING_ = 4;
  LOCK_BD_RLT_PONG_ = 5;

  BD_RLT_PING_B_ = 6;
  BD_RLT_PONG_B_ = 7;
  RLT_CHANNEL_B_ = 0;
  LOCK_BD_RLT_PING_B_ = 4;
  LOCK_BD_RLT_PONG_B_ = 5;

  IMG_BASE_ADDR_PING_ = 0x0000;  // 0x0000/4, need 32-bits aligned address
  IMG_BASE_ADDR_PONG_ = 0x1000;  // 0x4000/4

  AorB_ = false;

  ITER_PARAM_WORD_SIZE_ = 3;
  LAYER_PARAM_WORD_SIZE_ = 16;
  MAX_WLUT_WORD_SIZE_ = 64;
  // system 0x1000, <stacksize,paddr>:<1024, 0x1420> <1536, 0x1620>
  // system 0x1800, <stacksize,paddr>:<1024, 0x1C20>
  PARAM_BASE_ADDR_PING_ = 0x7CC0;
  PARAM_BASE_ADDR_PONG_ = 0x7C80;
  WLUT_BASE_ADDR_PING_ = 0x7E00;
  WLUT_BASE_ADDR_PONG_ = 0x7D00;
}
string uint2hex(uint32_t* tmp, int len, int WIDTH) {
  assert(WIDTH > 0);
  stringstream ss;
  ss.clear();
  ss.str("");
  int row = len/WIDTH;
  row += (len%WIDTH == 0) ? 0 : 1;
  for(int i=0; i<row-1; i++){
    ss<<"TDATA : ";
    int row_base = i*WIDTH;
    for(int j=0; j<WIDTH;j++){
      ss<<hex<<setw(8)<<setfill('0')<<tmp[row_base+3-j];
    }
    ss<<"\n";
  }
  ss<<"TLAST : ";
  int mod = len%WIDTH;
  mod = (mod == 0) ? WIDTH : mod;
  for(int i=0; i<WIDTH-mod; i++){
    ss<<hex<<setw(8)<<setfill('0')<<0;
  }
  for(int i=0; i<mod; i++){
    ss<<hex<<setw(8)<<setfill('0')<<tmp[len-i-1];
  }
  ss<<"\n";
  return ss.str();
}
string uint2str(uint32_t* pkts, int len, PKT_TYPE type, int WIDTH) {
  assert(type == PKT_TYPE_CTRL || type == PKT_TYPE_DATA);
  assert(WIDTH > 0);
  stringstream ss;
  ss.clear();
  ss.str("");
  int row = len/WIDTH;
  row += (len%WIDTH == 0) ? 0 : 1;
  for(int i=0; i<row; i++){
    if(i == row-1) ss<<"TLAST"<<endl;
    for(int j=0; j<WIDTH; j++){
      int idx = i*WIDTH+j;
      if(idx < len){
        /*
        if(type == PKT_TYPE_CTRL) ss<<pkts[idx]<<" ";
        else if(type == PKT_TYPE_DATA) ss<<(int32_t)pkts[idx]<<" ";
        */
        ss<<(int32_t)pkts[idx]<<" ";
      }
    }
    ss<<endl;
  }
  return ss.str();
}
