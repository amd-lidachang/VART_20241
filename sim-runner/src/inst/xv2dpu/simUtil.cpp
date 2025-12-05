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

#include "simUtil.hpp"
#include "SimCfg.hpp"
#include <iomanip>
#include <dirent.h>
#include <sys/stat.h>

std::string getLoadIfmDumpFileName(const int batch, const int mtRowIndex,
                                   const int mtColumnIndex) {
  UNI_LOG_CHECK(mtColumnIndex < SimCfgProxy::Params().LOAD_IFM_PARALLEL, SIM_CFG_FMT_NO_MATCH);

  std::string dataName = "ifm";
  std::string batchName = "b" + std::to_string(batch);
  std::string mtRowName = (0 == mtRowIndex) ? "c0" : "c1";
  std::string mtColName = "h23";
  // when LOAD_IFM_PARALLEL == 1, only mt h23 is involved, but mt h01 is also
  // visited, because LOAD_IFM_PARALLEL here indicates which channel used, not
  // mt used, the channel of mt h23 has the ability to visit mt h01
  if (2 == SimCfgProxy::Params().LOAD_IFM_PARALLEL) {
    // exchange the img dump file name
    // https://xilinx.slack.com/archives/D02UCR1PJ01/p1661137334363879
    mtColName = (0 == mtColumnIndex) ? "h23" : "h01";
  }

  return dataName + "_" + batchName + "_" + mtRowName + "_" + mtColName;
}

std::string getSaveOfmDumpFileName(const int batch, const int mtRowIndex,
                                   const int mtColumnIndex) {
  UNI_LOG_CHECK(mtColumnIndex < SimCfgProxy::Params().SAVE_OFM_PARALLEL, SIM_CFG_FMT_NO_MATCH);

  std::string dataName = "ofm";
  std::string batchName = "b" + std::to_string(batch);
  std::string mtRowName = (0 == mtRowIndex) ? "c0" : "c1";
  std::string mtColName = "h23";
  if(2 == SimCfgProxy::Params().SAVE_OFM_PARALLEL && 0 == mtColumnIndex) {
    mtColName = "h01";
  }

  return dataName + "_" + batchName + "_" + mtRowName + "_" + mtColName;
}

// apply for conv_wgt, conv_bias, misc_wgt, misc_bias
std::string getLoadConvWgtDumpFileName(const int batch, const int mtRowIndex,
                                       const int mtColumnIndex,
                                       const int parallelIndex) {
  UNI_LOG_CHECK(parallelIndex < SimCfgProxy::Params().LOAD_WGT_PARALLEL,
                SIM_CFG_FMT_NO_MATCH);

  std::string dataName = "wgt";
  std::string mtRowName = (0 == mtRowIndex) ? "oc0" : "oc1";
  std::string mtColName = (0 == mtColumnIndex) ? "ic0" : "ic1";
  std::string parallelName = (0 == parallelIndex) ? "in0" : "in1";

  return dataName + "_" + mtRowName + "_" + mtColName + "_" + parallelName;
}

void mkdir_check(const std::string& path) {
  if (access(path.c_str(), R_OK | W_OK | X_OK) == 0) {
      return;
  }

  if (mkdir(path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1) {
    UNI_LOG_FATAL(SIM_FILE_OPEN_FAILED) << "mkdir error: " << path;
  }
}

void DelFileOrDir_check(const string& name) {
  struct stat buf;
  const char* fname = name.c_str();
  if (lstat(fname, &buf) == -1) {
    return;
  }

  Util::DelFileOrDir(name);
}

void save_hex_smallEnd(const std::string& filePath, std::vector<char>& data, bool isLast) {
  UNI_LOG_CHECK(SimCfgProxy::Params().DUMP_LINE_DATA_NUM == (long int)(data.size()), SIM_OUT_OF_RANGE);
  static std::set<std::string> createdFilesSet;
  bool isFirstSave = (0 == createdFilesSet.count(filePath));
  std::ofstream f;
  auto mode = isFirstSave ? std::ios::trunc : std::ios::app;
  f.open(filePath, mode);
  createdFilesSet.insert(filePath);

  std::string header = isLast ? "TLAST" : "TDATA";
  int num = data.size();
  f << header << " : ";
  for (int j = num - 1; j >= 0; j--) {
    char tmp = data.at(j);
    uint8_t tmp_u8 = static_cast<uint8_t>(tmp);
    uint32_t tmp_u32 = static_cast<uint32_t>(tmp_u8);
    f << std::hex << std::setw(2) << std::setfill('0') << tmp_u32;
  }
  f << "\n";
  f.close();
}

void save_hex_bigEnd(const std::string& filePath, std::vector<char>& data, bool isLast) {
  UNI_LOG_CHECK(SimCfgProxy::Params().DUMP_LINE_DATA_NUM == (long int)(data.size()), SIM_OUT_OF_RANGE);
  static std::set<std::string> createdFilesSet;
  bool isFirstSave = (0 == createdFilesSet.count(filePath));
  std::ofstream f;
  auto mode = isFirstSave ? std::ios::trunc : std::ios::app;
  f.open(filePath, mode);
  createdFilesSet.insert(filePath);

  std::string header = isLast ? "TLAST" : "TDATA";
  int num = data.size();
  f << header << " : ";
  for (int j = 0; j < num; j++) {
    char tmp = data.at(j);
    uint8_t tmp_u8 = static_cast<uint8_t>(tmp);
    uint32_t tmp_u32 = static_cast<uint32_t>(tmp_u8);
    f << std::hex << std::setw(2) << std::setfill('0') << tmp_u32;
  }
  f << "\n";
  f.close();
}

void save_txt_smallEnd(const std::string& filePath, std::vector<char>& data, bool isLast) {
  UNI_LOG_CHECK(SimCfgProxy::Params().DUMP_LINE_DATA_NUM == (long int)(data.size()), SIM_OUT_OF_RANGE);
  static std::set<std::string> createdFilesSet;
  bool isFirstSave = (0 == createdFilesSet.count(filePath));
  std::ofstream f;
  auto mode = isFirstSave ? std::ios::trunc : std::ios::app;
  f.open(filePath, mode);
  createdFilesSet.insert(filePath);

  std::vector<char> data_reverse;
  std::copy(data.rbegin(), data.rend(), std::back_inserter(data_reverse));
  const int32_t* buf = reinterpret_cast<const int32_t*>(data_reverse.data());

  if (isLast) {
    f << "TLAST\n";
  } 
  int num = data_reverse.size();
  for (int j = 0; j < num / 4; j++) {
    f << buf[j] << " ";
  }
  f << "\n";
  f.close();
}

void save_txt_bigEnd(const std::string& filePath, std::vector<char>& data, bool isLast) {
  UNI_LOG_CHECK(SimCfgProxy::Params().DUMP_LINE_DATA_NUM == (long int)(data.size()), SIM_OUT_OF_RANGE);
  static std::set<std::string> createdFilesSet;
  bool isFirstSave = (0 == createdFilesSet.count(filePath));
  std::ofstream f;
  auto mode = isFirstSave ? std::ios::trunc : std::ios::app;
  f.open(filePath, mode);
  createdFilesSet.insert(filePath);

  const int32_t* buf = reinterpret_cast<const int32_t*>(data.data());
  if (isLast) {
    f << "TLAST\n";
  } 
  int num = data.size();
  for (int j = 0; j < num / 4; j++) {
    f << buf[j] << " ";
  }
  f << "\n";
  f.close();
}


void streamDump(const std::vector<char>& data, const std::vector<StreamDataFormat>& formats,
               const StreamDataSequence& sequence,
               const std::string& dir, const std::string& fileName, bool isLast) {
	const int lineDataNum = SimCfgProxy::Params().DUMP_LINE_DATA_NUM;
	int dumpLines = data.size() / lineDataNum;
	UNI_LOG_CHECK(dumpLines * lineDataNum == (long int)(data.size()), SIM_OUT_OF_RANGE);

	for(int idx_line = 0; idx_line < dumpLines; ++idx_line) {
    bool realIsLast = isLast && (idx_line == dumpLines - 1);
		int start = idx_line * lineDataNum;
		int end = start + lineDataNum;
		std::vector<char> lineData;
		std::copy(data.begin() + start, data.begin() + end, std::back_inserter(lineData));

		static std::map<StreamDataFormat, std::string> formatMap = {
			{StreamDataFormat::TXT, ".txt"},
			{StreamDataFormat::HEX, ".hex"}};
    
    static std::map<std::pair<StreamDataFormat, StreamDataSequence>, 
                    void(*)(const std::string&, std::vector<char>&, bool)> saveFuncMap = {
      {{StreamDataFormat::HEX, StreamDataSequence::BIGEND}, &save_hex_bigEnd},
      {{StreamDataFormat::HEX, StreamDataSequence::SMALLEND}, &save_hex_smallEnd},
      {{StreamDataFormat::TXT, StreamDataSequence::BIGEND}, &save_txt_bigEnd},
      {{StreamDataFormat::TXT, StreamDataSequence::SMALLEND}, &save_txt_smallEnd}};

		for(auto format : formats) {
			std::string filePath = dir + "/" + fileName + formatMap.at(format);
      auto saveFunc = *saveFuncMap.at({format, SimCfgProxy::Params().DUMP_SEQUENCE});
      saveFunc(filePath, lineData, realIsLast);
		}
	}
}

std::string getLoadIfmTokenFileName(const int batch, const int mtColumnIndex) {
  UNI_LOG_CHECK(mtColumnIndex < SimCfgProxy::Params().LOAD_IFM_PARALLEL, SIM_CFG_FMT_NO_MATCH);

  // ifm_token_b0_h23
  std::string dataName = "ifm_token";
  std::string batchName = "b" + std::to_string(batch);
  std::string mtColName = "h01";
  // std::string mtColName = "h23";
  // if(2 == SimCfgProxy::Params().LOAD_IFM_PARALLEL && 0 == mtColumnIndex) {
  //   mtColName = "h01";
  // }

  return dataName + "_" + batchName + "_" + mtColName;
}

std::string getSaveOfmTokenFileName(const int batch, const int mtColumnIndex) {
  UNI_LOG_CHECK(mtColumnIndex < SimCfgProxy::Params().SAVE_OFM_PARALLEL, SIM_CFG_FMT_NO_MATCH);

  // ifm_token_b0_h23
  std::string dataName = "ifm_token";
  std::string batchName = "b" + std::to_string(batch);
  std::string mtColName = "h01";
  // std::string mtColName = "h23";
  // if(2 == SimCfgProxy::Params().SAVE_OFM_PARALLEL && 0 == mtColumnIndex) {
  //   mtColName = "h01";
  // }

  return dataName + "_" + batchName + "_" + mtColName;
}

std::string getLoadWgtTokenFileName(const int batch, const int mtColumnIndex,
                                    const int idx_parallel) {
  // wgt_token_icg0
  std::string dataName = "wgt_token";
  std::string mtColName = (0 == mtColumnIndex) ? "icg0" : "icg1";
  std::string parallelName = (0 == idx_parallel) ? "L0" : "L1";

  return dataName + "_" + mtColName + "_" + parallelName;
}

std::string getConvTokenFileName(const int batch, const int mtColumnIndex,
                                 SIM_OP_TYPE dataType) {
  // img_token_b0_h01_conv_ifm.txt
  // img_token_b0_h01_conv_ofm.txt
  // wgt_token_b0_ic0_conv.txt
  // wgt_token_b0_ic1_conv.txt
  UNI_LOG_CHECK(mtColumnIndex == 0 || mtColumnIndex == 1, VART_INVALID_VALUE);
  std::string dataName = "";
  std::string batchName = "b" + std::to_string(batch);
  std::string mtColName = "";
  std::string outputName = "";
  if (SIM_OP_TYPE::CONV_IFM == dataType) {
    dataName = "img_token";
    // mtColName = (0 == mtColumnIndex) ? "h01" : "h23";
    mtColName = "h01";
    outputName = "conv_ifm";
  } else if (SIM_OP_TYPE::CONV_OFM == dataType) {
    dataName = "img_token";
    // mtColName = (0 == mtColumnIndex) ? "h01" : "h23";
    mtColName = "h01";
    outputName = "conv_ofm";
  } else if (SIM_OP_TYPE::CONV_WGTS == dataType ||
             SIM_OP_TYPE::CONV_BIAS == dataType) {
    dataName = "wgt_token";
    mtColName = (0 == mtColumnIndex) ? "ic0" : "ic1";
    outputName = "conv";
  } else {
    UNI_LOG_FATAL(VART_NOT_SUPPORT) << "unsupported data type";
  }

  return dataName + "_" + batchName + "_" + mtColName + "_" + outputName;
}

std::string getAluTokenFileName(const int batch, const int mtColumnIndex,
                                SIM_OP_TYPE dataType) {
  // img_token_b0_h01_alu_ifm.txt
  // img_token_b0_h23_alu_ofm.txt
  // wgt_token_b0_ic0_alu.txt
  UNI_LOG_CHECK(mtColumnIndex == 0 || mtColumnIndex == 1, VART_INVALID_VALUE);
  std::string dataName = "";
  std::string batchName = "b" + std::to_string(batch);
  std::string mtColName = "";
  std::string outputName = "";
  if (SIM_OP_TYPE::ALU_IFM == dataType) {
    dataName = "img_token";
    // mtColName = (0 == mtColumnIndex) ? "h01" : "h23";
    mtColName = "h01";
    outputName = "alu_ifm";
  } else if (SIM_OP_TYPE::ALU_OFM == dataType) {
    dataName = "img_token";
    UNI_LOG_CHECK(mtColumnIndex == 1, SIM_CFG_FMT_NO_MATCH);
    // mtColName = "h23";
    mtColName = "h01";
    outputName = "alu_ofm";
  } else if (SIM_OP_TYPE::ALU_WGTS == dataType ||
             SIM_OP_TYPE::ALU_BIAS == dataType) {
    dataName = "wgt_token";
    UNI_LOG_CHECK(mtColumnIndex == 0, SIM_CFG_FMT_NO_MATCH);
    mtColName = "ic0";
    outputName = "alu";
  } else {
    UNI_LOG_FATAL(VART_NOT_SUPPORT) << "unsupported data type";
  }

  return dataName + "_" + batchName + "_" + mtColName + "_" + outputName;
}

static uint32_t addOddParityBits(uint32_t in) {
  uint32_t odd = 1;
  for(int i=0; i<31; i++){
    odd ^= (in >> i);
  }
  return (in | (odd << 31));
}

static uint32_t genToken(uint32_t streamID, uint32_t actorID) {                 
  uint32_t Stream_ID = streamID;
  uint32_t Controller_ID = 0;
  uint32_t Actor_ID = actorID;
  uint32_t Packet_Type = 6;
  uint32_t Source_Row = 0;
  uint32_t Source_Column = 0;

  assert((((Stream_ID       >>  5 ) & 0xffffffff) == 0) && "Stream_ID        only  5 bits valid");
  assert((((Controller_ID   >>  3 ) & 0xffffffff) == 0) && "Controller_ID    only  3 bits valid");
  assert((((Actor_ID        >>  4 ) & 0xffffffff) == 0) && "Actor_ID         only  4 bits valid");
  assert((((Packet_Type     >>  3 ) & 0xffffffff) == 0) && "Packet_Type      only  3 bits valid");
  assert((((Source_Row      >>  5 ) & 0xffffffff) == 0) && "Source_Row       only  5 bits valid");
  assert((((Source_Column   >>  7 ) & 0xffffffff) == 0) && "Source_Column    only  7 bits valid");

  uint32_t ret = 0x00000000;
  ret |= Stream_ID;               // bit:4-0
  ret |= (Controller_ID << 5);    // bit:7-5
  ret |= (Actor_ID      << 8);    // bit:11-8
  ret |= (Packet_Type   << 12);   // bit:14:12
  ret |= (Source_Row    << 16);   // bit:20-16
  ret |= (Source_Column << 21);   // bit:27:21
  return addOddParityBits(ret);
}

static uint32_t get_Actor_ID_LOAD_SAVE_PL(SIM_OP_TYPE op, const int parallelIndex) {
  const int MT_IMG_CHN_LOAD     = SimCfg::Instance().get_MT_IMG_CHN_LOAD();
  const int MT_IMG_CHN_SAVE     = SimCfg::Instance().get_MT_IMG_CHN_SAVE();
  const int MT_WGT_CHN_LOAD_L1  = SimCfg::Instance().get_MT_WGT_CHN_LOAD_L1();
  const int MT_WGT_CHN_LOAD_L2  = SimCfg::Instance().get_MT_WGT_CHN_LOAD_L2();

  uint32_t actorID = 0;
  if (SIM_OP_TYPE::LOAD_CONV_WGT == op || SIM_OP_TYPE::LOAD_CONV_BIAS == op ||
      SIM_OP_TYPE::LOAD_MISC_WGT == op || SIM_OP_TYPE::LOAD_MISC_BIAS == op) {
    actorID = (0 == parallelIndex) ? MT_WGT_CHN_LOAD_L1 : MT_WGT_CHN_LOAD_L2;
  } else if (SIM_OP_TYPE::LOAD_IFM == op) {
    actorID = MT_IMG_CHN_LOAD;
  } else if (SIM_OP_TYPE::SAVE_OFM == op) {
    actorID = MT_IMG_CHN_SAVE;
  } else {
    UNI_LOG_FATAL(VART_INVALID_VALUE);
  }

  return actorID;
}

static uint32_t get_Actor_ID_CONV_PL(SIM_OP_TYPE op, int hIndex) {
  const int MT_IMG_CHN_CONV_IFM_H0 = SimCfg::Instance().get_MT_IMG_CHN_CONV_IFM_H0();
  const int MT_IMG_CHN_CONV_IFM_H1 = SimCfg::Instance().get_MT_IMG_CHN_CONV_IFM_H1();
  const int MT_IMG_CHN_CONV_OFM_H0 = SimCfg::Instance().get_MT_IMG_CHN_CONV_OFM_H0();
  const int MT_IMG_CHN_CONV_OFM_H1 = SimCfg::Instance().get_MT_IMG_CHN_CONV_OFM_H1();
  const int MT_WGT_CHN_CONV = SimCfg::Instance().get_MT_WGT_CHN_CONV();

  UNI_LOG_CHECK(hIndex == 0 || hIndex == 1 ||
                hIndex == 2 || hIndex == 3, VART_INVALID_VALUE);
  hIndex = hIndex % 2;
  uint32_t actorID = 0;
  if (SIM_OP_TYPE::CONV_IFM == op) {
    actorID = ((0 == hIndex) ? MT_IMG_CHN_CONV_IFM_H0 : MT_IMG_CHN_CONV_IFM_H1);
  } else if (SIM_OP_TYPE::CONV_OFM == op) {
    actorID = ((0 == hIndex) ? MT_IMG_CHN_CONV_OFM_H0 : MT_IMG_CHN_CONV_OFM_H1);
  } else if (SIM_OP_TYPE::CONV_WGTS == op || SIM_OP_TYPE::CONV_BIAS == op) {
    actorID = MT_WGT_CHN_CONV;
  } else {
    UNI_LOG_FATAL(VART_INVALID_VALUE);
  }

  return actorID;
}

static uint32_t get_Actor_ID_ALU_PL(SIM_OP_TYPE op) {
  const int MT_IMG_CHN_ALU_IFM = SimCfg::Instance().get_MT_IMG_CHN_ALU_IFM();
  const int MT_IMG_CHN_ALU_OFM = SimCfg::Instance().get_MT_IMG_CHN_ALU_OFM();
  const int MT_WGT_CHN_ALU = SimCfg::Instance().get_MT_WGT_CHN_ALU();

  uint32_t actorID = 0;
  if (SIM_OP_TYPE::ALU_IFM == op) {
    actorID = MT_IMG_CHN_ALU_IFM;
  } else if (SIM_OP_TYPE::ALU_OFM == op) {
    actorID = MT_IMG_CHN_ALU_OFM;
  } else if (SIM_OP_TYPE::ALU_WGTS == op || SIM_OP_TYPE::ALU_BIAS == op) {
    actorID = MT_WGT_CHN_ALU;
  } else {
    UNI_LOG_FATAL(VART_INVALID_VALUE);
  }

  return actorID;
}

void token2file_load_save_PL(SIM_OP_TYPE op, const int batch, const int mtRowIndex, 
                   const int mtColumnIndex, const int parallelIndex) {
  uint32_t streamID = mtRowIndex;
  if (SIM_OP_TYPE::LOAD_IFM == op || SIM_OP_TYPE::SAVE_OFM == op) {
    int real_mt_column_index = 1;
    if (2 == SimCfgProxy::Params().LOAD_IFM_PARALLEL && 0 == mtColumnIndex) {
      real_mt_column_index = 0;
    }
    streamID = 2 * real_mt_column_index + mtRowIndex;
  }
  uint32_t actorID = get_Actor_ID_LOAD_SAVE_PL(op, parallelIndex);
  uint32_t tokenValue = genToken(streamID, actorID);

  std::string tokenFileName = "";
  std::string subDir = "";
  if (SIM_OP_TYPE::LOAD_CONV_WGT == op || SIM_OP_TYPE::LOAD_CONV_BIAS == op ||
      SIM_OP_TYPE::LOAD_MISC_WGT == op || SIM_OP_TYPE::LOAD_MISC_BIAS == op) {
    UNI_LOG_CHECK(parallelIndex < SimCfgProxy::Params().LOAD_WGT_PARALLEL, SIM_CFG_FMT_NO_MATCH);
    tokenFileName =
        getLoadWgtTokenFileName(batch, mtColumnIndex, parallelIndex);
    subDir = "load";
  } else if (SIM_OP_TYPE::LOAD_IFM == op) {
    UNI_LOG_CHECK(parallelIndex < SimCfgProxy::Params().LOAD_IFM_PARALLEL, SIM_CFG_FMT_NO_MATCH);
    tokenFileName = getLoadIfmTokenFileName(batch, mtColumnIndex);
    subDir = "load";
  } else if (SIM_OP_TYPE::SAVE_OFM == op) {
    UNI_LOG_CHECK(parallelIndex < SimCfgProxy::Params().SAVE_OFM_PARALLEL, SIM_CFG_FMT_NO_MATCH);
    tokenFileName = getSaveOfmTokenFileName(batch, mtColumnIndex);
    subDir = "save";
  } else {
    UNI_LOG_FATAL(VART_INVALID_VALUE);
  }
  std::string filePath = SimCfgProxy::Params().DUMP_DIR + "/" + subDir +
                         "/" + tokenFileName + ".txt";

  static std::set<std::string> createdFilesSet;
  bool isFirstSave = (0 == createdFilesSet.count(filePath));
  std::ofstream f;
  auto mode = (isFirstSave ? std::ios::trunc : std::ios::app);
  f.open(filePath, mode);
  createdFilesSet.insert(filePath);

  f << "TLAST : ";
  // bigEnd
  f << std::hex << std::setw(8) << std::setfill('0') << tokenValue;
  f << "\n";
  f.close();
}

void token2file_conv_PL(SIM_OP_TYPE op, const int batch, const int mtRowIndex,
                        const int mtColumnIndex, const int hIndex) {
  uint32_t streamID = mtRowIndex;
  if (SIM_OP_TYPE::CONV_IFM == op || SIM_OP_TYPE::CONV_OFM == op) {
    streamID = 2 * mtColumnIndex + mtRowIndex;
  }
  uint32_t actorID = get_Actor_ID_CONV_PL(op, hIndex);
  uint32_t tokenValue = genToken(streamID, actorID);

  std::string fileDir = SimCfgProxy::Params().DUMP_DIR + "/conv";
  std::string tokenFileName = getConvTokenFileName(batch, mtColumnIndex, op);
  std::string filePath = fileDir + "/" + tokenFileName + ".txt";
  mkdir_check(fileDir);

  static std::set<std::string> createdFilesSet;
  bool isFirstSave = (0 == createdFilesSet.count(filePath));
  std::ofstream f;
  auto mode = (isFirstSave ? std::ios::trunc : std::ios::app);
  f.open(filePath, mode);
  createdFilesSet.insert(filePath);

  f << "TLAST : ";
  // bigEnd
  f << std::hex << std::setw(16) << std::setfill('0') << tokenValue;
  f << "\n";
  f.close();
}

void token2file_alu_PL(SIM_OP_TYPE op, const int batch, const int mtRowIndex,
                       const int mtColumnIndex) {
  uint32_t streamID = mtRowIndex;
  if (SIM_OP_TYPE::ALU_IFM == op || SIM_OP_TYPE::ALU_OFM == op) {
    streamID = 2 * mtColumnIndex + mtRowIndex;
  }
  uint32_t actorID = get_Actor_ID_ALU_PL(op);
  uint32_t tokenValue = genToken(streamID, actorID);

  std::string fileDir = SimCfgProxy::Params().DUMP_DIR + "/alu";
  std::string tokenFileName = getAluTokenFileName(batch, mtColumnIndex, op);
  std::string filePath = fileDir + "/" + tokenFileName + ".txt";
  mkdir_check(fileDir);

  static std::set<std::string> createdFilesSet;
  bool isFirstSave = (0 == createdFilesSet.count(filePath));
  std::ofstream f;
  auto mode = (isFirstSave ? std::ios::trunc : std::ios::app);
  f.open(filePath, mode);
  createdFilesSet.insert(filePath);

  f << "TLAST : ";
  // bigEnd
  f << std::hex << std::setw(16) << std::setfill('0') << tokenValue;
  f << "\n";
  f.close();
}
