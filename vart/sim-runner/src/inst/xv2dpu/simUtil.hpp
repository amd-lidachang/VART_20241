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

#include <string>
#include <vector>
#include "simCfg.hpp"

std::string getLoadIfmDumpFileName(const int batch, const int mtRowIndex,
                                   const int mtColumnIndex);
std::string getSaveOfmDumpFileName(const int batch, const int mtRowIndex,
                                   const int mtColumnIndex);
std::string getLoadConvWgtDumpFileName(const int batch, const int mtRowIndex,
                                       const int mtColumnIndex,
                                       const int parallelIndex);

void mkdir_check(const std::string& path);
void DelFileOrDir_check(const string& name);
void save_hex_smallEnd(const std::string& filePath, std::vector<char>& data, bool isLast);
void save_hex_bigEnd(const std::string& filePath, std::vector<char>& data, bool isLast);
void save_txt_smallEnd(const std::string& filePath, std::vector<char>& data, bool isLast);
void save_txt_bigEnd(const std::string& filePath, std::vector<char>& data, bool isLast);
void streamDump(const std::vector<char>& data,
                const std::vector<StreamDataFormat>& formats,
                const StreamDataSequence& sequence, const std::string& dir,
                const std::string& fileName, bool isLast);
std::string getLoadIfmTokenFileName(const int batch, const int mtColumnIndex);
std::string getSaveOfmTokenFileName(const int batch, const int mtColumnIndex);
std::string getLoadWgtTokenFileName(const int batch, const int mtColumnIndex,
                                    const int idx_parallel);
std::string getConvTokenFileName(const int batch, const int mtColumnIndex,
                                 SIM_OP_TYPE dataType);
std::string getAluTokenFileName(const int batch, const int mtColumnIndex,
                                SIM_OP_TYPE dataType);
void token2file_load_save_PL(SIM_OP_TYPE op, const int batch, const int mtRowIndex, 
                   const int mtColumnIndex, const int parallelIndex);
void token2file_conv_PL(SIM_OP_TYPE op, const int batch, const int mtRowIndex,
                        const int mtColumnIndex, const int hIndex);
void token2file_alu_PL(SIM_OP_TYPE op, const int batch, const int mtRowIndex,
                       const int mtColumnIndex);