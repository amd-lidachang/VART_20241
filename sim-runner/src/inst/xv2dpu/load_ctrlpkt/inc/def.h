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

namespace cosim {
namespace xv2dpu {
namespace load {

// data unit
constexpr int DATA_BW = 2;

// DDR or HBM
constexpr int AXI_BW = 16;
constexpr int DDR_WEIGHT_BLOCK = 32;
constexpr int DDR_DEPTH = 1024;
constexpr int DDR_NUM = 1;

// matrix multiplication process
/******************************************
                    COLUMNS
        +++++++++++++++++++++++++++++++
        + 							  +
        +                             +
UNITES  +         matrix unite        +
        +                             +
        +                             +
        +++++++++++++++++++++++++++++++
        

******************************************/
constexpr int COLUMNS = 32;
constexpr int UNITES_IN_COLUMN = 32;
constexpr int CASCADE_ADD_DATA_bW = 48;  // bitwidth of cascade data for adding
constexpr int NAX_CHANNELS = 32;         // max vector chnnels

// bank
/******************************************
weight_group: 0 2048Bx4096
vector_bank:  1 64Bx512
bias_bank:    2 64Bx512
shared_bank:  3- 64Bx512

******************************************/
constexpr int WEIGHT_GROUP_BANK_NUM = COLUMNS;
constexpr int WEIGHT_GROUP_BW = 2048;  //= COLUMNS x UNITES_IN_COLUMN x DATA_BW
constexpr int WEIGHT_BANK_BW = WEIGHT_GROUP_BW / WEIGHT_GROUP_BANK_NUM;
constexpr int WEIGHT_GROUP_DEPTH = 2048;  // 1024+512
constexpr int VECTOR_BANK_BW = 64;        //= UNITES_IN_COLUMN x DATA_BW
constexpr int VECTOR_BANK_DEPTH = 512;
constexpr int BIAS_BANK_BW = 64;  //= COLUMN x DATA_BW
constexpr int BIAS_BANK_DEPTH = 512;
constexpr int SHARED_BANK_BW = 64;
constexpr int SHARED_BANK_DEPTH = 512;

constexpr int SHARED_BANK_NUM = 8;
constexpr int BANK_NUM = 42;
constexpr int WEIGHT_BANK_ID = 0;
constexpr int VECTOR_BANK_ID = 1;
constexpr int BIAS_BANK_ID = 2;
constexpr int MAX_SHARED_BANK_ID = 10;
constexpr int ACTV_LOOKUP_DEPTH = 2048;

// sigmoid/tanh using Cordic or lookup
constexpr int ACTV_USE_LOOKUP = 1;
constexpr int SIGMOID_ITERATES = 50;
constexpr int TANH_ITERATES = 30;

}  // namespace load
}  // namespace xv2dpu
}  // namespace cosim