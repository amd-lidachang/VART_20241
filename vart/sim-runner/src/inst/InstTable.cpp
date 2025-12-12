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

#include <InstTable.hpp>


namespace DPU4F {
// instruction version
const string InstTable::INST_VERSION = "DPU4F";


// category name vector
const vector<string> InstTable::CategoryName = {
    "LOAD",
    "SAVE",
    "CONV",
    "MISC",
};


// inst name vector
const vector<string> InstTable::InstName = {
    "LOAD", // 0
    "SAVE", // 1
    "CONVINIT", // 2
    "CONV", // 3
    "POOLINIT", // 4
    "POOL", // 5
    "DWINIT", // 6
    "DPTWISE", // 7
    "ELEWINIT", // 8
    "ELEW", // 9
    "THD", // 10
    "END", // 11
    "DUMPBANK", // 12
    "DUMPDDR", // 13
    "DUMPDDRSLICE", // 14
};


// lower inst name vector
const vector<string> InstTable::InstNameLower = {
    "load", // 0
    "save", // 1
    "convinit", // 2
    "conv", // 3
    "poolinit", // 4
    "pool", // 5
    "dwinit", // 6
    "dptwise", // 7
    "elewinit", // 8
    "elew", // 9
    "thd", // 10
    "end", // 11
    "dumpbank", // 12
    "dumpddr", // 13
    "dumpddrslice", // 14
};


// inst category
const vector<Category> InstTable::InstCategory = {
    Category::INST_CATEGORY_LOAD, // LOAD
    Category::INST_CATEGORY_SAVE, // SAVE
    Category::INST_CATEGORY_CONV, // CONVINIT
    Category::INST_CATEGORY_CONV, // CONV
    Category::INST_CATEGORY_MISC, // POOLINIT
    Category::INST_CATEGORY_MISC, // POOL
    Category::INST_CATEGORY_MISC, // DWINIT
    Category::INST_CATEGORY_MISC, // DPTWISE
    Category::INST_CATEGORY_MISC, // ELEWINIT
    Category::INST_CATEGORY_MISC, // ELEW
    Category::INST_CATEGORY_MISC, // THD
    Category::INST_CATEGORY_SAVE, // END
    Category::INST_CATEGORY_MISC, // DUMPBANK
    Category::INST_CATEGORY_MISC, // DUMPDDR
    Category::INST_CATEGORY_MISC, // DUMPDDRSLICE
};


// inst word number
const vector<uint32_t> InstTable::WordNum = {
    4, // LOAD
    4, // SAVE
    4, // CONVINIT
    5, // CONV
    2, // POOLINIT
    5, // POOL
    3, // DWINIT
    5, // DPTWISE
    2, // ELEWINIT
    3, // ELEW
    5, // THD
    1, // END
    2, // DUMPBANK
    4, // DUMPDDR
    6, // DUMPDDRSLICE
};


// inst field number
const vector<uint32_t> InstTable::FieldNum = {
    LOAD_FIELD_MAX, // 0
    SAVE_FIELD_MAX, // 1
    CONVINIT_FIELD_MAX, // 2
    CONV_FIELD_MAX, // 3
    POOLINIT_FIELD_MAX, // 4
    POOL_FIELD_MAX, // 5
    DWINIT_FIELD_MAX, // 6
    DPTWISE_FIELD_MAX, // 7
    ELEWINIT_FIELD_MAX, // 8
    ELEW_FIELD_MAX, // 9
    THD_FIELD_MAX, // 10
    END_FIELD_MAX, // 11
    DUMPBANK_FIELD_MAX, // 12
    DUMPDDR_FIELD_MAX, // 13
    DUMPDDRSLICE_FIELD_MAX, // 14
};


// inst opcode
const vector<uint32_t> InstTable::OPCode = {
    0x00, // LOAD
    0x04, // SAVE
    0x09, // CONVINIT
    0x08, // CONV
    0x06, // POOLINIT
    0x0C, // POOL
    0x0B, // DWINIT
    0x0A, // DPTWISE
    0x0D, // ELEWINIT
    0x0E, // ELEW
    0x0F, // THD
    0x07, // END
    0xFF, // DUMPBANK
    0xFE, // DUMPDDR
    0xFD, // DUMPDDRSLICE
};


// Load field name
const vector<string> InstTable::LoadFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "hp_id", // 3
    "bank_id", // 4
    "bank_addr", // 5
    "quant_lth", // 6
    "pad_start", // 7
    "pad_end", // 8
    "pad_idx", // 9
    "jump_read", // 10
    "jump_write", // 11
    "length", // 12
    "mode_avg", // 13
    "channel", // 14
    "reg_id", // 15
    "ddr_addr", // 16
};


// Save field name
const vector<string> InstTable::SaveFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "hp_id", // 3
    "bank_id", // 4
    "bank_addr", // 5
    "quant_lth", // 6
    "jump_write", // 7
    "jump_read", // 8
    "length", // 9
    "channel", // 10
    "reg_id", // 11
    "ddr_addr", // 12
};


// ConvInit field name
const vector<string> InstTable::ConvInitFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "act_type", // 3
    "shift_cut", // 4
    "jump_read", // 5
    "stride_out", // 6
    "calt_mode", // 7
    "quant_lth", // 8
    "shift_bias", // 9
    "jump_read_endl", // 10
    "stride_h", // 11
    "kernel_h", // 12
    "valid_pixel_parallel", // 13
    "stride_offset_in", // 14
    "jump_write", // 15
    "stride_w", // 16
    "kernel_w", // 17
    "stride_offset_out", // 18
    "jump_write_endl", // 19
};


// Conv field name
const vector<string> InstTable::ConvFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "pad_left", // 3
    "pad_top", // 4
    "bank_addr_in", // 5
    "channel_group", // 6
    "channel_offset", // 7
    "pad_right", // 8
    "pad_bottom", // 9
    "bank_addr_out", // 10
    "length", // 11
    "bank_id_in", // 12
    "bank_addr_weights", // 13
    "bank_addr_in_1", // 14
    "bank_id_out", // 15
    "bank_addr_bias", // 16
    "bank_addr_in_3", // 17
    "bank_addr_in_2", // 18
};


// PoolInit field name
const vector<string> InstTable::PoolInitFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "shift_cut", // 3
    "kernel_w", // 4
    "kernel_h", // 5
    "jump_read", // 6
    "jump_write", // 7
    "stride_out", // 8
    "stride_offset_out", // 9
    "valid_pixel_parallel", // 10
    "stride_offset_in", // 11
    "stride_w", // 12
    "stride_h", // 13
    "pool_type", // 14
};


// Pool field name
const vector<string> InstTable::PoolFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "channel_group", // 3
    "bank_addr_in", // 4
    "jump_write_endl", // 5
    "pad_left", // 6
    "pad_top", // 7
    "pad_right", // 8
    "pad_bottom", // 9
    "bank_id_in", // 10
    "jump_read_endl", // 11
    "length", // 12
    "bank_id_out", // 13
    "bank_addr_in_1", // 14
    "bank_addr_out", // 15
    "bank_addr_in_3", // 16
    "bank_addr_in_2", // 17
};


// DWInit field name
const vector<string> InstTable::DWInitFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "jump_read", // 3
    "jump_read_endl", // 4
    "stride_offset_in", // 5
    "valid_pixel_parallel", // 6
    "stride_w", // 7
    "stride_h", // 8
    "kernel_w", // 9
    "kernel_h", // 10
    "jump_write_endl", // 11
    "jump_write", // 12
    "stride_out", // 13
    "stride_offset_out", // 14
};


// DptWise field name
const vector<string> InstTable::DptWiseFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "channel_group", // 3
    "bank_addr_in", // 4
    "pad_left", // 5
    "pad_right", // 6
    "pad_top", // 7
    "pad_bottom", // 8
    "channel_offset", // 9
    "bank_addr_out", // 10
    "act_type", // 11
    "length", // 12
    "bank_id_in", // 13
    "bank_addr_weights", // 14
    "bank_addr_in_1", // 15
    "bank_id_out", // 16
    "bank_addr_bias", // 17
    "bank_addr_in_3", // 18
    "bank_addr_in_2", // 19
};


// ElewInit field name
const vector<string> InstTable::ElewInitFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "bank_id_in", // 3
    "shift_read", // 4
    "jump_read", // 5
    "jump_read_endl", // 6
    "id", // 7
    "bank_addr_in", // 8
};


// Elew field name
const vector<string> InstTable::ElewFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "bank_id_out", // 3
    "shift_write", // 4
    "jump_write", // 5
    "num", // 6
    "channel_group", // 7
    "act_type", // 8
    "length", // 9
    "bank_addr_out", // 10
    "valid_pixel_parallel", // 11
    "jump_write_endl", // 12
};


// Thd field name
const vector<string> InstTable::ThdFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "channel_group", // 3
    "bank_addr_in", // 4
    "signed_mode", // 5
    "valid_pixel_parallel", // 6
    "length", // 7
    "bank_addr_out", // 8
    "stride_out", // 9
    "bank_id_in", // 10
    "jump_read", // 11
    "jump_read_endl", // 12
    "stride_offset_out", // 13
    "bank_id_out", // 14
    "jump_write", // 15
    "jump_write_endl", // 16
    "param_addr", // 17
};


// End field name
const vector<string> InstTable::EndFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
};


// DumpBank field name
const vector<string> InstTable::DumpBankFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "save_name", // 3
    "save_fmt", // 4
    "bank_start", // 5
    "bank_num", // 6
};


// DumpDDR field name
const vector<string> InstTable::DumpDDRFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "save_name", // 3
    "save_fmt", // 4
    "reg_id", // 5
    "ddr_start", // 6
    "ddr_size", // 7
};


// DumpDDRSlice field name
const vector<string> InstTable::DumpDDRSliceFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "save_name", // 3
    "save_fmt", // 4
    "reg_id", // 5
    "ddr_start", // 6
    "height", // 7
    "height_stride", // 8
    "width", // 9
    "width_stride", // 10
    "channel", // 11
    "channel_stride", // 12
};


// Load field data type
const vector<int> InstTable::LoadFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_UINT, // hp_id(3)
    INST_FIELD_DATA_TYPE_UINT, // bank_id(4)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr(5)
    INST_FIELD_DATA_TYPE_UINT, // quant_lth(6)
    INST_FIELD_DATA_TYPE_UINT, // pad_start(7)
    INST_FIELD_DATA_TYPE_UINT, // pad_end(8)
    INST_FIELD_DATA_TYPE_UINT, // pad_idx(9)
    INST_FIELD_DATA_TYPE_UINT, // jump_read(10)
    INST_FIELD_DATA_TYPE_UINT, // jump_write(11)
    INST_FIELD_DATA_TYPE_UINT, // length(12)
    INST_FIELD_DATA_TYPE_UINT, // mode_avg(13)
    INST_FIELD_DATA_TYPE_UINT, // channel(14)
    INST_FIELD_DATA_TYPE_UINT, // reg_id(15)
    INST_FIELD_DATA_TYPE_UINT, // ddr_addr(16)
};


// Save field data type
const vector<int> InstTable::SaveFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_UINT, // hp_id(3)
    INST_FIELD_DATA_TYPE_UINT, // bank_id(4)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr(5)
    INST_FIELD_DATA_TYPE_UINT, // quant_lth(6)
    INST_FIELD_DATA_TYPE_UINT, // jump_write(7)
    INST_FIELD_DATA_TYPE_UINT, // jump_read(8)
    INST_FIELD_DATA_TYPE_UINT, // length(9)
    INST_FIELD_DATA_TYPE_UINT, // channel(10)
    INST_FIELD_DATA_TYPE_UINT, // reg_id(11)
    INST_FIELD_DATA_TYPE_UINT, // ddr_addr(12)
};


// ConvInit field data type
const vector<int> InstTable::ConvInitFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_UINT, // act_type(3)
    INST_FIELD_DATA_TYPE_UINT, // shift_cut(4)
    INST_FIELD_DATA_TYPE_UINT, // jump_read(5)
    INST_FIELD_DATA_TYPE_UINT, // stride_out(6)
    INST_FIELD_DATA_TYPE_UINT, // calt_mode(7)
    INST_FIELD_DATA_TYPE_UINT, // quant_lth(8)
    INST_FIELD_DATA_TYPE_UINT, // shift_bias(9)
    INST_FIELD_DATA_TYPE_UINT, // jump_read_endl(10)
    INST_FIELD_DATA_TYPE_UINT, // stride_h(11)
    INST_FIELD_DATA_TYPE_UINT, // kernel_h(12)
    INST_FIELD_DATA_TYPE_UINT, // valid_pixel_parallel(13)
    INST_FIELD_DATA_TYPE_UINT, // stride_offset_in(14)
    INST_FIELD_DATA_TYPE_UINT, // jump_write(15)
    INST_FIELD_DATA_TYPE_UINT, // stride_w(16)
    INST_FIELD_DATA_TYPE_UINT, // kernel_w(17)
    INST_FIELD_DATA_TYPE_UINT, // stride_offset_out(18)
    INST_FIELD_DATA_TYPE_UINT, // jump_write_endl(19)
};


// Conv field data type
const vector<int> InstTable::ConvFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_UINT, // pad_left(3)
    INST_FIELD_DATA_TYPE_UINT, // pad_top(4)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in(5)
    INST_FIELD_DATA_TYPE_UINT, // channel_group(6)
    INST_FIELD_DATA_TYPE_UINT, // channel_offset(7)
    INST_FIELD_DATA_TYPE_UINT, // pad_right(8)
    INST_FIELD_DATA_TYPE_UINT, // pad_bottom(9)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_out(10)
    INST_FIELD_DATA_TYPE_UINT, // length(11)
    INST_FIELD_DATA_TYPE_UINT, // bank_id_in(12)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_weights(13)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in_1(14)
    INST_FIELD_DATA_TYPE_UINT, // bank_id_out(15)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_bias(16)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in_3(17)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in_2(18)
};


// PoolInit field data type
const vector<int> InstTable::PoolInitFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_UINT, // shift_cut(3)
    INST_FIELD_DATA_TYPE_UINT, // kernel_w(4)
    INST_FIELD_DATA_TYPE_UINT, // kernel_h(5)
    INST_FIELD_DATA_TYPE_UINT, // jump_read(6)
    INST_FIELD_DATA_TYPE_UINT, // jump_write(7)
    INST_FIELD_DATA_TYPE_UINT, // stride_out(8)
    INST_FIELD_DATA_TYPE_UINT, // stride_offset_out(9)
    INST_FIELD_DATA_TYPE_UINT, // valid_pixel_parallel(10)
    INST_FIELD_DATA_TYPE_UINT, // stride_offset_in(11)
    INST_FIELD_DATA_TYPE_UINT, // stride_w(12)
    INST_FIELD_DATA_TYPE_UINT, // stride_h(13)
    INST_FIELD_DATA_TYPE_UINT, // pool_type(14)
};


// Pool field data type
const vector<int> InstTable::PoolFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_UINT, // channel_group(3)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in(4)
    INST_FIELD_DATA_TYPE_UINT, // jump_write_endl(5)
    INST_FIELD_DATA_TYPE_UINT, // pad_left(6)
    INST_FIELD_DATA_TYPE_UINT, // pad_top(7)
    INST_FIELD_DATA_TYPE_UINT, // pad_right(8)
    INST_FIELD_DATA_TYPE_UINT, // pad_bottom(9)
    INST_FIELD_DATA_TYPE_UINT, // bank_id_in(10)
    INST_FIELD_DATA_TYPE_UINT, // jump_read_endl(11)
    INST_FIELD_DATA_TYPE_UINT, // length(12)
    INST_FIELD_DATA_TYPE_UINT, // bank_id_out(13)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in_1(14)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_out(15)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in_3(16)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in_2(17)
};


// DWInit field data type
const vector<int> InstTable::DWInitFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_UINT, // jump_read(3)
    INST_FIELD_DATA_TYPE_UINT, // jump_read_endl(4)
    INST_FIELD_DATA_TYPE_UINT, // stride_offset_in(5)
    INST_FIELD_DATA_TYPE_UINT, // valid_pixel_parallel(6)
    INST_FIELD_DATA_TYPE_UINT, // stride_w(7)
    INST_FIELD_DATA_TYPE_UINT, // stride_h(8)
    INST_FIELD_DATA_TYPE_UINT, // kernel_w(9)
    INST_FIELD_DATA_TYPE_UINT, // kernel_h(10)
    INST_FIELD_DATA_TYPE_UINT, // jump_write_endl(11)
    INST_FIELD_DATA_TYPE_UINT, // jump_write(12)
    INST_FIELD_DATA_TYPE_UINT, // stride_out(13)
    INST_FIELD_DATA_TYPE_UINT, // stride_offset_out(14)
};


// DptWise field data type
const vector<int> InstTable::DptWiseFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_UINT, // channel_group(3)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in(4)
    INST_FIELD_DATA_TYPE_UINT, // pad_left(5)
    INST_FIELD_DATA_TYPE_UINT, // pad_right(6)
    INST_FIELD_DATA_TYPE_UINT, // pad_top(7)
    INST_FIELD_DATA_TYPE_UINT, // pad_bottom(8)
    INST_FIELD_DATA_TYPE_UINT, // channel_offset(9)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_out(10)
    INST_FIELD_DATA_TYPE_UINT, // act_type(11)
    INST_FIELD_DATA_TYPE_UINT, // length(12)
    INST_FIELD_DATA_TYPE_UINT, // bank_id_in(13)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_weights(14)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in_1(15)
    INST_FIELD_DATA_TYPE_UINT, // bank_id_out(16)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_bias(17)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in_3(18)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in_2(19)
};


// ElewInit field data type
const vector<int> InstTable::ElewInitFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_UINT, // bank_id_in(3)
    INST_FIELD_DATA_TYPE_UINT, // shift_read(4)
    INST_FIELD_DATA_TYPE_UINT, // jump_read(5)
    INST_FIELD_DATA_TYPE_UINT, // jump_read_endl(6)
    INST_FIELD_DATA_TYPE_UINT, // id(7)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in(8)
};


// Elew field data type
const vector<int> InstTable::ElewFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_UINT, // bank_id_out(3)
    INST_FIELD_DATA_TYPE_UINT, // shift_write(4)
    INST_FIELD_DATA_TYPE_UINT, // jump_write(5)
    INST_FIELD_DATA_TYPE_UINT, // num(6)
    INST_FIELD_DATA_TYPE_UINT, // channel_group(7)
    INST_FIELD_DATA_TYPE_UINT, // act_type(8)
    INST_FIELD_DATA_TYPE_UINT, // length(9)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_out(10)
    INST_FIELD_DATA_TYPE_UINT, // valid_pixel_parallel(11)
    INST_FIELD_DATA_TYPE_UINT, // jump_write_endl(12)
};


// Thd field data type
const vector<int> InstTable::ThdFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_UINT, // channel_group(3)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in(4)
    INST_FIELD_DATA_TYPE_UINT, // signed_mode(5)
    INST_FIELD_DATA_TYPE_UINT, // valid_pixel_parallel(6)
    INST_FIELD_DATA_TYPE_UINT, // length(7)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_out(8)
    INST_FIELD_DATA_TYPE_UINT, // stride_out(9)
    INST_FIELD_DATA_TYPE_UINT, // bank_id_in(10)
    INST_FIELD_DATA_TYPE_UINT, // jump_read(11)
    INST_FIELD_DATA_TYPE_UINT, // jump_read_endl(12)
    INST_FIELD_DATA_TYPE_UINT, // stride_offset_out(13)
    INST_FIELD_DATA_TYPE_UINT, // bank_id_out(14)
    INST_FIELD_DATA_TYPE_UINT, // jump_write(15)
    INST_FIELD_DATA_TYPE_UINT, // jump_write_endl(16)
    INST_FIELD_DATA_TYPE_UINT, // param_addr(17)
};


// End field data type
const vector<int> InstTable::EndFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
};


// DumpBank field data type
const vector<int> InstTable::DumpBankFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_STRING, // save_name(3)
    INST_FIELD_DATA_TYPE_UINT, // save_fmt(4)
    INST_FIELD_DATA_TYPE_UINT, // bank_start(5)
    INST_FIELD_DATA_TYPE_UINT, // bank_num(6)
};


// DumpDDR field data type
const vector<int> InstTable::DumpDDRFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_STRING, // save_name(3)
    INST_FIELD_DATA_TYPE_UINT, // save_fmt(4)
    INST_FIELD_DATA_TYPE_UINT, // reg_id(5)
    INST_FIELD_DATA_TYPE_UINT, // ddr_start(6)
    INST_FIELD_DATA_TYPE_UINT, // ddr_size(7)
};


// DumpDDRSlice field data type
const vector<int> InstTable::DumpDDRSliceFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_STRING, // save_name(3)
    INST_FIELD_DATA_TYPE_UINT, // save_fmt(4)
    INST_FIELD_DATA_TYPE_UINT, // reg_id(5)
    INST_FIELD_DATA_TYPE_UINT, // ddr_start(6)
    INST_FIELD_DATA_TYPE_UINT, // height(7)
    INST_FIELD_DATA_TYPE_UINT, // height_stride(8)
    INST_FIELD_DATA_TYPE_UINT, // width(9)
    INST_FIELD_DATA_TYPE_UINT, // width_stride(10)
    INST_FIELD_DATA_TYPE_UINT, // channel(11)
    INST_FIELD_DATA_TYPE_UINT, // channel_stride(12)
};


// Load field minus value
const vector<int> InstTable::LoadFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // hp_id(3)
    0, // bank_id(4)
    0, // bank_addr(5)
    0, // quant_lth(6)
    0, // pad_start(7)
    0, // pad_end(8)
    1, // pad_idx(9)
    1, // jump_read(10)
    1, // jump_write(11)
    1, // length(12)
    0, // mode_avg(13)
    1, // channel(14)
    0, // reg_id(15)
    0, // ddr_addr(16)
};


// Save field minus value
const vector<int> InstTable::SaveFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // hp_id(3)
    0, // bank_id(4)
    0, // bank_addr(5)
    0, // quant_lth(6)
    1, // jump_write(7)
    1, // jump_read(8)
    1, // length(9)
    1, // channel(10)
    0, // reg_id(11)
    0, // ddr_addr(12)
};


// ConvInit field minus value
const vector<int> InstTable::ConvInitFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // act_type(3)
    0, // shift_cut(4)
    1, // jump_read(5)
    1, // stride_out(6)
    0, // calt_mode(7)
    0, // quant_lth(8)
    0, // shift_bias(9)
    1, // jump_read_endl(10)
    1, // stride_h(11)
    1, // kernel_h(12)
    1, // valid_pixel_parallel(13)
    0, // stride_offset_in(14)
    1, // jump_write(15)
    1, // stride_w(16)
    1, // kernel_w(17)
    0, // stride_offset_out(18)
    1, // jump_write_endl(19)
};


// Conv field minus value
const vector<int> InstTable::ConvFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // pad_left(3)
    0, // pad_top(4)
    0, // bank_addr_in(5)
    1, // channel_group(6)
    0, // channel_offset(7)
    0, // pad_right(8)
    0, // pad_bottom(9)
    0, // bank_addr_out(10)
    1, // length(11)
    0, // bank_id_in(12)
    0, // bank_addr_weights(13)
    0, // bank_addr_in_1(14)
    0, // bank_id_out(15)
    0, // bank_addr_bias(16)
    0, // bank_addr_in_3(17)
    0, // bank_addr_in_2(18)
};


// PoolInit field minus value
const vector<int> InstTable::PoolInitFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // shift_cut(3)
    1, // kernel_w(4)
    1, // kernel_h(5)
    1, // jump_read(6)
    1, // jump_write(7)
    1, // stride_out(8)
    0, // stride_offset_out(9)
    1, // valid_pixel_parallel(10)
    0, // stride_offset_in(11)
    1, // stride_w(12)
    1, // stride_h(13)
    0, // pool_type(14)
};


// Pool field minus value
const vector<int> InstTable::PoolFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    1, // channel_group(3)
    0, // bank_addr_in(4)
    1, // jump_write_endl(5)
    0, // pad_left(6)
    0, // pad_top(7)
    0, // pad_right(8)
    0, // pad_bottom(9)
    0, // bank_id_in(10)
    1, // jump_read_endl(11)
    1, // length(12)
    0, // bank_id_out(13)
    0, // bank_addr_in_1(14)
    0, // bank_addr_out(15)
    0, // bank_addr_in_3(16)
    0, // bank_addr_in_2(17)
};


// DWInit field minus value
const vector<int> InstTable::DWInitFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    1, // jump_read(3)
    1, // jump_read_endl(4)
    0, // stride_offset_in(5)
    1, // valid_pixel_parallel(6)
    1, // stride_w(7)
    1, // stride_h(8)
    1, // kernel_w(9)
    1, // kernel_h(10)
    1, // jump_write_endl(11)
    1, // jump_write(12)
    1, // stride_out(13)
    0, // stride_offset_out(14)
};


// DptWise field minus value
const vector<int> InstTable::DptWiseFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    1, // channel_group(3)
    0, // bank_addr_in(4)
    0, // pad_left(5)
    0, // pad_right(6)
    0, // pad_top(7)
    0, // pad_bottom(8)
    0, // channel_offset(9)
    0, // bank_addr_out(10)
    0, // act_type(11)
    1, // length(12)
    0, // bank_id_in(13)
    0, // bank_addr_weights(14)
    0, // bank_addr_in_1(15)
    0, // bank_id_out(16)
    0, // bank_addr_bias(17)
    0, // bank_addr_in_3(18)
    0, // bank_addr_in_2(19)
};


// ElewInit field minus value
const vector<int> InstTable::ElewInitFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // bank_id_in(3)
    0, // shift_read(4)
    1, // jump_read(5)
    1, // jump_read_endl(6)
    0, // id(7)
    0, // bank_addr_in(8)
};


// Elew field minus value
const vector<int> InstTable::ElewFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // bank_id_out(3)
    0, // shift_write(4)
    1, // jump_write(5)
    1, // num(6)
    1, // channel_group(7)
    0, // act_type(8)
    1, // length(9)
    0, // bank_addr_out(10)
    1, // valid_pixel_parallel(11)
    1, // jump_write_endl(12)
};


// Thd field minus value
const vector<int> InstTable::ThdFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    1, // channel_group(3)
    0, // bank_addr_in(4)
    0, // signed_mode(5)
    1, // valid_pixel_parallel(6)
    1, // length(7)
    0, // bank_addr_out(8)
    1, // stride_out(9)
    0, // bank_id_in(10)
    1, // jump_read(11)
    1, // jump_read_endl(12)
    0, // stride_offset_out(13)
    0, // bank_id_out(14)
    1, // jump_write(15)
    1, // jump_write_endl(16)
    0, // param_addr(17)
};


// End field minus value
const vector<int> InstTable::EndFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
};


// DumpBank field minus value
const vector<int> InstTable::DumpBankFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // save_name(3)
    0, // save_fmt(4)
    0, // bank_start(5)
    0, // bank_num(6)
};


// DumpDDR field minus value
const vector<int> InstTable::DumpDDRFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // save_name(3)
    0, // save_fmt(4)
    0, // reg_id(5)
    0, // ddr_start(6)
    0, // ddr_size(7)
};


// DumpDDRSlice field minus value
const vector<int> InstTable::DumpDDRSliceFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // save_name(3)
    0, // save_fmt(4)
    0, // reg_id(5)
    0, // ddr_start(6)
    0, // height(7)
    0, // height_stride(8)
    0, // width(9)
    0, // width_stride(10)
    0, // channel(11)
    0, // channel_stride(12)
};


// Load field position
const vector<map<uint32_t, uint32_t>> InstTable::LoadFieldPos = {
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_OPCODE, 28 }, // 0
        { LOAD_FIELD_DPDON, 24 }, // 1
        { LOAD_FIELD_DPDBY, 20 }, // 2
        { LOAD_FIELD_HP_ID, 18 }, // 3
        { LOAD_FIELD_BANK_ID, 12 }, // 4
        { LOAD_FIELD_BANK_ADDR, 0 }, // 5
    },
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_QUANT_LTH, 31 }, // 6
        { LOAD_FIELD_PAD_START, 26 }, // 7
        { LOAD_FIELD_PAD_END, 21 }, // 8
        { LOAD_FIELD_PAD_IDX, 16 }, // 9
        { LOAD_FIELD_JUMP_READ, 0 }, // 10
    },
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_JUMP_WRITE, 24 }, // 11
        { LOAD_FIELD_LENGTH, 14 }, // 12
        { LOAD_FIELD_MODE_AVG, 12 }, // 13
        { LOAD_FIELD_CHANNEL, 0 }, // 14
    },
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_REG_ID, 29 }, // 15
        { LOAD_FIELD_DDR_ADDR, 0 }, // 16
    },
};


// Save field position
const vector<map<uint32_t, uint32_t>> InstTable::SaveFieldPos = {
    map<uint32_t, uint32_t> {
        { SAVE_FIELD_OPCODE, 28 }, // 0
        { SAVE_FIELD_DPDON, 24 }, // 1
        { SAVE_FIELD_DPDBY, 20 }, // 2
        { SAVE_FIELD_HP_ID, 18 }, // 3
        { SAVE_FIELD_BANK_ID, 12 }, // 4
        { SAVE_FIELD_BANK_ADDR, 0 }, // 5
    },
    map<uint32_t, uint32_t> {
        { SAVE_FIELD_QUANT_LTH, 31 }, // 6
        { SAVE_FIELD_JUMP_WRITE, 0 }, // 7
    },
    map<uint32_t, uint32_t> {
        { SAVE_FIELD_JUMP_READ, 24 }, // 8
        { SAVE_FIELD_LENGTH, 14 }, // 9
        { SAVE_FIELD_CHANNEL, 0 }, // 10
    },
    map<uint32_t, uint32_t> {
        { SAVE_FIELD_REG_ID, 29 }, // 11
        { SAVE_FIELD_DDR_ADDR, 0 }, // 12
    },
};


// ConvInit field position
const vector<map<uint32_t, uint32_t>> InstTable::ConvInitFieldPos = {
    map<uint32_t, uint32_t> {
        { CONVINIT_FIELD_OPCODE, 28 }, // 0
        { CONVINIT_FIELD_DPDON, 24 }, // 1
        { CONVINIT_FIELD_DPDBY, 20 }, // 2
        { CONVINIT_FIELD_ACT_TYPE, 16 }, // 3
        { CONVINIT_FIELD_SHIFT_CUT, 10 }, // 4
        { CONVINIT_FIELD_JUMP_READ, 0 }, // 5
    },
    map<uint32_t, uint32_t> {
        { CONVINIT_FIELD_STRIDE_OUT, 28 }, // 6
        { CONVINIT_FIELD_CALT_MODE, 26 }, // 7
        { CONVINIT_FIELD_QUANT_LTH, 24 }, // 8
        { CONVINIT_FIELD_SHIFT_BIAS, 10 }, // 9
        { CONVINIT_FIELD_JUMP_READ_ENDL, 0 }, // 10
    },
    map<uint32_t, uint32_t> {
        { CONVINIT_FIELD_STRIDE_H, 28 }, // 11
        { CONVINIT_FIELD_KERNEL_H, 24 }, // 12
        { CONVINIT_FIELD_VALID_PIXEL_PARALLEL, 13 }, // 13
        { CONVINIT_FIELD_STRIDE_OFFSET_IN, 10 }, // 14
        { CONVINIT_FIELD_JUMP_WRITE, 0 }, // 15
    },
    map<uint32_t, uint32_t> {
        { CONVINIT_FIELD_STRIDE_W, 28 }, // 16
        { CONVINIT_FIELD_KERNEL_W, 24 }, // 17
        { CONVINIT_FIELD_STRIDE_OFFSET_OUT, 10 }, // 18
        { CONVINIT_FIELD_JUMP_WRITE_ENDL, 0 }, // 19
    },
};


// Conv field position
const vector<map<uint32_t, uint32_t>> InstTable::ConvFieldPos = {
    map<uint32_t, uint32_t> {
        { CONV_FIELD_OPCODE, 28 }, // 0
        { CONV_FIELD_DPDON, 24 }, // 1
        { CONV_FIELD_DPDBY, 20 }, // 2
        { CONV_FIELD_PAD_LEFT, 16 }, // 3
        { CONV_FIELD_PAD_TOP, 12 }, // 4
        { CONV_FIELD_BANK_ADDR_IN, 0 }, // 5
    },
    map<uint32_t, uint32_t> {
        { CONV_FIELD_CHANNEL_GROUP, 24 }, // 6
        { CONV_FIELD_CHANNEL_OFFSET, 20 }, // 7
        { CONV_FIELD_PAD_RIGHT, 16 }, // 8
        { CONV_FIELD_PAD_BOTTOM, 12 }, // 9
        { CONV_FIELD_BANK_ADDR_OUT, 0 }, // 10
    },
    map<uint32_t, uint32_t> {
        { CONV_FIELD_LENGTH, 22 }, // 11
        { CONV_FIELD_BANK_ID_IN, 12 }, // 12
        { CONV_FIELD_BANK_ADDR_WEIGHTS, 0 }, // 13
    },
    map<uint32_t, uint32_t> {
        { CONV_FIELD_BANK_ADDR_IN_1, 18 }, // 14
        { CONV_FIELD_BANK_ID_OUT, 12 }, // 15
        { CONV_FIELD_BANK_ADDR_BIAS, 0 }, // 16
    },
    map<uint32_t, uint32_t> {
        { CONV_FIELD_BANK_ADDR_IN_3, 12 }, // 17
        { CONV_FIELD_BANK_ADDR_IN_2, 0 }, // 18
    },
};


// PoolInit field position
const vector<map<uint32_t, uint32_t>> InstTable::PoolInitFieldPos = {
    map<uint32_t, uint32_t> {
        { POOLINIT_FIELD_OPCODE, 28 }, // 0
        { POOLINIT_FIELD_DPDON, 24 }, // 1
        { POOLINIT_FIELD_DPDBY, 20 }, // 2
        { POOLINIT_FIELD_SHIFT_CUT, 16 }, // 3
        { POOLINIT_FIELD_KERNEL_W, 13 }, // 4
        { POOLINIT_FIELD_KERNEL_H, 10 }, // 5
        { POOLINIT_FIELD_JUMP_READ, 0 }, // 6
    },
    map<uint32_t, uint32_t> {
        { POOLINIT_FIELD_JUMP_WRITE, 21 }, // 7
        { POOLINIT_FIELD_STRIDE_OUT, 17 }, // 8
        { POOLINIT_FIELD_STRIDE_OFFSET_OUT, 14 }, // 9
        { POOLINIT_FIELD_VALID_PIXEL_PARALLEL, 11 }, // 10
        { POOLINIT_FIELD_STRIDE_OFFSET_IN, 8 }, // 11
        { POOLINIT_FIELD_STRIDE_W, 5 }, // 12
        { POOLINIT_FIELD_STRIDE_H, 2 }, // 13
        { POOLINIT_FIELD_POOL_TYPE, 0 }, // 14
    },
};


// Pool field position
const vector<map<uint32_t, uint32_t>> InstTable::PoolFieldPos = {
    map<uint32_t, uint32_t> {
        { POOL_FIELD_OPCODE, 28 }, // 0
        { POOL_FIELD_DPDON, 24 }, // 1
        { POOL_FIELD_DPDBY, 20 }, // 2
        { POOL_FIELD_CHANNEL_GROUP, 12 }, // 3
        { POOL_FIELD_BANK_ADDR_IN, 0 }, // 4
    },
    map<uint32_t, uint32_t> {
        { POOL_FIELD_JUMP_WRITE_ENDL, 18 }, // 5
        { POOL_FIELD_PAD_LEFT, 15 }, // 6
        { POOL_FIELD_PAD_TOP, 12 }, // 7
        { POOL_FIELD_PAD_RIGHT, 9 }, // 8
        { POOL_FIELD_PAD_BOTTOM, 6 }, // 9
        { POOL_FIELD_BANK_ID_IN, 0 }, // 10
    },
    map<uint32_t, uint32_t> {
        { POOL_FIELD_JUMP_READ_ENDL, 16 }, // 11
        { POOL_FIELD_LENGTH, 6 }, // 12
        { POOL_FIELD_BANK_ID_OUT, 0 }, // 13
    },
    map<uint32_t, uint32_t> {
        { POOL_FIELD_BANK_ADDR_IN_1, 12 }, // 14
        { POOL_FIELD_BANK_ADDR_OUT, 0 }, // 15
    },
    map<uint32_t, uint32_t> {
        { POOL_FIELD_BANK_ADDR_IN_3, 12 }, // 16
        { POOL_FIELD_BANK_ADDR_IN_2, 0 }, // 17
    },
};


// DWInit field position
const vector<map<uint32_t, uint32_t>> InstTable::DWInitFieldPos = {
    map<uint32_t, uint32_t> {
        { DWINIT_FIELD_OPCODE, 28 }, // 0
        { DWINIT_FIELD_DPDON, 24 }, // 1
        { DWINIT_FIELD_DPDBY, 20 }, // 2
        { DWINIT_FIELD_JUMP_READ, 10 }, // 3
        { DWINIT_FIELD_JUMP_READ_ENDL, 0 }, // 4
    },
    map<uint32_t, uint32_t> {
        { DWINIT_FIELD_STRIDE_OFFSET_IN, 29 }, // 5
        { DWINIT_FIELD_VALID_PIXEL_PARALLEL, 26 }, // 6
        { DWINIT_FIELD_STRIDE_W, 22 }, // 7
        { DWINIT_FIELD_STRIDE_H, 18 }, // 8
        { DWINIT_FIELD_KERNEL_W, 14 }, // 9
        { DWINIT_FIELD_KERNEL_H, 10 }, // 10
        { DWINIT_FIELD_JUMP_WRITE_ENDL, 0 }, // 11
    },
    map<uint32_t, uint32_t> {
        { DWINIT_FIELD_JUMP_WRITE, 19 }, // 12
        { DWINIT_FIELD_STRIDE_OUT, 15 }, // 13
        { DWINIT_FIELD_STRIDE_OFFSET_OUT, 12 }, // 14
    },
};


// DptWise field position
const vector<map<uint32_t, uint32_t>> InstTable::DptWiseFieldPos = {
    map<uint32_t, uint32_t> {
        { DPTWISE_FIELD_OPCODE, 28 }, // 0
        { DPTWISE_FIELD_DPDON, 24 }, // 1
        { DPTWISE_FIELD_DPDBY, 20 }, // 2
        { DPTWISE_FIELD_CHANNEL_GROUP, 12 }, // 3
        { DPTWISE_FIELD_BANK_ADDR_IN, 0 }, // 4
    },
    map<uint32_t, uint32_t> {
        { DPTWISE_FIELD_PAD_LEFT, 28 }, // 5
        { DPTWISE_FIELD_PAD_RIGHT, 24 }, // 6
        { DPTWISE_FIELD_PAD_TOP, 20 }, // 7
        { DPTWISE_FIELD_PAD_BOTTOM, 16 }, // 8
        { DPTWISE_FIELD_CHANNEL_OFFSET, 12 }, // 9
        { DPTWISE_FIELD_BANK_ADDR_OUT, 0 }, // 10
    },
    map<uint32_t, uint32_t> {
        { DPTWISE_FIELD_ACT_TYPE, 28 }, // 11
        { DPTWISE_FIELD_LENGTH, 18 }, // 12
        { DPTWISE_FIELD_BANK_ID_IN, 12 }, // 13
        { DPTWISE_FIELD_BANK_ADDR_WEIGHTS, 0 }, // 14
    },
    map<uint32_t, uint32_t> {
        { DPTWISE_FIELD_BANK_ADDR_IN_1, 18 }, // 15
        { DPTWISE_FIELD_BANK_ID_OUT, 12 }, // 16
        { DPTWISE_FIELD_BANK_ADDR_BIAS, 0 }, // 17
    },
    map<uint32_t, uint32_t> {
        { DPTWISE_FIELD_BANK_ADDR_IN_3, 12 }, // 18
        { DPTWISE_FIELD_BANK_ADDR_IN_2, 0 }, // 19
    },
};


// ElewInit field position
const vector<map<uint32_t, uint32_t>> InstTable::ElewInitFieldPos = {
    map<uint32_t, uint32_t> {
        { ELEWINIT_FIELD_OPCODE, 28 }, // 0
        { ELEWINIT_FIELD_DPDON, 24 }, // 1
        { ELEWINIT_FIELD_DPDBY, 20 }, // 2
        { ELEWINIT_FIELD_BANK_ID_IN, 14 }, // 3
        { ELEWINIT_FIELD_SHIFT_READ, 10 }, // 4
        { ELEWINIT_FIELD_JUMP_READ, 0 }, // 5
    },
    map<uint32_t, uint32_t> {
        { ELEWINIT_FIELD_JUMP_READ_ENDL, 14 }, // 6
        { ELEWINIT_FIELD_ID, 12 }, // 7
        { ELEWINIT_FIELD_BANK_ADDR_IN, 0 }, // 8
    },
};


// Elew field position
const vector<map<uint32_t, uint32_t>> InstTable::ElewFieldPos = {
    map<uint32_t, uint32_t> {
        { ELEW_FIELD_OPCODE, 28 }, // 0
        { ELEW_FIELD_DPDON, 24 }, // 1
        { ELEW_FIELD_DPDBY, 20 }, // 2
        { ELEW_FIELD_BANK_ID_OUT, 14 }, // 3
        { ELEW_FIELD_SHIFT_WRITE, 10 }, // 4
        { ELEW_FIELD_JUMP_WRITE, 0 }, // 5
    },
    map<uint32_t, uint32_t> {
        { ELEW_FIELD_NUM, 30 }, // 6
        { ELEW_FIELD_CHANNEL_GROUP, 22 }, // 7
        { ELEW_FIELD_ACT_TYPE, 21 }, // 8
        { ELEW_FIELD_LENGTH, 12 }, // 9
        { ELEW_FIELD_BANK_ADDR_OUT, 0 }, // 10
    },
    map<uint32_t, uint32_t> {
        { ELEW_FIELD_VALID_PIXEL_PARALLEL, 12 }, // 11
        { ELEW_FIELD_JUMP_WRITE_ENDL, 0 }, // 12
    },
};


// Thd field position
const vector<map<uint32_t, uint32_t>> InstTable::ThdFieldPos = {
    map<uint32_t, uint32_t> {
        { THD_FIELD_OPCODE, 28 }, // 0
        { THD_FIELD_DPDON, 24 }, // 1
        { THD_FIELD_DPDBY, 20 }, // 2
        { THD_FIELD_CHANNEL_GROUP, 12 }, // 3
        { THD_FIELD_BANK_ADDR_IN, 0 }, // 4
    },
    map<uint32_t, uint32_t> {
        { THD_FIELD_SIGNED_MODE, 24 }, // 5
        { THD_FIELD_VALID_PIXEL_PARALLEL, 21 }, // 6
        { THD_FIELD_LENGTH, 12 }, // 7
        { THD_FIELD_BANK_ADDR_OUT, 0 }, // 8
    },
    map<uint32_t, uint32_t> {
        { THD_FIELD_STRIDE_OUT, 28 }, // 9
        { THD_FIELD_BANK_ID_IN, 22 }, // 10
        { THD_FIELD_JUMP_READ, 12 }, // 11
        { THD_FIELD_JUMP_READ_ENDL, 0 }, // 12
    },
    map<uint32_t, uint32_t> {
        { THD_FIELD_STRIDE_OFFSET_OUT, 28 }, // 13
        { THD_FIELD_BANK_ID_OUT, 22 }, // 14
        { THD_FIELD_JUMP_WRITE, 12 }, // 15
        { THD_FIELD_JUMP_WRITE_ENDL, 0 }, // 16
    },
    map<uint32_t, uint32_t> {
        { THD_FIELD_PARAM_ADDR, 0 }, // 17
    },
};


// End field position
const vector<map<uint32_t, uint32_t>> InstTable::EndFieldPos = {
    map<uint32_t, uint32_t> {
        { END_FIELD_OPCODE, 28 }, // 0
        { END_FIELD_DPDON, 24 }, // 1
        { END_FIELD_DPDBY, 20 }, // 2
    },
};


// DumpBank field position
const vector<map<uint32_t, uint32_t>> InstTable::DumpBankFieldPos = {
    map<uint32_t, uint32_t> {
        { DUMPBANK_FIELD_OPCODE, 28 }, // 0
        { DUMPBANK_FIELD_DPDON, 24 }, // 1
        { DUMPBANK_FIELD_DPDBY, 20 }, // 2
        { DUMPBANK_FIELD_SAVE_NAME, 0 }, // 3
    },
    map<uint32_t, uint32_t> {
        { DUMPBANK_FIELD_SAVE_FMT, 26 }, // 4
        { DUMPBANK_FIELD_BANK_START, 18 }, // 5
        { DUMPBANK_FIELD_BANK_NUM, 10 }, // 6
    },
};


// DumpDDR field position
const vector<map<uint32_t, uint32_t>> InstTable::DumpDDRFieldPos = {
    map<uint32_t, uint32_t> {
        { DUMPDDR_FIELD_OPCODE, 28 }, // 0
        { DUMPDDR_FIELD_DPDON, 24 }, // 1
        { DUMPDDR_FIELD_DPDBY, 20 }, // 2
        { DUMPDDR_FIELD_SAVE_NAME, 0 }, // 3
    },
    map<uint32_t, uint32_t> {
        { DUMPDDR_FIELD_SAVE_FMT, 26 }, // 4
        { DUMPDDR_FIELD_REG_ID, 20 }, // 5
    },
    map<uint32_t, uint32_t> {
        { DUMPDDR_FIELD_DDR_START, 0 }, // 6
    },
    map<uint32_t, uint32_t> {
        { DUMPDDR_FIELD_DDR_SIZE, 0 }, // 7
    },
};


// DumpDDRSlice field position
const vector<map<uint32_t, uint32_t>> InstTable::DumpDDRSliceFieldPos = {
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_OPCODE, 28 }, // 0
        { DUMPDDRSLICE_FIELD_DPDON, 24 }, // 1
        { DUMPDDRSLICE_FIELD_DPDBY, 20 }, // 2
        { DUMPDDRSLICE_FIELD_SAVE_NAME, 0 }, // 3
    },
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_SAVE_FMT, 26 }, // 4
        { DUMPDDRSLICE_FIELD_REG_ID, 20 }, // 5
    },
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_DDR_START, 0 }, // 6
    },
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_HEIGHT, 16 }, // 7
        { DUMPDDRSLICE_FIELD_HEIGHT_STRIDE, 0 }, // 8
    },
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_WIDTH, 16 }, // 9
        { DUMPDDRSLICE_FIELD_WIDTH_STRIDE, 0 }, // 10
    },
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_CHANNEL, 16 }, // 11
        { DUMPDDRSLICE_FIELD_CHANNEL_STRIDE, 0 }, // 12
    },
};


// Load field length
const vector<map<uint32_t, uint32_t>> InstTable::LoadFieldLen = {
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_OPCODE, 4 }, // 0
        { LOAD_FIELD_DPDON, 4 }, // 1
        { LOAD_FIELD_DPDBY, 4 }, // 2
        { LOAD_FIELD_HP_ID, 2 }, // 3
        { LOAD_FIELD_BANK_ID, 6 }, // 4
        { LOAD_FIELD_BANK_ADDR, 12 }, // 5
    },
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_QUANT_LTH, 1 }, // 6
        { LOAD_FIELD_PAD_START, 5 }, // 7
        { LOAD_FIELD_PAD_END, 5 }, // 8
        { LOAD_FIELD_PAD_IDX, 5 }, // 9
        { LOAD_FIELD_JUMP_READ, 16 }, // 10
    },
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_JUMP_WRITE, 8 }, // 11
        { LOAD_FIELD_LENGTH, 10 }, // 12
        { LOAD_FIELD_MODE_AVG, 2 }, // 13
        { LOAD_FIELD_CHANNEL, 12 }, // 14
    },
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_REG_ID, 3 }, // 15
        { LOAD_FIELD_DDR_ADDR, 29 }, // 16
    },
};


// Save field length
const vector<map<uint32_t, uint32_t>> InstTable::SaveFieldLen = {
    map<uint32_t, uint32_t> {
        { SAVE_FIELD_OPCODE, 4 }, // 0
        { SAVE_FIELD_DPDON, 4 }, // 1
        { SAVE_FIELD_DPDBY, 4 }, // 2
        { SAVE_FIELD_HP_ID, 2 }, // 3
        { SAVE_FIELD_BANK_ID, 6 }, // 4
        { SAVE_FIELD_BANK_ADDR, 12 }, // 5
    },
    map<uint32_t, uint32_t> {
        { SAVE_FIELD_QUANT_LTH, 1 }, // 6
        { SAVE_FIELD_JUMP_WRITE, 16 }, // 7
    },
    map<uint32_t, uint32_t> {
        { SAVE_FIELD_JUMP_READ, 8 }, // 8
        { SAVE_FIELD_LENGTH, 10 }, // 9
        { SAVE_FIELD_CHANNEL, 12 }, // 10
    },
    map<uint32_t, uint32_t> {
        { SAVE_FIELD_REG_ID, 3 }, // 11
        { SAVE_FIELD_DDR_ADDR, 29 }, // 12
    },
};


// ConvInit field length
const vector<map<uint32_t, uint32_t>> InstTable::ConvInitFieldLen = {
    map<uint32_t, uint32_t> {
        { CONVINIT_FIELD_OPCODE, 4 }, // 0
        { CONVINIT_FIELD_DPDON, 4 }, // 1
        { CONVINIT_FIELD_DPDBY, 4 }, // 2
        { CONVINIT_FIELD_ACT_TYPE, 4 }, // 3
        { CONVINIT_FIELD_SHIFT_CUT, 6 }, // 4
        { CONVINIT_FIELD_JUMP_READ, 10 }, // 5
    },
    map<uint32_t, uint32_t> {
        { CONVINIT_FIELD_STRIDE_OUT, 4 }, // 6
        { CONVINIT_FIELD_CALT_MODE, 2 }, // 7
        { CONVINIT_FIELD_QUANT_LTH, 2 }, // 8
        { CONVINIT_FIELD_SHIFT_BIAS, 6 }, // 9
        { CONVINIT_FIELD_JUMP_READ_ENDL, 10 }, // 10
    },
    map<uint32_t, uint32_t> {
        { CONVINIT_FIELD_STRIDE_H, 4 }, // 11
        { CONVINIT_FIELD_KERNEL_H, 4 }, // 12
        { CONVINIT_FIELD_VALID_PIXEL_PARALLEL, 3 }, // 13
        { CONVINIT_FIELD_STRIDE_OFFSET_IN, 3 }, // 14
        { CONVINIT_FIELD_JUMP_WRITE, 10 }, // 15
    },
    map<uint32_t, uint32_t> {
        { CONVINIT_FIELD_STRIDE_W, 4 }, // 16
        { CONVINIT_FIELD_KERNEL_W, 4 }, // 17
        { CONVINIT_FIELD_STRIDE_OFFSET_OUT, 3 }, // 18
        { CONVINIT_FIELD_JUMP_WRITE_ENDL, 10 }, // 19
    },
};


// Conv field length
const vector<map<uint32_t, uint32_t>> InstTable::ConvFieldLen = {
    map<uint32_t, uint32_t> {
        { CONV_FIELD_OPCODE, 4 }, // 0
        { CONV_FIELD_DPDON, 4 }, // 1
        { CONV_FIELD_DPDBY, 4 }, // 2
        { CONV_FIELD_PAD_LEFT, 4 }, // 3
        { CONV_FIELD_PAD_TOP, 4 }, // 4
        { CONV_FIELD_BANK_ADDR_IN, 12 }, // 5
    },
    map<uint32_t, uint32_t> {
        { CONV_FIELD_CHANNEL_GROUP, 8 }, // 6
        { CONV_FIELD_CHANNEL_OFFSET, 4 }, // 7
        { CONV_FIELD_PAD_RIGHT, 4 }, // 8
        { CONV_FIELD_PAD_BOTTOM, 4 }, // 9
        { CONV_FIELD_BANK_ADDR_OUT, 12 }, // 10
    },
    map<uint32_t, uint32_t> {
        { CONV_FIELD_LENGTH, 10 }, // 11
        { CONV_FIELD_BANK_ID_IN, 6 }, // 12
        { CONV_FIELD_BANK_ADDR_WEIGHTS, 12 }, // 13
    },
    map<uint32_t, uint32_t> {
        { CONV_FIELD_BANK_ADDR_IN_1, 12 }, // 14
        { CONV_FIELD_BANK_ID_OUT, 6 }, // 15
        { CONV_FIELD_BANK_ADDR_BIAS, 12 }, // 16
    },
    map<uint32_t, uint32_t> {
        { CONV_FIELD_BANK_ADDR_IN_3, 12 }, // 17
        { CONV_FIELD_BANK_ADDR_IN_2, 12 }, // 18
    },
};


// PoolInit field length
const vector<map<uint32_t, uint32_t>> InstTable::PoolInitFieldLen = {
    map<uint32_t, uint32_t> {
        { POOLINIT_FIELD_OPCODE, 4 }, // 0
        { POOLINIT_FIELD_DPDON, 4 }, // 1
        { POOLINIT_FIELD_DPDBY, 4 }, // 2
        { POOLINIT_FIELD_SHIFT_CUT, 4 }, // 3
        { POOLINIT_FIELD_KERNEL_W, 3 }, // 4
        { POOLINIT_FIELD_KERNEL_H, 3 }, // 5
        { POOLINIT_FIELD_JUMP_READ, 10 }, // 6
    },
    map<uint32_t, uint32_t> {
        { POOLINIT_FIELD_JUMP_WRITE, 10 }, // 7
        { POOLINIT_FIELD_STRIDE_OUT, 4 }, // 8
        { POOLINIT_FIELD_STRIDE_OFFSET_OUT, 3 }, // 9
        { POOLINIT_FIELD_VALID_PIXEL_PARALLEL, 3 }, // 10
        { POOLINIT_FIELD_STRIDE_OFFSET_IN, 3 }, // 11
        { POOLINIT_FIELD_STRIDE_W, 3 }, // 12
        { POOLINIT_FIELD_STRIDE_H, 3 }, // 13
        { POOLINIT_FIELD_POOL_TYPE, 2 }, // 14
    },
};


// Pool field length
const vector<map<uint32_t, uint32_t>> InstTable::PoolFieldLen = {
    map<uint32_t, uint32_t> {
        { POOL_FIELD_OPCODE, 4 }, // 0
        { POOL_FIELD_DPDON, 4 }, // 1
        { POOL_FIELD_DPDBY, 4 }, // 2
        { POOL_FIELD_CHANNEL_GROUP, 8 }, // 3
        { POOL_FIELD_BANK_ADDR_IN, 12 }, // 4
    },
    map<uint32_t, uint32_t> {
        { POOL_FIELD_JUMP_WRITE_ENDL, 12 }, // 5
        { POOL_FIELD_PAD_LEFT, 3 }, // 6
        { POOL_FIELD_PAD_TOP, 3 }, // 7
        { POOL_FIELD_PAD_RIGHT, 3 }, // 8
        { POOL_FIELD_PAD_BOTTOM, 3 }, // 9
        { POOL_FIELD_BANK_ID_IN, 6 }, // 10
    },
    map<uint32_t, uint32_t> {
        { POOL_FIELD_JUMP_READ_ENDL, 12 }, // 11
        { POOL_FIELD_LENGTH, 10 }, // 12
        { POOL_FIELD_BANK_ID_OUT, 6 }, // 13
    },
    map<uint32_t, uint32_t> {
        { POOL_FIELD_BANK_ADDR_IN_1, 12 }, // 14
        { POOL_FIELD_BANK_ADDR_OUT, 12 }, // 15
    },
    map<uint32_t, uint32_t> {
        { POOL_FIELD_BANK_ADDR_IN_3, 12 }, // 16
        { POOL_FIELD_BANK_ADDR_IN_2, 12 }, // 17
    },
};


// DWInit field length
const vector<map<uint32_t, uint32_t>> InstTable::DWInitFieldLen = {
    map<uint32_t, uint32_t> {
        { DWINIT_FIELD_OPCODE, 4 }, // 0
        { DWINIT_FIELD_DPDON, 4 }, // 1
        { DWINIT_FIELD_DPDBY, 4 }, // 2
        { DWINIT_FIELD_JUMP_READ, 10 }, // 3
        { DWINIT_FIELD_JUMP_READ_ENDL, 10 }, // 4
    },
    map<uint32_t, uint32_t> {
        { DWINIT_FIELD_STRIDE_OFFSET_IN, 3 }, // 5
        { DWINIT_FIELD_VALID_PIXEL_PARALLEL, 3 }, // 6
        { DWINIT_FIELD_STRIDE_W, 4 }, // 7
        { DWINIT_FIELD_STRIDE_H, 4 }, // 8
        { DWINIT_FIELD_KERNEL_W, 4 }, // 9
        { DWINIT_FIELD_KERNEL_H, 4 }, // 10
        { DWINIT_FIELD_JUMP_WRITE_ENDL, 10 }, // 11
    },
    map<uint32_t, uint32_t> {
        { DWINIT_FIELD_JUMP_WRITE, 10 }, // 12
        { DWINIT_FIELD_STRIDE_OUT, 4 }, // 13
        { DWINIT_FIELD_STRIDE_OFFSET_OUT, 3 }, // 14
    },
};


// DptWise field length
const vector<map<uint32_t, uint32_t>> InstTable::DptWiseFieldLen = {
    map<uint32_t, uint32_t> {
        { DPTWISE_FIELD_OPCODE, 4 }, // 0
        { DPTWISE_FIELD_DPDON, 4 }, // 1
        { DPTWISE_FIELD_DPDBY, 4 }, // 2
        { DPTWISE_FIELD_CHANNEL_GROUP, 8 }, // 3
        { DPTWISE_FIELD_BANK_ADDR_IN, 12 }, // 4
    },
    map<uint32_t, uint32_t> {
        { DPTWISE_FIELD_PAD_LEFT, 4 }, // 5
        { DPTWISE_FIELD_PAD_RIGHT, 4 }, // 6
        { DPTWISE_FIELD_PAD_TOP, 4 }, // 7
        { DPTWISE_FIELD_PAD_BOTTOM, 4 }, // 8
        { DPTWISE_FIELD_CHANNEL_OFFSET, 4 }, // 9
        { DPTWISE_FIELD_BANK_ADDR_OUT, 12 }, // 10
    },
    map<uint32_t, uint32_t> {
        { DPTWISE_FIELD_ACT_TYPE, 4 }, // 11
        { DPTWISE_FIELD_LENGTH, 10 }, // 12
        { DPTWISE_FIELD_BANK_ID_IN, 6 }, // 13
        { DPTWISE_FIELD_BANK_ADDR_WEIGHTS, 12 }, // 14
    },
    map<uint32_t, uint32_t> {
        { DPTWISE_FIELD_BANK_ADDR_IN_1, 12 }, // 15
        { DPTWISE_FIELD_BANK_ID_OUT, 6 }, // 16
        { DPTWISE_FIELD_BANK_ADDR_BIAS, 12 }, // 17
    },
    map<uint32_t, uint32_t> {
        { DPTWISE_FIELD_BANK_ADDR_IN_3, 12 }, // 18
        { DPTWISE_FIELD_BANK_ADDR_IN_2, 12 }, // 19
    },
};


// ElewInit field length
const vector<map<uint32_t, uint32_t>> InstTable::ElewInitFieldLen = {
    map<uint32_t, uint32_t> {
        { ELEWINIT_FIELD_OPCODE, 4 }, // 0
        { ELEWINIT_FIELD_DPDON, 4 }, // 1
        { ELEWINIT_FIELD_DPDBY, 4 }, // 2
        { ELEWINIT_FIELD_BANK_ID_IN, 6 }, // 3
        { ELEWINIT_FIELD_SHIFT_READ, 4 }, // 4
        { ELEWINIT_FIELD_JUMP_READ, 10 }, // 5
    },
    map<uint32_t, uint32_t> {
        { ELEWINIT_FIELD_JUMP_READ_ENDL, 12 }, // 6
        { ELEWINIT_FIELD_ID, 2 }, // 7
        { ELEWINIT_FIELD_BANK_ADDR_IN, 12 }, // 8
    },
};


// Elew field length
const vector<map<uint32_t, uint32_t>> InstTable::ElewFieldLen = {
    map<uint32_t, uint32_t> {
        { ELEW_FIELD_OPCODE, 4 }, // 0
        { ELEW_FIELD_DPDON, 4 }, // 1
        { ELEW_FIELD_DPDBY, 4 }, // 2
        { ELEW_FIELD_BANK_ID_OUT, 6 }, // 3
        { ELEW_FIELD_SHIFT_WRITE, 4 }, // 4
        { ELEW_FIELD_JUMP_WRITE, 10 }, // 5
    },
    map<uint32_t, uint32_t> {
        { ELEW_FIELD_NUM, 2 }, // 6
        { ELEW_FIELD_CHANNEL_GROUP, 8 }, // 7
        { ELEW_FIELD_ACT_TYPE, 1 }, // 8
        { ELEW_FIELD_LENGTH, 9 }, // 9
        { ELEW_FIELD_BANK_ADDR_OUT, 12 }, // 10
    },
    map<uint32_t, uint32_t> {
        { ELEW_FIELD_VALID_PIXEL_PARALLEL, 3 }, // 11
        { ELEW_FIELD_JUMP_WRITE_ENDL, 12 }, // 12
    },
};


// Thd field length
const vector<map<uint32_t, uint32_t>> InstTable::ThdFieldLen = {
    map<uint32_t, uint32_t> {
        { THD_FIELD_OPCODE, 4 }, // 0
        { THD_FIELD_DPDON, 4 }, // 1
        { THD_FIELD_DPDBY, 4 }, // 2
        { THD_FIELD_CHANNEL_GROUP, 8 }, // 3
        { THD_FIELD_BANK_ADDR_IN, 12 }, // 4
    },
    map<uint32_t, uint32_t> {
        { THD_FIELD_SIGNED_MODE, 1 }, // 5
        { THD_FIELD_VALID_PIXEL_PARALLEL, 3 }, // 6
        { THD_FIELD_LENGTH, 9 }, // 7
        { THD_FIELD_BANK_ADDR_OUT, 12 }, // 8
    },
    map<uint32_t, uint32_t> {
        { THD_FIELD_STRIDE_OUT, 4 }, // 9
        { THD_FIELD_BANK_ID_IN, 6 }, // 10
        { THD_FIELD_JUMP_READ, 10 }, // 11
        { THD_FIELD_JUMP_READ_ENDL, 12 }, // 12
    },
    map<uint32_t, uint32_t> {
        { THD_FIELD_STRIDE_OFFSET_OUT, 4 }, // 13
        { THD_FIELD_BANK_ID_OUT, 6 }, // 14
        { THD_FIELD_JUMP_WRITE, 10 }, // 15
        { THD_FIELD_JUMP_WRITE_ENDL, 12 }, // 16
    },
    map<uint32_t, uint32_t> {
        { THD_FIELD_PARAM_ADDR, 12 }, // 17
    },
};


// End field length
const vector<map<uint32_t, uint32_t>> InstTable::EndFieldLen = {
    map<uint32_t, uint32_t> {
        { END_FIELD_OPCODE, 4 }, // 0
        { END_FIELD_DPDON, 4 }, // 1
        { END_FIELD_DPDBY, 4 }, // 2
    },
};


// DumpBank field length
const vector<map<uint32_t, uint32_t>> InstTable::DumpBankFieldLen = {
    map<uint32_t, uint32_t> {
        { DUMPBANK_FIELD_OPCODE, 4 }, // 0
        { DUMPBANK_FIELD_DPDON, 4 }, // 1
        { DUMPBANK_FIELD_DPDBY, 4 }, // 2
        { DUMPBANK_FIELD_SAVE_NAME, 20 }, // 3
    },
    map<uint32_t, uint32_t> {
        { DUMPBANK_FIELD_SAVE_FMT, 6 }, // 4
        { DUMPBANK_FIELD_BANK_START, 8 }, // 5
        { DUMPBANK_FIELD_BANK_NUM, 8 }, // 6
    },
};


// DumpDDR field length
const vector<map<uint32_t, uint32_t>> InstTable::DumpDDRFieldLen = {
    map<uint32_t, uint32_t> {
        { DUMPDDR_FIELD_OPCODE, 4 }, // 0
        { DUMPDDR_FIELD_DPDON, 4 }, // 1
        { DUMPDDR_FIELD_DPDBY, 4 }, // 2
        { DUMPDDR_FIELD_SAVE_NAME, 20 }, // 3
    },
    map<uint32_t, uint32_t> {
        { DUMPDDR_FIELD_SAVE_FMT, 6 }, // 4
        { DUMPDDR_FIELD_REG_ID, 6 }, // 5
    },
    map<uint32_t, uint32_t> {
        { DUMPDDR_FIELD_DDR_START, 32 }, // 6
    },
    map<uint32_t, uint32_t> {
        { DUMPDDR_FIELD_DDR_SIZE, 32 }, // 7
    },
};


// DumpDDRSlice field length
const vector<map<uint32_t, uint32_t>> InstTable::DumpDDRSliceFieldLen = {
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_OPCODE, 4 }, // 0
        { DUMPDDRSLICE_FIELD_DPDON, 4 }, // 1
        { DUMPDDRSLICE_FIELD_DPDBY, 4 }, // 2
        { DUMPDDRSLICE_FIELD_SAVE_NAME, 20 }, // 3
    },
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_SAVE_FMT, 6 }, // 4
        { DUMPDDRSLICE_FIELD_REG_ID, 6 }, // 5
    },
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_DDR_START, 32 }, // 6
    },
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_HEIGHT, 16 }, // 7
        { DUMPDDRSLICE_FIELD_HEIGHT_STRIDE, 16 }, // 8
    },
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_WIDTH, 16 }, // 9
        { DUMPDDRSLICE_FIELD_WIDTH_STRIDE, 16 }, // 10
    },
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_CHANNEL, 16 }, // 11
        { DUMPDDRSLICE_FIELD_CHANNEL_STRIDE, 16 }, // 12
    },
};


// inst opcode to inst type map
const unordered_map<uint32_t, uint32_t> InstTable::OPCode2InstType = {
    { 0x00, INST_TYPE_LOAD }, // 0
    { 0x04, INST_TYPE_SAVE }, // 1
    { 0x09, INST_TYPE_CONVINIT }, // 2
    { 0x08, INST_TYPE_CONV }, // 3
    { 0x06, INST_TYPE_POOLINIT }, // 4
    { 0x0C, INST_TYPE_POOL }, // 5
    { 0x0B, INST_TYPE_DWINIT }, // 6
    { 0x0A, INST_TYPE_DPTWISE }, // 7
    { 0x0D, INST_TYPE_ELEWINIT }, // 8
    { 0x0E, INST_TYPE_ELEW }, // 9
    { 0x0F, INST_TYPE_THD }, // 10
    { 0x07, INST_TYPE_END }, // 11
    { 0xFF, INST_TYPE_DUMPBANK }, // 12
    { 0xFE, INST_TYPE_DUMPDDR }, // 13
    { 0xFD, INST_TYPE_DUMPDDRSLICE }, // 14
};


// inst name to inst type map
const unordered_map<string, uint32_t> InstTable::InstName2InstType = {
    { "LOAD", INST_TYPE_LOAD }, // 0
    { "SAVE", INST_TYPE_SAVE }, // 1
    { "CONVINIT", INST_TYPE_CONVINIT }, // 2
    { "CONV", INST_TYPE_CONV }, // 3
    { "POOLINIT", INST_TYPE_POOLINIT }, // 4
    { "POOL", INST_TYPE_POOL }, // 5
    { "DWINIT", INST_TYPE_DWINIT }, // 6
    { "DPTWISE", INST_TYPE_DPTWISE }, // 7
    { "ELEWINIT", INST_TYPE_ELEWINIT }, // 8
    { "ELEW", INST_TYPE_ELEW }, // 9
    { "THD", INST_TYPE_THD }, // 10
    { "END", INST_TYPE_END }, // 11
    { "DUMPBANK", INST_TYPE_DUMPBANK }, // 12
    { "DUMPDDR", INST_TYPE_DUMPDDR }, // 13
    { "DUMPDDRSLICE", INST_TYPE_DUMPDDRSLICE }, // 14
};


// inst type to field name map
const unordered_map<uint32_t, const vector<string> &> InstTable::InstFieldName = {
    { INST_TYPE_LOAD, LoadFieldName }, // 0
    { INST_TYPE_SAVE, SaveFieldName }, // 1
    { INST_TYPE_CONVINIT, ConvInitFieldName }, // 2
    { INST_TYPE_CONV, ConvFieldName }, // 3
    { INST_TYPE_POOLINIT, PoolInitFieldName }, // 4
    { INST_TYPE_POOL, PoolFieldName }, // 5
    { INST_TYPE_DWINIT, DWInitFieldName }, // 6
    { INST_TYPE_DPTWISE, DptWiseFieldName }, // 7
    { INST_TYPE_ELEWINIT, ElewInitFieldName }, // 8
    { INST_TYPE_ELEW, ElewFieldName }, // 9
    { INST_TYPE_THD, ThdFieldName }, // 10
    { INST_TYPE_END, EndFieldName }, // 11
    { INST_TYPE_DUMPBANK, DumpBankFieldName }, // 12
    { INST_TYPE_DUMPDDR, DumpDDRFieldName }, // 13
    { INST_TYPE_DUMPDDRSLICE, DumpDDRSliceFieldName }, // 14
};


// inst type to field data type map
const unordered_map<uint32_t, const vector<int> &> InstTable::InstFieldDataType = {
    { INST_TYPE_LOAD, LoadFieldDataType }, // 0
    { INST_TYPE_SAVE, SaveFieldDataType }, // 1
    { INST_TYPE_CONVINIT, ConvInitFieldDataType }, // 2
    { INST_TYPE_CONV, ConvFieldDataType }, // 3
    { INST_TYPE_POOLINIT, PoolInitFieldDataType }, // 4
    { INST_TYPE_POOL, PoolFieldDataType }, // 5
    { INST_TYPE_DWINIT, DWInitFieldDataType }, // 6
    { INST_TYPE_DPTWISE, DptWiseFieldDataType }, // 7
    { INST_TYPE_ELEWINIT, ElewInitFieldDataType }, // 8
    { INST_TYPE_ELEW, ElewFieldDataType }, // 9
    { INST_TYPE_THD, ThdFieldDataType }, // 10
    { INST_TYPE_END, EndFieldDataType }, // 11
    { INST_TYPE_DUMPBANK, DumpBankFieldDataType }, // 12
    { INST_TYPE_DUMPDDR, DumpDDRFieldDataType }, // 13
    { INST_TYPE_DUMPDDRSLICE, DumpDDRSliceFieldDataType }, // 14
};


// inst type to field minus map
const unordered_map<uint32_t, const vector<int> &> InstTable::InstFieldMinus = {
    { INST_TYPE_LOAD, LoadFieldMinus }, // 0
    { INST_TYPE_SAVE, SaveFieldMinus }, // 1
    { INST_TYPE_CONVINIT, ConvInitFieldMinus }, // 2
    { INST_TYPE_CONV, ConvFieldMinus }, // 3
    { INST_TYPE_POOLINIT, PoolInitFieldMinus }, // 4
    { INST_TYPE_POOL, PoolFieldMinus }, // 5
    { INST_TYPE_DWINIT, DWInitFieldMinus }, // 6
    { INST_TYPE_DPTWISE, DptWiseFieldMinus }, // 7
    { INST_TYPE_ELEWINIT, ElewInitFieldMinus }, // 8
    { INST_TYPE_ELEW, ElewFieldMinus }, // 9
    { INST_TYPE_THD, ThdFieldMinus }, // 10
    { INST_TYPE_END, EndFieldMinus }, // 11
    { INST_TYPE_DUMPBANK, DumpBankFieldMinus }, // 12
    { INST_TYPE_DUMPDDR, DumpDDRFieldMinus }, // 13
    { INST_TYPE_DUMPDDRSLICE, DumpDDRSliceFieldMinus }, // 14
};


// inst type to field position map
const unordered_map<uint32_t, const vector<map<uint32_t, uint32_t>> &> InstTable::InstFieldPos = {
    { INST_TYPE_LOAD, LoadFieldPos }, // 0
    { INST_TYPE_SAVE, SaveFieldPos }, // 1
    { INST_TYPE_CONVINIT, ConvInitFieldPos }, // 2
    { INST_TYPE_CONV, ConvFieldPos }, // 3
    { INST_TYPE_POOLINIT, PoolInitFieldPos }, // 4
    { INST_TYPE_POOL, PoolFieldPos }, // 5
    { INST_TYPE_DWINIT, DWInitFieldPos }, // 6
    { INST_TYPE_DPTWISE, DptWiseFieldPos }, // 7
    { INST_TYPE_ELEWINIT, ElewInitFieldPos }, // 8
    { INST_TYPE_ELEW, ElewFieldPos }, // 9
    { INST_TYPE_THD, ThdFieldPos }, // 10
    { INST_TYPE_END, EndFieldPos }, // 11
    { INST_TYPE_DUMPBANK, DumpBankFieldPos }, // 12
    { INST_TYPE_DUMPDDR, DumpDDRFieldPos }, // 13
    { INST_TYPE_DUMPDDRSLICE, DumpDDRSliceFieldPos }, // 14
};


// inst type to field length map
const unordered_map<uint32_t, const vector<map<uint32_t, uint32_t>> &> InstTable::InstFieldLen = {
    { INST_TYPE_LOAD, LoadFieldLen }, // 0
    { INST_TYPE_SAVE, SaveFieldLen }, // 1
    { INST_TYPE_CONVINIT, ConvInitFieldLen }, // 2
    { INST_TYPE_CONV, ConvFieldLen }, // 3
    { INST_TYPE_POOLINIT, PoolInitFieldLen }, // 4
    { INST_TYPE_POOL, PoolFieldLen }, // 5
    { INST_TYPE_DWINIT, DWInitFieldLen }, // 6
    { INST_TYPE_DPTWISE, DptWiseFieldLen }, // 7
    { INST_TYPE_ELEWINIT, ElewInitFieldLen }, // 8
    { INST_TYPE_ELEW, ElewFieldLen }, // 9
    { INST_TYPE_THD, ThdFieldLen }, // 10
    { INST_TYPE_END, EndFieldLen }, // 11
    { INST_TYPE_DUMPBANK, DumpBankFieldLen }, // 12
    { INST_TYPE_DUMPDDR, DumpDDRFieldLen }, // 13
    { INST_TYPE_DUMPDDRSLICE, DumpDDRSliceFieldLen }, // 14
};


}

namespace DPUV_2 {
// instruction version
const string InstTable::INST_VERSION = "DPUV2";


// category name vector
const vector<string> InstTable::CategoryName = {
    "LOAD",
    "SAVE",
    "CONV",
    "MISC",
};


// inst name vector
const vector<string> InstTable::InstName = {
    "LOAD", // 0
    "SAVE", // 1
    "CONVINIT", // 2
    "CONV", // 3
    "POOLINIT", // 4
    "POOL", // 5
    "ALUINIT", // 6
    "ALU", // 7
    "DWINIT", // 8
    "DPTWISE", // 9
    "ELEWINIT", // 10
    "ELEW", // 11
    "END", // 12
    "DUMPBANK", // 13
    "DUMPDDR", // 14
    "DUMPDDRSLICE", // 15
};


// lower inst name vector
const vector<string> InstTable::InstNameLower = {
    "load", // 0
    "save", // 1
    "convinit", // 2
    "conv", // 3
    "poolinit", // 4
    "pool", // 5
    "aluinit", // 6
    "alu", // 7
    "dwinit", // 8
    "dptwise", // 9
    "elewinit", // 10
    "elew", // 11
    "end", // 12
    "dumpbank", // 13
    "dumpddr", // 14
    "dumpddrslice", // 15
};


// inst category
const vector<Category> InstTable::InstCategory = {
    Category::INST_CATEGORY_LOAD, // LOAD
    Category::INST_CATEGORY_SAVE, // SAVE
    Category::INST_CATEGORY_CONV, // CONVINIT
    Category::INST_CATEGORY_CONV, // CONV
    Category::INST_CATEGORY_MISC, // POOLINIT
    Category::INST_CATEGORY_MISC, // POOL
    Category::INST_CATEGORY_MISC, // ALUINIT
    Category::INST_CATEGORY_MISC, // ALU
    Category::INST_CATEGORY_MISC, // DWINIT
    Category::INST_CATEGORY_MISC, // DPTWISE
    Category::INST_CATEGORY_MISC, // ELEWINIT
    Category::INST_CATEGORY_MISC, // ELEW
    Category::INST_CATEGORY_SAVE, // END
    Category::INST_CATEGORY_MISC, // DUMPBANK
    Category::INST_CATEGORY_MISC, // DUMPDDR
    Category::INST_CATEGORY_MISC, // DUMPDDRSLICE
};


// inst word number
const vector<uint32_t> InstTable::WordNum = {
    5, // LOAD
    5, // SAVE
    4, // CONVINIT
    5, // CONV
    2, // POOLINIT
    5, // POOL
    6, // ALUINIT
    5, // ALU
    3, // DWINIT
    5, // DPTWISE
    3, // ELEWINIT
    3, // ELEW
    1, // END
    2, // DUMPBANK
    4, // DUMPDDR
    6, // DUMPDDRSLICE
};


// inst field number
const vector<uint32_t> InstTable::FieldNum = {
    LOAD_FIELD_MAX, // 0
    SAVE_FIELD_MAX, // 1
    CONVINIT_FIELD_MAX, // 2
    CONV_FIELD_MAX, // 3
    POOLINIT_FIELD_MAX, // 4
    POOL_FIELD_MAX, // 5
    ALUINIT_FIELD_MAX, // 6
    ALU_FIELD_MAX, // 7
    DWINIT_FIELD_MAX, // 8
    DPTWISE_FIELD_MAX, // 9
    ELEWINIT_FIELD_MAX, // 10
    ELEW_FIELD_MAX, // 11
    END_FIELD_MAX, // 12
    DUMPBANK_FIELD_MAX, // 13
    DUMPDDR_FIELD_MAX, // 14
    DUMPDDRSLICE_FIELD_MAX, // 15
};


// inst opcode
const vector<uint32_t> InstTable::OPCode = {
    0x00, // LOAD
    0x04, // SAVE
    0x09, // CONVINIT
    0x08, // CONV
    0x06, // POOLINIT
    0x0C, // POOL
    0x01, // ALUINIT
    0x02, // ALU
    0x0B, // DWINIT
    0x0A, // DPTWISE
    0x0D, // ELEWINIT
    0x0E, // ELEW
    0x07, // END
    0xFF, // DUMPBANK
    0xFE, // DUMPDDR
    0xFD, // DUMPDDRSLICE
};


// Load field name
const vector<string> InstTable::LoadFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "bank_id", // 3
    "bank_addr", // 4
    "jump_read", // 5
    "jump_write", // 6
    "pad_idx", // 7
    "channel", // 8
    "length", // 9
    "pad_start", // 10
    "pad_end", // 11
    "mode_avg", // 12
    "broadcast", // 13
    "const_value", // 14
    "reg_id", // 15
    "ddr_addr", // 16
};


// Save field name
const vector<string> InstTable::SaveFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "bank_id", // 3
    "bank_addr", // 4
    "hp_id", // 5
    "argmax", // 6
    "jump_write", // 7
    "jump_read", // 8
    "channel", // 9
    "length", // 10
    "const_en", // 11
    "const_value", // 12
    "reg_id", // 13
    "ddr_addr", // 14
};


// ConvInit field name
const vector<string> InstTable::ConvInitFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "act_type", // 3
    "shift_cut", // 4
    "jump_read", // 5
    "stride_out", // 6
    "shift_bias", // 7
    "jump_read_endl", // 8
    "stride_h", // 9
    "kernel_h", // 10
    "valid_pixel_parallel", // 11
    "stride_offset_in", // 12
    "jump_write", // 13
    "stride_w", // 14
    "kernel_w", // 15
    "stride_offset_out", // 16
    "jump_write_endl", // 17
};


// Conv field name
const vector<string> InstTable::ConvFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "pad_left", // 3
    "pad_top", // 4
    "bank_addr_in", // 5
    "channel_group", // 6
    "pad_right", // 7
    "pad_bottom", // 8
    "bank_addr_out", // 9
    "length", // 10
    "bank_id_in", // 11
    "bank_addr_weights", // 12
    "bank_addr_in_1", // 13
    "bank_id_out", // 14
    "bank_addr_bias", // 15
    "channel_offset", // 16
    "bank_addr_in_3", // 17
    "bank_addr_in_2", // 18
};


// PoolInit field name
const vector<string> InstTable::PoolInitFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "shift_cut", // 3
    "kernel_w", // 4
    "kernel_h", // 5
    "jump_read", // 6
    "jump_write", // 7
    "stride_out", // 8
    "stride_offset_out", // 9
    "valid_pixel_parallel", // 10
    "stride_offset_in", // 11
    "stride_w", // 12
    "stride_h", // 13
    "pool_type", // 14
};


// Pool field name
const vector<string> InstTable::PoolFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "channel_group", // 3
    "bank_addr_in", // 4
    "jump_write_endl", // 5
    "pad_left", // 6
    "pad_top", // 7
    "pad_right", // 8
    "pad_bottom", // 9
    "bank_id_in", // 10
    "jump_read_endl", // 11
    "length", // 12
    "bank_id_out", // 13
    "downsample_kernel_w", // 14
    "bank_addr_in_1", // 15
    "bank_addr_out", // 16
    "downsample_kernel_h", // 17
    "bank_addr_in_3", // 18
    "bank_addr_in_2", // 19
};


// AluInit field name
const vector<string> InstTable::AluInitFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "exec_mode", // 3
    "kernel_h", // 4
    "kernel_w", // 5
    "stride_h", // 6
    "stride_w", // 7
    "stride_offset_in", // 8
    "stride_offset_out", // 9
    "stride_out", // 10
    "channel_group", // 11
    "channel_offset", // 12
    "length", // 13
    "shift_prelu_p", // 14
    "act_type", // 15
    "shift_bias", // 16
    "shift_cut", // 17
    "shift_prelu_n", // 18
    "multi_factor", // 19
    "jump_read", // 20
    "jump_read_endl", // 21
    "jump_read_weights", // 22
    "jump_write", // 23
    "jump_write_endl", // 24
};


// Alu field name
const vector<string> InstTable::AluFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "bank_id_out", // 3
    "bank_addr_out", // 4
    "pad_bottom", // 5
    "valid_pixel_parallel", // 6
    "bank_id_in", // 7
    "bank_addr_in", // 8
    "pad_top", // 9
    "pad_left", // 10
    "pad_right", // 11
    "bank_addr_in_1", // 12
    "kernel_d", // 13
    "bank_addr_weights", // 14
    "bank_addr_in_2", // 15
    "bank_addr_bias", // 16
    "bank_addr_in_3", // 17
};


// DWInit field name
const vector<string> InstTable::DWInitFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "jump_read", // 3
    "jump_read_endl", // 4
    "stride_offset_in", // 5
    "valid_pixel_parallel", // 6
    "stride_w", // 7
    "stride_h", // 8
    "kernel_w", // 9
    "kernel_h", // 10
    "jump_write_endl", // 11
    "jump_write", // 12
    "stride_out", // 13
    "stride_offset_out", // 14
    "shift_bias", // 15
    "shift_cut", // 16
};


// DptWise field name
const vector<string> InstTable::DptWiseFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "channel_group", // 3
    "bank_addr_in", // 4
    "pad_left", // 5
    "pad_right", // 6
    "pad_top", // 7
    "pad_bottom", // 8
    "bank_addr_out", // 9
    "act_type", // 10
    "length", // 11
    "bank_id_in", // 12
    "bank_addr_weights", // 13
    "bank_addr_in_1", // 14
    "bank_id_out", // 15
    "bank_addr_bias", // 16
    "channel_offset", // 17
    "bank_addr_in_3", // 18
    "bank_addr_in_2", // 19
};


// ElewInit field name
const vector<string> InstTable::ElewInitFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "bank_id_in", // 3
    "share_pp", // 4
    "jump_bank", // 5
    "id", // 6
    "jump_read_endl", // 7
    "bank_addr_in", // 8
    "shift_read", // 9
    "jump_read", // 10
};


// Elew field name
const vector<string> InstTable::ElewFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "bank_id_out", // 3
    "elew_type", // 4
    "act_type", // 5
    "length", // 6
    "valid_pixel_parallel", // 7
    "num", // 8
    "jump_write_endl", // 9
    "bank_addr_out", // 10
    "channel_group", // 11
    "shift_write", // 12
    "jump_write", // 13
};


// End field name
const vector<string> InstTable::EndFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
};


// DumpBank field name
const vector<string> InstTable::DumpBankFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "save_name", // 3
    "save_fmt", // 4
    "bank_start", // 5
    "bank_num", // 6
};


// DumpDDR field name
const vector<string> InstTable::DumpDDRFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "save_name", // 3
    "save_fmt", // 4
    "reg_id", // 5
    "ddr_start", // 6
    "ddr_size", // 7
};


// DumpDDRSlice field name
const vector<string> InstTable::DumpDDRSliceFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "save_name", // 3
    "save_fmt", // 4
    "reg_id", // 5
    "ddr_start", // 6
    "height", // 7
    "height_stride", // 8
    "width", // 9
    "width_stride", // 10
    "channel", // 11
    "channel_stride", // 12
};


// Load field data type
const vector<int> InstTable::LoadFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_UINT, // bank_id(3)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr(4)
    INST_FIELD_DATA_TYPE_UINT, // jump_read(5)
    INST_FIELD_DATA_TYPE_UINT, // jump_write(6)
    INST_FIELD_DATA_TYPE_UINT, // pad_idx(7)
    INST_FIELD_DATA_TYPE_UINT, // channel(8)
    INST_FIELD_DATA_TYPE_UINT, // length(9)
    INST_FIELD_DATA_TYPE_UINT, // pad_start(10)
    INST_FIELD_DATA_TYPE_UINT, // pad_end(11)
    INST_FIELD_DATA_TYPE_UINT, // mode_avg(12)
    INST_FIELD_DATA_TYPE_UINT, // broadcast(13)
    INST_FIELD_DATA_TYPE_UINT, // const_value(14)
    INST_FIELD_DATA_TYPE_UINT, // reg_id(15)
    INST_FIELD_DATA_TYPE_UINT, // ddr_addr(16)
};


// Save field data type
const vector<int> InstTable::SaveFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_UINT, // bank_id(3)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr(4)
    INST_FIELD_DATA_TYPE_UINT, // hp_id(5)
    INST_FIELD_DATA_TYPE_UINT, // argmax(6)
    INST_FIELD_DATA_TYPE_UINT, // jump_write(7)
    INST_FIELD_DATA_TYPE_UINT, // jump_read(8)
    INST_FIELD_DATA_TYPE_UINT, // channel(9)
    INST_FIELD_DATA_TYPE_UINT, // length(10)
    INST_FIELD_DATA_TYPE_UINT, // const_en(11)
    INST_FIELD_DATA_TYPE_UINT, // const_value(12)
    INST_FIELD_DATA_TYPE_UINT, // reg_id(13)
    INST_FIELD_DATA_TYPE_UINT, // ddr_addr(14)
};


// ConvInit field data type
const vector<int> InstTable::ConvInitFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_UINT, // act_type(3)
    INST_FIELD_DATA_TYPE_UINT, // shift_cut(4)
    INST_FIELD_DATA_TYPE_UINT, // jump_read(5)
    INST_FIELD_DATA_TYPE_UINT, // stride_out(6)
    INST_FIELD_DATA_TYPE_UINT, // shift_bias(7)
    INST_FIELD_DATA_TYPE_UINT, // jump_read_endl(8)
    INST_FIELD_DATA_TYPE_UINT, // stride_h(9)
    INST_FIELD_DATA_TYPE_UINT, // kernel_h(10)
    INST_FIELD_DATA_TYPE_UINT, // valid_pixel_parallel(11)
    INST_FIELD_DATA_TYPE_UINT, // stride_offset_in(12)
    INST_FIELD_DATA_TYPE_UINT, // jump_write(13)
    INST_FIELD_DATA_TYPE_UINT, // stride_w(14)
    INST_FIELD_DATA_TYPE_UINT, // kernel_w(15)
    INST_FIELD_DATA_TYPE_UINT, // stride_offset_out(16)
    INST_FIELD_DATA_TYPE_UINT, // jump_write_endl(17)
};


// Conv field data type
const vector<int> InstTable::ConvFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_UINT, // pad_left(3)
    INST_FIELD_DATA_TYPE_UINT, // pad_top(4)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in(5)
    INST_FIELD_DATA_TYPE_UINT, // channel_group(6)
    INST_FIELD_DATA_TYPE_UINT, // pad_right(7)
    INST_FIELD_DATA_TYPE_UINT, // pad_bottom(8)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_out(9)
    INST_FIELD_DATA_TYPE_UINT, // length(10)
    INST_FIELD_DATA_TYPE_UINT, // bank_id_in(11)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_weights(12)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in_1(13)
    INST_FIELD_DATA_TYPE_UINT, // bank_id_out(14)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_bias(15)
    INST_FIELD_DATA_TYPE_UINT, // channel_offset(16)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in_3(17)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in_2(18)
};


// PoolInit field data type
const vector<int> InstTable::PoolInitFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_UINT, // shift_cut(3)
    INST_FIELD_DATA_TYPE_UINT, // kernel_w(4)
    INST_FIELD_DATA_TYPE_UINT, // kernel_h(5)
    INST_FIELD_DATA_TYPE_UINT, // jump_read(6)
    INST_FIELD_DATA_TYPE_UINT, // jump_write(7)
    INST_FIELD_DATA_TYPE_UINT, // stride_out(8)
    INST_FIELD_DATA_TYPE_UINT, // stride_offset_out(9)
    INST_FIELD_DATA_TYPE_UINT, // valid_pixel_parallel(10)
    INST_FIELD_DATA_TYPE_UINT, // stride_offset_in(11)
    INST_FIELD_DATA_TYPE_UINT, // stride_w(12)
    INST_FIELD_DATA_TYPE_UINT, // stride_h(13)
    INST_FIELD_DATA_TYPE_UINT, // pool_type(14)
};


// Pool field data type
const vector<int> InstTable::PoolFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_UINT, // channel_group(3)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in(4)
    INST_FIELD_DATA_TYPE_UINT, // jump_write_endl(5)
    INST_FIELD_DATA_TYPE_UINT, // pad_left(6)
    INST_FIELD_DATA_TYPE_UINT, // pad_top(7)
    INST_FIELD_DATA_TYPE_UINT, // pad_right(8)
    INST_FIELD_DATA_TYPE_UINT, // pad_bottom(9)
    INST_FIELD_DATA_TYPE_UINT, // bank_id_in(10)
    INST_FIELD_DATA_TYPE_UINT, // jump_read_endl(11)
    INST_FIELD_DATA_TYPE_UINT, // length(12)
    INST_FIELD_DATA_TYPE_UINT, // bank_id_out(13)
    INST_FIELD_DATA_TYPE_UINT, // downsample_kernel_w(14)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in_1(15)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_out(16)
    INST_FIELD_DATA_TYPE_UINT, // downsample_kernel_h(17)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in_3(18)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in_2(19)
};


// AluInit field data type
const vector<int> InstTable::AluInitFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_UINT, // exec_mode(3)
    INST_FIELD_DATA_TYPE_UINT, // kernel_h(4)
    INST_FIELD_DATA_TYPE_UINT, // kernel_w(5)
    INST_FIELD_DATA_TYPE_UINT, // stride_h(6)
    INST_FIELD_DATA_TYPE_UINT, // stride_w(7)
    INST_FIELD_DATA_TYPE_UINT, // stride_offset_in(8)
    INST_FIELD_DATA_TYPE_UINT, // stride_offset_out(9)
    INST_FIELD_DATA_TYPE_UINT, // stride_out(10)
    INST_FIELD_DATA_TYPE_UINT, // channel_group(11)
    INST_FIELD_DATA_TYPE_UINT, // channel_offset(12)
    INST_FIELD_DATA_TYPE_UINT, // length(13)
    INST_FIELD_DATA_TYPE_UINT, // shift_prelu_p(14)
    INST_FIELD_DATA_TYPE_UINT, // act_type(15)
    INST_FIELD_DATA_TYPE_UINT, // shift_bias(16)
    INST_FIELD_DATA_TYPE_UINT, // shift_cut(17)
    INST_FIELD_DATA_TYPE_UINT, // shift_prelu_n(18)
    INST_FIELD_DATA_TYPE_UINT, // multi_factor(19)
    INST_FIELD_DATA_TYPE_UINT, // jump_read(20)
    INST_FIELD_DATA_TYPE_UINT, // jump_read_endl(21)
    INST_FIELD_DATA_TYPE_UINT, // jump_read_weights(22)
    INST_FIELD_DATA_TYPE_UINT, // jump_write(23)
    INST_FIELD_DATA_TYPE_UINT, // jump_write_endl(24)
};


// Alu field data type
const vector<int> InstTable::AluFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_UINT, // bank_id_out(3)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_out(4)
    INST_FIELD_DATA_TYPE_UINT, // pad_bottom(5)
    INST_FIELD_DATA_TYPE_UINT, // valid_pixel_parallel(6)
    INST_FIELD_DATA_TYPE_UINT, // bank_id_in(7)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in(8)
    INST_FIELD_DATA_TYPE_UINT, // pad_top(9)
    INST_FIELD_DATA_TYPE_UINT, // pad_left(10)
    INST_FIELD_DATA_TYPE_UINT, // pad_right(11)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in_1(12)
    INST_FIELD_DATA_TYPE_UINT, // kernel_d(13)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_weights(14)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in_2(15)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_bias(16)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in_3(17)
};


// DWInit field data type
const vector<int> InstTable::DWInitFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_UINT, // jump_read(3)
    INST_FIELD_DATA_TYPE_UINT, // jump_read_endl(4)
    INST_FIELD_DATA_TYPE_UINT, // stride_offset_in(5)
    INST_FIELD_DATA_TYPE_UINT, // valid_pixel_parallel(6)
    INST_FIELD_DATA_TYPE_UINT, // stride_w(7)
    INST_FIELD_DATA_TYPE_UINT, // stride_h(8)
    INST_FIELD_DATA_TYPE_UINT, // kernel_w(9)
    INST_FIELD_DATA_TYPE_UINT, // kernel_h(10)
    INST_FIELD_DATA_TYPE_UINT, // jump_write_endl(11)
    INST_FIELD_DATA_TYPE_UINT, // jump_write(12)
    INST_FIELD_DATA_TYPE_UINT, // stride_out(13)
    INST_FIELD_DATA_TYPE_UINT, // stride_offset_out(14)
    INST_FIELD_DATA_TYPE_UINT, // shift_bias(15)
    INST_FIELD_DATA_TYPE_UINT, // shift_cut(16)
};


// DptWise field data type
const vector<int> InstTable::DptWiseFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_UINT, // channel_group(3)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in(4)
    INST_FIELD_DATA_TYPE_UINT, // pad_left(5)
    INST_FIELD_DATA_TYPE_UINT, // pad_right(6)
    INST_FIELD_DATA_TYPE_UINT, // pad_top(7)
    INST_FIELD_DATA_TYPE_UINT, // pad_bottom(8)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_out(9)
    INST_FIELD_DATA_TYPE_UINT, // act_type(10)
    INST_FIELD_DATA_TYPE_UINT, // length(11)
    INST_FIELD_DATA_TYPE_UINT, // bank_id_in(12)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_weights(13)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in_1(14)
    INST_FIELD_DATA_TYPE_UINT, // bank_id_out(15)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_bias(16)
    INST_FIELD_DATA_TYPE_UINT, // channel_offset(17)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in_3(18)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in_2(19)
};


// ElewInit field data type
const vector<int> InstTable::ElewInitFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_UINT, // bank_id_in(3)
    INST_FIELD_DATA_TYPE_UINT, // share_pp(4)
    INST_FIELD_DATA_TYPE_UINT, // jump_bank(5)
    INST_FIELD_DATA_TYPE_UINT, // id(6)
    INST_FIELD_DATA_TYPE_UINT, // jump_read_endl(7)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in(8)
    INST_FIELD_DATA_TYPE_UINT, // shift_read(9)
    INST_FIELD_DATA_TYPE_UINT, // jump_read(10)
};


// Elew field data type
const vector<int> InstTable::ElewFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_UINT, // bank_id_out(3)
    INST_FIELD_DATA_TYPE_UINT, // elew_type(4)
    INST_FIELD_DATA_TYPE_UINT, // act_type(5)
    INST_FIELD_DATA_TYPE_UINT, // length(6)
    INST_FIELD_DATA_TYPE_UINT, // valid_pixel_parallel(7)
    INST_FIELD_DATA_TYPE_UINT, // num(8)
    INST_FIELD_DATA_TYPE_UINT, // jump_write_endl(9)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_out(10)
    INST_FIELD_DATA_TYPE_UINT, // channel_group(11)
    INST_FIELD_DATA_TYPE_UINT, // shift_write(12)
    INST_FIELD_DATA_TYPE_UINT, // jump_write(13)
};


// End field data type
const vector<int> InstTable::EndFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
};


// DumpBank field data type
const vector<int> InstTable::DumpBankFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_STRING, // save_name(3)
    INST_FIELD_DATA_TYPE_UINT, // save_fmt(4)
    INST_FIELD_DATA_TYPE_UINT, // bank_start(5)
    INST_FIELD_DATA_TYPE_UINT, // bank_num(6)
};


// DumpDDR field data type
const vector<int> InstTable::DumpDDRFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_STRING, // save_name(3)
    INST_FIELD_DATA_TYPE_UINT, // save_fmt(4)
    INST_FIELD_DATA_TYPE_UINT, // reg_id(5)
    INST_FIELD_DATA_TYPE_UINT, // ddr_start(6)
    INST_FIELD_DATA_TYPE_UINT, // ddr_size(7)
};


// DumpDDRSlice field data type
const vector<int> InstTable::DumpDDRSliceFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_STRING, // save_name(3)
    INST_FIELD_DATA_TYPE_UINT, // save_fmt(4)
    INST_FIELD_DATA_TYPE_UINT, // reg_id(5)
    INST_FIELD_DATA_TYPE_UINT, // ddr_start(6)
    INST_FIELD_DATA_TYPE_UINT, // height(7)
    INST_FIELD_DATA_TYPE_UINT, // height_stride(8)
    INST_FIELD_DATA_TYPE_UINT, // width(9)
    INST_FIELD_DATA_TYPE_UINT, // width_stride(10)
    INST_FIELD_DATA_TYPE_UINT, // channel(11)
    INST_FIELD_DATA_TYPE_UINT, // channel_stride(12)
};


// Load field minus value
const vector<int> InstTable::LoadFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // bank_id(3)
    0, // bank_addr(4)
    1, // jump_read(5)
    1, // jump_write(6)
    1, // pad_idx(7)
    1, // channel(8)
    1, // length(9)
    0, // pad_start(10)
    0, // pad_end(11)
    0, // mode_avg(12)
    0, // broadcast(13)
    0, // const_value(14)
    0, // reg_id(15)
    0, // ddr_addr(16)
};


// Save field minus value
const vector<int> InstTable::SaveFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // bank_id(3)
    0, // bank_addr(4)
    0, // hp_id(5)
    0, // argmax(6)
    1, // jump_write(7)
    1, // jump_read(8)
    1, // channel(9)
    1, // length(10)
    0, // const_en(11)
    0, // const_value(12)
    0, // reg_id(13)
    0, // ddr_addr(14)
};


// ConvInit field minus value
const vector<int> InstTable::ConvInitFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // act_type(3)
    0, // shift_cut(4)
    1, // jump_read(5)
    1, // stride_out(6)
    0, // shift_bias(7)
    1, // jump_read_endl(8)
    1, // stride_h(9)
    1, // kernel_h(10)
    1, // valid_pixel_parallel(11)
    0, // stride_offset_in(12)
    1, // jump_write(13)
    1, // stride_w(14)
    1, // kernel_w(15)
    0, // stride_offset_out(16)
    1, // jump_write_endl(17)
};


// Conv field minus value
const vector<int> InstTable::ConvFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // pad_left(3)
    0, // pad_top(4)
    0, // bank_addr_in(5)
    1, // channel_group(6)
    0, // pad_right(7)
    0, // pad_bottom(8)
    0, // bank_addr_out(9)
    1, // length(10)
    0, // bank_id_in(11)
    0, // bank_addr_weights(12)
    0, // bank_addr_in_1(13)
    0, // bank_id_out(14)
    0, // bank_addr_bias(15)
    0, // channel_offset(16)
    0, // bank_addr_in_3(17)
    0, // bank_addr_in_2(18)
};


// PoolInit field minus value
const vector<int> InstTable::PoolInitFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // shift_cut(3)
    1, // kernel_w(4)
    1, // kernel_h(5)
    1, // jump_read(6)
    1, // jump_write(7)
    1, // stride_out(8)
    0, // stride_offset_out(9)
    1, // valid_pixel_parallel(10)
    0, // stride_offset_in(11)
    1, // stride_w(12)
    1, // stride_h(13)
    0, // pool_type(14)
};


// Pool field minus value
const vector<int> InstTable::PoolFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    1, // channel_group(3)
    0, // bank_addr_in(4)
    1, // jump_write_endl(5)
    0, // pad_left(6)
    0, // pad_top(7)
    0, // pad_right(8)
    0, // pad_bottom(9)
    0, // bank_id_in(10)
    1, // jump_read_endl(11)
    1, // length(12)
    0, // bank_id_out(13)
    1, // downsample_kernel_w(14)
    0, // bank_addr_in_1(15)
    0, // bank_addr_out(16)
    1, // downsample_kernel_h(17)
    0, // bank_addr_in_3(18)
    0, // bank_addr_in_2(19)
};


// AluInit field minus value
const vector<int> InstTable::AluInitFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // exec_mode(3)
    1, // kernel_h(4)
    1, // kernel_w(5)
    1, // stride_h(6)
    1, // stride_w(7)
    0, // stride_offset_in(8)
    0, // stride_offset_out(9)
    1, // stride_out(10)
    1, // channel_group(11)
    0, // channel_offset(12)
    1, // length(13)
    0, // shift_prelu_p(14)
    0, // act_type(15)
    0, // shift_bias(16)
    0, // shift_cut(17)
    0, // shift_prelu_n(18)
    0, // multi_factor(19)
    1, // jump_read(20)
    1, // jump_read_endl(21)
    0, // jump_read_weights(22)
    1, // jump_write(23)
    1, // jump_write_endl(24)
};


// Alu field minus value
const vector<int> InstTable::AluFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // bank_id_out(3)
    0, // bank_addr_out(4)
    0, // pad_bottom(5)
    1, // valid_pixel_parallel(6)
    0, // bank_id_in(7)
    0, // bank_addr_in(8)
    0, // pad_top(9)
    0, // pad_left(10)
    0, // pad_right(11)
    0, // bank_addr_in_1(12)
    1, // kernel_d(13)
    0, // bank_addr_weights(14)
    0, // bank_addr_in_2(15)
    0, // bank_addr_bias(16)
    0, // bank_addr_in_3(17)
};


// DWInit field minus value
const vector<int> InstTable::DWInitFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    1, // jump_read(3)
    1, // jump_read_endl(4)
    0, // stride_offset_in(5)
    1, // valid_pixel_parallel(6)
    1, // stride_w(7)
    1, // stride_h(8)
    1, // kernel_w(9)
    1, // kernel_h(10)
    1, // jump_write_endl(11)
    1, // jump_write(12)
    1, // stride_out(13)
    0, // stride_offset_out(14)
    0, // shift_bias(15)
    0, // shift_cut(16)
};


// DptWise field minus value
const vector<int> InstTable::DptWiseFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    1, // channel_group(3)
    0, // bank_addr_in(4)
    0, // pad_left(5)
    0, // pad_right(6)
    0, // pad_top(7)
    0, // pad_bottom(8)
    0, // bank_addr_out(9)
    0, // act_type(10)
    1, // length(11)
    0, // bank_id_in(12)
    0, // bank_addr_weights(13)
    0, // bank_addr_in_1(14)
    0, // bank_id_out(15)
    0, // bank_addr_bias(16)
    0, // channel_offset(17)
    0, // bank_addr_in_3(18)
    0, // bank_addr_in_2(19)
};


// ElewInit field minus value
const vector<int> InstTable::ElewInitFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // bank_id_in(3)
    0, // share_pp(4)
    1, // jump_bank(5)
    0, // id(6)
    1, // jump_read_endl(7)
    0, // bank_addr_in(8)
    0, // shift_read(9)
    0, // jump_read(10)
};


// Elew field minus value
const vector<int> InstTable::ElewFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // bank_id_out(3)
    0, // elew_type(4)
    0, // act_type(5)
    1, // length(6)
    1, // valid_pixel_parallel(7)
    1, // num(8)
    1, // jump_write_endl(9)
    0, // bank_addr_out(10)
    1, // channel_group(11)
    0, // shift_write(12)
    1, // jump_write(13)
};


// End field minus value
const vector<int> InstTable::EndFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
};


// DumpBank field minus value
const vector<int> InstTable::DumpBankFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // save_name(3)
    0, // save_fmt(4)
    0, // bank_start(5)
    0, // bank_num(6)
};


// DumpDDR field minus value
const vector<int> InstTable::DumpDDRFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // save_name(3)
    0, // save_fmt(4)
    0, // reg_id(5)
    0, // ddr_start(6)
    0, // ddr_size(7)
};


// DumpDDRSlice field minus value
const vector<int> InstTable::DumpDDRSliceFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // save_name(3)
    0, // save_fmt(4)
    0, // reg_id(5)
    0, // ddr_start(6)
    0, // height(7)
    0, // height_stride(8)
    0, // width(9)
    0, // width_stride(10)
    0, // channel(11)
    0, // channel_stride(12)
};


// Load field position
const vector<map<uint32_t, uint32_t>> InstTable::LoadFieldPos = {
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_OPCODE, 28 }, // 0
        { LOAD_FIELD_DPDON, 24 }, // 1
        { LOAD_FIELD_DPDBY, 20 }, // 2
        { LOAD_FIELD_BANK_ID, 14 }, // 3
        { LOAD_FIELD_BANK_ADDR, 0 }, // 4
    },
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_JUMP_READ, 10 }, // 5
        { LOAD_FIELD_JUMP_WRITE, 0 }, // 6
    },
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_PAD_IDX, 27 }, // 7
        { LOAD_FIELD_CHANNEL, 10 }, // 8
        { LOAD_FIELD_LENGTH, 0 }, // 9
    },
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_PAD_START, 27 }, // 10
        { LOAD_FIELD_PAD_END, 22 }, // 11
        { LOAD_FIELD_MODE_AVG, 20 }, // 12
        { LOAD_FIELD_BROADCAST, 16 }, // 13
        { LOAD_FIELD_CONST_VALUE, 8 }, // 14
        { LOAD_FIELD_REG_ID, 0 }, // 15
    },
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_DDR_ADDR, 0 }, // 16
    },
};


// Save field position
const vector<map<uint32_t, uint32_t>> InstTable::SaveFieldPos = {
    map<uint32_t, uint32_t> {
        { SAVE_FIELD_OPCODE, 28 }, // 0
        { SAVE_FIELD_DPDON, 24 }, // 1
        { SAVE_FIELD_DPDBY, 20 }, // 2
        { SAVE_FIELD_BANK_ID, 14 }, // 3
        { SAVE_FIELD_BANK_ADDR, 0 }, // 4
    },
    map<uint32_t, uint32_t> {
        { SAVE_FIELD_HP_ID, 29 }, // 5
        { SAVE_FIELD_ARGMAX, 27 }, // 6
        { SAVE_FIELD_JUMP_WRITE, 10 }, // 7
        { SAVE_FIELD_JUMP_READ, 0 }, // 8
    },
    map<uint32_t, uint32_t> {
        { SAVE_FIELD_CHANNEL, 10 }, // 9
        { SAVE_FIELD_LENGTH, 0 }, // 10
    },
    map<uint32_t, uint32_t> {
        { SAVE_FIELD_CONST_EN, 16 }, // 11
        { SAVE_FIELD_CONST_VALUE, 8 }, // 12
        { SAVE_FIELD_REG_ID, 0 }, // 13
    },
    map<uint32_t, uint32_t> {
        { SAVE_FIELD_DDR_ADDR, 0 }, // 14
    },
};


// ConvInit field position
const vector<map<uint32_t, uint32_t>> InstTable::ConvInitFieldPos = {
    map<uint32_t, uint32_t> {
        { CONVINIT_FIELD_OPCODE, 28 }, // 0
        { CONVINIT_FIELD_DPDON, 24 }, // 1
        { CONVINIT_FIELD_DPDBY, 20 }, // 2
        { CONVINIT_FIELD_ACT_TYPE, 16 }, // 3
        { CONVINIT_FIELD_SHIFT_CUT, 10 }, // 4
        { CONVINIT_FIELD_JUMP_READ, 0 }, // 5
    },
    map<uint32_t, uint32_t> {
        { CONVINIT_FIELD_STRIDE_OUT, 28 }, // 6
        { CONVINIT_FIELD_SHIFT_BIAS, 10 }, // 7
        { CONVINIT_FIELD_JUMP_READ_ENDL, 0 }, // 8
    },
    map<uint32_t, uint32_t> {
        { CONVINIT_FIELD_STRIDE_H, 28 }, // 9
        { CONVINIT_FIELD_KERNEL_H, 24 }, // 10
        { CONVINIT_FIELD_VALID_PIXEL_PARALLEL, 13 }, // 11
        { CONVINIT_FIELD_STRIDE_OFFSET_IN, 10 }, // 12
        { CONVINIT_FIELD_JUMP_WRITE, 0 }, // 13
    },
    map<uint32_t, uint32_t> {
        { CONVINIT_FIELD_STRIDE_W, 28 }, // 14
        { CONVINIT_FIELD_KERNEL_W, 24 }, // 15
        { CONVINIT_FIELD_STRIDE_OFFSET_OUT, 10 }, // 16
        { CONVINIT_FIELD_JUMP_WRITE_ENDL, 0 }, // 17
    },
};


// Conv field position
const vector<map<uint32_t, uint32_t>> InstTable::ConvFieldPos = {
    map<uint32_t, uint32_t> {
        { CONV_FIELD_OPCODE, 28 }, // 0
        { CONV_FIELD_DPDON, 24 }, // 1
        { CONV_FIELD_DPDBY, 20 }, // 2
        { CONV_FIELD_PAD_LEFT, 16 }, // 3
        { CONV_FIELD_PAD_TOP, 12 }, // 4
        { CONV_FIELD_BANK_ADDR_IN, 0 }, // 5
    },
    map<uint32_t, uint32_t> {
        { CONV_FIELD_CHANNEL_GROUP, 24 }, // 6
        { CONV_FIELD_PAD_RIGHT, 16 }, // 7
        { CONV_FIELD_PAD_BOTTOM, 12 }, // 8
        { CONV_FIELD_BANK_ADDR_OUT, 0 }, // 9
    },
    map<uint32_t, uint32_t> {
        { CONV_FIELD_LENGTH, 22 }, // 10
        { CONV_FIELD_BANK_ID_IN, 12 }, // 11
        { CONV_FIELD_BANK_ADDR_WEIGHTS, 0 }, // 12
    },
    map<uint32_t, uint32_t> {
        { CONV_FIELD_BANK_ADDR_IN_1, 18 }, // 13
        { CONV_FIELD_BANK_ID_OUT, 12 }, // 14
        { CONV_FIELD_BANK_ADDR_BIAS, 0 }, // 15
    },
    map<uint32_t, uint32_t> {
        { CONV_FIELD_CHANNEL_OFFSET, 24 }, // 16
        { CONV_FIELD_BANK_ADDR_IN_3, 12 }, // 17
        { CONV_FIELD_BANK_ADDR_IN_2, 0 }, // 18
    },
};


// PoolInit field position
const vector<map<uint32_t, uint32_t>> InstTable::PoolInitFieldPos = {
    map<uint32_t, uint32_t> {
        { POOLINIT_FIELD_OPCODE, 28 }, // 0
        { POOLINIT_FIELD_DPDON, 24 }, // 1
        { POOLINIT_FIELD_DPDBY, 20 }, // 2
        { POOLINIT_FIELD_SHIFT_CUT, 16 }, // 3
        { POOLINIT_FIELD_KERNEL_W, 13 }, // 4
        { POOLINIT_FIELD_KERNEL_H, 10 }, // 5
        { POOLINIT_FIELD_JUMP_READ, 0 }, // 6
    },
    map<uint32_t, uint32_t> {
        { POOLINIT_FIELD_JUMP_WRITE, 21 }, // 7
        { POOLINIT_FIELD_STRIDE_OUT, 17 }, // 8
        { POOLINIT_FIELD_STRIDE_OFFSET_OUT, 14 }, // 9
        { POOLINIT_FIELD_VALID_PIXEL_PARALLEL, 11 }, // 10
        { POOLINIT_FIELD_STRIDE_OFFSET_IN, 8 }, // 11
        { POOLINIT_FIELD_STRIDE_W, 5 }, // 12
        { POOLINIT_FIELD_STRIDE_H, 2 }, // 13
        { POOLINIT_FIELD_POOL_TYPE, 0 }, // 14
    },
};


// Pool field position
const vector<map<uint32_t, uint32_t>> InstTable::PoolFieldPos = {
    map<uint32_t, uint32_t> {
        { POOL_FIELD_OPCODE, 28 }, // 0
        { POOL_FIELD_DPDON, 24 }, // 1
        { POOL_FIELD_DPDBY, 20 }, // 2
        { POOL_FIELD_CHANNEL_GROUP, 12 }, // 3
        { POOL_FIELD_BANK_ADDR_IN, 0 }, // 4
    },
    map<uint32_t, uint32_t> {
        { POOL_FIELD_JUMP_WRITE_ENDL, 18 }, // 5
        { POOL_FIELD_PAD_LEFT, 15 }, // 6
        { POOL_FIELD_PAD_TOP, 12 }, // 7
        { POOL_FIELD_PAD_RIGHT, 9 }, // 8
        { POOL_FIELD_PAD_BOTTOM, 6 }, // 9
        { POOL_FIELD_BANK_ID_IN, 0 }, // 10
    },
    map<uint32_t, uint32_t> {
        { POOL_FIELD_JUMP_READ_ENDL, 16 }, // 11
        { POOL_FIELD_LENGTH, 6 }, // 12
        { POOL_FIELD_BANK_ID_OUT, 0 }, // 13
    },
    map<uint32_t, uint32_t> {
        { POOL_FIELD_DOWNSAMPLE_KERNEL_W, 24 }, // 14
        { POOL_FIELD_BANK_ADDR_IN_1, 12 }, // 15
        { POOL_FIELD_BANK_ADDR_OUT, 0 }, // 16
    },
    map<uint32_t, uint32_t> {
        { POOL_FIELD_DOWNSAMPLE_KERNEL_H, 24 }, // 17
        { POOL_FIELD_BANK_ADDR_IN_3, 12 }, // 18
        { POOL_FIELD_BANK_ADDR_IN_2, 0 }, // 19
    },
};


// AluInit field position
const vector<map<uint32_t, uint32_t>> InstTable::AluInitFieldPos = {
    map<uint32_t, uint32_t> {
        { ALUINIT_FIELD_OPCODE, 28 }, // 0
        { ALUINIT_FIELD_DPDON, 24 }, // 1
        { ALUINIT_FIELD_DPDBY, 20 }, // 2
        { ALUINIT_FIELD_EXEC_MODE, 16 }, // 3
        { ALUINIT_FIELD_KERNEL_H, 8 }, // 4
        { ALUINIT_FIELD_KERNEL_W, 0 }, // 5
    },
    map<uint32_t, uint32_t> {
        { ALUINIT_FIELD_STRIDE_H, 24 }, // 6
        { ALUINIT_FIELD_STRIDE_W, 16 }, // 7
        { ALUINIT_FIELD_STRIDE_OFFSET_IN, 8 }, // 8
        { ALUINIT_FIELD_STRIDE_OFFSET_OUT, 0 }, // 9
    },
    map<uint32_t, uint32_t> {
        { ALUINIT_FIELD_STRIDE_OUT, 24 }, // 10
        { ALUINIT_FIELD_CHANNEL_GROUP, 16 }, // 11
        { ALUINIT_FIELD_CHANNEL_OFFSET, 11 }, // 12
        { ALUINIT_FIELD_LENGTH, 0 }, // 13
    },
    map<uint32_t, uint32_t> {
        { ALUINIT_FIELD_SHIFT_PRELU_P, 22 }, // 14
        { ALUINIT_FIELD_ACT_TYPE, 18 }, // 15
        { ALUINIT_FIELD_SHIFT_BIAS, 12 }, // 16
        { ALUINIT_FIELD_SHIFT_CUT, 6 }, // 17
        { ALUINIT_FIELD_SHIFT_PRELU_N, 0 }, // 18
    },
    map<uint32_t, uint32_t> {
        { ALUINIT_FIELD_MULTI_FACTOR, 23 }, // 19
        { ALUINIT_FIELD_JUMP_READ, 13 }, // 20
        { ALUINIT_FIELD_JUMP_READ_ENDL, 0 }, // 21
    },
    map<uint32_t, uint32_t> {
        { ALUINIT_FIELD_JUMP_READ_WEIGHTS, 23 }, // 22
        { ALUINIT_FIELD_JUMP_WRITE, 13 }, // 23
        { ALUINIT_FIELD_JUMP_WRITE_ENDL, 0 }, // 24
    },
};


// Alu field position
const vector<map<uint32_t, uint32_t>> InstTable::AluFieldPos = {
    map<uint32_t, uint32_t> {
        { ALU_FIELD_OPCODE, 28 }, // 0
        { ALU_FIELD_DPDON, 24 }, // 1
        { ALU_FIELD_DPDBY, 20 }, // 2
        { ALU_FIELD_BANK_ID_OUT, 14 }, // 3
        { ALU_FIELD_BANK_ADDR_OUT, 0 }, // 4
    },
    map<uint32_t, uint32_t> {
        { ALU_FIELD_PAD_BOTTOM, 23 }, // 5
        { ALU_FIELD_VALID_PIXEL_PARALLEL, 20 }, // 6
        { ALU_FIELD_BANK_ID_IN, 14 }, // 7
        { ALU_FIELD_BANK_ADDR_IN, 0 }, // 8
    },
    map<uint32_t, uint32_t> {
        { ALU_FIELD_PAD_TOP, 26 }, // 9
        { ALU_FIELD_PAD_LEFT, 22 }, // 10
        { ALU_FIELD_PAD_RIGHT, 14 }, // 11
        { ALU_FIELD_BANK_ADDR_IN_1, 0 }, // 12
    },
    map<uint32_t, uint32_t> {
        { ALU_FIELD_KERNEL_D, 28 }, // 13
        { ALU_FIELD_BANK_ADDR_WEIGHTS, 14 }, // 14
        { ALU_FIELD_BANK_ADDR_IN_2, 0 }, // 15
    },
    map<uint32_t, uint32_t> {
        { ALU_FIELD_BANK_ADDR_BIAS, 14 }, // 16
        { ALU_FIELD_BANK_ADDR_IN_3, 0 }, // 17
    },
};


// DWInit field position
const vector<map<uint32_t, uint32_t>> InstTable::DWInitFieldPos = {
    map<uint32_t, uint32_t> {
        { DWINIT_FIELD_OPCODE, 28 }, // 0
        { DWINIT_FIELD_DPDON, 24 }, // 1
        { DWINIT_FIELD_DPDBY, 20 }, // 2
        { DWINIT_FIELD_JUMP_READ, 10 }, // 3
        { DWINIT_FIELD_JUMP_READ_ENDL, 0 }, // 4
    },
    map<uint32_t, uint32_t> {
        { DWINIT_FIELD_STRIDE_OFFSET_IN, 29 }, // 5
        { DWINIT_FIELD_VALID_PIXEL_PARALLEL, 26 }, // 6
        { DWINIT_FIELD_STRIDE_W, 22 }, // 7
        { DWINIT_FIELD_STRIDE_H, 18 }, // 8
        { DWINIT_FIELD_KERNEL_W, 14 }, // 9
        { DWINIT_FIELD_KERNEL_H, 10 }, // 10
        { DWINIT_FIELD_JUMP_WRITE_ENDL, 0 }, // 11
    },
    map<uint32_t, uint32_t> {
        { DWINIT_FIELD_JUMP_WRITE, 19 }, // 12
        { DWINIT_FIELD_STRIDE_OUT, 15 }, // 13
        { DWINIT_FIELD_STRIDE_OFFSET_OUT, 12 }, // 14
        { DWINIT_FIELD_SHIFT_BIAS, 6 }, // 15
        { DWINIT_FIELD_SHIFT_CUT, 0 }, // 16
    },
};


// DptWise field position
const vector<map<uint32_t, uint32_t>> InstTable::DptWiseFieldPos = {
    map<uint32_t, uint32_t> {
        { DPTWISE_FIELD_OPCODE, 28 }, // 0
        { DPTWISE_FIELD_DPDON, 24 }, // 1
        { DPTWISE_FIELD_DPDBY, 20 }, // 2
        { DPTWISE_FIELD_CHANNEL_GROUP, 12 }, // 3
        { DPTWISE_FIELD_BANK_ADDR_IN, 0 }, // 4
    },
    map<uint32_t, uint32_t> {
        { DPTWISE_FIELD_PAD_LEFT, 28 }, // 5
        { DPTWISE_FIELD_PAD_RIGHT, 24 }, // 6
        { DPTWISE_FIELD_PAD_TOP, 20 }, // 7
        { DPTWISE_FIELD_PAD_BOTTOM, 16 }, // 8
        { DPTWISE_FIELD_BANK_ADDR_OUT, 0 }, // 9
    },
    map<uint32_t, uint32_t> {
        { DPTWISE_FIELD_ACT_TYPE, 28 }, // 10
        { DPTWISE_FIELD_LENGTH, 18 }, // 11
        { DPTWISE_FIELD_BANK_ID_IN, 12 }, // 12
        { DPTWISE_FIELD_BANK_ADDR_WEIGHTS, 0 }, // 13
    },
    map<uint32_t, uint32_t> {
        { DPTWISE_FIELD_BANK_ADDR_IN_1, 18 }, // 14
        { DPTWISE_FIELD_BANK_ID_OUT, 12 }, // 15
        { DPTWISE_FIELD_BANK_ADDR_BIAS, 0 }, // 16
    },
    map<uint32_t, uint32_t> {
        { DPTWISE_FIELD_CHANNEL_OFFSET, 24 }, // 17
        { DPTWISE_FIELD_BANK_ADDR_IN_3, 12 }, // 18
        { DPTWISE_FIELD_BANK_ADDR_IN_2, 0 }, // 19
    },
};


// ElewInit field position
const vector<map<uint32_t, uint32_t>> InstTable::ElewInitFieldPos = {
    map<uint32_t, uint32_t> {
        { ELEWINIT_FIELD_OPCODE, 28 }, // 0
        { ELEWINIT_FIELD_DPDON, 24 }, // 1
        { ELEWINIT_FIELD_DPDBY, 20 }, // 2
        { ELEWINIT_FIELD_BANK_ID_IN, 14 }, // 3
        { ELEWINIT_FIELD_SHARE_PP, 13 }, // 4
    },
    map<uint32_t, uint32_t> {
        { ELEWINIT_FIELD_JUMP_BANK, 29 }, // 5
        { ELEWINIT_FIELD_ID, 27 }, // 6
        { ELEWINIT_FIELD_JUMP_READ_ENDL, 14 }, // 7
        { ELEWINIT_FIELD_BANK_ADDR_IN, 0 }, // 8
    },
    map<uint32_t, uint32_t> {
        { ELEWINIT_FIELD_SHIFT_READ, 12 }, // 9
        { ELEWINIT_FIELD_JUMP_READ, 0 }, // 10
    },
};


// Elew field position
const vector<map<uint32_t, uint32_t>> InstTable::ElewFieldPos = {
    map<uint32_t, uint32_t> {
        { ELEW_FIELD_OPCODE, 28 }, // 0
        { ELEW_FIELD_DPDON, 24 }, // 1
        { ELEW_FIELD_DPDBY, 20 }, // 2
        { ELEW_FIELD_BANK_ID_OUT, 14 }, // 3
        { ELEW_FIELD_ELEW_TYPE, 12 }, // 4
        { ELEW_FIELD_ACT_TYPE, 11 }, // 5
        { ELEW_FIELD_LENGTH, 0 }, // 6
    },
    map<uint32_t, uint32_t> {
        { ELEW_FIELD_VALID_PIXEL_PARALLEL, 29 }, // 7
        { ELEW_FIELD_NUM, 27 }, // 8
        { ELEW_FIELD_JUMP_WRITE_ENDL, 14 }, // 9
        { ELEW_FIELD_BANK_ADDR_OUT, 0 }, // 10
    },
    map<uint32_t, uint32_t> {
        { ELEW_FIELD_CHANNEL_GROUP, 17 }, // 11
        { ELEW_FIELD_SHIFT_WRITE, 12 }, // 12
        { ELEW_FIELD_JUMP_WRITE, 0 }, // 13
    },
};


// End field position
const vector<map<uint32_t, uint32_t>> InstTable::EndFieldPos = {
    map<uint32_t, uint32_t> {
        { END_FIELD_OPCODE, 28 }, // 0
        { END_FIELD_DPDON, 24 }, // 1
        { END_FIELD_DPDBY, 20 }, // 2
    },
};


// DumpBank field position
const vector<map<uint32_t, uint32_t>> InstTable::DumpBankFieldPos = {
    map<uint32_t, uint32_t> {
        { DUMPBANK_FIELD_OPCODE, 28 }, // 0
        { DUMPBANK_FIELD_DPDON, 24 }, // 1
        { DUMPBANK_FIELD_DPDBY, 20 }, // 2
        { DUMPBANK_FIELD_SAVE_NAME, 0 }, // 3
    },
    map<uint32_t, uint32_t> {
        { DUMPBANK_FIELD_SAVE_FMT, 26 }, // 4
        { DUMPBANK_FIELD_BANK_START, 18 }, // 5
        { DUMPBANK_FIELD_BANK_NUM, 10 }, // 6
    },
};


// DumpDDR field position
const vector<map<uint32_t, uint32_t>> InstTable::DumpDDRFieldPos = {
    map<uint32_t, uint32_t> {
        { DUMPDDR_FIELD_OPCODE, 28 }, // 0
        { DUMPDDR_FIELD_DPDON, 24 }, // 1
        { DUMPDDR_FIELD_DPDBY, 20 }, // 2
        { DUMPDDR_FIELD_SAVE_NAME, 0 }, // 3
    },
    map<uint32_t, uint32_t> {
        { DUMPDDR_FIELD_SAVE_FMT, 26 }, // 4
        { DUMPDDR_FIELD_REG_ID, 20 }, // 5
    },
    map<uint32_t, uint32_t> {
        { DUMPDDR_FIELD_DDR_START, 0 }, // 6
    },
    map<uint32_t, uint32_t> {
        { DUMPDDR_FIELD_DDR_SIZE, 0 }, // 7
    },
};


// DumpDDRSlice field position
const vector<map<uint32_t, uint32_t>> InstTable::DumpDDRSliceFieldPos = {
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_OPCODE, 28 }, // 0
        { DUMPDDRSLICE_FIELD_DPDON, 24 }, // 1
        { DUMPDDRSLICE_FIELD_DPDBY, 20 }, // 2
        { DUMPDDRSLICE_FIELD_SAVE_NAME, 0 }, // 3
    },
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_SAVE_FMT, 26 }, // 4
        { DUMPDDRSLICE_FIELD_REG_ID, 20 }, // 5
    },
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_DDR_START, 0 }, // 6
    },
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_HEIGHT, 16 }, // 7
        { DUMPDDRSLICE_FIELD_HEIGHT_STRIDE, 0 }, // 8
    },
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_WIDTH, 16 }, // 9
        { DUMPDDRSLICE_FIELD_WIDTH_STRIDE, 0 }, // 10
    },
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_CHANNEL, 16 }, // 11
        { DUMPDDRSLICE_FIELD_CHANNEL_STRIDE, 0 }, // 12
    },
};


// Load field length
const vector<map<uint32_t, uint32_t>> InstTable::LoadFieldLen = {
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_OPCODE, 4 }, // 0
        { LOAD_FIELD_DPDON, 4 }, // 1
        { LOAD_FIELD_DPDBY, 4 }, // 2
        { LOAD_FIELD_BANK_ID, 6 }, // 3
        { LOAD_FIELD_BANK_ADDR, 12 }, // 4
    },
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_JUMP_READ, 16 }, // 5
        { LOAD_FIELD_JUMP_WRITE, 10 }, // 6
    },
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_PAD_IDX, 5 }, // 7
        { LOAD_FIELD_CHANNEL, 14 }, // 8
        { LOAD_FIELD_LENGTH, 10 }, // 9
    },
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_PAD_START, 5 }, // 10
        { LOAD_FIELD_PAD_END, 5 }, // 11
        { LOAD_FIELD_MODE_AVG, 2 }, // 12
        { LOAD_FIELD_BROADCAST, 4 }, // 13
        { LOAD_FIELD_CONST_VALUE, 8 }, // 14
        { LOAD_FIELD_REG_ID, 3 }, // 15
    },
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_DDR_ADDR, 29 }, // 16
    },
};


// Save field length
const vector<map<uint32_t, uint32_t>> InstTable::SaveFieldLen = {
    map<uint32_t, uint32_t> {
        { SAVE_FIELD_OPCODE, 4 }, // 0
        { SAVE_FIELD_DPDON, 4 }, // 1
        { SAVE_FIELD_DPDBY, 4 }, // 2
        { SAVE_FIELD_BANK_ID, 6 }, // 3
        { SAVE_FIELD_BANK_ADDR, 12 }, // 4
    },
    map<uint32_t, uint32_t> {
        { SAVE_FIELD_HP_ID, 2 }, // 5
        { SAVE_FIELD_ARGMAX, 2 }, // 6
        { SAVE_FIELD_JUMP_WRITE, 16 }, // 7
        { SAVE_FIELD_JUMP_READ, 10 }, // 8
    },
    map<uint32_t, uint32_t> {
        { SAVE_FIELD_CHANNEL, 12 }, // 9
        { SAVE_FIELD_LENGTH, 10 }, // 10
    },
    map<uint32_t, uint32_t> {
        { SAVE_FIELD_CONST_EN, 1 }, // 11
        { SAVE_FIELD_CONST_VALUE, 8 }, // 12
        { SAVE_FIELD_REG_ID, 3 }, // 13
    },
    map<uint32_t, uint32_t> {
        { SAVE_FIELD_DDR_ADDR, 29 }, // 14
    },
};


// ConvInit field length
const vector<map<uint32_t, uint32_t>> InstTable::ConvInitFieldLen = {
    map<uint32_t, uint32_t> {
        { CONVINIT_FIELD_OPCODE, 4 }, // 0
        { CONVINIT_FIELD_DPDON, 4 }, // 1
        { CONVINIT_FIELD_DPDBY, 4 }, // 2
        { CONVINIT_FIELD_ACT_TYPE, 4 }, // 3
        { CONVINIT_FIELD_SHIFT_CUT, 6 }, // 4
        { CONVINIT_FIELD_JUMP_READ, 10 }, // 5
    },
    map<uint32_t, uint32_t> {
        { CONVINIT_FIELD_STRIDE_OUT, 4 }, // 6
        { CONVINIT_FIELD_SHIFT_BIAS, 6 }, // 7
        { CONVINIT_FIELD_JUMP_READ_ENDL, 10 }, // 8
    },
    map<uint32_t, uint32_t> {
        { CONVINIT_FIELD_STRIDE_H, 4 }, // 9
        { CONVINIT_FIELD_KERNEL_H, 4 }, // 10
        { CONVINIT_FIELD_VALID_PIXEL_PARALLEL, 3 }, // 11
        { CONVINIT_FIELD_STRIDE_OFFSET_IN, 3 }, // 12
        { CONVINIT_FIELD_JUMP_WRITE, 10 }, // 13
    },
    map<uint32_t, uint32_t> {
        { CONVINIT_FIELD_STRIDE_W, 4 }, // 14
        { CONVINIT_FIELD_KERNEL_W, 4 }, // 15
        { CONVINIT_FIELD_STRIDE_OFFSET_OUT, 3 }, // 16
        { CONVINIT_FIELD_JUMP_WRITE_ENDL, 10 }, // 17
    },
};


// Conv field length
const vector<map<uint32_t, uint32_t>> InstTable::ConvFieldLen = {
    map<uint32_t, uint32_t> {
        { CONV_FIELD_OPCODE, 4 }, // 0
        { CONV_FIELD_DPDON, 4 }, // 1
        { CONV_FIELD_DPDBY, 4 }, // 2
        { CONV_FIELD_PAD_LEFT, 4 }, // 3
        { CONV_FIELD_PAD_TOP, 4 }, // 4
        { CONV_FIELD_BANK_ADDR_IN, 12 }, // 5
    },
    map<uint32_t, uint32_t> {
        { CONV_FIELD_CHANNEL_GROUP, 8 }, // 6
        { CONV_FIELD_PAD_RIGHT, 4 }, // 7
        { CONV_FIELD_PAD_BOTTOM, 4 }, // 8
        { CONV_FIELD_BANK_ADDR_OUT, 12 }, // 9
    },
    map<uint32_t, uint32_t> {
        { CONV_FIELD_LENGTH, 10 }, // 10
        { CONV_FIELD_BANK_ID_IN, 6 }, // 11
        { CONV_FIELD_BANK_ADDR_WEIGHTS, 12 }, // 12
    },
    map<uint32_t, uint32_t> {
        { CONV_FIELD_BANK_ADDR_IN_1, 12 }, // 13
        { CONV_FIELD_BANK_ID_OUT, 6 }, // 14
        { CONV_FIELD_BANK_ADDR_BIAS, 12 }, // 15
    },
    map<uint32_t, uint32_t> {
        { CONV_FIELD_CHANNEL_OFFSET, 5 }, // 16
        { CONV_FIELD_BANK_ADDR_IN_3, 12 }, // 17
        { CONV_FIELD_BANK_ADDR_IN_2, 12 }, // 18
    },
};


// PoolInit field length
const vector<map<uint32_t, uint32_t>> InstTable::PoolInitFieldLen = {
    map<uint32_t, uint32_t> {
        { POOLINIT_FIELD_OPCODE, 4 }, // 0
        { POOLINIT_FIELD_DPDON, 4 }, // 1
        { POOLINIT_FIELD_DPDBY, 4 }, // 2
        { POOLINIT_FIELD_SHIFT_CUT, 4 }, // 3
        { POOLINIT_FIELD_KERNEL_W, 3 }, // 4
        { POOLINIT_FIELD_KERNEL_H, 3 }, // 5
        { POOLINIT_FIELD_JUMP_READ, 10 }, // 6
    },
    map<uint32_t, uint32_t> {
        { POOLINIT_FIELD_JUMP_WRITE, 10 }, // 7
        { POOLINIT_FIELD_STRIDE_OUT, 4 }, // 8
        { POOLINIT_FIELD_STRIDE_OFFSET_OUT, 3 }, // 9
        { POOLINIT_FIELD_VALID_PIXEL_PARALLEL, 3 }, // 10
        { POOLINIT_FIELD_STRIDE_OFFSET_IN, 3 }, // 11
        { POOLINIT_FIELD_STRIDE_W, 3 }, // 12
        { POOLINIT_FIELD_STRIDE_H, 3 }, // 13
        { POOLINIT_FIELD_POOL_TYPE, 2 }, // 14
    },
};


// Pool field length
const vector<map<uint32_t, uint32_t>> InstTable::PoolFieldLen = {
    map<uint32_t, uint32_t> {
        { POOL_FIELD_OPCODE, 4 }, // 0
        { POOL_FIELD_DPDON, 4 }, // 1
        { POOL_FIELD_DPDBY, 4 }, // 2
        { POOL_FIELD_CHANNEL_GROUP, 8 }, // 3
        { POOL_FIELD_BANK_ADDR_IN, 12 }, // 4
    },
    map<uint32_t, uint32_t> {
        { POOL_FIELD_JUMP_WRITE_ENDL, 12 }, // 5
        { POOL_FIELD_PAD_LEFT, 3 }, // 6
        { POOL_FIELD_PAD_TOP, 3 }, // 7
        { POOL_FIELD_PAD_RIGHT, 3 }, // 8
        { POOL_FIELD_PAD_BOTTOM, 3 }, // 9
        { POOL_FIELD_BANK_ID_IN, 6 }, // 10
    },
    map<uint32_t, uint32_t> {
        { POOL_FIELD_JUMP_READ_ENDL, 12 }, // 11
        { POOL_FIELD_LENGTH, 10 }, // 12
        { POOL_FIELD_BANK_ID_OUT, 6 }, // 13
    },
    map<uint32_t, uint32_t> {
        { POOL_FIELD_DOWNSAMPLE_KERNEL_W, 8 }, // 14
        { POOL_FIELD_BANK_ADDR_IN_1, 12 }, // 15
        { POOL_FIELD_BANK_ADDR_OUT, 12 }, // 16
    },
    map<uint32_t, uint32_t> {
        { POOL_FIELD_DOWNSAMPLE_KERNEL_H, 8 }, // 17
        { POOL_FIELD_BANK_ADDR_IN_3, 12 }, // 18
        { POOL_FIELD_BANK_ADDR_IN_2, 12 }, // 19
    },
};


// AluInit field length
const vector<map<uint32_t, uint32_t>> InstTable::AluInitFieldLen = {
    map<uint32_t, uint32_t> {
        { ALUINIT_FIELD_OPCODE, 4 }, // 0
        { ALUINIT_FIELD_DPDON, 4 }, // 1
        { ALUINIT_FIELD_DPDBY, 4 }, // 2
        { ALUINIT_FIELD_EXEC_MODE, 4 }, // 3
        { ALUINIT_FIELD_KERNEL_H, 8 }, // 4
        { ALUINIT_FIELD_KERNEL_W, 8 }, // 5
    },
    map<uint32_t, uint32_t> {
        { ALUINIT_FIELD_STRIDE_H, 8 }, // 6
        { ALUINIT_FIELD_STRIDE_W, 8 }, // 7
        { ALUINIT_FIELD_STRIDE_OFFSET_IN, 8 }, // 8
        { ALUINIT_FIELD_STRIDE_OFFSET_OUT, 8 }, // 9
    },
    map<uint32_t, uint32_t> {
        { ALUINIT_FIELD_STRIDE_OUT, 8 }, // 10
        { ALUINIT_FIELD_CHANNEL_GROUP, 8 }, // 11
        { ALUINIT_FIELD_CHANNEL_OFFSET, 5 }, // 12
        { ALUINIT_FIELD_LENGTH, 11 }, // 13
    },
    map<uint32_t, uint32_t> {
        { ALUINIT_FIELD_SHIFT_PRELU_P, 6 }, // 14
        { ALUINIT_FIELD_ACT_TYPE, 4 }, // 15
        { ALUINIT_FIELD_SHIFT_BIAS, 6 }, // 16
        { ALUINIT_FIELD_SHIFT_CUT, 6 }, // 17
        { ALUINIT_FIELD_SHIFT_PRELU_N, 6 }, // 18
    },
    map<uint32_t, uint32_t> {
        { ALUINIT_FIELD_MULTI_FACTOR, 8 }, // 19
        { ALUINIT_FIELD_JUMP_READ, 10 }, // 20
        { ALUINIT_FIELD_JUMP_READ_ENDL, 13 }, // 21
    },
    map<uint32_t, uint32_t> {
        { ALUINIT_FIELD_JUMP_READ_WEIGHTS, 9 }, // 22
        { ALUINIT_FIELD_JUMP_WRITE, 10 }, // 23
        { ALUINIT_FIELD_JUMP_WRITE_ENDL, 13 }, // 24
    },
};


// Alu field length
const vector<map<uint32_t, uint32_t>> InstTable::AluFieldLen = {
    map<uint32_t, uint32_t> {
        { ALU_FIELD_OPCODE, 4 }, // 0
        { ALU_FIELD_DPDON, 4 }, // 1
        { ALU_FIELD_DPDBY, 4 }, // 2
        { ALU_FIELD_BANK_ID_OUT, 6 }, // 3
        { ALU_FIELD_BANK_ADDR_OUT, 14 }, // 4
    },
    map<uint32_t, uint32_t> {
        { ALU_FIELD_PAD_BOTTOM, 8 }, // 5
        { ALU_FIELD_VALID_PIXEL_PARALLEL, 3 }, // 6
        { ALU_FIELD_BANK_ID_IN, 6 }, // 7
        { ALU_FIELD_BANK_ADDR_IN, 14 }, // 8
    },
    map<uint32_t, uint32_t> {
        { ALU_FIELD_PAD_TOP, 4 }, // 9
        { ALU_FIELD_PAD_LEFT, 4 }, // 10
        { ALU_FIELD_PAD_RIGHT, 8 }, // 11
        { ALU_FIELD_BANK_ADDR_IN_1, 14 }, // 12
    },
    map<uint32_t, uint32_t> {
        { ALU_FIELD_KERNEL_D, 4 }, // 13
        { ALU_FIELD_BANK_ADDR_WEIGHTS, 14 }, // 14
        { ALU_FIELD_BANK_ADDR_IN_2, 14 }, // 15
    },
    map<uint32_t, uint32_t> {
        { ALU_FIELD_BANK_ADDR_BIAS, 14 }, // 16
        { ALU_FIELD_BANK_ADDR_IN_3, 14 }, // 17
    },
};


// DWInit field length
const vector<map<uint32_t, uint32_t>> InstTable::DWInitFieldLen = {
    map<uint32_t, uint32_t> {
        { DWINIT_FIELD_OPCODE, 4 }, // 0
        { DWINIT_FIELD_DPDON, 4 }, // 1
        { DWINIT_FIELD_DPDBY, 4 }, // 2
        { DWINIT_FIELD_JUMP_READ, 10 }, // 3
        { DWINIT_FIELD_JUMP_READ_ENDL, 10 }, // 4
    },
    map<uint32_t, uint32_t> {
        { DWINIT_FIELD_STRIDE_OFFSET_IN, 3 }, // 5
        { DWINIT_FIELD_VALID_PIXEL_PARALLEL, 3 }, // 6
        { DWINIT_FIELD_STRIDE_W, 4 }, // 7
        { DWINIT_FIELD_STRIDE_H, 4 }, // 8
        { DWINIT_FIELD_KERNEL_W, 4 }, // 9
        { DWINIT_FIELD_KERNEL_H, 4 }, // 10
        { DWINIT_FIELD_JUMP_WRITE_ENDL, 10 }, // 11
    },
    map<uint32_t, uint32_t> {
        { DWINIT_FIELD_JUMP_WRITE, 10 }, // 12
        { DWINIT_FIELD_STRIDE_OUT, 4 }, // 13
        { DWINIT_FIELD_STRIDE_OFFSET_OUT, 3 }, // 14
        { DWINIT_FIELD_SHIFT_BIAS, 6 }, // 15
        { DWINIT_FIELD_SHIFT_CUT, 6 }, // 16
    },
};


// DptWise field length
const vector<map<uint32_t, uint32_t>> InstTable::DptWiseFieldLen = {
    map<uint32_t, uint32_t> {
        { DPTWISE_FIELD_OPCODE, 4 }, // 0
        { DPTWISE_FIELD_DPDON, 4 }, // 1
        { DPTWISE_FIELD_DPDBY, 4 }, // 2
        { DPTWISE_FIELD_CHANNEL_GROUP, 8 }, // 3
        { DPTWISE_FIELD_BANK_ADDR_IN, 12 }, // 4
    },
    map<uint32_t, uint32_t> {
        { DPTWISE_FIELD_PAD_LEFT, 4 }, // 5
        { DPTWISE_FIELD_PAD_RIGHT, 4 }, // 6
        { DPTWISE_FIELD_PAD_TOP, 4 }, // 7
        { DPTWISE_FIELD_PAD_BOTTOM, 4 }, // 8
        { DPTWISE_FIELD_BANK_ADDR_OUT, 12 }, // 9
    },
    map<uint32_t, uint32_t> {
        { DPTWISE_FIELD_ACT_TYPE, 4 }, // 10
        { DPTWISE_FIELD_LENGTH, 10 }, // 11
        { DPTWISE_FIELD_BANK_ID_IN, 6 }, // 12
        { DPTWISE_FIELD_BANK_ADDR_WEIGHTS, 12 }, // 13
    },
    map<uint32_t, uint32_t> {
        { DPTWISE_FIELD_BANK_ADDR_IN_1, 12 }, // 14
        { DPTWISE_FIELD_BANK_ID_OUT, 6 }, // 15
        { DPTWISE_FIELD_BANK_ADDR_BIAS, 12 }, // 16
    },
    map<uint32_t, uint32_t> {
        { DPTWISE_FIELD_CHANNEL_OFFSET, 5 }, // 17
        { DPTWISE_FIELD_BANK_ADDR_IN_3, 12 }, // 18
        { DPTWISE_FIELD_BANK_ADDR_IN_2, 12 }, // 19
    },
};


// ElewInit field length
const vector<map<uint32_t, uint32_t>> InstTable::ElewInitFieldLen = {
    map<uint32_t, uint32_t> {
        { ELEWINIT_FIELD_OPCODE, 4 }, // 0
        { ELEWINIT_FIELD_DPDON, 4 }, // 1
        { ELEWINIT_FIELD_DPDBY, 4 }, // 2
        { ELEWINIT_FIELD_BANK_ID_IN, 6 }, // 3
        { ELEWINIT_FIELD_SHARE_PP, 1 }, // 4
    },
    map<uint32_t, uint32_t> {
        { ELEWINIT_FIELD_JUMP_BANK, 3 }, // 5
        { ELEWINIT_FIELD_ID, 2 }, // 6
        { ELEWINIT_FIELD_JUMP_READ_ENDL, 13 }, // 7
        { ELEWINIT_FIELD_BANK_ADDR_IN, 14 }, // 8
    },
    map<uint32_t, uint32_t> {
        { ELEWINIT_FIELD_SHIFT_READ, 4 }, // 9
        { ELEWINIT_FIELD_JUMP_READ, 12 }, // 10
    },
};


// Elew field length
const vector<map<uint32_t, uint32_t>> InstTable::ElewFieldLen = {
    map<uint32_t, uint32_t> {
        { ELEW_FIELD_OPCODE, 4 }, // 0
        { ELEW_FIELD_DPDON, 4 }, // 1
        { ELEW_FIELD_DPDBY, 4 }, // 2
        { ELEW_FIELD_BANK_ID_OUT, 6 }, // 3
        { ELEW_FIELD_ELEW_TYPE, 2 }, // 4
        { ELEW_FIELD_ACT_TYPE, 1 }, // 5
        { ELEW_FIELD_LENGTH, 11 }, // 6
    },
    map<uint32_t, uint32_t> {
        { ELEW_FIELD_VALID_PIXEL_PARALLEL, 3 }, // 7
        { ELEW_FIELD_NUM, 2 }, // 8
        { ELEW_FIELD_JUMP_WRITE_ENDL, 13 }, // 9
        { ELEW_FIELD_BANK_ADDR_OUT, 14 }, // 10
    },
    map<uint32_t, uint32_t> {
        { ELEW_FIELD_CHANNEL_GROUP, 8 }, // 11
        { ELEW_FIELD_SHIFT_WRITE, 5 }, // 12
        { ELEW_FIELD_JUMP_WRITE, 12 }, // 13
    },
};


// End field length
const vector<map<uint32_t, uint32_t>> InstTable::EndFieldLen = {
    map<uint32_t, uint32_t> {
        { END_FIELD_OPCODE, 4 }, // 0
        { END_FIELD_DPDON, 4 }, // 1
        { END_FIELD_DPDBY, 4 }, // 2
    },
};


// DumpBank field length
const vector<map<uint32_t, uint32_t>> InstTable::DumpBankFieldLen = {
    map<uint32_t, uint32_t> {
        { DUMPBANK_FIELD_OPCODE, 4 }, // 0
        { DUMPBANK_FIELD_DPDON, 4 }, // 1
        { DUMPBANK_FIELD_DPDBY, 4 }, // 2
        { DUMPBANK_FIELD_SAVE_NAME, 20 }, // 3
    },
    map<uint32_t, uint32_t> {
        { DUMPBANK_FIELD_SAVE_FMT, 6 }, // 4
        { DUMPBANK_FIELD_BANK_START, 8 }, // 5
        { DUMPBANK_FIELD_BANK_NUM, 8 }, // 6
    },
};


// DumpDDR field length
const vector<map<uint32_t, uint32_t>> InstTable::DumpDDRFieldLen = {
    map<uint32_t, uint32_t> {
        { DUMPDDR_FIELD_OPCODE, 4 }, // 0
        { DUMPDDR_FIELD_DPDON, 4 }, // 1
        { DUMPDDR_FIELD_DPDBY, 4 }, // 2
        { DUMPDDR_FIELD_SAVE_NAME, 20 }, // 3
    },
    map<uint32_t, uint32_t> {
        { DUMPDDR_FIELD_SAVE_FMT, 6 }, // 4
        { DUMPDDR_FIELD_REG_ID, 6 }, // 5
    },
    map<uint32_t, uint32_t> {
        { DUMPDDR_FIELD_DDR_START, 32 }, // 6
    },
    map<uint32_t, uint32_t> {
        { DUMPDDR_FIELD_DDR_SIZE, 32 }, // 7
    },
};


// DumpDDRSlice field length
const vector<map<uint32_t, uint32_t>> InstTable::DumpDDRSliceFieldLen = {
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_OPCODE, 4 }, // 0
        { DUMPDDRSLICE_FIELD_DPDON, 4 }, // 1
        { DUMPDDRSLICE_FIELD_DPDBY, 4 }, // 2
        { DUMPDDRSLICE_FIELD_SAVE_NAME, 20 }, // 3
    },
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_SAVE_FMT, 6 }, // 4
        { DUMPDDRSLICE_FIELD_REG_ID, 6 }, // 5
    },
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_DDR_START, 32 }, // 6
    },
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_HEIGHT, 16 }, // 7
        { DUMPDDRSLICE_FIELD_HEIGHT_STRIDE, 16 }, // 8
    },
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_WIDTH, 16 }, // 9
        { DUMPDDRSLICE_FIELD_WIDTH_STRIDE, 16 }, // 10
    },
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_CHANNEL, 16 }, // 11
        { DUMPDDRSLICE_FIELD_CHANNEL_STRIDE, 16 }, // 12
    },
};


// inst opcode to inst type map
const unordered_map<uint32_t, uint32_t> InstTable::OPCode2InstType = {
    { 0x00, INST_TYPE_LOAD }, // 0
    { 0x04, INST_TYPE_SAVE }, // 1
    { 0x09, INST_TYPE_CONVINIT }, // 2
    { 0x08, INST_TYPE_CONV }, // 3
    { 0x06, INST_TYPE_POOLINIT }, // 4
    { 0x0C, INST_TYPE_POOL }, // 5
    { 0x01, INST_TYPE_ALUINIT }, // 6
    { 0x02, INST_TYPE_ALU }, // 7
    { 0x0B, INST_TYPE_DWINIT }, // 8
    { 0x0A, INST_TYPE_DPTWISE }, // 9
    { 0x0D, INST_TYPE_ELEWINIT }, // 10
    { 0x0E, INST_TYPE_ELEW }, // 11
    { 0x07, INST_TYPE_END }, // 12
    { 0xFF, INST_TYPE_DUMPBANK }, // 13
    { 0xFE, INST_TYPE_DUMPDDR }, // 14
    { 0xFD, INST_TYPE_DUMPDDRSLICE }, // 15
};


// inst name to inst type map
const unordered_map<string, uint32_t> InstTable::InstName2InstType = {
    { "LOAD", INST_TYPE_LOAD }, // 0
    { "SAVE", INST_TYPE_SAVE }, // 1
    { "CONVINIT", INST_TYPE_CONVINIT }, // 2
    { "CONV", INST_TYPE_CONV }, // 3
    { "POOLINIT", INST_TYPE_POOLINIT }, // 4
    { "POOL", INST_TYPE_POOL }, // 5
    { "ALUINIT", INST_TYPE_ALUINIT }, // 6
    { "ALU", INST_TYPE_ALU }, // 7
    { "DWINIT", INST_TYPE_DWINIT }, // 8
    { "DPTWISE", INST_TYPE_DPTWISE }, // 9
    { "ELEWINIT", INST_TYPE_ELEWINIT }, // 10
    { "ELEW", INST_TYPE_ELEW }, // 11
    { "END", INST_TYPE_END }, // 12
    { "DUMPBANK", INST_TYPE_DUMPBANK }, // 13
    { "DUMPDDR", INST_TYPE_DUMPDDR }, // 14
    { "DUMPDDRSLICE", INST_TYPE_DUMPDDRSLICE }, // 15
};


// inst type to field name map
const unordered_map<uint32_t, const vector<string> &> InstTable::InstFieldName = {
    { INST_TYPE_LOAD, LoadFieldName }, // 0
    { INST_TYPE_SAVE, SaveFieldName }, // 1
    { INST_TYPE_CONVINIT, ConvInitFieldName }, // 2
    { INST_TYPE_CONV, ConvFieldName }, // 3
    { INST_TYPE_POOLINIT, PoolInitFieldName }, // 4
    { INST_TYPE_POOL, PoolFieldName }, // 5
    { INST_TYPE_ALUINIT, AluInitFieldName }, // 6
    { INST_TYPE_ALU, AluFieldName }, // 7
    { INST_TYPE_DWINIT, DWInitFieldName }, // 8
    { INST_TYPE_DPTWISE, DptWiseFieldName }, // 9
    { INST_TYPE_ELEWINIT, ElewInitFieldName }, // 10
    { INST_TYPE_ELEW, ElewFieldName }, // 11
    { INST_TYPE_END, EndFieldName }, // 12
    { INST_TYPE_DUMPBANK, DumpBankFieldName }, // 13
    { INST_TYPE_DUMPDDR, DumpDDRFieldName }, // 14
    { INST_TYPE_DUMPDDRSLICE, DumpDDRSliceFieldName }, // 15
};


// inst type to field data type map
const unordered_map<uint32_t, const vector<int> &> InstTable::InstFieldDataType = {
    { INST_TYPE_LOAD, LoadFieldDataType }, // 0
    { INST_TYPE_SAVE, SaveFieldDataType }, // 1
    { INST_TYPE_CONVINIT, ConvInitFieldDataType }, // 2
    { INST_TYPE_CONV, ConvFieldDataType }, // 3
    { INST_TYPE_POOLINIT, PoolInitFieldDataType }, // 4
    { INST_TYPE_POOL, PoolFieldDataType }, // 5
    { INST_TYPE_ALUINIT, AluInitFieldDataType }, // 6
    { INST_TYPE_ALU, AluFieldDataType }, // 7
    { INST_TYPE_DWINIT, DWInitFieldDataType }, // 8
    { INST_TYPE_DPTWISE, DptWiseFieldDataType }, // 9
    { INST_TYPE_ELEWINIT, ElewInitFieldDataType }, // 10
    { INST_TYPE_ELEW, ElewFieldDataType }, // 11
    { INST_TYPE_END, EndFieldDataType }, // 12
    { INST_TYPE_DUMPBANK, DumpBankFieldDataType }, // 13
    { INST_TYPE_DUMPDDR, DumpDDRFieldDataType }, // 14
    { INST_TYPE_DUMPDDRSLICE, DumpDDRSliceFieldDataType }, // 15
};


// inst type to field minus map
const unordered_map<uint32_t, const vector<int> &> InstTable::InstFieldMinus = {
    { INST_TYPE_LOAD, LoadFieldMinus }, // 0
    { INST_TYPE_SAVE, SaveFieldMinus }, // 1
    { INST_TYPE_CONVINIT, ConvInitFieldMinus }, // 2
    { INST_TYPE_CONV, ConvFieldMinus }, // 3
    { INST_TYPE_POOLINIT, PoolInitFieldMinus }, // 4
    { INST_TYPE_POOL, PoolFieldMinus }, // 5
    { INST_TYPE_ALUINIT, AluInitFieldMinus }, // 6
    { INST_TYPE_ALU, AluFieldMinus }, // 7
    { INST_TYPE_DWINIT, DWInitFieldMinus }, // 8
    { INST_TYPE_DPTWISE, DptWiseFieldMinus }, // 9
    { INST_TYPE_ELEWINIT, ElewInitFieldMinus }, // 10
    { INST_TYPE_ELEW, ElewFieldMinus }, // 11
    { INST_TYPE_END, EndFieldMinus }, // 12
    { INST_TYPE_DUMPBANK, DumpBankFieldMinus }, // 13
    { INST_TYPE_DUMPDDR, DumpDDRFieldMinus }, // 14
    { INST_TYPE_DUMPDDRSLICE, DumpDDRSliceFieldMinus }, // 15
};


// inst type to field position map
const unordered_map<uint32_t, const vector<map<uint32_t, uint32_t>> &> InstTable::InstFieldPos = {
    { INST_TYPE_LOAD, LoadFieldPos }, // 0
    { INST_TYPE_SAVE, SaveFieldPos }, // 1
    { INST_TYPE_CONVINIT, ConvInitFieldPos }, // 2
    { INST_TYPE_CONV, ConvFieldPos }, // 3
    { INST_TYPE_POOLINIT, PoolInitFieldPos }, // 4
    { INST_TYPE_POOL, PoolFieldPos }, // 5
    { INST_TYPE_ALUINIT, AluInitFieldPos }, // 6
    { INST_TYPE_ALU, AluFieldPos }, // 7
    { INST_TYPE_DWINIT, DWInitFieldPos }, // 8
    { INST_TYPE_DPTWISE, DptWiseFieldPos }, // 9
    { INST_TYPE_ELEWINIT, ElewInitFieldPos }, // 10
    { INST_TYPE_ELEW, ElewFieldPos }, // 11
    { INST_TYPE_END, EndFieldPos }, // 12
    { INST_TYPE_DUMPBANK, DumpBankFieldPos }, // 13
    { INST_TYPE_DUMPDDR, DumpDDRFieldPos }, // 14
    { INST_TYPE_DUMPDDRSLICE, DumpDDRSliceFieldPos }, // 15
};


// inst type to field length map
const unordered_map<uint32_t, const vector<map<uint32_t, uint32_t>> &> InstTable::InstFieldLen = {
    { INST_TYPE_LOAD, LoadFieldLen }, // 0
    { INST_TYPE_SAVE, SaveFieldLen }, // 1
    { INST_TYPE_CONVINIT, ConvInitFieldLen }, // 2
    { INST_TYPE_CONV, ConvFieldLen }, // 3
    { INST_TYPE_POOLINIT, PoolInitFieldLen }, // 4
    { INST_TYPE_POOL, PoolFieldLen }, // 5
    { INST_TYPE_ALUINIT, AluInitFieldLen }, // 6
    { INST_TYPE_ALU, AluFieldLen }, // 7
    { INST_TYPE_DWINIT, DWInitFieldLen }, // 8
    { INST_TYPE_DPTWISE, DptWiseFieldLen }, // 9
    { INST_TYPE_ELEWINIT, ElewInitFieldLen }, // 10
    { INST_TYPE_ELEW, ElewFieldLen }, // 11
    { INST_TYPE_END, EndFieldLen }, // 12
    { INST_TYPE_DUMPBANK, DumpBankFieldLen }, // 13
    { INST_TYPE_DUMPDDR, DumpDDRFieldLen }, // 14
    { INST_TYPE_DUMPDDRSLICE, DumpDDRSliceFieldLen }, // 15
};


}

namespace DPUV_3E {
// instruction version
const string InstTable::INST_VERSION = "DPUV3E";


// category name vector
const vector<string> InstTable::CategoryName = {
    "LOAD",
    "SAVE",
    "CONV",
    "MISC",
};


// inst name vector
const vector<string> InstTable::InstName = {
    "LOAD", // 0
    "SAVE", // 1
    "CONVINIT", // 2
    "CONV", // 3
    "POOLINIT", // 4
    "POOL", // 5
    "DWINIT", // 6
    "DPTWISE", // 7
    "ELEWINIT", // 8
    "ELEW", // 9
    "END", // 10
    "DUMPBANK", // 11
    "DUMPDDR", // 12
    "DUMPDDRSLICE", // 13
};


// lower inst name vector
const vector<string> InstTable::InstNameLower = {
    "load", // 0
    "save", // 1
    "convinit", // 2
    "conv", // 3
    "poolinit", // 4
    "pool", // 5
    "dwinit", // 6
    "dptwise", // 7
    "elewinit", // 8
    "elew", // 9
    "end", // 10
    "dumpbank", // 11
    "dumpddr", // 12
    "dumpddrslice", // 13
};


// inst category
const vector<Category> InstTable::InstCategory = {
    Category::INST_CATEGORY_LOAD, // LOAD
    Category::INST_CATEGORY_SAVE, // SAVE
    Category::INST_CATEGORY_CONV, // CONVINIT
    Category::INST_CATEGORY_CONV, // CONV
    Category::INST_CATEGORY_MISC, // POOLINIT
    Category::INST_CATEGORY_MISC, // POOL
    Category::INST_CATEGORY_MISC, // DWINIT
    Category::INST_CATEGORY_MISC, // DPTWISE
    Category::INST_CATEGORY_MISC, // ELEWINIT
    Category::INST_CATEGORY_MISC, // ELEW
    Category::INST_CATEGORY_SAVE, // END
    Category::INST_CATEGORY_MISC, // DUMPBANK
    Category::INST_CATEGORY_MISC, // DUMPDDR
    Category::INST_CATEGORY_MISC, // DUMPDDRSLICE
};


// inst word number
const vector<uint32_t> InstTable::WordNum = {
    6, // LOAD
    4, // SAVE
    4, // CONVINIT
    5, // CONV
    2, // POOLINIT
    5, // POOL
    4, // DWINIT
    5, // DPTWISE
    2, // ELEWINIT
    3, // ELEW
    1, // END
    2, // DUMPBANK
    4, // DUMPDDR
    6, // DUMPDDRSLICE
};


// inst field number
const vector<uint32_t> InstTable::FieldNum = {
    LOAD_FIELD_MAX, // 0
    SAVE_FIELD_MAX, // 1
    CONVINIT_FIELD_MAX, // 2
    CONV_FIELD_MAX, // 3
    POOLINIT_FIELD_MAX, // 4
    POOL_FIELD_MAX, // 5
    DWINIT_FIELD_MAX, // 6
    DPTWISE_FIELD_MAX, // 7
    ELEWINIT_FIELD_MAX, // 8
    ELEW_FIELD_MAX, // 9
    END_FIELD_MAX, // 10
    DUMPBANK_FIELD_MAX, // 11
    DUMPDDR_FIELD_MAX, // 12
    DUMPDDRSLICE_FIELD_MAX, // 13
};


// inst opcode
const vector<uint32_t> InstTable::OPCode = {
    0x00, // LOAD
    0x04, // SAVE
    0x09, // CONVINIT
    0x08, // CONV
    0x06, // POOLINIT
    0x0C, // POOL
    0x0B, // DWINIT
    0x0A, // DPTWISE
    0x0D, // ELEWINIT
    0x0E, // ELEW
    0x07, // END
    0xFF, // DUMPBANK
    0xFE, // DUMPDDR
    0xFD, // DUMPDDRSLICE
};


// Load field name
const vector<string> InstTable::LoadFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "bank_id", // 3
    "bank_addr", // 4
    "pad_start", // 5
    "pad_end", // 6
    "pad_idx", // 7
    "jump_read", // 8
    "jump_write", // 9
    "length", // 10
    "mode_avg", // 11
    "channel", // 12
    "reg_id", // 13
    "ddr_addr", // 14
    "ddr_mode", // 15
    "output_channel_num", // 16
    "jump_read_endl", // 17
    "reg_id_1", // 18
    "ddr_addr_1", // 19
};


// Save field name
const vector<string> InstTable::SaveFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "bank_id", // 3
    "bank_addr", // 4
    "jump_write", // 5
    "jump_read", // 6
    "length", // 7
    "channel", // 8
    "reg_id", // 9
    "ddr_addr", // 10
};


// ConvInit field name
const vector<string> InstTable::ConvInitFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "act_type", // 3
    "shift_cut", // 4
    "jump_read", // 5
    "stride_out", // 6
    "shift_bias", // 7
    "jump_read_endl", // 8
    "stride_h", // 9
    "kernel_h", // 10
    "valid_pixel_parallel", // 11
    "stride_offset_in", // 12
    "jump_write", // 13
    "stride_w", // 14
    "kernel_w", // 15
    "stride_offset_out", // 16
    "jump_write_endl", // 17
};


// Conv field name
const vector<string> InstTable::ConvFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "bank_id_in", // 3
    "bank_addr_in", // 4
    "pad_top", // 5
    "channel_group", // 6
    "bank_id_out", // 7
    "bank_addr_out", // 8
    "pad_bottom", // 9
    "channel_offset", // 10
    "length", // 11
    "bank_addr_in_1", // 12
    "pad_left", // 13
    "bank_addr_weights", // 14
    "bank_addr_in_2", // 15
    "pad_right", // 16
    "bank_addr_bias", // 17
    "bank_addr_in_3", // 18
};


// PoolInit field name
const vector<string> InstTable::PoolInitFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "shift_cut", // 3
    "kernel_w", // 4
    "kernel_h", // 5
    "jump_read", // 6
    "jump_write", // 7
    "stride_out", // 8
    "stride_offset_out", // 9
    "valid_pixel_parallel", // 10
    "stride_offset_in", // 11
    "stride_w", // 12
    "stride_h", // 13
    "pool_type", // 14
};


// Pool field name
const vector<string> InstTable::PoolFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "bank_id_in", // 3
    "bank_addr_in", // 4
    "jump_read_endl", // 5
    "bank_id_out", // 6
    "pad_top", // 7
    "pad_bottom", // 8
    "pad_left", // 9
    "pad_right", // 10
    "jump_write_endl", // 11
    "channel_group", // 12
    "length", // 13
    "bank_addr_in_1", // 14
    "bank_addr_in_2", // 15
    "bank_addr_in_3", // 16
    "bank_addr_out", // 17
};


// DWInit field name
const vector<string> InstTable::DWInitFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "stride_out", // 3
    "act_type", // 4
    "jump_read_endl", // 5
    "kernel_w", // 6
    "kernel_h", // 7
    "stride_w", // 8
    "stride_h", // 9
    "valid_pixel_parallel", // 10
    "jump_write_endl", // 11
    "stride_offset_in", // 12
    "stride_offset_out", // 13
    "shift_bias", // 14
    "shift_cut", // 15
    "jump_read", // 16
    "jump_write", // 17
};


// DptWise field name
const vector<string> InstTable::DptWiseFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "bank_id_in", // 3
    "bank_addr_in", // 4
    "pad_top", // 5
    "channel_group", // 6
    "bank_id_out", // 7
    "bank_addr_out", // 8
    "pad_bottom", // 9
    "channel_offset", // 10
    "length", // 11
    "bank_addr_in_1", // 12
    "pad_left", // 13
    "bank_addr_weights", // 14
    "bank_addr_in_2", // 15
    "pad_right", // 16
    "bank_addr_bias", // 17
    "bank_addr_in_3", // 18
};


// ElewInit field name
const vector<string> InstTable::ElewInitFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "bank_id_in", // 3
    "shift_read", // 4
    "jump_read", // 5
    "id", // 6
    "jump_read_endl", // 7
    "bank_addr_in", // 8
};


// Elew field name
const vector<string> InstTable::ElewFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "bank_id_out", // 3
    "shift_write", // 4
    "jump_write", // 5
    "channel_group", // 6
    "length", // 7
    "act_type", // 8
    "bank_addr_out", // 9
    "num", // 10
    "valid_pixel_parallel", // 11
    "jump_write_endl", // 12
};


// End field name
const vector<string> InstTable::EndFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
};


// DumpBank field name
const vector<string> InstTable::DumpBankFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "save_name", // 3
    "save_fmt", // 4
    "bank_start", // 5
    "bank_num", // 6
};


// DumpDDR field name
const vector<string> InstTable::DumpDDRFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "save_name", // 3
    "save_fmt", // 4
    "reg_id", // 5
    "ddr_start", // 6
    "ddr_size", // 7
};


// DumpDDRSlice field name
const vector<string> InstTable::DumpDDRSliceFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "save_name", // 3
    "save_fmt", // 4
    "reg_id", // 5
    "ddr_start", // 6
    "height", // 7
    "height_stride", // 8
    "width", // 9
    "width_stride", // 10
    "channel", // 11
    "channel_stride", // 12
};


// Load field data type
const vector<int> InstTable::LoadFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_UINT, // bank_id(3)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr(4)
    INST_FIELD_DATA_TYPE_UINT, // pad_start(5)
    INST_FIELD_DATA_TYPE_UINT, // pad_end(6)
    INST_FIELD_DATA_TYPE_UINT, // pad_idx(7)
    INST_FIELD_DATA_TYPE_UINT, // jump_read(8)
    INST_FIELD_DATA_TYPE_UINT, // jump_write(9)
    INST_FIELD_DATA_TYPE_UINT, // length(10)
    INST_FIELD_DATA_TYPE_UINT, // mode_avg(11)
    INST_FIELD_DATA_TYPE_UINT, // channel(12)
    INST_FIELD_DATA_TYPE_UINT, // reg_id(13)
    INST_FIELD_DATA_TYPE_UINT, // ddr_addr(14)
    INST_FIELD_DATA_TYPE_UINT, // ddr_mode(15)
    INST_FIELD_DATA_TYPE_UINT, // output_channel_num(16)
    INST_FIELD_DATA_TYPE_UINT, // jump_read_endl(17)
    INST_FIELD_DATA_TYPE_UINT, // reg_id_1(18)
    INST_FIELD_DATA_TYPE_UINT, // ddr_addr_1(19)
};


// Save field data type
const vector<int> InstTable::SaveFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_UINT, // bank_id(3)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr(4)
    INST_FIELD_DATA_TYPE_UINT, // jump_write(5)
    INST_FIELD_DATA_TYPE_UINT, // jump_read(6)
    INST_FIELD_DATA_TYPE_UINT, // length(7)
    INST_FIELD_DATA_TYPE_UINT, // channel(8)
    INST_FIELD_DATA_TYPE_UINT, // reg_id(9)
    INST_FIELD_DATA_TYPE_UINT, // ddr_addr(10)
};


// ConvInit field data type
const vector<int> InstTable::ConvInitFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_UINT, // act_type(3)
    INST_FIELD_DATA_TYPE_UINT, // shift_cut(4)
    INST_FIELD_DATA_TYPE_UINT, // jump_read(5)
    INST_FIELD_DATA_TYPE_UINT, // stride_out(6)
    INST_FIELD_DATA_TYPE_UINT, // shift_bias(7)
    INST_FIELD_DATA_TYPE_UINT, // jump_read_endl(8)
    INST_FIELD_DATA_TYPE_UINT, // stride_h(9)
    INST_FIELD_DATA_TYPE_UINT, // kernel_h(10)
    INST_FIELD_DATA_TYPE_UINT, // valid_pixel_parallel(11)
    INST_FIELD_DATA_TYPE_UINT, // stride_offset_in(12)
    INST_FIELD_DATA_TYPE_UINT, // jump_write(13)
    INST_FIELD_DATA_TYPE_UINT, // stride_w(14)
    INST_FIELD_DATA_TYPE_UINT, // kernel_w(15)
    INST_FIELD_DATA_TYPE_UINT, // stride_offset_out(16)
    INST_FIELD_DATA_TYPE_UINT, // jump_write_endl(17)
};


// Conv field data type
const vector<int> InstTable::ConvFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_UINT, // bank_id_in(3)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in(4)
    INST_FIELD_DATA_TYPE_UINT, // pad_top(5)
    INST_FIELD_DATA_TYPE_UINT, // channel_group(6)
    INST_FIELD_DATA_TYPE_UINT, // bank_id_out(7)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_out(8)
    INST_FIELD_DATA_TYPE_UINT, // pad_bottom(9)
    INST_FIELD_DATA_TYPE_UINT, // channel_offset(10)
    INST_FIELD_DATA_TYPE_UINT, // length(11)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in_1(12)
    INST_FIELD_DATA_TYPE_UINT, // pad_left(13)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_weights(14)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in_2(15)
    INST_FIELD_DATA_TYPE_UINT, // pad_right(16)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_bias(17)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in_3(18)
};


// PoolInit field data type
const vector<int> InstTable::PoolInitFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_UINT, // shift_cut(3)
    INST_FIELD_DATA_TYPE_UINT, // kernel_w(4)
    INST_FIELD_DATA_TYPE_UINT, // kernel_h(5)
    INST_FIELD_DATA_TYPE_UINT, // jump_read(6)
    INST_FIELD_DATA_TYPE_UINT, // jump_write(7)
    INST_FIELD_DATA_TYPE_UINT, // stride_out(8)
    INST_FIELD_DATA_TYPE_UINT, // stride_offset_out(9)
    INST_FIELD_DATA_TYPE_UINT, // valid_pixel_parallel(10)
    INST_FIELD_DATA_TYPE_UINT, // stride_offset_in(11)
    INST_FIELD_DATA_TYPE_UINT, // stride_w(12)
    INST_FIELD_DATA_TYPE_UINT, // stride_h(13)
    INST_FIELD_DATA_TYPE_UINT, // pool_type(14)
};


// Pool field data type
const vector<int> InstTable::PoolFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_UINT, // bank_id_in(3)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in(4)
    INST_FIELD_DATA_TYPE_UINT, // jump_read_endl(5)
    INST_FIELD_DATA_TYPE_UINT, // bank_id_out(6)
    INST_FIELD_DATA_TYPE_UINT, // pad_top(7)
    INST_FIELD_DATA_TYPE_UINT, // pad_bottom(8)
    INST_FIELD_DATA_TYPE_UINT, // pad_left(9)
    INST_FIELD_DATA_TYPE_UINT, // pad_right(10)
    INST_FIELD_DATA_TYPE_UINT, // jump_write_endl(11)
    INST_FIELD_DATA_TYPE_UINT, // channel_group(12)
    INST_FIELD_DATA_TYPE_UINT, // length(13)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in_1(14)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in_2(15)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in_3(16)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_out(17)
};


// DWInit field data type
const vector<int> InstTable::DWInitFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_UINT, // stride_out(3)
    INST_FIELD_DATA_TYPE_UINT, // act_type(4)
    INST_FIELD_DATA_TYPE_UINT, // jump_read_endl(5)
    INST_FIELD_DATA_TYPE_UINT, // kernel_w(6)
    INST_FIELD_DATA_TYPE_UINT, // kernel_h(7)
    INST_FIELD_DATA_TYPE_UINT, // stride_w(8)
    INST_FIELD_DATA_TYPE_UINT, // stride_h(9)
    INST_FIELD_DATA_TYPE_UINT, // valid_pixel_parallel(10)
    INST_FIELD_DATA_TYPE_UINT, // jump_write_endl(11)
    INST_FIELD_DATA_TYPE_UINT, // stride_offset_in(12)
    INST_FIELD_DATA_TYPE_UINT, // stride_offset_out(13)
    INST_FIELD_DATA_TYPE_UINT, // shift_bias(14)
    INST_FIELD_DATA_TYPE_UINT, // shift_cut(15)
    INST_FIELD_DATA_TYPE_UINT, // jump_read(16)
    INST_FIELD_DATA_TYPE_UINT, // jump_write(17)
};


// DptWise field data type
const vector<int> InstTable::DptWiseFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_UINT, // bank_id_in(3)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in(4)
    INST_FIELD_DATA_TYPE_UINT, // pad_top(5)
    INST_FIELD_DATA_TYPE_UINT, // channel_group(6)
    INST_FIELD_DATA_TYPE_UINT, // bank_id_out(7)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_out(8)
    INST_FIELD_DATA_TYPE_UINT, // pad_bottom(9)
    INST_FIELD_DATA_TYPE_UINT, // channel_offset(10)
    INST_FIELD_DATA_TYPE_UINT, // length(11)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in_1(12)
    INST_FIELD_DATA_TYPE_UINT, // pad_left(13)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_weights(14)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in_2(15)
    INST_FIELD_DATA_TYPE_UINT, // pad_right(16)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_bias(17)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in_3(18)
};


// ElewInit field data type
const vector<int> InstTable::ElewInitFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_UINT, // bank_id_in(3)
    INST_FIELD_DATA_TYPE_UINT, // shift_read(4)
    INST_FIELD_DATA_TYPE_UINT, // jump_read(5)
    INST_FIELD_DATA_TYPE_UINT, // id(6)
    INST_FIELD_DATA_TYPE_UINT, // jump_read_endl(7)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in(8)
};


// Elew field data type
const vector<int> InstTable::ElewFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_UINT, // bank_id_out(3)
    INST_FIELD_DATA_TYPE_UINT, // shift_write(4)
    INST_FIELD_DATA_TYPE_UINT, // jump_write(5)
    INST_FIELD_DATA_TYPE_UINT, // channel_group(6)
    INST_FIELD_DATA_TYPE_UINT, // length(7)
    INST_FIELD_DATA_TYPE_UINT, // act_type(8)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_out(9)
    INST_FIELD_DATA_TYPE_UINT, // num(10)
    INST_FIELD_DATA_TYPE_UINT, // valid_pixel_parallel(11)
    INST_FIELD_DATA_TYPE_UINT, // jump_write_endl(12)
};


// End field data type
const vector<int> InstTable::EndFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
};


// DumpBank field data type
const vector<int> InstTable::DumpBankFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_STRING, // save_name(3)
    INST_FIELD_DATA_TYPE_UINT, // save_fmt(4)
    INST_FIELD_DATA_TYPE_UINT, // bank_start(5)
    INST_FIELD_DATA_TYPE_UINT, // bank_num(6)
};


// DumpDDR field data type
const vector<int> InstTable::DumpDDRFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_STRING, // save_name(3)
    INST_FIELD_DATA_TYPE_UINT, // save_fmt(4)
    INST_FIELD_DATA_TYPE_UINT, // reg_id(5)
    INST_FIELD_DATA_TYPE_UINT, // ddr_start(6)
    INST_FIELD_DATA_TYPE_UINT, // ddr_size(7)
};


// DumpDDRSlice field data type
const vector<int> InstTable::DumpDDRSliceFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_STRING, // save_name(3)
    INST_FIELD_DATA_TYPE_UINT, // save_fmt(4)
    INST_FIELD_DATA_TYPE_UINT, // reg_id(5)
    INST_FIELD_DATA_TYPE_UINT, // ddr_start(6)
    INST_FIELD_DATA_TYPE_UINT, // height(7)
    INST_FIELD_DATA_TYPE_UINT, // height_stride(8)
    INST_FIELD_DATA_TYPE_UINT, // width(9)
    INST_FIELD_DATA_TYPE_UINT, // width_stride(10)
    INST_FIELD_DATA_TYPE_UINT, // channel(11)
    INST_FIELD_DATA_TYPE_UINT, // channel_stride(12)
};


// Load field minus value
const vector<int> InstTable::LoadFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // bank_id(3)
    0, // bank_addr(4)
    0, // pad_start(5)
    0, // pad_end(6)
    1, // pad_idx(7)
    1, // jump_read(8)
    1, // jump_write(9)
    1, // length(10)
    0, // mode_avg(11)
    1, // channel(12)
    0, // reg_id(13)
    0, // ddr_addr(14)
    0, // ddr_mode(15)
    1, // output_channel_num(16)
    1, // jump_read_endl(17)
    0, // reg_id_1(18)
    0, // ddr_addr_1(19)
};


// Save field minus value
const vector<int> InstTable::SaveFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // bank_id(3)
    0, // bank_addr(4)
    1, // jump_write(5)
    1, // jump_read(6)
    1, // length(7)
    1, // channel(8)
    0, // reg_id(9)
    0, // ddr_addr(10)
};


// ConvInit field minus value
const vector<int> InstTable::ConvInitFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // act_type(3)
    0, // shift_cut(4)
    1, // jump_read(5)
    1, // stride_out(6)
    0, // shift_bias(7)
    1, // jump_read_endl(8)
    1, // stride_h(9)
    1, // kernel_h(10)
    1, // valid_pixel_parallel(11)
    0, // stride_offset_in(12)
    1, // jump_write(13)
    1, // stride_w(14)
    1, // kernel_w(15)
    0, // stride_offset_out(16)
    1, // jump_write_endl(17)
};


// Conv field minus value
const vector<int> InstTable::ConvFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // bank_id_in(3)
    0, // bank_addr_in(4)
    0, // pad_top(5)
    1, // channel_group(6)
    0, // bank_id_out(7)
    0, // bank_addr_out(8)
    0, // pad_bottom(9)
    0, // channel_offset(10)
    1, // length(11)
    0, // bank_addr_in_1(12)
    0, // pad_left(13)
    0, // bank_addr_weights(14)
    0, // bank_addr_in_2(15)
    0, // pad_right(16)
    0, // bank_addr_bias(17)
    0, // bank_addr_in_3(18)
};


// PoolInit field minus value
const vector<int> InstTable::PoolInitFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // shift_cut(3)
    1, // kernel_w(4)
    1, // kernel_h(5)
    1, // jump_read(6)
    1, // jump_write(7)
    1, // stride_out(8)
    0, // stride_offset_out(9)
    1, // valid_pixel_parallel(10)
    0, // stride_offset_in(11)
    1, // stride_w(12)
    1, // stride_h(13)
    0, // pool_type(14)
};


// Pool field minus value
const vector<int> InstTable::PoolFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // bank_id_in(3)
    0, // bank_addr_in(4)
    1, // jump_read_endl(5)
    0, // bank_id_out(6)
    0, // pad_top(7)
    0, // pad_bottom(8)
    0, // pad_left(9)
    0, // pad_right(10)
    1, // jump_write_endl(11)
    1, // channel_group(12)
    1, // length(13)
    0, // bank_addr_in_1(14)
    0, // bank_addr_in_2(15)
    0, // bank_addr_in_3(16)
    0, // bank_addr_out(17)
};


// DWInit field minus value
const vector<int> InstTable::DWInitFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    1, // stride_out(3)
    0, // act_type(4)
    1, // jump_read_endl(5)
    1, // kernel_w(6)
    1, // kernel_h(7)
    1, // stride_w(8)
    1, // stride_h(9)
    1, // valid_pixel_parallel(10)
    1, // jump_write_endl(11)
    0, // stride_offset_in(12)
    0, // stride_offset_out(13)
    0, // shift_bias(14)
    0, // shift_cut(15)
    1, // jump_read(16)
    1, // jump_write(17)
};


// DptWise field minus value
const vector<int> InstTable::DptWiseFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // bank_id_in(3)
    0, // bank_addr_in(4)
    0, // pad_top(5)
    1, // channel_group(6)
    0, // bank_id_out(7)
    0, // bank_addr_out(8)
    0, // pad_bottom(9)
    0, // channel_offset(10)
    1, // length(11)
    0, // bank_addr_in_1(12)
    0, // pad_left(13)
    0, // bank_addr_weights(14)
    0, // bank_addr_in_2(15)
    0, // pad_right(16)
    0, // bank_addr_bias(17)
    0, // bank_addr_in_3(18)
};


// ElewInit field minus value
const vector<int> InstTable::ElewInitFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // bank_id_in(3)
    0, // shift_read(4)
    1, // jump_read(5)
    0, // id(6)
    1, // jump_read_endl(7)
    0, // bank_addr_in(8)
};


// Elew field minus value
const vector<int> InstTable::ElewFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // bank_id_out(3)
    0, // shift_write(4)
    1, // jump_write(5)
    1, // channel_group(6)
    1, // length(7)
    0, // act_type(8)
    0, // bank_addr_out(9)
    1, // num(10)
    1, // valid_pixel_parallel(11)
    1, // jump_write_endl(12)
};


// End field minus value
const vector<int> InstTable::EndFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
};


// DumpBank field minus value
const vector<int> InstTable::DumpBankFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // save_name(3)
    0, // save_fmt(4)
    0, // bank_start(5)
    0, // bank_num(6)
};


// DumpDDR field minus value
const vector<int> InstTable::DumpDDRFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // save_name(3)
    0, // save_fmt(4)
    0, // reg_id(5)
    0, // ddr_start(6)
    0, // ddr_size(7)
};


// DumpDDRSlice field minus value
const vector<int> InstTable::DumpDDRSliceFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // save_name(3)
    0, // save_fmt(4)
    0, // reg_id(5)
    0, // ddr_start(6)
    0, // height(7)
    0, // height_stride(8)
    0, // width(9)
    0, // width_stride(10)
    0, // channel(11)
    0, // channel_stride(12)
};


// Load field position
const vector<map<uint32_t, uint32_t>> InstTable::LoadFieldPos = {
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_OPCODE, 28 }, // 0
        { LOAD_FIELD_DPDON, 24 }, // 1
        { LOAD_FIELD_DPDBY, 20 }, // 2
        { LOAD_FIELD_BANK_ID, 14 }, // 3
        { LOAD_FIELD_BANK_ADDR, 0 }, // 4
    },
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_PAD_START, 26 }, // 5
        { LOAD_FIELD_PAD_END, 21 }, // 6
        { LOAD_FIELD_PAD_IDX, 16 }, // 7
        { LOAD_FIELD_JUMP_READ, 0 }, // 8
    },
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_JUMP_WRITE, 24 }, // 9
        { LOAD_FIELD_LENGTH, 14 }, // 10
        { LOAD_FIELD_MODE_AVG, 12 }, // 11
        { LOAD_FIELD_CHANNEL, 0 }, // 12
    },
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_REG_ID, 29 }, // 13
        { LOAD_FIELD_DDR_ADDR, 0 }, // 14
    },
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_DDR_MODE, 23 }, // 15
        { LOAD_FIELD_OUTPUT_CHANNEL_NUM, 16 }, // 16
        { LOAD_FIELD_JUMP_READ_ENDL, 0 }, // 17
    },
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_REG_ID_1, 29 }, // 18
        { LOAD_FIELD_DDR_ADDR_1, 0 }, // 19
    },
};


// Save field position
const vector<map<uint32_t, uint32_t>> InstTable::SaveFieldPos = {
    map<uint32_t, uint32_t> {
        { SAVE_FIELD_OPCODE, 28 }, // 0
        { SAVE_FIELD_DPDON, 24 }, // 1
        { SAVE_FIELD_DPDBY, 20 }, // 2
        { SAVE_FIELD_BANK_ID, 14 }, // 3
        { SAVE_FIELD_BANK_ADDR, 0 }, // 4
    },
    map<uint32_t, uint32_t> {
        { SAVE_FIELD_JUMP_WRITE, 0 }, // 5
    },
    map<uint32_t, uint32_t> {
        { SAVE_FIELD_JUMP_READ, 24 }, // 6
        { SAVE_FIELD_LENGTH, 14 }, // 7
        { SAVE_FIELD_CHANNEL, 0 }, // 8
    },
    map<uint32_t, uint32_t> {
        { SAVE_FIELD_REG_ID, 29 }, // 9
        { SAVE_FIELD_DDR_ADDR, 0 }, // 10
    },
};


// ConvInit field position
const vector<map<uint32_t, uint32_t>> InstTable::ConvInitFieldPos = {
    map<uint32_t, uint32_t> {
        { CONVINIT_FIELD_OPCODE, 28 }, // 0
        { CONVINIT_FIELD_DPDON, 24 }, // 1
        { CONVINIT_FIELD_DPDBY, 20 }, // 2
        { CONVINIT_FIELD_ACT_TYPE, 16 }, // 3
        { CONVINIT_FIELD_SHIFT_CUT, 10 }, // 4
        { CONVINIT_FIELD_JUMP_READ, 0 }, // 5
    },
    map<uint32_t, uint32_t> {
        { CONVINIT_FIELD_STRIDE_OUT, 28 }, // 6
        { CONVINIT_FIELD_SHIFT_BIAS, 13 }, // 7
        { CONVINIT_FIELD_JUMP_READ_ENDL, 0 }, // 8
    },
    map<uint32_t, uint32_t> {
        { CONVINIT_FIELD_STRIDE_H, 28 }, // 9
        { CONVINIT_FIELD_KERNEL_H, 24 }, // 10
        { CONVINIT_FIELD_VALID_PIXEL_PARALLEL, 13 }, // 11
        { CONVINIT_FIELD_STRIDE_OFFSET_IN, 10 }, // 12
        { CONVINIT_FIELD_JUMP_WRITE, 0 }, // 13
    },
    map<uint32_t, uint32_t> {
        { CONVINIT_FIELD_STRIDE_W, 28 }, // 14
        { CONVINIT_FIELD_KERNEL_W, 24 }, // 15
        { CONVINIT_FIELD_STRIDE_OFFSET_OUT, 13 }, // 16
        { CONVINIT_FIELD_JUMP_WRITE_ENDL, 0 }, // 17
    },
};


// Conv field position
const vector<map<uint32_t, uint32_t>> InstTable::ConvFieldPos = {
    map<uint32_t, uint32_t> {
        { CONV_FIELD_OPCODE, 28 }, // 0
        { CONV_FIELD_DPDON, 24 }, // 1
        { CONV_FIELD_DPDBY, 20 }, // 2
        { CONV_FIELD_BANK_ID_IN, 14 }, // 3
        { CONV_FIELD_BANK_ADDR_IN, 0 }, // 4
    },
    map<uint32_t, uint32_t> {
        { CONV_FIELD_PAD_TOP, 28 }, // 5
        { CONV_FIELD_CHANNEL_GROUP, 20 }, // 6
        { CONV_FIELD_BANK_ID_OUT, 14 }, // 7
        { CONV_FIELD_BANK_ADDR_OUT, 0 }, // 8
    },
    map<uint32_t, uint32_t> {
        { CONV_FIELD_PAD_BOTTOM, 28 }, // 9
        { CONV_FIELD_CHANNEL_OFFSET, 24 }, // 10
        { CONV_FIELD_LENGTH, 14 }, // 11
        { CONV_FIELD_BANK_ADDR_IN_1, 0 }, // 12
    },
    map<uint32_t, uint32_t> {
        { CONV_FIELD_PAD_LEFT, 28 }, // 13
        { CONV_FIELD_BANK_ADDR_WEIGHTS, 14 }, // 14
        { CONV_FIELD_BANK_ADDR_IN_2, 0 }, // 15
    },
    map<uint32_t, uint32_t> {
        { CONV_FIELD_PAD_RIGHT, 28 }, // 16
        { CONV_FIELD_BANK_ADDR_BIAS, 14 }, // 17
        { CONV_FIELD_BANK_ADDR_IN_3, 0 }, // 18
    },
};


// PoolInit field position
const vector<map<uint32_t, uint32_t>> InstTable::PoolInitFieldPos = {
    map<uint32_t, uint32_t> {
        { POOLINIT_FIELD_OPCODE, 28 }, // 0
        { POOLINIT_FIELD_DPDON, 24 }, // 1
        { POOLINIT_FIELD_DPDBY, 20 }, // 2
        { POOLINIT_FIELD_SHIFT_CUT, 16 }, // 3
        { POOLINIT_FIELD_KERNEL_W, 13 }, // 4
        { POOLINIT_FIELD_KERNEL_H, 10 }, // 5
        { POOLINIT_FIELD_JUMP_READ, 0 }, // 6
    },
    map<uint32_t, uint32_t> {
        { POOLINIT_FIELD_JUMP_WRITE, 21 }, // 7
        { POOLINIT_FIELD_STRIDE_OUT, 17 }, // 8
        { POOLINIT_FIELD_STRIDE_OFFSET_OUT, 14 }, // 9
        { POOLINIT_FIELD_VALID_PIXEL_PARALLEL, 11 }, // 10
        { POOLINIT_FIELD_STRIDE_OFFSET_IN, 8 }, // 11
        { POOLINIT_FIELD_STRIDE_W, 5 }, // 12
        { POOLINIT_FIELD_STRIDE_H, 2 }, // 13
        { POOLINIT_FIELD_POOL_TYPE, 0 }, // 14
    },
};


// Pool field position
const vector<map<uint32_t, uint32_t>> InstTable::PoolFieldPos = {
    map<uint32_t, uint32_t> {
        { POOL_FIELD_OPCODE, 28 }, // 0
        { POOL_FIELD_DPDON, 24 }, // 1
        { POOL_FIELD_DPDBY, 20 }, // 2
        { POOL_FIELD_BANK_ID_IN, 14 }, // 3
        { POOL_FIELD_BANK_ADDR_IN, 0 }, // 4
    },
    map<uint32_t, uint32_t> {
        { POOL_FIELD_JUMP_READ_ENDL, 19 }, // 5
        { POOL_FIELD_BANK_ID_OUT, 13 }, // 6
        { POOL_FIELD_PAD_TOP, 9 }, // 7
        { POOL_FIELD_PAD_BOTTOM, 6 }, // 8
        { POOL_FIELD_PAD_LEFT, 3 }, // 9
        { POOL_FIELD_PAD_RIGHT, 0 }, // 10
    },
    map<uint32_t, uint32_t> {
        { POOL_FIELD_JUMP_WRITE_ENDL, 19 }, // 11
        { POOL_FIELD_CHANNEL_GROUP, 11 }, // 12
        { POOL_FIELD_LENGTH, 0 }, // 13
    },
    map<uint32_t, uint32_t> {
        { POOL_FIELD_BANK_ADDR_IN_1, 14 }, // 14
        { POOL_FIELD_BANK_ADDR_IN_2, 0 }, // 15
    },
    map<uint32_t, uint32_t> {
        { POOL_FIELD_BANK_ADDR_IN_3, 14 }, // 16
        { POOL_FIELD_BANK_ADDR_OUT, 0 }, // 17
    },
};


// DWInit field position
const vector<map<uint32_t, uint32_t>> InstTable::DWInitFieldPos = {
    map<uint32_t, uint32_t> {
        { DWINIT_FIELD_OPCODE, 28 }, // 0
        { DWINIT_FIELD_DPDON, 24 }, // 1
        { DWINIT_FIELD_DPDBY, 20 }, // 2
        { DWINIT_FIELD_STRIDE_OUT, 16 }, // 3
        { DWINIT_FIELD_ACT_TYPE, 13 }, // 4
        { DWINIT_FIELD_JUMP_READ_ENDL, 0 }, // 5
    },
    map<uint32_t, uint32_t> {
        { DWINIT_FIELD_KERNEL_W, 28 }, // 6
        { DWINIT_FIELD_KERNEL_H, 24 }, // 7
        { DWINIT_FIELD_STRIDE_W, 20 }, // 8
        { DWINIT_FIELD_STRIDE_H, 16 }, // 9
        { DWINIT_FIELD_VALID_PIXEL_PARALLEL, 13 }, // 10
        { DWINIT_FIELD_JUMP_WRITE_ENDL, 0 }, // 11
    },
    map<uint32_t, uint32_t> {
        { DWINIT_FIELD_STRIDE_OFFSET_IN, 25 }, // 12
        { DWINIT_FIELD_STRIDE_OFFSET_OUT, 22 }, // 13
        { DWINIT_FIELD_SHIFT_BIAS, 16 }, // 14
        { DWINIT_FIELD_SHIFT_CUT, 10 }, // 15
        { DWINIT_FIELD_JUMP_READ, 0 }, // 16
    },
    map<uint32_t, uint32_t> {
        { DWINIT_FIELD_JUMP_WRITE, 0 }, // 17
    },
};


// DptWise field position
const vector<map<uint32_t, uint32_t>> InstTable::DptWiseFieldPos = {
    map<uint32_t, uint32_t> {
        { DPTWISE_FIELD_OPCODE, 28 }, // 0
        { DPTWISE_FIELD_DPDON, 24 }, // 1
        { DPTWISE_FIELD_DPDBY, 20 }, // 2
        { DPTWISE_FIELD_BANK_ID_IN, 14 }, // 3
        { DPTWISE_FIELD_BANK_ADDR_IN, 0 }, // 4
    },
    map<uint32_t, uint32_t> {
        { DPTWISE_FIELD_PAD_TOP, 28 }, // 5
        { DPTWISE_FIELD_CHANNEL_GROUP, 20 }, // 6
        { DPTWISE_FIELD_BANK_ID_OUT, 14 }, // 7
        { DPTWISE_FIELD_BANK_ADDR_OUT, 0 }, // 8
    },
    map<uint32_t, uint32_t> {
        { DPTWISE_FIELD_PAD_BOTTOM, 28 }, // 9
        { DPTWISE_FIELD_CHANNEL_OFFSET, 24 }, // 10
        { DPTWISE_FIELD_LENGTH, 14 }, // 11
        { DPTWISE_FIELD_BANK_ADDR_IN_1, 0 }, // 12
    },
    map<uint32_t, uint32_t> {
        { DPTWISE_FIELD_PAD_LEFT, 28 }, // 13
        { DPTWISE_FIELD_BANK_ADDR_WEIGHTS, 14 }, // 14
        { DPTWISE_FIELD_BANK_ADDR_IN_2, 0 }, // 15
    },
    map<uint32_t, uint32_t> {
        { DPTWISE_FIELD_PAD_RIGHT, 28 }, // 16
        { DPTWISE_FIELD_BANK_ADDR_BIAS, 14 }, // 17
        { DPTWISE_FIELD_BANK_ADDR_IN_3, 0 }, // 18
    },
};


// ElewInit field position
const vector<map<uint32_t, uint32_t>> InstTable::ElewInitFieldPos = {
    map<uint32_t, uint32_t> {
        { ELEWINIT_FIELD_OPCODE, 28 }, // 0
        { ELEWINIT_FIELD_DPDON, 24 }, // 1
        { ELEWINIT_FIELD_DPDBY, 20 }, // 2
        { ELEWINIT_FIELD_BANK_ID_IN, 14 }, // 3
        { ELEWINIT_FIELD_SHIFT_READ, 10 }, // 4
        { ELEWINIT_FIELD_JUMP_READ, 0 }, // 5
    },
    map<uint32_t, uint32_t> {
        { ELEWINIT_FIELD_ID, 27 }, // 6
        { ELEWINIT_FIELD_JUMP_READ_ENDL, 14 }, // 7
        { ELEWINIT_FIELD_BANK_ADDR_IN, 0 }, // 8
    },
};


// Elew field position
const vector<map<uint32_t, uint32_t>> InstTable::ElewFieldPos = {
    map<uint32_t, uint32_t> {
        { ELEW_FIELD_OPCODE, 28 }, // 0
        { ELEW_FIELD_DPDON, 24 }, // 1
        { ELEW_FIELD_DPDBY, 20 }, // 2
        { ELEW_FIELD_BANK_ID_OUT, 14 }, // 3
        { ELEW_FIELD_SHIFT_WRITE, 10 }, // 4
        { ELEW_FIELD_JUMP_WRITE, 0 }, // 5
    },
    map<uint32_t, uint32_t> {
        { ELEW_FIELD_CHANNEL_GROUP, 24 }, // 6
        { ELEW_FIELD_LENGTH, 15 }, // 7
        { ELEW_FIELD_ACT_TYPE, 14 }, // 8
        { ELEW_FIELD_BANK_ADDR_OUT, 0 }, // 9
    },
    map<uint32_t, uint32_t> {
        { ELEW_FIELD_NUM, 16 }, // 10
        { ELEW_FIELD_VALID_PIXEL_PARALLEL, 13 }, // 11
        { ELEW_FIELD_JUMP_WRITE_ENDL, 0 }, // 12
    },
};


// End field position
const vector<map<uint32_t, uint32_t>> InstTable::EndFieldPos = {
    map<uint32_t, uint32_t> {
        { END_FIELD_OPCODE, 28 }, // 0
        { END_FIELD_DPDON, 24 }, // 1
        { END_FIELD_DPDBY, 20 }, // 2
    },
};


// DumpBank field position
const vector<map<uint32_t, uint32_t>> InstTable::DumpBankFieldPos = {
    map<uint32_t, uint32_t> {
        { DUMPBANK_FIELD_OPCODE, 28 }, // 0
        { DUMPBANK_FIELD_DPDON, 24 }, // 1
        { DUMPBANK_FIELD_DPDBY, 20 }, // 2
        { DUMPBANK_FIELD_SAVE_NAME, 0 }, // 3
    },
    map<uint32_t, uint32_t> {
        { DUMPBANK_FIELD_SAVE_FMT, 26 }, // 4
        { DUMPBANK_FIELD_BANK_START, 18 }, // 5
        { DUMPBANK_FIELD_BANK_NUM, 10 }, // 6
    },
};


// DumpDDR field position
const vector<map<uint32_t, uint32_t>> InstTable::DumpDDRFieldPos = {
    map<uint32_t, uint32_t> {
        { DUMPDDR_FIELD_OPCODE, 28 }, // 0
        { DUMPDDR_FIELD_DPDON, 24 }, // 1
        { DUMPDDR_FIELD_DPDBY, 20 }, // 2
        { DUMPDDR_FIELD_SAVE_NAME, 0 }, // 3
    },
    map<uint32_t, uint32_t> {
        { DUMPDDR_FIELD_SAVE_FMT, 26 }, // 4
        { DUMPDDR_FIELD_REG_ID, 20 }, // 5
    },
    map<uint32_t, uint32_t> {
        { DUMPDDR_FIELD_DDR_START, 0 }, // 6
    },
    map<uint32_t, uint32_t> {
        { DUMPDDR_FIELD_DDR_SIZE, 0 }, // 7
    },
};


// DumpDDRSlice field position
const vector<map<uint32_t, uint32_t>> InstTable::DumpDDRSliceFieldPos = {
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_OPCODE, 28 }, // 0
        { DUMPDDRSLICE_FIELD_DPDON, 24 }, // 1
        { DUMPDDRSLICE_FIELD_DPDBY, 20 }, // 2
        { DUMPDDRSLICE_FIELD_SAVE_NAME, 0 }, // 3
    },
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_SAVE_FMT, 26 }, // 4
        { DUMPDDRSLICE_FIELD_REG_ID, 20 }, // 5
    },
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_DDR_START, 0 }, // 6
    },
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_HEIGHT, 16 }, // 7
        { DUMPDDRSLICE_FIELD_HEIGHT_STRIDE, 0 }, // 8
    },
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_WIDTH, 16 }, // 9
        { DUMPDDRSLICE_FIELD_WIDTH_STRIDE, 0 }, // 10
    },
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_CHANNEL, 16 }, // 11
        { DUMPDDRSLICE_FIELD_CHANNEL_STRIDE, 0 }, // 12
    },
};


// Load field length
const vector<map<uint32_t, uint32_t>> InstTable::LoadFieldLen = {
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_OPCODE, 4 }, // 0
        { LOAD_FIELD_DPDON, 4 }, // 1
        { LOAD_FIELD_DPDBY, 4 }, // 2
        { LOAD_FIELD_BANK_ID, 6 }, // 3
        { LOAD_FIELD_BANK_ADDR, 13 }, // 4
    },
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_PAD_START, 5 }, // 5
        { LOAD_FIELD_PAD_END, 5 }, // 6
        { LOAD_FIELD_PAD_IDX, 5 }, // 7
        { LOAD_FIELD_JUMP_READ, 16 }, // 8
    },
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_JUMP_WRITE, 8 }, // 9
        { LOAD_FIELD_LENGTH, 10 }, // 10
        { LOAD_FIELD_MODE_AVG, 2 }, // 11
        { LOAD_FIELD_CHANNEL, 12 }, // 12
    },
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_REG_ID, 3 }, // 13
        { LOAD_FIELD_DDR_ADDR, 29 }, // 14
    },
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_DDR_MODE, 1 }, // 15
        { LOAD_FIELD_OUTPUT_CHANNEL_NUM, 7 }, // 16
        { LOAD_FIELD_JUMP_READ_ENDL, 16 }, // 17
    },
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_REG_ID_1, 3 }, // 18
        { LOAD_FIELD_DDR_ADDR_1, 29 }, // 19
    },
};


// Save field length
const vector<map<uint32_t, uint32_t>> InstTable::SaveFieldLen = {
    map<uint32_t, uint32_t> {
        { SAVE_FIELD_OPCODE, 4 }, // 0
        { SAVE_FIELD_DPDON, 4 }, // 1
        { SAVE_FIELD_DPDBY, 4 }, // 2
        { SAVE_FIELD_BANK_ID, 6 }, // 3
        { SAVE_FIELD_BANK_ADDR, 13 }, // 4
    },
    map<uint32_t, uint32_t> {
        { SAVE_FIELD_JUMP_WRITE, 16 }, // 5
    },
    map<uint32_t, uint32_t> {
        { SAVE_FIELD_JUMP_READ, 8 }, // 6
        { SAVE_FIELD_LENGTH, 10 }, // 7
        { SAVE_FIELD_CHANNEL, 12 }, // 8
    },
    map<uint32_t, uint32_t> {
        { SAVE_FIELD_REG_ID, 3 }, // 9
        { SAVE_FIELD_DDR_ADDR, 29 }, // 10
    },
};


// ConvInit field length
const vector<map<uint32_t, uint32_t>> InstTable::ConvInitFieldLen = {
    map<uint32_t, uint32_t> {
        { CONVINIT_FIELD_OPCODE, 4 }, // 0
        { CONVINIT_FIELD_DPDON, 4 }, // 1
        { CONVINIT_FIELD_DPDBY, 4 }, // 2
        { CONVINIT_FIELD_ACT_TYPE, 4 }, // 3
        { CONVINIT_FIELD_SHIFT_CUT, 6 }, // 4
        { CONVINIT_FIELD_JUMP_READ, 10 }, // 5
    },
    map<uint32_t, uint32_t> {
        { CONVINIT_FIELD_STRIDE_OUT, 4 }, // 6
        { CONVINIT_FIELD_SHIFT_BIAS, 6 }, // 7
        { CONVINIT_FIELD_JUMP_READ_ENDL, 13 }, // 8
    },
    map<uint32_t, uint32_t> {
        { CONVINIT_FIELD_STRIDE_H, 4 }, // 9
        { CONVINIT_FIELD_KERNEL_H, 4 }, // 10
        { CONVINIT_FIELD_VALID_PIXEL_PARALLEL, 3 }, // 11
        { CONVINIT_FIELD_STRIDE_OFFSET_IN, 3 }, // 12
        { CONVINIT_FIELD_JUMP_WRITE, 10 }, // 13
    },
    map<uint32_t, uint32_t> {
        { CONVINIT_FIELD_STRIDE_W, 4 }, // 14
        { CONVINIT_FIELD_KERNEL_W, 4 }, // 15
        { CONVINIT_FIELD_STRIDE_OFFSET_OUT, 3 }, // 16
        { CONVINIT_FIELD_JUMP_WRITE_ENDL, 13 }, // 17
    },
};


// Conv field length
const vector<map<uint32_t, uint32_t>> InstTable::ConvFieldLen = {
    map<uint32_t, uint32_t> {
        { CONV_FIELD_OPCODE, 4 }, // 0
        { CONV_FIELD_DPDON, 4 }, // 1
        { CONV_FIELD_DPDBY, 4 }, // 2
        { CONV_FIELD_BANK_ID_IN, 6 }, // 3
        { CONV_FIELD_BANK_ADDR_IN, 13 }, // 4
    },
    map<uint32_t, uint32_t> {
        { CONV_FIELD_PAD_TOP, 4 }, // 5
        { CONV_FIELD_CHANNEL_GROUP, 8 }, // 6
        { CONV_FIELD_BANK_ID_OUT, 6 }, // 7
        { CONV_FIELD_BANK_ADDR_OUT, 13 }, // 8
    },
    map<uint32_t, uint32_t> {
        { CONV_FIELD_PAD_BOTTOM, 4 }, // 9
        { CONV_FIELD_CHANNEL_OFFSET, 4 }, // 10
        { CONV_FIELD_LENGTH, 10 }, // 11
        { CONV_FIELD_BANK_ADDR_IN_1, 13 }, // 12
    },
    map<uint32_t, uint32_t> {
        { CONV_FIELD_PAD_LEFT, 4 }, // 13
        { CONV_FIELD_BANK_ADDR_WEIGHTS, 12 }, // 14
        { CONV_FIELD_BANK_ADDR_IN_2, 13 }, // 15
    },
    map<uint32_t, uint32_t> {
        { CONV_FIELD_PAD_RIGHT, 4 }, // 16
        { CONV_FIELD_BANK_ADDR_BIAS, 12 }, // 17
        { CONV_FIELD_BANK_ADDR_IN_3, 13 }, // 18
    },
};


// PoolInit field length
const vector<map<uint32_t, uint32_t>> InstTable::PoolInitFieldLen = {
    map<uint32_t, uint32_t> {
        { POOLINIT_FIELD_OPCODE, 4 }, // 0
        { POOLINIT_FIELD_DPDON, 4 }, // 1
        { POOLINIT_FIELD_DPDBY, 4 }, // 2
        { POOLINIT_FIELD_SHIFT_CUT, 4 }, // 3
        { POOLINIT_FIELD_KERNEL_W, 3 }, // 4
        { POOLINIT_FIELD_KERNEL_H, 3 }, // 5
        { POOLINIT_FIELD_JUMP_READ, 10 }, // 6
    },
    map<uint32_t, uint32_t> {
        { POOLINIT_FIELD_JUMP_WRITE, 10 }, // 7
        { POOLINIT_FIELD_STRIDE_OUT, 4 }, // 8
        { POOLINIT_FIELD_STRIDE_OFFSET_OUT, 3 }, // 9
        { POOLINIT_FIELD_VALID_PIXEL_PARALLEL, 3 }, // 10
        { POOLINIT_FIELD_STRIDE_OFFSET_IN, 3 }, // 11
        { POOLINIT_FIELD_STRIDE_W, 3 }, // 12
        { POOLINIT_FIELD_STRIDE_H, 3 }, // 13
        { POOLINIT_FIELD_POOL_TYPE, 2 }, // 14
    },
};


// Pool field length
const vector<map<uint32_t, uint32_t>> InstTable::PoolFieldLen = {
    map<uint32_t, uint32_t> {
        { POOL_FIELD_OPCODE, 4 }, // 0
        { POOL_FIELD_DPDON, 4 }, // 1
        { POOL_FIELD_DPDBY, 4 }, // 2
        { POOL_FIELD_BANK_ID_IN, 6 }, // 3
        { POOL_FIELD_BANK_ADDR_IN, 13 }, // 4
    },
    map<uint32_t, uint32_t> {
        { POOL_FIELD_JUMP_READ_ENDL, 13 }, // 5
        { POOL_FIELD_BANK_ID_OUT, 6 }, // 6
        { POOL_FIELD_PAD_TOP, 3 }, // 7
        { POOL_FIELD_PAD_BOTTOM, 3 }, // 8
        { POOL_FIELD_PAD_LEFT, 3 }, // 9
        { POOL_FIELD_PAD_RIGHT, 3 }, // 10
    },
    map<uint32_t, uint32_t> {
        { POOL_FIELD_JUMP_WRITE_ENDL, 13 }, // 11
        { POOL_FIELD_CHANNEL_GROUP, 8 }, // 12
        { POOL_FIELD_LENGTH, 10 }, // 13
    },
    map<uint32_t, uint32_t> {
        { POOL_FIELD_BANK_ADDR_IN_1, 13 }, // 14
        { POOL_FIELD_BANK_ADDR_IN_2, 13 }, // 15
    },
    map<uint32_t, uint32_t> {
        { POOL_FIELD_BANK_ADDR_IN_3, 13 }, // 16
        { POOL_FIELD_BANK_ADDR_OUT, 13 }, // 17
    },
};


// DWInit field length
const vector<map<uint32_t, uint32_t>> InstTable::DWInitFieldLen = {
    map<uint32_t, uint32_t> {
        { DWINIT_FIELD_OPCODE, 4 }, // 0
        { DWINIT_FIELD_DPDON, 4 }, // 1
        { DWINIT_FIELD_DPDBY, 4 }, // 2
        { DWINIT_FIELD_STRIDE_OUT, 4 }, // 3
        { DWINIT_FIELD_ACT_TYPE, 3 }, // 4
        { DWINIT_FIELD_JUMP_READ_ENDL, 13 }, // 5
    },
    map<uint32_t, uint32_t> {
        { DWINIT_FIELD_KERNEL_W, 4 }, // 6
        { DWINIT_FIELD_KERNEL_H, 4 }, // 7
        { DWINIT_FIELD_STRIDE_W, 4 }, // 8
        { DWINIT_FIELD_STRIDE_H, 4 }, // 9
        { DWINIT_FIELD_VALID_PIXEL_PARALLEL, 3 }, // 10
        { DWINIT_FIELD_JUMP_WRITE_ENDL, 13 }, // 11
    },
    map<uint32_t, uint32_t> {
        { DWINIT_FIELD_STRIDE_OFFSET_IN, 3 }, // 12
        { DWINIT_FIELD_STRIDE_OFFSET_OUT, 3 }, // 13
        { DWINIT_FIELD_SHIFT_BIAS, 6 }, // 14
        { DWINIT_FIELD_SHIFT_CUT, 6 }, // 15
        { DWINIT_FIELD_JUMP_READ, 10 }, // 16
    },
    map<uint32_t, uint32_t> {
        { DWINIT_FIELD_JUMP_WRITE, 10 }, // 17
    },
};


// DptWise field length
const vector<map<uint32_t, uint32_t>> InstTable::DptWiseFieldLen = {
    map<uint32_t, uint32_t> {
        { DPTWISE_FIELD_OPCODE, 4 }, // 0
        { DPTWISE_FIELD_DPDON, 4 }, // 1
        { DPTWISE_FIELD_DPDBY, 4 }, // 2
        { DPTWISE_FIELD_BANK_ID_IN, 6 }, // 3
        { DPTWISE_FIELD_BANK_ADDR_IN, 13 }, // 4
    },
    map<uint32_t, uint32_t> {
        { DPTWISE_FIELD_PAD_TOP, 4 }, // 5
        { DPTWISE_FIELD_CHANNEL_GROUP, 8 }, // 6
        { DPTWISE_FIELD_BANK_ID_OUT, 6 }, // 7
        { DPTWISE_FIELD_BANK_ADDR_OUT, 13 }, // 8
    },
    map<uint32_t, uint32_t> {
        { DPTWISE_FIELD_PAD_BOTTOM, 4 }, // 9
        { DPTWISE_FIELD_CHANNEL_OFFSET, 4 }, // 10
        { DPTWISE_FIELD_LENGTH, 10 }, // 11
        { DPTWISE_FIELD_BANK_ADDR_IN_1, 13 }, // 12
    },
    map<uint32_t, uint32_t> {
        { DPTWISE_FIELD_PAD_LEFT, 4 }, // 13
        { DPTWISE_FIELD_BANK_ADDR_WEIGHTS, 12 }, // 14
        { DPTWISE_FIELD_BANK_ADDR_IN_2, 13 }, // 15
    },
    map<uint32_t, uint32_t> {
        { DPTWISE_FIELD_PAD_RIGHT, 4 }, // 16
        { DPTWISE_FIELD_BANK_ADDR_BIAS, 12 }, // 17
        { DPTWISE_FIELD_BANK_ADDR_IN_3, 13 }, // 18
    },
};


// ElewInit field length
const vector<map<uint32_t, uint32_t>> InstTable::ElewInitFieldLen = {
    map<uint32_t, uint32_t> {
        { ELEWINIT_FIELD_OPCODE, 4 }, // 0
        { ELEWINIT_FIELD_DPDON, 4 }, // 1
        { ELEWINIT_FIELD_DPDBY, 4 }, // 2
        { ELEWINIT_FIELD_BANK_ID_IN, 6 }, // 3
        { ELEWINIT_FIELD_SHIFT_READ, 4 }, // 4
        { ELEWINIT_FIELD_JUMP_READ, 10 }, // 5
    },
    map<uint32_t, uint32_t> {
        { ELEWINIT_FIELD_ID, 2 }, // 6
        { ELEWINIT_FIELD_JUMP_READ_ENDL, 13 }, // 7
        { ELEWINIT_FIELD_BANK_ADDR_IN, 13 }, // 8
    },
};


// Elew field length
const vector<map<uint32_t, uint32_t>> InstTable::ElewFieldLen = {
    map<uint32_t, uint32_t> {
        { ELEW_FIELD_OPCODE, 4 }, // 0
        { ELEW_FIELD_DPDON, 4 }, // 1
        { ELEW_FIELD_DPDBY, 4 }, // 2
        { ELEW_FIELD_BANK_ID_OUT, 6 }, // 3
        { ELEW_FIELD_SHIFT_WRITE, 4 }, // 4
        { ELEW_FIELD_JUMP_WRITE, 10 }, // 5
    },
    map<uint32_t, uint32_t> {
        { ELEW_FIELD_CHANNEL_GROUP, 8 }, // 6
        { ELEW_FIELD_LENGTH, 9 }, // 7
        { ELEW_FIELD_ACT_TYPE, 1 }, // 8
        { ELEW_FIELD_BANK_ADDR_OUT, 13 }, // 9
    },
    map<uint32_t, uint32_t> {
        { ELEW_FIELD_NUM, 2 }, // 10
        { ELEW_FIELD_VALID_PIXEL_PARALLEL, 3 }, // 11
        { ELEW_FIELD_JUMP_WRITE_ENDL, 13 }, // 12
    },
};


// End field length
const vector<map<uint32_t, uint32_t>> InstTable::EndFieldLen = {
    map<uint32_t, uint32_t> {
        { END_FIELD_OPCODE, 4 }, // 0
        { END_FIELD_DPDON, 4 }, // 1
        { END_FIELD_DPDBY, 4 }, // 2
    },
};


// DumpBank field length
const vector<map<uint32_t, uint32_t>> InstTable::DumpBankFieldLen = {
    map<uint32_t, uint32_t> {
        { DUMPBANK_FIELD_OPCODE, 4 }, // 0
        { DUMPBANK_FIELD_DPDON, 4 }, // 1
        { DUMPBANK_FIELD_DPDBY, 4 }, // 2
        { DUMPBANK_FIELD_SAVE_NAME, 20 }, // 3
    },
    map<uint32_t, uint32_t> {
        { DUMPBANK_FIELD_SAVE_FMT, 6 }, // 4
        { DUMPBANK_FIELD_BANK_START, 8 }, // 5
        { DUMPBANK_FIELD_BANK_NUM, 8 }, // 6
    },
};


// DumpDDR field length
const vector<map<uint32_t, uint32_t>> InstTable::DumpDDRFieldLen = {
    map<uint32_t, uint32_t> {
        { DUMPDDR_FIELD_OPCODE, 4 }, // 0
        { DUMPDDR_FIELD_DPDON, 4 }, // 1
        { DUMPDDR_FIELD_DPDBY, 4 }, // 2
        { DUMPDDR_FIELD_SAVE_NAME, 20 }, // 3
    },
    map<uint32_t, uint32_t> {
        { DUMPDDR_FIELD_SAVE_FMT, 6 }, // 4
        { DUMPDDR_FIELD_REG_ID, 6 }, // 5
    },
    map<uint32_t, uint32_t> {
        { DUMPDDR_FIELD_DDR_START, 32 }, // 6
    },
    map<uint32_t, uint32_t> {
        { DUMPDDR_FIELD_DDR_SIZE, 32 }, // 7
    },
};


// DumpDDRSlice field length
const vector<map<uint32_t, uint32_t>> InstTable::DumpDDRSliceFieldLen = {
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_OPCODE, 4 }, // 0
        { DUMPDDRSLICE_FIELD_DPDON, 4 }, // 1
        { DUMPDDRSLICE_FIELD_DPDBY, 4 }, // 2
        { DUMPDDRSLICE_FIELD_SAVE_NAME, 20 }, // 3
    },
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_SAVE_FMT, 6 }, // 4
        { DUMPDDRSLICE_FIELD_REG_ID, 6 }, // 5
    },
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_DDR_START, 32 }, // 6
    },
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_HEIGHT, 16 }, // 7
        { DUMPDDRSLICE_FIELD_HEIGHT_STRIDE, 16 }, // 8
    },
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_WIDTH, 16 }, // 9
        { DUMPDDRSLICE_FIELD_WIDTH_STRIDE, 16 }, // 10
    },
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_CHANNEL, 16 }, // 11
        { DUMPDDRSLICE_FIELD_CHANNEL_STRIDE, 16 }, // 12
    },
};


// inst opcode to inst type map
const unordered_map<uint32_t, uint32_t> InstTable::OPCode2InstType = {
    { 0x00, INST_TYPE_LOAD }, // 0
    { 0x04, INST_TYPE_SAVE }, // 1
    { 0x09, INST_TYPE_CONVINIT }, // 2
    { 0x08, INST_TYPE_CONV }, // 3
    { 0x06, INST_TYPE_POOLINIT }, // 4
    { 0x0C, INST_TYPE_POOL }, // 5
    { 0x0B, INST_TYPE_DWINIT }, // 6
    { 0x0A, INST_TYPE_DPTWISE }, // 7
    { 0x0D, INST_TYPE_ELEWINIT }, // 8
    { 0x0E, INST_TYPE_ELEW }, // 9
    { 0x07, INST_TYPE_END }, // 10
    { 0xFF, INST_TYPE_DUMPBANK }, // 11
    { 0xFE, INST_TYPE_DUMPDDR }, // 12
    { 0xFD, INST_TYPE_DUMPDDRSLICE }, // 13
};


// inst name to inst type map
const unordered_map<string, uint32_t> InstTable::InstName2InstType = {
    { "LOAD", INST_TYPE_LOAD }, // 0
    { "SAVE", INST_TYPE_SAVE }, // 1
    { "CONVINIT", INST_TYPE_CONVINIT }, // 2
    { "CONV", INST_TYPE_CONV }, // 3
    { "POOLINIT", INST_TYPE_POOLINIT }, // 4
    { "POOL", INST_TYPE_POOL }, // 5
    { "DWINIT", INST_TYPE_DWINIT }, // 6
    { "DPTWISE", INST_TYPE_DPTWISE }, // 7
    { "ELEWINIT", INST_TYPE_ELEWINIT }, // 8
    { "ELEW", INST_TYPE_ELEW }, // 9
    { "END", INST_TYPE_END }, // 10
    { "DUMPBANK", INST_TYPE_DUMPBANK }, // 11
    { "DUMPDDR", INST_TYPE_DUMPDDR }, // 12
    { "DUMPDDRSLICE", INST_TYPE_DUMPDDRSLICE }, // 13
};


// inst type to field name map
const unordered_map<uint32_t, const vector<string> &> InstTable::InstFieldName = {
    { INST_TYPE_LOAD, LoadFieldName }, // 0
    { INST_TYPE_SAVE, SaveFieldName }, // 1
    { INST_TYPE_CONVINIT, ConvInitFieldName }, // 2
    { INST_TYPE_CONV, ConvFieldName }, // 3
    { INST_TYPE_POOLINIT, PoolInitFieldName }, // 4
    { INST_TYPE_POOL, PoolFieldName }, // 5
    { INST_TYPE_DWINIT, DWInitFieldName }, // 6
    { INST_TYPE_DPTWISE, DptWiseFieldName }, // 7
    { INST_TYPE_ELEWINIT, ElewInitFieldName }, // 8
    { INST_TYPE_ELEW, ElewFieldName }, // 9
    { INST_TYPE_END, EndFieldName }, // 10
    { INST_TYPE_DUMPBANK, DumpBankFieldName }, // 11
    { INST_TYPE_DUMPDDR, DumpDDRFieldName }, // 12
    { INST_TYPE_DUMPDDRSLICE, DumpDDRSliceFieldName }, // 13
};


// inst type to field data type map
const unordered_map<uint32_t, const vector<int> &> InstTable::InstFieldDataType = {
    { INST_TYPE_LOAD, LoadFieldDataType }, // 0
    { INST_TYPE_SAVE, SaveFieldDataType }, // 1
    { INST_TYPE_CONVINIT, ConvInitFieldDataType }, // 2
    { INST_TYPE_CONV, ConvFieldDataType }, // 3
    { INST_TYPE_POOLINIT, PoolInitFieldDataType }, // 4
    { INST_TYPE_POOL, PoolFieldDataType }, // 5
    { INST_TYPE_DWINIT, DWInitFieldDataType }, // 6
    { INST_TYPE_DPTWISE, DptWiseFieldDataType }, // 7
    { INST_TYPE_ELEWINIT, ElewInitFieldDataType }, // 8
    { INST_TYPE_ELEW, ElewFieldDataType }, // 9
    { INST_TYPE_END, EndFieldDataType }, // 10
    { INST_TYPE_DUMPBANK, DumpBankFieldDataType }, // 11
    { INST_TYPE_DUMPDDR, DumpDDRFieldDataType }, // 12
    { INST_TYPE_DUMPDDRSLICE, DumpDDRSliceFieldDataType }, // 13
};


// inst type to field minus map
const unordered_map<uint32_t, const vector<int> &> InstTable::InstFieldMinus = {
    { INST_TYPE_LOAD, LoadFieldMinus }, // 0
    { INST_TYPE_SAVE, SaveFieldMinus }, // 1
    { INST_TYPE_CONVINIT, ConvInitFieldMinus }, // 2
    { INST_TYPE_CONV, ConvFieldMinus }, // 3
    { INST_TYPE_POOLINIT, PoolInitFieldMinus }, // 4
    { INST_TYPE_POOL, PoolFieldMinus }, // 5
    { INST_TYPE_DWINIT, DWInitFieldMinus }, // 6
    { INST_TYPE_DPTWISE, DptWiseFieldMinus }, // 7
    { INST_TYPE_ELEWINIT, ElewInitFieldMinus }, // 8
    { INST_TYPE_ELEW, ElewFieldMinus }, // 9
    { INST_TYPE_END, EndFieldMinus }, // 10
    { INST_TYPE_DUMPBANK, DumpBankFieldMinus }, // 11
    { INST_TYPE_DUMPDDR, DumpDDRFieldMinus }, // 12
    { INST_TYPE_DUMPDDRSLICE, DumpDDRSliceFieldMinus }, // 13
};


// inst type to field position map
const unordered_map<uint32_t, const vector<map<uint32_t, uint32_t>> &> InstTable::InstFieldPos = {
    { INST_TYPE_LOAD, LoadFieldPos }, // 0
    { INST_TYPE_SAVE, SaveFieldPos }, // 1
    { INST_TYPE_CONVINIT, ConvInitFieldPos }, // 2
    { INST_TYPE_CONV, ConvFieldPos }, // 3
    { INST_TYPE_POOLINIT, PoolInitFieldPos }, // 4
    { INST_TYPE_POOL, PoolFieldPos }, // 5
    { INST_TYPE_DWINIT, DWInitFieldPos }, // 6
    { INST_TYPE_DPTWISE, DptWiseFieldPos }, // 7
    { INST_TYPE_ELEWINIT, ElewInitFieldPos }, // 8
    { INST_TYPE_ELEW, ElewFieldPos }, // 9
    { INST_TYPE_END, EndFieldPos }, // 10
    { INST_TYPE_DUMPBANK, DumpBankFieldPos }, // 11
    { INST_TYPE_DUMPDDR, DumpDDRFieldPos }, // 12
    { INST_TYPE_DUMPDDRSLICE, DumpDDRSliceFieldPos }, // 13
};


// inst type to field length map
const unordered_map<uint32_t, const vector<map<uint32_t, uint32_t>> &> InstTable::InstFieldLen = {
    { INST_TYPE_LOAD, LoadFieldLen }, // 0
    { INST_TYPE_SAVE, SaveFieldLen }, // 1
    { INST_TYPE_CONVINIT, ConvInitFieldLen }, // 2
    { INST_TYPE_CONV, ConvFieldLen }, // 3
    { INST_TYPE_POOLINIT, PoolInitFieldLen }, // 4
    { INST_TYPE_POOL, PoolFieldLen }, // 5
    { INST_TYPE_DWINIT, DWInitFieldLen }, // 6
    { INST_TYPE_DPTWISE, DptWiseFieldLen }, // 7
    { INST_TYPE_ELEWINIT, ElewInitFieldLen }, // 8
    { INST_TYPE_ELEW, ElewFieldLen }, // 9
    { INST_TYPE_END, EndFieldLen }, // 10
    { INST_TYPE_DUMPBANK, DumpBankFieldLen }, // 11
    { INST_TYPE_DUMPDDR, DumpDDRFieldLen }, // 12
    { INST_TYPE_DUMPDDRSLICE, DumpDDRSliceFieldLen }, // 13
};


}

namespace DPUV_3ME {
// instruction version
const string InstTable::INST_VERSION = "DPUV3ME";


// category name vector
const vector<string> InstTable::CategoryName = {
    "LOAD",
    "SAVE",
    "CONV",
    "MISC",
};


// inst name vector
const vector<string> InstTable::InstName = {
    "LOAD", // 0
    "SAVE", // 1
    "CBLOAD", // 2
    "CONVINIT", // 3
    "CONV", // 4
    "POOLINIT", // 5
    "POOL", // 6
    "DWINIT", // 7
    "DPTWISE", // 8
    "ELEWINIT", // 9
    "ELEW", // 10
    "END", // 11
    "DUMPBANK", // 12
    "DUMPDDR", // 13
    "DUMPDDRSLICE", // 14
};


// lower inst name vector
const vector<string> InstTable::InstNameLower = {
    "load", // 0
    "save", // 1
    "cbload", // 2
    "convinit", // 3
    "conv", // 4
    "poolinit", // 5
    "pool", // 6
    "dwinit", // 7
    "dptwise", // 8
    "elewinit", // 9
    "elew", // 10
    "end", // 11
    "dumpbank", // 12
    "dumpddr", // 13
    "dumpddrslice", // 14
};


// inst category
const vector<Category> InstTable::InstCategory = {
    Category::INST_CATEGORY_LOAD, // LOAD
    Category::INST_CATEGORY_SAVE, // SAVE
    Category::INST_CATEGORY_CONV, // CBLOAD
    Category::INST_CATEGORY_CONV, // CONVINIT
    Category::INST_CATEGORY_CONV, // CONV
    Category::INST_CATEGORY_MISC, // POOLINIT
    Category::INST_CATEGORY_MISC, // POOL
    Category::INST_CATEGORY_MISC, // DWINIT
    Category::INST_CATEGORY_MISC, // DPTWISE
    Category::INST_CATEGORY_MISC, // ELEWINIT
    Category::INST_CATEGORY_MISC, // ELEW
    Category::INST_CATEGORY_SAVE, // END
    Category::INST_CATEGORY_MISC, // DUMPBANK
    Category::INST_CATEGORY_MISC, // DUMPDDR
    Category::INST_CATEGORY_MISC, // DUMPDDRSLICE
};


// inst word number
const vector<uint32_t> InstTable::WordNum = {
    7, // LOAD
    4, // SAVE
    3, // CBLOAD
    5, // CONVINIT
    7, // CONV
    2, // POOLINIT
    5, // POOL
    4, // DWINIT
    6, // DPTWISE
    3, // ELEWINIT
    4, // ELEW
    1, // END
    2, // DUMPBANK
    4, // DUMPDDR
    6, // DUMPDDRSLICE
};


// inst field number
const vector<uint32_t> InstTable::FieldNum = {
    LOAD_FIELD_MAX, // 0
    SAVE_FIELD_MAX, // 1
    CBLOAD_FIELD_MAX, // 2
    CONVINIT_FIELD_MAX, // 3
    CONV_FIELD_MAX, // 4
    POOLINIT_FIELD_MAX, // 5
    POOL_FIELD_MAX, // 6
    DWINIT_FIELD_MAX, // 7
    DPTWISE_FIELD_MAX, // 8
    ELEWINIT_FIELD_MAX, // 9
    ELEW_FIELD_MAX, // 10
    END_FIELD_MAX, // 11
    DUMPBANK_FIELD_MAX, // 12
    DUMPDDR_FIELD_MAX, // 13
    DUMPDDRSLICE_FIELD_MAX, // 14
};


// inst opcode
const vector<uint32_t> InstTable::OPCode = {
    0x00, // LOAD
    0x04, // SAVE
    0x05, // CBLOAD
    0x09, // CONVINIT
    0x08, // CONV
    0x06, // POOLINIT
    0x0C, // POOL
    0x0B, // DWINIT
    0x0A, // DPTWISE
    0x0D, // ELEWINIT
    0x0E, // ELEW
    0x07, // END
    0xFF, // DUMPBANK
    0xFE, // DUMPDDR
    0xFD, // DUMPDDRSLICE
};


// Load field name
const vector<string> InstTable::LoadFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "mode", // 3
    "bank_id", // 4
    "length", // 5
    "jump_write", // 6
    "channel", // 7
    "jump_read", // 8
    "channel_offset", // 9
    "pad_bottom", // 10
    "pad_top", // 11
    "mode_avg", // 12
    "pad_start", // 13
    "pad_end", // 14
    "pad_idx", // 15
    "bank_addr", // 16
    "reg_id", // 17
    "ddr_addr", // 18
    "reg_id_1", // 19
    "ddr_addr_1", // 20
    "reg_id_2", // 21
    "ddr_addr_2", // 22
};


// Save field name
const vector<string> InstTable::SaveFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "hp_id", // 3
    "bank_id", // 4
    "length", // 5
    "jump_read", // 6
    "channel", // 7
    "jump_write", // 8
    "bank_addr", // 9
    "reg_id", // 10
    "ddr_addr", // 11
};


// CBLoad field name
const vector<string> InstTable::CBLoadFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "channel_group", // 3
    "length", // 4
    "cb_sel", // 5
    "group_id_in", // 6
    "bank_addr_in", // 7
    "bank_addr_out", // 8
};


// ConvInit field name
const vector<string> InstTable::ConvInitFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "act_type", // 3
    "shift_cut", // 4
    "jump_read", // 5
    "stride_h", // 6
    "kernel_h", // 7
    "stride_offset_out_hbm", // 8
    "valid_pixel_parallel", // 9
    "stride_offset_in", // 10
    "jump_write_cb", // 11
    "stride_w", // 12
    "kernel_w", // 13
    "stride_offset_out_cb", // 14
    "jump_read_endl", // 15
    "stride_out_cb", // 16
    "shift_relusix", // 17
    "shift_bias", // 18
    "jump_write_endl_cb", // 19
    "stride_out_hbm", // 20
    "jump_write_hbm", // 21
    "jump_write_endl_hbm", // 22
};


// Conv field name
const vector<string> InstTable::ConvFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "channel_group", // 3
    "length", // 4
    "output_channel_offset", // 5
    "channel_offset", // 6
    "bank_id_in", // 7
    "bank_id_out_cb", // 8
    "pad_left", // 9
    "pad_top", // 10
    "bank_addr_in", // 11
    "pad_right", // 12
    "pad_bottom", // 13
    "bank_addr_out_cb", // 14
    "bank_addr_weights", // 15
    "bank_addr_bias", // 16
    "dest_mode", // 17
    "bank_id_out_hbm", // 18
    "bank_addr_out_hbm", // 19
};


// PoolInit field name
const vector<string> InstTable::PoolInitFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "shift_cut", // 3
    "kernel_w", // 4
    "kernel_h", // 5
    "jump_read", // 6
    "jump_write", // 7
    "stride_out", // 8
    "stride_offset_out", // 9
    "valid_pixel_parallel", // 10
    "stride_offset_in", // 11
    "stride_w", // 12
    "stride_h", // 13
    "pool_type", // 14
};


// Pool field name
const vector<string> InstTable::PoolFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "channel_group", // 3
    "pad_left", // 4
    "pad_top", // 5
    "pad_right", // 6
    "pad_bottom", // 7
    "bank_id_in", // 8
    "bank_id_out", // 9
    "jump_read_endl", // 10
    "length", // 11
    "jump_write_endl", // 12
    "bank_addr_in", // 13
    "bank_addr_out", // 14
};


// DWInit field name
const vector<string> InstTable::DWInitFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "act_type", // 3
    "shift_cut", // 4
    "jump_read", // 5
    "stride_h", // 6
    "kernel_h", // 7
    "valid_pixel_parallel", // 8
    "stride_offset_in", // 9
    "jump_write", // 10
    "stride_w", // 11
    "kernel_w", // 12
    "stride_offset_out", // 13
    "jump_read_endl", // 14
    "stride_out", // 15
    "shift_relusix", // 16
    "shift_bias", // 17
    "jump_write_endl", // 18
};


// DptWise field name
const vector<string> InstTable::DptWiseFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "channel_group", // 3
    "length", // 4
    "channel_offset", // 5
    "bank_id_in", // 6
    "bank_id_out", // 7
    "pad_left", // 8
    "pad_top", // 9
    "bank_addr_in", // 10
    "pad_right", // 11
    "pad_bottom", // 12
    "bank_addr_out", // 13
    "bank_addr_weights", // 14
    "bank_addr_bias", // 15
};


// ElewInit field name
const vector<string> InstTable::ElewInitFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "bank_id_in", // 3
    "shift_read", // 4
    "jump_read", // 5
    "id", // 6
    "jump_read_endl", // 7
    "bank_addr_in", // 8
};


// Elew field name
const vector<string> InstTable::ElewFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "bank_id_out", // 3
    "shift_write", // 4
    "jump_write", // 5
    "num", // 6
    "channel_group", // 7
    "act_type", // 8
    "valid_pixel_parallel", // 9
    "length", // 10
    "jump_write_endl", // 11
    "bank_addr_out", // 12
};


// End field name
const vector<string> InstTable::EndFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
};


// DumpBank field name
const vector<string> InstTable::DumpBankFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "save_name", // 3
    "save_fmt", // 4
    "bank_start", // 5
    "bank_num", // 6
};


// DumpDDR field name
const vector<string> InstTable::DumpDDRFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "save_name", // 3
    "save_fmt", // 4
    "reg_id", // 5
    "ddr_start", // 6
    "ddr_size", // 7
};


// DumpDDRSlice field name
const vector<string> InstTable::DumpDDRSliceFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "save_name", // 3
    "save_fmt", // 4
    "reg_id", // 5
    "ddr_start", // 6
    "height", // 7
    "height_stride", // 8
    "width", // 9
    "width_stride", // 10
    "channel", // 11
    "channel_stride", // 12
};


// Load field data type
const vector<int> InstTable::LoadFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_UINT, // mode(3)
    INST_FIELD_DATA_TYPE_UINT, // bank_id(4)
    INST_FIELD_DATA_TYPE_UINT, // length(5)
    INST_FIELD_DATA_TYPE_UINT, // jump_write(6)
    INST_FIELD_DATA_TYPE_UINT, // channel(7)
    INST_FIELD_DATA_TYPE_UINT, // jump_read(8)
    INST_FIELD_DATA_TYPE_UINT, // channel_offset(9)
    INST_FIELD_DATA_TYPE_UINT, // pad_bottom(10)
    INST_FIELD_DATA_TYPE_UINT, // pad_top(11)
    INST_FIELD_DATA_TYPE_UINT, // mode_avg(12)
    INST_FIELD_DATA_TYPE_UINT, // pad_start(13)
    INST_FIELD_DATA_TYPE_UINT, // pad_end(14)
    INST_FIELD_DATA_TYPE_UINT, // pad_idx(15)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr(16)
    INST_FIELD_DATA_TYPE_UINT, // reg_id(17)
    INST_FIELD_DATA_TYPE_UINT, // ddr_addr(18)
    INST_FIELD_DATA_TYPE_UINT, // reg_id_1(19)
    INST_FIELD_DATA_TYPE_UINT, // ddr_addr_1(20)
    INST_FIELD_DATA_TYPE_UINT, // reg_id_2(21)
    INST_FIELD_DATA_TYPE_UINT, // ddr_addr_2(22)
};


// Save field data type
const vector<int> InstTable::SaveFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_UINT, // hp_id(3)
    INST_FIELD_DATA_TYPE_UINT, // bank_id(4)
    INST_FIELD_DATA_TYPE_UINT, // length(5)
    INST_FIELD_DATA_TYPE_UINT, // jump_read(6)
    INST_FIELD_DATA_TYPE_UINT, // channel(7)
    INST_FIELD_DATA_TYPE_UINT, // jump_write(8)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr(9)
    INST_FIELD_DATA_TYPE_UINT, // reg_id(10)
    INST_FIELD_DATA_TYPE_UINT, // ddr_addr(11)
};


// CBLoad field data type
const vector<int> InstTable::CBLoadFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_UINT, // channel_group(3)
    INST_FIELD_DATA_TYPE_UINT, // length(4)
    INST_FIELD_DATA_TYPE_UINT, // cb_sel(5)
    INST_FIELD_DATA_TYPE_UINT, // group_id_in(6)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in(7)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_out(8)
};


// ConvInit field data type
const vector<int> InstTable::ConvInitFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_UINT, // act_type(3)
    INST_FIELD_DATA_TYPE_UINT, // shift_cut(4)
    INST_FIELD_DATA_TYPE_UINT, // jump_read(5)
    INST_FIELD_DATA_TYPE_UINT, // stride_h(6)
    INST_FIELD_DATA_TYPE_UINT, // kernel_h(7)
    INST_FIELD_DATA_TYPE_UINT, // stride_offset_out_hbm(8)
    INST_FIELD_DATA_TYPE_UINT, // valid_pixel_parallel(9)
    INST_FIELD_DATA_TYPE_UINT, // stride_offset_in(10)
    INST_FIELD_DATA_TYPE_UINT, // jump_write_cb(11)
    INST_FIELD_DATA_TYPE_UINT, // stride_w(12)
    INST_FIELD_DATA_TYPE_UINT, // kernel_w(13)
    INST_FIELD_DATA_TYPE_UINT, // stride_offset_out_cb(14)
    INST_FIELD_DATA_TYPE_UINT, // jump_read_endl(15)
    INST_FIELD_DATA_TYPE_UINT, // stride_out_cb(16)
    INST_FIELD_DATA_TYPE_UINT, // shift_relusix(17)
    INST_FIELD_DATA_TYPE_UINT, // shift_bias(18)
    INST_FIELD_DATA_TYPE_UINT, // jump_write_endl_cb(19)
    INST_FIELD_DATA_TYPE_UINT, // stride_out_hbm(20)
    INST_FIELD_DATA_TYPE_UINT, // jump_write_hbm(21)
    INST_FIELD_DATA_TYPE_UINT, // jump_write_endl_hbm(22)
};


// Conv field data type
const vector<int> InstTable::ConvFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_UINT, // channel_group(3)
    INST_FIELD_DATA_TYPE_UINT, // length(4)
    INST_FIELD_DATA_TYPE_UINT, // output_channel_offset(5)
    INST_FIELD_DATA_TYPE_UINT, // channel_offset(6)
    INST_FIELD_DATA_TYPE_UINT, // bank_id_in(7)
    INST_FIELD_DATA_TYPE_UINT, // bank_id_out_cb(8)
    INST_FIELD_DATA_TYPE_UINT, // pad_left(9)
    INST_FIELD_DATA_TYPE_UINT, // pad_top(10)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in(11)
    INST_FIELD_DATA_TYPE_UINT, // pad_right(12)
    INST_FIELD_DATA_TYPE_UINT, // pad_bottom(13)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_out_cb(14)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_weights(15)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_bias(16)
    INST_FIELD_DATA_TYPE_UINT, // dest_mode(17)
    INST_FIELD_DATA_TYPE_UINT, // bank_id_out_hbm(18)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_out_hbm(19)
};


// PoolInit field data type
const vector<int> InstTable::PoolInitFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_UINT, // shift_cut(3)
    INST_FIELD_DATA_TYPE_UINT, // kernel_w(4)
    INST_FIELD_DATA_TYPE_UINT, // kernel_h(5)
    INST_FIELD_DATA_TYPE_UINT, // jump_read(6)
    INST_FIELD_DATA_TYPE_UINT, // jump_write(7)
    INST_FIELD_DATA_TYPE_UINT, // stride_out(8)
    INST_FIELD_DATA_TYPE_UINT, // stride_offset_out(9)
    INST_FIELD_DATA_TYPE_UINT, // valid_pixel_parallel(10)
    INST_FIELD_DATA_TYPE_UINT, // stride_offset_in(11)
    INST_FIELD_DATA_TYPE_UINT, // stride_w(12)
    INST_FIELD_DATA_TYPE_UINT, // stride_h(13)
    INST_FIELD_DATA_TYPE_UINT, // pool_type(14)
};


// Pool field data type
const vector<int> InstTable::PoolFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_UINT, // channel_group(3)
    INST_FIELD_DATA_TYPE_UINT, // pad_left(4)
    INST_FIELD_DATA_TYPE_UINT, // pad_top(5)
    INST_FIELD_DATA_TYPE_UINT, // pad_right(6)
    INST_FIELD_DATA_TYPE_UINT, // pad_bottom(7)
    INST_FIELD_DATA_TYPE_UINT, // bank_id_in(8)
    INST_FIELD_DATA_TYPE_UINT, // bank_id_out(9)
    INST_FIELD_DATA_TYPE_UINT, // jump_read_endl(10)
    INST_FIELD_DATA_TYPE_UINT, // length(11)
    INST_FIELD_DATA_TYPE_UINT, // jump_write_endl(12)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in(13)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_out(14)
};


// DWInit field data type
const vector<int> InstTable::DWInitFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_UINT, // act_type(3)
    INST_FIELD_DATA_TYPE_UINT, // shift_cut(4)
    INST_FIELD_DATA_TYPE_UINT, // jump_read(5)
    INST_FIELD_DATA_TYPE_UINT, // stride_h(6)
    INST_FIELD_DATA_TYPE_UINT, // kernel_h(7)
    INST_FIELD_DATA_TYPE_UINT, // valid_pixel_parallel(8)
    INST_FIELD_DATA_TYPE_UINT, // stride_offset_in(9)
    INST_FIELD_DATA_TYPE_UINT, // jump_write(10)
    INST_FIELD_DATA_TYPE_UINT, // stride_w(11)
    INST_FIELD_DATA_TYPE_UINT, // kernel_w(12)
    INST_FIELD_DATA_TYPE_UINT, // stride_offset_out(13)
    INST_FIELD_DATA_TYPE_UINT, // jump_read_endl(14)
    INST_FIELD_DATA_TYPE_UINT, // stride_out(15)
    INST_FIELD_DATA_TYPE_UINT, // shift_relusix(16)
    INST_FIELD_DATA_TYPE_UINT, // shift_bias(17)
    INST_FIELD_DATA_TYPE_UINT, // jump_write_endl(18)
};


// DptWise field data type
const vector<int> InstTable::DptWiseFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_UINT, // channel_group(3)
    INST_FIELD_DATA_TYPE_UINT, // length(4)
    INST_FIELD_DATA_TYPE_UINT, // channel_offset(5)
    INST_FIELD_DATA_TYPE_UINT, // bank_id_in(6)
    INST_FIELD_DATA_TYPE_UINT, // bank_id_out(7)
    INST_FIELD_DATA_TYPE_UINT, // pad_left(8)
    INST_FIELD_DATA_TYPE_UINT, // pad_top(9)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in(10)
    INST_FIELD_DATA_TYPE_UINT, // pad_right(11)
    INST_FIELD_DATA_TYPE_UINT, // pad_bottom(12)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_out(13)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_weights(14)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_bias(15)
};


// ElewInit field data type
const vector<int> InstTable::ElewInitFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_UINT, // bank_id_in(3)
    INST_FIELD_DATA_TYPE_UINT, // shift_read(4)
    INST_FIELD_DATA_TYPE_UINT, // jump_read(5)
    INST_FIELD_DATA_TYPE_UINT, // id(6)
    INST_FIELD_DATA_TYPE_UINT, // jump_read_endl(7)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in(8)
};


// Elew field data type
const vector<int> InstTable::ElewFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_UINT, // bank_id_out(3)
    INST_FIELD_DATA_TYPE_UINT, // shift_write(4)
    INST_FIELD_DATA_TYPE_UINT, // jump_write(5)
    INST_FIELD_DATA_TYPE_UINT, // num(6)
    INST_FIELD_DATA_TYPE_UINT, // channel_group(7)
    INST_FIELD_DATA_TYPE_UINT, // act_type(8)
    INST_FIELD_DATA_TYPE_UINT, // valid_pixel_parallel(9)
    INST_FIELD_DATA_TYPE_UINT, // length(10)
    INST_FIELD_DATA_TYPE_UINT, // jump_write_endl(11)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_out(12)
};


// End field data type
const vector<int> InstTable::EndFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
};


// DumpBank field data type
const vector<int> InstTable::DumpBankFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_STRING, // save_name(3)
    INST_FIELD_DATA_TYPE_UINT, // save_fmt(4)
    INST_FIELD_DATA_TYPE_UINT, // bank_start(5)
    INST_FIELD_DATA_TYPE_UINT, // bank_num(6)
};


// DumpDDR field data type
const vector<int> InstTable::DumpDDRFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_STRING, // save_name(3)
    INST_FIELD_DATA_TYPE_UINT, // save_fmt(4)
    INST_FIELD_DATA_TYPE_UINT, // reg_id(5)
    INST_FIELD_DATA_TYPE_UINT, // ddr_start(6)
    INST_FIELD_DATA_TYPE_UINT, // ddr_size(7)
};


// DumpDDRSlice field data type
const vector<int> InstTable::DumpDDRSliceFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_STRING, // save_name(3)
    INST_FIELD_DATA_TYPE_UINT, // save_fmt(4)
    INST_FIELD_DATA_TYPE_UINT, // reg_id(5)
    INST_FIELD_DATA_TYPE_UINT, // ddr_start(6)
    INST_FIELD_DATA_TYPE_UINT, // height(7)
    INST_FIELD_DATA_TYPE_UINT, // height_stride(8)
    INST_FIELD_DATA_TYPE_UINT, // width(9)
    INST_FIELD_DATA_TYPE_UINT, // width_stride(10)
    INST_FIELD_DATA_TYPE_UINT, // channel(11)
    INST_FIELD_DATA_TYPE_UINT, // channel_stride(12)
};


// Load field minus value
const vector<int> InstTable::LoadFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // mode(3)
    0, // bank_id(4)
    1, // length(5)
    1, // jump_write(6)
    1, // channel(7)
    1, // jump_read(8)
    0, // channel_offset(9)
    0, // pad_bottom(10)
    0, // pad_top(11)
    0, // mode_avg(12)
    0, // pad_start(13)
    0, // pad_end(14)
    1, // pad_idx(15)
    0, // bank_addr(16)
    0, // reg_id(17)
    0, // ddr_addr(18)
    0, // reg_id_1(19)
    0, // ddr_addr_1(20)
    0, // reg_id_2(21)
    0, // ddr_addr_2(22)
};


// Save field minus value
const vector<int> InstTable::SaveFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // hp_id(3)
    0, // bank_id(4)
    1, // length(5)
    1, // jump_read(6)
    1, // channel(7)
    1, // jump_write(8)
    0, // bank_addr(9)
    0, // reg_id(10)
    0, // ddr_addr(11)
};


// CBLoad field minus value
const vector<int> InstTable::CBLoadFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    1, // channel_group(3)
    1, // length(4)
    0, // cb_sel(5)
    0, // group_id_in(6)
    0, // bank_addr_in(7)
    0, // bank_addr_out(8)
};


// ConvInit field minus value
const vector<int> InstTable::ConvInitFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // act_type(3)
    0, // shift_cut(4)
    1, // jump_read(5)
    1, // stride_h(6)
    1, // kernel_h(7)
    0, // stride_offset_out_hbm(8)
    1, // valid_pixel_parallel(9)
    0, // stride_offset_in(10)
    1, // jump_write_cb(11)
    1, // stride_w(12)
    1, // kernel_w(13)
    0, // stride_offset_out_cb(14)
    1, // jump_read_endl(15)
    1, // stride_out_cb(16)
    0, // shift_relusix(17)
    0, // shift_bias(18)
    1, // jump_write_endl_cb(19)
    1, // stride_out_hbm(20)
    1, // jump_write_hbm(21)
    1, // jump_write_endl_hbm(22)
};


// Conv field minus value
const vector<int> InstTable::ConvFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    1, // channel_group(3)
    1, // length(4)
    0, // output_channel_offset(5)
    0, // channel_offset(6)
    0, // bank_id_in(7)
    0, // bank_id_out_cb(8)
    0, // pad_left(9)
    0, // pad_top(10)
    0, // bank_addr_in(11)
    0, // pad_right(12)
    0, // pad_bottom(13)
    0, // bank_addr_out_cb(14)
    0, // bank_addr_weights(15)
    0, // bank_addr_bias(16)
    0, // dest_mode(17)
    0, // bank_id_out_hbm(18)
    0, // bank_addr_out_hbm(19)
};


// PoolInit field minus value
const vector<int> InstTable::PoolInitFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // shift_cut(3)
    1, // kernel_w(4)
    1, // kernel_h(5)
    1, // jump_read(6)
    1, // jump_write(7)
    1, // stride_out(8)
    0, // stride_offset_out(9)
    1, // valid_pixel_parallel(10)
    0, // stride_offset_in(11)
    1, // stride_w(12)
    1, // stride_h(13)
    0, // pool_type(14)
};


// Pool field minus value
const vector<int> InstTable::PoolFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    1, // channel_group(3)
    0, // pad_left(4)
    0, // pad_top(5)
    0, // pad_right(6)
    0, // pad_bottom(7)
    0, // bank_id_in(8)
    0, // bank_id_out(9)
    1, // jump_read_endl(10)
    1, // length(11)
    1, // jump_write_endl(12)
    0, // bank_addr_in(13)
    0, // bank_addr_out(14)
};


// DWInit field minus value
const vector<int> InstTable::DWInitFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // act_type(3)
    0, // shift_cut(4)
    1, // jump_read(5)
    1, // stride_h(6)
    1, // kernel_h(7)
    1, // valid_pixel_parallel(8)
    0, // stride_offset_in(9)
    1, // jump_write(10)
    1, // stride_w(11)
    1, // kernel_w(12)
    0, // stride_offset_out(13)
    1, // jump_read_endl(14)
    1, // stride_out(15)
    0, // shift_relusix(16)
    0, // shift_bias(17)
    1, // jump_write_endl(18)
};


// DptWise field minus value
const vector<int> InstTable::DptWiseFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    1, // channel_group(3)
    1, // length(4)
    0, // channel_offset(5)
    0, // bank_id_in(6)
    0, // bank_id_out(7)
    0, // pad_left(8)
    0, // pad_top(9)
    0, // bank_addr_in(10)
    0, // pad_right(11)
    0, // pad_bottom(12)
    0, // bank_addr_out(13)
    0, // bank_addr_weights(14)
    0, // bank_addr_bias(15)
};


// ElewInit field minus value
const vector<int> InstTable::ElewInitFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // bank_id_in(3)
    0, // shift_read(4)
    1, // jump_read(5)
    0, // id(6)
    1, // jump_read_endl(7)
    0, // bank_addr_in(8)
};


// Elew field minus value
const vector<int> InstTable::ElewFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // bank_id_out(3)
    0, // shift_write(4)
    1, // jump_write(5)
    1, // num(6)
    1, // channel_group(7)
    0, // act_type(8)
    1, // valid_pixel_parallel(9)
    1, // length(10)
    1, // jump_write_endl(11)
    0, // bank_addr_out(12)
};


// End field minus value
const vector<int> InstTable::EndFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
};


// DumpBank field minus value
const vector<int> InstTable::DumpBankFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // save_name(3)
    0, // save_fmt(4)
    0, // bank_start(5)
    0, // bank_num(6)
};


// DumpDDR field minus value
const vector<int> InstTable::DumpDDRFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // save_name(3)
    0, // save_fmt(4)
    0, // reg_id(5)
    0, // ddr_start(6)
    0, // ddr_size(7)
};


// DumpDDRSlice field minus value
const vector<int> InstTable::DumpDDRSliceFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // save_name(3)
    0, // save_fmt(4)
    0, // reg_id(5)
    0, // ddr_start(6)
    0, // height(7)
    0, // height_stride(8)
    0, // width(9)
    0, // width_stride(10)
    0, // channel(11)
    0, // channel_stride(12)
};


// Load field position
const vector<map<uint32_t, uint32_t>> InstTable::LoadFieldPos = {
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_OPCODE, 28 }, // 0
        { LOAD_FIELD_DPDON, 24 }, // 1
        { LOAD_FIELD_DPDBY, 20 }, // 2
        { LOAD_FIELD_MODE, 19 }, // 3
        { LOAD_FIELD_BANK_ID, 12 }, // 4
        { LOAD_FIELD_LENGTH, 0 }, // 5
    },
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_JUMP_WRITE, 24 }, // 6
        { LOAD_FIELD_CHANNEL, 12 }, // 7
        { LOAD_FIELD_JUMP_READ, 0 }, // 8
    },
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_CHANNEL_OFFSET, 25 }, // 9
        { LOAD_FIELD_PAD_BOTTOM, 21 }, // 10
        { LOAD_FIELD_PAD_TOP, 17 }, // 11
        { LOAD_FIELD_MODE_AVG, 15 }, // 12
        { LOAD_FIELD_PAD_START, 10 }, // 13
        { LOAD_FIELD_PAD_END, 5 }, // 14
        { LOAD_FIELD_PAD_IDX, 0 }, // 15
    },
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_BANK_ADDR, 0 }, // 16
    },
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_REG_ID, 29 }, // 17
        { LOAD_FIELD_DDR_ADDR, 0 }, // 18
    },
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_REG_ID_1, 29 }, // 19
        { LOAD_FIELD_DDR_ADDR_1, 0 }, // 20
    },
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_REG_ID_2, 29 }, // 21
        { LOAD_FIELD_DDR_ADDR_2, 0 }, // 22
    },
};


// Save field position
const vector<map<uint32_t, uint32_t>> InstTable::SaveFieldPos = {
    map<uint32_t, uint32_t> {
        { SAVE_FIELD_OPCODE, 28 }, // 0
        { SAVE_FIELD_DPDON, 24 }, // 1
        { SAVE_FIELD_DPDBY, 20 }, // 2
        { SAVE_FIELD_HP_ID, 18 }, // 3
        { SAVE_FIELD_BANK_ID, 12 }, // 4
        { SAVE_FIELD_LENGTH, 0 }, // 5
    },
    map<uint32_t, uint32_t> {
        { SAVE_FIELD_JUMP_READ, 24 }, // 6
        { SAVE_FIELD_CHANNEL, 12 }, // 7
        { SAVE_FIELD_JUMP_WRITE, 0 }, // 8
    },
    map<uint32_t, uint32_t> {
        { SAVE_FIELD_BANK_ADDR, 0 }, // 9
    },
    map<uint32_t, uint32_t> {
        { SAVE_FIELD_REG_ID, 29 }, // 10
        { SAVE_FIELD_DDR_ADDR, 0 }, // 11
    },
};


// CBLoad field position
const vector<map<uint32_t, uint32_t>> InstTable::CBLoadFieldPos = {
    map<uint32_t, uint32_t> {
        { CBLOAD_FIELD_OPCODE, 28 }, // 0
        { CBLOAD_FIELD_DPDON, 24 }, // 1
        { CBLOAD_FIELD_DPDBY, 20 }, // 2
        { CBLOAD_FIELD_CHANNEL_GROUP, 12 }, // 3
        { CBLOAD_FIELD_LENGTH, 0 }, // 4
    },
    map<uint32_t, uint32_t> {
        { CBLOAD_FIELD_CB_SEL, 28 }, // 5
        { CBLOAD_FIELD_GROUP_ID_IN, 24 }, // 6
        { CBLOAD_FIELD_BANK_ADDR_IN, 0 }, // 7
    },
    map<uint32_t, uint32_t> {
        { CBLOAD_FIELD_BANK_ADDR_OUT, 0 }, // 8
    },
};


// ConvInit field position
const vector<map<uint32_t, uint32_t>> InstTable::ConvInitFieldPos = {
    map<uint32_t, uint32_t> {
        { CONVINIT_FIELD_OPCODE, 28 }, // 0
        { CONVINIT_FIELD_DPDON, 24 }, // 1
        { CONVINIT_FIELD_DPDBY, 20 }, // 2
        { CONVINIT_FIELD_ACT_TYPE, 16 }, // 3
        { CONVINIT_FIELD_SHIFT_CUT, 10 }, // 4
        { CONVINIT_FIELD_JUMP_READ, 0 }, // 5
    },
    map<uint32_t, uint32_t> {
        { CONVINIT_FIELD_STRIDE_H, 28 }, // 6
        { CONVINIT_FIELD_KERNEL_H, 24 }, // 7
        { CONVINIT_FIELD_STRIDE_OFFSET_OUT_HBM, 16 }, // 8
        { CONVINIT_FIELD_VALID_PIXEL_PARALLEL, 13 }, // 9
        { CONVINIT_FIELD_STRIDE_OFFSET_IN, 10 }, // 10
        { CONVINIT_FIELD_JUMP_WRITE_CB, 0 }, // 11
    },
    map<uint32_t, uint32_t> {
        { CONVINIT_FIELD_STRIDE_W, 28 }, // 12
        { CONVINIT_FIELD_KERNEL_W, 24 }, // 13
        { CONVINIT_FIELD_STRIDE_OFFSET_OUT_CB, 16 }, // 14
        { CONVINIT_FIELD_JUMP_READ_ENDL, 0 }, // 15
    },
    map<uint32_t, uint32_t> {
        { CONVINIT_FIELD_STRIDE_OUT_CB, 26 }, // 16
        { CONVINIT_FIELD_SHIFT_RELUSIX, 22 }, // 17
        { CONVINIT_FIELD_SHIFT_BIAS, 16 }, // 18
        { CONVINIT_FIELD_JUMP_WRITE_ENDL_CB, 0 }, // 19
    },
    map<uint32_t, uint32_t> {
        { CONVINIT_FIELD_STRIDE_OUT_HBM, 26 }, // 20
        { CONVINIT_FIELD_JUMP_WRITE_HBM, 16 }, // 21
        { CONVINIT_FIELD_JUMP_WRITE_ENDL_HBM, 0 }, // 22
    },
};


// Conv field position
const vector<map<uint32_t, uint32_t>> InstTable::ConvFieldPos = {
    map<uint32_t, uint32_t> {
        { CONV_FIELD_OPCODE, 28 }, // 0
        { CONV_FIELD_DPDON, 24 }, // 1
        { CONV_FIELD_DPDBY, 20 }, // 2
        { CONV_FIELD_CHANNEL_GROUP, 12 }, // 3
        { CONV_FIELD_LENGTH, 0 }, // 4
    },
    map<uint32_t, uint32_t> {
        { CONV_FIELD_OUTPUT_CHANNEL_OFFSET, 18 }, // 5
        { CONV_FIELD_CHANNEL_OFFSET, 12 }, // 6
        { CONV_FIELD_BANK_ID_IN, 6 }, // 7
        { CONV_FIELD_BANK_ID_OUT_CB, 0 }, // 8
    },
    map<uint32_t, uint32_t> {
        { CONV_FIELD_PAD_LEFT, 28 }, // 9
        { CONV_FIELD_PAD_TOP, 24 }, // 10
        { CONV_FIELD_BANK_ADDR_IN, 0 }, // 11
    },
    map<uint32_t, uint32_t> {
        { CONV_FIELD_PAD_RIGHT, 28 }, // 12
        { CONV_FIELD_PAD_BOTTOM, 24 }, // 13
        { CONV_FIELD_BANK_ADDR_OUT_CB, 0 }, // 14
    },
    map<uint32_t, uint32_t> {
        { CONV_FIELD_BANK_ADDR_WEIGHTS, 0 }, // 15
    },
    map<uint32_t, uint32_t> {
        { CONV_FIELD_BANK_ADDR_BIAS, 0 }, // 16
    },
    map<uint32_t, uint32_t> {
        { CONV_FIELD_DEST_MODE, 30 }, // 17
        { CONV_FIELD_BANK_ID_OUT_HBM, 24 }, // 18
        { CONV_FIELD_BANK_ADDR_OUT_HBM, 0 }, // 19
    },
};


// PoolInit field position
const vector<map<uint32_t, uint32_t>> InstTable::PoolInitFieldPos = {
    map<uint32_t, uint32_t> {
        { POOLINIT_FIELD_OPCODE, 28 }, // 0
        { POOLINIT_FIELD_DPDON, 24 }, // 1
        { POOLINIT_FIELD_DPDBY, 20 }, // 2
        { POOLINIT_FIELD_SHIFT_CUT, 16 }, // 3
        { POOLINIT_FIELD_KERNEL_W, 13 }, // 4
        { POOLINIT_FIELD_KERNEL_H, 10 }, // 5
        { POOLINIT_FIELD_JUMP_READ, 0 }, // 6
    },
    map<uint32_t, uint32_t> {
        { POOLINIT_FIELD_JUMP_WRITE, 21 }, // 7
        { POOLINIT_FIELD_STRIDE_OUT, 17 }, // 8
        { POOLINIT_FIELD_STRIDE_OFFSET_OUT, 14 }, // 9
        { POOLINIT_FIELD_VALID_PIXEL_PARALLEL, 11 }, // 10
        { POOLINIT_FIELD_STRIDE_OFFSET_IN, 8 }, // 11
        { POOLINIT_FIELD_STRIDE_W, 5 }, // 12
        { POOLINIT_FIELD_STRIDE_H, 2 }, // 13
        { POOLINIT_FIELD_POOL_TYPE, 0 }, // 14
    },
};


// Pool field position
const vector<map<uint32_t, uint32_t>> InstTable::PoolFieldPos = {
    map<uint32_t, uint32_t> {
        { POOL_FIELD_OPCODE, 28 }, // 0
        { POOL_FIELD_DPDON, 24 }, // 1
        { POOL_FIELD_DPDBY, 20 }, // 2
        { POOL_FIELD_CHANNEL_GROUP, 12 }, // 3
        { POOL_FIELD_PAD_LEFT, 9 }, // 4
        { POOL_FIELD_PAD_TOP, 6 }, // 5
        { POOL_FIELD_PAD_RIGHT, 3 }, // 6
        { POOL_FIELD_PAD_BOTTOM, 0 }, // 7
    },
    map<uint32_t, uint32_t> {
        { POOL_FIELD_BANK_ID_IN, 22 }, // 8
        { POOL_FIELD_BANK_ID_OUT, 16 }, // 9
        { POOL_FIELD_JUMP_READ_ENDL, 0 }, // 10
    },
    map<uint32_t, uint32_t> {
        { POOL_FIELD_LENGTH, 16 }, // 11
        { POOL_FIELD_JUMP_WRITE_ENDL, 0 }, // 12
    },
    map<uint32_t, uint32_t> {
        { POOL_FIELD_BANK_ADDR_IN, 0 }, // 13
    },
    map<uint32_t, uint32_t> {
        { POOL_FIELD_BANK_ADDR_OUT, 0 }, // 14
    },
};


// DWInit field position
const vector<map<uint32_t, uint32_t>> InstTable::DWInitFieldPos = {
    map<uint32_t, uint32_t> {
        { DWINIT_FIELD_OPCODE, 28 }, // 0
        { DWINIT_FIELD_DPDON, 24 }, // 1
        { DWINIT_FIELD_DPDBY, 20 }, // 2
        { DWINIT_FIELD_ACT_TYPE, 16 }, // 3
        { DWINIT_FIELD_SHIFT_CUT, 10 }, // 4
        { DWINIT_FIELD_JUMP_READ, 0 }, // 5
    },
    map<uint32_t, uint32_t> {
        { DWINIT_FIELD_STRIDE_H, 28 }, // 6
        { DWINIT_FIELD_KERNEL_H, 24 }, // 7
        { DWINIT_FIELD_VALID_PIXEL_PARALLEL, 13 }, // 8
        { DWINIT_FIELD_STRIDE_OFFSET_IN, 10 }, // 9
        { DWINIT_FIELD_JUMP_WRITE, 0 }, // 10
    },
    map<uint32_t, uint32_t> {
        { DWINIT_FIELD_STRIDE_W, 28 }, // 11
        { DWINIT_FIELD_KERNEL_W, 24 }, // 12
        { DWINIT_FIELD_STRIDE_OFFSET_OUT, 16 }, // 13
        { DWINIT_FIELD_JUMP_READ_ENDL, 0 }, // 14
    },
    map<uint32_t, uint32_t> {
        { DWINIT_FIELD_STRIDE_OUT, 26 }, // 15
        { DWINIT_FIELD_SHIFT_RELUSIX, 22 }, // 16
        { DWINIT_FIELD_SHIFT_BIAS, 16 }, // 17
        { DWINIT_FIELD_JUMP_WRITE_ENDL, 0 }, // 18
    },
};


// DptWise field position
const vector<map<uint32_t, uint32_t>> InstTable::DptWiseFieldPos = {
    map<uint32_t, uint32_t> {
        { DPTWISE_FIELD_OPCODE, 28 }, // 0
        { DPTWISE_FIELD_DPDON, 24 }, // 1
        { DPTWISE_FIELD_DPDBY, 20 }, // 2
        { DPTWISE_FIELD_CHANNEL_GROUP, 12 }, // 3
        { DPTWISE_FIELD_LENGTH, 0 }, // 4
    },
    map<uint32_t, uint32_t> {
        { DPTWISE_FIELD_CHANNEL_OFFSET, 12 }, // 5
        { DPTWISE_FIELD_BANK_ID_IN, 6 }, // 6
        { DPTWISE_FIELD_BANK_ID_OUT, 0 }, // 7
    },
    map<uint32_t, uint32_t> {
        { DPTWISE_FIELD_PAD_LEFT, 28 }, // 8
        { DPTWISE_FIELD_PAD_TOP, 24 }, // 9
        { DPTWISE_FIELD_BANK_ADDR_IN, 0 }, // 10
    },
    map<uint32_t, uint32_t> {
        { DPTWISE_FIELD_PAD_RIGHT, 28 }, // 11
        { DPTWISE_FIELD_PAD_BOTTOM, 24 }, // 12
        { DPTWISE_FIELD_BANK_ADDR_OUT, 0 }, // 13
    },
    map<uint32_t, uint32_t> {
        { DPTWISE_FIELD_BANK_ADDR_WEIGHTS, 0 }, // 14
    },
    map<uint32_t, uint32_t> {
        { DPTWISE_FIELD_BANK_ADDR_BIAS, 0 }, // 15
    },
};


// ElewInit field position
const vector<map<uint32_t, uint32_t>> InstTable::ElewInitFieldPos = {
    map<uint32_t, uint32_t> {
        { ELEWINIT_FIELD_OPCODE, 28 }, // 0
        { ELEWINIT_FIELD_DPDON, 24 }, // 1
        { ELEWINIT_FIELD_DPDBY, 20 }, // 2
        { ELEWINIT_FIELD_BANK_ID_IN, 14 }, // 3
        { ELEWINIT_FIELD_SHIFT_READ, 10 }, // 4
        { ELEWINIT_FIELD_JUMP_READ, 0 }, // 5
    },
    map<uint32_t, uint32_t> {
        { ELEWINIT_FIELD_ID, 16 }, // 6
        { ELEWINIT_FIELD_JUMP_READ_ENDL, 0 }, // 7
    },
    map<uint32_t, uint32_t> {
        { ELEWINIT_FIELD_BANK_ADDR_IN, 0 }, // 8
    },
};


// Elew field position
const vector<map<uint32_t, uint32_t>> InstTable::ElewFieldPos = {
    map<uint32_t, uint32_t> {
        { ELEW_FIELD_OPCODE, 28 }, // 0
        { ELEW_FIELD_DPDON, 24 }, // 1
        { ELEW_FIELD_DPDBY, 20 }, // 2
        { ELEW_FIELD_BANK_ID_OUT, 14 }, // 3
        { ELEW_FIELD_SHIFT_WRITE, 10 }, // 4
        { ELEW_FIELD_JUMP_WRITE, 0 }, // 5
    },
    map<uint32_t, uint32_t> {
        { ELEW_FIELD_NUM, 12 }, // 6
        { ELEW_FIELD_CHANNEL_GROUP, 4 }, // 7
        { ELEW_FIELD_ACT_TYPE, 3 }, // 8
        { ELEW_FIELD_VALID_PIXEL_PARALLEL, 0 }, // 9
    },
    map<uint32_t, uint32_t> {
        { ELEW_FIELD_LENGTH, 16 }, // 10
        { ELEW_FIELD_JUMP_WRITE_ENDL, 0 }, // 11
    },
    map<uint32_t, uint32_t> {
        { ELEW_FIELD_BANK_ADDR_OUT, 0 }, // 12
    },
};


// End field position
const vector<map<uint32_t, uint32_t>> InstTable::EndFieldPos = {
    map<uint32_t, uint32_t> {
        { END_FIELD_OPCODE, 28 }, // 0
        { END_FIELD_DPDON, 24 }, // 1
        { END_FIELD_DPDBY, 20 }, // 2
    },
};


// DumpBank field position
const vector<map<uint32_t, uint32_t>> InstTable::DumpBankFieldPos = {
    map<uint32_t, uint32_t> {
        { DUMPBANK_FIELD_OPCODE, 28 }, // 0
        { DUMPBANK_FIELD_DPDON, 24 }, // 1
        { DUMPBANK_FIELD_DPDBY, 20 }, // 2
        { DUMPBANK_FIELD_SAVE_NAME, 0 }, // 3
    },
    map<uint32_t, uint32_t> {
        { DUMPBANK_FIELD_SAVE_FMT, 26 }, // 4
        { DUMPBANK_FIELD_BANK_START, 18 }, // 5
        { DUMPBANK_FIELD_BANK_NUM, 10 }, // 6
    },
};


// DumpDDR field position
const vector<map<uint32_t, uint32_t>> InstTable::DumpDDRFieldPos = {
    map<uint32_t, uint32_t> {
        { DUMPDDR_FIELD_OPCODE, 28 }, // 0
        { DUMPDDR_FIELD_DPDON, 24 }, // 1
        { DUMPDDR_FIELD_DPDBY, 20 }, // 2
        { DUMPDDR_FIELD_SAVE_NAME, 0 }, // 3
    },
    map<uint32_t, uint32_t> {
        { DUMPDDR_FIELD_SAVE_FMT, 26 }, // 4
        { DUMPDDR_FIELD_REG_ID, 20 }, // 5
    },
    map<uint32_t, uint32_t> {
        { DUMPDDR_FIELD_DDR_START, 0 }, // 6
    },
    map<uint32_t, uint32_t> {
        { DUMPDDR_FIELD_DDR_SIZE, 0 }, // 7
    },
};


// DumpDDRSlice field position
const vector<map<uint32_t, uint32_t>> InstTable::DumpDDRSliceFieldPos = {
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_OPCODE, 28 }, // 0
        { DUMPDDRSLICE_FIELD_DPDON, 24 }, // 1
        { DUMPDDRSLICE_FIELD_DPDBY, 20 }, // 2
        { DUMPDDRSLICE_FIELD_SAVE_NAME, 0 }, // 3
    },
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_SAVE_FMT, 26 }, // 4
        { DUMPDDRSLICE_FIELD_REG_ID, 20 }, // 5
    },
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_DDR_START, 0 }, // 6
    },
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_HEIGHT, 16 }, // 7
        { DUMPDDRSLICE_FIELD_HEIGHT_STRIDE, 0 }, // 8
    },
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_WIDTH, 16 }, // 9
        { DUMPDDRSLICE_FIELD_WIDTH_STRIDE, 0 }, // 10
    },
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_CHANNEL, 16 }, // 11
        { DUMPDDRSLICE_FIELD_CHANNEL_STRIDE, 0 }, // 12
    },
};


// Load field length
const vector<map<uint32_t, uint32_t>> InstTable::LoadFieldLen = {
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_OPCODE, 4 }, // 0
        { LOAD_FIELD_DPDON, 4 }, // 1
        { LOAD_FIELD_DPDBY, 4 }, // 2
        { LOAD_FIELD_MODE, 1 }, // 3
        { LOAD_FIELD_BANK_ID, 7 }, // 4
        { LOAD_FIELD_LENGTH, 12 }, // 5
    },
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_JUMP_WRITE, 8 }, // 6
        { LOAD_FIELD_CHANNEL, 12 }, // 7
        { LOAD_FIELD_JUMP_READ, 12 }, // 8
    },
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_CHANNEL_OFFSET, 6 }, // 9
        { LOAD_FIELD_PAD_BOTTOM, 4 }, // 10
        { LOAD_FIELD_PAD_TOP, 4 }, // 11
        { LOAD_FIELD_MODE_AVG, 2 }, // 12
        { LOAD_FIELD_PAD_START, 5 }, // 13
        { LOAD_FIELD_PAD_END, 5 }, // 14
        { LOAD_FIELD_PAD_IDX, 5 }, // 15
    },
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_BANK_ADDR, 24 }, // 16
    },
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_REG_ID, 3 }, // 17
        { LOAD_FIELD_DDR_ADDR, 29 }, // 18
    },
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_REG_ID_1, 3 }, // 19
        { LOAD_FIELD_DDR_ADDR_1, 29 }, // 20
    },
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_REG_ID_2, 3 }, // 21
        { LOAD_FIELD_DDR_ADDR_2, 29 }, // 22
    },
};


// Save field length
const vector<map<uint32_t, uint32_t>> InstTable::SaveFieldLen = {
    map<uint32_t, uint32_t> {
        { SAVE_FIELD_OPCODE, 4 }, // 0
        { SAVE_FIELD_DPDON, 4 }, // 1
        { SAVE_FIELD_DPDBY, 4 }, // 2
        { SAVE_FIELD_HP_ID, 2 }, // 3
        { SAVE_FIELD_BANK_ID, 6 }, // 4
        { SAVE_FIELD_LENGTH, 12 }, // 5
    },
    map<uint32_t, uint32_t> {
        { SAVE_FIELD_JUMP_READ, 8 }, // 6
        { SAVE_FIELD_CHANNEL, 12 }, // 7
        { SAVE_FIELD_JUMP_WRITE, 12 }, // 8
    },
    map<uint32_t, uint32_t> {
        { SAVE_FIELD_BANK_ADDR, 24 }, // 9
    },
    map<uint32_t, uint32_t> {
        { SAVE_FIELD_REG_ID, 3 }, // 10
        { SAVE_FIELD_DDR_ADDR, 29 }, // 11
    },
};


// CBLoad field length
const vector<map<uint32_t, uint32_t>> InstTable::CBLoadFieldLen = {
    map<uint32_t, uint32_t> {
        { CBLOAD_FIELD_OPCODE, 4 }, // 0
        { CBLOAD_FIELD_DPDON, 4 }, // 1
        { CBLOAD_FIELD_DPDBY, 4 }, // 2
        { CBLOAD_FIELD_CHANNEL_GROUP, 8 }, // 3
        { CBLOAD_FIELD_LENGTH, 12 }, // 4
    },
    map<uint32_t, uint32_t> {
        { CBLOAD_FIELD_CB_SEL, 4 }, // 5
        { CBLOAD_FIELD_GROUP_ID_IN, 4 }, // 6
        { CBLOAD_FIELD_BANK_ADDR_IN, 24 }, // 7
    },
    map<uint32_t, uint32_t> {
        { CBLOAD_FIELD_BANK_ADDR_OUT, 24 }, // 8
    },
};


// ConvInit field length
const vector<map<uint32_t, uint32_t>> InstTable::ConvInitFieldLen = {
    map<uint32_t, uint32_t> {
        { CONVINIT_FIELD_OPCODE, 4 }, // 0
        { CONVINIT_FIELD_DPDON, 4 }, // 1
        { CONVINIT_FIELD_DPDBY, 4 }, // 2
        { CONVINIT_FIELD_ACT_TYPE, 4 }, // 3
        { CONVINIT_FIELD_SHIFT_CUT, 6 }, // 4
        { CONVINIT_FIELD_JUMP_READ, 10 }, // 5
    },
    map<uint32_t, uint32_t> {
        { CONVINIT_FIELD_STRIDE_H, 4 }, // 6
        { CONVINIT_FIELD_KERNEL_H, 4 }, // 7
        { CONVINIT_FIELD_STRIDE_OFFSET_OUT_HBM, 3 }, // 8
        { CONVINIT_FIELD_VALID_PIXEL_PARALLEL, 3 }, // 9
        { CONVINIT_FIELD_STRIDE_OFFSET_IN, 3 }, // 10
        { CONVINIT_FIELD_JUMP_WRITE_CB, 10 }, // 11
    },
    map<uint32_t, uint32_t> {
        { CONVINIT_FIELD_STRIDE_W, 4 }, // 12
        { CONVINIT_FIELD_KERNEL_W, 4 }, // 13
        { CONVINIT_FIELD_STRIDE_OFFSET_OUT_CB, 3 }, // 14
        { CONVINIT_FIELD_JUMP_READ_ENDL, 16 }, // 15
    },
    map<uint32_t, uint32_t> {
        { CONVINIT_FIELD_STRIDE_OUT_CB, 4 }, // 16
        { CONVINIT_FIELD_SHIFT_RELUSIX, 4 }, // 17
        { CONVINIT_FIELD_SHIFT_BIAS, 6 }, // 18
        { CONVINIT_FIELD_JUMP_WRITE_ENDL_CB, 16 }, // 19
    },
    map<uint32_t, uint32_t> {
        { CONVINIT_FIELD_STRIDE_OUT_HBM, 4 }, // 20
        { CONVINIT_FIELD_JUMP_WRITE_HBM, 10 }, // 21
        { CONVINIT_FIELD_JUMP_WRITE_ENDL_HBM, 16 }, // 22
    },
};


// Conv field length
const vector<map<uint32_t, uint32_t>> InstTable::ConvFieldLen = {
    map<uint32_t, uint32_t> {
        { CONV_FIELD_OPCODE, 4 }, // 0
        { CONV_FIELD_DPDON, 4 }, // 1
        { CONV_FIELD_DPDBY, 4 }, // 2
        { CONV_FIELD_CHANNEL_GROUP, 8 }, // 3
        { CONV_FIELD_LENGTH, 12 }, // 4
    },
    map<uint32_t, uint32_t> {
        { CONV_FIELD_OUTPUT_CHANNEL_OFFSET, 6 }, // 5
        { CONV_FIELD_CHANNEL_OFFSET, 6 }, // 6
        { CONV_FIELD_BANK_ID_IN, 6 }, // 7
        { CONV_FIELD_BANK_ID_OUT_CB, 6 }, // 8
    },
    map<uint32_t, uint32_t> {
        { CONV_FIELD_PAD_LEFT, 4 }, // 9
        { CONV_FIELD_PAD_TOP, 4 }, // 10
        { CONV_FIELD_BANK_ADDR_IN, 24 }, // 11
    },
    map<uint32_t, uint32_t> {
        { CONV_FIELD_PAD_RIGHT, 4 }, // 12
        { CONV_FIELD_PAD_BOTTOM, 4 }, // 13
        { CONV_FIELD_BANK_ADDR_OUT_CB, 24 }, // 14
    },
    map<uint32_t, uint32_t> {
        { CONV_FIELD_BANK_ADDR_WEIGHTS, 24 }, // 15
    },
    map<uint32_t, uint32_t> {
        { CONV_FIELD_BANK_ADDR_BIAS, 24 }, // 16
    },
    map<uint32_t, uint32_t> {
        { CONV_FIELD_DEST_MODE, 2 }, // 17
        { CONV_FIELD_BANK_ID_OUT_HBM, 6 }, // 18
        { CONV_FIELD_BANK_ADDR_OUT_HBM, 24 }, // 19
    },
};


// PoolInit field length
const vector<map<uint32_t, uint32_t>> InstTable::PoolInitFieldLen = {
    map<uint32_t, uint32_t> {
        { POOLINIT_FIELD_OPCODE, 4 }, // 0
        { POOLINIT_FIELD_DPDON, 4 }, // 1
        { POOLINIT_FIELD_DPDBY, 4 }, // 2
        { POOLINIT_FIELD_SHIFT_CUT, 4 }, // 3
        { POOLINIT_FIELD_KERNEL_W, 3 }, // 4
        { POOLINIT_FIELD_KERNEL_H, 3 }, // 5
        { POOLINIT_FIELD_JUMP_READ, 10 }, // 6
    },
    map<uint32_t, uint32_t> {
        { POOLINIT_FIELD_JUMP_WRITE, 10 }, // 7
        { POOLINIT_FIELD_STRIDE_OUT, 4 }, // 8
        { POOLINIT_FIELD_STRIDE_OFFSET_OUT, 3 }, // 9
        { POOLINIT_FIELD_VALID_PIXEL_PARALLEL, 3 }, // 10
        { POOLINIT_FIELD_STRIDE_OFFSET_IN, 3 }, // 11
        { POOLINIT_FIELD_STRIDE_W, 3 }, // 12
        { POOLINIT_FIELD_STRIDE_H, 3 }, // 13
        { POOLINIT_FIELD_POOL_TYPE, 2 }, // 14
    },
};


// Pool field length
const vector<map<uint32_t, uint32_t>> InstTable::PoolFieldLen = {
    map<uint32_t, uint32_t> {
        { POOL_FIELD_OPCODE, 4 }, // 0
        { POOL_FIELD_DPDON, 4 }, // 1
        { POOL_FIELD_DPDBY, 4 }, // 2
        { POOL_FIELD_CHANNEL_GROUP, 8 }, // 3
        { POOL_FIELD_PAD_LEFT, 3 }, // 4
        { POOL_FIELD_PAD_TOP, 3 }, // 5
        { POOL_FIELD_PAD_RIGHT, 3 }, // 6
        { POOL_FIELD_PAD_BOTTOM, 3 }, // 7
    },
    map<uint32_t, uint32_t> {
        { POOL_FIELD_BANK_ID_IN, 6 }, // 8
        { POOL_FIELD_BANK_ID_OUT, 6 }, // 9
        { POOL_FIELD_JUMP_READ_ENDL, 16 }, // 10
    },
    map<uint32_t, uint32_t> {
        { POOL_FIELD_LENGTH, 12 }, // 11
        { POOL_FIELD_JUMP_WRITE_ENDL, 16 }, // 12
    },
    map<uint32_t, uint32_t> {
        { POOL_FIELD_BANK_ADDR_IN, 24 }, // 13
    },
    map<uint32_t, uint32_t> {
        { POOL_FIELD_BANK_ADDR_OUT, 24 }, // 14
    },
};


// DWInit field length
const vector<map<uint32_t, uint32_t>> InstTable::DWInitFieldLen = {
    map<uint32_t, uint32_t> {
        { DWINIT_FIELD_OPCODE, 4 }, // 0
        { DWINIT_FIELD_DPDON, 4 }, // 1
        { DWINIT_FIELD_DPDBY, 4 }, // 2
        { DWINIT_FIELD_ACT_TYPE, 4 }, // 3
        { DWINIT_FIELD_SHIFT_CUT, 6 }, // 4
        { DWINIT_FIELD_JUMP_READ, 10 }, // 5
    },
    map<uint32_t, uint32_t> {
        { DWINIT_FIELD_STRIDE_H, 4 }, // 6
        { DWINIT_FIELD_KERNEL_H, 4 }, // 7
        { DWINIT_FIELD_VALID_PIXEL_PARALLEL, 3 }, // 8
        { DWINIT_FIELD_STRIDE_OFFSET_IN, 3 }, // 9
        { DWINIT_FIELD_JUMP_WRITE, 10 }, // 10
    },
    map<uint32_t, uint32_t> {
        { DWINIT_FIELD_STRIDE_W, 4 }, // 11
        { DWINIT_FIELD_KERNEL_W, 4 }, // 12
        { DWINIT_FIELD_STRIDE_OFFSET_OUT, 3 }, // 13
        { DWINIT_FIELD_JUMP_READ_ENDL, 16 }, // 14
    },
    map<uint32_t, uint32_t> {
        { DWINIT_FIELD_STRIDE_OUT, 4 }, // 15
        { DWINIT_FIELD_SHIFT_RELUSIX, 4 }, // 16
        { DWINIT_FIELD_SHIFT_BIAS, 6 }, // 17
        { DWINIT_FIELD_JUMP_WRITE_ENDL, 16 }, // 18
    },
};


// DptWise field length
const vector<map<uint32_t, uint32_t>> InstTable::DptWiseFieldLen = {
    map<uint32_t, uint32_t> {
        { DPTWISE_FIELD_OPCODE, 4 }, // 0
        { DPTWISE_FIELD_DPDON, 4 }, // 1
        { DPTWISE_FIELD_DPDBY, 4 }, // 2
        { DPTWISE_FIELD_CHANNEL_GROUP, 8 }, // 3
        { DPTWISE_FIELD_LENGTH, 12 }, // 4
    },
    map<uint32_t, uint32_t> {
        { DPTWISE_FIELD_CHANNEL_OFFSET, 6 }, // 5
        { DPTWISE_FIELD_BANK_ID_IN, 6 }, // 6
        { DPTWISE_FIELD_BANK_ID_OUT, 6 }, // 7
    },
    map<uint32_t, uint32_t> {
        { DPTWISE_FIELD_PAD_LEFT, 4 }, // 8
        { DPTWISE_FIELD_PAD_TOP, 4 }, // 9
        { DPTWISE_FIELD_BANK_ADDR_IN, 24 }, // 10
    },
    map<uint32_t, uint32_t> {
        { DPTWISE_FIELD_PAD_RIGHT, 4 }, // 11
        { DPTWISE_FIELD_PAD_BOTTOM, 4 }, // 12
        { DPTWISE_FIELD_BANK_ADDR_OUT, 24 }, // 13
    },
    map<uint32_t, uint32_t> {
        { DPTWISE_FIELD_BANK_ADDR_WEIGHTS, 24 }, // 14
    },
    map<uint32_t, uint32_t> {
        { DPTWISE_FIELD_BANK_ADDR_BIAS, 24 }, // 15
    },
};


// ElewInit field length
const vector<map<uint32_t, uint32_t>> InstTable::ElewInitFieldLen = {
    map<uint32_t, uint32_t> {
        { ELEWINIT_FIELD_OPCODE, 4 }, // 0
        { ELEWINIT_FIELD_DPDON, 4 }, // 1
        { ELEWINIT_FIELD_DPDBY, 4 }, // 2
        { ELEWINIT_FIELD_BANK_ID_IN, 6 }, // 3
        { ELEWINIT_FIELD_SHIFT_READ, 4 }, // 4
        { ELEWINIT_FIELD_JUMP_READ, 10 }, // 5
    },
    map<uint32_t, uint32_t> {
        { ELEWINIT_FIELD_ID, 2 }, // 6
        { ELEWINIT_FIELD_JUMP_READ_ENDL, 16 }, // 7
    },
    map<uint32_t, uint32_t> {
        { ELEWINIT_FIELD_BANK_ADDR_IN, 24 }, // 8
    },
};


// Elew field length
const vector<map<uint32_t, uint32_t>> InstTable::ElewFieldLen = {
    map<uint32_t, uint32_t> {
        { ELEW_FIELD_OPCODE, 4 }, // 0
        { ELEW_FIELD_DPDON, 4 }, // 1
        { ELEW_FIELD_DPDBY, 4 }, // 2
        { ELEW_FIELD_BANK_ID_OUT, 6 }, // 3
        { ELEW_FIELD_SHIFT_WRITE, 4 }, // 4
        { ELEW_FIELD_JUMP_WRITE, 10 }, // 5
    },
    map<uint32_t, uint32_t> {
        { ELEW_FIELD_NUM, 2 }, // 6
        { ELEW_FIELD_CHANNEL_GROUP, 8 }, // 7
        { ELEW_FIELD_ACT_TYPE, 1 }, // 8
        { ELEW_FIELD_VALID_PIXEL_PARALLEL, 3 }, // 9
    },
    map<uint32_t, uint32_t> {
        { ELEW_FIELD_LENGTH, 12 }, // 10
        { ELEW_FIELD_JUMP_WRITE_ENDL, 16 }, // 11
    },
    map<uint32_t, uint32_t> {
        { ELEW_FIELD_BANK_ADDR_OUT, 24 }, // 12
    },
};


// End field length
const vector<map<uint32_t, uint32_t>> InstTable::EndFieldLen = {
    map<uint32_t, uint32_t> {
        { END_FIELD_OPCODE, 4 }, // 0
        { END_FIELD_DPDON, 4 }, // 1
        { END_FIELD_DPDBY, 4 }, // 2
    },
};


// DumpBank field length
const vector<map<uint32_t, uint32_t>> InstTable::DumpBankFieldLen = {
    map<uint32_t, uint32_t> {
        { DUMPBANK_FIELD_OPCODE, 4 }, // 0
        { DUMPBANK_FIELD_DPDON, 4 }, // 1
        { DUMPBANK_FIELD_DPDBY, 4 }, // 2
        { DUMPBANK_FIELD_SAVE_NAME, 20 }, // 3
    },
    map<uint32_t, uint32_t> {
        { DUMPBANK_FIELD_SAVE_FMT, 6 }, // 4
        { DUMPBANK_FIELD_BANK_START, 8 }, // 5
        { DUMPBANK_FIELD_BANK_NUM, 8 }, // 6
    },
};


// DumpDDR field length
const vector<map<uint32_t, uint32_t>> InstTable::DumpDDRFieldLen = {
    map<uint32_t, uint32_t> {
        { DUMPDDR_FIELD_OPCODE, 4 }, // 0
        { DUMPDDR_FIELD_DPDON, 4 }, // 1
        { DUMPDDR_FIELD_DPDBY, 4 }, // 2
        { DUMPDDR_FIELD_SAVE_NAME, 20 }, // 3
    },
    map<uint32_t, uint32_t> {
        { DUMPDDR_FIELD_SAVE_FMT, 6 }, // 4
        { DUMPDDR_FIELD_REG_ID, 6 }, // 5
    },
    map<uint32_t, uint32_t> {
        { DUMPDDR_FIELD_DDR_START, 32 }, // 6
    },
    map<uint32_t, uint32_t> {
        { DUMPDDR_FIELD_DDR_SIZE, 32 }, // 7
    },
};


// DumpDDRSlice field length
const vector<map<uint32_t, uint32_t>> InstTable::DumpDDRSliceFieldLen = {
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_OPCODE, 4 }, // 0
        { DUMPDDRSLICE_FIELD_DPDON, 4 }, // 1
        { DUMPDDRSLICE_FIELD_DPDBY, 4 }, // 2
        { DUMPDDRSLICE_FIELD_SAVE_NAME, 20 }, // 3
    },
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_SAVE_FMT, 6 }, // 4
        { DUMPDDRSLICE_FIELD_REG_ID, 6 }, // 5
    },
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_DDR_START, 32 }, // 6
    },
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_HEIGHT, 16 }, // 7
        { DUMPDDRSLICE_FIELD_HEIGHT_STRIDE, 16 }, // 8
    },
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_WIDTH, 16 }, // 9
        { DUMPDDRSLICE_FIELD_WIDTH_STRIDE, 16 }, // 10
    },
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_CHANNEL, 16 }, // 11
        { DUMPDDRSLICE_FIELD_CHANNEL_STRIDE, 16 }, // 12
    },
};


// inst opcode to inst type map
const unordered_map<uint32_t, uint32_t> InstTable::OPCode2InstType = {
    { 0x00, INST_TYPE_LOAD }, // 0
    { 0x04, INST_TYPE_SAVE }, // 1
    { 0x05, INST_TYPE_CBLOAD }, // 2
    { 0x09, INST_TYPE_CONVINIT }, // 3
    { 0x08, INST_TYPE_CONV }, // 4
    { 0x06, INST_TYPE_POOLINIT }, // 5
    { 0x0C, INST_TYPE_POOL }, // 6
    { 0x0B, INST_TYPE_DWINIT }, // 7
    { 0x0A, INST_TYPE_DPTWISE }, // 8
    { 0x0D, INST_TYPE_ELEWINIT }, // 9
    { 0x0E, INST_TYPE_ELEW }, // 10
    { 0x07, INST_TYPE_END }, // 11
    { 0xFF, INST_TYPE_DUMPBANK }, // 12
    { 0xFE, INST_TYPE_DUMPDDR }, // 13
    { 0xFD, INST_TYPE_DUMPDDRSLICE }, // 14
};


// inst name to inst type map
const unordered_map<string, uint32_t> InstTable::InstName2InstType = {
    { "LOAD", INST_TYPE_LOAD }, // 0
    { "SAVE", INST_TYPE_SAVE }, // 1
    { "CBLOAD", INST_TYPE_CBLOAD }, // 2
    { "CONVINIT", INST_TYPE_CONVINIT }, // 3
    { "CONV", INST_TYPE_CONV }, // 4
    { "POOLINIT", INST_TYPE_POOLINIT }, // 5
    { "POOL", INST_TYPE_POOL }, // 6
    { "DWINIT", INST_TYPE_DWINIT }, // 7
    { "DPTWISE", INST_TYPE_DPTWISE }, // 8
    { "ELEWINIT", INST_TYPE_ELEWINIT }, // 9
    { "ELEW", INST_TYPE_ELEW }, // 10
    { "END", INST_TYPE_END }, // 11
    { "DUMPBANK", INST_TYPE_DUMPBANK }, // 12
    { "DUMPDDR", INST_TYPE_DUMPDDR }, // 13
    { "DUMPDDRSLICE", INST_TYPE_DUMPDDRSLICE }, // 14
};


// inst type to field name map
const unordered_map<uint32_t, const vector<string> &> InstTable::InstFieldName = {
    { INST_TYPE_LOAD, LoadFieldName }, // 0
    { INST_TYPE_SAVE, SaveFieldName }, // 1
    { INST_TYPE_CBLOAD, CBLoadFieldName }, // 2
    { INST_TYPE_CONVINIT, ConvInitFieldName }, // 3
    { INST_TYPE_CONV, ConvFieldName }, // 4
    { INST_TYPE_POOLINIT, PoolInitFieldName }, // 5
    { INST_TYPE_POOL, PoolFieldName }, // 6
    { INST_TYPE_DWINIT, DWInitFieldName }, // 7
    { INST_TYPE_DPTWISE, DptWiseFieldName }, // 8
    { INST_TYPE_ELEWINIT, ElewInitFieldName }, // 9
    { INST_TYPE_ELEW, ElewFieldName }, // 10
    { INST_TYPE_END, EndFieldName }, // 11
    { INST_TYPE_DUMPBANK, DumpBankFieldName }, // 12
    { INST_TYPE_DUMPDDR, DumpDDRFieldName }, // 13
    { INST_TYPE_DUMPDDRSLICE, DumpDDRSliceFieldName }, // 14
};


// inst type to field data type map
const unordered_map<uint32_t, const vector<int> &> InstTable::InstFieldDataType = {
    { INST_TYPE_LOAD, LoadFieldDataType }, // 0
    { INST_TYPE_SAVE, SaveFieldDataType }, // 1
    { INST_TYPE_CBLOAD, CBLoadFieldDataType }, // 2
    { INST_TYPE_CONVINIT, ConvInitFieldDataType }, // 3
    { INST_TYPE_CONV, ConvFieldDataType }, // 4
    { INST_TYPE_POOLINIT, PoolInitFieldDataType }, // 5
    { INST_TYPE_POOL, PoolFieldDataType }, // 6
    { INST_TYPE_DWINIT, DWInitFieldDataType }, // 7
    { INST_TYPE_DPTWISE, DptWiseFieldDataType }, // 8
    { INST_TYPE_ELEWINIT, ElewInitFieldDataType }, // 9
    { INST_TYPE_ELEW, ElewFieldDataType }, // 10
    { INST_TYPE_END, EndFieldDataType }, // 11
    { INST_TYPE_DUMPBANK, DumpBankFieldDataType }, // 12
    { INST_TYPE_DUMPDDR, DumpDDRFieldDataType }, // 13
    { INST_TYPE_DUMPDDRSLICE, DumpDDRSliceFieldDataType }, // 14
};


// inst type to field minus map
const unordered_map<uint32_t, const vector<int> &> InstTable::InstFieldMinus = {
    { INST_TYPE_LOAD, LoadFieldMinus }, // 0
    { INST_TYPE_SAVE, SaveFieldMinus }, // 1
    { INST_TYPE_CBLOAD, CBLoadFieldMinus }, // 2
    { INST_TYPE_CONVINIT, ConvInitFieldMinus }, // 3
    { INST_TYPE_CONV, ConvFieldMinus }, // 4
    { INST_TYPE_POOLINIT, PoolInitFieldMinus }, // 5
    { INST_TYPE_POOL, PoolFieldMinus }, // 6
    { INST_TYPE_DWINIT, DWInitFieldMinus }, // 7
    { INST_TYPE_DPTWISE, DptWiseFieldMinus }, // 8
    { INST_TYPE_ELEWINIT, ElewInitFieldMinus }, // 9
    { INST_TYPE_ELEW, ElewFieldMinus }, // 10
    { INST_TYPE_END, EndFieldMinus }, // 11
    { INST_TYPE_DUMPBANK, DumpBankFieldMinus }, // 12
    { INST_TYPE_DUMPDDR, DumpDDRFieldMinus }, // 13
    { INST_TYPE_DUMPDDRSLICE, DumpDDRSliceFieldMinus }, // 14
};


// inst type to field position map
const unordered_map<uint32_t, const vector<map<uint32_t, uint32_t>> &> InstTable::InstFieldPos = {
    { INST_TYPE_LOAD, LoadFieldPos }, // 0
    { INST_TYPE_SAVE, SaveFieldPos }, // 1
    { INST_TYPE_CBLOAD, CBLoadFieldPos }, // 2
    { INST_TYPE_CONVINIT, ConvInitFieldPos }, // 3
    { INST_TYPE_CONV, ConvFieldPos }, // 4
    { INST_TYPE_POOLINIT, PoolInitFieldPos }, // 5
    { INST_TYPE_POOL, PoolFieldPos }, // 6
    { INST_TYPE_DWINIT, DWInitFieldPos }, // 7
    { INST_TYPE_DPTWISE, DptWiseFieldPos }, // 8
    { INST_TYPE_ELEWINIT, ElewInitFieldPos }, // 9
    { INST_TYPE_ELEW, ElewFieldPos }, // 10
    { INST_TYPE_END, EndFieldPos }, // 11
    { INST_TYPE_DUMPBANK, DumpBankFieldPos }, // 12
    { INST_TYPE_DUMPDDR, DumpDDRFieldPos }, // 13
    { INST_TYPE_DUMPDDRSLICE, DumpDDRSliceFieldPos }, // 14
};


// inst type to field length map
const unordered_map<uint32_t, const vector<map<uint32_t, uint32_t>> &> InstTable::InstFieldLen = {
    { INST_TYPE_LOAD, LoadFieldLen }, // 0
    { INST_TYPE_SAVE, SaveFieldLen }, // 1
    { INST_TYPE_CBLOAD, CBLoadFieldLen }, // 2
    { INST_TYPE_CONVINIT, ConvInitFieldLen }, // 3
    { INST_TYPE_CONV, ConvFieldLen }, // 4
    { INST_TYPE_POOLINIT, PoolInitFieldLen }, // 5
    { INST_TYPE_POOL, PoolFieldLen }, // 6
    { INST_TYPE_DWINIT, DWInitFieldLen }, // 7
    { INST_TYPE_DPTWISE, DptWiseFieldLen }, // 8
    { INST_TYPE_ELEWINIT, ElewInitFieldLen }, // 9
    { INST_TYPE_ELEW, ElewFieldLen }, // 10
    { INST_TYPE_END, EndFieldLen }, // 11
    { INST_TYPE_DUMPBANK, DumpBankFieldLen }, // 12
    { INST_TYPE_DUMPDDR, DumpDDRFieldLen }, // 13
    { INST_TYPE_DUMPDDRSLICE, DumpDDRSliceFieldLen }, // 14
};


}

namespace DPUV_4E {
// instruction version
const string InstTable::INST_VERSION = "dpuv4e";


// category name vector
const vector<string> InstTable::CategoryName = {
    "LOAD",
    "SAVE",
    "CONV",
    "MISC",
};


// inst name vector
const vector<string> InstTable::InstName = {
    "LOAD", // 0
    "SAVE", // 1
    "CONVINIT", // 2
    "CONV", // 3
    "POOLINIT", // 4
    "POOL", // 5
    "DWINIT", // 6
    "DPTWISE", // 7
    "ELEWINIT", // 8
    "ELEW", // 9
    "END", // 10
    "DUMPBANK", // 11
    "DUMPDDR", // 12
    "DUMPDDRSLICE", // 13
};


// lower inst name vector
const vector<string> InstTable::InstNameLower = {
    "load", // 0
    "save", // 1
    "convinit", // 2
    "conv", // 3
    "poolinit", // 4
    "pool", // 5
    "dwinit", // 6
    "dptwise", // 7
    "elewinit", // 8
    "elew", // 9
    "end", // 10
    "dumpbank", // 11
    "dumpddr", // 12
    "dumpddrslice", // 13
};


// inst category
const vector<Category> InstTable::InstCategory = {
    Category::INST_CATEGORY_LOAD, // LOAD
    Category::INST_CATEGORY_SAVE, // SAVE
    Category::INST_CATEGORY_CONV, // CONVINIT
    Category::INST_CATEGORY_CONV, // CONV
    Category::INST_CATEGORY_MISC, // POOLINIT
    Category::INST_CATEGORY_MISC, // POOL
    Category::INST_CATEGORY_MISC, // DWINIT
    Category::INST_CATEGORY_MISC, // DPTWISE
    Category::INST_CATEGORY_MISC, // ELEWINIT
    Category::INST_CATEGORY_MISC, // ELEW
    Category::INST_CATEGORY_SAVE, // END
    Category::INST_CATEGORY_MISC, // DUMPBANK
    Category::INST_CATEGORY_MISC, // DUMPDDR
    Category::INST_CATEGORY_MISC, // DUMPDDRSLICE
};


// inst word number
const vector<uint32_t> InstTable::WordNum = {
    5, // LOAD
    4, // SAVE
    5, // CONVINIT
    5, // CONV
    4, // POOLINIT
    5, // POOL
    4, // DWINIT
    5, // DPTWISE
    2, // ELEWINIT
    4, // ELEW
    1, // END
    2, // DUMPBANK
    4, // DUMPDDR
    6, // DUMPDDRSLICE
};


// inst field number
const vector<uint32_t> InstTable::FieldNum = {
    LOAD_FIELD_MAX, // 0
    SAVE_FIELD_MAX, // 1
    CONVINIT_FIELD_MAX, // 2
    CONV_FIELD_MAX, // 3
    POOLINIT_FIELD_MAX, // 4
    POOL_FIELD_MAX, // 5
    DWINIT_FIELD_MAX, // 6
    DPTWISE_FIELD_MAX, // 7
    ELEWINIT_FIELD_MAX, // 8
    ELEW_FIELD_MAX, // 9
    END_FIELD_MAX, // 10
    DUMPBANK_FIELD_MAX, // 11
    DUMPDDR_FIELD_MAX, // 12
    DUMPDDRSLICE_FIELD_MAX, // 13
};


// inst opcode
const vector<uint32_t> InstTable::OPCode = {
    0x00, // LOAD
    0x04, // SAVE
    0x09, // CONVINIT
    0x08, // CONV
    0x06, // POOLINIT
    0x0C, // POOL
    0x0B, // DWINIT
    0x0A, // DPTWISE
    0x0D, // ELEWINIT
    0x0E, // ELEW
    0x07, // END
    0xFF, // DUMPBANK
    0xFE, // DUMPDDR
    0xFD, // DUMPDDRSLICE
};


// Load field name
const vector<string> InstTable::LoadFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "output_channel_num", // 3
    "bank_addr", // 4
    "load_mode", // 5
    "pad_idx", // 6
    "ddr_mode", // 7
    "bank_id", // 8
    "jump_read", // 9
    "jump_write", // 10
    "length", // 11
    "mode_avg", // 12
    "channel", // 13
    "reg_id", // 14
    "ddr_addr", // 15
    "pad_start", // 16
    "pad_end", // 17
    "jump_read_endl", // 18
};


// Save field name
const vector<string> InstTable::SaveFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "bank_addr", // 3
    "bank_id", // 4
    "jump_write", // 5
    "jump_read", // 6
    "length", // 7
    "channel", // 8
    "reg_id", // 9
    "ddr_addr", // 10
};


// ConvInit field name
const vector<string> InstTable::ConvInitFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "stride_out", // 3
    "stride_h", // 4
    "stride_w", // 5
    "kernel_h", // 6
    "kernel_w", // 7
    "act_type", // 8
    "shift_hswish", // 9
    "channel_group", // 10
    "channel_offset", // 11
    "length", // 12
    "shift_bias", // 13
    "stride_offset_in", // 14
    "jump_read", // 15
    "jump_read_endl", // 16
    "shift_cut", // 17
    "stride_offset_out", // 18
    "jump_write", // 19
    "jump_write_endl", // 20
    "shift_hsigmoid", // 21
    "hsigmoid_in", // 22
    "batch_num", // 23
    "stride_batch", // 24
};


// Conv field name
const vector<string> InstTable::ConvFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "bank_addr_bias", // 3
    "bank_addr_weights", // 4
    "pad_left", // 5
    "pad_right", // 6
    "pad_top", // 7
    "pad_bottom", // 8
    "valid_pixel_parallel", // 9
    "bank_addr_out", // 10
    "bank_id_in", // 11
    "bank_addr_in_1", // 12
    "bank_addr_in", // 13
    "bank_id_out", // 14
    "bank_addr_in_3", // 15
    "bank_addr_in_2", // 16
};


// PoolInit field name
const vector<string> InstTable::PoolInitFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "pool_type", // 3
    "kernel_h", // 4
    "kernel_w", // 5
    "stride_h", // 6
    "stride_w", // 7
    "stride_offset_in", // 8
    "stride_offset_out", // 9
    "stride_out", // 10
    "jump_read", // 11
    "jump_read_endl", // 12
    "multi_factor", // 13
    "jump_write", // 14
    "jump_write_endl", // 15
};


// Pool field name
const vector<string> InstTable::PoolFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "channel_group", // 3
    "shift_cut", // 4
    "bank_id_out", // 5
    "pad_left", // 6
    "pad_right", // 7
    "pad_top", // 8
    "pad_bottom", // 9
    "bank_id_in", // 10
    "valid_pixel_parallel", // 11
    "length", // 12
    "bank_addr_out", // 13
    "bank_addr_in_1", // 14
    "bank_addr_in", // 15
    "bank_addr_in_3", // 16
    "bank_addr_in_2", // 17
};


// DWInit field name
const vector<string> InstTable::DWInitFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "stride_out", // 3
    "stride_h", // 4
    "stride_w", // 5
    "kernel_h", // 6
    "kernel_w", // 7
    "act_type", // 8
    "channel_group", // 9
    "channel_offset", // 10
    "length", // 11
    "shift_bias", // 12
    "stride_offset_in", // 13
    "jump_read", // 14
    "jump_read_endl", // 15
    "shift_cut", // 16
    "stride_offset_out", // 17
    "jump_write", // 18
    "jump_write_endl", // 19
};


// DptWise field name
const vector<string> InstTable::DptWiseFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "bank_addr_bias", // 3
    "bank_addr_weights", // 4
    "pad_left", // 5
    "pad_right", // 6
    "pad_top", // 7
    "pad_bottom", // 8
    "valid_pixel_parallel", // 9
    "bank_addr_out", // 10
    "bank_id_in", // 11
    "bank_addr_in_1", // 12
    "bank_addr_in", // 13
    "bank_id_out", // 14
    "bank_addr_in_3", // 15
    "bank_addr_in_2", // 16
};


// ElewInit field name
const vector<string> InstTable::ElewInitFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "shift_read", // 3
    "bank_id_in", // 4
    "jump_read", // 5
    "id", // 6
    "jump_read_endl", // 7
    "bank_addr_in", // 8
};


// Elew field name
const vector<string> InstTable::ElewFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "bank_id_out", // 3
    "jump_write", // 4
    "stride_out", // 5
    "num", // 6
    "jump_write_endl", // 7
    "bank_addr_out", // 8
    "act_type", // 9
    "elew_type", // 10
    "channel_group", // 11
    "stride_offset_out", // 12
    "valid_pixel_parallel", // 13
    "length", // 14
    "shift_write", // 15
    "hsigmoid_in", // 16
    "shift_hsigmoid", // 17
};


// End field name
const vector<string> InstTable::EndFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
};


// DumpBank field name
const vector<string> InstTable::DumpBankFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "save_name", // 3
    "save_fmt", // 4
    "bank_start", // 5
    "bank_num", // 6
};


// DumpDDR field name
const vector<string> InstTable::DumpDDRFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "save_name", // 3
    "save_fmt", // 4
    "reg_id", // 5
    "ddr_start", // 6
    "ddr_size", // 7
};


// DumpDDRSlice field name
const vector<string> InstTable::DumpDDRSliceFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "save_name", // 3
    "save_fmt", // 4
    "reg_id", // 5
    "ddr_start", // 6
    "height", // 7
    "height_stride", // 8
    "width", // 9
    "width_stride", // 10
    "channel", // 11
    "channel_stride", // 12
};


// Load field data type
const vector<int> InstTable::LoadFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_UINT, // output_channel_num(3)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr(4)
    INST_FIELD_DATA_TYPE_UINT, // load_mode(5)
    INST_FIELD_DATA_TYPE_UINT, // pad_idx(6)
    INST_FIELD_DATA_TYPE_UINT, // ddr_mode(7)
    INST_FIELD_DATA_TYPE_UINT, // bank_id(8)
    INST_FIELD_DATA_TYPE_UINT, // jump_read(9)
    INST_FIELD_DATA_TYPE_UINT, // jump_write(10)
    INST_FIELD_DATA_TYPE_UINT, // length(11)
    INST_FIELD_DATA_TYPE_UINT, // mode_avg(12)
    INST_FIELD_DATA_TYPE_UINT, // channel(13)
    INST_FIELD_DATA_TYPE_UINT, // reg_id(14)
    INST_FIELD_DATA_TYPE_UINT, // ddr_addr(15)
    INST_FIELD_DATA_TYPE_UINT, // pad_start(16)
    INST_FIELD_DATA_TYPE_UINT, // pad_end(17)
    INST_FIELD_DATA_TYPE_UINT, // jump_read_endl(18)
};


// Save field data type
const vector<int> InstTable::SaveFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr(3)
    INST_FIELD_DATA_TYPE_UINT, // bank_id(4)
    INST_FIELD_DATA_TYPE_UINT, // jump_write(5)
    INST_FIELD_DATA_TYPE_UINT, // jump_read(6)
    INST_FIELD_DATA_TYPE_UINT, // length(7)
    INST_FIELD_DATA_TYPE_UINT, // channel(8)
    INST_FIELD_DATA_TYPE_UINT, // reg_id(9)
    INST_FIELD_DATA_TYPE_UINT, // ddr_addr(10)
};


// ConvInit field data type
const vector<int> InstTable::ConvInitFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_UINT, // stride_out(3)
    INST_FIELD_DATA_TYPE_UINT, // stride_h(4)
    INST_FIELD_DATA_TYPE_UINT, // stride_w(5)
    INST_FIELD_DATA_TYPE_UINT, // kernel_h(6)
    INST_FIELD_DATA_TYPE_UINT, // kernel_w(7)
    INST_FIELD_DATA_TYPE_UINT, // act_type(8)
    INST_FIELD_DATA_TYPE_UINT, // shift_hswish(9)
    INST_FIELD_DATA_TYPE_UINT, // channel_group(10)
    INST_FIELD_DATA_TYPE_UINT, // channel_offset(11)
    INST_FIELD_DATA_TYPE_UINT, // length(12)
    INST_FIELD_DATA_TYPE_UINT, // shift_bias(13)
    INST_FIELD_DATA_TYPE_UINT, // stride_offset_in(14)
    INST_FIELD_DATA_TYPE_UINT, // jump_read(15)
    INST_FIELD_DATA_TYPE_UINT, // jump_read_endl(16)
    INST_FIELD_DATA_TYPE_UINT, // shift_cut(17)
    INST_FIELD_DATA_TYPE_UINT, // stride_offset_out(18)
    INST_FIELD_DATA_TYPE_UINT, // jump_write(19)
    INST_FIELD_DATA_TYPE_UINT, // jump_write_endl(20)
    INST_FIELD_DATA_TYPE_UINT, // shift_hsigmoid(21)
    INST_FIELD_DATA_TYPE_UINT, // hsigmoid_in(22)
    INST_FIELD_DATA_TYPE_UINT, // batch_num(23)
    INST_FIELD_DATA_TYPE_UINT, // stride_batch(24)
};


// Conv field data type
const vector<int> InstTable::ConvFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_bias(3)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_weights(4)
    INST_FIELD_DATA_TYPE_UINT, // pad_left(5)
    INST_FIELD_DATA_TYPE_UINT, // pad_right(6)
    INST_FIELD_DATA_TYPE_UINT, // pad_top(7)
    INST_FIELD_DATA_TYPE_UINT, // pad_bottom(8)
    INST_FIELD_DATA_TYPE_UINT, // valid_pixel_parallel(9)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_out(10)
    INST_FIELD_DATA_TYPE_UINT, // bank_id_in(11)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in_1(12)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in(13)
    INST_FIELD_DATA_TYPE_UINT, // bank_id_out(14)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in_3(15)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in_2(16)
};


// PoolInit field data type
const vector<int> InstTable::PoolInitFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_UINT, // pool_type(3)
    INST_FIELD_DATA_TYPE_UINT, // kernel_h(4)
    INST_FIELD_DATA_TYPE_UINT, // kernel_w(5)
    INST_FIELD_DATA_TYPE_UINT, // stride_h(6)
    INST_FIELD_DATA_TYPE_UINT, // stride_w(7)
    INST_FIELD_DATA_TYPE_UINT, // stride_offset_in(8)
    INST_FIELD_DATA_TYPE_UINT, // stride_offset_out(9)
    INST_FIELD_DATA_TYPE_UINT, // stride_out(10)
    INST_FIELD_DATA_TYPE_UINT, // jump_read(11)
    INST_FIELD_DATA_TYPE_UINT, // jump_read_endl(12)
    INST_FIELD_DATA_TYPE_UINT, // multi_factor(13)
    INST_FIELD_DATA_TYPE_UINT, // jump_write(14)
    INST_FIELD_DATA_TYPE_UINT, // jump_write_endl(15)
};


// Pool field data type
const vector<int> InstTable::PoolFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_UINT, // channel_group(3)
    INST_FIELD_DATA_TYPE_UINT, // shift_cut(4)
    INST_FIELD_DATA_TYPE_UINT, // bank_id_out(5)
    INST_FIELD_DATA_TYPE_UINT, // pad_left(6)
    INST_FIELD_DATA_TYPE_UINT, // pad_right(7)
    INST_FIELD_DATA_TYPE_UINT, // pad_top(8)
    INST_FIELD_DATA_TYPE_UINT, // pad_bottom(9)
    INST_FIELD_DATA_TYPE_UINT, // bank_id_in(10)
    INST_FIELD_DATA_TYPE_UINT, // valid_pixel_parallel(11)
    INST_FIELD_DATA_TYPE_UINT, // length(12)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_out(13)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in_1(14)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in(15)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in_3(16)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in_2(17)
};


// DWInit field data type
const vector<int> InstTable::DWInitFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_UINT, // stride_out(3)
    INST_FIELD_DATA_TYPE_UINT, // stride_h(4)
    INST_FIELD_DATA_TYPE_UINT, // stride_w(5)
    INST_FIELD_DATA_TYPE_UINT, // kernel_h(6)
    INST_FIELD_DATA_TYPE_UINT, // kernel_w(7)
    INST_FIELD_DATA_TYPE_UINT, // act_type(8)
    INST_FIELD_DATA_TYPE_UINT, // channel_group(9)
    INST_FIELD_DATA_TYPE_UINT, // channel_offset(10)
    INST_FIELD_DATA_TYPE_UINT, // length(11)
    INST_FIELD_DATA_TYPE_UINT, // shift_bias(12)
    INST_FIELD_DATA_TYPE_UINT, // stride_offset_in(13)
    INST_FIELD_DATA_TYPE_UINT, // jump_read(14)
    INST_FIELD_DATA_TYPE_UINT, // jump_read_endl(15)
    INST_FIELD_DATA_TYPE_UINT, // shift_cut(16)
    INST_FIELD_DATA_TYPE_UINT, // stride_offset_out(17)
    INST_FIELD_DATA_TYPE_UINT, // jump_write(18)
    INST_FIELD_DATA_TYPE_UINT, // jump_write_endl(19)
};


// DptWise field data type
const vector<int> InstTable::DptWiseFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_bias(3)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_weights(4)
    INST_FIELD_DATA_TYPE_UINT, // pad_left(5)
    INST_FIELD_DATA_TYPE_UINT, // pad_right(6)
    INST_FIELD_DATA_TYPE_UINT, // pad_top(7)
    INST_FIELD_DATA_TYPE_UINT, // pad_bottom(8)
    INST_FIELD_DATA_TYPE_UINT, // valid_pixel_parallel(9)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_out(10)
    INST_FIELD_DATA_TYPE_UINT, // bank_id_in(11)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in_1(12)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in(13)
    INST_FIELD_DATA_TYPE_UINT, // bank_id_out(14)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in_3(15)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in_2(16)
};


// ElewInit field data type
const vector<int> InstTable::ElewInitFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_UINT, // shift_read(3)
    INST_FIELD_DATA_TYPE_UINT, // bank_id_in(4)
    INST_FIELD_DATA_TYPE_UINT, // jump_read(5)
    INST_FIELD_DATA_TYPE_UINT, // id(6)
    INST_FIELD_DATA_TYPE_UINT, // jump_read_endl(7)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in(8)
};


// Elew field data type
const vector<int> InstTable::ElewFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_UINT, // bank_id_out(3)
    INST_FIELD_DATA_TYPE_UINT, // jump_write(4)
    INST_FIELD_DATA_TYPE_UINT, // stride_out(5)
    INST_FIELD_DATA_TYPE_UINT, // num(6)
    INST_FIELD_DATA_TYPE_UINT, // jump_write_endl(7)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_out(8)
    INST_FIELD_DATA_TYPE_UINT, // act_type(9)
    INST_FIELD_DATA_TYPE_UINT, // elew_type(10)
    INST_FIELD_DATA_TYPE_UINT, // channel_group(11)
    INST_FIELD_DATA_TYPE_UINT, // stride_offset_out(12)
    INST_FIELD_DATA_TYPE_UINT, // valid_pixel_parallel(13)
    INST_FIELD_DATA_TYPE_UINT, // length(14)
    INST_FIELD_DATA_TYPE_UINT, // shift_write(15)
    INST_FIELD_DATA_TYPE_UINT, // hsigmoid_in(16)
    INST_FIELD_DATA_TYPE_UINT, // shift_hsigmoid(17)
};


// End field data type
const vector<int> InstTable::EndFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
};


// DumpBank field data type
const vector<int> InstTable::DumpBankFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_STRING, // save_name(3)
    INST_FIELD_DATA_TYPE_UINT, // save_fmt(4)
    INST_FIELD_DATA_TYPE_UINT, // bank_start(5)
    INST_FIELD_DATA_TYPE_UINT, // bank_num(6)
};


// DumpDDR field data type
const vector<int> InstTable::DumpDDRFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_STRING, // save_name(3)
    INST_FIELD_DATA_TYPE_UINT, // save_fmt(4)
    INST_FIELD_DATA_TYPE_UINT, // reg_id(5)
    INST_FIELD_DATA_TYPE_UINT, // ddr_start(6)
    INST_FIELD_DATA_TYPE_UINT, // ddr_size(7)
};


// DumpDDRSlice field data type
const vector<int> InstTable::DumpDDRSliceFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_STRING, // save_name(3)
    INST_FIELD_DATA_TYPE_UINT, // save_fmt(4)
    INST_FIELD_DATA_TYPE_UINT, // reg_id(5)
    INST_FIELD_DATA_TYPE_UINT, // ddr_start(6)
    INST_FIELD_DATA_TYPE_UINT, // height(7)
    INST_FIELD_DATA_TYPE_UINT, // height_stride(8)
    INST_FIELD_DATA_TYPE_UINT, // width(9)
    INST_FIELD_DATA_TYPE_UINT, // width_stride(10)
    INST_FIELD_DATA_TYPE_UINT, // channel(11)
    INST_FIELD_DATA_TYPE_UINT, // channel_stride(12)
};


// Load field minus value
const vector<int> InstTable::LoadFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    1, // output_channel_num(3)
    0, // bank_addr(4)
    0, // load_mode(5)
    1, // pad_idx(6)
    0, // ddr_mode(7)
    0, // bank_id(8)
    1, // jump_read(9)
    1, // jump_write(10)
    1, // length(11)
    0, // mode_avg(12)
    1, // channel(13)
    0, // reg_id(14)
    0, // ddr_addr(15)
    0, // pad_start(16)
    0, // pad_end(17)
    1, // jump_read_endl(18)
};


// Save field minus value
const vector<int> InstTable::SaveFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // bank_addr(3)
    0, // bank_id(4)
    1, // jump_write(5)
    1, // jump_read(6)
    1, // length(7)
    1, // channel(8)
    0, // reg_id(9)
    0, // ddr_addr(10)
};


// ConvInit field minus value
const vector<int> InstTable::ConvInitFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    1, // stride_out(3)
    1, // stride_h(4)
    1, // stride_w(5)
    1, // kernel_h(6)
    1, // kernel_w(7)
    0, // act_type(8)
    0, // shift_hswish(9)
    1, // channel_group(10)
    0, // channel_offset(11)
    1, // length(12)
    0, // shift_bias(13)
    0, // stride_offset_in(14)
    1, // jump_read(15)
    1, // jump_read_endl(16)
    0, // shift_cut(17)
    0, // stride_offset_out(18)
    1, // jump_write(19)
    1, // jump_write_endl(20)
    0, // shift_hsigmoid(21)
    0, // hsigmoid_in(22)
    1, // batch_num(23)
    1, // stride_batch(24)
};


// Conv field minus value
const vector<int> InstTable::ConvFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // bank_addr_bias(3)
    0, // bank_addr_weights(4)
    0, // pad_left(5)
    0, // pad_right(6)
    0, // pad_top(7)
    0, // pad_bottom(8)
    1, // valid_pixel_parallel(9)
    0, // bank_addr_out(10)
    0, // bank_id_in(11)
    0, // bank_addr_in_1(12)
    0, // bank_addr_in(13)
    0, // bank_id_out(14)
    0, // bank_addr_in_3(15)
    0, // bank_addr_in_2(16)
};


// PoolInit field minus value
const vector<int> InstTable::PoolInitFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // pool_type(3)
    1, // kernel_h(4)
    1, // kernel_w(5)
    1, // stride_h(6)
    1, // stride_w(7)
    0, // stride_offset_in(8)
    0, // stride_offset_out(9)
    1, // stride_out(10)
    1, // jump_read(11)
    1, // jump_read_endl(12)
    0, // multi_factor(13)
    1, // jump_write(14)
    1, // jump_write_endl(15)
};


// Pool field minus value
const vector<int> InstTable::PoolFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    1, // channel_group(3)
    0, // shift_cut(4)
    0, // bank_id_out(5)
    0, // pad_left(6)
    0, // pad_right(7)
    0, // pad_top(8)
    0, // pad_bottom(9)
    0, // bank_id_in(10)
    1, // valid_pixel_parallel(11)
    1, // length(12)
    0, // bank_addr_out(13)
    0, // bank_addr_in_1(14)
    0, // bank_addr_in(15)
    0, // bank_addr_in_3(16)
    0, // bank_addr_in_2(17)
};


// DWInit field minus value
const vector<int> InstTable::DWInitFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    1, // stride_out(3)
    1, // stride_h(4)
    1, // stride_w(5)
    1, // kernel_h(6)
    1, // kernel_w(7)
    0, // act_type(8)
    1, // channel_group(9)
    0, // channel_offset(10)
    1, // length(11)
    0, // shift_bias(12)
    0, // stride_offset_in(13)
    1, // jump_read(14)
    1, // jump_read_endl(15)
    0, // shift_cut(16)
    0, // stride_offset_out(17)
    1, // jump_write(18)
    1, // jump_write_endl(19)
};


// DptWise field minus value
const vector<int> InstTable::DptWiseFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // bank_addr_bias(3)
    0, // bank_addr_weights(4)
    0, // pad_left(5)
    0, // pad_right(6)
    0, // pad_top(7)
    0, // pad_bottom(8)
    1, // valid_pixel_parallel(9)
    0, // bank_addr_out(10)
    0, // bank_id_in(11)
    0, // bank_addr_in_1(12)
    0, // bank_addr_in(13)
    0, // bank_id_out(14)
    0, // bank_addr_in_3(15)
    0, // bank_addr_in_2(16)
};


// ElewInit field minus value
const vector<int> InstTable::ElewInitFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // shift_read(3)
    0, // bank_id_in(4)
    1, // jump_read(5)
    0, // id(6)
    1, // jump_read_endl(7)
    0, // bank_addr_in(8)
};


// Elew field minus value
const vector<int> InstTable::ElewFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // bank_id_out(3)
    1, // jump_write(4)
    1, // stride_out(5)
    1, // num(6)
    1, // jump_write_endl(7)
    0, // bank_addr_out(8)
    0, // act_type(9)
    0, // elew_type(10)
    1, // channel_group(11)
    0, // stride_offset_out(12)
    1, // valid_pixel_parallel(13)
    1, // length(14)
    0, // shift_write(15)
    0, // hsigmoid_in(16)
    0, // shift_hsigmoid(17)
};


// End field minus value
const vector<int> InstTable::EndFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
};


// DumpBank field minus value
const vector<int> InstTable::DumpBankFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // save_name(3)
    0, // save_fmt(4)
    0, // bank_start(5)
    0, // bank_num(6)
};


// DumpDDR field minus value
const vector<int> InstTable::DumpDDRFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // save_name(3)
    0, // save_fmt(4)
    0, // reg_id(5)
    0, // ddr_start(6)
    0, // ddr_size(7)
};


// DumpDDRSlice field minus value
const vector<int> InstTable::DumpDDRSliceFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // save_name(3)
    0, // save_fmt(4)
    0, // reg_id(5)
    0, // ddr_start(6)
    0, // height(7)
    0, // height_stride(8)
    0, // width(9)
    0, // width_stride(10)
    0, // channel(11)
    0, // channel_stride(12)
};


// Load field position
const vector<map<uint32_t, uint32_t>> InstTable::LoadFieldPos = {
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_OPCODE, 28 }, // 0
        { LOAD_FIELD_DPDON, 24 }, // 1
        { LOAD_FIELD_DPDBY, 20 }, // 2
        { LOAD_FIELD_OUTPUT_CHANNEL_NUM, 13 }, // 3
        { LOAD_FIELD_BANK_ADDR, 0 }, // 4
    },
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_LOAD_MODE, 30 }, // 5
        { LOAD_FIELD_PAD_IDX, 25 }, // 6
        { LOAD_FIELD_DDR_MODE, 24 }, // 7
        { LOAD_FIELD_BANK_ID, 16 }, // 8
        { LOAD_FIELD_JUMP_READ, 0 }, // 9
    },
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_JUMP_WRITE, 24 }, // 10
        { LOAD_FIELD_LENGTH, 14 }, // 11
        { LOAD_FIELD_MODE_AVG, 12 }, // 12
        { LOAD_FIELD_CHANNEL, 0 }, // 13
    },
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_REG_ID, 29 }, // 14
        { LOAD_FIELD_DDR_ADDR, 0 }, // 15
    },
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_PAD_START, 27 }, // 16
        { LOAD_FIELD_PAD_END, 21 }, // 17
        { LOAD_FIELD_JUMP_READ_ENDL, 0 }, // 18
    },
};


// Save field position
const vector<map<uint32_t, uint32_t>> InstTable::SaveFieldPos = {
    map<uint32_t, uint32_t> {
        { SAVE_FIELD_OPCODE, 28 }, // 0
        { SAVE_FIELD_DPDON, 24 }, // 1
        { SAVE_FIELD_DPDBY, 20 }, // 2
        { SAVE_FIELD_BANK_ADDR, 0 }, // 3
    },
    map<uint32_t, uint32_t> {
        { SAVE_FIELD_BANK_ID, 16 }, // 4
        { SAVE_FIELD_JUMP_WRITE, 0 }, // 5
    },
    map<uint32_t, uint32_t> {
        { SAVE_FIELD_JUMP_READ, 24 }, // 6
        { SAVE_FIELD_LENGTH, 14 }, // 7
        { SAVE_FIELD_CHANNEL, 0 }, // 8
    },
    map<uint32_t, uint32_t> {
        { SAVE_FIELD_REG_ID, 29 }, // 9
        { SAVE_FIELD_DDR_ADDR, 0 }, // 10
    },
};


// ConvInit field position
const vector<map<uint32_t, uint32_t>> InstTable::ConvInitFieldPos = {
    map<uint32_t, uint32_t> {
        { CONVINIT_FIELD_OPCODE, 28 }, // 0
        { CONVINIT_FIELD_DPDON, 24 }, // 1
        { CONVINIT_FIELD_DPDBY, 20 }, // 2
        { CONVINIT_FIELD_STRIDE_OUT, 16 }, // 3
        { CONVINIT_FIELD_STRIDE_H, 12 }, // 4
        { CONVINIT_FIELD_STRIDE_W, 8 }, // 5
        { CONVINIT_FIELD_KERNEL_H, 4 }, // 6
        { CONVINIT_FIELD_KERNEL_W, 0 }, // 7
    },
    map<uint32_t, uint32_t> {
        { CONVINIT_FIELD_ACT_TYPE, 28 }, // 8
        { CONVINIT_FIELD_SHIFT_HSWISH, 24 }, // 9
        { CONVINIT_FIELD_CHANNEL_GROUP, 16 }, // 10
        { CONVINIT_FIELD_CHANNEL_OFFSET, 10 }, // 11
        { CONVINIT_FIELD_LENGTH, 0 }, // 12
    },
    map<uint32_t, uint32_t> {
        { CONVINIT_FIELD_SHIFT_BIAS, 26 }, // 13
        { CONVINIT_FIELD_STRIDE_OFFSET_IN, 23 }, // 14
        { CONVINIT_FIELD_JUMP_READ, 13 }, // 15
        { CONVINIT_FIELD_JUMP_READ_ENDL, 0 }, // 16
    },
    map<uint32_t, uint32_t> {
        { CONVINIT_FIELD_SHIFT_CUT, 26 }, // 17
        { CONVINIT_FIELD_STRIDE_OFFSET_OUT, 23 }, // 18
        { CONVINIT_FIELD_JUMP_WRITE, 13 }, // 19
        { CONVINIT_FIELD_JUMP_WRITE_ENDL, 0 }, // 20
    },
    map<uint32_t, uint32_t> {
        { CONVINIT_FIELD_SHIFT_HSIGMOID, 16 }, // 21
        { CONVINIT_FIELD_HSIGMOID_IN, 12 }, // 22
        { CONVINIT_FIELD_BATCH_NUM, 10 }, // 23
        { CONVINIT_FIELD_STRIDE_BATCH, 0 }, // 24
    },
};


// Conv field position
const vector<map<uint32_t, uint32_t>> InstTable::ConvFieldPos = {
    map<uint32_t, uint32_t> {
        { CONV_FIELD_OPCODE, 28 }, // 0
        { CONV_FIELD_DPDON, 24 }, // 1
        { CONV_FIELD_DPDBY, 20 }, // 2
        { CONV_FIELD_BANK_ADDR_BIAS, 0 }, // 3
    },
    map<uint32_t, uint32_t> {
        { CONV_FIELD_BANK_ADDR_WEIGHTS, 0 }, // 4
    },
    map<uint32_t, uint32_t> {
        { CONV_FIELD_PAD_LEFT, 28 }, // 5
        { CONV_FIELD_PAD_RIGHT, 24 }, // 6
        { CONV_FIELD_PAD_TOP, 20 }, // 7
        { CONV_FIELD_PAD_BOTTOM, 16 }, // 8
        { CONV_FIELD_VALID_PIXEL_PARALLEL, 13 }, // 9
        { CONV_FIELD_BANK_ADDR_OUT, 0 }, // 10
    },
    map<uint32_t, uint32_t> {
        { CONV_FIELD_BANK_ID_IN, 26 }, // 11
        { CONV_FIELD_BANK_ADDR_IN_1, 13 }, // 12
        { CONV_FIELD_BANK_ADDR_IN, 0 }, // 13
    },
    map<uint32_t, uint32_t> {
        { CONV_FIELD_BANK_ID_OUT, 26 }, // 14
        { CONV_FIELD_BANK_ADDR_IN_3, 13 }, // 15
        { CONV_FIELD_BANK_ADDR_IN_2, 0 }, // 16
    },
};


// PoolInit field position
const vector<map<uint32_t, uint32_t>> InstTable::PoolInitFieldPos = {
    map<uint32_t, uint32_t> {
        { POOLINIT_FIELD_OPCODE, 28 }, // 0
        { POOLINIT_FIELD_DPDON, 24 }, // 1
        { POOLINIT_FIELD_DPDBY, 20 }, // 2
        { POOLINIT_FIELD_POOL_TYPE, 16 }, // 3
        { POOLINIT_FIELD_KERNEL_H, 8 }, // 4
        { POOLINIT_FIELD_KERNEL_W, 0 }, // 5
    },
    map<uint32_t, uint32_t> {
        { POOLINIT_FIELD_STRIDE_H, 24 }, // 6
        { POOLINIT_FIELD_STRIDE_W, 16 }, // 7
        { POOLINIT_FIELD_STRIDE_OFFSET_IN, 8 }, // 8
        { POOLINIT_FIELD_STRIDE_OFFSET_OUT, 0 }, // 9
    },
    map<uint32_t, uint32_t> {
        { POOLINIT_FIELD_STRIDE_OUT, 23 }, // 10
        { POOLINIT_FIELD_JUMP_READ, 13 }, // 11
        { POOLINIT_FIELD_JUMP_READ_ENDL, 0 }, // 12
    },
    map<uint32_t, uint32_t> {
        { POOLINIT_FIELD_MULTI_FACTOR, 23 }, // 13
        { POOLINIT_FIELD_JUMP_WRITE, 13 }, // 14
        { POOLINIT_FIELD_JUMP_WRITE_ENDL, 0 }, // 15
    },
};


// Pool field position
const vector<map<uint32_t, uint32_t>> InstTable::PoolFieldPos = {
    map<uint32_t, uint32_t> {
        { POOL_FIELD_OPCODE, 28 }, // 0
        { POOL_FIELD_DPDON, 24 }, // 1
        { POOL_FIELD_DPDBY, 20 }, // 2
        { POOL_FIELD_CHANNEL_GROUP, 12 }, // 3
        { POOL_FIELD_SHIFT_CUT, 6 }, // 4
        { POOL_FIELD_BANK_ID_OUT, 0 }, // 5
    },
    map<uint32_t, uint32_t> {
        { POOL_FIELD_PAD_LEFT, 24 }, // 6
        { POOL_FIELD_PAD_RIGHT, 16 }, // 7
        { POOL_FIELD_PAD_TOP, 8 }, // 8
        { POOL_FIELD_PAD_BOTTOM, 0 }, // 9
    },
    map<uint32_t, uint32_t> {
        { POOL_FIELD_BANK_ID_IN, 26 }, // 10
        { POOL_FIELD_VALID_PIXEL_PARALLEL, 23 }, // 11
        { POOL_FIELD_LENGTH, 13 }, // 12
        { POOL_FIELD_BANK_ADDR_OUT, 0 }, // 13
    },
    map<uint32_t, uint32_t> {
        { POOL_FIELD_BANK_ADDR_IN_1, 13 }, // 14
        { POOL_FIELD_BANK_ADDR_IN, 0 }, // 15
    },
    map<uint32_t, uint32_t> {
        { POOL_FIELD_BANK_ADDR_IN_3, 13 }, // 16
        { POOL_FIELD_BANK_ADDR_IN_2, 0 }, // 17
    },
};


// DWInit field position
const vector<map<uint32_t, uint32_t>> InstTable::DWInitFieldPos = {
    map<uint32_t, uint32_t> {
        { DWINIT_FIELD_OPCODE, 28 }, // 0
        { DWINIT_FIELD_DPDON, 24 }, // 1
        { DWINIT_FIELD_DPDBY, 20 }, // 2
        { DWINIT_FIELD_STRIDE_OUT, 16 }, // 3
        { DWINIT_FIELD_STRIDE_H, 12 }, // 4
        { DWINIT_FIELD_STRIDE_W, 8 }, // 5
        { DWINIT_FIELD_KERNEL_H, 4 }, // 6
        { DWINIT_FIELD_KERNEL_W, 0 }, // 7
    },
    map<uint32_t, uint32_t> {
        { DWINIT_FIELD_ACT_TYPE, 28 }, // 8
        { DWINIT_FIELD_CHANNEL_GROUP, 16 }, // 9
        { DWINIT_FIELD_CHANNEL_OFFSET, 10 }, // 10
        { DWINIT_FIELD_LENGTH, 0 }, // 11
    },
    map<uint32_t, uint32_t> {
        { DWINIT_FIELD_SHIFT_BIAS, 26 }, // 12
        { DWINIT_FIELD_STRIDE_OFFSET_IN, 23 }, // 13
        { DWINIT_FIELD_JUMP_READ, 13 }, // 14
        { DWINIT_FIELD_JUMP_READ_ENDL, 0 }, // 15
    },
    map<uint32_t, uint32_t> {
        { DWINIT_FIELD_SHIFT_CUT, 26 }, // 16
        { DWINIT_FIELD_STRIDE_OFFSET_OUT, 23 }, // 17
        { DWINIT_FIELD_JUMP_WRITE, 13 }, // 18
        { DWINIT_FIELD_JUMP_WRITE_ENDL, 0 }, // 19
    },
};


// DptWise field position
const vector<map<uint32_t, uint32_t>> InstTable::DptWiseFieldPos = {
    map<uint32_t, uint32_t> {
        { DPTWISE_FIELD_OPCODE, 28 }, // 0
        { DPTWISE_FIELD_DPDON, 24 }, // 1
        { DPTWISE_FIELD_DPDBY, 20 }, // 2
        { DPTWISE_FIELD_BANK_ADDR_BIAS, 0 }, // 3
    },
    map<uint32_t, uint32_t> {
        { DPTWISE_FIELD_BANK_ADDR_WEIGHTS, 0 }, // 4
    },
    map<uint32_t, uint32_t> {
        { DPTWISE_FIELD_PAD_LEFT, 28 }, // 5
        { DPTWISE_FIELD_PAD_RIGHT, 24 }, // 6
        { DPTWISE_FIELD_PAD_TOP, 20 }, // 7
        { DPTWISE_FIELD_PAD_BOTTOM, 16 }, // 8
        { DPTWISE_FIELD_VALID_PIXEL_PARALLEL, 13 }, // 9
        { DPTWISE_FIELD_BANK_ADDR_OUT, 0 }, // 10
    },
    map<uint32_t, uint32_t> {
        { DPTWISE_FIELD_BANK_ID_IN, 26 }, // 11
        { DPTWISE_FIELD_BANK_ADDR_IN_1, 13 }, // 12
        { DPTWISE_FIELD_BANK_ADDR_IN, 0 }, // 13
    },
    map<uint32_t, uint32_t> {
        { DPTWISE_FIELD_BANK_ID_OUT, 26 }, // 14
        { DPTWISE_FIELD_BANK_ADDR_IN_3, 13 }, // 15
        { DPTWISE_FIELD_BANK_ADDR_IN_2, 0 }, // 16
    },
};


// ElewInit field position
const vector<map<uint32_t, uint32_t>> InstTable::ElewInitFieldPos = {
    map<uint32_t, uint32_t> {
        { ELEWINIT_FIELD_OPCODE, 28 }, // 0
        { ELEWINIT_FIELD_DPDON, 24 }, // 1
        { ELEWINIT_FIELD_DPDBY, 20 }, // 2
        { ELEWINIT_FIELD_SHIFT_READ, 16 }, // 3
        { ELEWINIT_FIELD_BANK_ID_IN, 10 }, // 4
        { ELEWINIT_FIELD_JUMP_READ, 0 }, // 5
    },
    map<uint32_t, uint32_t> {
        { ELEWINIT_FIELD_ID, 26 }, // 6
        { ELEWINIT_FIELD_JUMP_READ_ENDL, 13 }, // 7
        { ELEWINIT_FIELD_BANK_ADDR_IN, 0 }, // 8
    },
};


// Elew field position
const vector<map<uint32_t, uint32_t>> InstTable::ElewFieldPos = {
    map<uint32_t, uint32_t> {
        { ELEW_FIELD_OPCODE, 28 }, // 0
        { ELEW_FIELD_DPDON, 24 }, // 1
        { ELEW_FIELD_DPDBY, 20 }, // 2
        { ELEW_FIELD_BANK_ID_OUT, 10 }, // 3
        { ELEW_FIELD_JUMP_WRITE, 0 }, // 4
    },
    map<uint32_t, uint32_t> {
        { ELEW_FIELD_STRIDE_OUT, 28 }, // 5
        { ELEW_FIELD_NUM, 26 }, // 6
        { ELEW_FIELD_JUMP_WRITE_ENDL, 13 }, // 7
        { ELEW_FIELD_BANK_ADDR_OUT, 0 }, // 8
    },
    map<uint32_t, uint32_t> {
        { ELEW_FIELD_ACT_TYPE, 28 }, // 9
        { ELEW_FIELD_ELEW_TYPE, 26 }, // 10
        { ELEW_FIELD_CHANNEL_GROUP, 16 }, // 11
        { ELEW_FIELD_STRIDE_OFFSET_OUT, 13 }, // 12
        { ELEW_FIELD_VALID_PIXEL_PARALLEL, 10 }, // 13
        { ELEW_FIELD_LENGTH, 0 }, // 14
    },
    map<uint32_t, uint32_t> {
        { ELEW_FIELD_SHIFT_WRITE, 12 }, // 15
        { ELEW_FIELD_HSIGMOID_IN, 6 }, // 16
        { ELEW_FIELD_SHIFT_HSIGMOID, 0 }, // 17
    },
};


// End field position
const vector<map<uint32_t, uint32_t>> InstTable::EndFieldPos = {
    map<uint32_t, uint32_t> {
        { END_FIELD_OPCODE, 28 }, // 0
        { END_FIELD_DPDON, 24 }, // 1
        { END_FIELD_DPDBY, 20 }, // 2
    },
};


// DumpBank field position
const vector<map<uint32_t, uint32_t>> InstTable::DumpBankFieldPos = {
    map<uint32_t, uint32_t> {
        { DUMPBANK_FIELD_OPCODE, 28 }, // 0
        { DUMPBANK_FIELD_DPDON, 24 }, // 1
        { DUMPBANK_FIELD_DPDBY, 20 }, // 2
        { DUMPBANK_FIELD_SAVE_NAME, 0 }, // 3
    },
    map<uint32_t, uint32_t> {
        { DUMPBANK_FIELD_SAVE_FMT, 26 }, // 4
        { DUMPBANK_FIELD_BANK_START, 18 }, // 5
        { DUMPBANK_FIELD_BANK_NUM, 10 }, // 6
    },
};


// DumpDDR field position
const vector<map<uint32_t, uint32_t>> InstTable::DumpDDRFieldPos = {
    map<uint32_t, uint32_t> {
        { DUMPDDR_FIELD_OPCODE, 28 }, // 0
        { DUMPDDR_FIELD_DPDON, 24 }, // 1
        { DUMPDDR_FIELD_DPDBY, 20 }, // 2
        { DUMPDDR_FIELD_SAVE_NAME, 0 }, // 3
    },
    map<uint32_t, uint32_t> {
        { DUMPDDR_FIELD_SAVE_FMT, 26 }, // 4
        { DUMPDDR_FIELD_REG_ID, 20 }, // 5
    },
    map<uint32_t, uint32_t> {
        { DUMPDDR_FIELD_DDR_START, 0 }, // 6
    },
    map<uint32_t, uint32_t> {
        { DUMPDDR_FIELD_DDR_SIZE, 0 }, // 7
    },
};


// DumpDDRSlice field position
const vector<map<uint32_t, uint32_t>> InstTable::DumpDDRSliceFieldPos = {
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_OPCODE, 28 }, // 0
        { DUMPDDRSLICE_FIELD_DPDON, 24 }, // 1
        { DUMPDDRSLICE_FIELD_DPDBY, 20 }, // 2
        { DUMPDDRSLICE_FIELD_SAVE_NAME, 0 }, // 3
    },
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_SAVE_FMT, 26 }, // 4
        { DUMPDDRSLICE_FIELD_REG_ID, 20 }, // 5
    },
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_DDR_START, 0 }, // 6
    },
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_HEIGHT, 16 }, // 7
        { DUMPDDRSLICE_FIELD_HEIGHT_STRIDE, 0 }, // 8
    },
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_WIDTH, 16 }, // 9
        { DUMPDDRSLICE_FIELD_WIDTH_STRIDE, 0 }, // 10
    },
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_CHANNEL, 16 }, // 11
        { DUMPDDRSLICE_FIELD_CHANNEL_STRIDE, 0 }, // 12
    },
};


// Load field length
const vector<map<uint32_t, uint32_t>> InstTable::LoadFieldLen = {
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_OPCODE, 4 }, // 0
        { LOAD_FIELD_DPDON, 4 }, // 1
        { LOAD_FIELD_DPDBY, 4 }, // 2
        { LOAD_FIELD_OUTPUT_CHANNEL_NUM, 7 }, // 3
        { LOAD_FIELD_BANK_ADDR, 13 }, // 4
    },
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_LOAD_MODE, 2 }, // 5
        { LOAD_FIELD_PAD_IDX, 5 }, // 6
        { LOAD_FIELD_DDR_MODE, 1 }, // 7
        { LOAD_FIELD_BANK_ID, 8 }, // 8
        { LOAD_FIELD_JUMP_READ, 16 }, // 9
    },
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_JUMP_WRITE, 8 }, // 10
        { LOAD_FIELD_LENGTH, 10 }, // 11
        { LOAD_FIELD_MODE_AVG, 2 }, // 12
        { LOAD_FIELD_CHANNEL, 12 }, // 13
    },
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_REG_ID, 3 }, // 14
        { LOAD_FIELD_DDR_ADDR, 29 }, // 15
    },
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_PAD_START, 5 }, // 16
        { LOAD_FIELD_PAD_END, 6 }, // 17
        { LOAD_FIELD_JUMP_READ_ENDL, 21 }, // 18
    },
};


// Save field length
const vector<map<uint32_t, uint32_t>> InstTable::SaveFieldLen = {
    map<uint32_t, uint32_t> {
        { SAVE_FIELD_OPCODE, 4 }, // 0
        { SAVE_FIELD_DPDON, 4 }, // 1
        { SAVE_FIELD_DPDBY, 4 }, // 2
        { SAVE_FIELD_BANK_ADDR, 13 }, // 3
    },
    map<uint32_t, uint32_t> {
        { SAVE_FIELD_BANK_ID, 8 }, // 4
        { SAVE_FIELD_JUMP_WRITE, 16 }, // 5
    },
    map<uint32_t, uint32_t> {
        { SAVE_FIELD_JUMP_READ, 8 }, // 6
        { SAVE_FIELD_LENGTH, 10 }, // 7
        { SAVE_FIELD_CHANNEL, 12 }, // 8
    },
    map<uint32_t, uint32_t> {
        { SAVE_FIELD_REG_ID, 3 }, // 9
        { SAVE_FIELD_DDR_ADDR, 29 }, // 10
    },
};


// ConvInit field length
const vector<map<uint32_t, uint32_t>> InstTable::ConvInitFieldLen = {
    map<uint32_t, uint32_t> {
        { CONVINIT_FIELD_OPCODE, 4 }, // 0
        { CONVINIT_FIELD_DPDON, 4 }, // 1
        { CONVINIT_FIELD_DPDBY, 4 }, // 2
        { CONVINIT_FIELD_STRIDE_OUT, 4 }, // 3
        { CONVINIT_FIELD_STRIDE_H, 4 }, // 4
        { CONVINIT_FIELD_STRIDE_W, 4 }, // 5
        { CONVINIT_FIELD_KERNEL_H, 4 }, // 6
        { CONVINIT_FIELD_KERNEL_W, 4 }, // 7
    },
    map<uint32_t, uint32_t> {
        { CONVINIT_FIELD_ACT_TYPE, 4 }, // 8
        { CONVINIT_FIELD_SHIFT_HSWISH, 4 }, // 9
        { CONVINIT_FIELD_CHANNEL_GROUP, 8 }, // 10
        { CONVINIT_FIELD_CHANNEL_OFFSET, 6 }, // 11
        { CONVINIT_FIELD_LENGTH, 10 }, // 12
    },
    map<uint32_t, uint32_t> {
        { CONVINIT_FIELD_SHIFT_BIAS, 6 }, // 13
        { CONVINIT_FIELD_STRIDE_OFFSET_IN, 3 }, // 14
        { CONVINIT_FIELD_JUMP_READ, 10 }, // 15
        { CONVINIT_FIELD_JUMP_READ_ENDL, 13 }, // 16
    },
    map<uint32_t, uint32_t> {
        { CONVINIT_FIELD_SHIFT_CUT, 6 }, // 17
        { CONVINIT_FIELD_STRIDE_OFFSET_OUT, 3 }, // 18
        { CONVINIT_FIELD_JUMP_WRITE, 10 }, // 19
        { CONVINIT_FIELD_JUMP_WRITE_ENDL, 13 }, // 20
    },
    map<uint32_t, uint32_t> {
        { CONVINIT_FIELD_SHIFT_HSIGMOID, 5 }, // 21
        { CONVINIT_FIELD_HSIGMOID_IN, 4 }, // 22
        { CONVINIT_FIELD_BATCH_NUM, 2 }, // 23
        { CONVINIT_FIELD_STRIDE_BATCH, 10 }, // 24
    },
};


// Conv field length
const vector<map<uint32_t, uint32_t>> InstTable::ConvFieldLen = {
    map<uint32_t, uint32_t> {
        { CONV_FIELD_OPCODE, 4 }, // 0
        { CONV_FIELD_DPDON, 4 }, // 1
        { CONV_FIELD_DPDBY, 4 }, // 2
        { CONV_FIELD_BANK_ADDR_BIAS, 12 }, // 3
    },
    map<uint32_t, uint32_t> {
        { CONV_FIELD_BANK_ADDR_WEIGHTS, 12 }, // 4
    },
    map<uint32_t, uint32_t> {
        { CONV_FIELD_PAD_LEFT, 4 }, // 5
        { CONV_FIELD_PAD_RIGHT, 4 }, // 6
        { CONV_FIELD_PAD_TOP, 4 }, // 7
        { CONV_FIELD_PAD_BOTTOM, 4 }, // 8
        { CONV_FIELD_VALID_PIXEL_PARALLEL, 3 }, // 9
        { CONV_FIELD_BANK_ADDR_OUT, 13 }, // 10
    },
    map<uint32_t, uint32_t> {
        { CONV_FIELD_BANK_ID_IN, 6 }, // 11
        { CONV_FIELD_BANK_ADDR_IN_1, 13 }, // 12
        { CONV_FIELD_BANK_ADDR_IN, 13 }, // 13
    },
    map<uint32_t, uint32_t> {
        { CONV_FIELD_BANK_ID_OUT, 6 }, // 14
        { CONV_FIELD_BANK_ADDR_IN_3, 13 }, // 15
        { CONV_FIELD_BANK_ADDR_IN_2, 13 }, // 16
    },
};


// PoolInit field length
const vector<map<uint32_t, uint32_t>> InstTable::PoolInitFieldLen = {
    map<uint32_t, uint32_t> {
        { POOLINIT_FIELD_OPCODE, 4 }, // 0
        { POOLINIT_FIELD_DPDON, 4 }, // 1
        { POOLINIT_FIELD_DPDBY, 4 }, // 2
        { POOLINIT_FIELD_POOL_TYPE, 4 }, // 3
        { POOLINIT_FIELD_KERNEL_H, 8 }, // 4
        { POOLINIT_FIELD_KERNEL_W, 8 }, // 5
    },
    map<uint32_t, uint32_t> {
        { POOLINIT_FIELD_STRIDE_H, 8 }, // 6
        { POOLINIT_FIELD_STRIDE_W, 8 }, // 7
        { POOLINIT_FIELD_STRIDE_OFFSET_IN, 8 }, // 8
        { POOLINIT_FIELD_STRIDE_OFFSET_OUT, 8 }, // 9
    },
    map<uint32_t, uint32_t> {
        { POOLINIT_FIELD_STRIDE_OUT, 8 }, // 10
        { POOLINIT_FIELD_JUMP_READ, 10 }, // 11
        { POOLINIT_FIELD_JUMP_READ_ENDL, 13 }, // 12
    },
    map<uint32_t, uint32_t> {
        { POOLINIT_FIELD_MULTI_FACTOR, 8 }, // 13
        { POOLINIT_FIELD_JUMP_WRITE, 10 }, // 14
        { POOLINIT_FIELD_JUMP_WRITE_ENDL, 13 }, // 15
    },
};


// Pool field length
const vector<map<uint32_t, uint32_t>> InstTable::PoolFieldLen = {
    map<uint32_t, uint32_t> {
        { POOL_FIELD_OPCODE, 4 }, // 0
        { POOL_FIELD_DPDON, 4 }, // 1
        { POOL_FIELD_DPDBY, 4 }, // 2
        { POOL_FIELD_CHANNEL_GROUP, 8 }, // 3
        { POOL_FIELD_SHIFT_CUT, 6 }, // 4
        { POOL_FIELD_BANK_ID_OUT, 6 }, // 5
    },
    map<uint32_t, uint32_t> {
        { POOL_FIELD_PAD_LEFT, 8 }, // 6
        { POOL_FIELD_PAD_RIGHT, 8 }, // 7
        { POOL_FIELD_PAD_TOP, 8 }, // 8
        { POOL_FIELD_PAD_BOTTOM, 8 }, // 9
    },
    map<uint32_t, uint32_t> {
        { POOL_FIELD_BANK_ID_IN, 6 }, // 10
        { POOL_FIELD_VALID_PIXEL_PARALLEL, 3 }, // 11
        { POOL_FIELD_LENGTH, 10 }, // 12
        { POOL_FIELD_BANK_ADDR_OUT, 13 }, // 13
    },
    map<uint32_t, uint32_t> {
        { POOL_FIELD_BANK_ADDR_IN_1, 13 }, // 14
        { POOL_FIELD_BANK_ADDR_IN, 13 }, // 15
    },
    map<uint32_t, uint32_t> {
        { POOL_FIELD_BANK_ADDR_IN_3, 13 }, // 16
        { POOL_FIELD_BANK_ADDR_IN_2, 13 }, // 17
    },
};


// DWInit field length
const vector<map<uint32_t, uint32_t>> InstTable::DWInitFieldLen = {
    map<uint32_t, uint32_t> {
        { DWINIT_FIELD_OPCODE, 4 }, // 0
        { DWINIT_FIELD_DPDON, 4 }, // 1
        { DWINIT_FIELD_DPDBY, 4 }, // 2
        { DWINIT_FIELD_STRIDE_OUT, 4 }, // 3
        { DWINIT_FIELD_STRIDE_H, 4 }, // 4
        { DWINIT_FIELD_STRIDE_W, 4 }, // 5
        { DWINIT_FIELD_KERNEL_H, 4 }, // 6
        { DWINIT_FIELD_KERNEL_W, 4 }, // 7
    },
    map<uint32_t, uint32_t> {
        { DWINIT_FIELD_ACT_TYPE, 4 }, // 8
        { DWINIT_FIELD_CHANNEL_GROUP, 8 }, // 9
        { DWINIT_FIELD_CHANNEL_OFFSET, 6 }, // 10
        { DWINIT_FIELD_LENGTH, 10 }, // 11
    },
    map<uint32_t, uint32_t> {
        { DWINIT_FIELD_SHIFT_BIAS, 6 }, // 12
        { DWINIT_FIELD_STRIDE_OFFSET_IN, 3 }, // 13
        { DWINIT_FIELD_JUMP_READ, 10 }, // 14
        { DWINIT_FIELD_JUMP_READ_ENDL, 13 }, // 15
    },
    map<uint32_t, uint32_t> {
        { DWINIT_FIELD_SHIFT_CUT, 6 }, // 16
        { DWINIT_FIELD_STRIDE_OFFSET_OUT, 3 }, // 17
        { DWINIT_FIELD_JUMP_WRITE, 10 }, // 18
        { DWINIT_FIELD_JUMP_WRITE_ENDL, 13 }, // 19
    },
};


// DptWise field length
const vector<map<uint32_t, uint32_t>> InstTable::DptWiseFieldLen = {
    map<uint32_t, uint32_t> {
        { DPTWISE_FIELD_OPCODE, 4 }, // 0
        { DPTWISE_FIELD_DPDON, 4 }, // 1
        { DPTWISE_FIELD_DPDBY, 4 }, // 2
        { DPTWISE_FIELD_BANK_ADDR_BIAS, 12 }, // 3
    },
    map<uint32_t, uint32_t> {
        { DPTWISE_FIELD_BANK_ADDR_WEIGHTS, 13 }, // 4
    },
    map<uint32_t, uint32_t> {
        { DPTWISE_FIELD_PAD_LEFT, 4 }, // 5
        { DPTWISE_FIELD_PAD_RIGHT, 4 }, // 6
        { DPTWISE_FIELD_PAD_TOP, 4 }, // 7
        { DPTWISE_FIELD_PAD_BOTTOM, 4 }, // 8
        { DPTWISE_FIELD_VALID_PIXEL_PARALLEL, 3 }, // 9
        { DPTWISE_FIELD_BANK_ADDR_OUT, 13 }, // 10
    },
    map<uint32_t, uint32_t> {
        { DPTWISE_FIELD_BANK_ID_IN, 6 }, // 11
        { DPTWISE_FIELD_BANK_ADDR_IN_1, 13 }, // 12
        { DPTWISE_FIELD_BANK_ADDR_IN, 13 }, // 13
    },
    map<uint32_t, uint32_t> {
        { DPTWISE_FIELD_BANK_ID_OUT, 6 }, // 14
        { DPTWISE_FIELD_BANK_ADDR_IN_3, 13 }, // 15
        { DPTWISE_FIELD_BANK_ADDR_IN_2, 13 }, // 16
    },
};


// ElewInit field length
const vector<map<uint32_t, uint32_t>> InstTable::ElewInitFieldLen = {
    map<uint32_t, uint32_t> {
        { ELEWINIT_FIELD_OPCODE, 4 }, // 0
        { ELEWINIT_FIELD_DPDON, 4 }, // 1
        { ELEWINIT_FIELD_DPDBY, 4 }, // 2
        { ELEWINIT_FIELD_SHIFT_READ, 4 }, // 3
        { ELEWINIT_FIELD_BANK_ID_IN, 6 }, // 4
        { ELEWINIT_FIELD_JUMP_READ, 10 }, // 5
    },
    map<uint32_t, uint32_t> {
        { ELEWINIT_FIELD_ID, 2 }, // 6
        { ELEWINIT_FIELD_JUMP_READ_ENDL, 13 }, // 7
        { ELEWINIT_FIELD_BANK_ADDR_IN, 13 }, // 8
    },
};


// Elew field length
const vector<map<uint32_t, uint32_t>> InstTable::ElewFieldLen = {
    map<uint32_t, uint32_t> {
        { ELEW_FIELD_OPCODE, 4 }, // 0
        { ELEW_FIELD_DPDON, 4 }, // 1
        { ELEW_FIELD_DPDBY, 4 }, // 2
        { ELEW_FIELD_BANK_ID_OUT, 6 }, // 3
        { ELEW_FIELD_JUMP_WRITE, 10 }, // 4
    },
    map<uint32_t, uint32_t> {
        { ELEW_FIELD_STRIDE_OUT, 4 }, // 5
        { ELEW_FIELD_NUM, 2 }, // 6
        { ELEW_FIELD_JUMP_WRITE_ENDL, 13 }, // 7
        { ELEW_FIELD_BANK_ADDR_OUT, 13 }, // 8
    },
    map<uint32_t, uint32_t> {
        { ELEW_FIELD_ACT_TYPE, 4 }, // 9
        { ELEW_FIELD_ELEW_TYPE, 2 }, // 10
        { ELEW_FIELD_CHANNEL_GROUP, 8 }, // 11
        { ELEW_FIELD_STRIDE_OFFSET_OUT, 3 }, // 12
        { ELEW_FIELD_VALID_PIXEL_PARALLEL, 3 }, // 13
        { ELEW_FIELD_LENGTH, 10 }, // 14
    },
    map<uint32_t, uint32_t> {
        { ELEW_FIELD_SHIFT_WRITE, 5 }, // 15
        { ELEW_FIELD_HSIGMOID_IN, 6 }, // 16
        { ELEW_FIELD_SHIFT_HSIGMOID, 6 }, // 17
    },
};


// End field length
const vector<map<uint32_t, uint32_t>> InstTable::EndFieldLen = {
    map<uint32_t, uint32_t> {
        { END_FIELD_OPCODE, 4 }, // 0
        { END_FIELD_DPDON, 4 }, // 1
        { END_FIELD_DPDBY, 4 }, // 2
    },
};


// DumpBank field length
const vector<map<uint32_t, uint32_t>> InstTable::DumpBankFieldLen = {
    map<uint32_t, uint32_t> {
        { DUMPBANK_FIELD_OPCODE, 4 }, // 0
        { DUMPBANK_FIELD_DPDON, 4 }, // 1
        { DUMPBANK_FIELD_DPDBY, 4 }, // 2
        { DUMPBANK_FIELD_SAVE_NAME, 20 }, // 3
    },
    map<uint32_t, uint32_t> {
        { DUMPBANK_FIELD_SAVE_FMT, 6 }, // 4
        { DUMPBANK_FIELD_BANK_START, 8 }, // 5
        { DUMPBANK_FIELD_BANK_NUM, 8 }, // 6
    },
};


// DumpDDR field length
const vector<map<uint32_t, uint32_t>> InstTable::DumpDDRFieldLen = {
    map<uint32_t, uint32_t> {
        { DUMPDDR_FIELD_OPCODE, 4 }, // 0
        { DUMPDDR_FIELD_DPDON, 4 }, // 1
        { DUMPDDR_FIELD_DPDBY, 4 }, // 2
        { DUMPDDR_FIELD_SAVE_NAME, 20 }, // 3
    },
    map<uint32_t, uint32_t> {
        { DUMPDDR_FIELD_SAVE_FMT, 6 }, // 4
        { DUMPDDR_FIELD_REG_ID, 6 }, // 5
    },
    map<uint32_t, uint32_t> {
        { DUMPDDR_FIELD_DDR_START, 32 }, // 6
    },
    map<uint32_t, uint32_t> {
        { DUMPDDR_FIELD_DDR_SIZE, 32 }, // 7
    },
};


// DumpDDRSlice field length
const vector<map<uint32_t, uint32_t>> InstTable::DumpDDRSliceFieldLen = {
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_OPCODE, 4 }, // 0
        { DUMPDDRSLICE_FIELD_DPDON, 4 }, // 1
        { DUMPDDRSLICE_FIELD_DPDBY, 4 }, // 2
        { DUMPDDRSLICE_FIELD_SAVE_NAME, 20 }, // 3
    },
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_SAVE_FMT, 6 }, // 4
        { DUMPDDRSLICE_FIELD_REG_ID, 6 }, // 5
    },
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_DDR_START, 32 }, // 6
    },
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_HEIGHT, 16 }, // 7
        { DUMPDDRSLICE_FIELD_HEIGHT_STRIDE, 16 }, // 8
    },
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_WIDTH, 16 }, // 9
        { DUMPDDRSLICE_FIELD_WIDTH_STRIDE, 16 }, // 10
    },
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_CHANNEL, 16 }, // 11
        { DUMPDDRSLICE_FIELD_CHANNEL_STRIDE, 16 }, // 12
    },
};


// inst opcode to inst type map
const unordered_map<uint32_t, uint32_t> InstTable::OPCode2InstType = {
    { 0x00, INST_TYPE_LOAD }, // 0
    { 0x04, INST_TYPE_SAVE }, // 1
    { 0x09, INST_TYPE_CONVINIT }, // 2
    { 0x08, INST_TYPE_CONV }, // 3
    { 0x06, INST_TYPE_POOLINIT }, // 4
    { 0x0C, INST_TYPE_POOL }, // 5
    { 0x0B, INST_TYPE_DWINIT }, // 6
    { 0x0A, INST_TYPE_DPTWISE }, // 7
    { 0x0D, INST_TYPE_ELEWINIT }, // 8
    { 0x0E, INST_TYPE_ELEW }, // 9
    { 0x07, INST_TYPE_END }, // 10
    { 0xFF, INST_TYPE_DUMPBANK }, // 11
    { 0xFE, INST_TYPE_DUMPDDR }, // 12
    { 0xFD, INST_TYPE_DUMPDDRSLICE }, // 13
};


// inst name to inst type map
const unordered_map<string, uint32_t> InstTable::InstName2InstType = {
    { "LOAD", INST_TYPE_LOAD }, // 0
    { "SAVE", INST_TYPE_SAVE }, // 1
    { "CONVINIT", INST_TYPE_CONVINIT }, // 2
    { "CONV", INST_TYPE_CONV }, // 3
    { "POOLINIT", INST_TYPE_POOLINIT }, // 4
    { "POOL", INST_TYPE_POOL }, // 5
    { "DWINIT", INST_TYPE_DWINIT }, // 6
    { "DPTWISE", INST_TYPE_DPTWISE }, // 7
    { "ELEWINIT", INST_TYPE_ELEWINIT }, // 8
    { "ELEW", INST_TYPE_ELEW }, // 9
    { "END", INST_TYPE_END }, // 10
    { "DUMPBANK", INST_TYPE_DUMPBANK }, // 11
    { "DUMPDDR", INST_TYPE_DUMPDDR }, // 12
    { "DUMPDDRSLICE", INST_TYPE_DUMPDDRSLICE }, // 13
};


// inst type to field name map
const unordered_map<uint32_t, const vector<string> &> InstTable::InstFieldName = {
    { INST_TYPE_LOAD, LoadFieldName }, // 0
    { INST_TYPE_SAVE, SaveFieldName }, // 1
    { INST_TYPE_CONVINIT, ConvInitFieldName }, // 2
    { INST_TYPE_CONV, ConvFieldName }, // 3
    { INST_TYPE_POOLINIT, PoolInitFieldName }, // 4
    { INST_TYPE_POOL, PoolFieldName }, // 5
    { INST_TYPE_DWINIT, DWInitFieldName }, // 6
    { INST_TYPE_DPTWISE, DptWiseFieldName }, // 7
    { INST_TYPE_ELEWINIT, ElewInitFieldName }, // 8
    { INST_TYPE_ELEW, ElewFieldName }, // 9
    { INST_TYPE_END, EndFieldName }, // 10
    { INST_TYPE_DUMPBANK, DumpBankFieldName }, // 11
    { INST_TYPE_DUMPDDR, DumpDDRFieldName }, // 12
    { INST_TYPE_DUMPDDRSLICE, DumpDDRSliceFieldName }, // 13
};


// inst type to field data type map
const unordered_map<uint32_t, const vector<int> &> InstTable::InstFieldDataType = {
    { INST_TYPE_LOAD, LoadFieldDataType }, // 0
    { INST_TYPE_SAVE, SaveFieldDataType }, // 1
    { INST_TYPE_CONVINIT, ConvInitFieldDataType }, // 2
    { INST_TYPE_CONV, ConvFieldDataType }, // 3
    { INST_TYPE_POOLINIT, PoolInitFieldDataType }, // 4
    { INST_TYPE_POOL, PoolFieldDataType }, // 5
    { INST_TYPE_DWINIT, DWInitFieldDataType }, // 6
    { INST_TYPE_DPTWISE, DptWiseFieldDataType }, // 7
    { INST_TYPE_ELEWINIT, ElewInitFieldDataType }, // 8
    { INST_TYPE_ELEW, ElewFieldDataType }, // 9
    { INST_TYPE_END, EndFieldDataType }, // 10
    { INST_TYPE_DUMPBANK, DumpBankFieldDataType }, // 11
    { INST_TYPE_DUMPDDR, DumpDDRFieldDataType }, // 12
    { INST_TYPE_DUMPDDRSLICE, DumpDDRSliceFieldDataType }, // 13
};


// inst type to field minus map
const unordered_map<uint32_t, const vector<int> &> InstTable::InstFieldMinus = {
    { INST_TYPE_LOAD, LoadFieldMinus }, // 0
    { INST_TYPE_SAVE, SaveFieldMinus }, // 1
    { INST_TYPE_CONVINIT, ConvInitFieldMinus }, // 2
    { INST_TYPE_CONV, ConvFieldMinus }, // 3
    { INST_TYPE_POOLINIT, PoolInitFieldMinus }, // 4
    { INST_TYPE_POOL, PoolFieldMinus }, // 5
    { INST_TYPE_DWINIT, DWInitFieldMinus }, // 6
    { INST_TYPE_DPTWISE, DptWiseFieldMinus }, // 7
    { INST_TYPE_ELEWINIT, ElewInitFieldMinus }, // 8
    { INST_TYPE_ELEW, ElewFieldMinus }, // 9
    { INST_TYPE_END, EndFieldMinus }, // 10
    { INST_TYPE_DUMPBANK, DumpBankFieldMinus }, // 11
    { INST_TYPE_DUMPDDR, DumpDDRFieldMinus }, // 12
    { INST_TYPE_DUMPDDRSLICE, DumpDDRSliceFieldMinus }, // 13
};


// inst type to field position map
const unordered_map<uint32_t, const vector<map<uint32_t, uint32_t>> &> InstTable::InstFieldPos = {
    { INST_TYPE_LOAD, LoadFieldPos }, // 0
    { INST_TYPE_SAVE, SaveFieldPos }, // 1
    { INST_TYPE_CONVINIT, ConvInitFieldPos }, // 2
    { INST_TYPE_CONV, ConvFieldPos }, // 3
    { INST_TYPE_POOLINIT, PoolInitFieldPos }, // 4
    { INST_TYPE_POOL, PoolFieldPos }, // 5
    { INST_TYPE_DWINIT, DWInitFieldPos }, // 6
    { INST_TYPE_DPTWISE, DptWiseFieldPos }, // 7
    { INST_TYPE_ELEWINIT, ElewInitFieldPos }, // 8
    { INST_TYPE_ELEW, ElewFieldPos }, // 9
    { INST_TYPE_END, EndFieldPos }, // 10
    { INST_TYPE_DUMPBANK, DumpBankFieldPos }, // 11
    { INST_TYPE_DUMPDDR, DumpDDRFieldPos }, // 12
    { INST_TYPE_DUMPDDRSLICE, DumpDDRSliceFieldPos }, // 13
};


// inst type to field length map
const unordered_map<uint32_t, const vector<map<uint32_t, uint32_t>> &> InstTable::InstFieldLen = {
    { INST_TYPE_LOAD, LoadFieldLen }, // 0
    { INST_TYPE_SAVE, SaveFieldLen }, // 1
    { INST_TYPE_CONVINIT, ConvInitFieldLen }, // 2
    { INST_TYPE_CONV, ConvFieldLen }, // 3
    { INST_TYPE_POOLINIT, PoolInitFieldLen }, // 4
    { INST_TYPE_POOL, PoolFieldLen }, // 5
    { INST_TYPE_DWINIT, DWInitFieldLen }, // 6
    { INST_TYPE_DPTWISE, DptWiseFieldLen }, // 7
    { INST_TYPE_ELEWINIT, ElewInitFieldLen }, // 8
    { INST_TYPE_ELEW, ElewFieldLen }, // 9
    { INST_TYPE_END, EndFieldLen }, // 10
    { INST_TYPE_DUMPBANK, DumpBankFieldLen }, // 11
    { INST_TYPE_DUMPDDR, DumpDDRFieldLen }, // 12
    { INST_TYPE_DUMPDDRSLICE, DumpDDRSliceFieldLen }, // 13
};


}

namespace XV_2DPU {
// instruction version
const string InstTable::INST_VERSION = "xv2dpu";


// category name vector
const vector<string> InstTable::CategoryName = {
    "LOAD",
    "SAVE",
    "CONV",
    "MISC",
};


// inst name vector
const vector<string> InstTable::InstName = {
    "LOAD", // 0
    "SAVE", // 1
    "CONVINIT", // 2
    "CONVADDR", // 3
    "CONV", // 4
    "ALUINIT", // 5
    "ALUADDR", // 6
    "ALU", // 7
    "END", // 8
    "DUMPBANK", // 9
    "DUMPDDR", // 10
    "DUMPDDRSLICE", // 11
};


// lower inst name vector
const vector<string> InstTable::InstNameLower = {
    "load", // 0
    "save", // 1
    "convinit", // 2
    "convaddr", // 3
    "conv", // 4
    "aluinit", // 5
    "aluaddr", // 6
    "alu", // 7
    "end", // 8
    "dumpbank", // 9
    "dumpddr", // 10
    "dumpddrslice", // 11
};


// inst category
const vector<Category> InstTable::InstCategory = {
    Category::INST_CATEGORY_LOAD, // LOAD
    Category::INST_CATEGORY_SAVE, // SAVE
    Category::INST_CATEGORY_CONV, // CONVINIT
    Category::INST_CATEGORY_CONV, // CONVADDR
    Category::INST_CATEGORY_CONV, // CONV
    Category::INST_CATEGORY_MISC, // ALUINIT
    Category::INST_CATEGORY_MISC, // ALUADDR
    Category::INST_CATEGORY_MISC, // ALU
    Category::INST_CATEGORY_SAVE, // END
    Category::INST_CATEGORY_MISC, // DUMPBANK
    Category::INST_CATEGORY_MISC, // DUMPDDR
    Category::INST_CATEGORY_MISC, // DUMPDDRSLICE
};


// inst word number
const vector<uint32_t> InstTable::WordNum = {
    6, // LOAD
    4, // SAVE
    7, // CONVINIT
    3, // CONVADDR
    2, // CONV
    8, // ALUINIT
    3, // ALUADDR
    2, // ALU
    1, // END
    2, // DUMPBANK
    4, // DUMPDDR
    6, // DUMPDDRSLICE
};


// inst field number
const vector<uint32_t> InstTable::FieldNum = {
    LOAD_FIELD_MAX, // 0
    SAVE_FIELD_MAX, // 1
    CONVINIT_FIELD_MAX, // 2
    CONVADDR_FIELD_MAX, // 3
    CONV_FIELD_MAX, // 4
    ALUINIT_FIELD_MAX, // 5
    ALUADDR_FIELD_MAX, // 6
    ALU_FIELD_MAX, // 7
    END_FIELD_MAX, // 8
    DUMPBANK_FIELD_MAX, // 9
    DUMPDDR_FIELD_MAX, // 10
    DUMPDDRSLICE_FIELD_MAX, // 11
};


// inst opcode
const vector<uint32_t> InstTable::OPCode = {
    0x00, // LOAD
    0x04, // SAVE
    0x09, // CONVINIT
    0x05, // CONVADDR
    0x08, // CONV
    0x01, // ALUINIT
    0x03, // ALUADDR
    0x02, // ALU
    0x07, // END
    0xFF, // DUMPBANK
    0xFE, // DUMPDDR
    0xFD, // DUMPDDRSLICE
};


// Load field name
const vector<string> InstTable::LoadFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "mt_dst", // 3
    "mt_addr", // 4
    "block_num", // 5
    "broadcast", // 6
    "pad_start", // 7
    "pad_end", // 8
    "pad_idx", // 9
    "reg_id", // 10
    "const_value", // 11
    "jump_read", // 12
    "mode_avg", // 13
    "length", // 14
    "channel", // 15
    "ddr_addr", // 16
    "jump_write", // 17
    "jump_write_endl", // 18
};


// Save field name
const vector<string> InstTable::SaveFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "argmax", // 3
    "const_en", // 4
    "mt_addr", // 5
    "reg_id", // 6
    "const_value", // 7
    "jump_write", // 8
    "jump_read", // 9
    "length", // 10
    "channel", // 11
    "ddr_addr", // 12
};


// ConvInit field name
const vector<string> InstTable::ConvInitFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "exec_mode", // 3
    "kernel_h", // 4
    "kernel_w", // 5
    "stride_h", // 6
    "stride_w", // 7
    "total_in", // 8
    "tile_icg", // 9
    "tile_ocg", // 10
    "tile_ohg", // 11
    "tile_owg", // 12
    "ic_iter", // 13
    "ow_iter", // 14
    "ow_offset", // 15
    "shift_hswish", // 16
    "shift_hsigmoid", // 17
    "hsigmoid_in", // 18
    "kw_iter", // 19
    "kh_iter", // 20
    "act_type", // 21
    "aie_mode", // 22
    "conv_num", // 23
    "shift_bias", // 24
    "shift_cut", // 25
    "one_height", // 26
    "one_width", // 27
    "one_cube", // 28
    "tile_en", // 29
    "total_tile", // 30
    "one_size", // 31
    "shift_elew0", // 32
    "shift_elew1", // 33
    "elew_cut", // 34
    "elew_act_type", // 35
};


// ConvAddr field name
const vector<string> InstTable::ConvAddrFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "type", // 3
    "last", // 4
    "compression", // 5
    "invalid", // 6
    "h_num", // 7
    "mt_addr", // 8
    "jump", // 9
    "jump_endl", // 10
};


// Conv field name
const vector<string> InstTable::ConvFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "icg_offset", // 3
    "ocg_offset", // 4
    "pad_right", // 5
    "pad_left", // 6
    "pad_bottom", // 7
    "pad_top", // 8
    "reuse", // 9
    "macc_cont", // 10
};


// AluInit field name
const vector<string> InstTable::AluInitFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "exec_mode", // 3
    "kernel_h", // 4
    "kernel_w", // 5
    "stride_h", // 6
    "stride_w", // 7
    "b_mode", // 8
    "num", // 9
    "share_kernel", // 10
    "share_channel_group", // 11
    "tile_owg", // 12
    "tile_ohg", // 13
    "pad_type", // 14
    "tile_cg", // 15
    "ow_iter", // 16
    "ow_offset", // 17
    "shift_hswish", // 18
    "shift_hsigmoid", // 19
    "hsigmoid_in", // 20
    "act_type", // 21
    "shift_bias", // 22
    "shift_cut", // 23
    "alu_num", // 24
    "weights_lines", // 25
    "shift_read_3", // 26
    "shift_read_2", // 27
    "shift_read_1", // 28
    "shift_read_0", // 29
    "total_tile", // 30
    "one_height", // 31
    "one_width", // 32
    "kh_iter", // 33
    "incAO3", // 34
    "kw_iter", // 35
    "incAO2", // 36
};


// AluAddr field name
const vector<string> InstTable::AluAddrFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "type", // 3
    "last", // 4
    "compression", // 5
    "invalid", // 6
    "h_num", // 7
    "macc_dim", // 8
    "mt_addr", // 9
    "id", // 10
    "jump", // 11
    "jump_endl", // 12
};


// Alu field name
const vector<string> InstTable::AluFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "reuse", // 3
    "macc_cont", // 4
    "pad_left", // 5
    "pad_top", // 6
    "pad_right_mt", // 7
    "pad_bottom_mt", // 8
    "pad_bottom_at", // 9
    "pad_right_at", // 10
};


// End field name
const vector<string> InstTable::EndFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
};


// DumpBank field name
const vector<string> InstTable::DumpBankFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "save_name", // 3
    "save_fmt", // 4
    "bank_start", // 5
    "bank_num", // 6
};


// DumpDDR field name
const vector<string> InstTable::DumpDDRFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "save_name", // 3
    "save_fmt", // 4
    "reg_id", // 5
    "ddr_start", // 6
    "ddr_size", // 7
};


// DumpDDRSlice field name
const vector<string> InstTable::DumpDDRSliceFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "save_name", // 3
    "save_fmt", // 4
    "reg_id", // 5
    "ddr_start", // 6
    "height", // 7
    "height_stride", // 8
    "width", // 9
    "width_stride", // 10
    "channel", // 11
    "channel_stride", // 12
};


// Load field data type
const vector<int> InstTable::LoadFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_UINT, // mt_dst(3)
    INST_FIELD_DATA_TYPE_UINT, // mt_addr(4)
    INST_FIELD_DATA_TYPE_UINT, // block_num(5)
    INST_FIELD_DATA_TYPE_UINT, // broadcast(6)
    INST_FIELD_DATA_TYPE_UINT, // pad_start(7)
    INST_FIELD_DATA_TYPE_UINT, // pad_end(8)
    INST_FIELD_DATA_TYPE_UINT, // pad_idx(9)
    INST_FIELD_DATA_TYPE_UINT, // reg_id(10)
    INST_FIELD_DATA_TYPE_UINT, // const_value(11)
    INST_FIELD_DATA_TYPE_UINT, // jump_read(12)
    INST_FIELD_DATA_TYPE_UINT, // mode_avg(13)
    INST_FIELD_DATA_TYPE_UINT, // length(14)
    INST_FIELD_DATA_TYPE_UINT, // channel(15)
    INST_FIELD_DATA_TYPE_UINT, // ddr_addr(16)
    INST_FIELD_DATA_TYPE_UINT, // jump_write(17)
    INST_FIELD_DATA_TYPE_UINT, // jump_write_endl(18)
};


// Save field data type
const vector<int> InstTable::SaveFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_UINT, // argmax(3)
    INST_FIELD_DATA_TYPE_UINT, // const_en(4)
    INST_FIELD_DATA_TYPE_UINT, // mt_addr(5)
    INST_FIELD_DATA_TYPE_UINT, // reg_id(6)
    INST_FIELD_DATA_TYPE_UINT, // const_value(7)
    INST_FIELD_DATA_TYPE_UINT, // jump_write(8)
    INST_FIELD_DATA_TYPE_UINT, // jump_read(9)
    INST_FIELD_DATA_TYPE_UINT, // length(10)
    INST_FIELD_DATA_TYPE_UINT, // channel(11)
    INST_FIELD_DATA_TYPE_UINT, // ddr_addr(12)
};


// ConvInit field data type
const vector<int> InstTable::ConvInitFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_UINT, // exec_mode(3)
    INST_FIELD_DATA_TYPE_UINT, // kernel_h(4)
    INST_FIELD_DATA_TYPE_UINT, // kernel_w(5)
    INST_FIELD_DATA_TYPE_UINT, // stride_h(6)
    INST_FIELD_DATA_TYPE_UINT, // stride_w(7)
    INST_FIELD_DATA_TYPE_UINT, // total_in(8)
    INST_FIELD_DATA_TYPE_UINT, // tile_icg(9)
    INST_FIELD_DATA_TYPE_UINT, // tile_ocg(10)
    INST_FIELD_DATA_TYPE_UINT, // tile_ohg(11)
    INST_FIELD_DATA_TYPE_UINT, // tile_owg(12)
    INST_FIELD_DATA_TYPE_UINT, // ic_iter(13)
    INST_FIELD_DATA_TYPE_UINT, // ow_iter(14)
    INST_FIELD_DATA_TYPE_UINT, // ow_offset(15)
    INST_FIELD_DATA_TYPE_UINT, // shift_hswish(16)
    INST_FIELD_DATA_TYPE_UINT, // shift_hsigmoid(17)
    INST_FIELD_DATA_TYPE_UINT, // hsigmoid_in(18)
    INST_FIELD_DATA_TYPE_UINT, // kw_iter(19)
    INST_FIELD_DATA_TYPE_UINT, // kh_iter(20)
    INST_FIELD_DATA_TYPE_UINT, // act_type(21)
    INST_FIELD_DATA_TYPE_UINT, // aie_mode(22)
    INST_FIELD_DATA_TYPE_UINT, // conv_num(23)
    INST_FIELD_DATA_TYPE_UINT, // shift_bias(24)
    INST_FIELD_DATA_TYPE_UINT, // shift_cut(25)
    INST_FIELD_DATA_TYPE_UINT, // one_height(26)
    INST_FIELD_DATA_TYPE_UINT, // one_width(27)
    INST_FIELD_DATA_TYPE_UINT, // one_cube(28)
    INST_FIELD_DATA_TYPE_UINT, // tile_en(29)
    INST_FIELD_DATA_TYPE_UINT, // total_tile(30)
    INST_FIELD_DATA_TYPE_UINT, // one_size(31)
    INST_FIELD_DATA_TYPE_UINT, // shift_elew0(32)
    INST_FIELD_DATA_TYPE_UINT, // shift_elew1(33)
    INST_FIELD_DATA_TYPE_UINT, // elew_cut(34)
    INST_FIELD_DATA_TYPE_UINT, // elew_act_type(35)
};


// ConvAddr field data type
const vector<int> InstTable::ConvAddrFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_UINT, // type(3)
    INST_FIELD_DATA_TYPE_UINT, // last(4)
    INST_FIELD_DATA_TYPE_UINT, // compression(5)
    INST_FIELD_DATA_TYPE_UINT, // invalid(6)
    INST_FIELD_DATA_TYPE_UINT, // h_num(7)
    INST_FIELD_DATA_TYPE_UINT, // mt_addr(8)
    INST_FIELD_DATA_TYPE_UINT, // jump(9)
    INST_FIELD_DATA_TYPE_UINT, // jump_endl(10)
};


// Conv field data type
const vector<int> InstTable::ConvFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_UINT, // icg_offset(3)
    INST_FIELD_DATA_TYPE_UINT, // ocg_offset(4)
    INST_FIELD_DATA_TYPE_UINT, // pad_right(5)
    INST_FIELD_DATA_TYPE_UINT, // pad_left(6)
    INST_FIELD_DATA_TYPE_UINT, // pad_bottom(7)
    INST_FIELD_DATA_TYPE_UINT, // pad_top(8)
    INST_FIELD_DATA_TYPE_UINT, // reuse(9)
    INST_FIELD_DATA_TYPE_UINT, // macc_cont(10)
};


// AluInit field data type
const vector<int> InstTable::AluInitFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_UINT, // exec_mode(3)
    INST_FIELD_DATA_TYPE_UINT, // kernel_h(4)
    INST_FIELD_DATA_TYPE_UINT, // kernel_w(5)
    INST_FIELD_DATA_TYPE_UINT, // stride_h(6)
    INST_FIELD_DATA_TYPE_UINT, // stride_w(7)
    INST_FIELD_DATA_TYPE_UINT, // b_mode(8)
    INST_FIELD_DATA_TYPE_UINT, // num(9)
    INST_FIELD_DATA_TYPE_UINT, // share_kernel(10)
    INST_FIELD_DATA_TYPE_UINT, // share_channel_group(11)
    INST_FIELD_DATA_TYPE_UINT, // tile_owg(12)
    INST_FIELD_DATA_TYPE_UINT, // tile_ohg(13)
    INST_FIELD_DATA_TYPE_UINT, // pad_type(14)
    INST_FIELD_DATA_TYPE_UINT, // tile_cg(15)
    INST_FIELD_DATA_TYPE_UINT, // ow_iter(16)
    INST_FIELD_DATA_TYPE_UINT, // ow_offset(17)
    INST_FIELD_DATA_TYPE_UINT, // shift_hswish(18)
    INST_FIELD_DATA_TYPE_UINT, // shift_hsigmoid(19)
    INST_FIELD_DATA_TYPE_UINT, // hsigmoid_in(20)
    INST_FIELD_DATA_TYPE_UINT, // act_type(21)
    INST_FIELD_DATA_TYPE_UINT, // shift_bias(22)
    INST_FIELD_DATA_TYPE_UINT, // shift_cut(23)
    INST_FIELD_DATA_TYPE_UINT, // alu_num(24)
    INST_FIELD_DATA_TYPE_UINT, // weights_lines(25)
    INST_FIELD_DATA_TYPE_UINT, // shift_read_3(26)
    INST_FIELD_DATA_TYPE_UINT, // shift_read_2(27)
    INST_FIELD_DATA_TYPE_UINT, // shift_read_1(28)
    INST_FIELD_DATA_TYPE_UINT, // shift_read_0(29)
    INST_FIELD_DATA_TYPE_UINT, // total_tile(30)
    INST_FIELD_DATA_TYPE_UINT, // one_height(31)
    INST_FIELD_DATA_TYPE_UINT, // one_width(32)
    INST_FIELD_DATA_TYPE_UINT, // kh_iter(33)
    INST_FIELD_DATA_TYPE_UINT, // incAO3(34)
    INST_FIELD_DATA_TYPE_UINT, // kw_iter(35)
    INST_FIELD_DATA_TYPE_UINT, // incAO2(36)
};


// AluAddr field data type
const vector<int> InstTable::AluAddrFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_UINT, // type(3)
    INST_FIELD_DATA_TYPE_UINT, // last(4)
    INST_FIELD_DATA_TYPE_UINT, // compression(5)
    INST_FIELD_DATA_TYPE_UINT, // invalid(6)
    INST_FIELD_DATA_TYPE_UINT, // h_num(7)
    INST_FIELD_DATA_TYPE_UINT, // macc_dim(8)
    INST_FIELD_DATA_TYPE_UINT, // mt_addr(9)
    INST_FIELD_DATA_TYPE_UINT, // id(10)
    INST_FIELD_DATA_TYPE_UINT, // jump(11)
    INST_FIELD_DATA_TYPE_UINT, // jump_endl(12)
};


// Alu field data type
const vector<int> InstTable::AluFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_UINT, // reuse(3)
    INST_FIELD_DATA_TYPE_UINT, // macc_cont(4)
    INST_FIELD_DATA_TYPE_UINT, // pad_left(5)
    INST_FIELD_DATA_TYPE_UINT, // pad_top(6)
    INST_FIELD_DATA_TYPE_UINT, // pad_right_mt(7)
    INST_FIELD_DATA_TYPE_UINT, // pad_bottom_mt(8)
    INST_FIELD_DATA_TYPE_UINT, // pad_bottom_at(9)
    INST_FIELD_DATA_TYPE_UINT, // pad_right_at(10)
};


// End field data type
const vector<int> InstTable::EndFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
};


// DumpBank field data type
const vector<int> InstTable::DumpBankFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_STRING, // save_name(3)
    INST_FIELD_DATA_TYPE_UINT, // save_fmt(4)
    INST_FIELD_DATA_TYPE_UINT, // bank_start(5)
    INST_FIELD_DATA_TYPE_UINT, // bank_num(6)
};


// DumpDDR field data type
const vector<int> InstTable::DumpDDRFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_STRING, // save_name(3)
    INST_FIELD_DATA_TYPE_UINT, // save_fmt(4)
    INST_FIELD_DATA_TYPE_UINT, // reg_id(5)
    INST_FIELD_DATA_TYPE_UINT, // ddr_start(6)
    INST_FIELD_DATA_TYPE_UINT, // ddr_size(7)
};


// DumpDDRSlice field data type
const vector<int> InstTable::DumpDDRSliceFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_STRING, // save_name(3)
    INST_FIELD_DATA_TYPE_UINT, // save_fmt(4)
    INST_FIELD_DATA_TYPE_UINT, // reg_id(5)
    INST_FIELD_DATA_TYPE_UINT, // ddr_start(6)
    INST_FIELD_DATA_TYPE_UINT, // height(7)
    INST_FIELD_DATA_TYPE_UINT, // height_stride(8)
    INST_FIELD_DATA_TYPE_UINT, // width(9)
    INST_FIELD_DATA_TYPE_UINT, // width_stride(10)
    INST_FIELD_DATA_TYPE_UINT, // channel(11)
    INST_FIELD_DATA_TYPE_UINT, // channel_stride(12)
};


// Load field minus value
const vector<int> InstTable::LoadFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // mt_dst(3)
    0, // mt_addr(4)
    1, // block_num(5)
    0, // broadcast(6)
    0, // pad_start(7)
    0, // pad_end(8)
    1, // pad_idx(9)
    0, // reg_id(10)
    0, // const_value(11)
    1, // jump_read(12)
    0, // mode_avg(13)
    1, // length(14)
    1, // channel(15)
    0, // ddr_addr(16)
    1, // jump_write(17)
    1, // jump_write_endl(18)
};


// Save field minus value
const vector<int> InstTable::SaveFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // argmax(3)
    0, // const_en(4)
    0, // mt_addr(5)
    0, // reg_id(6)
    0, // const_value(7)
    1, // jump_write(8)
    1, // jump_read(9)
    1, // length(10)
    1, // channel(11)
    0, // ddr_addr(12)
};


// ConvInit field minus value
const vector<int> InstTable::ConvInitFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // exec_mode(3)
    1, // kernel_h(4)
    1, // kernel_w(5)
    1, // stride_h(6)
    1, // stride_w(7)
    0, // total_in(8)
    1, // tile_icg(9)
    1, // tile_ocg(10)
    1, // tile_ohg(11)
    0, // tile_owg(12)
    1, // ic_iter(13)
    1, // ow_iter(14)
    0, // ow_offset(15)
    0, // shift_hswish(16)
    0, // shift_hsigmoid(17)
    0, // hsigmoid_in(18)
    1, // kw_iter(19)
    1, // kh_iter(20)
    0, // act_type(21)
    0, // aie_mode(22)
    1, // conv_num(23)
    0, // shift_bias(24)
    0, // shift_cut(25)
    0, // one_height(26)
    0, // one_width(27)
    0, // one_cube(28)
    0, // tile_en(29)
    0, // total_tile(30)
    0, // one_size(31)
    0, // shift_elew0(32)
    0, // shift_elew1(33)
    0, // elew_cut(34)
    0, // elew_act_type(35)
};


// ConvAddr field minus value
const vector<int> InstTable::ConvAddrFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // type(3)
    0, // last(4)
    0, // compression(5)
    0, // invalid(6)
    1, // h_num(7)
    0, // mt_addr(8)
    0, // jump(9)
    0, // jump_endl(10)
};


// Conv field minus value
const vector<int> InstTable::ConvFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // icg_offset(3)
    0, // ocg_offset(4)
    0, // pad_right(5)
    0, // pad_left(6)
    0, // pad_bottom(7)
    0, // pad_top(8)
    0, // reuse(9)
    0, // macc_cont(10)
};


// AluInit field minus value
const vector<int> InstTable::AluInitFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // exec_mode(3)
    1, // kernel_h(4)
    1, // kernel_w(5)
    1, // stride_h(6)
    1, // stride_w(7)
    0, // b_mode(8)
    1, // num(9)
    0, // share_kernel(10)
    0, // share_channel_group(11)
    0, // tile_owg(12)
    1, // tile_ohg(13)
    0, // pad_type(14)
    1, // tile_cg(15)
    1, // ow_iter(16)
    0, // ow_offset(17)
    0, // shift_hswish(18)
    0, // shift_hsigmoid(19)
    0, // hsigmoid_in(20)
    0, // act_type(21)
    0, // shift_bias(22)
    0, // shift_cut(23)
    1, // alu_num(24)
    0, // weights_lines(25)
    0, // shift_read_3(26)
    0, // shift_read_2(27)
    0, // shift_read_1(28)
    0, // shift_read_0(29)
    0, // total_tile(30)
    0, // one_height(31)
    0, // one_width(32)
    1, // kh_iter(33)
    0, // incAO3(34)
    1, // kw_iter(35)
    0, // incAO2(36)
};


// AluAddr field minus value
const vector<int> InstTable::AluAddrFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // type(3)
    0, // last(4)
    0, // compression(5)
    0, // invalid(6)
    1, // h_num(7)
    0, // macc_dim(8)
    0, // mt_addr(9)
    0, // id(10)
    0, // jump(11)
    0, // jump_endl(12)
};


// Alu field minus value
const vector<int> InstTable::AluFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // reuse(3)
    0, // macc_cont(4)
    0, // pad_left(5)
    0, // pad_top(6)
    0, // pad_right_mt(7)
    0, // pad_bottom_mt(8)
    0, // pad_bottom_at(9)
    0, // pad_right_at(10)
};


// End field minus value
const vector<int> InstTable::EndFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
};


// DumpBank field minus value
const vector<int> InstTable::DumpBankFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // save_name(3)
    0, // save_fmt(4)
    0, // bank_start(5)
    0, // bank_num(6)
};


// DumpDDR field minus value
const vector<int> InstTable::DumpDDRFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // save_name(3)
    0, // save_fmt(4)
    0, // reg_id(5)
    0, // ddr_start(6)
    0, // ddr_size(7)
};


// DumpDDRSlice field minus value
const vector<int> InstTable::DumpDDRSliceFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // save_name(3)
    0, // save_fmt(4)
    0, // reg_id(5)
    0, // ddr_start(6)
    0, // height(7)
    0, // height_stride(8)
    0, // width(9)
    0, // width_stride(10)
    0, // channel(11)
    0, // channel_stride(12)
};


// Load field position
const vector<map<uint32_t, uint32_t>> InstTable::LoadFieldPos = {
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_OPCODE, 28 }, // 0
        { LOAD_FIELD_DPDON, 24 }, // 1
        { LOAD_FIELD_DPDBY, 20 }, // 2
        { LOAD_FIELD_MT_DST, 17 }, // 3
        { LOAD_FIELD_MT_ADDR, 0 }, // 4
    },
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_BLOCK_NUM, 22 }, // 5
        { LOAD_FIELD_BROADCAST, 18 }, // 6
        { LOAD_FIELD_PAD_START, 11 }, // 7
        { LOAD_FIELD_PAD_END, 5 }, // 8
        { LOAD_FIELD_PAD_IDX, 0 }, // 9
    },
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_REG_ID, 24 }, // 10
        { LOAD_FIELD_CONST_VALUE, 16 }, // 11
        { LOAD_FIELD_JUMP_READ, 0 }, // 12
    },
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_MODE_AVG, 30 }, // 13
        { LOAD_FIELD_LENGTH, 15 }, // 14
        { LOAD_FIELD_CHANNEL, 0 }, // 15
    },
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_DDR_ADDR, 0 }, // 16
    },
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_JUMP_WRITE, 18 }, // 17
        { LOAD_FIELD_JUMP_WRITE_ENDL, 0 }, // 18
    },
};


// Save field position
const vector<map<uint32_t, uint32_t>> InstTable::SaveFieldPos = {
    map<uint32_t, uint32_t> {
        { SAVE_FIELD_OPCODE, 28 }, // 0
        { SAVE_FIELD_DPDON, 24 }, // 1
        { SAVE_FIELD_DPDBY, 20 }, // 2
        { SAVE_FIELD_ARGMAX, 18 }, // 3
        { SAVE_FIELD_CONST_EN, 17 }, // 4
        { SAVE_FIELD_MT_ADDR, 0 }, // 5
    },
    map<uint32_t, uint32_t> {
        { SAVE_FIELD_REG_ID, 24 }, // 6
        { SAVE_FIELD_CONST_VALUE, 16 }, // 7
        { SAVE_FIELD_JUMP_WRITE, 0 }, // 8
    },
    map<uint32_t, uint32_t> {
        { SAVE_FIELD_JUMP_READ, 22 }, // 9
        { SAVE_FIELD_LENGTH, 12 }, // 10
        { SAVE_FIELD_CHANNEL, 0 }, // 11
    },
    map<uint32_t, uint32_t> {
        { SAVE_FIELD_DDR_ADDR, 0 }, // 12
    },
};


// ConvInit field position
const vector<map<uint32_t, uint32_t>> InstTable::ConvInitFieldPos = {
    map<uint32_t, uint32_t> {
        { CONVINIT_FIELD_OPCODE, 28 }, // 0
        { CONVINIT_FIELD_DPDON, 24 }, // 1
        { CONVINIT_FIELD_DPDBY, 20 }, // 2
        { CONVINIT_FIELD_EXEC_MODE, 16 }, // 3
        { CONVINIT_FIELD_KERNEL_H, 12 }, // 4
        { CONVINIT_FIELD_KERNEL_W, 8 }, // 5
        { CONVINIT_FIELD_STRIDE_H, 4 }, // 6
        { CONVINIT_FIELD_STRIDE_W, 0 }, // 7
    },
    map<uint32_t, uint32_t> {
        { CONVINIT_FIELD_TOTAL_IN, 21 }, // 8
        { CONVINIT_FIELD_TILE_ICG, 15 }, // 9
        { CONVINIT_FIELD_TILE_OCG, 9 }, // 10
        { CONVINIT_FIELD_TILE_OHG, 3 }, // 11
        { CONVINIT_FIELD_TILE_OWG, 0 }, // 12
    },
    map<uint32_t, uint32_t> {
        { CONVINIT_FIELD_IC_ITER, 24 }, // 13
        { CONVINIT_FIELD_OW_ITER, 19 }, // 14
        { CONVINIT_FIELD_OW_OFFSET, 13 }, // 15
        { CONVINIT_FIELD_SHIFT_HSWISH, 9 }, // 16
        { CONVINIT_FIELD_SHIFT_HSIGMOID, 4 }, // 17
        { CONVINIT_FIELD_HSIGMOID_IN, 0 }, // 18
    },
    map<uint32_t, uint32_t> {
        { CONVINIT_FIELD_KW_ITER, 26 }, // 19
        { CONVINIT_FIELD_KH_ITER, 20 }, // 20
        { CONVINIT_FIELD_ACT_TYPE, 13 }, // 21
        { CONVINIT_FIELD_AIE_MODE, 10 }, // 22
        { CONVINIT_FIELD_CONV_NUM, 0 }, // 23
    },
    map<uint32_t, uint32_t> {
        { CONVINIT_FIELD_SHIFT_BIAS, 26 }, // 24
        { CONVINIT_FIELD_SHIFT_CUT, 20 }, // 25
        { CONVINIT_FIELD_ONE_HEIGHT, 10 }, // 26
        { CONVINIT_FIELD_ONE_WIDTH, 0 }, // 27
    },
    map<uint32_t, uint32_t> {
        { CONVINIT_FIELD_ONE_CUBE, 20 }, // 28
        { CONVINIT_FIELD_TILE_EN, 19 }, // 29
        { CONVINIT_FIELD_TOTAL_TILE, 10 }, // 30
        { CONVINIT_FIELD_ONE_SIZE, 0 }, // 31
    },
    map<uint32_t, uint32_t> {
        { CONVINIT_FIELD_SHIFT_ELEW0, 26 }, // 32
        { CONVINIT_FIELD_SHIFT_ELEW1, 20 }, // 33
        { CONVINIT_FIELD_ELEW_CUT, 13 }, // 34
        { CONVINIT_FIELD_ELEW_ACT_TYPE, 9 }, // 35
    },
};


// ConvAddr field position
const vector<map<uint32_t, uint32_t>> InstTable::ConvAddrFieldPos = {
    map<uint32_t, uint32_t> {
        { CONVADDR_FIELD_OPCODE, 28 }, // 0
        { CONVADDR_FIELD_DPDON, 24 }, // 1
        { CONVADDR_FIELD_DPDBY, 20 }, // 2
        { CONVADDR_FIELD_TYPE, 17 }, // 3
        { CONVADDR_FIELD_LAST, 10 }, // 4
        { CONVADDR_FIELD_COMPRESSION, 9 }, // 5
        { CONVADDR_FIELD_INVALID, 8 }, // 6
        { CONVADDR_FIELD_H_NUM, 0 }, // 7
    },
    map<uint32_t, uint32_t> {
        { CONVADDR_FIELD_MT_ADDR, 0 }, // 8
    },
    map<uint32_t, uint32_t> {
        { CONVADDR_FIELD_JUMP, 16 }, // 9
        { CONVADDR_FIELD_JUMP_ENDL, 0 }, // 10
    },
};


// Conv field position
const vector<map<uint32_t, uint32_t>> InstTable::ConvFieldPos = {
    map<uint32_t, uint32_t> {
        { CONV_FIELD_OPCODE, 28 }, // 0
        { CONV_FIELD_DPDON, 24 }, // 1
        { CONV_FIELD_DPDBY, 20 }, // 2
        { CONV_FIELD_ICG_OFFSET, 16 }, // 3
        { CONV_FIELD_OCG_OFFSET, 12 }, // 4
        { CONV_FIELD_PAD_RIGHT, 4 }, // 5
        { CONV_FIELD_PAD_LEFT, 0 }, // 6
    },
    map<uint32_t, uint32_t> {
        { CONV_FIELD_PAD_BOTTOM, 24 }, // 7
        { CONV_FIELD_PAD_TOP, 20 }, // 8
        { CONV_FIELD_REUSE, 19 }, // 9
        { CONV_FIELD_MACC_CONT, 17 }, // 10
    },
};


// AluInit field position
const vector<map<uint32_t, uint32_t>> InstTable::AluInitFieldPos = {
    map<uint32_t, uint32_t> {
        { ALUINIT_FIELD_OPCODE, 28 }, // 0
        { ALUINIT_FIELD_DPDON, 24 }, // 1
        { ALUINIT_FIELD_DPDBY, 20 }, // 2
        { ALUINIT_FIELD_EXEC_MODE, 16 }, // 3
        { ALUINIT_FIELD_KERNEL_H, 8 }, // 4
        { ALUINIT_FIELD_KERNEL_W, 0 }, // 5
    },
    map<uint32_t, uint32_t> {
        { ALUINIT_FIELD_STRIDE_H, 24 }, // 6
        { ALUINIT_FIELD_STRIDE_W, 16 }, // 7
        { ALUINIT_FIELD_B_MODE, 13 }, // 8
        { ALUINIT_FIELD_NUM, 11 }, // 9
        { ALUINIT_FIELD_SHARE_KERNEL, 10 }, // 10
        { ALUINIT_FIELD_SHARE_CHANNEL_GROUP, 9 }, // 11
        { ALUINIT_FIELD_TILE_OWG, 6 }, // 12
        { ALUINIT_FIELD_TILE_OHG, 0 }, // 13
    },
    map<uint32_t, uint32_t> {
        { ALUINIT_FIELD_PAD_TYPE, 30 }, // 14
        { ALUINIT_FIELD_TILE_CG, 24 }, // 15
        { ALUINIT_FIELD_OW_ITER, 19 }, // 16
        { ALUINIT_FIELD_OW_OFFSET, 13 }, // 17
        { ALUINIT_FIELD_SHIFT_HSWISH, 9 }, // 18
        { ALUINIT_FIELD_SHIFT_HSIGMOID, 4 }, // 19
        { ALUINIT_FIELD_HSIGMOID_IN, 0 }, // 20
    },
    map<uint32_t, uint32_t> {
        { ALUINIT_FIELD_ACT_TYPE, 29 }, // 21
        { ALUINIT_FIELD_SHIFT_BIAS, 23 }, // 22
        { ALUINIT_FIELD_SHIFT_CUT, 16 }, // 23
        { ALUINIT_FIELD_ALU_NUM, 0 }, // 24
    },
    map<uint32_t, uint32_t> {
        { ALUINIT_FIELD_WEIGHTS_LINES, 17 }, // 25
        { ALUINIT_FIELD_SHIFT_READ_3, 12 }, // 26
        { ALUINIT_FIELD_SHIFT_READ_2, 8 }, // 27
        { ALUINIT_FIELD_SHIFT_READ_1, 4 }, // 28
        { ALUINIT_FIELD_SHIFT_READ_0, 0 }, // 29
    },
    map<uint32_t, uint32_t> {
        { ALUINIT_FIELD_TOTAL_TILE, 20 }, // 30
        { ALUINIT_FIELD_ONE_HEIGHT, 10 }, // 31
        { ALUINIT_FIELD_ONE_WIDTH, 0 }, // 32
    },
    map<uint32_t, uint32_t> {
        { ALUINIT_FIELD_KH_ITER, 24 }, // 33
        { ALUINIT_FIELD_INCAO3, 0 }, // 34
    },
    map<uint32_t, uint32_t> {
        { ALUINIT_FIELD_KW_ITER, 24 }, // 35
        { ALUINIT_FIELD_INCAO2, 0 }, // 36
    },
};


// AluAddr field position
const vector<map<uint32_t, uint32_t>> InstTable::AluAddrFieldPos = {
    map<uint32_t, uint32_t> {
        { ALUADDR_FIELD_OPCODE, 28 }, // 0
        { ALUADDR_FIELD_DPDON, 24 }, // 1
        { ALUADDR_FIELD_DPDBY, 20 }, // 2
        { ALUADDR_FIELD_TYPE, 17 }, // 3
        { ALUADDR_FIELD_LAST, 10 }, // 4
        { ALUADDR_FIELD_COMPRESSION, 9 }, // 5
        { ALUADDR_FIELD_INVALID, 8 }, // 6
        { ALUADDR_FIELD_H_NUM, 0 }, // 7
    },
    map<uint32_t, uint32_t> {
        { ALUADDR_FIELD_MACC_DIM, 26 }, // 8
        { ALUADDR_FIELD_MT_ADDR, 0 }, // 9
    },
    map<uint32_t, uint32_t> {
        { ALUADDR_FIELD_ID, 30 }, // 10
        { ALUADDR_FIELD_JUMP, 15 }, // 11
        { ALUADDR_FIELD_JUMP_ENDL, 0 }, // 12
    },
};


// Alu field position
const vector<map<uint32_t, uint32_t>> InstTable::AluFieldPos = {
    map<uint32_t, uint32_t> {
        { ALU_FIELD_OPCODE, 28 }, // 0
        { ALU_FIELD_DPDON, 24 }, // 1
        { ALU_FIELD_DPDBY, 20 }, // 2
        { ALU_FIELD_REUSE, 18 }, // 3
        { ALU_FIELD_MACC_CONT, 16 }, // 4
        { ALU_FIELD_PAD_LEFT, 12 }, // 5
        { ALU_FIELD_PAD_TOP, 8 }, // 6
        { ALU_FIELD_PAD_RIGHT_MT, 0 }, // 7
    },
    map<uint32_t, uint32_t> {
        { ALU_FIELD_PAD_BOTTOM_MT, 24 }, // 8
        { ALU_FIELD_PAD_BOTTOM_AT, 4 }, // 9
        { ALU_FIELD_PAD_RIGHT_AT, 0 }, // 10
    },
};


// End field position
const vector<map<uint32_t, uint32_t>> InstTable::EndFieldPos = {
    map<uint32_t, uint32_t> {
        { END_FIELD_OPCODE, 28 }, // 0
        { END_FIELD_DPDON, 24 }, // 1
        { END_FIELD_DPDBY, 20 }, // 2
    },
};


// DumpBank field position
const vector<map<uint32_t, uint32_t>> InstTable::DumpBankFieldPos = {
    map<uint32_t, uint32_t> {
        { DUMPBANK_FIELD_OPCODE, 28 }, // 0
        { DUMPBANK_FIELD_DPDON, 24 }, // 1
        { DUMPBANK_FIELD_DPDBY, 20 }, // 2
        { DUMPBANK_FIELD_SAVE_NAME, 0 }, // 3
    },
    map<uint32_t, uint32_t> {
        { DUMPBANK_FIELD_SAVE_FMT, 26 }, // 4
        { DUMPBANK_FIELD_BANK_START, 18 }, // 5
        { DUMPBANK_FIELD_BANK_NUM, 10 }, // 6
    },
};


// DumpDDR field position
const vector<map<uint32_t, uint32_t>> InstTable::DumpDDRFieldPos = {
    map<uint32_t, uint32_t> {
        { DUMPDDR_FIELD_OPCODE, 28 }, // 0
        { DUMPDDR_FIELD_DPDON, 24 }, // 1
        { DUMPDDR_FIELD_DPDBY, 20 }, // 2
        { DUMPDDR_FIELD_SAVE_NAME, 0 }, // 3
    },
    map<uint32_t, uint32_t> {
        { DUMPDDR_FIELD_SAVE_FMT, 26 }, // 4
        { DUMPDDR_FIELD_REG_ID, 20 }, // 5
    },
    map<uint32_t, uint32_t> {
        { DUMPDDR_FIELD_DDR_START, 0 }, // 6
    },
    map<uint32_t, uint32_t> {
        { DUMPDDR_FIELD_DDR_SIZE, 0 }, // 7
    },
};


// DumpDDRSlice field position
const vector<map<uint32_t, uint32_t>> InstTable::DumpDDRSliceFieldPos = {
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_OPCODE, 28 }, // 0
        { DUMPDDRSLICE_FIELD_DPDON, 24 }, // 1
        { DUMPDDRSLICE_FIELD_DPDBY, 20 }, // 2
        { DUMPDDRSLICE_FIELD_SAVE_NAME, 0 }, // 3
    },
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_SAVE_FMT, 26 }, // 4
        { DUMPDDRSLICE_FIELD_REG_ID, 20 }, // 5
    },
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_DDR_START, 0 }, // 6
    },
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_HEIGHT, 16 }, // 7
        { DUMPDDRSLICE_FIELD_HEIGHT_STRIDE, 0 }, // 8
    },
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_WIDTH, 16 }, // 9
        { DUMPDDRSLICE_FIELD_WIDTH_STRIDE, 0 }, // 10
    },
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_CHANNEL, 16 }, // 11
        { DUMPDDRSLICE_FIELD_CHANNEL_STRIDE, 0 }, // 12
    },
};


// Load field length
const vector<map<uint32_t, uint32_t>> InstTable::LoadFieldLen = {
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_OPCODE, 4 }, // 0
        { LOAD_FIELD_DPDON, 4 }, // 1
        { LOAD_FIELD_DPDBY, 4 }, // 2
        { LOAD_FIELD_MT_DST, 3 }, // 3
        { LOAD_FIELD_MT_ADDR, 17 }, // 4
    },
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_BLOCK_NUM, 10 }, // 5
        { LOAD_FIELD_BROADCAST, 4 }, // 6
        { LOAD_FIELD_PAD_START, 6 }, // 7
        { LOAD_FIELD_PAD_END, 6 }, // 8
        { LOAD_FIELD_PAD_IDX, 5 }, // 9
    },
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_REG_ID, 8 }, // 10
        { LOAD_FIELD_CONST_VALUE, 8 }, // 11
        { LOAD_FIELD_JUMP_READ, 16 }, // 12
    },
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_MODE_AVG, 2 }, // 13
        { LOAD_FIELD_LENGTH, 10 }, // 14
        { LOAD_FIELD_CHANNEL, 15 }, // 15
    },
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_DDR_ADDR, 29 }, // 16
    },
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_JUMP_WRITE, 12 }, // 17
        { LOAD_FIELD_JUMP_WRITE_ENDL, 18 }, // 18
    },
};


// Save field length
const vector<map<uint32_t, uint32_t>> InstTable::SaveFieldLen = {
    map<uint32_t, uint32_t> {
        { SAVE_FIELD_OPCODE, 4 }, // 0
        { SAVE_FIELD_DPDON, 4 }, // 1
        { SAVE_FIELD_DPDBY, 4 }, // 2
        { SAVE_FIELD_ARGMAX, 2 }, // 3
        { SAVE_FIELD_CONST_EN, 1 }, // 4
        { SAVE_FIELD_MT_ADDR, 17 }, // 5
    },
    map<uint32_t, uint32_t> {
        { SAVE_FIELD_REG_ID, 8 }, // 6
        { SAVE_FIELD_CONST_VALUE, 8 }, // 7
        { SAVE_FIELD_JUMP_WRITE, 16 }, // 8
    },
    map<uint32_t, uint32_t> {
        { SAVE_FIELD_JUMP_READ, 8 }, // 9
        { SAVE_FIELD_LENGTH, 10 }, // 10
        { SAVE_FIELD_CHANNEL, 12 }, // 11
    },
    map<uint32_t, uint32_t> {
        { SAVE_FIELD_DDR_ADDR, 29 }, // 12
    },
};


// ConvInit field length
const vector<map<uint32_t, uint32_t>> InstTable::ConvInitFieldLen = {
    map<uint32_t, uint32_t> {
        { CONVINIT_FIELD_OPCODE, 4 }, // 0
        { CONVINIT_FIELD_DPDON, 4 }, // 1
        { CONVINIT_FIELD_DPDBY, 4 }, // 2
        { CONVINIT_FIELD_EXEC_MODE, 4 }, // 3
        { CONVINIT_FIELD_KERNEL_H, 4 }, // 4
        { CONVINIT_FIELD_KERNEL_W, 4 }, // 5
        { CONVINIT_FIELD_STRIDE_H, 4 }, // 6
        { CONVINIT_FIELD_STRIDE_W, 4 }, // 7
    },
    map<uint32_t, uint32_t> {
        { CONVINIT_FIELD_TOTAL_IN, 11 }, // 8
        { CONVINIT_FIELD_TILE_ICG, 6 }, // 9
        { CONVINIT_FIELD_TILE_OCG, 6 }, // 10
        { CONVINIT_FIELD_TILE_OHG, 6 }, // 11
        { CONVINIT_FIELD_TILE_OWG, 3 }, // 12
    },
    map<uint32_t, uint32_t> {
        { CONVINIT_FIELD_IC_ITER, 6 }, // 13
        { CONVINIT_FIELD_OW_ITER, 5 }, // 14
        { CONVINIT_FIELD_OW_OFFSET, 6 }, // 15
        { CONVINIT_FIELD_SHIFT_HSWISH, 4 }, // 16
        { CONVINIT_FIELD_SHIFT_HSIGMOID, 5 }, // 17
        { CONVINIT_FIELD_HSIGMOID_IN, 4 }, // 18
    },
    map<uint32_t, uint32_t> {
        { CONVINIT_FIELD_KW_ITER, 6 }, // 19
        { CONVINIT_FIELD_KH_ITER, 6 }, // 20
        { CONVINIT_FIELD_ACT_TYPE, 4 }, // 21
        { CONVINIT_FIELD_AIE_MODE, 3 }, // 22
        { CONVINIT_FIELD_CONV_NUM, 10 }, // 23
    },
    map<uint32_t, uint32_t> {
        { CONVINIT_FIELD_SHIFT_BIAS, 6 }, // 24
        { CONVINIT_FIELD_SHIFT_CUT, 6 }, // 25
        { CONVINIT_FIELD_ONE_HEIGHT, 10 }, // 26
        { CONVINIT_FIELD_ONE_WIDTH, 10 }, // 27
    },
    map<uint32_t, uint32_t> {
        { CONVINIT_FIELD_ONE_CUBE, 12 }, // 28
        { CONVINIT_FIELD_TILE_EN, 1 }, // 29
        { CONVINIT_FIELD_TOTAL_TILE, 9 }, // 30
        { CONVINIT_FIELD_ONE_SIZE, 10 }, // 31
    },
    map<uint32_t, uint32_t> {
        { CONVINIT_FIELD_SHIFT_ELEW0, 6 }, // 32
        { CONVINIT_FIELD_SHIFT_ELEW1, 6 }, // 33
        { CONVINIT_FIELD_ELEW_CUT, 7 }, // 34
        { CONVINIT_FIELD_ELEW_ACT_TYPE, 4 }, // 35
    },
};


// ConvAddr field length
const vector<map<uint32_t, uint32_t>> InstTable::ConvAddrFieldLen = {
    map<uint32_t, uint32_t> {
        { CONVADDR_FIELD_OPCODE, 4 }, // 0
        { CONVADDR_FIELD_DPDON, 4 }, // 1
        { CONVADDR_FIELD_DPDBY, 4 }, // 2
        { CONVADDR_FIELD_TYPE, 3 }, // 3
        { CONVADDR_FIELD_LAST, 1 }, // 4
        { CONVADDR_FIELD_COMPRESSION, 1 }, // 5
        { CONVADDR_FIELD_INVALID, 1 }, // 6
        { CONVADDR_FIELD_H_NUM, 8 }, // 7
    },
    map<uint32_t, uint32_t> {
        { CONVADDR_FIELD_MT_ADDR, 17 }, // 8
    },
    map<uint32_t, uint32_t> {
        { CONVADDR_FIELD_JUMP, 14 }, // 9
        { CONVADDR_FIELD_JUMP_ENDL, 16 }, // 10
    },
};


// Conv field length
const vector<map<uint32_t, uint32_t>> InstTable::ConvFieldLen = {
    map<uint32_t, uint32_t> {
        { CONV_FIELD_OPCODE, 4 }, // 0
        { CONV_FIELD_DPDON, 4 }, // 1
        { CONV_FIELD_DPDBY, 4 }, // 2
        { CONV_FIELD_ICG_OFFSET, 4 }, // 3
        { CONV_FIELD_OCG_OFFSET, 4 }, // 4
        { CONV_FIELD_PAD_RIGHT, 8 }, // 5
        { CONV_FIELD_PAD_LEFT, 4 }, // 6
    },
    map<uint32_t, uint32_t> {
        { CONV_FIELD_PAD_BOTTOM, 8 }, // 7
        { CONV_FIELD_PAD_TOP, 4 }, // 8
        { CONV_FIELD_REUSE, 1 }, // 9
        { CONV_FIELD_MACC_CONT, 2 }, // 10
    },
};


// AluInit field length
const vector<map<uint32_t, uint32_t>> InstTable::AluInitFieldLen = {
    map<uint32_t, uint32_t> {
        { ALUINIT_FIELD_OPCODE, 4 }, // 0
        { ALUINIT_FIELD_DPDON, 4 }, // 1
        { ALUINIT_FIELD_DPDBY, 4 }, // 2
        { ALUINIT_FIELD_EXEC_MODE, 4 }, // 3
        { ALUINIT_FIELD_KERNEL_H, 8 }, // 4
        { ALUINIT_FIELD_KERNEL_W, 8 }, // 5
    },
    map<uint32_t, uint32_t> {
        { ALUINIT_FIELD_STRIDE_H, 8 }, // 6
        { ALUINIT_FIELD_STRIDE_W, 8 }, // 7
        { ALUINIT_FIELD_B_MODE, 3 }, // 8
        { ALUINIT_FIELD_NUM, 2 }, // 9
        { ALUINIT_FIELD_SHARE_KERNEL, 1 }, // 10
        { ALUINIT_FIELD_SHARE_CHANNEL_GROUP, 1 }, // 11
        { ALUINIT_FIELD_TILE_OWG, 3 }, // 12
        { ALUINIT_FIELD_TILE_OHG, 6 }, // 13
    },
    map<uint32_t, uint32_t> {
        { ALUINIT_FIELD_PAD_TYPE, 2 }, // 14
        { ALUINIT_FIELD_TILE_CG, 6 }, // 15
        { ALUINIT_FIELD_OW_ITER, 5 }, // 16
        { ALUINIT_FIELD_OW_OFFSET, 6 }, // 17
        { ALUINIT_FIELD_SHIFT_HSWISH, 4 }, // 18
        { ALUINIT_FIELD_SHIFT_HSIGMOID, 5 }, // 19
        { ALUINIT_FIELD_HSIGMOID_IN, 4 }, // 20
    },
    map<uint32_t, uint32_t> {
        { ALUINIT_FIELD_ACT_TYPE, 3 }, // 21
        { ALUINIT_FIELD_SHIFT_BIAS, 6 }, // 22
        { ALUINIT_FIELD_SHIFT_CUT, 7 }, // 23
        { ALUINIT_FIELD_ALU_NUM, 12 }, // 24
    },
    map<uint32_t, uint32_t> {
        { ALUINIT_FIELD_WEIGHTS_LINES, 15 }, // 25
        { ALUINIT_FIELD_SHIFT_READ_3, 4 }, // 26
        { ALUINIT_FIELD_SHIFT_READ_2, 4 }, // 27
        { ALUINIT_FIELD_SHIFT_READ_1, 4 }, // 28
        { ALUINIT_FIELD_SHIFT_READ_0, 4 }, // 29
    },
    map<uint32_t, uint32_t> {
        { ALUINIT_FIELD_TOTAL_TILE, 9 }, // 30
        { ALUINIT_FIELD_ONE_HEIGHT, 10 }, // 31
        { ALUINIT_FIELD_ONE_WIDTH, 10 }, // 32
    },
    map<uint32_t, uint32_t> {
        { ALUINIT_FIELD_KH_ITER, 8 }, // 33
        { ALUINIT_FIELD_INCAO3, 24 }, // 34
    },
    map<uint32_t, uint32_t> {
        { ALUINIT_FIELD_KW_ITER, 8 }, // 35
        { ALUINIT_FIELD_INCAO2, 18 }, // 36
    },
};


// AluAddr field length
const vector<map<uint32_t, uint32_t>> InstTable::AluAddrFieldLen = {
    map<uint32_t, uint32_t> {
        { ALUADDR_FIELD_OPCODE, 4 }, // 0
        { ALUADDR_FIELD_DPDON, 4 }, // 1
        { ALUADDR_FIELD_DPDBY, 4 }, // 2
        { ALUADDR_FIELD_TYPE, 3 }, // 3
        { ALUADDR_FIELD_LAST, 1 }, // 4
        { ALUADDR_FIELD_COMPRESSION, 1 }, // 5
        { ALUADDR_FIELD_INVALID, 1 }, // 6
        { ALUADDR_FIELD_H_NUM, 8 }, // 7
    },
    map<uint32_t, uint32_t> {
        { ALUADDR_FIELD_MACC_DIM, 2 }, // 8
        { ALUADDR_FIELD_MT_ADDR, 17 }, // 9
    },
    map<uint32_t, uint32_t> {
        { ALUADDR_FIELD_ID, 2 }, // 10
        { ALUADDR_FIELD_JUMP, 15 }, // 11
        { ALUADDR_FIELD_JUMP_ENDL, 15 }, // 12
    },
};


// Alu field length
const vector<map<uint32_t, uint32_t>> InstTable::AluFieldLen = {
    map<uint32_t, uint32_t> {
        { ALU_FIELD_OPCODE, 4 }, // 0
        { ALU_FIELD_DPDON, 4 }, // 1
        { ALU_FIELD_DPDBY, 4 }, // 2
        { ALU_FIELD_REUSE, 1 }, // 3
        { ALU_FIELD_MACC_CONT, 2 }, // 4
        { ALU_FIELD_PAD_LEFT, 4 }, // 5
        { ALU_FIELD_PAD_TOP, 4 }, // 6
        { ALU_FIELD_PAD_RIGHT_MT, 8 }, // 7
    },
    map<uint32_t, uint32_t> {
        { ALU_FIELD_PAD_BOTTOM_MT, 8 }, // 8
        { ALU_FIELD_PAD_BOTTOM_AT, 4 }, // 9
        { ALU_FIELD_PAD_RIGHT_AT, 4 }, // 10
    },
};


// End field length
const vector<map<uint32_t, uint32_t>> InstTable::EndFieldLen = {
    map<uint32_t, uint32_t> {
        { END_FIELD_OPCODE, 4 }, // 0
        { END_FIELD_DPDON, 4 }, // 1
        { END_FIELD_DPDBY, 4 }, // 2
    },
};


// DumpBank field length
const vector<map<uint32_t, uint32_t>> InstTable::DumpBankFieldLen = {
    map<uint32_t, uint32_t> {
        { DUMPBANK_FIELD_OPCODE, 4 }, // 0
        { DUMPBANK_FIELD_DPDON, 4 }, // 1
        { DUMPBANK_FIELD_DPDBY, 4 }, // 2
        { DUMPBANK_FIELD_SAVE_NAME, 20 }, // 3
    },
    map<uint32_t, uint32_t> {
        { DUMPBANK_FIELD_SAVE_FMT, 6 }, // 4
        { DUMPBANK_FIELD_BANK_START, 8 }, // 5
        { DUMPBANK_FIELD_BANK_NUM, 8 }, // 6
    },
};


// DumpDDR field length
const vector<map<uint32_t, uint32_t>> InstTable::DumpDDRFieldLen = {
    map<uint32_t, uint32_t> {
        { DUMPDDR_FIELD_OPCODE, 4 }, // 0
        { DUMPDDR_FIELD_DPDON, 4 }, // 1
        { DUMPDDR_FIELD_DPDBY, 4 }, // 2
        { DUMPDDR_FIELD_SAVE_NAME, 20 }, // 3
    },
    map<uint32_t, uint32_t> {
        { DUMPDDR_FIELD_SAVE_FMT, 6 }, // 4
        { DUMPDDR_FIELD_REG_ID, 6 }, // 5
    },
    map<uint32_t, uint32_t> {
        { DUMPDDR_FIELD_DDR_START, 32 }, // 6
    },
    map<uint32_t, uint32_t> {
        { DUMPDDR_FIELD_DDR_SIZE, 32 }, // 7
    },
};


// DumpDDRSlice field length
const vector<map<uint32_t, uint32_t>> InstTable::DumpDDRSliceFieldLen = {
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_OPCODE, 4 }, // 0
        { DUMPDDRSLICE_FIELD_DPDON, 4 }, // 1
        { DUMPDDRSLICE_FIELD_DPDBY, 4 }, // 2
        { DUMPDDRSLICE_FIELD_SAVE_NAME, 20 }, // 3
    },
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_SAVE_FMT, 6 }, // 4
        { DUMPDDRSLICE_FIELD_REG_ID, 6 }, // 5
    },
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_DDR_START, 32 }, // 6
    },
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_HEIGHT, 16 }, // 7
        { DUMPDDRSLICE_FIELD_HEIGHT_STRIDE, 16 }, // 8
    },
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_WIDTH, 16 }, // 9
        { DUMPDDRSLICE_FIELD_WIDTH_STRIDE, 16 }, // 10
    },
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_CHANNEL, 16 }, // 11
        { DUMPDDRSLICE_FIELD_CHANNEL_STRIDE, 16 }, // 12
    },
};


// inst opcode to inst type map
const unordered_map<uint32_t, uint32_t> InstTable::OPCode2InstType = {
    { 0x00, INST_TYPE_LOAD }, // 0
    { 0x04, INST_TYPE_SAVE }, // 1
    { 0x09, INST_TYPE_CONVINIT }, // 2
    { 0x05, INST_TYPE_CONVADDR }, // 3
    { 0x08, INST_TYPE_CONV }, // 4
    { 0x01, INST_TYPE_ALUINIT }, // 5
    { 0x03, INST_TYPE_ALUADDR }, // 6
    { 0x02, INST_TYPE_ALU }, // 7
    { 0x07, INST_TYPE_END }, // 8
    { 0xFF, INST_TYPE_DUMPBANK }, // 9
    { 0xFE, INST_TYPE_DUMPDDR }, // 10
    { 0xFD, INST_TYPE_DUMPDDRSLICE }, // 11
};


// inst name to inst type map
const unordered_map<string, uint32_t> InstTable::InstName2InstType = {
    { "LOAD", INST_TYPE_LOAD }, // 0
    { "SAVE", INST_TYPE_SAVE }, // 1
    { "CONVINIT", INST_TYPE_CONVINIT }, // 2
    { "CONVADDR", INST_TYPE_CONVADDR }, // 3
    { "CONV", INST_TYPE_CONV }, // 4
    { "ALUINIT", INST_TYPE_ALUINIT }, // 5
    { "ALUADDR", INST_TYPE_ALUADDR }, // 6
    { "ALU", INST_TYPE_ALU }, // 7
    { "END", INST_TYPE_END }, // 8
    { "DUMPBANK", INST_TYPE_DUMPBANK }, // 9
    { "DUMPDDR", INST_TYPE_DUMPDDR }, // 10
    { "DUMPDDRSLICE", INST_TYPE_DUMPDDRSLICE }, // 11
};


// inst type to field name map
const unordered_map<uint32_t, const vector<string> &> InstTable::InstFieldName = {
    { INST_TYPE_LOAD, LoadFieldName }, // 0
    { INST_TYPE_SAVE, SaveFieldName }, // 1
    { INST_TYPE_CONVINIT, ConvInitFieldName }, // 2
    { INST_TYPE_CONVADDR, ConvAddrFieldName }, // 3
    { INST_TYPE_CONV, ConvFieldName }, // 4
    { INST_TYPE_ALUINIT, AluInitFieldName }, // 5
    { INST_TYPE_ALUADDR, AluAddrFieldName }, // 6
    { INST_TYPE_ALU, AluFieldName }, // 7
    { INST_TYPE_END, EndFieldName }, // 8
    { INST_TYPE_DUMPBANK, DumpBankFieldName }, // 9
    { INST_TYPE_DUMPDDR, DumpDDRFieldName }, // 10
    { INST_TYPE_DUMPDDRSLICE, DumpDDRSliceFieldName }, // 11
};


// inst type to field data type map
const unordered_map<uint32_t, const vector<int> &> InstTable::InstFieldDataType = {
    { INST_TYPE_LOAD, LoadFieldDataType }, // 0
    { INST_TYPE_SAVE, SaveFieldDataType }, // 1
    { INST_TYPE_CONVINIT, ConvInitFieldDataType }, // 2
    { INST_TYPE_CONVADDR, ConvAddrFieldDataType }, // 3
    { INST_TYPE_CONV, ConvFieldDataType }, // 4
    { INST_TYPE_ALUINIT, AluInitFieldDataType }, // 5
    { INST_TYPE_ALUADDR, AluAddrFieldDataType }, // 6
    { INST_TYPE_ALU, AluFieldDataType }, // 7
    { INST_TYPE_END, EndFieldDataType }, // 8
    { INST_TYPE_DUMPBANK, DumpBankFieldDataType }, // 9
    { INST_TYPE_DUMPDDR, DumpDDRFieldDataType }, // 10
    { INST_TYPE_DUMPDDRSLICE, DumpDDRSliceFieldDataType }, // 11
};


// inst type to field minus map
const unordered_map<uint32_t, const vector<int> &> InstTable::InstFieldMinus = {
    { INST_TYPE_LOAD, LoadFieldMinus }, // 0
    { INST_TYPE_SAVE, SaveFieldMinus }, // 1
    { INST_TYPE_CONVINIT, ConvInitFieldMinus }, // 2
    { INST_TYPE_CONVADDR, ConvAddrFieldMinus }, // 3
    { INST_TYPE_CONV, ConvFieldMinus }, // 4
    { INST_TYPE_ALUINIT, AluInitFieldMinus }, // 5
    { INST_TYPE_ALUADDR, AluAddrFieldMinus }, // 6
    { INST_TYPE_ALU, AluFieldMinus }, // 7
    { INST_TYPE_END, EndFieldMinus }, // 8
    { INST_TYPE_DUMPBANK, DumpBankFieldMinus }, // 9
    { INST_TYPE_DUMPDDR, DumpDDRFieldMinus }, // 10
    { INST_TYPE_DUMPDDRSLICE, DumpDDRSliceFieldMinus }, // 11
};


// inst type to field position map
const unordered_map<uint32_t, const vector<map<uint32_t, uint32_t>> &> InstTable::InstFieldPos = {
    { INST_TYPE_LOAD, LoadFieldPos }, // 0
    { INST_TYPE_SAVE, SaveFieldPos }, // 1
    { INST_TYPE_CONVINIT, ConvInitFieldPos }, // 2
    { INST_TYPE_CONVADDR, ConvAddrFieldPos }, // 3
    { INST_TYPE_CONV, ConvFieldPos }, // 4
    { INST_TYPE_ALUINIT, AluInitFieldPos }, // 5
    { INST_TYPE_ALUADDR, AluAddrFieldPos }, // 6
    { INST_TYPE_ALU, AluFieldPos }, // 7
    { INST_TYPE_END, EndFieldPos }, // 8
    { INST_TYPE_DUMPBANK, DumpBankFieldPos }, // 9
    { INST_TYPE_DUMPDDR, DumpDDRFieldPos }, // 10
    { INST_TYPE_DUMPDDRSLICE, DumpDDRSliceFieldPos }, // 11
};


// inst type to field length map
const unordered_map<uint32_t, const vector<map<uint32_t, uint32_t>> &> InstTable::InstFieldLen = {
    { INST_TYPE_LOAD, LoadFieldLen }, // 0
    { INST_TYPE_SAVE, SaveFieldLen }, // 1
    { INST_TYPE_CONVINIT, ConvInitFieldLen }, // 2
    { INST_TYPE_CONVADDR, ConvAddrFieldLen }, // 3
    { INST_TYPE_CONV, ConvFieldLen }, // 4
    { INST_TYPE_ALUINIT, AluInitFieldLen }, // 5
    { INST_TYPE_ALUADDR, AluAddrFieldLen }, // 6
    { INST_TYPE_ALU, AluFieldLen }, // 7
    { INST_TYPE_END, EndFieldLen }, // 8
    { INST_TYPE_DUMPBANK, DumpBankFieldLen }, // 9
    { INST_TYPE_DUMPDDR, DumpDDRFieldLen }, // 10
    { INST_TYPE_DUMPDDRSLICE, DumpDDRSliceFieldLen }, // 11
};


}

namespace XV_DPU {
// instruction version
const string InstTable::INST_VERSION = "xvdpu";


// category name vector
const vector<string> InstTable::CategoryName = {
    "LOAD",
    "SAVE",
    "CONV",
    "MISC",
};


// inst name vector
const vector<string> InstTable::InstName = {
    "LOAD", // 0
    "SAVE", // 1
    "CONVINIT", // 2
    "CONV", // 3
    "POOLINIT", // 4
    "POOL", // 5
    "ALUINIT", // 6
    "ALU", // 7
    "DWINIT", // 8
    "DPTWISE", // 9
    "ELEWINIT", // 10
    "ELEW", // 11
    "END", // 12
    "DUMPBANK", // 13
    "DUMPDDR", // 14
    "DUMPDDRSLICE", // 15
};


// lower inst name vector
const vector<string> InstTable::InstNameLower = {
    "load", // 0
    "save", // 1
    "convinit", // 2
    "conv", // 3
    "poolinit", // 4
    "pool", // 5
    "aluinit", // 6
    "alu", // 7
    "dwinit", // 8
    "dptwise", // 9
    "elewinit", // 10
    "elew", // 11
    "end", // 12
    "dumpbank", // 13
    "dumpddr", // 14
    "dumpddrslice", // 15
};


// inst category
const vector<Category> InstTable::InstCategory = {
    Category::INST_CATEGORY_LOAD, // LOAD
    Category::INST_CATEGORY_SAVE, // SAVE
    Category::INST_CATEGORY_CONV, // CONVINIT
    Category::INST_CATEGORY_CONV, // CONV
    Category::INST_CATEGORY_MISC, // POOLINIT
    Category::INST_CATEGORY_MISC, // POOL
    Category::INST_CATEGORY_MISC, // ALUINIT
    Category::INST_CATEGORY_MISC, // ALU
    Category::INST_CATEGORY_MISC, // DWINIT
    Category::INST_CATEGORY_MISC, // DPTWISE
    Category::INST_CATEGORY_MISC, // ELEWINIT
    Category::INST_CATEGORY_MISC, // ELEW
    Category::INST_CATEGORY_SAVE, // END
    Category::INST_CATEGORY_MISC, // DUMPBANK
    Category::INST_CATEGORY_MISC, // DUMPDDR
    Category::INST_CATEGORY_MISC, // DUMPDDRSLICE
};


// inst word number
const vector<uint32_t> InstTable::WordNum = {
    6, // LOAD
    5, // SAVE
    6, // CONVINIT
    5, // CONV
    2, // POOLINIT
    5, // POOL
    6, // ALUINIT
    5, // ALU
    4, // DWINIT
    5, // DPTWISE
    3, // ELEWINIT
    3, // ELEW
    1, // END
    2, // DUMPBANK
    4, // DUMPDDR
    6, // DUMPDDRSLICE
};


// inst field number
const vector<uint32_t> InstTable::FieldNum = {
    LOAD_FIELD_MAX, // 0
    SAVE_FIELD_MAX, // 1
    CONVINIT_FIELD_MAX, // 2
    CONV_FIELD_MAX, // 3
    POOLINIT_FIELD_MAX, // 4
    POOL_FIELD_MAX, // 5
    ALUINIT_FIELD_MAX, // 6
    ALU_FIELD_MAX, // 7
    DWINIT_FIELD_MAX, // 8
    DPTWISE_FIELD_MAX, // 9
    ELEWINIT_FIELD_MAX, // 10
    ELEW_FIELD_MAX, // 11
    END_FIELD_MAX, // 12
    DUMPBANK_FIELD_MAX, // 13
    DUMPDDR_FIELD_MAX, // 14
    DUMPDDRSLICE_FIELD_MAX, // 15
};


// inst opcode
const vector<uint32_t> InstTable::OPCode = {
    0x00, // LOAD
    0x04, // SAVE
    0x09, // CONVINIT
    0x08, // CONV
    0x06, // POOLINIT
    0x0C, // POOL
    0x01, // ALUINIT
    0x02, // ALU
    0x0B, // DWINIT
    0x0A, // DPTWISE
    0x0D, // ELEWINIT
    0x0E, // ELEW
    0x07, // END
    0xFF, // DUMPBANK
    0xFE, // DUMPDDR
    0xFD, // DUMPDDRSLICE
};


// Load field name
const vector<string> InstTable::LoadFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "bank_id", // 3
    "bank_addr", // 4
    "jump_read", // 5
    "jump_write", // 6
    "jump_write_endl", // 7
    "block_num", // 8
    "pad_idx", // 9
    "channel", // 10
    "length", // 11
    "pad_start", // 12
    "pad_end", // 13
    "mode_avg", // 14
    "broadcast", // 15
    "const_value", // 16
    "reg_id", // 17
    "ddr_addr", // 18
};


// Save field name
const vector<string> InstTable::SaveFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "bank_id", // 3
    "bank_addr", // 4
    "argmax", // 5
    "jump_write", // 6
    "jump_read", // 7
    "channel", // 8
    "length", // 9
    "const_en", // 10
    "const_value", // 11
    "reg_id", // 12
    "ddr_addr", // 13
};


// ConvInit field name
const vector<string> InstTable::ConvInitFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "tile_en", // 3
    "shift_cut", // 4
    "jump_read_endl", // 5
    "stride_h", // 6
    "stride_w", // 7
    "kernel_h", // 8
    "kernel_w", // 9
    "shift_bias", // 10
    "jump_read", // 11
    "conv_num", // 12
    "act_type", // 13
    "stride_offset_in", // 14
    "stride_offset_out", // 15
    "jump_write_endl", // 16
    "stride_out", // 17
    "tile_icg", // 18
    "tile_ocg", // 19
    "tile_owg", // 20
    "jump_write", // 21
    "ic_iter", // 22
    "channel_offset", // 23
    "ow_iter", // 24
    "ow_offset", // 25
    "jump_read_weights", // 26
    "shift_hswish", // 27
    "shift_hsigmoid", // 28
    "hsigmoid_in", // 29
    "jump_read_weights_endl", // 30
};


// Conv field name
const vector<string> InstTable::ConvFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "bank_id_out", // 3
    "bank_addr_out", // 4
    "icg_offset", // 5
    "bank_id_in", // 6
    "bank_addr_in", // 7
    "pad_left", // 8
    "pad_right", // 9
    "pad_top", // 10
    "pad_bottom", // 11
    "bank_addr_in_1", // 12
    "valid_pixel_parallel", // 13
    "bank_addr_weights", // 14
    "bank_addr_in_2", // 15
    "ocg_offset", // 16
    "bank_addr_bias", // 17
    "bank_addr_in_3", // 18
};


// PoolInit field name
const vector<string> InstTable::PoolInitFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "shift_cut", // 3
    "kernel_w", // 4
    "kernel_h", // 5
    "jump_read", // 6
    "jump_write", // 7
    "stride_out", // 8
    "stride_offset_out", // 9
    "valid_pixel_parallel", // 10
    "stride_offset_in", // 11
    "stride_w", // 12
    "stride_h", // 13
    "pool_type", // 14
};


// Pool field name
const vector<string> InstTable::PoolFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "pad_left", // 3
    "pad_top", // 4
    "bank_addr_in", // 5
    "length", // 6
    "pad_right", // 7
    "pad_bottom", // 8
    "channel_group", // 9
    "bank_id_in", // 10
    "jump_read_endl", // 11
    "jump_write_endl", // 12
    "bank_id_out", // 13
    "bank_addr_in_1", // 14
    "bank_addr_out", // 15
    "bank_addr_in_3", // 16
    "bank_addr_in_2", // 17
};


// AluInit field name
const vector<string> InstTable::AluInitFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "exec_mode", // 3
    "kernel_h", // 4
    "kernel_w", // 5
    "stride_h", // 6
    "stride_w", // 7
    "stride_offset_in", // 8
    "stride_offset_out", // 9
    "stride_out", // 10
    "channel_group", // 11
    "channel_offset", // 12
    "length", // 13
    "shift_prelu_p", // 14
    "act_type", // 15
    "shift_bias", // 16
    "shift_cut", // 17
    "shift_prelu_n", // 18
    "multi_factor", // 19
    "jump_read", // 20
    "jump_read_endl", // 21
    "jump_read_weights", // 22
    "jump_write", // 23
    "jump_write_endl", // 24
};


// Alu field name
const vector<string> InstTable::AluFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "bank_id_out", // 3
    "bank_addr_out", // 4
    "pad_bottom", // 5
    "valid_pixel_parallel", // 6
    "bank_id_in", // 7
    "bank_addr_in", // 8
    "pad_top", // 9
    "pad_left", // 10
    "pad_right", // 11
    "bank_addr_in_1", // 12
    "kernel_d", // 13
    "bank_addr_weights", // 14
    "bank_addr_in_2", // 15
    "bank_addr_bias", // 16
    "bank_addr_in_3", // 17
};


// DWInit field name
const vector<string> InstTable::DWInitFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "channel_group", // 3
    "jump_read_endl", // 4
    "stride_h", // 5
    "stride_w", // 6
    "kernel_h", // 7
    "kernel_w", // 8
    "stride_offset_in", // 9
    "stride_offset_out", // 10
    "jump_read", // 11
    "stride_out", // 12
    "act_type", // 13
    "channel_offset", // 14
    "jump_write_endl", // 15
    "shift_cut", // 16
    "shift_bias", // 17
    "length", // 18
    "jump_write", // 19
};


// DptWise field name
const vector<string> InstTable::DptWiseFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "bank_id_out", // 3
    "bank_addr_out", // 4
    "bank_id_in", // 5
    "bank_addr_in", // 6
    "pad_top", // 7
    "pad_bottom", // 8
    "pad_left", // 9
    "pad_right", // 10
    "bank_addr_in_1", // 11
    "valid_pixel_parallel", // 12
    "bank_addr_weights", // 13
    "bank_addr_in_2", // 14
    "bank_addr_bias", // 15
    "bank_addr_in_3", // 16
};


// ElewInit field name
const vector<string> InstTable::ElewInitFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "bank_id_in", // 3
    "share_pp", // 4
    "jump_bank", // 5
    "id", // 6
    "jump_read_endl", // 7
    "bank_addr_in", // 8
    "shift_read", // 9
    "jump_read", // 10
};


// Elew field name
const vector<string> InstTable::ElewFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "bank_id_out", // 3
    "elew_type", // 4
    "act_type", // 5
    "length", // 6
    "valid_pixel_parallel", // 7
    "num", // 8
    "jump_write_endl", // 9
    "bank_addr_out", // 10
    "channel_group", // 11
    "shift_write", // 12
    "jump_write", // 13
};


// End field name
const vector<string> InstTable::EndFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
};


// DumpBank field name
const vector<string> InstTable::DumpBankFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "save_name", // 3
    "save_fmt", // 4
    "bank_start", // 5
    "bank_num", // 6
};


// DumpDDR field name
const vector<string> InstTable::DumpDDRFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "save_name", // 3
    "save_fmt", // 4
    "reg_id", // 5
    "ddr_start", // 6
    "ddr_size", // 7
};


// DumpDDRSlice field name
const vector<string> InstTable::DumpDDRSliceFieldName = {
    "opcode", // 0
    "dpdon", // 1
    "dpdby", // 2
    "save_name", // 3
    "save_fmt", // 4
    "reg_id", // 5
    "ddr_start", // 6
    "height", // 7
    "height_stride", // 8
    "width", // 9
    "width_stride", // 10
    "channel", // 11
    "channel_stride", // 12
};


// Load field data type
const vector<int> InstTable::LoadFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_UINT, // bank_id(3)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr(4)
    INST_FIELD_DATA_TYPE_UINT, // jump_read(5)
    INST_FIELD_DATA_TYPE_UINT, // jump_write(6)
    INST_FIELD_DATA_TYPE_UINT, // jump_write_endl(7)
    INST_FIELD_DATA_TYPE_UINT, // block_num(8)
    INST_FIELD_DATA_TYPE_UINT, // pad_idx(9)
    INST_FIELD_DATA_TYPE_UINT, // channel(10)
    INST_FIELD_DATA_TYPE_UINT, // length(11)
    INST_FIELD_DATA_TYPE_UINT, // pad_start(12)
    INST_FIELD_DATA_TYPE_UINT, // pad_end(13)
    INST_FIELD_DATA_TYPE_UINT, // mode_avg(14)
    INST_FIELD_DATA_TYPE_UINT, // broadcast(15)
    INST_FIELD_DATA_TYPE_UINT, // const_value(16)
    INST_FIELD_DATA_TYPE_UINT, // reg_id(17)
    INST_FIELD_DATA_TYPE_UINT, // ddr_addr(18)
};


// Save field data type
const vector<int> InstTable::SaveFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_UINT, // bank_id(3)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr(4)
    INST_FIELD_DATA_TYPE_UINT, // argmax(5)
    INST_FIELD_DATA_TYPE_UINT, // jump_write(6)
    INST_FIELD_DATA_TYPE_UINT, // jump_read(7)
    INST_FIELD_DATA_TYPE_UINT, // channel(8)
    INST_FIELD_DATA_TYPE_UINT, // length(9)
    INST_FIELD_DATA_TYPE_UINT, // const_en(10)
    INST_FIELD_DATA_TYPE_UINT, // const_value(11)
    INST_FIELD_DATA_TYPE_UINT, // reg_id(12)
    INST_FIELD_DATA_TYPE_UINT, // ddr_addr(13)
};


// ConvInit field data type
const vector<int> InstTable::ConvInitFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_UINT, // tile_en(3)
    INST_FIELD_DATA_TYPE_UINT, // shift_cut(4)
    INST_FIELD_DATA_TYPE_UINT, // jump_read_endl(5)
    INST_FIELD_DATA_TYPE_UINT, // stride_h(6)
    INST_FIELD_DATA_TYPE_UINT, // stride_w(7)
    INST_FIELD_DATA_TYPE_UINT, // kernel_h(8)
    INST_FIELD_DATA_TYPE_UINT, // kernel_w(9)
    INST_FIELD_DATA_TYPE_UINT, // shift_bias(10)
    INST_FIELD_DATA_TYPE_UINT, // jump_read(11)
    INST_FIELD_DATA_TYPE_UINT, // conv_num(12)
    INST_FIELD_DATA_TYPE_UINT, // act_type(13)
    INST_FIELD_DATA_TYPE_UINT, // stride_offset_in(14)
    INST_FIELD_DATA_TYPE_UINT, // stride_offset_out(15)
    INST_FIELD_DATA_TYPE_UINT, // jump_write_endl(16)
    INST_FIELD_DATA_TYPE_UINT, // stride_out(17)
    INST_FIELD_DATA_TYPE_UINT, // tile_icg(18)
    INST_FIELD_DATA_TYPE_UINT, // tile_ocg(19)
    INST_FIELD_DATA_TYPE_UINT, // tile_owg(20)
    INST_FIELD_DATA_TYPE_UINT, // jump_write(21)
    INST_FIELD_DATA_TYPE_UINT, // ic_iter(22)
    INST_FIELD_DATA_TYPE_UINT, // channel_offset(23)
    INST_FIELD_DATA_TYPE_UINT, // ow_iter(24)
    INST_FIELD_DATA_TYPE_UINT, // ow_offset(25)
    INST_FIELD_DATA_TYPE_UINT, // jump_read_weights(26)
    INST_FIELD_DATA_TYPE_UINT, // shift_hswish(27)
    INST_FIELD_DATA_TYPE_UINT, // shift_hsigmoid(28)
    INST_FIELD_DATA_TYPE_UINT, // hsigmoid_in(29)
    INST_FIELD_DATA_TYPE_UINT, // jump_read_weights_endl(30)
};


// Conv field data type
const vector<int> InstTable::ConvFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_UINT, // bank_id_out(3)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_out(4)
    INST_FIELD_DATA_TYPE_UINT, // icg_offset(5)
    INST_FIELD_DATA_TYPE_UINT, // bank_id_in(6)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in(7)
    INST_FIELD_DATA_TYPE_UINT, // pad_left(8)
    INST_FIELD_DATA_TYPE_UINT, // pad_right(9)
    INST_FIELD_DATA_TYPE_UINT, // pad_top(10)
    INST_FIELD_DATA_TYPE_UINT, // pad_bottom(11)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in_1(12)
    INST_FIELD_DATA_TYPE_UINT, // valid_pixel_parallel(13)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_weights(14)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in_2(15)
    INST_FIELD_DATA_TYPE_UINT, // ocg_offset(16)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_bias(17)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in_3(18)
};


// PoolInit field data type
const vector<int> InstTable::PoolInitFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_UINT, // shift_cut(3)
    INST_FIELD_DATA_TYPE_UINT, // kernel_w(4)
    INST_FIELD_DATA_TYPE_UINT, // kernel_h(5)
    INST_FIELD_DATA_TYPE_UINT, // jump_read(6)
    INST_FIELD_DATA_TYPE_UINT, // jump_write(7)
    INST_FIELD_DATA_TYPE_UINT, // stride_out(8)
    INST_FIELD_DATA_TYPE_UINT, // stride_offset_out(9)
    INST_FIELD_DATA_TYPE_UINT, // valid_pixel_parallel(10)
    INST_FIELD_DATA_TYPE_UINT, // stride_offset_in(11)
    INST_FIELD_DATA_TYPE_UINT, // stride_w(12)
    INST_FIELD_DATA_TYPE_UINT, // stride_h(13)
    INST_FIELD_DATA_TYPE_UINT, // pool_type(14)
};


// Pool field data type
const vector<int> InstTable::PoolFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_UINT, // pad_left(3)
    INST_FIELD_DATA_TYPE_UINT, // pad_top(4)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in(5)
    INST_FIELD_DATA_TYPE_UINT, // length(6)
    INST_FIELD_DATA_TYPE_UINT, // pad_right(7)
    INST_FIELD_DATA_TYPE_UINT, // pad_bottom(8)
    INST_FIELD_DATA_TYPE_UINT, // channel_group(9)
    INST_FIELD_DATA_TYPE_UINT, // bank_id_in(10)
    INST_FIELD_DATA_TYPE_UINT, // jump_read_endl(11)
    INST_FIELD_DATA_TYPE_UINT, // jump_write_endl(12)
    INST_FIELD_DATA_TYPE_UINT, // bank_id_out(13)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in_1(14)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_out(15)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in_3(16)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in_2(17)
};


// AluInit field data type
const vector<int> InstTable::AluInitFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_UINT, // exec_mode(3)
    INST_FIELD_DATA_TYPE_UINT, // kernel_h(4)
    INST_FIELD_DATA_TYPE_UINT, // kernel_w(5)
    INST_FIELD_DATA_TYPE_UINT, // stride_h(6)
    INST_FIELD_DATA_TYPE_UINT, // stride_w(7)
    INST_FIELD_DATA_TYPE_UINT, // stride_offset_in(8)
    INST_FIELD_DATA_TYPE_UINT, // stride_offset_out(9)
    INST_FIELD_DATA_TYPE_UINT, // stride_out(10)
    INST_FIELD_DATA_TYPE_UINT, // channel_group(11)
    INST_FIELD_DATA_TYPE_UINT, // channel_offset(12)
    INST_FIELD_DATA_TYPE_UINT, // length(13)
    INST_FIELD_DATA_TYPE_UINT, // shift_prelu_p(14)
    INST_FIELD_DATA_TYPE_UINT, // act_type(15)
    INST_FIELD_DATA_TYPE_UINT, // shift_bias(16)
    INST_FIELD_DATA_TYPE_UINT, // shift_cut(17)
    INST_FIELD_DATA_TYPE_UINT, // shift_prelu_n(18)
    INST_FIELD_DATA_TYPE_UINT, // multi_factor(19)
    INST_FIELD_DATA_TYPE_UINT, // jump_read(20)
    INST_FIELD_DATA_TYPE_UINT, // jump_read_endl(21)
    INST_FIELD_DATA_TYPE_UINT, // jump_read_weights(22)
    INST_FIELD_DATA_TYPE_UINT, // jump_write(23)
    INST_FIELD_DATA_TYPE_UINT, // jump_write_endl(24)
};


// Alu field data type
const vector<int> InstTable::AluFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_UINT, // bank_id_out(3)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_out(4)
    INST_FIELD_DATA_TYPE_UINT, // pad_bottom(5)
    INST_FIELD_DATA_TYPE_UINT, // valid_pixel_parallel(6)
    INST_FIELD_DATA_TYPE_UINT, // bank_id_in(7)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in(8)
    INST_FIELD_DATA_TYPE_UINT, // pad_top(9)
    INST_FIELD_DATA_TYPE_UINT, // pad_left(10)
    INST_FIELD_DATA_TYPE_UINT, // pad_right(11)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in_1(12)
    INST_FIELD_DATA_TYPE_UINT, // kernel_d(13)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_weights(14)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in_2(15)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_bias(16)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in_3(17)
};


// DWInit field data type
const vector<int> InstTable::DWInitFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_UINT, // channel_group(3)
    INST_FIELD_DATA_TYPE_UINT, // jump_read_endl(4)
    INST_FIELD_DATA_TYPE_UINT, // stride_h(5)
    INST_FIELD_DATA_TYPE_UINT, // stride_w(6)
    INST_FIELD_DATA_TYPE_UINT, // kernel_h(7)
    INST_FIELD_DATA_TYPE_UINT, // kernel_w(8)
    INST_FIELD_DATA_TYPE_UINT, // stride_offset_in(9)
    INST_FIELD_DATA_TYPE_UINT, // stride_offset_out(10)
    INST_FIELD_DATA_TYPE_UINT, // jump_read(11)
    INST_FIELD_DATA_TYPE_UINT, // stride_out(12)
    INST_FIELD_DATA_TYPE_UINT, // act_type(13)
    INST_FIELD_DATA_TYPE_UINT, // channel_offset(14)
    INST_FIELD_DATA_TYPE_UINT, // jump_write_endl(15)
    INST_FIELD_DATA_TYPE_UINT, // shift_cut(16)
    INST_FIELD_DATA_TYPE_UINT, // shift_bias(17)
    INST_FIELD_DATA_TYPE_UINT, // length(18)
    INST_FIELD_DATA_TYPE_UINT, // jump_write(19)
};


// DptWise field data type
const vector<int> InstTable::DptWiseFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_UINT, // bank_id_out(3)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_out(4)
    INST_FIELD_DATA_TYPE_UINT, // bank_id_in(5)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in(6)
    INST_FIELD_DATA_TYPE_UINT, // pad_top(7)
    INST_FIELD_DATA_TYPE_UINT, // pad_bottom(8)
    INST_FIELD_DATA_TYPE_UINT, // pad_left(9)
    INST_FIELD_DATA_TYPE_UINT, // pad_right(10)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in_1(11)
    INST_FIELD_DATA_TYPE_UINT, // valid_pixel_parallel(12)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_weights(13)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in_2(14)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_bias(15)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in_3(16)
};


// ElewInit field data type
const vector<int> InstTable::ElewInitFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_UINT, // bank_id_in(3)
    INST_FIELD_DATA_TYPE_UINT, // share_pp(4)
    INST_FIELD_DATA_TYPE_UINT, // jump_bank(5)
    INST_FIELD_DATA_TYPE_UINT, // id(6)
    INST_FIELD_DATA_TYPE_UINT, // jump_read_endl(7)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_in(8)
    INST_FIELD_DATA_TYPE_UINT, // shift_read(9)
    INST_FIELD_DATA_TYPE_UINT, // jump_read(10)
};


// Elew field data type
const vector<int> InstTable::ElewFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_UINT, // bank_id_out(3)
    INST_FIELD_DATA_TYPE_UINT, // elew_type(4)
    INST_FIELD_DATA_TYPE_UINT, // act_type(5)
    INST_FIELD_DATA_TYPE_UINT, // length(6)
    INST_FIELD_DATA_TYPE_UINT, // valid_pixel_parallel(7)
    INST_FIELD_DATA_TYPE_UINT, // num(8)
    INST_FIELD_DATA_TYPE_UINT, // jump_write_endl(9)
    INST_FIELD_DATA_TYPE_UINT, // bank_addr_out(10)
    INST_FIELD_DATA_TYPE_UINT, // channel_group(11)
    INST_FIELD_DATA_TYPE_UINT, // shift_write(12)
    INST_FIELD_DATA_TYPE_UINT, // jump_write(13)
};


// End field data type
const vector<int> InstTable::EndFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
};


// DumpBank field data type
const vector<int> InstTable::DumpBankFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_STRING, // save_name(3)
    INST_FIELD_DATA_TYPE_UINT, // save_fmt(4)
    INST_FIELD_DATA_TYPE_UINT, // bank_start(5)
    INST_FIELD_DATA_TYPE_UINT, // bank_num(6)
};


// DumpDDR field data type
const vector<int> InstTable::DumpDDRFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_STRING, // save_name(3)
    INST_FIELD_DATA_TYPE_UINT, // save_fmt(4)
    INST_FIELD_DATA_TYPE_UINT, // reg_id(5)
    INST_FIELD_DATA_TYPE_UINT, // ddr_start(6)
    INST_FIELD_DATA_TYPE_UINT, // ddr_size(7)
};


// DumpDDRSlice field data type
const vector<int> InstTable::DumpDDRSliceFieldDataType = {
    INST_FIELD_DATA_TYPE_STRING, // opcode(0)
    INST_FIELD_DATA_TYPE_STRING, // dpdon(1)
    INST_FIELD_DATA_TYPE_STRING, // dpdby(2)
    INST_FIELD_DATA_TYPE_STRING, // save_name(3)
    INST_FIELD_DATA_TYPE_UINT, // save_fmt(4)
    INST_FIELD_DATA_TYPE_UINT, // reg_id(5)
    INST_FIELD_DATA_TYPE_UINT, // ddr_start(6)
    INST_FIELD_DATA_TYPE_UINT, // height(7)
    INST_FIELD_DATA_TYPE_UINT, // height_stride(8)
    INST_FIELD_DATA_TYPE_UINT, // width(9)
    INST_FIELD_DATA_TYPE_UINT, // width_stride(10)
    INST_FIELD_DATA_TYPE_UINT, // channel(11)
    INST_FIELD_DATA_TYPE_UINT, // channel_stride(12)
};


// Load field minus value
const vector<int> InstTable::LoadFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // bank_id(3)
    0, // bank_addr(4)
    1, // jump_read(5)
    1, // jump_write(6)
    1, // jump_write_endl(7)
    1, // block_num(8)
    1, // pad_idx(9)
    1, // channel(10)
    1, // length(11)
    0, // pad_start(12)
    0, // pad_end(13)
    0, // mode_avg(14)
    0, // broadcast(15)
    0, // const_value(16)
    0, // reg_id(17)
    0, // ddr_addr(18)
};


// Save field minus value
const vector<int> InstTable::SaveFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // bank_id(3)
    0, // bank_addr(4)
    0, // argmax(5)
    1, // jump_write(6)
    1, // jump_read(7)
    1, // channel(8)
    1, // length(9)
    0, // const_en(10)
    0, // const_value(11)
    0, // reg_id(12)
    0, // ddr_addr(13)
};


// ConvInit field minus value
const vector<int> InstTable::ConvInitFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // tile_en(3)
    0, // shift_cut(4)
    1, // jump_read_endl(5)
    1, // stride_h(6)
    1, // stride_w(7)
    1, // kernel_h(8)
    1, // kernel_w(9)
    0, // shift_bias(10)
    1, // jump_read(11)
    1, // conv_num(12)
    0, // act_type(13)
    0, // stride_offset_in(14)
    0, // stride_offset_out(15)
    1, // jump_write_endl(16)
    1, // stride_out(17)
    1, // tile_icg(18)
    1, // tile_ocg(19)
    1, // tile_owg(20)
    1, // jump_write(21)
    1, // ic_iter(22)
    0, // channel_offset(23)
    1, // ow_iter(24)
    0, // ow_offset(25)
    1, // jump_read_weights(26)
    0, // shift_hswish(27)
    0, // shift_hsigmoid(28)
    0, // hsigmoid_in(29)
    1, // jump_read_weights_endl(30)
};


// Conv field minus value
const vector<int> InstTable::ConvFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // bank_id_out(3)
    0, // bank_addr_out(4)
    0, // icg_offset(5)
    0, // bank_id_in(6)
    0, // bank_addr_in(7)
    0, // pad_left(8)
    0, // pad_right(9)
    0, // pad_top(10)
    0, // pad_bottom(11)
    0, // bank_addr_in_1(12)
    1, // valid_pixel_parallel(13)
    0, // bank_addr_weights(14)
    0, // bank_addr_in_2(15)
    0, // ocg_offset(16)
    0, // bank_addr_bias(17)
    0, // bank_addr_in_3(18)
};


// PoolInit field minus value
const vector<int> InstTable::PoolInitFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // shift_cut(3)
    1, // kernel_w(4)
    1, // kernel_h(5)
    1, // jump_read(6)
    1, // jump_write(7)
    1, // stride_out(8)
    0, // stride_offset_out(9)
    1, // valid_pixel_parallel(10)
    0, // stride_offset_in(11)
    1, // stride_w(12)
    1, // stride_h(13)
    0, // pool_type(14)
};


// Pool field minus value
const vector<int> InstTable::PoolFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // pad_left(3)
    0, // pad_top(4)
    0, // bank_addr_in(5)
    1, // length(6)
    0, // pad_right(7)
    0, // pad_bottom(8)
    1, // channel_group(9)
    0, // bank_id_in(10)
    1, // jump_read_endl(11)
    1, // jump_write_endl(12)
    0, // bank_id_out(13)
    0, // bank_addr_in_1(14)
    0, // bank_addr_out(15)
    0, // bank_addr_in_3(16)
    0, // bank_addr_in_2(17)
};


// AluInit field minus value
const vector<int> InstTable::AluInitFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // exec_mode(3)
    1, // kernel_h(4)
    1, // kernel_w(5)
    1, // stride_h(6)
    1, // stride_w(7)
    0, // stride_offset_in(8)
    0, // stride_offset_out(9)
    1, // stride_out(10)
    1, // channel_group(11)
    0, // channel_offset(12)
    1, // length(13)
    0, // shift_prelu_p(14)
    0, // act_type(15)
    0, // shift_bias(16)
    0, // shift_cut(17)
    0, // shift_prelu_n(18)
    0, // multi_factor(19)
    1, // jump_read(20)
    1, // jump_read_endl(21)
    0, // jump_read_weights(22)
    1, // jump_write(23)
    1, // jump_write_endl(24)
};


// Alu field minus value
const vector<int> InstTable::AluFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // bank_id_out(3)
    0, // bank_addr_out(4)
    0, // pad_bottom(5)
    1, // valid_pixel_parallel(6)
    0, // bank_id_in(7)
    0, // bank_addr_in(8)
    0, // pad_top(9)
    0, // pad_left(10)
    0, // pad_right(11)
    0, // bank_addr_in_1(12)
    1, // kernel_d(13)
    0, // bank_addr_weights(14)
    0, // bank_addr_in_2(15)
    0, // bank_addr_bias(16)
    0, // bank_addr_in_3(17)
};


// DWInit field minus value
const vector<int> InstTable::DWInitFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    1, // channel_group(3)
    1, // jump_read_endl(4)
    1, // stride_h(5)
    1, // stride_w(6)
    1, // kernel_h(7)
    1, // kernel_w(8)
    0, // stride_offset_in(9)
    0, // stride_offset_out(10)
    1, // jump_read(11)
    1, // stride_out(12)
    0, // act_type(13)
    0, // channel_offset(14)
    1, // jump_write_endl(15)
    0, // shift_cut(16)
    0, // shift_bias(17)
    1, // length(18)
    1, // jump_write(19)
};


// DptWise field minus value
const vector<int> InstTable::DptWiseFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // bank_id_out(3)
    0, // bank_addr_out(4)
    0, // bank_id_in(5)
    0, // bank_addr_in(6)
    0, // pad_top(7)
    0, // pad_bottom(8)
    0, // pad_left(9)
    0, // pad_right(10)
    0, // bank_addr_in_1(11)
    1, // valid_pixel_parallel(12)
    0, // bank_addr_weights(13)
    0, // bank_addr_in_2(14)
    0, // bank_addr_bias(15)
    0, // bank_addr_in_3(16)
};


// ElewInit field minus value
const vector<int> InstTable::ElewInitFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // bank_id_in(3)
    0, // share_pp(4)
    1, // jump_bank(5)
    0, // id(6)
    1, // jump_read_endl(7)
    0, // bank_addr_in(8)
    0, // shift_read(9)
    0, // jump_read(10)
};


// Elew field minus value
const vector<int> InstTable::ElewFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // bank_id_out(3)
    0, // elew_type(4)
    0, // act_type(5)
    1, // length(6)
    1, // valid_pixel_parallel(7)
    1, // num(8)
    1, // jump_write_endl(9)
    0, // bank_addr_out(10)
    1, // channel_group(11)
    0, // shift_write(12)
    1, // jump_write(13)
};


// End field minus value
const vector<int> InstTable::EndFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
};


// DumpBank field minus value
const vector<int> InstTable::DumpBankFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // save_name(3)
    0, // save_fmt(4)
    0, // bank_start(5)
    0, // bank_num(6)
};


// DumpDDR field minus value
const vector<int> InstTable::DumpDDRFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // save_name(3)
    0, // save_fmt(4)
    0, // reg_id(5)
    0, // ddr_start(6)
    0, // ddr_size(7)
};


// DumpDDRSlice field minus value
const vector<int> InstTable::DumpDDRSliceFieldMinus = {
    0, // opcode(0)
    0, // dpdon(1)
    0, // dpdby(2)
    0, // save_name(3)
    0, // save_fmt(4)
    0, // reg_id(5)
    0, // ddr_start(6)
    0, // height(7)
    0, // height_stride(8)
    0, // width(9)
    0, // width_stride(10)
    0, // channel(11)
    0, // channel_stride(12)
};


// Load field position
const vector<map<uint32_t, uint32_t>> InstTable::LoadFieldPos = {
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_OPCODE, 28 }, // 0
        { LOAD_FIELD_DPDON, 24 }, // 1
        { LOAD_FIELD_DPDBY, 20 }, // 2
        { LOAD_FIELD_BANK_ID, 14 }, // 3
        { LOAD_FIELD_BANK_ADDR, 0 }, // 4
    },
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_JUMP_READ, 10 }, // 5
        { LOAD_FIELD_JUMP_WRITE, 0 }, // 6
    },
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_JUMP_WRITE_ENDL, 10 }, // 7
        { LOAD_FIELD_BLOCK_NUM, 0 }, // 8
    },
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_PAD_IDX, 27 }, // 9
        { LOAD_FIELD_CHANNEL, 10 }, // 10
        { LOAD_FIELD_LENGTH, 0 }, // 11
    },
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_PAD_START, 27 }, // 12
        { LOAD_FIELD_PAD_END, 22 }, // 13
        { LOAD_FIELD_MODE_AVG, 20 }, // 14
        { LOAD_FIELD_BROADCAST, 16 }, // 15
        { LOAD_FIELD_CONST_VALUE, 8 }, // 16
        { LOAD_FIELD_REG_ID, 0 }, // 17
    },
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_DDR_ADDR, 0 }, // 18
    },
};


// Save field position
const vector<map<uint32_t, uint32_t>> InstTable::SaveFieldPos = {
    map<uint32_t, uint32_t> {
        { SAVE_FIELD_OPCODE, 28 }, // 0
        { SAVE_FIELD_DPDON, 24 }, // 1
        { SAVE_FIELD_DPDBY, 20 }, // 2
        { SAVE_FIELD_BANK_ID, 14 }, // 3
        { SAVE_FIELD_BANK_ADDR, 0 }, // 4
    },
    map<uint32_t, uint32_t> {
        { SAVE_FIELD_ARGMAX, 30 }, // 5
        { SAVE_FIELD_JUMP_WRITE, 10 }, // 6
        { SAVE_FIELD_JUMP_READ, 0 }, // 7
    },
    map<uint32_t, uint32_t> {
        { SAVE_FIELD_CHANNEL, 10 }, // 8
        { SAVE_FIELD_LENGTH, 0 }, // 9
    },
    map<uint32_t, uint32_t> {
        { SAVE_FIELD_CONST_EN, 16 }, // 10
        { SAVE_FIELD_CONST_VALUE, 8 }, // 11
        { SAVE_FIELD_REG_ID, 0 }, // 12
    },
    map<uint32_t, uint32_t> {
        { SAVE_FIELD_DDR_ADDR, 0 }, // 13
    },
};


// ConvInit field position
const vector<map<uint32_t, uint32_t>> InstTable::ConvInitFieldPos = {
    map<uint32_t, uint32_t> {
        { CONVINIT_FIELD_OPCODE, 28 }, // 0
        { CONVINIT_FIELD_DPDON, 24 }, // 1
        { CONVINIT_FIELD_DPDBY, 20 }, // 2
        { CONVINIT_FIELD_TILE_EN, 19 }, // 3
        { CONVINIT_FIELD_SHIFT_CUT, 13 }, // 4
        { CONVINIT_FIELD_JUMP_READ_ENDL, 0 }, // 5
    },
    map<uint32_t, uint32_t> {
        { CONVINIT_FIELD_STRIDE_H, 28 }, // 6
        { CONVINIT_FIELD_STRIDE_W, 24 }, // 7
        { CONVINIT_FIELD_KERNEL_H, 20 }, // 8
        { CONVINIT_FIELD_KERNEL_W, 16 }, // 9
        { CONVINIT_FIELD_SHIFT_BIAS, 10 }, // 10
        { CONVINIT_FIELD_JUMP_READ, 0 }, // 11
    },
    map<uint32_t, uint32_t> {
        { CONVINIT_FIELD_CONV_NUM, 22 }, // 12
        { CONVINIT_FIELD_ACT_TYPE, 19 }, // 13
        { CONVINIT_FIELD_STRIDE_OFFSET_IN, 16 }, // 14
        { CONVINIT_FIELD_STRIDE_OFFSET_OUT, 13 }, // 15
        { CONVINIT_FIELD_JUMP_WRITE_ENDL, 0 }, // 16
    },
    map<uint32_t, uint32_t> {
        { CONVINIT_FIELD_STRIDE_OUT, 28 }, // 17
        { CONVINIT_FIELD_TILE_ICG, 22 }, // 18
        { CONVINIT_FIELD_TILE_OCG, 16 }, // 19
        { CONVINIT_FIELD_TILE_OWG, 10 }, // 20
        { CONVINIT_FIELD_JUMP_WRITE, 0 }, // 21
    },
    map<uint32_t, uint32_t> {
        { CONVINIT_FIELD_IC_ITER, 26 }, // 22
        { CONVINIT_FIELD_CHANNEL_OFFSET, 21 }, // 23
        { CONVINIT_FIELD_OW_ITER, 16 }, // 24
        { CONVINIT_FIELD_OW_OFFSET, 10 }, // 25
        { CONVINIT_FIELD_JUMP_READ_WEIGHTS, 0 }, // 26
    },
    map<uint32_t, uint32_t> {
        { CONVINIT_FIELD_SHIFT_HSWISH, 22 }, // 27
        { CONVINIT_FIELD_SHIFT_HSIGMOID, 17 }, // 28
        { CONVINIT_FIELD_HSIGMOID_IN, 13 }, // 29
        { CONVINIT_FIELD_JUMP_READ_WEIGHTS_ENDL, 0 }, // 30
    },
};


// Conv field position
const vector<map<uint32_t, uint32_t>> InstTable::ConvFieldPos = {
    map<uint32_t, uint32_t> {
        { CONV_FIELD_OPCODE, 28 }, // 0
        { CONV_FIELD_DPDON, 24 }, // 1
        { CONV_FIELD_DPDBY, 20 }, // 2
        { CONV_FIELD_BANK_ID_OUT, 14 }, // 3
        { CONV_FIELD_BANK_ADDR_OUT, 0 }, // 4
    },
    map<uint32_t, uint32_t> {
        { CONV_FIELD_ICG_OFFSET, 20 }, // 5
        { CONV_FIELD_BANK_ID_IN, 14 }, // 6
        { CONV_FIELD_BANK_ADDR_IN, 0 }, // 7
    },
    map<uint32_t, uint32_t> {
        { CONV_FIELD_PAD_LEFT, 26 }, // 8
        { CONV_FIELD_PAD_RIGHT, 22 }, // 9
        { CONV_FIELD_PAD_TOP, 18 }, // 10
        { CONV_FIELD_PAD_BOTTOM, 14 }, // 11
        { CONV_FIELD_BANK_ADDR_IN_1, 0 }, // 12
    },
    map<uint32_t, uint32_t> {
        { CONV_FIELD_VALID_PIXEL_PARALLEL, 28 }, // 13
        { CONV_FIELD_BANK_ADDR_WEIGHTS, 14 }, // 14
        { CONV_FIELD_BANK_ADDR_IN_2, 0 }, // 15
    },
    map<uint32_t, uint32_t> {
        { CONV_FIELD_OCG_OFFSET, 28 }, // 16
        { CONV_FIELD_BANK_ADDR_BIAS, 14 }, // 17
        { CONV_FIELD_BANK_ADDR_IN_3, 0 }, // 18
    },
};


// PoolInit field position
const vector<map<uint32_t, uint32_t>> InstTable::PoolInitFieldPos = {
    map<uint32_t, uint32_t> {
        { POOLINIT_FIELD_OPCODE, 28 }, // 0
        { POOLINIT_FIELD_DPDON, 24 }, // 1
        { POOLINIT_FIELD_DPDBY, 20 }, // 2
        { POOLINIT_FIELD_SHIFT_CUT, 16 }, // 3
        { POOLINIT_FIELD_KERNEL_W, 13 }, // 4
        { POOLINIT_FIELD_KERNEL_H, 10 }, // 5
        { POOLINIT_FIELD_JUMP_READ, 0 }, // 6
    },
    map<uint32_t, uint32_t> {
        { POOLINIT_FIELD_JUMP_WRITE, 21 }, // 7
        { POOLINIT_FIELD_STRIDE_OUT, 17 }, // 8
        { POOLINIT_FIELD_STRIDE_OFFSET_OUT, 14 }, // 9
        { POOLINIT_FIELD_VALID_PIXEL_PARALLEL, 11 }, // 10
        { POOLINIT_FIELD_STRIDE_OFFSET_IN, 8 }, // 11
        { POOLINIT_FIELD_STRIDE_W, 5 }, // 12
        { POOLINIT_FIELD_STRIDE_H, 2 }, // 13
        { POOLINIT_FIELD_POOL_TYPE, 0 }, // 14
    },
};


// Pool field position
const vector<map<uint32_t, uint32_t>> InstTable::PoolFieldPos = {
    map<uint32_t, uint32_t> {
        { POOL_FIELD_OPCODE, 28 }, // 0
        { POOL_FIELD_DPDON, 24 }, // 1
        { POOL_FIELD_DPDBY, 20 }, // 2
        { POOL_FIELD_PAD_LEFT, 17 }, // 3
        { POOL_FIELD_PAD_TOP, 14 }, // 4
        { POOL_FIELD_BANK_ADDR_IN, 0 }, // 5
    },
    map<uint32_t, uint32_t> {
        { POOL_FIELD_LENGTH, 20 }, // 6
        { POOL_FIELD_PAD_RIGHT, 17 }, // 7
        { POOL_FIELD_PAD_BOTTOM, 14 }, // 8
        { POOL_FIELD_CHANNEL_GROUP, 6 }, // 9
        { POOL_FIELD_BANK_ID_IN, 0 }, // 10
    },
    map<uint32_t, uint32_t> {
        { POOL_FIELD_JUMP_READ_ENDL, 19 }, // 11
        { POOL_FIELD_JUMP_WRITE_ENDL, 6 }, // 12
        { POOL_FIELD_BANK_ID_OUT, 0 }, // 13
    },
    map<uint32_t, uint32_t> {
        { POOL_FIELD_BANK_ADDR_IN_1, 14 }, // 14
        { POOL_FIELD_BANK_ADDR_OUT, 0 }, // 15
    },
    map<uint32_t, uint32_t> {
        { POOL_FIELD_BANK_ADDR_IN_3, 14 }, // 16
        { POOL_FIELD_BANK_ADDR_IN_2, 0 }, // 17
    },
};


// AluInit field position
const vector<map<uint32_t, uint32_t>> InstTable::AluInitFieldPos = {
    map<uint32_t, uint32_t> {
        { ALUINIT_FIELD_OPCODE, 28 }, // 0
        { ALUINIT_FIELD_DPDON, 24 }, // 1
        { ALUINIT_FIELD_DPDBY, 20 }, // 2
        { ALUINIT_FIELD_EXEC_MODE, 16 }, // 3
        { ALUINIT_FIELD_KERNEL_H, 8 }, // 4
        { ALUINIT_FIELD_KERNEL_W, 0 }, // 5
    },
    map<uint32_t, uint32_t> {
        { ALUINIT_FIELD_STRIDE_H, 24 }, // 6
        { ALUINIT_FIELD_STRIDE_W, 16 }, // 7
        { ALUINIT_FIELD_STRIDE_OFFSET_IN, 8 }, // 8
        { ALUINIT_FIELD_STRIDE_OFFSET_OUT, 0 }, // 9
    },
    map<uint32_t, uint32_t> {
        { ALUINIT_FIELD_STRIDE_OUT, 24 }, // 10
        { ALUINIT_FIELD_CHANNEL_GROUP, 16 }, // 11
        { ALUINIT_FIELD_CHANNEL_OFFSET, 11 }, // 12
        { ALUINIT_FIELD_LENGTH, 0 }, // 13
    },
    map<uint32_t, uint32_t> {
        { ALUINIT_FIELD_SHIFT_PRELU_P, 22 }, // 14
        { ALUINIT_FIELD_ACT_TYPE, 18 }, // 15
        { ALUINIT_FIELD_SHIFT_BIAS, 12 }, // 16
        { ALUINIT_FIELD_SHIFT_CUT, 6 }, // 17
        { ALUINIT_FIELD_SHIFT_PRELU_N, 0 }, // 18
    },
    map<uint32_t, uint32_t> {
        { ALUINIT_FIELD_MULTI_FACTOR, 23 }, // 19
        { ALUINIT_FIELD_JUMP_READ, 13 }, // 20
        { ALUINIT_FIELD_JUMP_READ_ENDL, 0 }, // 21
    },
    map<uint32_t, uint32_t> {
        { ALUINIT_FIELD_JUMP_READ_WEIGHTS, 23 }, // 22
        { ALUINIT_FIELD_JUMP_WRITE, 13 }, // 23
        { ALUINIT_FIELD_JUMP_WRITE_ENDL, 0 }, // 24
    },
};


// Alu field position
const vector<map<uint32_t, uint32_t>> InstTable::AluFieldPos = {
    map<uint32_t, uint32_t> {
        { ALU_FIELD_OPCODE, 28 }, // 0
        { ALU_FIELD_DPDON, 24 }, // 1
        { ALU_FIELD_DPDBY, 20 }, // 2
        { ALU_FIELD_BANK_ID_OUT, 14 }, // 3
        { ALU_FIELD_BANK_ADDR_OUT, 0 }, // 4
    },
    map<uint32_t, uint32_t> {
        { ALU_FIELD_PAD_BOTTOM, 23 }, // 5
        { ALU_FIELD_VALID_PIXEL_PARALLEL, 20 }, // 6
        { ALU_FIELD_BANK_ID_IN, 14 }, // 7
        { ALU_FIELD_BANK_ADDR_IN, 0 }, // 8
    },
    map<uint32_t, uint32_t> {
        { ALU_FIELD_PAD_TOP, 26 }, // 9
        { ALU_FIELD_PAD_LEFT, 22 }, // 10
        { ALU_FIELD_PAD_RIGHT, 14 }, // 11
        { ALU_FIELD_BANK_ADDR_IN_1, 0 }, // 12
    },
    map<uint32_t, uint32_t> {
        { ALU_FIELD_KERNEL_D, 28 }, // 13
        { ALU_FIELD_BANK_ADDR_WEIGHTS, 14 }, // 14
        { ALU_FIELD_BANK_ADDR_IN_2, 0 }, // 15
    },
    map<uint32_t, uint32_t> {
        { ALU_FIELD_BANK_ADDR_BIAS, 14 }, // 16
        { ALU_FIELD_BANK_ADDR_IN_3, 0 }, // 17
    },
};


// DWInit field position
const vector<map<uint32_t, uint32_t>> InstTable::DWInitFieldPos = {
    map<uint32_t, uint32_t> {
        { DWINIT_FIELD_OPCODE, 28 }, // 0
        { DWINIT_FIELD_DPDON, 24 }, // 1
        { DWINIT_FIELD_DPDBY, 20 }, // 2
        { DWINIT_FIELD_CHANNEL_GROUP, 12 }, // 3
        { DWINIT_FIELD_JUMP_READ_ENDL, 0 }, // 4
    },
    map<uint32_t, uint32_t> {
        { DWINIT_FIELD_STRIDE_H, 28 }, // 5
        { DWINIT_FIELD_STRIDE_W, 24 }, // 6
        { DWINIT_FIELD_KERNEL_H, 20 }, // 7
        { DWINIT_FIELD_KERNEL_W, 16 }, // 8
        { DWINIT_FIELD_STRIDE_OFFSET_IN, 13 }, // 9
        { DWINIT_FIELD_STRIDE_OFFSET_OUT, 10 }, // 10
        { DWINIT_FIELD_JUMP_READ, 0 }, // 11
    },
    map<uint32_t, uint32_t> {
        { DWINIT_FIELD_STRIDE_OUT, 28 }, // 12
        { DWINIT_FIELD_ACT_TYPE, 24 }, // 13
        { DWINIT_FIELD_CHANNEL_OFFSET, 19 }, // 14
        { DWINIT_FIELD_JUMP_WRITE_ENDL, 0 }, // 15
    },
    map<uint32_t, uint32_t> {
        { DWINIT_FIELD_SHIFT_CUT, 26 }, // 16
        { DWINIT_FIELD_SHIFT_BIAS, 20 }, // 17
        { DWINIT_FIELD_LENGTH, 10 }, // 18
        { DWINIT_FIELD_JUMP_WRITE, 0 }, // 19
    },
};


// DptWise field position
const vector<map<uint32_t, uint32_t>> InstTable::DptWiseFieldPos = {
    map<uint32_t, uint32_t> {
        { DPTWISE_FIELD_OPCODE, 28 }, // 0
        { DPTWISE_FIELD_DPDON, 24 }, // 1
        { DPTWISE_FIELD_DPDBY, 20 }, // 2
        { DPTWISE_FIELD_BANK_ID_OUT, 14 }, // 3
        { DPTWISE_FIELD_BANK_ADDR_OUT, 0 }, // 4
    },
    map<uint32_t, uint32_t> {
        { DPTWISE_FIELD_BANK_ID_IN, 14 }, // 5
        { DPTWISE_FIELD_BANK_ADDR_IN, 0 }, // 6
    },
    map<uint32_t, uint32_t> {
        { DPTWISE_FIELD_PAD_TOP, 26 }, // 7
        { DPTWISE_FIELD_PAD_BOTTOM, 22 }, // 8
        { DPTWISE_FIELD_PAD_LEFT, 18 }, // 9
        { DPTWISE_FIELD_PAD_RIGHT, 14 }, // 10
        { DPTWISE_FIELD_BANK_ADDR_IN_1, 0 }, // 11
    },
    map<uint32_t, uint32_t> {
        { DPTWISE_FIELD_VALID_PIXEL_PARALLEL, 28 }, // 12
        { DPTWISE_FIELD_BANK_ADDR_WEIGHTS, 14 }, // 13
        { DPTWISE_FIELD_BANK_ADDR_IN_2, 0 }, // 14
    },
    map<uint32_t, uint32_t> {
        { DPTWISE_FIELD_BANK_ADDR_BIAS, 14 }, // 15
        { DPTWISE_FIELD_BANK_ADDR_IN_3, 0 }, // 16
    },
};


// ElewInit field position
const vector<map<uint32_t, uint32_t>> InstTable::ElewInitFieldPos = {
    map<uint32_t, uint32_t> {
        { ELEWINIT_FIELD_OPCODE, 28 }, // 0
        { ELEWINIT_FIELD_DPDON, 24 }, // 1
        { ELEWINIT_FIELD_DPDBY, 20 }, // 2
        { ELEWINIT_FIELD_BANK_ID_IN, 14 }, // 3
        { ELEWINIT_FIELD_SHARE_PP, 13 }, // 4
    },
    map<uint32_t, uint32_t> {
        { ELEWINIT_FIELD_JUMP_BANK, 29 }, // 5
        { ELEWINIT_FIELD_ID, 27 }, // 6
        { ELEWINIT_FIELD_JUMP_READ_ENDL, 14 }, // 7
        { ELEWINIT_FIELD_BANK_ADDR_IN, 0 }, // 8
    },
    map<uint32_t, uint32_t> {
        { ELEWINIT_FIELD_SHIFT_READ, 12 }, // 9
        { ELEWINIT_FIELD_JUMP_READ, 0 }, // 10
    },
};


// Elew field position
const vector<map<uint32_t, uint32_t>> InstTable::ElewFieldPos = {
    map<uint32_t, uint32_t> {
        { ELEW_FIELD_OPCODE, 28 }, // 0
        { ELEW_FIELD_DPDON, 24 }, // 1
        { ELEW_FIELD_DPDBY, 20 }, // 2
        { ELEW_FIELD_BANK_ID_OUT, 14 }, // 3
        { ELEW_FIELD_ELEW_TYPE, 12 }, // 4
        { ELEW_FIELD_ACT_TYPE, 11 }, // 5
        { ELEW_FIELD_LENGTH, 0 }, // 6
    },
    map<uint32_t, uint32_t> {
        { ELEW_FIELD_VALID_PIXEL_PARALLEL, 29 }, // 7
        { ELEW_FIELD_NUM, 27 }, // 8
        { ELEW_FIELD_JUMP_WRITE_ENDL, 14 }, // 9
        { ELEW_FIELD_BANK_ADDR_OUT, 0 }, // 10
    },
    map<uint32_t, uint32_t> {
        { ELEW_FIELD_CHANNEL_GROUP, 17 }, // 11
        { ELEW_FIELD_SHIFT_WRITE, 12 }, // 12
        { ELEW_FIELD_JUMP_WRITE, 0 }, // 13
    },
};


// End field position
const vector<map<uint32_t, uint32_t>> InstTable::EndFieldPos = {
    map<uint32_t, uint32_t> {
        { END_FIELD_OPCODE, 28 }, // 0
        { END_FIELD_DPDON, 24 }, // 1
        { END_FIELD_DPDBY, 20 }, // 2
    },
};


// DumpBank field position
const vector<map<uint32_t, uint32_t>> InstTable::DumpBankFieldPos = {
    map<uint32_t, uint32_t> {
        { DUMPBANK_FIELD_OPCODE, 28 }, // 0
        { DUMPBANK_FIELD_DPDON, 24 }, // 1
        { DUMPBANK_FIELD_DPDBY, 20 }, // 2
        { DUMPBANK_FIELD_SAVE_NAME, 0 }, // 3
    },
    map<uint32_t, uint32_t> {
        { DUMPBANK_FIELD_SAVE_FMT, 26 }, // 4
        { DUMPBANK_FIELD_BANK_START, 18 }, // 5
        { DUMPBANK_FIELD_BANK_NUM, 10 }, // 6
    },
};


// DumpDDR field position
const vector<map<uint32_t, uint32_t>> InstTable::DumpDDRFieldPos = {
    map<uint32_t, uint32_t> {
        { DUMPDDR_FIELD_OPCODE, 28 }, // 0
        { DUMPDDR_FIELD_DPDON, 24 }, // 1
        { DUMPDDR_FIELD_DPDBY, 20 }, // 2
        { DUMPDDR_FIELD_SAVE_NAME, 0 }, // 3
    },
    map<uint32_t, uint32_t> {
        { DUMPDDR_FIELD_SAVE_FMT, 26 }, // 4
        { DUMPDDR_FIELD_REG_ID, 20 }, // 5
    },
    map<uint32_t, uint32_t> {
        { DUMPDDR_FIELD_DDR_START, 0 }, // 6
    },
    map<uint32_t, uint32_t> {
        { DUMPDDR_FIELD_DDR_SIZE, 0 }, // 7
    },
};


// DumpDDRSlice field position
const vector<map<uint32_t, uint32_t>> InstTable::DumpDDRSliceFieldPos = {
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_OPCODE, 28 }, // 0
        { DUMPDDRSLICE_FIELD_DPDON, 24 }, // 1
        { DUMPDDRSLICE_FIELD_DPDBY, 20 }, // 2
        { DUMPDDRSLICE_FIELD_SAVE_NAME, 0 }, // 3
    },
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_SAVE_FMT, 26 }, // 4
        { DUMPDDRSLICE_FIELD_REG_ID, 20 }, // 5
    },
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_DDR_START, 0 }, // 6
    },
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_HEIGHT, 16 }, // 7
        { DUMPDDRSLICE_FIELD_HEIGHT_STRIDE, 0 }, // 8
    },
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_WIDTH, 16 }, // 9
        { DUMPDDRSLICE_FIELD_WIDTH_STRIDE, 0 }, // 10
    },
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_CHANNEL, 16 }, // 11
        { DUMPDDRSLICE_FIELD_CHANNEL_STRIDE, 0 }, // 12
    },
};


// Load field length
const vector<map<uint32_t, uint32_t>> InstTable::LoadFieldLen = {
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_OPCODE, 4 }, // 0
        { LOAD_FIELD_DPDON, 4 }, // 1
        { LOAD_FIELD_DPDBY, 4 }, // 2
        { LOAD_FIELD_BANK_ID, 6 }, // 3
        { LOAD_FIELD_BANK_ADDR, 14 }, // 4
    },
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_JUMP_READ, 16 }, // 5
        { LOAD_FIELD_JUMP_WRITE, 10 }, // 6
    },
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_JUMP_WRITE_ENDL, 14 }, // 7
        { LOAD_FIELD_BLOCK_NUM, 10 }, // 8
    },
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_PAD_IDX, 5 }, // 9
        { LOAD_FIELD_CHANNEL, 14 }, // 10
        { LOAD_FIELD_LENGTH, 10 }, // 11
    },
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_PAD_START, 5 }, // 12
        { LOAD_FIELD_PAD_END, 5 }, // 13
        { LOAD_FIELD_MODE_AVG, 2 }, // 14
        { LOAD_FIELD_BROADCAST, 4 }, // 15
        { LOAD_FIELD_CONST_VALUE, 8 }, // 16
        { LOAD_FIELD_REG_ID, 3 }, // 17
    },
    map<uint32_t, uint32_t> {
        { LOAD_FIELD_DDR_ADDR, 29 }, // 18
    },
};


// Save field length
const vector<map<uint32_t, uint32_t>> InstTable::SaveFieldLen = {
    map<uint32_t, uint32_t> {
        { SAVE_FIELD_OPCODE, 4 }, // 0
        { SAVE_FIELD_DPDON, 4 }, // 1
        { SAVE_FIELD_DPDBY, 4 }, // 2
        { SAVE_FIELD_BANK_ID, 6 }, // 3
        { SAVE_FIELD_BANK_ADDR, 14 }, // 4
    },
    map<uint32_t, uint32_t> {
        { SAVE_FIELD_ARGMAX, 2 }, // 5
        { SAVE_FIELD_JUMP_WRITE, 16 }, // 6
        { SAVE_FIELD_JUMP_READ, 10 }, // 7
    },
    map<uint32_t, uint32_t> {
        { SAVE_FIELD_CHANNEL, 12 }, // 8
        { SAVE_FIELD_LENGTH, 10 }, // 9
    },
    map<uint32_t, uint32_t> {
        { SAVE_FIELD_CONST_EN, 1 }, // 10
        { SAVE_FIELD_CONST_VALUE, 8 }, // 11
        { SAVE_FIELD_REG_ID, 3 }, // 12
    },
    map<uint32_t, uint32_t> {
        { SAVE_FIELD_DDR_ADDR, 29 }, // 13
    },
};


// ConvInit field length
const vector<map<uint32_t, uint32_t>> InstTable::ConvInitFieldLen = {
    map<uint32_t, uint32_t> {
        { CONVINIT_FIELD_OPCODE, 4 }, // 0
        { CONVINIT_FIELD_DPDON, 4 }, // 1
        { CONVINIT_FIELD_DPDBY, 4 }, // 2
        { CONVINIT_FIELD_TILE_EN, 1 }, // 3
        { CONVINIT_FIELD_SHIFT_CUT, 6 }, // 4
        { CONVINIT_FIELD_JUMP_READ_ENDL, 13 }, // 5
    },
    map<uint32_t, uint32_t> {
        { CONVINIT_FIELD_STRIDE_H, 4 }, // 6
        { CONVINIT_FIELD_STRIDE_W, 4 }, // 7
        { CONVINIT_FIELD_KERNEL_H, 4 }, // 8
        { CONVINIT_FIELD_KERNEL_W, 4 }, // 9
        { CONVINIT_FIELD_SHIFT_BIAS, 6 }, // 10
        { CONVINIT_FIELD_JUMP_READ, 10 }, // 11
    },
    map<uint32_t, uint32_t> {
        { CONVINIT_FIELD_CONV_NUM, 10 }, // 12
        { CONVINIT_FIELD_ACT_TYPE, 3 }, // 13
        { CONVINIT_FIELD_STRIDE_OFFSET_IN, 3 }, // 14
        { CONVINIT_FIELD_STRIDE_OFFSET_OUT, 3 }, // 15
        { CONVINIT_FIELD_JUMP_WRITE_ENDL, 13 }, // 16
    },
    map<uint32_t, uint32_t> {
        { CONVINIT_FIELD_STRIDE_OUT, 4 }, // 17
        { CONVINIT_FIELD_TILE_ICG, 6 }, // 18
        { CONVINIT_FIELD_TILE_OCG, 6 }, // 19
        { CONVINIT_FIELD_TILE_OWG, 6 }, // 20
        { CONVINIT_FIELD_JUMP_WRITE, 10 }, // 21
    },
    map<uint32_t, uint32_t> {
        { CONVINIT_FIELD_IC_ITER, 6 }, // 22
        { CONVINIT_FIELD_CHANNEL_OFFSET, 5 }, // 23
        { CONVINIT_FIELD_OW_ITER, 5 }, // 24
        { CONVINIT_FIELD_OW_OFFSET, 6 }, // 25
        { CONVINIT_FIELD_JUMP_READ_WEIGHTS, 9 }, // 26
    },
    map<uint32_t, uint32_t> {
        { CONVINIT_FIELD_SHIFT_HSWISH, 4 }, // 27
        { CONVINIT_FIELD_SHIFT_HSIGMOID, 5 }, // 28
        { CONVINIT_FIELD_HSIGMOID_IN, 4 }, // 29
        { CONVINIT_FIELD_JUMP_READ_WEIGHTS_ENDL, 13 }, // 30
    },
};


// Conv field length
const vector<map<uint32_t, uint32_t>> InstTable::ConvFieldLen = {
    map<uint32_t, uint32_t> {
        { CONV_FIELD_OPCODE, 4 }, // 0
        { CONV_FIELD_DPDON, 4 }, // 1
        { CONV_FIELD_DPDBY, 4 }, // 2
        { CONV_FIELD_BANK_ID_OUT, 6 }, // 3
        { CONV_FIELD_BANK_ADDR_OUT, 14 }, // 4
    },
    map<uint32_t, uint32_t> {
        { CONV_FIELD_ICG_OFFSET, 4 }, // 5
        { CONV_FIELD_BANK_ID_IN, 6 }, // 6
        { CONV_FIELD_BANK_ADDR_IN, 14 }, // 7
    },
    map<uint32_t, uint32_t> {
        { CONV_FIELD_PAD_LEFT, 4 }, // 8
        { CONV_FIELD_PAD_RIGHT, 4 }, // 9
        { CONV_FIELD_PAD_TOP, 4 }, // 10
        { CONV_FIELD_PAD_BOTTOM, 4 }, // 11
        { CONV_FIELD_BANK_ADDR_IN_1, 14 }, // 12
    },
    map<uint32_t, uint32_t> {
        { CONV_FIELD_VALID_PIXEL_PARALLEL, 4 }, // 13
        { CONV_FIELD_BANK_ADDR_WEIGHTS, 14 }, // 14
        { CONV_FIELD_BANK_ADDR_IN_2, 14 }, // 15
    },
    map<uint32_t, uint32_t> {
        { CONV_FIELD_OCG_OFFSET, 4 }, // 16
        { CONV_FIELD_BANK_ADDR_BIAS, 14 }, // 17
        { CONV_FIELD_BANK_ADDR_IN_3, 14 }, // 18
    },
};


// PoolInit field length
const vector<map<uint32_t, uint32_t>> InstTable::PoolInitFieldLen = {
    map<uint32_t, uint32_t> {
        { POOLINIT_FIELD_OPCODE, 4 }, // 0
        { POOLINIT_FIELD_DPDON, 4 }, // 1
        { POOLINIT_FIELD_DPDBY, 4 }, // 2
        { POOLINIT_FIELD_SHIFT_CUT, 4 }, // 3
        { POOLINIT_FIELD_KERNEL_W, 3 }, // 4
        { POOLINIT_FIELD_KERNEL_H, 3 }, // 5
        { POOLINIT_FIELD_JUMP_READ, 10 }, // 6
    },
    map<uint32_t, uint32_t> {
        { POOLINIT_FIELD_JUMP_WRITE, 10 }, // 7
        { POOLINIT_FIELD_STRIDE_OUT, 4 }, // 8
        { POOLINIT_FIELD_STRIDE_OFFSET_OUT, 3 }, // 9
        { POOLINIT_FIELD_VALID_PIXEL_PARALLEL, 3 }, // 10
        { POOLINIT_FIELD_STRIDE_OFFSET_IN, 3 }, // 11
        { POOLINIT_FIELD_STRIDE_W, 3 }, // 12
        { POOLINIT_FIELD_STRIDE_H, 3 }, // 13
        { POOLINIT_FIELD_POOL_TYPE, 2 }, // 14
    },
};


// Pool field length
const vector<map<uint32_t, uint32_t>> InstTable::PoolFieldLen = {
    map<uint32_t, uint32_t> {
        { POOL_FIELD_OPCODE, 4 }, // 0
        { POOL_FIELD_DPDON, 4 }, // 1
        { POOL_FIELD_DPDBY, 4 }, // 2
        { POOL_FIELD_PAD_LEFT, 3 }, // 3
        { POOL_FIELD_PAD_TOP, 3 }, // 4
        { POOL_FIELD_BANK_ADDR_IN, 14 }, // 5
    },
    map<uint32_t, uint32_t> {
        { POOL_FIELD_LENGTH, 10 }, // 6
        { POOL_FIELD_PAD_RIGHT, 3 }, // 7
        { POOL_FIELD_PAD_BOTTOM, 3 }, // 8
        { POOL_FIELD_CHANNEL_GROUP, 8 }, // 9
        { POOL_FIELD_BANK_ID_IN, 6 }, // 10
    },
    map<uint32_t, uint32_t> {
        { POOL_FIELD_JUMP_READ_ENDL, 13 }, // 11
        { POOL_FIELD_JUMP_WRITE_ENDL, 13 }, // 12
        { POOL_FIELD_BANK_ID_OUT, 6 }, // 13
    },
    map<uint32_t, uint32_t> {
        { POOL_FIELD_BANK_ADDR_IN_1, 14 }, // 14
        { POOL_FIELD_BANK_ADDR_OUT, 14 }, // 15
    },
    map<uint32_t, uint32_t> {
        { POOL_FIELD_BANK_ADDR_IN_3, 14 }, // 16
        { POOL_FIELD_BANK_ADDR_IN_2, 14 }, // 17
    },
};


// AluInit field length
const vector<map<uint32_t, uint32_t>> InstTable::AluInitFieldLen = {
    map<uint32_t, uint32_t> {
        { ALUINIT_FIELD_OPCODE, 4 }, // 0
        { ALUINIT_FIELD_DPDON, 4 }, // 1
        { ALUINIT_FIELD_DPDBY, 4 }, // 2
        { ALUINIT_FIELD_EXEC_MODE, 4 }, // 3
        { ALUINIT_FIELD_KERNEL_H, 8 }, // 4
        { ALUINIT_FIELD_KERNEL_W, 8 }, // 5
    },
    map<uint32_t, uint32_t> {
        { ALUINIT_FIELD_STRIDE_H, 8 }, // 6
        { ALUINIT_FIELD_STRIDE_W, 8 }, // 7
        { ALUINIT_FIELD_STRIDE_OFFSET_IN, 8 }, // 8
        { ALUINIT_FIELD_STRIDE_OFFSET_OUT, 8 }, // 9
    },
    map<uint32_t, uint32_t> {
        { ALUINIT_FIELD_STRIDE_OUT, 8 }, // 10
        { ALUINIT_FIELD_CHANNEL_GROUP, 8 }, // 11
        { ALUINIT_FIELD_CHANNEL_OFFSET, 5 }, // 12
        { ALUINIT_FIELD_LENGTH, 11 }, // 13
    },
    map<uint32_t, uint32_t> {
        { ALUINIT_FIELD_SHIFT_PRELU_P, 6 }, // 14
        { ALUINIT_FIELD_ACT_TYPE, 4 }, // 15
        { ALUINIT_FIELD_SHIFT_BIAS, 6 }, // 16
        { ALUINIT_FIELD_SHIFT_CUT, 6 }, // 17
        { ALUINIT_FIELD_SHIFT_PRELU_N, 6 }, // 18
    },
    map<uint32_t, uint32_t> {
        { ALUINIT_FIELD_MULTI_FACTOR, 8 }, // 19
        { ALUINIT_FIELD_JUMP_READ, 10 }, // 20
        { ALUINIT_FIELD_JUMP_READ_ENDL, 13 }, // 21
    },
    map<uint32_t, uint32_t> {
        { ALUINIT_FIELD_JUMP_READ_WEIGHTS, 9 }, // 22
        { ALUINIT_FIELD_JUMP_WRITE, 10 }, // 23
        { ALUINIT_FIELD_JUMP_WRITE_ENDL, 13 }, // 24
    },
};


// Alu field length
const vector<map<uint32_t, uint32_t>> InstTable::AluFieldLen = {
    map<uint32_t, uint32_t> {
        { ALU_FIELD_OPCODE, 4 }, // 0
        { ALU_FIELD_DPDON, 4 }, // 1
        { ALU_FIELD_DPDBY, 4 }, // 2
        { ALU_FIELD_BANK_ID_OUT, 6 }, // 3
        { ALU_FIELD_BANK_ADDR_OUT, 14 }, // 4
    },
    map<uint32_t, uint32_t> {
        { ALU_FIELD_PAD_BOTTOM, 8 }, // 5
        { ALU_FIELD_VALID_PIXEL_PARALLEL, 3 }, // 6
        { ALU_FIELD_BANK_ID_IN, 6 }, // 7
        { ALU_FIELD_BANK_ADDR_IN, 14 }, // 8
    },
    map<uint32_t, uint32_t> {
        { ALU_FIELD_PAD_TOP, 4 }, // 9
        { ALU_FIELD_PAD_LEFT, 4 }, // 10
        { ALU_FIELD_PAD_RIGHT, 8 }, // 11
        { ALU_FIELD_BANK_ADDR_IN_1, 14 }, // 12
    },
    map<uint32_t, uint32_t> {
        { ALU_FIELD_KERNEL_D, 4 }, // 13
        { ALU_FIELD_BANK_ADDR_WEIGHTS, 14 }, // 14
        { ALU_FIELD_BANK_ADDR_IN_2, 14 }, // 15
    },
    map<uint32_t, uint32_t> {
        { ALU_FIELD_BANK_ADDR_BIAS, 14 }, // 16
        { ALU_FIELD_BANK_ADDR_IN_3, 14 }, // 17
    },
};


// DWInit field length
const vector<map<uint32_t, uint32_t>> InstTable::DWInitFieldLen = {
    map<uint32_t, uint32_t> {
        { DWINIT_FIELD_OPCODE, 4 }, // 0
        { DWINIT_FIELD_DPDON, 4 }, // 1
        { DWINIT_FIELD_DPDBY, 4 }, // 2
        { DWINIT_FIELD_CHANNEL_GROUP, 8 }, // 3
        { DWINIT_FIELD_JUMP_READ_ENDL, 12 }, // 4
    },
    map<uint32_t, uint32_t> {
        { DWINIT_FIELD_STRIDE_H, 4 }, // 5
        { DWINIT_FIELD_STRIDE_W, 4 }, // 6
        { DWINIT_FIELD_KERNEL_H, 4 }, // 7
        { DWINIT_FIELD_KERNEL_W, 4 }, // 8
        { DWINIT_FIELD_STRIDE_OFFSET_IN, 3 }, // 9
        { DWINIT_FIELD_STRIDE_OFFSET_OUT, 3 }, // 10
        { DWINIT_FIELD_JUMP_READ, 10 }, // 11
    },
    map<uint32_t, uint32_t> {
        { DWINIT_FIELD_STRIDE_OUT, 4 }, // 12
        { DWINIT_FIELD_ACT_TYPE, 4 }, // 13
        { DWINIT_FIELD_CHANNEL_OFFSET, 5 }, // 14
        { DWINIT_FIELD_JUMP_WRITE_ENDL, 12 }, // 15
    },
    map<uint32_t, uint32_t> {
        { DWINIT_FIELD_SHIFT_CUT, 6 }, // 16
        { DWINIT_FIELD_SHIFT_BIAS, 6 }, // 17
        { DWINIT_FIELD_LENGTH, 10 }, // 18
        { DWINIT_FIELD_JUMP_WRITE, 10 }, // 19
    },
};


// DptWise field length
const vector<map<uint32_t, uint32_t>> InstTable::DptWiseFieldLen = {
    map<uint32_t, uint32_t> {
        { DPTWISE_FIELD_OPCODE, 4 }, // 0
        { DPTWISE_FIELD_DPDON, 4 }, // 1
        { DPTWISE_FIELD_DPDBY, 4 }, // 2
        { DPTWISE_FIELD_BANK_ID_OUT, 6 }, // 3
        { DPTWISE_FIELD_BANK_ADDR_OUT, 14 }, // 4
    },
    map<uint32_t, uint32_t> {
        { DPTWISE_FIELD_BANK_ID_IN, 6 }, // 5
        { DPTWISE_FIELD_BANK_ADDR_IN, 14 }, // 6
    },
    map<uint32_t, uint32_t> {
        { DPTWISE_FIELD_PAD_TOP, 4 }, // 7
        { DPTWISE_FIELD_PAD_BOTTOM, 4 }, // 8
        { DPTWISE_FIELD_PAD_LEFT, 4 }, // 9
        { DPTWISE_FIELD_PAD_RIGHT, 4 }, // 10
        { DPTWISE_FIELD_BANK_ADDR_IN_1, 14 }, // 11
    },
    map<uint32_t, uint32_t> {
        { DPTWISE_FIELD_VALID_PIXEL_PARALLEL, 4 }, // 12
        { DPTWISE_FIELD_BANK_ADDR_WEIGHTS, 14 }, // 13
        { DPTWISE_FIELD_BANK_ADDR_IN_2, 14 }, // 14
    },
    map<uint32_t, uint32_t> {
        { DPTWISE_FIELD_BANK_ADDR_BIAS, 14 }, // 15
        { DPTWISE_FIELD_BANK_ADDR_IN_3, 14 }, // 16
    },
};


// ElewInit field length
const vector<map<uint32_t, uint32_t>> InstTable::ElewInitFieldLen = {
    map<uint32_t, uint32_t> {
        { ELEWINIT_FIELD_OPCODE, 4 }, // 0
        { ELEWINIT_FIELD_DPDON, 4 }, // 1
        { ELEWINIT_FIELD_DPDBY, 4 }, // 2
        { ELEWINIT_FIELD_BANK_ID_IN, 6 }, // 3
        { ELEWINIT_FIELD_SHARE_PP, 1 }, // 4
    },
    map<uint32_t, uint32_t> {
        { ELEWINIT_FIELD_JUMP_BANK, 3 }, // 5
        { ELEWINIT_FIELD_ID, 2 }, // 6
        { ELEWINIT_FIELD_JUMP_READ_ENDL, 13 }, // 7
        { ELEWINIT_FIELD_BANK_ADDR_IN, 14 }, // 8
    },
    map<uint32_t, uint32_t> {
        { ELEWINIT_FIELD_SHIFT_READ, 4 }, // 9
        { ELEWINIT_FIELD_JUMP_READ, 12 }, // 10
    },
};


// Elew field length
const vector<map<uint32_t, uint32_t>> InstTable::ElewFieldLen = {
    map<uint32_t, uint32_t> {
        { ELEW_FIELD_OPCODE, 4 }, // 0
        { ELEW_FIELD_DPDON, 4 }, // 1
        { ELEW_FIELD_DPDBY, 4 }, // 2
        { ELEW_FIELD_BANK_ID_OUT, 6 }, // 3
        { ELEW_FIELD_ELEW_TYPE, 2 }, // 4
        { ELEW_FIELD_ACT_TYPE, 1 }, // 5
        { ELEW_FIELD_LENGTH, 11 }, // 6
    },
    map<uint32_t, uint32_t> {
        { ELEW_FIELD_VALID_PIXEL_PARALLEL, 3 }, // 7
        { ELEW_FIELD_NUM, 2 }, // 8
        { ELEW_FIELD_JUMP_WRITE_ENDL, 13 }, // 9
        { ELEW_FIELD_BANK_ADDR_OUT, 14 }, // 10
    },
    map<uint32_t, uint32_t> {
        { ELEW_FIELD_CHANNEL_GROUP, 8 }, // 11
        { ELEW_FIELD_SHIFT_WRITE, 5 }, // 12
        { ELEW_FIELD_JUMP_WRITE, 12 }, // 13
    },
};


// End field length
const vector<map<uint32_t, uint32_t>> InstTable::EndFieldLen = {
    map<uint32_t, uint32_t> {
        { END_FIELD_OPCODE, 4 }, // 0
        { END_FIELD_DPDON, 4 }, // 1
        { END_FIELD_DPDBY, 4 }, // 2
    },
};


// DumpBank field length
const vector<map<uint32_t, uint32_t>> InstTable::DumpBankFieldLen = {
    map<uint32_t, uint32_t> {
        { DUMPBANK_FIELD_OPCODE, 4 }, // 0
        { DUMPBANK_FIELD_DPDON, 4 }, // 1
        { DUMPBANK_FIELD_DPDBY, 4 }, // 2
        { DUMPBANK_FIELD_SAVE_NAME, 20 }, // 3
    },
    map<uint32_t, uint32_t> {
        { DUMPBANK_FIELD_SAVE_FMT, 6 }, // 4
        { DUMPBANK_FIELD_BANK_START, 8 }, // 5
        { DUMPBANK_FIELD_BANK_NUM, 8 }, // 6
    },
};


// DumpDDR field length
const vector<map<uint32_t, uint32_t>> InstTable::DumpDDRFieldLen = {
    map<uint32_t, uint32_t> {
        { DUMPDDR_FIELD_OPCODE, 4 }, // 0
        { DUMPDDR_FIELD_DPDON, 4 }, // 1
        { DUMPDDR_FIELD_DPDBY, 4 }, // 2
        { DUMPDDR_FIELD_SAVE_NAME, 20 }, // 3
    },
    map<uint32_t, uint32_t> {
        { DUMPDDR_FIELD_SAVE_FMT, 6 }, // 4
        { DUMPDDR_FIELD_REG_ID, 6 }, // 5
    },
    map<uint32_t, uint32_t> {
        { DUMPDDR_FIELD_DDR_START, 32 }, // 6
    },
    map<uint32_t, uint32_t> {
        { DUMPDDR_FIELD_DDR_SIZE, 32 }, // 7
    },
};


// DumpDDRSlice field length
const vector<map<uint32_t, uint32_t>> InstTable::DumpDDRSliceFieldLen = {
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_OPCODE, 4 }, // 0
        { DUMPDDRSLICE_FIELD_DPDON, 4 }, // 1
        { DUMPDDRSLICE_FIELD_DPDBY, 4 }, // 2
        { DUMPDDRSLICE_FIELD_SAVE_NAME, 20 }, // 3
    },
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_SAVE_FMT, 6 }, // 4
        { DUMPDDRSLICE_FIELD_REG_ID, 6 }, // 5
    },
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_DDR_START, 32 }, // 6
    },
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_HEIGHT, 16 }, // 7
        { DUMPDDRSLICE_FIELD_HEIGHT_STRIDE, 16 }, // 8
    },
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_WIDTH, 16 }, // 9
        { DUMPDDRSLICE_FIELD_WIDTH_STRIDE, 16 }, // 10
    },
    map<uint32_t, uint32_t> {
        { DUMPDDRSLICE_FIELD_CHANNEL, 16 }, // 11
        { DUMPDDRSLICE_FIELD_CHANNEL_STRIDE, 16 }, // 12
    },
};


// inst opcode to inst type map
const unordered_map<uint32_t, uint32_t> InstTable::OPCode2InstType = {
    { 0x00, INST_TYPE_LOAD }, // 0
    { 0x04, INST_TYPE_SAVE }, // 1
    { 0x09, INST_TYPE_CONVINIT }, // 2
    { 0x08, INST_TYPE_CONV }, // 3
    { 0x06, INST_TYPE_POOLINIT }, // 4
    { 0x0C, INST_TYPE_POOL }, // 5
    { 0x01, INST_TYPE_ALUINIT }, // 6
    { 0x02, INST_TYPE_ALU }, // 7
    { 0x0B, INST_TYPE_DWINIT }, // 8
    { 0x0A, INST_TYPE_DPTWISE }, // 9
    { 0x0D, INST_TYPE_ELEWINIT }, // 10
    { 0x0E, INST_TYPE_ELEW }, // 11
    { 0x07, INST_TYPE_END }, // 12
    { 0xFF, INST_TYPE_DUMPBANK }, // 13
    { 0xFE, INST_TYPE_DUMPDDR }, // 14
    { 0xFD, INST_TYPE_DUMPDDRSLICE }, // 15
};


// inst name to inst type map
const unordered_map<string, uint32_t> InstTable::InstName2InstType = {
    { "LOAD", INST_TYPE_LOAD }, // 0
    { "SAVE", INST_TYPE_SAVE }, // 1
    { "CONVINIT", INST_TYPE_CONVINIT }, // 2
    { "CONV", INST_TYPE_CONV }, // 3
    { "POOLINIT", INST_TYPE_POOLINIT }, // 4
    { "POOL", INST_TYPE_POOL }, // 5
    { "ALUINIT", INST_TYPE_ALUINIT }, // 6
    { "ALU", INST_TYPE_ALU }, // 7
    { "DWINIT", INST_TYPE_DWINIT }, // 8
    { "DPTWISE", INST_TYPE_DPTWISE }, // 9
    { "ELEWINIT", INST_TYPE_ELEWINIT }, // 10
    { "ELEW", INST_TYPE_ELEW }, // 11
    { "END", INST_TYPE_END }, // 12
    { "DUMPBANK", INST_TYPE_DUMPBANK }, // 13
    { "DUMPDDR", INST_TYPE_DUMPDDR }, // 14
    { "DUMPDDRSLICE", INST_TYPE_DUMPDDRSLICE }, // 15
};


// inst type to field name map
const unordered_map<uint32_t, const vector<string> &> InstTable::InstFieldName = {
    { INST_TYPE_LOAD, LoadFieldName }, // 0
    { INST_TYPE_SAVE, SaveFieldName }, // 1
    { INST_TYPE_CONVINIT, ConvInitFieldName }, // 2
    { INST_TYPE_CONV, ConvFieldName }, // 3
    { INST_TYPE_POOLINIT, PoolInitFieldName }, // 4
    { INST_TYPE_POOL, PoolFieldName }, // 5
    { INST_TYPE_ALUINIT, AluInitFieldName }, // 6
    { INST_TYPE_ALU, AluFieldName }, // 7
    { INST_TYPE_DWINIT, DWInitFieldName }, // 8
    { INST_TYPE_DPTWISE, DptWiseFieldName }, // 9
    { INST_TYPE_ELEWINIT, ElewInitFieldName }, // 10
    { INST_TYPE_ELEW, ElewFieldName }, // 11
    { INST_TYPE_END, EndFieldName }, // 12
    { INST_TYPE_DUMPBANK, DumpBankFieldName }, // 13
    { INST_TYPE_DUMPDDR, DumpDDRFieldName }, // 14
    { INST_TYPE_DUMPDDRSLICE, DumpDDRSliceFieldName }, // 15
};


// inst type to field data type map
const unordered_map<uint32_t, const vector<int> &> InstTable::InstFieldDataType = {
    { INST_TYPE_LOAD, LoadFieldDataType }, // 0
    { INST_TYPE_SAVE, SaveFieldDataType }, // 1
    { INST_TYPE_CONVINIT, ConvInitFieldDataType }, // 2
    { INST_TYPE_CONV, ConvFieldDataType }, // 3
    { INST_TYPE_POOLINIT, PoolInitFieldDataType }, // 4
    { INST_TYPE_POOL, PoolFieldDataType }, // 5
    { INST_TYPE_ALUINIT, AluInitFieldDataType }, // 6
    { INST_TYPE_ALU, AluFieldDataType }, // 7
    { INST_TYPE_DWINIT, DWInitFieldDataType }, // 8
    { INST_TYPE_DPTWISE, DptWiseFieldDataType }, // 9
    { INST_TYPE_ELEWINIT, ElewInitFieldDataType }, // 10
    { INST_TYPE_ELEW, ElewFieldDataType }, // 11
    { INST_TYPE_END, EndFieldDataType }, // 12
    { INST_TYPE_DUMPBANK, DumpBankFieldDataType }, // 13
    { INST_TYPE_DUMPDDR, DumpDDRFieldDataType }, // 14
    { INST_TYPE_DUMPDDRSLICE, DumpDDRSliceFieldDataType }, // 15
};


// inst type to field minus map
const unordered_map<uint32_t, const vector<int> &> InstTable::InstFieldMinus = {
    { INST_TYPE_LOAD, LoadFieldMinus }, // 0
    { INST_TYPE_SAVE, SaveFieldMinus }, // 1
    { INST_TYPE_CONVINIT, ConvInitFieldMinus }, // 2
    { INST_TYPE_CONV, ConvFieldMinus }, // 3
    { INST_TYPE_POOLINIT, PoolInitFieldMinus }, // 4
    { INST_TYPE_POOL, PoolFieldMinus }, // 5
    { INST_TYPE_ALUINIT, AluInitFieldMinus }, // 6
    { INST_TYPE_ALU, AluFieldMinus }, // 7
    { INST_TYPE_DWINIT, DWInitFieldMinus }, // 8
    { INST_TYPE_DPTWISE, DptWiseFieldMinus }, // 9
    { INST_TYPE_ELEWINIT, ElewInitFieldMinus }, // 10
    { INST_TYPE_ELEW, ElewFieldMinus }, // 11
    { INST_TYPE_END, EndFieldMinus }, // 12
    { INST_TYPE_DUMPBANK, DumpBankFieldMinus }, // 13
    { INST_TYPE_DUMPDDR, DumpDDRFieldMinus }, // 14
    { INST_TYPE_DUMPDDRSLICE, DumpDDRSliceFieldMinus }, // 15
};


// inst type to field position map
const unordered_map<uint32_t, const vector<map<uint32_t, uint32_t>> &> InstTable::InstFieldPos = {
    { INST_TYPE_LOAD, LoadFieldPos }, // 0
    { INST_TYPE_SAVE, SaveFieldPos }, // 1
    { INST_TYPE_CONVINIT, ConvInitFieldPos }, // 2
    { INST_TYPE_CONV, ConvFieldPos }, // 3
    { INST_TYPE_POOLINIT, PoolInitFieldPos }, // 4
    { INST_TYPE_POOL, PoolFieldPos }, // 5
    { INST_TYPE_ALUINIT, AluInitFieldPos }, // 6
    { INST_TYPE_ALU, AluFieldPos }, // 7
    { INST_TYPE_DWINIT, DWInitFieldPos }, // 8
    { INST_TYPE_DPTWISE, DptWiseFieldPos }, // 9
    { INST_TYPE_ELEWINIT, ElewInitFieldPos }, // 10
    { INST_TYPE_ELEW, ElewFieldPos }, // 11
    { INST_TYPE_END, EndFieldPos }, // 12
    { INST_TYPE_DUMPBANK, DumpBankFieldPos }, // 13
    { INST_TYPE_DUMPDDR, DumpDDRFieldPos }, // 14
    { INST_TYPE_DUMPDDRSLICE, DumpDDRSliceFieldPos }, // 15
};


// inst type to field length map
const unordered_map<uint32_t, const vector<map<uint32_t, uint32_t>> &> InstTable::InstFieldLen = {
    { INST_TYPE_LOAD, LoadFieldLen }, // 0
    { INST_TYPE_SAVE, SaveFieldLen }, // 1
    { INST_TYPE_CONVINIT, ConvInitFieldLen }, // 2
    { INST_TYPE_CONV, ConvFieldLen }, // 3
    { INST_TYPE_POOLINIT, PoolInitFieldLen }, // 4
    { INST_TYPE_POOL, PoolFieldLen }, // 5
    { INST_TYPE_ALUINIT, AluInitFieldLen }, // 6
    { INST_TYPE_ALU, AluFieldLen }, // 7
    { INST_TYPE_DWINIT, DWInitFieldLen }, // 8
    { INST_TYPE_DPTWISE, DptWiseFieldLen }, // 9
    { INST_TYPE_ELEWINIT, ElewInitFieldLen }, // 10
    { INST_TYPE_ELEW, ElewFieldLen }, // 11
    { INST_TYPE_END, EndFieldLen }, // 12
    { INST_TYPE_DUMPBANK, DumpBankFieldLen }, // 13
    { INST_TYPE_DUMPDDR, DumpDDRFieldLen }, // 14
    { INST_TYPE_DUMPDDRSLICE, DumpDDRSliceFieldLen }, // 15
};


}

