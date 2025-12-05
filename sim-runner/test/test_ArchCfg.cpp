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
#include <fcntl.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <iostream>

#include "conf/ArchCfg.hpp"

using google::protobuf::io::FileInputStream;

int main() {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  vitis::ai::Target arch;

  auto fd = open("./config/ArchCfg.txt", ios_base::in);
  google::protobuf::io::FileInputStream fstream(fd);
  if (!google::protobuf::TextFormat::Parse(&fstream, &arch)) {
    UNI_LOG_FATAL(SIM_FILE_OPEN_FAILED) << "Failed to parse arch param.";
    abort();
  }

  ArchCfg::Instance().set_param(arch);

  //  GOOGLE_PROTOBUF_VERIFY_VERSION;
  std::cout << "Target name: " << ArchCfg::Instance().get_param().name()
            << std::endl;
  std::cout << "Target type: " << ArchCfg::Instance().get_param().type()
            << std::endl;
  std::cout << "Bank group size: "
            << ArchCfg::Instance().get_param().bank_group_size() << std::endl;
  std::cout << "Load engine cp: "
            << ArchCfg::Instance().get_param().load_engine().channel_parallel()
            << std::endl;

  ArchCfg::Instance().init_white_lists();
  std::cout << "Test bank access white list" << std::endl;
  std::cout << "elew-in" << std::endl;
  auto inst_white_list =
      ArchCfg::Instance().get_bank_access_white_list("elew-in");
  std::for_each(inst_white_list.cbegin(), inst_white_list.cend(),
                [](std::int32_t bank_id) { std::cout << bank_id << ' '; });
  std::cout << std::endl << "load-out" << std::endl;
  inst_white_list = ArchCfg::Instance().get_bank_access_white_list("load-out");
  std::for_each(inst_white_list.cbegin(), inst_white_list.cend(),
                [](std::int32_t bank_id) { std::cout << bank_id << ' '; });

  std::cout << std::endl << "Test instr limit white list" << std::endl;
  std::cout << "conv-limit kernel_size" << std::endl;
  auto limit_white_list =
      ArchCfg::Instance().get_instr_limit_white_list("pool-max-kernel");
  std::for_each(limit_white_list.cbegin(), limit_white_list.cend(),
                [](std::int32_t stride) { std::cout << stride << ' '; });
  std::cout << std::endl;
}
