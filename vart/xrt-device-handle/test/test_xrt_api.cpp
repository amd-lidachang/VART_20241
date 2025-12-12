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
#include <glog/logging.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_uuid.h>

#include <iostream>
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

int main(int argc, char* argv[]) {
  CHECK(argc <= 2) << "usage: " << argv[0] << " [xclbin_file]";
  auto xclbin_file = std::string("/run/media/mmcblk0p1/dpu.xclbin");
  if (argc == 2) {
    xclbin_file = std::string(argv[1]);
    CHECK(!xclbin_file.empty()) << "input file invalid";
  }

  unsigned int dev_index = 0;
  auto device = xrt::device(dev_index);
  auto xclbin = xrt::xclbin(xclbin_file);
  auto ips = xclbin.get_ips();
  std::cout << "cu number: " << ips.size() << std::endl;

  auto uuid = device.load_xclbin(xclbin);
  auto index = 0u;
  for (auto& ip : ips) {
    auto ip_name = ip.get_name();
    auto kernel_name = ip_name.substr(0, ip_name.find(':'));
    auto instance_name = ip_name.substr(ip_name.find(':') + 1);
    auto cu_name = kernel_name + ":{" + instance_name + "}";
    auto kernel = xrt::kernel(device, uuid, cu_name);
    xrt::set_read_range(kernel, 0x10, 0x1f0);
    auto l_value = kernel.read_register(0x1F0);
    auto h_value = kernel.read_register(0x1F0 + sizeof(l_value));
    uint64_t fingerprint = h_value;
    fingerprint = (fingerprint << 32) + l_value;
    std::cout << "cu[" << index << "]: ip_name " << ip_name                 //
              << ", cu_name " << cu_name                                    //
              << ", fingerprint 0x" << std::hex << fingerprint << std::dec  //
              << std::endl;
    index++;
  }

  return 0;
}
