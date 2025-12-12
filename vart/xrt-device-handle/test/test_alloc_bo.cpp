
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
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <algorithm>
#include <array>
#include <iostream>

#include "xir/xrt_device_handle.hpp"
using namespace std;

int main(int argc, char* argv[]) {
  auto device = xrt::device(0);
  LOG(INFO) << "bo1 25u*1024*1024 alloating in memgroup(0)...";
  auto bo1 = xrt::bo(device, 25u * 1024 * 1024, 0);
  auto data1 = bo1.map<char*>();
  auto phy1 = bo1.address();
  std::generate(data1, data1 + bo1.size(), []() { return rand() % 256; });  
  LOG(INFO) << "bo1 data 0x" << std::hex << data1 << " phy 0x" << phy1
            << std::dec << ", actual size " << bo1.size();
  
  LOG(INFO) << "bo2 4u*1024*1024 alloating in memgroup(1)...";
  auto bo2 = xrt::bo(device, 4u * 1024 * 1024, 1);
  auto data2 = bo2.map<void*>();
  auto phy2 = bo2.address();
  LOG(INFO) << "bo2 data 0x" << std::hex << data2 << " phy 0x" << phy2
            << std::dec << ", actual size " << bo2.size();

  auto h = xir::XrtDeviceHandle::get_instance();
  std::string cu = "DPU";
  auto* dev = reinterpret_cast<const xrt::device*>(h->get_device_handle(cu, 0));
  auto* kel = reinterpret_cast<const xrt::kernel*>(h->get_kernel_handle(cu, 0));
  LOG(INFO) << "bo3 8u*1024*1024 alloating in memgroup(" << kel->group_id(2)
            << ")...";
  auto bo3 = xrt::bo(*dev, 8u * 1024 * 1024, kel->group_id(2));
  auto data3 = bo3.map<void*>();
  auto phy3 = bo3.address();
  LOG(INFO) << "bo3 data 0x" << std::hex << data3 << " phy 0x" << phy3
            << std::dec << ", actual size " << bo3.size();

  return 0;
}
