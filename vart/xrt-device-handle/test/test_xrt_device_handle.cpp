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

#include <iostream>

#include "xir/xrt_device_handle.hpp"
using namespace std;

static void show(xir::XrtDeviceHandle* h, const std::string& cu_name) {
  auto num_of_cus = h->get_num_of_cus(cu_name);
  // device info can be showed here if needed.
  LOG(INFO) << "cu_name = " << cu_name << ", num_of_cus = " << num_of_cus;
  for (auto i = 0u; i < num_of_cus; ++i) {
    LOG(INFO) << "cu[" << i << "]: "                                       //
              << "device_id=" << h->get_device_id(cu_name, i)              //
              << ", cu_full_name=" << h->get_cu_full_name(cu_name, i)      //
              << ", cu_kernel_name=" << h->get_cu_kernel_name(cu_name, i)  //
              << ", cu_instance_name="
              << h->get_cu_instance_name(cu_name, i)            //
              << ", cu_fingerprint=" << std::hex << "0x"        //
              << h->get_cu_fingerprint(cu_name, i) << std::dec  //
        ;
  }
}
int main(int argc, char* argv[]) {
  auto cu_name = std::string{argv[1]};
  auto h1 = xir::XrtDeviceHandle::get_instance();
  show(h1.get(), cu_name);
  std::cout << "h1 show finished." << std::endl;
  /* auto h2 = xir::XrtDeviceHandle::get_instance();
  show(h2.get(), cu_name);
  auto h3 = xir::XrtDeviceHandle::create();
  show(h3.get(), cu_name);
  auto h4 = xir::XrtDeviceHandle::create();
  show(h4.get(), cu_name);*/

  return 0;
}
