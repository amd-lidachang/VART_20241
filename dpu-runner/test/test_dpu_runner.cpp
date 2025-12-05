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
#include <google/protobuf/message.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <fstream>
#include<unistd.h>
#include <iostream>
#include <xir/tensor/tensor.hpp>
#include "vart/dpu/vitis_dpu_runner_factory.hpp"
#include "vart/runner_ext.hpp"

using namespace std;

void run_inference(const std::string& xmodel, const std::string& kernel,
                    const std::string& firmware_path) {
  setenv("XLNX_VART_FIRMWARE", firmware_path.c_str(), 1);
  // const std::string filepath = "/etc/vart.conf";
  // std::ofstream file(filepath, std::ios::out | std::ios::trunc);
  //  if (!file) {
  //       std::cerr << "Failed to open " << filepath << std::endl;
  //       return;
  //   }
  //   file<<firmware_path;
  //   file.close();
  auto runner = vart::dpu::DpuRunnerFactory::create_dpu_runner(xmodel, kernel);
  sleep(1);
  // runner.reset();
                   
}

int main(int argc, char* argv[]) {
  auto xmodel = argv[1];
  auto kernel = argv[2];

for(int i =0; i<100;i++){
    run_inference(xmodel, kernel, "/usr/lib/rm.xclbin");
    std::cout<<"version info"<<std::endl;
    system("devmem 0xa4200000");
    // system("env XLNX_VART_FIRMWARE=/usr/lib/rp.xclbin show_dpu");
    // std::cout<<"version info"<<std::endl;
    // system("devmem 0xa4200000");
    run_inference(xmodel, kernel, "/usr/lib/rp.xclbin");
     std::cout<<"version info"<<std::endl;
     system("devmem 0xa4200000");
}
  return 0;
}
