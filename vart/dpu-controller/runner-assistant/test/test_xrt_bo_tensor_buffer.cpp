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

#include <iostream>

#include "vart/assistant/xrt_bo_tensor_buffer.hpp"
#include "vart/zero_copy_helper.hpp"
#include "xir/graph/graph.hpp"
// xrt.h must be included after. otherwise, name pollution.
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
using namespace std;
int main(int argc, char* argv[]) {
  auto graph = xir::Graph::deserialize(argv[1]);
  auto root = graph->get_root_subgraph();
  xir::Subgraph* s = nullptr;
  for (auto c : root->get_children()) {
    if (c->get_attr<std::string>("device") == "DPU") {
      s = c;
      break;
    }
  }

  auto dev = std::make_shared<xrt::device>(0);
  auto input_tensor_buffer_size = vart::get_input_buffer_size(s);
  auto bo = std::make_unique<xrt::bo>(*dev, input_tensor_buffer_size, 1);
  auto bo_new = std::make_unique<xrt::bo>(*dev, input_tensor_buffer_size, 1);
  auto tensors = s->get_sorted_output_tensors();
  CHECK(!tensors.empty());
  auto tensor = *tensors.begin();
  tensor = tensor->get_producer()->get_input_op("input")->get_output_tensor();
  LOG(INFO) << "tensor = " << tensor->to_string();
  auto tensor_buffer = vart::assistant::XrtBoTensorBuffer::create(
      vart::xrt_bo_t{dev.get(), bo.get()}, tensor);
  auto tensor_buffer_new = vart::assistant::XrtBoTensorBuffer::create(
      vart::xrt_bo_t{dev.get(), bo_new.get()}, tensor);
  LOG(INFO) << "tensor_buffer=" << tensor_buffer->to_string();
  auto data_phy = tensor_buffer->data_phy({0, 0, 0, 0});
  auto data_phy_new = tensor_buffer_new->data_phy({0, 0, 0, 0});
  
  LOG(INFO) << "phy = " << std::hex << "0x" << data_phy.first << std::dec
            << " data_size=" << data_phy.second
            << " request bo_size=" << input_tensor_buffer_size
            << " allocated bo_size= " << bo->size();
  LOG(INFO) << "phy = " << std::hex << "0x" << data_phy_new.first << std::dec
            << " data_size=" << data_phy_new.second
            << " request bo_size=" << input_tensor_buffer_size
            << " allocated bo_size= " << bo_new->size();
  CHECK_LE(input_tensor_buffer_size, bo->size())
      << "allcate and request size mismatch";
  return 0;
}
