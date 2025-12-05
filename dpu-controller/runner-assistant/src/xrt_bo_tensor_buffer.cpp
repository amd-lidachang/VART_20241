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

#include "vart/assistant/xrt_bo_tensor_buffer.hpp"
#ifdef XRT_FOUND
#  include <xrt/xrt_bo.h>
#endif

#include <UniLog/UniLog.hpp>

#include "vart/runner.hpp"

namespace vart {
namespace assistant {

#ifdef XRT_FOUND
static uint64_t get_physical_address(const void* bo) {
  CHECK(nullptr != bo) << "bo ptr null";
  auto xrt_bo = reinterpret_cast<const xrt::bo*>(bo);
  uint64_t phy = xrt_bo->address();
  CHECK(phy != 0u) << "get nullptr phy address";
  return phy;
}
#else
static uint64_t get_physical_address(const void* bo) {
  LOG(FATAL) << "not implemented, no XRT found";
  return 0ull;
}
#endif

std::unique_ptr<vart::TensorBuffer> XrtBoTensorBuffer::create(
    vart::xrt_bo_t bo, const xir::Tensor* tensor) {
  return std::make_unique<XrtBoTensorBuffer>(bo, tensor);
}

XrtBoTensorBuffer::XrtBoTensorBuffer(vart::xrt_bo_t bo,
                                     const xir::Tensor* tensor)
    : TensorBuffer(tensor), bo_{bo} {
  UNI_LOG_CHECK(tensor->has_attr("reg_id"), VART_TENSOR_INFO_ERROR)
      << "tensor: " << tensor->to_string();
  UNI_LOG_CHECK(tensor->has_attr("ddr_addr"), VART_TENSOR_INFO_ERROR)
      << "tensor: " << tensor->to_string();
  UNI_LOG_CHECK(tensor->has_attr("location"), VART_TENSOR_INFO_ERROR)
      << "tensor: " << tensor->to_string();
  // auto reg_id = (size_t)tensor->template get_attr<int>("reg_id");
  ddr_addr_ = (size_t)tensor->template get_attr<int>("ddr_addr");
  auto location = (size_t)tensor->template get_attr<int>("location");
  UNI_LOG_CHECK(location == 1, VART_TENSOR_INFO_ERROR);
  phy_addr_ = get_physical_address(bo.xrt_bo_handle);
  // TODO: assumue one bo one tensor, and the tensor should be on the
  // TODO: this is the bug for image bundling.
  size_ = tensor->get_data_size() / tensor->get_shape()[0];
}

std::pair<std::uint64_t, std::size_t> XrtBoTensorBuffer::data(
    const std::vector<std::int32_t> idx) {
  return std::make_pair((uint64_t)0u, (size_t)0u);
}
vart::TensorBuffer::location_t XrtBoTensorBuffer::get_location() const {
  // TODO: how to get device id from a XRT handle.
  return vart::TensorBuffer::location_t::DEVICE_0;
}
std::pair<uint64_t, size_t> XrtBoTensorBuffer::data_phy(
    const std::vector<std::int32_t> idx) {
  return std::make_pair((uint64_t)phy_addr_ + ddr_addr_, size_);
}
void XrtBoTensorBuffer::sync_for_read(uint64_t offset, size_t size) {
  // because it is not mapped, so that it is not so meaningful to  flush cache
  return;
}

void XrtBoTensorBuffer::sync_for_write(uint64_t offset, size_t size) {
  // because it is not mapped, so that it is not so meaningful to invalidate
  // cache
  return;
}

void XrtBoTensorBuffer::copy_from_host(size_t batch_idx, const void* buf,
                                       size_t size, size_t offset) {
  LOG(FATAL) << "TODO: not implemented yet";
}

void XrtBoTensorBuffer::copy_to_host(size_t batch_idx, void* buf, size_t size,
                                     size_t offset) {
  LOG(FATAL) << "TODO: not implemented yet";
}
}  // namespace assistant
}  // namespace vart
