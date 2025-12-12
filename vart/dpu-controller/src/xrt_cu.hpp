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
#include <experimental/xrt_ip.h>
#include <xrt/xrt_kernel.h>

#include <array>
#include <cstdlib>
#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "xir/xrt_device_handle.hpp"
class ert_start_kernel_cmd;
namespace xir {
struct my_bo_handle {
  const xrt::kernel* kernel;
  std::unique_ptr<xrt::run> run;
  ert_start_kernel_cmd* ecmd;

  size_t device_id;
  uint64_t cu_fingerprint;
  std::string cu_full_name;
  std::string cu_kernel_name;
  std::string cu_instance_name;
};

class XrtCu {
 public:
  explicit XrtCu(const std::string& cu_name);
  virtual ~XrtCu();
  XrtCu(const XrtCu& other) = delete;
  XrtCu& operator=(const XrtCu& rhs) = delete;
  using prepare_ecmd_t = std::function<void(ert_start_kernel_cmd*)>;
  using callback_t = std::function<void()>;

 public:
  void run(size_t core_idx, prepare_ecmd_t prepare, callback_t on_success,
           callback_t on_failure);

  size_t get_num_of_cu() const;
  size_t get_device_id(size_t device_core_idx) const;
  std::string get_full_name(size_t device_core_idx) const;
  std::string get_kernel_name(size_t device_core_idx) const;
  std::string get_instance_name(size_t device_core_idx) const;
  uint64_t get_fingerprint(size_t device_core_idx) const;
  uint32_t read_register(size_t device_core_idx, uint32_t offset) const;
  void write_register(size_t device_core_idx, uint32_t offset,
                      uint32_t value) const;
  ert_start_kernel_cmd* get_cmd(size_t device_core_id);

  // private:
  // void init_cmd(size_t device_core_id);

 private:
  std::string cu_name_;
  std::shared_ptr<xir::XrtDeviceHandle> handle_;
  std::vector<my_bo_handle> bo_handles_;
};

}  // namespace xir
