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
// #include <experimental/xrt_ip.h>
#include <experimental/xrt_xclbin.h>
#include <glog/logging.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_uuid.h>

#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vitis/ai/lock.hpp>

#include "xir/xrt_device_handle.hpp"
namespace {
// xrt device info
struct XrtDeviceInfo {
  std::string dev_name;
  std::string dev_bdf;
  xrt::uuid dev_interface_uuid;
  uint32_t dev_kdma;
  unsigned long dev_max_frequency;  // mhz
  bool dev_m2m;
  bool dev_nodma;
  bool dev_offline;
  std::string dev_electrical;
  std::string dev_thermal;
  std::string dev_mechanical;
  std::string dev_memory;
  std::string dev_platform;
  std::string dev_pcie_info;
  std::string dev_host;
  std::string dev_aie;
  std::string dev_aie_shim;
  std::string dev_aie_mem;
  std::string dev_dynamic_regions;
  std::string dev_vmr;
};

struct DeviceObject {
  // device info
  size_t device_index;
  size_t device_id;
  std::shared_ptr<xrt::uuid> uuid;
  XrtDeviceInfo dev_info;
  // cu info
  std::string cu_full_name;
  std::string cu_kernel_name;
  std::string cu_instance_name;
  uint64_t cu_fingerprint;
  // cu handle object
  std::shared_ptr<xrt::device> device;
  std::unique_ptr<xrt::kernel> kernel;
};

class XrtDeviceHandleImp : public xir::XrtDeviceHandle {
 public:
  XrtDeviceHandleImp();
  virtual ~XrtDeviceHandleImp();

  virtual size_t get_num_of_devices() const override;
  virtual size_t get_num_of_cus() const override;
  virtual size_t get_num_of_cus(const std::string& cu_name) const override;
  virtual size_t get_device_index(const std::string& cu_name,
                                  size_t cu_index) const override;
  virtual size_t get_device_id(const std::string& cu_name,
                               size_t cu_index) const override;
  virtual const void* get_device_uuid(const std::string& cu_name,
                                      size_t cu_index) const override;
  virtual std::string get_cu_full_name(const std::string& cu_name,
                                       size_t cu_index) const override;
  virtual std::string get_cu_kernel_name(const std::string& cu_name,
                                         size_t cu_index) const override;
  virtual std::string get_cu_instance_name(const std::string& cu_name,
                                           size_t cu_index) const override;
  virtual uint64_t get_cu_fingerprint(const std::string& cu_name,
                                      size_t cu_index) const override;
  virtual const void* get_device_handle(const std::string& cu_name,
                                        size_t cu_index) const override;
  virtual const void* get_kernel_handle(const std::string& cu_name,
                                        size_t cu_index) const override;
  virtual uint32_t get_memory_bank_index() const override;
  virtual uint32_t read_register(const std::string& cu_name, size_t cu_index,
                                 uint32_t offset) const override;
  virtual void write_register(const std::string& cu_name, size_t cu_index,
                              uint32_t offset, uint32_t value) const override;

 private:
  XrtDeviceHandleImp(const XrtDeviceHandleImp& rhs) = delete;
  XrtDeviceHandleImp& operator=(const XrtDeviceHandleImp& rhs) = delete;

 private:
  const DeviceObject& find_cu(const std::string& cu_name,
                              size_t cu_index) const;
  void init_bank_index();

 private:
  size_t device_num_;
  uint32_t bank_index_;
  std::unique_ptr<xrt::xclbin> xclbin_;
  std::map<std::string, DeviceObject> cu_handles_;

  std::vector<std::unique_ptr<vitis::ai::Lock>> mtx_;
  std::vector<std::unique_ptr<std::unique_lock<vitis::ai::Lock>>> locks_;
};
}  // namespace
