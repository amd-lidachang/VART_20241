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
#if _WIN32
// for get_env
#  pragma warning(disable : 4996)
// for type conversion
#  pragma warning(disable : 4267)
#endif
#if ! _WIN32
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
#include "./xrt_device_handle_imp.hpp"

#include <experimental/xrt_system.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <UniLog/UniLog.hpp>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <set>
#include <string>
#include <utility>

#include "vitis/ai/env_config.hpp"
#include "vitis/ai/lock.hpp"
#include "vitis/ai/simple_config.hpp"
#include "vitis/ai/weak.hpp"

DEF_ENV_PARAM(DEBUG_XRT_DEVICE_HANDLE, "0");
DEF_ENV_PARAM(XLNX_DISABLE_LOAD_XCLBIN, "0");
DEF_ENV_PARAM_2(XLNX_ENABLE_DEVICES, "ALL", std::string);
DEF_ENV_PARAM_2(XLNX_VART_FIRMWARE, "", std::string);

namespace {

const std::string get_dpu_xclbin() {
  // For mindray to use XLNX_VART_FIRMWARE
  const char* firmware_path = std::getenv("XLNX_VART_FIRMWARE");
  if (firmware_path && std::string(firmware_path).length() > 0) {
    return std::string(firmware_path);
  }
  auto ret = std::string("/usr/lib/dpu.xclbin");
  if (!ENV_PARAM(XLNX_VART_FIRMWARE).empty()) {
    ret = ENV_PARAM(XLNX_VART_FIRMWARE);
    return ret;
  }


  auto config =
      vitis::ai::SimpleConfig::getOrCreateSimpleConfig("/etc/vart.conf");

 
  if (!config) {
    LOG_IF(INFO, ENV_PARAM(DEBUG_XRT_DEVICE_HANDLE))
        << "/etc/vart.conf does not exits. use default value "
           "/usr/lib/dpu.xclbin";
    return ret;
  }
  auto has_firmware = (*config).has("firmware");
  if (!has_firmware) {
    LOG_IF(INFO, ENV_PARAM(DEBUG_XRT_DEVICE_HANDLE))
        << "/etc/vart.conf does not contains firmware: xxx. use default value "
           "/usr/lib/dpu.xclbin";
    return ret;
  }
  ret = (*config)("firmware").as<std::string>();
  return ret;
}

static bool start_with(const std::string& a, const std::string& b) {
  return a.find(b) == 0u;
}

static uint64_t my_get_fingerprint(const xrt::kernel* knl,
                                   const std::string& cu_name) {
  auto env_fingerprint = getenv((cu_name + ".fingerprint").c_str());
  if (env_fingerprint) {
    return std::stoul(env_fingerprint);
  }
  uint64_t cu_offset = 0x1F0;
  auto l_value = knl->read_register(cu_offset);
  auto h_value = knl->read_register(cu_offset + sizeof(l_value));
  uint64_t ret = h_value;
  ret = (ret << 32) + l_value;
  LOG_IF(INFO, ENV_PARAM(DEBUG_XRT_DEVICE_HANDLE))
      << cu_name << " fingerprint: 0x" << std::hex << ret << " with h_value"
      << " 0x" << h_value << " and l_value 0x" << l_value << std::dec;
  return ret;
}
/*
static void get_device_info(const xrt::device& device,
                            XrtDeviceInfo& dev_info) {
  dev_info.dev_bdf = device.get_info<xrt::info::device::bdf>();
  // dev_info.dev_interface_uuid =
  //     device.get_info<xrt::info::device::interface_uuid>();
  dev_info.dev_kdma = device.get_info<xrt::info::device::kdma>();
  dev_info.dev_max_frequency =
      device.get_info<xrt::info::device::max_clock_frequency_mhz>();
  dev_info.dev_m2m = device.get_info<xrt::info::device::m2m>();
  dev_info.dev_name = device.get_info<xrt::info::device::name>();
  dev_info.dev_nodma = device.get_info<xrt::info::device::nodma>();
  dev_info.dev_offline = device.get_info<xrt::info::device::offline>();
  dev_info.dev_electrical = device.get_info<xrt::info::device::electrical>();
  dev_info.dev_thermal = device.get_info<xrt::info::device::thermal>();
  dev_info.dev_mechanical = device.get_info<xrt::info::device::mechanical>();
  dev_info.dev_memory = device.get_info<xrt::info::device::memory>();
  dev_info.dev_platform = device.get_info<xrt::info::device::platform>();
  dev_info.dev_pcie_info = device.get_info<xrt::info::device::pcie_info>();
  dev_info.dev_host = device.get_info<xrt::info::device::host>();
  dev_info.dev_aie = device.get_info<xrt::info::device::aie>();
  dev_info.dev_aie_shim = device.get_info<xrt::info::device::aie_shim>();
  // dev_info.dev_aie_mem = device.get_info<xrt::info::device::aie_mem>();
  dev_info.dev_dynamic_regions =
      device.get_info<xrt::info::device::dynamic_regions>();
  dev_info.dev_vmr = device.get_info<xrt::info::device::vmr>();

  LOG_IF(INFO, ENV_PARAM(DEBUG_XRT_DEVICE_HANDLE) >= 2)
      << "bdf = " << dev_info.dev_bdf << "\n"                            //
      << "interface_uuid = " << dev_info.dev_interface_uuid << "\n"      //
      << "kdma = " << dev_info.dev_kdma << "\n"                          //
      << "max CL Frequency = " << dev_info.dev_max_frequency << "MHz\n"  //
      << "m2m = " << std::boolalpha << dev_info.dev_m2m << "\n"          //
      << "nodma = " << dev_info.dev_nodma << "\n"                        //
      << "offline = " << dev_info.dev_offline << "\n"                    //
      << "electrical = " << dev_info.dev_electrical << "\n"              //
      << "thermal = " << dev_info.dev_thermal << "\n"                    //
      << "mechanical = " << dev_info.dev_mechanical << "\n"              //
      << "memory = " << dev_info.dev_memory << "\n"                      //
      << "platform = " << dev_info.dev_platform << "\n"                  //
      << "pcie_info = " << dev_info.dev_pcie_info << "\n"                //
      << "host = " << dev_info.dev_host << "\n"                          //
      << "aie = " << dev_info.dev_aie << "\n"                            //
      << "aie_shim = " << dev_info.dev_aie_shim << "\n"                  //
      << "aie_mem = " << dev_info.dev_aie_mem << "\n"                    //
      << "dynamic_regions = " << dev_info.dev_dynamic_regions << "\n"    //
      << "vmr = " << dev_info.dev_vmr << "\n"                            //
      ;
  return;
}
*/
static std::pair<std::string, std::string> split_at(const std::string& str,
                                                    const char delimiter) {
  auto ret = std::pair<std::string, std::string>();

  std::istringstream names_istr(str);
  std::getline(names_istr, std::get<0>(ret), delimiter);
  std::getline(names_istr, std::get<1>(ret), delimiter);
  return ret;
}

static std::vector<std::string> split_str(const std::string& str,
                                          const char delim = ',') {
  auto list = std::vector<std::string>();
  std::istringstream ss(str);
  std::string item;
  while (std::getline(ss, item, delim)) {
    list.push_back(item);
  }
  return list;
}

static std::vector<uint32_t> get_device_id_list(const uint32_t num_of_devices) {
  std::string enable_devices = ENV_PARAM(XLNX_ENABLE_DEVICES);
  if (enable_devices == "ALL") {
    auto device_id_list = std::vector<uint32_t>(num_of_devices);
    std::iota(device_id_list.begin(), device_id_list.end(), 0);
    return device_id_list;
  }
  auto device_id_list = std::vector<uint32_t>();
  for (auto d : split_str(enable_devices, ',')) {
    auto device_id = (uint32_t)std::stoi(d);
    if (device_id < num_of_devices) {
      device_id_list.push_back(device_id);
    }
  }
  return device_id_list;
}

XrtDeviceHandleImp::XrtDeviceHandleImp() {
  static_assert(sizeof(xuid_t) == SIZE_OF_UUID, "ERROR: UUID size mismatch");

  auto num_of_devices = xrt::system::enumerate_devices();
  LOG_IF(INFO, ENV_PARAM(DEBUG_XRT_DEVICE_HANDLE))
      << "probe num of devices = " << num_of_devices;
  PCHECK(num_of_devices > 0) << "No devices can be used";

  // get xclbin file
  auto filename = get_dpu_xclbin();
  LOG_IF(INFO, ENV_PARAM(DEBUG_XRT_DEVICE_HANDLE))
      << "open firmware " << filename;
  xclbin_ = std::make_unique<xrt::xclbin>(filename);
  PCHECK(xclbin_ != nullptr) << "xclbin object initialized failed";

  // select devices to be used
  auto device_id_list = get_device_id_list(num_of_devices);
  PCHECK(device_id_list.size() > 0) << "No device selected";
  device_num_ = device_id_list.size();

  for (auto deviceIndex : device_id_list) {
    // device lock
    auto mtx = vitis::ai::Lock::create("DPU_" + std::to_string(deviceIndex));
    mtx_.push_back(std::move(mtx));
    auto lock = std::make_unique<std::unique_lock<vitis::ai::Lock>>(
        *(mtx_.back().get()), std::try_to_lock_t());
    if (!lock->owns_lock()) {
      LOG(INFO) << "waiting for process to release the resource:"
                << " DPU_" + std::to_string(deviceIndex);
      lock->lock();
    }
    locks_.push_back(std::move(lock));

    auto device = std::make_shared<xrt::device>(deviceIndex);

    // get device info, defer
    XrtDeviceInfo dev_info;
    // get_device_info(device, dev_info);

    // load xclbin
    auto xclbin_uuid = xclbin_->get_uuid();
    if (!ENV_PARAM(XLNX_DISABLE_LOAD_XCLBIN)) {
      LOG_IF(INFO, ENV_PARAM(DEBUG_XRT_DEVICE_HANDLE))
          << "load xclbin begin, device " << deviceIndex;
      auto uuid = device->load_xclbin(*xclbin_);
      PCHECK(uuid == xclbin_uuid) << "uuid loaded mismatched with xclbin uuid";
      LOG_IF(INFO, ENV_PARAM(DEBUG_XRT_DEVICE_HANDLE))
          << "load xclbin success, device " << deviceIndex << " uuid "
          << uuid.to_string();
    } else {
      LOG(INFO) << "load xclbin skipped";
    }
    // get uuid on device
    auto device_uuid = device->get_xclbin_uuid();
    LOG_IF(INFO, ENV_PARAM(DEBUG_XRT_DEVICE_HANDLE))
        << "device[" << deviceIndex << "]:"                         //
        << " uuid loaded on device is " << device_uuid.to_string()  //
        << " and uuid of xclbin file is " << xclbin_uuid.to_string();

    init_bank_index();

    // process for cu on this device
    auto xclbin_ips = xclbin_->get_ips();
    auto cu_num = xclbin_ips.size();
    PCHECK(cu_num > 0) << "no cus can be used";
    for (auto i = 0u; i < cu_num; ++i) {
      auto cu_full_name = xclbin_ips[i].get_name();
      // only load dpu and sfm kernel
      if (cu_full_name.find("DPU") == std::string::npos &&
          cu_full_name.find("dpu") == std::string::npos &&
          cu_full_name.find("sfm") == std::string::npos) {
        continue;
      }

      LOG_IF(INFO, ENV_PARAM(DEBUG_XRT_DEVICE_HANDLE))
          << "device_index/cu_index " << deviceIndex << "/" << i
          << " cu_full_name " << cu_full_name << " is loading... ";

      auto& cu = cu_handles_[cu_full_name + ":" + std::to_string(deviceIndex)];
      // device info
      cu.device_index = deviceIndex;
      cu.device_id = device_id_list[deviceIndex];
      cu.uuid = std::make_shared<xrt::uuid>(device_uuid);
      cu.dev_info = dev_info;
      // cu info
      cu.cu_full_name = cu_full_name;
      std::tie(cu.cu_kernel_name, cu.cu_instance_name) =
          split_at(cu_full_name, ':');
      auto cu_name = cu.cu_kernel_name + ":{" + cu.cu_instance_name + "}";
      // cu handle object
      cu.device = device;
      cu.kernel = std::make_unique<xrt::kernel>(*device, device_uuid, cu_name);
      // dpu DPUCV2DX8G register range [0x10,0x500]
      xrt::set_read_range(*cu.kernel, 0x10, 0x4F0);
      cu.cu_fingerprint = my_get_fingerprint(cu.kernel.get(), cu_full_name);
      LOG_IF(INFO, ENV_PARAM(DEBUG_XRT_DEVICE_HANDLE))
          << "device_id " << cu.device_id << ", "           //
          << "cu_full_name " << cu.cu_full_name << ", "     //
          << std::hex                                       //
          << "fingerprint 0x" << cu.cu_fingerprint << ", "  //
          << "device handle 0x" << cu.device.get() << ", "  //
          << "kernel handle 0x" << cu.kernel.get()          //
          << std::dec                                       //
          ;
    }
  }
  PCHECK(cu_handles_.size() > 0) << "No device initialized";
}  // namespace

XrtDeviceHandleImp::~XrtDeviceHandleImp() {
  LOG_IF(INFO, ENV_PARAM(DEBUG_XRT_DEVICE_HANDLE))
      << "handle is destroyed " << (void*)this << ", num of devices "
      << cu_handles_.size();
}

size_t XrtDeviceHandleImp::get_num_of_devices() const { return device_num_; }
size_t XrtDeviceHandleImp::get_num_of_cus() const { return cu_handles_.size(); }
size_t XrtDeviceHandleImp::get_num_of_cus(const std::string& cu_name) const {
  auto cnt = 0u;
  for (auto& x : cu_handles_) {
    if (start_with(x.first, cu_name)) {
      cnt++;
    }
  }
  return cnt;
}

size_t XrtDeviceHandleImp::get_device_index(const std::string& cu_name,
                                            size_t cu_index) const {
  return find_cu(cu_name, cu_index).device_index;
}

size_t XrtDeviceHandleImp::get_device_id(const std::string& cu_name,
                                         size_t cu_index) const {
  return find_cu(cu_name, cu_index).device_id;
}

const void* XrtDeviceHandleImp::get_device_uuid(const std::string& cu_name,
                                                size_t cu_index) const {
  return find_cu(cu_name, cu_index).uuid.get();
}

std::string XrtDeviceHandleImp::get_cu_full_name(const std::string& cu_name,
                                                 size_t cu_index) const {
  return find_cu(cu_name, cu_index).cu_full_name;
}

std::string XrtDeviceHandleImp::get_cu_kernel_name(const std::string& cu_name,
                                                   size_t cu_index) const {
  return find_cu(cu_name, cu_index).cu_kernel_name;
}

std::string XrtDeviceHandleImp::get_cu_instance_name(const std::string& cu_name,
                                                     size_t cu_index) const {
  return find_cu(cu_name, cu_index).cu_instance_name;
}

uint64_t XrtDeviceHandleImp::get_cu_fingerprint(const std::string& cu_name,
                                                size_t cu_index) const {
  return find_cu(cu_name, cu_index).cu_fingerprint;
}

const void* XrtDeviceHandleImp::get_device_handle(const std::string& cu_name,
                                                  size_t cu_index) const {
  return find_cu(cu_name, cu_index).device.get();
}

const void* XrtDeviceHandleImp::get_kernel_handle(const std::string& cu_name,
                                                  size_t cu_index) const {
  return find_cu(cu_name, cu_index).kernel.get();
}

uint32_t XrtDeviceHandleImp::get_memory_bank_index() const {
  return bank_index_ | XCL_BO_FLAGS_CACHEABLE;
}

uint32_t XrtDeviceHandleImp::read_register(const std::string& cu_name,
                                           size_t cu_index,
                                           uint32_t offset) const {
  return find_cu(cu_name, cu_index).kernel->read_register(offset);
}

void XrtDeviceHandleImp::write_register(const std::string& cu_name,
                                        size_t cu_index, uint32_t offset,
                                        uint32_t value) const {
  return find_cu(cu_name, cu_index).kernel->write_register(offset, value);
}

void XrtDeviceHandleImp::init_bank_index() {
  int32_t bank = -1;
  auto* mem_topo = xclbin_->get_axlf_section<const mem_topology*>(MEM_TOPOLOGY);
  for (int32_t idx = 0; idx < mem_topo->m_count; ++idx) {
    if (mem_topo->m_mem_data[idx].m_used) {
      bank = idx;
      break;
    }
  }
  PCHECK(bank >= 0) << "cannot find any valid bank index!";
  bank_index_ = bank;
  LOG_IF(INFO, ENV_PARAM(DEBUG_XRT_DEVICE_HANDLE))
      << "find the first valid bank index " << bank_index_;

  return;
}

const DeviceObject& XrtDeviceHandleImp::find_cu(const std::string& cu_name,
                                                size_t cu_index) const {
  auto cnt = 0u;
  const DeviceObject* ret = nullptr;
  for (auto& x : cu_handles_) {
    if (start_with(x.first, cu_name)) {
      if (cnt == cu_index) {
        ret = &x.second;
        break;
      }
      cnt++;
    }
  }
  UNI_LOG_CHECK(ret != nullptr, VART_XRT_NULL_PTR)
      << "cannot find cu handle with "
      << "cu_name " << cu_name << " and "  //
      << "cu_index " << cu_index << " "    //
      ;
  return *ret;
}
}  // namespace

// when we need one xrt-device-handle per xrt-cu, not sharing, we need the
// following function
/* std::unique_ptr<xir::XrtDeviceHandle> my_xir_device_handle_create() {
  return std::make_unique<XrtDeviceHandleImp>();
  }*/

namespace {
static struct Registar {
  Registar() {
    xir::XrtDeviceHandle::registar("03_xrt_edge", []() {
      return vitis::ai::WeakSingleton<XrtDeviceHandleImp>::create();
    });
  }
} g_registar;
}  // namespace
