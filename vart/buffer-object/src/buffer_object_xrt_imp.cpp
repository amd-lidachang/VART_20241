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
// for type conversion
#  pragma warning(disable : 4267)
#endif

#include "./buffer_object_xrt_imp.hpp"

#include <experimental/xrt_system.h>
#include <glog/logging.h>

#include <UniLog/UniLog.hpp>
#include <cstring>
#include <string>

#include "vitis/ai/env_config.hpp"
#include "vitis/ai/weak.hpp"
DEF_ENV_PARAM(DEBUG_BUFFER_OBJECT, "0")
DEF_ENV_PARAM(DRAM_ADDRESS_MAPPING, "1")
DEF_ENV_PARAM(DRAM_NUM_OF_BANK, "8")
DEF_ENV_PARAM(DRAM_BANK_RANGE, "16384")

namespace {
static device_info_t find_xrt_info(xir::XrtDeviceHandle* handle,
                                   size_t device_id,
                                   const std::string& cu_name) {
  auto num_of_cus = handle->get_num_of_cus(cu_name);
  for (auto i = 0u; i < num_of_cus; ++i) {
    if (device_id == handle->get_device_id(cu_name, i)) {
      return device_info_t{
          device_id,                                   //
          reinterpret_cast<const xrt::device*>(
              handle->get_device_handle(cu_name, i)),  //
          reinterpret_cast<const xrt::kernel*>(
              handle->get_kernel_handle(cu_name, i))   //
      };
    }
  }
  LOG(FATAL) << "cannot find xrt device handle for device id " << device_id;
  return device_info_t{};
}

BufferObjectXrtEdgeImp::BufferObjectXrtEdgeImp(size_t size, size_t device_id,
                                               const std::string& cu_name)
    : BufferObject(),                                          //
      holder_{xir::XrtDeviceHandle::get_instance()},           //
      xrt_{find_xrt_info(holder_.get(), device_id, cu_name)},  //
      size_{size},                                             //
      bank_offset_{0} {
  bo_size_ = size;
  bank_offset_ = 0;
  // This optimization utilizes the parallel access feature of DRAM banks to
  // increase the memory access bandwidth for XVDPU multiple batches. This is
  // achieved by assigning different bank ranges to the physical addresses of
  // the input, output, and workspace.
  if (ENV_PARAM(DRAM_ADDRESS_MAPPING)) {
    int num_of_bank = ENV_PARAM(DRAM_NUM_OF_BANK);
    size_t range_per_bank = ENV_PARAM(DRAM_BANK_RANGE);
    static int bo_idx = 0;
    if (size > range_per_bank) {
      bo_size_ = size + (range_per_bank * num_of_bank);
      bank_offset_ = ((bo_idx++) % num_of_bank) * range_per_bank;
    }
  }
  // temp to hard code memory bank to 0
  // need a rule for all cards, edge and cloud
  auto mem_group = holder_->get_memory_bank_index();
  // bo_ = std::make_unique<xrt::bo>(*xrt_.device, bo_size_,
                                  // mem_group); 
    bo_ = std::make_unique<xrt::bo>(*xrt_.device, bo_size_, 0, 1);                         //
  CHECK(bo_ != nullptr) << "bo initialization failed with "          //
                        << "device_id " << device_id << " "          //
                        << "cu_name " << cu_name << " "              //
                        << "in bank group " << mem_group << " "      //
                        << std::hex                                  //
                        << "device handle 0x" << xrt_.device << " "  //
                        << "kernel handle 0x" << xrt_.kernel << " "  //
                        << std::dec                                  //
      ;
  // default bo buffer type is normal
  data_ = bo_->map<int*>();
  // XRT memory allocation is dynamic. After map<int*>() , only the vritual
  // address is returned, and the physical memory is not immediately applied.
  // Only when a 'page falut' is generated when the memory is written for the
  // first time, the physical memory is applied for and then the write
  // operation is performed.
  // It takes 26ms to apply for 106MB of physical memory.
  // In order not to affect the performance measurement of the first write
  // operation, mmset is executed after mmap to actively apply for physical
  // memory.
  CHECK(data_ != nullptr) << "bo map failed with "                     //
                          << "device_id " << device_id << " "          //
                          << "cu_name " << cu_name << " "              //
                          << "in bank group " << mem_group << " "      //
                          << std::hex                                  //
                          << "device handle 0x" << xrt_.device << " "  //
                          << "kernel handle 0x" << xrt_.kernel << " "  //
                          << std::dec                                  //
      ;
  std::memset(data_, 0, bo_size_);
  // cache flush.
  // When the cache is enabled, after memset is executed, the time for cpu to
  // write back to the cache is not fixed, If it alternats with DPU writing,
  // dirty data will appear in the cache, causeing some very difficult to
  // debug errors. eg. DPU outputs random 64 bits zero .
  // Execute cache flush ， write data to cache immediately.
  bo_->sync(XCL_BO_SYNC_BO_TO_DEVICE, bo_size_, 0);                       //
  data_ = data_ + (bank_offset_ / sizeof(int));
  phy_ = bo_->address() + bank_offset_;                                   //
  CHECK(phy_ != 0u) << "bo phy address failed with "                      //
                    << "device_id " << device_id << " "                   //
                    << "cu_name " << cu_name << " "                       //
                    << "in bank group " << mem_group << " "               //
                    << std::hex                                           //
                    << "device handle 0x" << xrt_.device << " "           //
                    << "kernel handle 0x" << xrt_.kernel << " "           //
                    << std::dec                                           //
      ;
  LOG_IF(INFO, ENV_PARAM(DEBUG_BUFFER_OBJECT))                            //
      << "create bufferobject 0x"                                         //
      << std::hex                                                         //
      << this << " "                                                      //
      << "bo_ 0x" << bo_.get() << " "                                     //
      << "phy_ 0x" << phy_ << " "                                         //
      << "data_ 0x" << data_ << " "                                       //
      << std::dec                                                         //
      << "desiredsize/actualsize " << size_ << "/" << bo_->size() << " "  //
      << "bo_size" << bo_size_ << " "
      << "memory group " << bo_->get_memory_group() << " "                //
      << "flags " << (uint32_t)bo_->get_flags() << " "                    //
      ;
}

BufferObjectXrtEdgeImp::~BufferObjectXrtEdgeImp() {
  LOG_IF(INFO, ENV_PARAM(DEBUG_BUFFER_OBJECT)) << "delete bufferobject 0x"    //
                                               << std::hex                    //
                                               << this << " "                 //
                                               << "bo_ " << bo_.get() << " "  //
      ;
}

const void* BufferObjectXrtEdgeImp::data_r() const {  //
  return data_;
}

void* BufferObjectXrtEdgeImp::data_w() {  //
  return data_;
}

size_t BufferObjectXrtEdgeImp::size() { return size_; }

uint64_t BufferObjectXrtEdgeImp::phy(size_t offset) { return phy_ + offset; }

void BufferObjectXrtEdgeImp::sync_for_read(uint64_t offset, size_t size) {
  if (data_ == nullptr) {
    LOG_IF(INFO, ENV_PARAM(DEBUG_BUFFER_OBJECT))
        << " meaningless for sync a device only memory";
    return;
  }
  bo_->sync(XCL_BO_SYNC_BO_FROM_DEVICE, size,
            offset + bank_offset_);    //
  LOG_IF(INFO, ENV_PARAM(DEBUG_BUFFER_OBJECT))
      << "sync from device "           //
      << std::hex                      //
      << "phy_ 0x" << phy_ << " "      //
      << "offset 0x" << offset << " "  //
      << std::dec                      //
      << "size " << size << " "        //
      ;
}

void BufferObjectXrtEdgeImp::sync_for_write(uint64_t offset, size_t size) {
  if (data_ == nullptr) {
    LOG_IF(INFO, ENV_PARAM(DEBUG_BUFFER_OBJECT))
        << " meaningless for sync a device only memory";
    return;
  }
  bo_->sync(XCL_BO_SYNC_BO_TO_DEVICE, size,
            offset + bank_offset_);    //
  LOG_IF(INFO, ENV_PARAM(DEBUG_BUFFER_OBJECT))
      << "sync to device "             //
      << std::hex                      //
      << "phy_ 0x" << phy_ << " "      //
      << "offset 0x" << offset << " "  //
      << std::dec                      //
      << "size " << size << " "        //
      ;
}

void BufferObjectXrtEdgeImp::copy_from_host(const void* buf, size_t size,
                                            size_t offset) {
  LOG_IF(INFO, ENV_PARAM(DEBUG_BUFFER_OBJECT))
      << "copy from host to device "   //
      << std::hex                      //
      << "phy_ 0x" << phy_ << " "      //
      << "offset 0x" << offset << " "  //
      << std::dec                      //
      << "size " << size << " "        //
      ;
  UNI_LOG_CHECK(offset + size <= size_, VART_OUT_OF_RANGE);
#if IS_EDGE
  memcpy(static_cast<char*>(data_w()) + offset, buf, size);
  sync_for_write(offset, size);
#else
  bo_->write(buf, size, offset);
#endif
}
void BufferObjectXrtEdgeImp::copy_to_host(void* buf, size_t size,
                                          size_t offset) {
  LOG_IF(INFO, ENV_PARAM(DEBUG_BUFFER_OBJECT))
      << "copy from device to host "   //
      << std::hex                      //
      << "phy_ 0x" << phy_ << " "      //
      << "offset 0x" << offset << " "  //
      << std::dec                      //
      << "size " << size << " "        //
      ;
  UNI_LOG_CHECK(offset + size <= size_, VART_OUT_OF_RANGE);
#if IS_EDGE
  sync_for_read(offset, size);
  memcpy(buf, static_cast<const char*>(data_r()) + offset, size);
#else
  bo_->read(buf, size, offset);
#endif
}

xir::XclBo BufferObjectXrtEdgeImp::get_xcl_bo() const {
  return xir::XclBo{xrt_.device, bo_.get()};
}
}  // namespace

REGISTER_INJECTION_BEGIN(xir::BufferObject, 1, BufferObjectXrtEdgeImp, size_t&,
                         size_t&, const std::string&) {
  auto ret = xrt::system::enumerate_devices() > 0;
  LOG_IF(INFO, ENV_PARAM(DEBUG_BUFFER_OBJECT))
      << " ret=" << ret
      << " register factory methord of BufferObjectXrtEdgeImp for "
         " xir::BufferObject with priority `1`";
  return ret;
}
REGISTER_INJECTION_END
