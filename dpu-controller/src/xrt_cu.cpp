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
#include "xrt_cu.hpp"

#include <ert.h>
#include <glog/logging.h>

#include <UniLog/UniLog.hpp>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>
#include <vitis/ai/xxd.hpp>

DEF_ENV_PARAM(DEBUG_XRT_CU, "0");
DEF_ENV_PARAM(XLNX_DPU_TIMEOUT, "10000");
DEF_ENV_PARAM(XLNX_XRT_CU_DRY_RUN, "0");

namespace xir {

XrtCu::XrtCu(const std::string& cu_name)
    : cu_name_{cu_name}, handle_{xir::XrtDeviceHandle::get_instance()} {
  auto num_of_cus = handle_->get_num_of_cus(cu_name_);
  bo_handles_.reserve(num_of_cus);

  for (auto idx = 0u; idx < num_of_cus; ++idx) {
    auto device_index = handle_->get_device_index(cu_name_, idx);
    auto device_id = handle_->get_device_id(cu_name_, idx);
    auto cu_full_name = handle_->get_cu_full_name(cu_name_, idx);
    auto cu_kernel_name = handle_->get_cu_kernel_name(cu_name_, idx);
    auto cu_instance_name = handle_->get_cu_instance_name(cu_name_, idx);
    auto cu_fingerprint = handle_->get_cu_fingerprint(cu_name_, idx);
    auto* kernel = reinterpret_cast<const xrt::kernel*>(
        handle_->get_kernel_handle(cu_name_, idx));
    auto run = std::make_unique<xrt::run>(*kernel);
    auto ecmd = reinterpret_cast<ert_start_kernel_cmd*>(run->get_ert_packet());

    LOG_IF(INFO, ENV_PARAM(DEBUG_XRT_CU))
        << "device_index/cu_index: " << device_index << "/" << idx << ", "  //
        << "device_id " << device_id << ", "                                //
        << "cu_full_name " << cu_full_name << ", "                          //
        << "cu_kernel_name " << cu_kernel_name << ", "                      //
        << "cu_instance_name " << cu_instance_name << ", "                  //
        << std::hex                                                         //
        << "fingerprint 0x" << cu_fingerprint << ", "                       //
        << "kernel handle 0x" << kernel << ", "                             //
        << "run handle 0x" << run.get() << " "                              //
        << "ecmd 0x" << ecmd                                                //
        << std::dec                                                         //
        ;
    bo_handles_.emplace_back(
        my_bo_handle{kernel, std::move(run), ecmd, device_id, cu_fingerprint,
                     cu_full_name, cu_kernel_name, cu_instance_name});
    // init_cmd(idx);
  }
}

XrtCu::~XrtCu() {
  int idx = 0;
  for (const auto& cu : bo_handles_) {
    LOG_IF(INFO, ENV_PARAM(DEBUG_XRT_CU))
        << "idx " << idx << " "  //
        << "cu_full_name " << cu.cu_full_name << " "
        << "device_id " << cu.device_id << " "       //
        << std::hex                                  //
        << "kernel handle 0x" << cu.kernel << " "    //
        << "run handle 0x" << cu.run.get() << " "    //
        << "cu_fingerprint 0x" << cu.cu_fingerprint  //
        << std::dec                                  //
        ;
    idx++;
  }
}

struct timestamps {
  uint64_t total;
  uint64_t to_driver;
  uint64_t to_cu;
  uint64_t cu_complete;
  uint64_t done;
};
static inline uint64_t tp2ns(struct timespec* tp) {
  return (uint64_t)tp->tv_sec * 1000000000UL + tp->tv_nsec;
}
static void print_one_timestamp(const timestamps& ts) {
  LOG(INFO) << "Total: " << ts.total / 1000 << "us\t"
            << "ToDriver: " << ts.to_driver / 1000 << "us\t"
            << "ToCU: " << ts.to_cu / 1000 << "us\t"
            << "Complete: " << ts.cu_complete / 1000 << "us\t"
            << "Done: " << ts.done / 1000 << "us" << std::endl;
}
static void print_timestamp(const uint64_t start, const uint64_t end,
                            cu_cmd_state_timestamps* c) {
  struct timestamps ts;
  ts.total = end - start;
  ts.to_driver = c->skc_timestamps[ERT_CMD_STATE_NEW] - start;
  ts.to_cu = c->skc_timestamps[ERT_CMD_STATE_RUNNING] -
             c->skc_timestamps[ERT_CMD_STATE_NEW];
  ts.cu_complete = c->skc_timestamps[ERT_CMD_STATE_COMPLETED] -
                   c->skc_timestamps[ERT_CMD_STATE_RUNNING];
  ts.done = end - c->skc_timestamps[ERT_CMD_STATE_COMPLETED];
  print_one_timestamp(ts);
}

static std::string ert_state_to_string(ert_cmd_state state) {
  std::vector<std::string> state_str{
      "ERT_CMD_STATE_NEW",         //
      "ERT_CMD_STATE_QUEUED",      //
      "ERT_CMD_STATE_RUNNING",     //
      "ERT_CMD_STATE_COMPLETED",   //
      "ERT_CMD_STATE_ERROR",       //
      "ERT_CMD_STATE_ABORT",       //
      "ERT_CMD_STATE_SUBMITTED",   //
      "ERT_CMD_STATE_TIMEOUT",     //
      "ERT_CMD_STATE_NORESPONSE",  //
      "ERT_CMD_STATE_SKERROR",     //
      "ERT_CMD_STATE_SKCRASHED",   //
      "ERT_CMD_STATE_MAX"          //
  };
  return state_str.at(state - 1);
}

void XrtCu::run(size_t device_core_idx, XrtCu::prepare_ecmd_t prepare,
                callback_t on_success, callback_t on_failure = []() {
           LOG(WARNING) << "[XrtCu] DPU execution timeout! No handler provided.";
         }) {
  UNI_LOG_CHECK(bo_handles_.size() > 0u, VART_XRT_DEVICE_BUSY)
      << "no cu availabe. cu_name=" << cu_name_;
  struct timespec tp;
#ifdef _WIN32
  uint64_t start = 0;  // TODO; implemented it on windows.
#else
  clock_gettime(CLOCK_MONOTONIC, &tp);
  uint64_t start = tp2ns(&tp);
#endif

  device_core_idx = device_core_idx % bo_handles_.size();
  auto ecmd = bo_handles_[device_core_idx].ecmd;
  ecmd->type = ERT_CTRL;
  ecmd->stat_enabled = 1;
  prepare(ecmd);
  LOG_IF(INFO, ENV_PARAM(DEBUG_XRT_CU))
      << "sizeof(ecmd) " << sizeof(*ecmd) << " "                 //
      << "ecmd->state " << ecmd->state << " "                    //
      << "ecmd->cu_mask " << ecmd->cu_mask << " "                //
      << "ecmd->extra_cu_masks " << ecmd->extra_cu_masks << " "  //
      << "ecmd->count " << ecmd->count << " "                    //
      << "ecmd->opcode " << ecmd->opcode << " "                  //
      << "ecmd->type " << ecmd->type << " "                      //
      << ((ENV_PARAM(DEBUG_XRT_CU) >= 2)
              ? vitis::ai::xxd((unsigned char*)ecmd,
                               (sizeof *ecmd) + ecmd->count * 4, 8, 1)
              : std::string(""));
  ;
  __TIC__(XRT_RUN)
  bool is_done = false;
  auto start_from_0 = std::chrono::steady_clock::now();
  auto start_from = start_from_0;
  auto state = ERT_CMD_STATE_NEW;
  auto& r = *bo_handles_[device_core_idx].run;
  if (ENV_PARAM(XLNX_XRT_CU_DRY_RUN)) {
    is_done = true;
    state = ERT_CMD_STATE_COMPLETED;
  } else {
    r.start();

    start_from = std::chrono::steady_clock::now();
    auto count = 1u;
    while (!is_done) {
      state = r.wait(1000);
      LOG_IF(INFO, ENV_PARAM(DEBUG_XRT_CU) >= 2)
          << "wait state is " << ert_state_to_string(state)  //
          << " in " << count << "s";
          is_done = true;
      // if (state >= ERT_CMD_STATE_COMPLETED && state != ERT_CMD_STATE_TIMEOUT) {
      //   is_done = true;
      // }
      if (!is_done) {
        auto now = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      now - start_from)
                      .count();
        if (ms > ENV_PARAM(XLNX_DPU_TIMEOUT)) {

          LOG(INFO)<<"====================dpu time out need to be reset=============================";
          return ;
        }
      }
      count++;
    }
  };
  if (!is_done) {
#ifdef _WIN32
    uint64_t end = 0;  // todo , implemented it on windows
#else
    clock_gettime(CLOCK_MONOTONIC, &tp);
    uint64_t end = tp2ns(&tp);
#endif
    auto now = std::chrono::steady_clock::now();
    auto ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(now - start_from)
            .count();
    LOG_IF(WARNING, !is_done)
        << " cu timeout!"                                                   //
        << " device_core_idx=" << device_core_idx                           //
        << ", cu_name=" << bo_handles_[device_core_idx].cu_full_name        //
        << ", ENV_PARAM(XLNX_DPU_TIMEOUT)=" << ENV_PARAM(XLNX_DPU_TIMEOUT)  //
        << ", state is " << ert_state_to_string(state)                      //
        << ", wait_time=" << ms                                             //
        ;
#ifndef _WIN32
    print_timestamp(start, end, ert_start_kernel_timestamps(ecmd));
#endif

  } else if (ENV_PARAM(DEBUG_XRT_CU)) {
#ifdef _WIN32
    uint64_t end = 0;  // todo: implemented on windows.
#else
    clock_gettime(CLOCK_MONOTONIC, &tp);
    uint64_t end = tp2ns(&tp);
#endif
    auto now = std::chrono::steady_clock::now();
    auto ms =
        std::chrono::duration_cast<std::chrono::microseconds>(now - start_from)
            .count();
    auto ms0 = std::chrono::duration_cast<std::chrono::microseconds>(
                   now - start_from_0)
                   .count();
#ifndef _WIN32
    auto c = ert_start_kernel_timestamps(ecmd);
    LOG(INFO) << " device_core_idx=" << device_core_idx                     //
              << ", cu_name=" << bo_handles_[device_core_idx].cu_full_name  //
              << ", state is " << ert_state_to_string(state)                //
              << ", wait_time = " << ms                                     //
              << ", run_time = " << ms0                                     //
              << ", ts0 = " << c->skc_timestamps[0]                         //
              << ", ts1 = " << c->skc_timestamps[1]                         //
        ;
    print_timestamp(start, end, c);
#endif
  }
  __TOC__(XRT_RUN)
  if (is_done) {
    on_success();
  } else {
    on_failure();
  }
}

size_t XrtCu::get_num_of_cu() const { return bo_handles_.size(); }

size_t XrtCu::get_device_id(size_t device_core_idx) const {
  return bo_handles_[device_core_idx].device_id;
}

std::string XrtCu::get_full_name(size_t device_core_idx) const {
  return bo_handles_[device_core_idx].cu_full_name;
}

std::string XrtCu::get_kernel_name(size_t device_core_idx) const {
  return bo_handles_[device_core_idx].cu_kernel_name;
}

std::string XrtCu::get_instance_name(size_t device_core_idx) const {
  return bo_handles_[device_core_idx].cu_instance_name;
}

uint64_t XrtCu::get_fingerprint(size_t device_core_idx) const {
  return bo_handles_[device_core_idx].cu_fingerprint;
}

uint32_t XrtCu::read_register(size_t device_core_idx, uint32_t offset) const {
  return handle_->read_register(cu_name_, device_core_idx, offset);
}

void XrtCu::write_register(size_t device_core_idx, uint32_t offset,
                           uint32_t value) const {
  return handle_->write_register(cu_name_, device_core_idx, offset, value);
}

ert_start_kernel_cmd* XrtCu::get_cmd(size_t device_core_id) {
  return bo_handles_[device_core_id].ecmd;
}

/*
void XrtCu::init_cmd(size_t device_core_id) {
  auto ecmd = get_cmd(device_core_id);
  for (size_t i = 4; i < 128; ++i) {
    ecmd->data[i] = read_register(device_core_id, i * sizeof(uint32_t));
  }
}*/

}  // namespace xir
