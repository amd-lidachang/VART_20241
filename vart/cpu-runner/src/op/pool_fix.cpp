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

#include "pool_fix.hpp"

#include "pair_hash.hpp"

namespace {
struct ApproximateParam {
  ApproximateParam(int t_kh, int t_kw, int t_N, int t_M)
      : kh(t_kh), kw(t_kw), N(t_N), M(t_M) {
    scale = (float)N / powf(2, M);
  }

  int kh;  // pool kernel height
  int kw;  // pool kernel width
  int N;
  int M;

  float scale;
};

/// return the avgpool <multi_factor, shift_factor> according to the kernel size
std::pair<int32_t, int32_t> get_avgpool_dpu_factors(
    const std::vector<std::int32_t>& kernels) {
  auto rec = kernels[0] * kernels[1];
  auto multi_factor = 0;
  auto shift_factor = 0;
  auto diff = 1.f;
  if (kernels[0] == 3 && kernels[1] == 3) {
    multi_factor = 7;
    shift_factor = 6;
  } else if (kernels[0] == 5 && kernels[1] == 5) {
    multi_factor = 10;
    shift_factor = 8;
  } else if (kernels[0] == 6 && kernels[1] == 6) {
    multi_factor = 7;
    shift_factor = 8;
  } else if (kernels[0] == 7 && kernels[1] == 7) {
    multi_factor = 21;
    shift_factor = 10;
  } else if (kernels[0] == 14 && kernels[1] == 14) {
    multi_factor = 21;
    shift_factor = 12;
  } else {
    auto max_factor = std::ceil(std::log2(rec * 128));
    for (auto shift_factor_ = 0; shift_factor_ < max_factor; shift_factor_++) {
      auto factor = std::round(std::exp2(shift_factor_) / rec);
      auto diff_ = std::abs(factor / std::exp2(shift_factor_) - 1.f / rec);
      if (diff_ < diff) {
        multi_factor = factor;
        diff = diff_;
        shift_factor = shift_factor_;
      }
    }
  }
  return {multi_factor, shift_factor};
}

/// return the avgpool coefficient according to the kernel size
float get_avgpool_dpu_coefficient(const std::vector<std::int32_t>& kernels) {
  auto factors = get_avgpool_dpu_factors(kernels);
  return float(factors.first) / std::exp2(factors.second);
}

}  // anonymous namespace

namespace vart {
namespace cpu {

template <typename DType>
PoolFix<DType>::PoolFix(const xir::Subgraph* subg, const xir::Op* op,
                        IMapTBs_t inputs, CPUTBPtr_t output)
    : PoolBase<DType>(subg, op, inputs, output),
      AvgPool<DType>(subg, op, inputs, output),
      MaxPool<DType>(subg, op, inputs, output) {
  auto type_str = op->get_attr<std::string>("type");
  if (type_str == "MAX")
    pool_type_ = 0;
  else if (type_str == "AVG")
    pool_type_ = 1;
  else
    UNI_LOG_FATAL(VART_NOT_SUPPORT) << "Unknown avgpool type: " << type_str;

  auto xir_tensor_i = CPUOPBase::xir_op_->get_input_tensor("input");
  auto xir_tensor_o = CPUOPBase::xir_op_->get_output_tensor();

  fp_input_ = xir_tensor_i->get_attr<int>("fix_point");
  fp_output_ = xir_tensor_o->get_attr<int>("fix_point");
  shift_pool_ = fp_output_ - fp_input_;
  pow_shift_ = pow(2.0, shift_pool_);
}

template <typename DType>
void PoolFix<DType>::run() {
  // // do necessary check
  // print_param();
  // this->check_param();

  // // read data
  // this->read();

  // calc avgpool
  if (pool_type_ == PoolBase<DType>::MAX_POOL) {
    this->max_pool();
    max_pool_fix();
  } else {
    this->acc_pool();
    avg_pool_fix();
  }

  // // do save, debug...
  // this->save();
}

template <typename DType>
void PoolFix<DType>::print_param() {
  PoolBase<DType>::print_param();

  UNI_LOG_DEBUG_INFO << "pooltype = " << pool_type_ << endl;
  UNI_LOG_DEBUG_INFO << "fix_width = " << fix_width_ << endl;
  UNI_LOG_DEBUG_INFO << "fp_input = " << fp_input_ << endl;
  UNI_LOG_DEBUG_INFO << "fp_output = " << fp_output_ << endl;
  UNI_LOG_DEBUG_INFO << "shift_pool = " << shift_pool_ << endl;
  UNI_LOG_DEBUG_INFO << "pow_shift = " << pow_shift_ << endl;
}

template <typename DType>
void PoolFix<DType>::max_pool_fix() {
  if (CPU_RUN_MODE == CPURunMode::NORMAL_THREAD ||
      CPU_RUN_MODE == CPURunMode::GEMM_THREAD) {
    max_pool_fix_thread();
  } else {
    max_pool_fix_normal();
  }
}

template <typename DType>
void PoolFix<DType>::max_pool_fix_normal() {
  for (auto i = 0; i < fmap_o_.num(); i++) {
    data_out_ptr_[i] =
        DPURound<DType>(data_out_ptr_[i] * pow_shift_, CPUOPBase::data_min_,
                        CPUOPBase::data_max_);
  }
}

template <typename DType>
void PoolFix<DType>::max_pool_fix_thread() {
  vector<std::future<int>> thr_fut(THREAD_NUM);

  for (auto i = 0U; i < THREAD_NUM; i++) {
    // thr_fut[i] = std::async(std::launch::async | std::launch::deferred,
    thr_fut[i] = std::async(std::launch::async,
                            [this](decltype(i) i) {
                              auto BASE_POS = i * THREAD_WORKLOAD;
                              for (auto j = 0U; j < THREAD_WORKLOAD; j++) {
                                auto pos = BASE_POS + j;
                                if (pos >= FMAP_SIZE) break;

                                data_out_ptr_[pos] = DPURound<DType>(
                                    data_out_ptr_[pos] * pow_shift_,
                                    CPUOPBase::data_min_, CPUOPBase::data_max_);
                              }

                              return 0;
                            },
                            i);
  }

  for (auto i = 0U; i < THREAD_NUM; i++) {
    thr_fut[i].wait();
  }
}

template <typename DType>
void PoolFix<DType>::avg_pool_fix() {
  if (CPU_RUN_MODE == CPURunMode::NORMAL_THREAD ||
      CPU_RUN_MODE == CPURunMode::GEMM_THREAD) {
    avg_pool_fix_thread();
  } else {
    avg_pool_fix_normal();
  }
}

template <typename DType>
void PoolFix<DType>::avg_pool_fix_normal() {
  for (auto i = 0; i < fmap_o_.num(); i++) {
    avg_pool_fix_one(i);
  }
}

template <typename DType>
void PoolFix<DType>::avg_pool_fix_thread() {
  vector<std::future<int>> thr_fut(THREAD_NUM);

  for (auto i = 0U; i < THREAD_NUM; i++) {
    // thr_fut[i] = std::async(std::launch::async | std::launch::deferred,
    thr_fut[i] = std::async(std::launch::async,
                            [this](decltype(i) i) {
                              auto BASE_POS = i * THREAD_WORKLOAD;
                              for (auto j = 0U; j < THREAD_WORKLOAD; j++) {
                                auto pos = BASE_POS + j;
                                if (pos >= FMAP_SIZE) break;

                                avg_pool_fix_one(pos);
                              }

                              return 0;
                            },
                            i);
  }

  for (auto i = 0U; i < THREAD_NUM; i++) {
    thr_fut[i].wait();
  }
}

template <typename DType>
void PoolFix<DType>::avg_pool_fix_one(int i) {
  auto tmp = 0.f;

  tmp = (float)data_out_ptr_[i] *
        get_avgpool_dpu_coefficient({kernel_.h, kernel_.w});

  data_out_ptr_[i] = DPURound<DType>(tmp * pow_shift_, CPUOPBase::data_min_,
                                     CPUOPBase::data_max_);
}

INSTANTIATE_TPCLASS(PoolFix);
REG_OP_INSTANCE_FUNC("pool-fix", PoolFix);

}  // namespace cpu
}  // namespace vart
