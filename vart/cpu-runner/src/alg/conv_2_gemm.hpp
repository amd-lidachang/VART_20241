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

#include "cpu_base_inc.hpp"

namespace vart {
namespace cpu {
class Conv2Gemm {
 public:
  Conv2Gemm(const FMap_t& fmap_src, const FMap_t& fmap_dst,
            const FMap_t& fmap_w, const Kernel_t kernel, const Stride_t& stride)
      : fmap_src_(fmap_src),
        fmap_dst_(fmap_dst),
        fmap_w_(fmap_w),
        kernel_(kernel),
        stride_(stride),
        group_(fmap_src.c / fmap_w.c) {
    UNI_LOG_CHECK(
        fmap_src_.n == fmap_dst_.n / group_ &&
            fmap_dst_.c == kernel_.h * kernel_.w * kernel_.d * fmap_w_.c,
        VART_INVALID_VALUE)
        << ", fmap err!";
    THREAD_NUM = CPU_NUM;
    THREAD_WORKLOAD = ceil((double)(fmap_dst_.num() / fmap_w_.c) / THREAD_NUM);
  }

  VART_BIG_THREE_LAW(Conv2Gemm);

 public:
  template <typename T>
  void transform(const T* src, T* dst) {
    for (auto group_iter = 0; group_iter < group_; group_iter++) {
      for (auto n = 0; n < fmap_src_.n; n++) {
        trans_one_batch(src + n * fmap_src_.ncod(), dst + group_iter * (fmap_dst_.num() / group_) + n * (fmap_dst_.ncod()), group_iter);
      }
    }
  }

  template <typename T>
  void transform_thread(const T* src, T* dst) {
    vector<std::future<void>> fut(THREAD_NUM);

    for (auto i = 0U; i < THREAD_NUM; i++) {
      fut[i] = std::async(
          std::launch::async,
          [this, src, dst](decltype(i) i) {
            auto BASE_POS = i * THREAD_WORKLOAD;
            auto FMAP_SIZE = fmap_dst_.num() / fmap_w_.c;
            for (auto j = 0U; j < THREAD_WORKLOAD; j++) {
              int pos = BASE_POS + j;
              if (pos >= FMAP_SIZE) break;

              int32_t group_iter = (pos / (fmap_src_.n * (fmap_dst_.ncod() / fmap_w_.c)));
              int32_t batch_iter = (pos / (fmap_dst_.ncod() / fmap_w_.c)) % fmap_src_.n;
              int32_t ho_iter    = (pos / (fmap_dst_.hcod() / fmap_w_.c)) % fmap_dst_.h;
              int32_t wo_iter    = (pos / (fmap_dst_.wcod() / fmap_w_.c)) % fmap_dst_.w;
              int32_t do_iter    = (pos / (fmap_dst_.dcod() / fmap_w_.c)) % fmap_dst_.d;
              int32_t kh_iter    = (pos / (fmap_w_.w * fmap_w_.d)) % fmap_w_.h;
              int32_t kw_iter    = (pos / (fmap_w_.d)) % fmap_w_.w;
              int32_t kd_iter = pos % fmap_w_.d;

              auto* src_addr = src + batch_iter * fmap_src_.ncod() +
                               (ho_iter * stride_.h + kh_iter) * fmap_src_.hcod() +
                               (wo_iter * stride_.w + kw_iter) * fmap_src_.wcod() +
                               (do_iter * stride_.d + kd_iter) * fmap_src_.dcod() +
                               group_iter * fmap_w_.c;
              auto* dst_addr = dst + pos * fmap_w_.c;
              std::copy_n(src_addr, fmap_w_.c, dst_addr);
            }
          },
          i);
    }
    for (auto i = 0U; i < THREAD_NUM; i++) {
      fut[i].wait();
    }
  }

 private:
  template <typename T>
  void trans_one_batch(const T* src, T* dst, int32_t group_iter) {
    for (auto h = 0; h < fmap_dst_.h; h++) {
      for (auto w = 0; w < fmap_dst_.w; w++) {
        for (auto d = 0; d < fmap_dst_.d; d++) {
          auto src_h = h * stride_.h;
          auto src_w = w * stride_.w;
          auto src_d = d * stride_.d;

          auto* src_addr = src + src_h * fmap_src_.hcod() +
                           src_w * fmap_src_.wcod() + src_d * fmap_src_.dcod();
          auto* dst_addr = dst + h * fmap_dst_.hcod() + w * fmap_dst_.wcod() +
                           d * fmap_dst_.dcod();
          trans_one_kernel(src_addr, dst_addr, group_iter);
        }
      }
    }
  }

  template <typename T>
  void trans_one_kernel(const T* src, T* dst, int32_t group_iter) {
    for (auto h = 0; h < kernel_.h; h++) {
      for (auto w = 0; w < kernel_.w; w++) {
        for (auto d = 0; d < kernel_.d; d++) {
          auto* src_addr = src + h * fmap_src_.hcod() + w * fmap_src_.wcod() +
                           d * fmap_src_.dcod() + group_iter * fmap_w_.c;
          auto* dst_addr = dst + h * fmap_w_.hcod() + w * fmap_w_.wcod() +
                           d * fmap_w_.dcod();

          std::copy_n(src_addr, fmap_w_.c, dst_addr);
        }
      }
    }
  }

 private:
  FMap_t fmap_src_;
  FMap_t fmap_dst_;
  FMap_t fmap_w_;
  Kernel_t kernel_;
  Stride_t stride_{1, 1};
  int32_t group_;
  std::uint32_t THREAD_NUM;
  std::uint32_t THREAD_WORKLOAD;
};  // namespace cpu

}  // namespace cpu
}  // namespace vart
