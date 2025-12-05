/*
 * Copyright 2019 Xilinx Inc.
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

#include "general_util.h"

std::pair<int, int> reverse_kernel_iterate(int kernel_h_, int kernel_w_,
                                           int kh_iter_, int kw_iter_,
                                           int min_kh, int max_kh) {
  UNI_LOG_CHECK(kw_iter_ == 1, SIM_OUT_OF_RANGE)
      << "currently only support kw_iter(" << kw_iter_ << ") = 1";
  UNI_LOG_CHECK(kh_iter_ >= min_kh && kh_iter_ <= max_kh, SIM_OUT_OF_RANGE)
      << "kh_iter (" << kh_iter_ << ") out of conv range [" << min_kh << ","
      << max_kh << "].";

  kernel_h_ *= kh_iter_;
  kernel_w_ *= kw_iter_;

  UNI_LOG_CHECK(kernel_h_ >= min_kh && kernel_h_ <= max_kh, SIM_OUT_OF_RANGE)
      << "kernel_h (" << kernel_h_ << ") out of range conv [" << min_kh << ","
      << max_kh << "].";

  UNI_LOG_CHECK(kernel_w_ >= min_kh && kernel_w_ <= max_kh, SIM_OUT_OF_RANGE)
      << "kernel_w_ (" << kernel_w_ << ") out of range conv [" << min_kh << ","
      << max_kh << "].";
  return {kernel_h_, kernel_w_};
}
