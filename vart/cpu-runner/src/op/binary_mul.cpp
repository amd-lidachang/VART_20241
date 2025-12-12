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

#include "binary_mul.hpp"

namespace vart {
namespace cpu {

template <typename DTypeIn0, typename DTypeIn1, typename DTypeOut>
void BinaryMul<DTypeIn0, DTypeIn1, DTypeOut>::calculate(
    std::uint32_t start_index, std::uint32_t end_index, bool no_broadcast) {
  auto dst_coord = fmap_o_.pos2coord(0);
  auto src_coord0 = dst_coord;
  auto src_coord1 = dst_coord;
  dim_t pos0, pos1;
  for (auto pos_iter = start_index; pos_iter < end_index; pos_iter++) {
    if (no_broadcast) {
      pos0 = pos_iter;
      pos1 = pos_iter;
    } else {
      fmap_o_.pos2coord(pos_iter, dst_coord);
      for (auto dim_iter = 0U; dim_iter < src_coord0.size(); dim_iter++) {
        src_coord0[dim_iter] = dst_coord[dim_iter] % fmap_i_[0][dim_iter];
      }
      pos0 = fmap_i_[0].coord2pos(src_coord0);
      for (auto dim_iter = 0U; dim_iter < src_coord1.size(); dim_iter++) {
        src_coord1[dim_iter] = dst_coord[dim_iter] % fmap_i_[1][dim_iter];
      }
      pos1 = fmap_i_[1].coord2pos(src_coord1);
    }
    data_out_ptr_[pos_iter] = data_ina_ptr_[pos0] * data_inb_ptr_[pos1];
  }
}

INSTANTIATE_TPCLASS_BINARY(BinaryMul);
REG_BINARY_OP_INSTANCE_FUNC("binary_mul", BinaryMul);
}  // namespace cpu
}  // namespace vart
