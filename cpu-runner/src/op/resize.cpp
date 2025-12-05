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

#include "resize.hpp"

namespace vart {
namespace cpu {

template <typename DType>
Resize<DType>::Resize(const xir::Subgraph* subg, const xir::Op* op,
                      IMapTBs_t inputs, CPUTBPtr_t output)
    : CPUOPBase(subg, op, inputs, output) {
  GET_INPUT_DIMX_FMAP(fmap_i_, input, 5);
  GET_OUTPUT_DIMX_FMAP(fmap_o_, 5);
  mode_ = xir_op_->get_attr<std::string>("mode");
  align_corners_ = xir_op_->get_attr<bool>("align_corners");
  half_pixel_centers_ = xir_op_->get_attr<bool>("half_pixel_centers");
  output_f_.resize(fmap_o_.num());
}

template <typename DType>
void Resize<DType>::print_param() {
  fmap_i_.print_param("fmap_i");
  fmap_o_.print_param("fmap_o");

  UNI_LOG_DEBUG_INFO << "mode = " << mode_;
  UNI_LOG_DEBUG_INFO << "align_corners = " << align_corners_;
  UNI_LOG_DEBUG_INFO << "half_pixel_centers = " << half_pixel_centers_;
}

template <typename DType>
void Resize<DType>::run() {
  read();
  resize();
}

template <typename DType>
void Resize<DType>::read() {
  auto* cputb = inputs_.at("input").at(0);
  data_in_ptr_ = GET_CPUTB_DType_PTR(DType, cputb);
  data_out_ptr_ = GET_CPUTB_DType_PTR(DType, output_);
}

struct CachedInterpolation {
  int lower;
  int upper;
  float lerp;
};

template <typename DType>
void Resize<DType>::resize() {
  auto cal_scale = [](std::int64_t in, std::int64_t out,
                      bool align_corners) -> float {
    return (align_corners && out > 1) ? (in - 1) / static_cast<float>(out - 1)
                                      : in / static_cast<float>(out);
  };
  // The calculation of scaler is slightly different bewteen nearest neighbor
  // and linear/bilinear/trilinear when half_pixel_corner is True. In the
  // nearest neighbor mode, tf uses floor((x + 0.5) * s) instead of round((x +
  // 0.5) * s - 0.5). In the linear mode, coordinates are used to derive the
  // interpolation coefficients. tf uses ((x + 0.5) * s - 0.5) directly.
  auto scaler = [](std::int32_t out, float scale, bool half_pixel_centers,
                   float half_pixel_bias) -> float {
    return (half_pixel_centers)
               ? (static_cast<float>(out) + 0.5f) * scale - half_pixel_bias
               : static_cast<float>(out) * scale;
  };
  auto compute_interpolation_weights = [&](const int out_size,
                                           const int in_size, const float scale,
                                           CachedInterpolation* interpolation) {
    interpolation[out_size].lower = 0;
    interpolation[out_size].upper = 0;
    for (int i = out_size - 1; i >= 0; --i) {
      const float in = scaler(i, scale, half_pixel_centers_, 0.5);
      interpolation[i].lower =
          std::max(static_cast<int>(std::floor(in)), static_cast<int>(0));
      interpolation[i].upper = std::min(static_cast<int>(std::ceil(in)),
                                        static_cast<int>(in_size - 1));
      interpolation[i].lerp = in - std::floor(in);
    }
  };
  if (mode_ == "NEAREST") {
    auto h_scale = cal_scale(fmap_i_.h, fmap_o_.h, align_corners_);
    auto w_scale = cal_scale(fmap_i_.w, fmap_o_.w, align_corners_);
    for (int n = 0; n < fmap_o_.n; ++n) {
      for (int h = 0; h < fmap_o_.h; ++h) {
        auto h_scaler = scaler(h, h_scale, half_pixel_centers_, 0.0);
        auto h_idx = std::min(
            (align_corners_) ? static_cast<std::int64_t>(std::round(h_scaler))
                             : static_cast<std::int64_t>(std::floor(h_scaler)),
            fmap_i_.h - 1);
        for (int w = 0; w < fmap_o_.w; ++w) {
          auto w_scaler = scaler(w, w_scale, half_pixel_centers_, 0.0);
          auto w_idx =
              std::min((align_corners_)
                           ? static_cast<std::int64_t>(std::round(w_scaler))
                           : static_cast<std::int64_t>(std::floor(w_scaler)),
                       fmap_i_.w - 1);
          auto idx =
              n * fmap_o_.ncod() + h * fmap_o_.hcod() + w * fmap_o_.wcod();
          auto i_idx = n * fmap_i_.ncod() + h_idx * fmap_i_.hcod() +
                       w_idx * fmap_i_.wcod();
          for (int c = 0; c < fmap_o_.c; ++c) {
            output_f_[idx + c] = data_in_ptr_[i_idx + c];
            data_out_ptr_[idx + c] = data_in_ptr_[i_idx + c];
          }
        }
      }
    }
  } else if (mode_ == "BILINEAR") {
    std::vector<CachedInterpolation> xs(fmap_o_.w + 1);
    std::vector<CachedInterpolation> ys(fmap_o_.h + 1);
    auto h_scale = cal_scale(fmap_i_.h, fmap_o_.h, align_corners_);
    auto w_scale = cal_scale(fmap_i_.w, fmap_o_.w, align_corners_);
    compute_interpolation_weights(fmap_o_.h, fmap_i_.h, h_scale, ys.data());
    compute_interpolation_weights(fmap_o_.w, fmap_i_.w, w_scale, xs.data());
    for (auto& x : xs) {
      x.lower *= fmap_i_.c;
      x.upper *= fmap_i_.c;
    }
    const int in_row_size = fmap_i_.w * fmap_i_.c;
    const int in_batch_num_values = fmap_i_.h * in_row_size;
    const int out_row_size = fmap_o_.w * fmap_o_.c;

    auto compute_lerp = [&](const float top_left, const float top_right,
                            const float bottom_left, const float bottom_right,
                            const float x_lerp, const float y_lerp) -> float {
      const float top = top_left + (top_right - top_left) * x_lerp;
      const float bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;
      return top + (bottom - top) * y_lerp;
    };
    DType* in_ptr = data_in_ptr_;
    int idx_start = 0;
    for (int n = 0; n < fmap_o_.n; ++n) {
      for (int h = 0; h < fmap_o_.h; ++h) {
        const DType* ys_input_lower_ptr = in_ptr + ys[h].lower * in_row_size;
        const DType* ys_input_upper_ptr = in_ptr + ys[h].upper * in_row_size;
        const float ys_lerp = ys[h].lerp;
        for (int w = 0; w < fmap_o_.w; ++w) {
          auto xs_lower = xs[w].lower;
          auto xs_upper = xs[w].upper;
          auto xs_lerp = xs[w].lerp;
          for (int c = 0; c < fmap_o_.c; ++c) {
            const float top_left(ys_input_lower_ptr[xs_lower + c]);
            const float top_right(ys_input_lower_ptr[xs_upper + c]);
            const float bottom_left(ys_input_upper_ptr[xs_lower + c]);
            const float bottom_right(ys_input_upper_ptr[xs_upper + c]);
            auto idx = w * fmap_i_.wcod() + c + idx_start;
            output_f_[idx] = compute_lerp(top_left, top_right, bottom_left,
                                          bottom_right, xs_lerp, ys_lerp);
            data_out_ptr_[idx] = output_f_[idx];
          }
        }
        idx_start += out_row_size;
      }
      in_ptr += in_batch_num_values;
    }
  } else if (mode_ == "TRILINEAR") {
    std::vector<CachedInterpolation> xs(fmap_o_.w + 1);
    std::vector<CachedInterpolation> ys(fmap_o_.h + 1);
    std::vector<CachedInterpolation> ds(fmap_o_.d + 1);
    auto h_scale = cal_scale(fmap_i_.h, fmap_o_.h, align_corners_);
    auto w_scale = cal_scale(fmap_i_.w, fmap_o_.w, align_corners_);
    auto d_scale = cal_scale(fmap_i_.d, fmap_o_.d, align_corners_);

    compute_interpolation_weights(fmap_o_.h, fmap_i_.h, h_scale, ys.data());
    compute_interpolation_weights(fmap_o_.w, fmap_i_.w, w_scale, xs.data());
    compute_interpolation_weights(fmap_o_.d, fmap_i_.d, d_scale, ds.data());

    auto compute_lerp =
        [&](const float top_left_near, const float top_left_far,
            const float top_right_near, const float top_right_far,
            const float bottom_left_near, const float bottom_left_far,
            const float bottom_right_near, const float bottom_right_far,
            const float x_lerp, const float y_lerp,
            const float d_lerp) -> float {
      const float top_near =
          top_left_near * (1 - x_lerp) + top_right_near * x_lerp;
      const float top_far =
          top_left_far * (1 - x_lerp) + top_right_far * x_lerp;
      const float bottom_near =
          bottom_left_near * (1 - x_lerp) + bottom_right_near * x_lerp;
      const float bottom_far =
          bottom_left_far * (1 - x_lerp) + bottom_right_far * x_lerp;
      const float y_near = bottom_near * y_lerp + top_near * (1 - y_lerp);
      const float y_far = bottom_far * y_lerp + top_far * (1 - y_lerp);
      return y_near * (1 - d_lerp) + y_far * d_lerp;
    };
    int start = 0;
    for (auto n = 0; n < fmap_o_.n; ++n) {
      for (auto h = 0; h < fmap_o_.h; ++h) {
        for (auto w = 0; w < fmap_o_.w; ++w) {
          for (auto d = 0; d < fmap_o_.d; ++d) {
            for (auto c = 0; c < fmap_o_.c; ++c) {
              int top_left_near_addr = ys[h].lower * fmap_i_.hcod() +
                                       xs[w].lower * fmap_i_.wcod() +
                                       ds[d].lower * fmap_i_.dcod() + c;
              int top_left_far_addr = ys[h].lower * fmap_i_.hcod() +
                                      xs[w].lower * fmap_i_.wcod() +
                                      ds[d].upper * fmap_i_.dcod() + c;
              int top_right_near_addr = ys[h].lower * fmap_i_.hcod() +
                                        xs[w].upper * fmap_i_.wcod() +
                                        ds[d].lower * fmap_i_.dcod() + c;
              int top_right_far_addr = ys[h].lower * fmap_i_.hcod() +
                                       xs[w].upper * fmap_i_.wcod() +
                                       ds[d].upper * fmap_i_.dcod() + c;
              int bottom_left_near_addr = ys[h].upper * fmap_i_.hcod() +
                                          xs[w].lower * fmap_i_.wcod() +
                                          ds[d].lower * fmap_i_.dcod() + c;
              int bottom_left_far_addr = ys[h].upper * fmap_i_.hcod() +
                                         xs[w].lower * fmap_i_.wcod() +
                                         ds[d].upper * fmap_i_.dcod() + c;
              int bottom_right_near_addr = ys[h].upper * fmap_i_.hcod() +
                                           xs[w].upper * fmap_i_.wcod() +
                                           ds[d].lower * fmap_i_.dcod() + c;
              int bottom_right_far_addr = ys[h].upper * fmap_i_.hcod() +
                                          xs[w].upper * fmap_i_.wcod() +
                                          ds[d].upper * fmap_i_.dcod() + c;
              auto idx = start++;
              output_f_[idx] =
                  compute_lerp(data_in_ptr_[top_left_near_addr],
                               data_in_ptr_[top_left_far_addr],
                               data_in_ptr_[top_right_near_addr],
                               data_in_ptr_[top_right_far_addr],
                               data_in_ptr_[bottom_left_near_addr],
                               data_in_ptr_[bottom_left_far_addr],
                               data_in_ptr_[bottom_right_near_addr],
                               data_in_ptr_[bottom_right_far_addr], xs[w].lerp,
                               ys[h].lerp, ds[d].lerp);
              data_out_ptr_[idx] = output_f_[idx];
            }  // c loop
          }    // d loop
        }      // w loop
      }        // h loop
    }          // n loop
  }            // trilinear
}  // resize

INSTANTIATE_TPCLASS(Resize);
REG_OP_INSTANCE_FUNC("resize", Resize);
REG_OP_INSTANCE_FUNC("upsample", Resize);

}  // namespace cpu
}  // namespace vart
