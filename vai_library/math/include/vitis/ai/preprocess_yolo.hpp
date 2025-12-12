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

// the file is designed for providing pre-processing operations for YOLO series
// models.

#include <opencv2/core.hpp>

namespace vitis {
namespace ai {
namespace math {

/**
 * @brief Converts an input image to a specific format and scales the pixel
 * values. Used for yolov3, yolov4, yolov5 those tf_flag_=false, and yolov2.
 *
 * @param frame The input image as a cv::Mat.
 * @param width The desired width of the converted image.
 * @param height The desired height of the converted image.
 * @param channel The number of channels in the converted image.
 * @param scale The scaling factor for the pixel values.
 * @param[out] data The pointer to the output data array.
 */
void convertInputImage(const cv::Mat& frame, const int width, const int height,
                       const int channel, const float scale, int8_t* data);

/**
 * @brief Overload method with float input data for DPUV1
 *
 * @param frame The input image as a cv::Mat.
 * @param width The desired width of the converted image.
 * @param height The desired height of the converted image.
 * @param channel The number of channels in the converted image.
 * @param scale The scaling factor for the pixel values.
 * @param[out] data The pointer to the output data array.
 */
void convertInputImage(const cv::Mat& frame, int width, int height, int channel,
                       float scale, float* data);

/**
 * @brief Has the same functionality as the convertInputImage function.
 *
 * It takes an input image of any dimensions and resizes it proportionally
 * to fit within a new width and height. The resultant image is then centered
 * and padded with a constant value 128.
 *
 * @param im The input cv::Mat image to be resized and padded
 * @param w The target width of the output image
 * @param h The target height of the output image
 * @return The resized and padded cv::Mat image with the specified width and
 * height
 */
cv::Mat letterbox_v2(const cv::Mat& im, const int w, const int h);

/**
 * @brief Resize and pad an image while preserving its aspect ratio. Used for
 * yolov4, yolov5 models those tf_flag_=true and type=1 or 2 as well as yolov7.
 *
 * It takes an input image of any dimensions and resizes it proportionally
 * to fit within a new width and height. The resultant image is then centered
 * and padded with a constant value 114.
 *
 * @param im The input cv::Mat image to be resized and padded
 * @param w The target width of the output image
 * @param h The target height of the output image
 * @return The resized and padded cv::Mat image with the specified width and
 * height
 */
cv::Mat letterbox_v4(const cv::Mat& im, const int w, const int h);

/**
 * @brief Resize and pad an image using cv::warpAffine while preserving its
 * aspect ratio. No model use it.
 *
 * @param im The input cv::Mat image to be resized and padded
 * @param w The target width of the output image
 * @param h The target height of the output image
 * @return The resized and padded cv::Mat image with the specified width and
 * height
 */
cv::Mat letterbox_tf(const cv::Mat& im, const int w, const int h);

/**
 * @brief Applies letterboxing to an input image, resizing and padding it to a
 * target size. Used for yolov6.
 *
 * It takes an input image of any dimensions and resizes it proportionally
 * to fit within a load_size. The resultant image is then centered
 * and padded with a constant value 114.
 *
 * @param im The input image to be letterboxed.
 * @param w The target width of the letterboxed image.
 * @param h The target height of the letterboxed image.
 * @param load_size The size to which the input image is scaled before
 * letterboxing.
 * @param om The output matrix where the letterboxed image will be stored.
 * @param scale The scaling factor applied to the input image.
 * @param left The number of pixels added as padding on the left side of the
 * letterboxed image.
 * @param top The number of pixels added as padding on the top side of the
 * letterboxed image.
 */
void letterbox_v6(const cv::Mat& im, const int w, const int h, int load_size,
                  cv::Mat& om, float& scale, int& left, int& top);

/**
 * @brief Applies letterboxing to an input image, resizing and padding it to a
 * target size. Used for yolov8.
 *
 * It takes an input image of any dimensions and resizes it proportionally
 * to fit within a new width and height. The resultant image is then centered
 * and padded with a constant value 114.
 *
 * @param input_image The input image to be letterboxed.
 * @param output_image [out] The output matrix where the letterboxed image will
 * be stored.
 * @param height The target height of the letterboxed image.
 * @param width The target width of the letterboxed image.
 * @param scale [out] The scaling factor applied to the input image.
 * @param left [out] The number of pixels added as padding on the left side of
 * the letterboxed image.
 * @param top [out] The number of pixels added as padding on the top side of the
 * letterboxed image.
 */
void letterbox_v8(const cv::Mat input_image, cv::Mat& output_image,
                  const int height, const int width, float& scale, int& left,
                  int& top);

/**
 * @brief Applies letterboxing to an input image.Used for yolovx.
 *
 * It takes an input image of any dimensions and resizes it proportionally
 * to fit within a new width and height. The resultant image is then centered
 * and padded with a constant value 114.
 *
 * @param im The input image.
 * @param w The desired width of the output image.
 * @param h The desired height of the output image.
 * @param om The output image. It should be preallocated with size cv::Size(w,
 * h) and type CV_8UC3.
 * @param scale The scale factor applied to the input image. It is updated with
 * the calculated scale factor.
 */
void letterbox_vx(const cv::Mat& im, const int w, const int h, cv::Mat& om,
                  float& scale);

/**
 * @brief Applies letterboxing to an input image and returns the result. Used
 * for ofa-yolo.
 *
 * It takes an input image of any dimensions and resizes it proportionally
 * to fit within a new width and height. The resultant image is then centered
 * and padded with a constant value 114.
 *
 * @param im The input image.
 * @param w The desired width of the output image.
 * @param h The desired height of the output image.
 * @return The letterboxed image.
 */
cv::Mat letterbox_ofa(const cv::Mat& im, const int w, const int h);

}  // namespace math
}  // namespace ai
}  // namespace vitis

