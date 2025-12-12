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

/*
  The following source code derives from Darknet
*/
#include <opencv2/core.hpp>

namespace vitis {
namespace ai {
namespace math {

typedef struct {
  int w;
  int h;
  int c;
  float *data;
} image;

/**
 * @brief Frees the memory allocated for an image.
 *
 * @param m The image to be freed.
 */
void free_image(image m);

/**
 * @brief Creates an empty image with the specified dimensions and channels.
 *
 * @param w Width of the image.
 * @param h Height of the image.
 * @param c Number of channels in the image.
 * @return The created empty image.
 */
image make_empty_image(int w, int h, int c);

/**
 * @brief Creates an image with the specified dimensions and channels, and
 * allocates memory for it.
 *
 * @param w Width of the image.
 * @param h Height of the image.
 * @param c Number of channels in the image.
 * @return The created image.
 */
image make_image(int w, int h, int c);

/**
 * @brief Retrieves the pixel value at the specified coordinates in the image.
 *
 * @param m The image.
 * @param x x-coordinate of the pixel.
 * @param y y-coordinate of the pixel.
 * @param c Channel index of the pixel.
 * @return The pixel value at the specified coordinates.
 */
float get_pixel(image m, int x, int y, int c);

/**
 * @brief Sets the pixel value at the specified coordinates in the image.
 *
 * @param m The image.
 * @param x x-coordinate of the pixel.
 * @param y y-coordinate of the pixel.
 * @param c Channel index of the pixel.
 * @param val The value to set at the specified coordinates.
 */
void set_pixel(image m, int x, int y, int c, float val);

/**
 * @brief Adds a value to the pixel at the specified coordinates in the image.
 *
 * @param m The image.
 * @param x x-coordinate of the pixel.
 * @param y y-coordinate of the pixel.
 * @param c Channel index of the pixel.
 * @param val The value to add to the pixel at the specified coordinates.
 */
void add_pixel(image m, int x, int y, int c, float val);

/**
 * @brief Fills the entire image with the specified value.
 *
 * @param m The image.
 * @param s The value to fill the image with.
 */
void fill_image(image m, float s);

/**
 * @brief Embeds the source image into the destination image at the specified
 * coordinates.
 *
 * @param source The source image to be embedded.
 * @param dest The destination image.
 * @param dx x-coordinate in the destination image where the source image will
 * be embedded.
 * @param dy y-coordinate in the destination image where the source image will
 * be embedded.
 */
void embed_image(image source, image dest, int dx, int dy);

/**
 * @brief Resizes the image to the specified dimensions.
 *
 * @param im The image to be resized.
 * @param w Width of the resized image.
 * @param h Height of the resized image.
 * @return The resized image.
 */
image resize_image(image im, int w, int h);

/**
 * @brief Loads an image from the given OpenCV matrix.
 *
 * @param img The OpenCV matrix representing the image.
 * @return The loaded image.
 */
image load_image_cv(const cv::Mat &img);

/**
 * @brief Creates a letterboxed version of the image with the specified
 * dimensions.
 *
 * @param im The image to be letterboxed.
 * @param w Width of the letterboxed image.
 * @param h Height of the letterboxed image.
 * @return The letterboxed image.
 */
image letterbox_image(image im, int w, int h);

}  // namespace math
}  // namespace ai
}  // namespace vitis

