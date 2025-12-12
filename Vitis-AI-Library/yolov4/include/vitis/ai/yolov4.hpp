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
/*
 * Filename: yolov4.hpp
 *
 * Description:
 * This network is used to detecting object from an image, it will return
 * its coordinate, label and confidence.
 *
 * Please refer to document "Xilinx_AI_SDK_User_Guide.pdf" for more details
 * of these APIs.
 */
#pragma once
#include <memory>
#include <opencv2/core.hpp>
#include <vector>
#include <vitis/ai/configurable_dpu_task.hpp>
#include <vitis/ai/nnpp/yolov4.hpp>
namespace vitis {
namespace ai {

/**
 * @brief Base class for detecting objects in the input image (cv::Mat).
 *
 * Input is an image (cv::Mat).
 *
 * Output is the position of the pedestrians in the input image.
 *
 * Sample code:
 * @code
    auto yolo =
 vitis::ai::YOLOv4::create("yolov4", true);
    Mat img = cv::imread("sample_yolov4.jpg");

    auto results = yolo->run(img);

    for(auto &box : results.bboxes){
      int label = box.label;
      float xmin = box.x * img.cols + 1;
      float ymin = box.y * img.rows + 1;
      float xmax = xmin + box.width * img.cols;
      float ymax = ymin + box.height * img.rows;
      if(xmin < 0.) xmin = 1.;
      if(ymin < 0.) ymin = 1.;
      if(xmax > img.cols) xmax = img.cols;
      if(ymax > img.rows) ymax = img.rows;
      float confidence = box.score;

      cout << "RESULT: " << label << "\t" << xmin << "\t" << ymin << "\t"
           << xmax << "\t" << ymax << "\t" << confidence << "\n";
      if (label == 0) {
        rectangle(img, Point(xmin, ymin), Point(xmax, ymax), Scalar(0, 255, 0),
                  1, 1, 0);
      } else if (label == 1) {
        rectangle(img, Point(xmin, ymin), Point(xmax, ymax), Scalar(255, 0, 0),
                  1, 1, 0);
      } else if (label == 2) {
        rectangle(img, Point(xmin, ymin), Point(xmax, ymax), Scalar(0, 0, 255),
                  1, 1, 0);
      } else if (label == 3) {
        rectangle(img, Point(xmin, ymin), Point(xmax, ymax),
                  Scalar(0, 255, 255), 1, 1, 0);
      }

    }
    imwrite("sample_yolov4_result.jpg", img);
   @endcode
 *
 * Display of the model results:
 * @image latex images/sample_yolov4_result.jpg "out image" width=\textwidth
 */
class YOLOv4 : public ConfigurableDpuTaskBase {
 public:
  /**
   * @brief Factory function to get an instance of derived classes of class
   * YOLOv4.
   *
   * @param model_name Model name
   *
   * @param need_preprocess Normalize with mean/scale or not, default
   *value is true.
   *
   * @return An instance of YOLOv4 class.
   *
   */
  static std::unique_ptr<YOLOv4> create(const std::string& model_name,
                                        bool need_preprocess = true);
  /**
   * @cond NOCOMMENTS
   */
 protected:
  explicit YOLOv4(const std::string& model_name, bool need_preprocess);
  YOLOv4(const YOLOv4&) = delete;

 public:
  virtual ~YOLOv4();
  /**
   * @endcond
   */
 public:
  /**
   * @brief Function to get running result of the YOLOv4 neural network.
   *
   * @param image Input data of input image (cv::Mat).
   *
   * @return YOLOv4Result.
   *
   */
  virtual YOLOv4Result run(const cv::Mat& image) = 0;
  /**
   * @brief Function to get running result of the YOLOv4 neural network
   * in batch mode.
   *
   * @param images Input data of input images (std:vector<cv::Mat>). The size of
   * input images equals batch size obtained by get_input_batch.
   *
   * @return The vector of YOLOv4Result.
   *
   */
  virtual std::vector<YOLOv4Result> run(const std::vector<cv::Mat>& images) = 0;
  /**
   * @brief Function to get running result of the YOLOv4 neural network
   * in batch mode, used to receive user's xrt_bo to support zero copy.
   *
   * @param input_bos The vector of vart::xrt_bo_t.
   *
   * @return The vector of YOLOv4Result.
   *
   */
  virtual std::vector<YOLOv4Result> run(
      const std::vector<vart::xrt_bo_t>& input_bos) = 0;
};
}  // namespace ai
}  // namespace vitis
