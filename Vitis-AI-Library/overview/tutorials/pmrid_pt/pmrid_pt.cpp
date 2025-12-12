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
#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <memory>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <xir/graph/graph.hpp>
#include <vitis/ai/profiling.hpp>

#include "vart/runner_ext.hpp"

static std::vector<std::pair<std::string, float>> parse_raw_info(const std::string& filename);
static float K(const float iso);
static float Sigma(const float iso);
static float KSigma(const float iso, const float value, const float scale);
static float get_input_scale(const xir::Tensor* tensor);
static float get_output_scale(const xir::Tensor* tensor);
static cv::Mat read_raw_image(const std::string& filename, const int height, const int width);
static void bayer2rggb_KSigma(const unsigned short* input, std::vector<signed char>& rggb, 
		const float iso, const int rows, const int cols, const int channels, 
		const float scale, const float input_scale);
static void pad(const std::vector<signed char>& input, const int rows, const int cols, 
		const int channels, const int ph, const int pw, const signed char pad_value, void* data);
static void set_input(const cv::Mat& raw, void* data, const float iso,
		const int height, const int width, const int channels, 
		const int ph, const int pw, const float fix_scale);
static std::vector<float> invKSigma_unpad_rggb2bayer(void* output_data, void* input_data, const float output_scale, 
		const float input_scale, const int rows, const int cols, const int input_width, const int channels, 
		const int ph, const int pw, const float iso, const float scale);
static void dump_output(const std::vector<float>& result, const std::string& filename);

//parse raw_info from txt file, first para is raw fileneme and second para is ISO for every line.
static std::vector<std::pair<std::string, float>> parse_raw_info(const std::string& filename) {
  auto ret = std::vector<std::pair<std::string, float>>{};
  std::ifstream fin;
  fin.open(filename, std::ios_base::in);
  if(!fin)  {
     std::cout<<"Can't open the file " << filename << "\n";
     exit(-1);
  }
  std::string line;
  while(std::getline(fin, line)) {
    if (line.empty()) continue;
    std::istringstream ss(line);
    std::string name = "";  //raw fileneme
    int iso = 0;  //ISO parameter, KSigma transform need this parameter
    ss >> name >> iso;
    ret.push_back(std::make_pair(name, (float)iso));
  }
  fin.close();
  for(const auto &r : ret) {
    LOG(INFO) << "input filename : " << r.first << "   ISO : " << r.second;
  }
  return ret;
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cout << "usage: " << argv[0]
         << " <pmrid_pt.xmodel> raw_info.txt \n";
    return 0;
  }
  auto xmodel_file = std::string(argv[1]);
  auto raw_info = parse_raw_info(argv[2]);

  //  create dpu runner
  auto graph = xir::Graph::deserialize(xmodel_file);
  auto root = graph->get_root_subgraph();
  xir::Subgraph* subgraph = nullptr;
  for (auto c : root->children_topological_sort()) {
    CHECK(c->has_attr("device"));
    if (c->get_attr<std::string>("device") == "DPU") {
      subgraph = c;
      break;
    }
  }
  auto attrs = xir::Attrs::create();
  std::unique_ptr<vart::RunnerExt> runner =
      vart::RunnerExt::create_runner(subgraph, attrs.get());

  // get input & output tensor buffers
  auto input_tensor_buffers = runner->get_inputs();
  auto output_tensor_buffers = runner->get_outputs();
  CHECK_EQ(input_tensor_buffers.size(), 1u) << "only support pmrid_pt model";
  CHECK_EQ(output_tensor_buffers.size(), 1u) << "only support pmrid_pt model";

  // get input_scale & output_scale
  auto input_tensor = input_tensor_buffers[0]->get_tensor();
  auto input_scale = get_input_scale(input_tensor);

  auto output_tensor = output_tensor_buffers[0]->get_tensor();
  auto output_scale = get_output_scale(output_tensor);

  auto batch = input_tensor->get_shape().at(0);
  auto input_height = input_tensor->get_shape().at(1);
  auto input_width = input_tensor->get_shape().at(2);
  auto channels = input_tensor->get_shape().at(3);
  
  //raw image shape and pad parameter
  auto raw_height = 3000;
  auto raw_width = 4000;
  auto ph = (32 - ((raw_height / 2) % 32)) / 2;
  auto pw = (32 - ((raw_width / 2) % 32)) / 2;
  CHECK((raw_height % 2) || (raw_width % 2) == 0)
      << "raw_height is not 2N, or raw_width is not 2N.";
  CHECK((raw_height / 2 + ph * 2 == input_height) && (raw_width / 2 + pw * 2 == input_width))
      << "raw image size not match with xmodel input(N*1504*2016*4).";

  // loop for running raw_data in raw_info.txt
  for (auto i = 0; i < raw_info.size(); i += batch) {
    auto run_batch = std::min(((int)raw_info.size() - i), batch);
    auto raw_images = std::vector<cv::Mat>(run_batch);

    //read raw data
    for (auto idx = 0; idx < run_batch; ++idx) {
      raw_images[idx] = read_raw_image(raw_info[i + idx].first, raw_height, raw_width);
    }

    // preprocessing and set the input
    __TIC__(PRE_PROCESS)
    for (auto batch_idx = 0; batch_idx < run_batch; ++batch_idx) {
      uint64_t data_in = 0u;
      size_t size_in = 0u;
      std::tie(data_in, size_in) =
          input_tensor_buffers[0]->data(std::vector<int>{batch_idx, 0, 0, 0});
      CHECK_NE(size_in, 0u);
      set_input(raw_images[batch_idx], (void*)data_in, raw_info[i + batch_idx].second, raw_height, raw_width, channels, ph, pw, input_scale);
    }
    __TOC__(PRE_PROCESS)

    // sync data for input
    for (auto& input : input_tensor_buffers) {
      input->sync_for_write(0, input->get_tensor()->get_data_size() /
                                   input->get_tensor()->get_shape()[0]);
    }
    // start the dpu
    auto v = runner->execute_async(input_tensor_buffers, output_tensor_buffers);
    auto status = runner->wait((int)v.first, -1);
    CHECK_EQ(status, 0) << "failed to run dpu";
    // sync data for output
    for (auto& output : output_tensor_buffers) {
      output->sync_for_read(0, output->get_tensor()->get_data_size() /
                                   output->get_tensor()->get_shape()[0]);
    }

    // postprocessing
    __TIC__(POST_PROCESS)
    auto result = std::vector<std::vector<float>>(run_batch);
    for (auto batch_idx = 0; batch_idx < run_batch; ++batch_idx) {
      uint64_t data_out = 0u;
      size_t size_out = 0u;
      std::tie(data_out, size_out) = output_tensor_buffers[0]->data(std::vector<int>{batch_idx, 0, 0, 0});
    
      uint64_t data_in = 0u;
      size_t size_in = 0u;
      std::tie(data_in, size_in) = input_tensor_buffers[0]->data(std::vector<int>{batch_idx, 0, 0, 0});

      CHECK_EQ(size_out, size_in) << "output size must equal to input size";

      result[batch_idx] = invKSigma_unpad_rggb2bayer((void*)data_out, (void*)data_in, output_scale, 1 / input_scale, raw_height, raw_width, input_width, channels, ph, pw, raw_info[i + batch_idx].second, 256.0f);
    }
    __TOC__(POST_PROCESS)

    //dump output file
    for (auto batch_idx = 0; batch_idx < run_batch; ++batch_idx) {
      dump_output(result[batch_idx], raw_info[i + batch_idx].first);
    }

    LOG(INFO) << "Completed " << i + run_batch << " raw images Denoising.";
    LOG(INFO) << " ";
    LOG(INFO) << " ";
  }

  return 0;
}

//used to KSigma transform
static float K(const float iso) {
  float k_coeff[2] = {0.0005995267, 0.00868861};
  return k_coeff[0] * iso + k_coeff[1];
}

//used to KSigma transform
static float Sigma(const float iso) {
  float s_coeff[3] = {7.11772e-7, 6.514934e-4, 0.11492713};
  return s_coeff[0] * iso * iso + s_coeff[1] * iso + s_coeff[2];
}

//KSigma transform
static float KSigma(const float iso, const float value, const float scale) {
  float ret;
  float anchor = 1600;
  float v = 959.0;
  auto k = K(iso);
  auto sigma = Sigma(iso);
  auto k_a = K(anchor);
  auto sigma_a = Sigma(anchor);
  auto cvt_k = k_a / k;
  auto cvt_b = (sigma / (k * k) - sigma_a / (k_a * k_a)) * k_a;
  ret = (value * v * cvt_k + cvt_b) / v * scale;
  return ret;
}

// fix_point to scale for input tensor
static float get_input_scale(const xir::Tensor* tensor) {
  int fixpos = tensor->template get_attr<int>("fix_point");
  return std::exp2f(1.0f * (float)fixpos);
}

// fix_point to scale for output tensor
static float get_output_scale(const xir::Tensor* tensor) {
  int fixpos = tensor->template get_attr<int>("fix_point");
  return std::exp2f(-1.0f * (float)fixpos);
}

//read raw data, raw data is 16bit and 1 channels
static cv::Mat read_raw_image(const std::string& filename, const int height, const int width) {
  cv::Mat ret;
  ret.create(height, width, CV_16UC1);
  auto mode = std::ios_base::in | std::ios_base::binary;
  auto flag = std::ifstream(filename, mode)
                  .read((char*)ret.data, height * width * 2)
                  .good();

  LOG_IF(INFO, !flag) << "fail to read! filename=" << filename;
  return ret;
}

//bayer to rggb and KSigma transform, combine to 1 operator to improve performance
static void bayer2rggb_KSigma(const unsigned short* input, std::vector<signed char>& rggb, 
		const float iso, const int rows, const int cols, const int channels, 
		const float scale, const float input_scale) {
  float anchor = 1600;
  float v = 959.0;
  auto k = K(iso);
  auto sigma = Sigma(iso);
  auto k_a = K(anchor);
  auto sigma_a = Sigma(anchor);
  auto cvt_k = k_a / k;
  auto cvt_b = (sigma / (k * k) - sigma_a / (k_a * k_a)) * k_a;

  CHECK((rows % 2) || (cols % 2) == 0)
      << "rows is not 2N, or cols is not 2N";
  auto new_rows = rows / 2;
  auto new_cols = cols / 2;
  //Normalized    input / 65535
  //KSigma        (input * v * cvt_k + cvt_b) / v
  //Normalized    input * scale
  //fixed         int(input * input_scale + 0.5f),  +0.5f because rounding mode
  //bayer2rggb    R G R G   ->RGGBRGGB
  //              G B G B
  float a = 1.0f / 65535 * v * cvt_k / v * scale * input_scale;
  float b = cvt_b / v * scale * input_scale + 0.5f;
  for (auto h = 0; h < new_rows; ++h) {
    for (auto w = 0; w < new_cols; ++w) {
      auto idx = h * new_cols * channels + w * channels; 
//      rggb[idx] = (signed char)((float)input[h * 2 * cols + w * 2 + 0] * a + b);
//      rggb[idx + 1] = (signed char)((float)input[h * 2 * cols + w * 2 + 1] * a + b);
//      rggb[idx + 2] = (signed char)((float)input[(h * 2 + 1) * cols + w * 2 + 0] * a + b);
//      rggb[idx + 3] = (signed char)((float)input[(h * 2 + 1) * cols + w * 2 + 1] * a + b);

//      rggb[idx]     = (int)std::max(std::min((float)input[h * 2 * cols + w * 2 + 0] * a + b, 127.0f), -128.0f);
//      rggb[idx + 1] = (int)std::max(std::min((float)input[h * 2 * cols + w * 2 + 1] * a + b, 127.0f), -128.0f);
//      rggb[idx + 2] = (int)std::max(std::min((float)input[(h * 2 + 1) * cols + w * 2 + 0] * a + b, 127.0f), -128.0f);
//      rggb[idx + 3] = (int)std::max(std::min((float)input[(h * 2 + 1) * cols + w * 2 + 1] * a + b, 127.0f), -128.0f);

      rggb[idx]     = (int)std::min((float)input[h * 2 * cols + w * 2 + 0] * a + b, 127.0f);
      rggb[idx + 1] = (int)std::min((float)input[h * 2 * cols + w * 2 + 1] * a + b, 127.0f);
      rggb[idx + 2] = (int)std::min((float)input[(h * 2 + 1) * cols + w * 2 + 0] * a + b, 127.0f);
      rggb[idx + 3] = (int)std::min((float)input[(h * 2 + 1) * cols + w * 2 + 1] * a + b, 127.0f);
    }
  }
}

//pad operator, pad_value use KSigma and fixed value of 0.0f
static void pad(const std::vector<signed char>& input, const int rows, const int cols, 
		const int channels, const int ph, const int pw, const signed char pad_value, void* data) {
//  __TIC__(pad)
  auto dst_rows = rows + ph * 2;
  auto dst_cols = cols + pw * 2;
  auto src_row_size = cols * channels;
  auto dst_row_size = dst_cols * channels;
  auto src = input.data();
  signed char* dst = (signed char*)data;
  // pad top and bottom
  std::fill_n(dst, ph * dst_row_size, pad_value);
  std::fill_n(dst + (dst_rows - ph) * dst_row_size,
                      ph * dst_row_size, pad_value);
  // pad left and right
  for (auto h = ph; h < dst_rows - ph; h++) {
    auto offset = h * dst_row_size;
    std::fill_n(dst + offset, pw * channels, pad_value);
  }
  for (auto h = ph; h < dst_rows - ph; h++) {
    auto offset =
         h * dst_row_size + (dst_cols - pw) * channels;
    std::fill_n(dst + offset, pw * channels, pad_value);
  }
  // copy source data
  for (auto h = ph; h < dst_rows - ph; h++) {
    auto src_offset = (h - ph) * src_row_size;
    auto dst_offset = h * dst_row_size + pw * channels;
    std::copy_n(src + src_offset, src_row_size, dst + dst_offset);
  }
//  __TOC__(pad)
}

static void set_input(const cv::Mat& raw, void* data, const float iso,
		const int height, const int width, const int channels, 
		const int ph, const int pw, const float fix_scale) {
  //reverse order all the data
  cv::Mat raw_flip;
  cv::flip(raw, raw_flip, -1);

  //Normalized, bayer2rggb, KSigma, fixed all the raw data
  auto rggb = std::vector<signed char>(height * width);
  bayer2rggb_KSigma(raw_flip.ptr<ushort>(0), rggb, iso, raw_flip.rows, raw_flip.cols, channels, 256.0f, fix_scale);

  //pad_value use KSigma and fixed value of 0.0f
  signed char pad_value = (int)std::max(std::min(KSigma(iso, 0.0f, 256.0f) * fix_scale + 0.5f, 127.0f), -128.0f);
  pad(rggb, height / 2, width / 2, channels, ph, pw, pad_value, (void*)data);
}

//inverse KSigma transform, unpad and rggb to bayer, combine to 1 operator to improve performance
static std::vector<float> invKSigma_unpad_rggb2bayer(void* output_data, void* input_data, const float output_scale, 
		const float input_scale, const int rows, const int cols, const int input_width, const int channels, 
		const int ph, const int pw, const float iso, const float scale) {
  auto ret = std::vector<float>(rows * cols);
  float anchor = 1600;
  float v = 959.0;
  auto k = K(iso);
  auto sigma = Sigma(iso);
  auto k_a = K(anchor);
  auto sigma_a = Sigma(anchor);
  auto cvt_k = k_a / k;
  auto cvt_b = (sigma / (k * k) - sigma_a / (k_a * k_a)) * k_a;

  signed char* output = (signed char*)output_data;
  signed char* input = (signed char*)input_data;

  //output+input in float    output * output_scale + input * input_scale
  //Normalized               output / scale
  //inverse KSigma           (output * v - cvt_b) / cvt_k / v
  //rggb2bayer               RGGBRGGB  ->  R G R G
  //                                       G B G B

  float a = input_scale / scale * v / cvt_k / v;
  float b = output_scale / scale * v / cvt_k / v;
  float c = cvt_b / cvt_k / v;

//  float a = input_scale / scale * v;
//  float b = output_scale / scale * v;

  for (auto h = 0; h < rows; h = h + 2) {
    for(auto w = 0; w < cols; w = w + 2) {
      auto idx = input_width * channels * (ph + h / 2) + channels * (pw + w / 2); 

      ret[h * cols + w] = (float)input[idx] * a + (float)output[idx] * b - c; 
      ret[h * cols + w + 1] = (float)input[idx + 1] * a + (float)output[idx + 1] * b - c; 
      ret[(h + 1) * cols + w] = (float)input[idx + 2] * a + (float)output[idx + 2] * b - c; 
      ret[(h + 1) * cols + w + 1] = (float)input[idx + 3] * a + (float)output[idx + 3] * b - c; 

//      ret[h * cols + w] = ((float)input[idx] * a + (float)output[idx] * b - cvt_b) / cvt_k / v; 
//      ret[h * cols + w + 1] = ((float)input[idx + 1] * a + (float)output[idx + 1] * b - cvt_b) / cvt_k / v; 
//      ret[(h + 1) * cols + w] = ((float)input[idx + 2] * a + (float)output[idx + 2] * b - cvt_b) / cvt_k / v; 
//      ret[(h + 1) * cols + w + 1] = ((float)input[idx + 3] * a + (float)output[idx + 3] * b - cvt_b) / cvt_k / v; 
    }
  }
  return ret;
}

//dump output to file to calculate PSNR
static void dump_output(const std::vector<float>& result, const std::string& filename) { 
  auto mode = std::ios_base::out | std::ios_base::binary | std::ios_base::trunc;
  auto output_file = filename + std::string(".out");
  LOG(INFO) << "dump output to " << output_file;
  CHECK(std::ofstream(output_file, mode)
            .write((char*)result.data(), sizeof(float) * result.size())
            .good())
      << " faild to write to " << output_file;
}
