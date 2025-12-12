
#include <assert.h>
#include <core/session/experimental_onnxruntime_cxx_api.h>
#include <glog/logging.h>

#include <algorithm>  // std::generate
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <vector>
#include <vitis/ai/profiling.hpp>

#include "vitis/ai/env_config.hpp"

#include "getbatch.hpp"
using namespace std;

struct CenterPointResult {
  float bbox[9];
  /// Bounding box 3d: {x, y, z, x_size, y_size, z_size, yaw,vel1,vel2}
  float score;
  /// Score
  uint32_t label;
  //'car',         'truck',   'construction_vehicle',
  //'bus',         'trailer', 'barrier',
  //'motorcycle',  'bicycle', 'pedestrian',
  //'traffic_cone'
};

#include "utils.hpp"

class BEVDetPtOnnx {
 public:
  static std::unique_ptr<BEVDetPtOnnx> create(const std::string& model_name);
  ~BEVDetPtOnnx();
  BEVDetPtOnnx(const std::string& model_name);

  BEVDetPtOnnx(const BEVDetPtOnnx&) = delete;

  size_t getInputWidth();
  size_t getInputHeight();
  size_t get_input_batch();

  std::vector<CenterPointResult> run(
      const std::vector<cv::Mat>& images,
      const std::vector<std::vector<char>>& input_bins);

 private:
  std::string model_name_;
  Ort::SessionOptions session_options_;
  Ort::Env env_;
  Ort::Experimental::Session* session_;
  // std::unique_ptr<Ort::Experimental::Session> session_;
  std::vector<std::vector<int64_t>> input_shapes_;
};



//(image_data - mean) * scale, BRG2RGB and hwc2chw
void set_input_image(const cv::Mat& image, float* data) {
  float mean[3] = {103.53f, 116.28f, 123.675f};
  float scale[3] = {0.01742919f, 0.017507f, 0.01712475f};
  auto channels = 3;
  auto rows = image.rows;
  auto cols = image.cols;
  auto input = image.data;
  for (auto h = 0; h < rows; ++h) {
    for (auto w = 0; w < cols; ++w) {
      for (auto c = 0; c < channels; ++c) {
        float value =
            (float(input[h * cols * channels + w * channels + c]) - mean[c]) *
            scale[c];
        data[abs(c - 2) * rows * cols + h * cols + w] = value;
      }
    }
  }
}

#if 0
static void CheckStatus(OrtStatus* status) {
  if (status != NULL) {
    const char* msg = Ort::GetApi().GetErrorMessage(status);
    fprintf(stderr, "%s\n", msg);
    Ort::GetApi().ReleaseStatus(status);
    exit(1);
  }
}
#endif

// pretty prints a shape dimension vector
// static std::string print_shape(const std::vector<int64_t>& v) {
//  std::stringstream ss("");
//  for (size_t i = 0; i < v.size() - 1; i++) ss << v[i] << "x";
//  ss << v[v.size() - 1];
//  return ss.str();
//}

static int calculate_product(const std::vector<int64_t>& v) {
  int total = 1;
  for (auto& i : v) total *= (int)i;
  return total;
}

std::unique_ptr<BEVDetPtOnnx> BEVDetPtOnnx::create(
    const std::string& model_name) {
  return std::make_unique<BEVDetPtOnnx>(model_name);
}
BEVDetPtOnnx::~BEVDetPtOnnx() {
  LOG(INFO) << "~BEVDetPtOnnx()";
  delete session_;
}
BEVDetPtOnnx::BEVDetPtOnnx(const std::string& model_name)
    : model_name_{model_name},
      session_options_{Ort::SessionOptions()},
      env_{Ort::Env(ORT_LOGGING_LEVEL_WARNING, "bevdet")},
      session_{nullptr} {
  get_batch();

    auto options = std::unordered_map<std::string,std::string>({});
    options["config_file"] = "/usr/bin/vaip_config.json";
    // optional, eg: cache path and cache key: /tmp/my_cache/abcdefg
    // options["CacheDir"] = "/tmp/my_cache";
    // options["CacheKey"] = "abcdefg";
    session_options_.AppendExecutionProvider("VitisAI", options );
}

size_t BEVDetPtOnnx::getInputWidth() { return input_shapes_[0][3]; }
size_t BEVDetPtOnnx::getInputHeight() { return input_shapes_[0][2]; }
size_t BEVDetPtOnnx::get_input_batch() { return input_shapes_[0][0]; }

std::vector<CenterPointResult> BEVDetPtOnnx::run(
    const std::vector<cv::Mat>& images,
    const std::vector<std::vector<char>>& input_bins) {
  if (session_ == nullptr) {
    __TIC__(ONNX_INIT_BEVDET)
    session_ =
        new Ort::Experimental::Session(env_, model_name_, session_options_);
    input_shapes_ = session_->GetInputShapes();
    input_shapes_[0][0] = g_batchnum;
    __TOC__(ONNX_INIT_BEVDET)
  }
  std::vector<std::string> input_names = session_->GetInputNames();
  __TIC__(ONNX_per_process)
  std::vector<std::vector<int64_t>> input_shapes = session_->GetInputShapes();
  std::vector<std::vector<float>> input_tensor_values_;
  input_tensor_values_.resize(input_shapes_.size());

  // 0. input0  images
  for (size_t j = 0; j < images.size(); j++) {
    auto image_resize = resize_and_crop_image(images[j]);
    int datasize = calculate_product(input_shapes_[0]);
    input_tensor_values_[0].resize(datasize);
    set_input_image(image_resize,
                    input_tensor_values_[0].data() + 3 * 256 * 704 * j);
  }

  // 1. input2 bins1 bins0
  {
    float* bins1 = (float*)input_bins[1].data();
    float* bins0 = (float*)input_bins[0].data();

    int datasize = calculate_product(input_shapes_[1]);
    input_tensor_values_[1].resize(datasize);
    for (size_t i = 0; i < input_tensor_values_[1].size() / 16; i++) {
      auto dst = input_tensor_values_[1].data() + i * 16;
      memcpy(dst, bins1 + i * 9, 3 * sizeof(float));
      memcpy(dst + 4, bins1 + 3 + i * 9, 3 * sizeof(float));
      memcpy(dst + 8, bins1 + 6 + i * 9, 3 * sizeof(float));
      dst[3] = bins0[0 + i * 3];
      dst[7] = bins0[1 + i * 3];
      dst[11] = bins0[2 + i * 3];
    }
  }

  // 2.input_3 bins3
  {
    int datasize = calculate_product(input_shapes_[2]);
    input_tensor_values_[2].resize(datasize);
    CHECK_EQ(datasize * 4, input_bins[3].size());
    memcpy(input_tensor_values_[2].data(), input_bins[3].data(), datasize * 4);
  }

  // 3.input_4 bins2
  {
    int datasize = calculate_product(input_shapes_[3]);
    input_tensor_values_[3].resize(datasize);
    float* bins2 = (float*)input_bins[2].data();
    float* input3 = input_tensor_values_[3].data();
    for (int i = 0; i < datasize; i++) {
      // bins2[i] = input3[i];
      input3[i] = bins2[i * 3];
    }
  }
  std::vector<Ort::Value> input_tensors;
  for (size_t i = 0; i < input_shapes_.size(); i++) {
    input_tensors.emplace_back(Ort::Experimental::Value::CreateTensor<float>(
        input_tensor_values_[i].data(), input_tensor_values_[i].size(),
        input_shapes_[i]));
  }

  __TOC__(ONNX_per_process)
  __TIC__(ONNX_RUN)
  auto output_tensors = session_->Run(session_->GetInputNames(), input_tensors,
                                      session_->GetOutputNames());
  __TOC__(ONNX_RUN)
  __TIC__(ONNX_post_process)
  auto res = post_process(output_tensors, 0.05);
  __TOC__(ONNX_post_process)

  return res;
}
//
// protected:
// std::vector<CenterPointResult> postprocess(Ort::Value& output_tensor) {
//  std::vector<CenterPointResult> results;
//
//  // auto output_tensor_ptr = output_tensor.GetTensorMutableData<float>();
//  // for (auto index = 0; index < batch; ++index) {
//  //   auto softmax_output =
//  //       softmax(output_tensor_ptr + channel * index, channel);
//  //   auto tb_top5 = topk(softmax_output, 5);
//  //   // std::cout << "batch_index: " << index << std::endl;
//  //   // print_topk(tb_top5);
//  //   Resnet50PtOnnxResult r;
//  //
//  //  for (const auto& v : tb_top5) {
//  //    r.scores.push_back(Resnet50PtOnnxResult::Score{v.first, v.second});
//  //  }
//  //  results.emplace_back(r);
//  //}
//  return results;
//}
