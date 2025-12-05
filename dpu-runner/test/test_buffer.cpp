#include <glog/logging.h>
#include <google/protobuf/message.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <vector>
#include <cstdlib>

#include <xir/tensor/tensor.hpp>
#include "vart/dpu/vitis_dpu_runner_factory.hpp"
#include "vart/runner_ext.hpp"
#include "xir/xrt_device_handle.hpp"

using namespace std;

int main(int argc, char* argv[]) {
  if (argc < 6) {
    std::cerr << "Usage: " << argv[0]
              << " <xmodel_path> <kernel_name> <input_file> <runner_num>\n";
    return 1;
  }

  const string xmodel = argv[1];
  const string kernel = argv[2];
  const string input_file = argv[3];
  const int runner_num = std::stoi(argv[4]);
  // const int batch_count = std::stoi(argv[5]);

  const string xclbin1 = "/usr/lib/rp.xclbin";
  setenv("XLNX_VART_FIRMWARE", xclbin1.c_str(), 1);  // 固定设定 firmware

  vector<std::unique_ptr<vart::Runner>> runners;
  for (int rr = 0; rr < runner_num; rr++) {
    runners.emplace_back(
        vart::dpu::DpuRunnerFactory::create_dpu_runner(xmodel, kernel));
  }

  for (int rr = 0; rr < runner_num; rr++) {
    auto& runner = runners[rr];
    auto r = dynamic_cast<vart::RunnerExt*>(runner.get());
    auto input = r->get_inputs();
    auto output = r->get_outputs();

    CHECK_EQ(input.size(), 1u) << "Only support single input.";
    CHECK_EQ(output.size(), 1u) << "Only support single output.";

    size_t batch_size = input[0]->get_tensor()->get_shape()[0];
    size_t size_per_batch = input[0]->get_tensor()->get_data_size() / batch_size;
    uint64_t input_data = 0u, input_size = 0u;

    // 加载输入数据（只使用一个文件，batch 内部复制同一份数据）
    for (size_t i = 0; i < batch_size; ++i) {
      std::tie(input_data, input_size) = input[0]->data({(int)i, 0, 0, 0});
      CHECK(std::ifstream(input_file).read((char*)input_data, size_per_batch).good())
          << "Failed to read input data from " << input_file;
    }

    // 同步写
    for (auto in : input) {
      in->sync_for_write(0, in->get_tensor()->get_data_size() / batch_size);
    }

    // 执行一次推理
    auto job_id = runner->execute_async(input, output);
    runner->wait(job_id.first, -1);

    // 同步读
    for (auto out : output) {
      out->sync_for_read(0, out->get_tensor()->get_data_size() / batch_size);
    }

    // 保存输出为 int8 bin 文件

    uint64_t output_data = 0u, output_size = 0u;
    std::tie(output_data, output_size) = output[0]->data({0, 0, 0, 0});
    auto data_phy = input[0]->data_phy({0, 0, 0, 0});
  
  LOG(INFO) << "Input tesor buffer phy address = " << std::hex << "0x" << data_phy.first << std::dec
            << " data_size=" << data_phy.second;


    std::ofstream ofs("output.bin", std::ios::binary);
    ofs.write(reinterpret_cast<char*>(output_data), output_size);
    ofs.close();
    std::cout << "Output saved to output.bin (" << output_size << " bytes)." << std::endl;

    // 打印前几个输出值以供验证
    auto out_ptr = reinterpret_cast<int8_t*>(output_data);
    std::cout << "First 10 output values (int8): ";
    for (size_t i = 0; i < std::min<size_t>(10, output_size); ++i) {
      std::cout << static_cast<int>(out_ptr[i]) << " ";
    }
    std::cout << std::endl;
  }

  return 0;
}
