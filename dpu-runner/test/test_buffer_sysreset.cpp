#include <glog/logging.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <vector>
#include <thread>
#include <chrono>
#include <filesystem>
#include <cstdlib>
#include <unistd.h>
#include <stdexcept>

#include <xir/tensor/tensor.hpp>
#include "vart/dpu/vitis_dpu_runner_factory.hpp"
#include "vart/runner_ext.hpp"
#include "xir/xrt_device_handle.hpp"

namespace fs = std::filesystem;

// 强制 reload xclbin
bool load_xclbin(int device_idx, const std::string& xclbin_path) {
  std::ifstream stream(xclbin_path, std::ios::binary | std::ios::ate);
  if (!stream) {
    std::cerr << "Failed to open xclbin: " << xclbin_path << "\n";
    return false;
  }
  auto size = stream.tellg();
  stream.seekg(0, std::ios::beg);
  std::vector<char> data(size);
  if (!stream.read(data.data(), size)) {
    std::cerr << "Failed to read xclbin data.\n";
    return false;
  }

  try {
    auto device = xrt::device(device_idx);
    auto uuid = device.load_xclbin(data.data());
    std::cout << "[INFO] Reloaded xclbin with UUID: " << uuid.to_string() << "\n";
    return true;
  } catch (const std::exception& e) {
    std::cerr << "load_xclbin failed: " << e.what() << "\n";
    return false;
  }
}

// 更强的 CU context 检测逻辑
void force_clear_kds_context() {
  for (const auto& entry : fs::directory_iterator("/sys/class/drm")) {
    if (!entry.is_directory()) continue;
    if (entry.path().filename().string().rfind("renderD", 0) != 0) continue;

    auto kds_stat = entry.path() / "device" / "kds_stat";
    if (!fs::exists(kds_stat)) continue;

    std::ifstream ifs(kds_stat);
    std::string line;
    while (std::getline(ifs, line)) {
      if (line.find("usage(1)") != std::string::npos &&
          line.find("refcnt(0)") != std::string::npos &&
          line.find("intr(disable)") != std::string::npos) {
        std::cout << "[WARN] Hanging CU context found in: " << kds_stat << "\n";
        system("echo 1 > /sys/class/drm/renderD128/device/zocl_reset");
        std::this_thread::sleep_for(std::chrono::seconds(1));
      }
    }
  }
}

// 重置后 reload xclbin
bool safe_zocl_reset_with_reload(const std::string& xclbin_path) {
  std::cout << "[INFO] Triggering zocl_reset...\n";
  system("echo 1 > /sys/class/drm/renderD128/device/zocl_reset");
  std::this_thread::sleep_for(std::chrono::seconds(1));
  return load_xclbin(0, xclbin_path);
}

void run_inference(const std::string& xmodel, const std::string& kernel,
                   const std::string& input_file, int runner_num, int count,
                   const std::string& xclbin_path) {
  // CHECK(load_xclbin(0, xclbin_path)) << "Failed to load xclbin.";

  // 设置超时重置策略
  setenv("XLNX_DPU_TIMEOUT", "10000", 1); // 10s
  setenv("XLNX_DPU_TIMEOUT_RESET", "soft", 1);

  // 让 VART 知道我们要使用哪个 xclbin
  setenv("XLNX_VART_FIRMWARE", xclbin_path.c_str(), 1);

  std::vector<std::unique_ptr<vart::Runner>> runners;
  for (int i = 0; i < runner_num; ++i) {
    runners.emplace_back(vart::dpu::DpuRunnerFactory::create_dpu_runner(xmodel, kernel));
    sleep(3);
  }

  for (int i = 0; i < runner_num; ++i) {
    auto r = dynamic_cast<vart::RunnerExt*>(runners[i].get());
    auto input = r->get_inputs();
    auto output = r->get_outputs();

    size_t batch = input[0]->get_tensor()->get_shape()[0];
    size_t size_per_batch = input[0]->get_tensor()->get_data_size() / batch;
    for (size_t b = 0; b < batch; ++b) {
      uint64_t input_data = 0, input_size = 0;
      std::tie(input_data, input_size) = input[0]->data({(int)b, 0, 0, 0});
      CHECK(std::ifstream(input_file).read((char*)input_data, size_per_batch).good())
          << "Failed to read input";
    }

    for (int c = 0; c < count; ++c) {
      for (auto& in : input)
        in->sync_for_write(0, in->get_tensor()->get_data_size() / batch);

      auto job_id = runners[i]->execute_async(input, output);
      int status = runners[i]->wait(job_id.first, -1);
      if (status != 0) {
        LOG(ERROR) << "Timeout detected. Cleaning up...";
        runners.clear();
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        safe_zocl_reset_with_reload(xclbin_path);
        force_clear_kds_context();
        throw std::runtime_error("DPU execution timeout.");
      }

      for (auto& out : output)
        out->sync_for_read(0, out->get_tensor()->get_data_size() / batch);
    }
  }
  runners.clear();
}

int main(int argc, char* argv[]) {
  if (argc < 6) {
    std::cerr << "Usage: " << argv[0]


              << " <xmodel> <kernel_name> <input_file> <runner_num> <count>\n";
    return 1;
  }

  std::string xmodel = argv[1];
  std::string kernel = argv[2];
  std::string input_file = argv[3];
  int runner_num = std::stoi(argv[4]);
  int count = std::stoi(argv[5]);

  std::string xclbin1 = "/usr/lib/rp.xclbin";
  std::string xclbin2 = "/usr/lib/rm.xclbin";

  for (int i = 0; i < 1000; ++i) {
    std::string xclbin = (i % 2 == 0) ? xclbin1 : xclbin2;
    LOG(INFO) << "\n=== Iteration " << i + 1 << ": " << xclbin << " ===";
    try {
      run_inference(xmodel, kernel, input_file, runner_num, count, xclbin);
    } catch (const std::exception& e) {
      LOG(ERROR) << "Exception: " << e.what();
      force_clear_kds_context();
    }
  }

  return 0;
}
