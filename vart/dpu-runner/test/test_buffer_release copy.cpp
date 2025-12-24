#include <glog/logging.h>

#include <xir/graph/graph.hpp>
#include <xir/attrs/attrs.hpp>
#include "vart/runner_ext.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

struct Job {
  std::string xclbin_path;
  std::string xmodel_path;
  std::string input_file;
  int execute_count = 200;
    // 每个线程跑多少次 execute_async
  // 每个模型一个 golden（int8 raw），覆盖所有输出（按 outputs 顺序拼接）
  std::string golden_file;
};

static xir::Subgraph* find_dpu_subgraph(xir::Graph* graph) {
  auto root = graph->get_root_subgraph();
  for (auto c : root->children_topological_sort()) {
    if (c->has_attr("device") && c->get_attr<std::string>("device") == "DPU") {
      return c;
    }
  }
  return nullptr;
}


static size_t total_outputs_bytes(const std::vector<vart::TensorBuffer*>& outputs) {
  size_t total = 0;
  for (auto* o : outputs) total += o->get_tensor()->get_data_size();
  return total;
}

static std::vector<uint8_t> read_file_exact(const std::string& path, size_t expected_size) {
  std::ifstream ifs(path, std::ios::binary);
  CHECK(ifs.good()) << "Failed to open golden file: " << path;

  std::vector<uint8_t> buf(expected_size);
  ifs.read(reinterpret_cast<char*>(buf.data()), expected_size);
  CHECK(ifs.gcount() == static_cast<std::streamsize>(expected_size))
      << "Golden size mismatch, file=" << path
      << " expected=" << expected_size
      << " got=" << ifs.gcount();
  return buf;
}



static std::pair<uint64_t,uint64_t> tb_linear_ptr(vart::TensorBuffer* tb) {
  // 常见 DPU 输出可以用 index {0,0,0,0} 取线性起始地址
  return tb->data({0,0,0,0});
}

static bool compare_outputs_with_golden_concat(
    const std::vector<vart::TensorBuffer*>& outputs,
    const std::vector<uint8_t>& golden,
    std::string* why) {

  size_t expected_total = total_outputs_bytes(outputs);
  CHECK_EQ(golden.size(), expected_total);

  size_t golden_off = 0;
  for (size_t oi = 0; oi < outputs.size(); ++oi) {
    auto* out_tb = outputs[oi];
    const size_t bytes = out_tb->get_tensor()->get_data_size();

    uint64_t ptr_u64 = 0, sz = 0;
    std::tie(ptr_u64, sz) = tb_linear_ptr(out_tb);
    CHECK(sz >= bytes);

    const uint8_t* out = reinterpret_cast<const uint8_t*>(ptr_u64);
    const uint8_t* ref = golden.data() + golden_off;

    for (size_t k = 0; k < bytes; ++k) {
      if (out[k] != ref[k]) {
        if (why) {
          *why = "mismatch: output[" + std::to_string(oi) + "] byte=" + std::to_string(k) +
                 " out=" + std::to_string(out[k]) +
                 " ref=" + std::to_string(ref[k]) +
                 " golden_off=" + std::to_string(golden_off);
        }
        return false;
      }
    }
    golden_off += bytes;
  }
  return true;
}


static void fill_input_from_file(const std::vector<vart::TensorBuffer*>& inputs,
                                 const std::string& input_file) {
  CHECK(!inputs.empty());
  auto tb = inputs[0];
  auto tensor = tb->get_tensor();
  auto shape = tensor->get_shape();
  CHECK(!shape.empty());
  const size_t batch = inputs[0]->get_tensor()->get_shape()[0];
  // const size_t total_size = tensor->get_data_size();
  CHECK(batch > 0);
  // const size_t size_per_batch = total_size / batch;
  size_t size_per_batch = inputs[0]->get_tensor()->get_data_size() / batch;

  for (size_t b = 0; b < batch; ++b) {
    uint64_t input_data = 0;
    uint64_t input_size = 0;
    std::tie(input_data, input_size) = tb->data({(int)b, 0, 0, 0});
    CHECK(input_size >= size_per_batch);

    std::ifstream ifs(input_file, std::ios::binary);
    CHECK(ifs.good()) << "Failed to open input file: " << input_file;
    CHECK(ifs.read(reinterpret_cast<char*>(input_data), size_per_batch).good())
        << "Failed to read input data, file=" << input_file;
  }
}

class Worker {
 public:
  explicit Worker(std::string name) : name_(std::move(name)) {
    th_ = std::thread([this] { this->loop(); });
  }

  ~Worker() {
    stop();
    if (th_.joinable()) th_.join();
  }

  // 提交一个任务（阻塞等待 worker 接收）
  void submit(const Job& job) {
    std::unique_lock<std::mutex> lk(mu_);
    cv_idle_.wait(lk, [&] { return state_ == State::IDLE; });
    job_ = job;
    state_ = State::HAS_JOB;
    cv_.notify_one();
  }

  // 等待任务完成（阻塞）
  void wait_done() {
    std::unique_lock<std::mutex> lk(mu_);
    cv_done_.wait(lk, [&] { return state_ == State::IDLE; });
    if (!last_error_.empty()) {
      // 你也可以选择不抛异常，只打印
      throw std::runtime_error(name_ + " failed: " + last_error_);
    }
  }

  void stop() {
    std::unique_lock<std::mutex> lk(mu_);
    if (stop_) return;
    stop_ = true;
    cv_.notify_one();
  }

 private:
  enum class State { IDLE, HAS_JOB, RUNNING };

  void loop() {
    for (;;) {
      Job job;
      {
        std::unique_lock<std::mutex> lk(mu_);
        state_ = State::IDLE;
        last_error_.clear();
        cv_idle_.notify_all();
        cv_done_.notify_all();

        cv_.wait(lk, [&] { return stop_ || state_ == State::HAS_JOB; });
        if (stop_) return;

        job = job_;
        state_ = State::RUNNING;
      }

      try {
        run_once(job);
      } catch (const std::exception& e) {
        std::lock_guard<std::mutex> lk(mu_);
        last_error_ = e.what();
      } catch (...) {
        std::lock_guard<std::mutex> lk(mu_);
        last_error_ = "unknown error";
      }

      // 任务结束后，资源在 run_once 结束时已释放；线程回到 IDLE 等下一次提交
    }
  }

  void run_once(const Job& job) {

    
    // 必须由主线程在提交任务前 setenv，并保证上一阶段 runner 都已释放。
    LOG(INFO) << "[" << name_ << "] start job"
              << " xclbin=" << job.xclbin_path
              << " xmodel=" << job.xmodel_path
              << " count=" << job.execute_count;

    // 1) 反序列化 graph
    // LOG(INFO) << "[start to install zocl module";
    // system("modprobe -i zocl");
    auto graph = xir::Graph::deserialize(job.xmodel_path);
    auto* dpu_subgraph = find_dpu_subgraph(graph.get());
    CHECK(dpu_subgraph != nullptr) << "No DPU subgraph found in model: " << job.xmodel_path;
   
    // 2) 创建 runner（每次 job 都新建，满足“清除 runner”）
    sleep(1);
    auto attrs = xir::Attrs::create();
    std::unique_ptr<vart::Runner> runner =
        vart::RunnerExt::create_runner(dpu_subgraph, attrs.get());
    auto* r = dynamic_cast<vart::RunnerExt*>(runner.get());
    CHECK(r != nullptr) << "Runner is not RunnerExt";

    // 3) 获取 input/output TensorBuffer（由 runner 管理/返回；job 结束即“清除”）
    auto inputs = r->get_inputs();
    auto outputs = r->get_outputs();
    CHECK_EQ(inputs.size(), 1u) << "Only support single input (as your original code)";

    // 4) 准备输入
    fill_input_from_file(inputs, job.input_file);

    // 5) 跑 200 次 execute_async（你要求的次数）
    //    注意：execute_async 返回 job id；wait 必须等对应 job id
    for (int c = 0; c < job.execute_count; ++c) {
      // sync input -> device
      for (auto* in : inputs) {
        in->sync_for_write(0, in->get_tensor()->get_data_size());
      }

      // int jobid = -1;
      try {
         runner->execute_async(inputs, outputs);
      } catch (const std::runtime_error& e) {
        LOG(ERROR) << "[" << name_ << "] execute_async runtime_error: " << e.what();
        throw;  // 你也可以选择 continue，但这里按“失败就让主线程知道”
      }
          const size_t golden_bytes = total_outputs_bytes(outputs);
          auto golden_blob = read_file_exact(job.golden_file, golden_bytes);

      // 等待完成（-1 表示阻塞等待，具体语义看你 VART 版本；你也可设超时）
      // runner->wait(ret.first(), -1);

      // sync output -> host
      for (auto* out : outputs) {
        out->sync_for_read(0, out->get_tensor()->get_data_size());
      } 

      std::string why;
      bool ok = compare_outputs_with_golden_concat(outputs, golden_blob, &why);
      if (!ok) {
        LOG(ERROR) << "[" << name_ << "] GOLDEN MISMATCH at iter=" << c << " : " << why;
        throw std::runtime_error("golden mismatch");
      }

      if ((c + 1) % 5 == 0) {
        LOG(INFO) << "[" << name_ << "] finished " << (c + 1) << "/" << job.execute_count;
      }
    }

    


//     std::this_thread::sleep_for(std::chrono::milliseconds(500));    // 注意：XLNX_VART_FIRMWARE 是进程级环境变量，

//     runner.reset();
  
//     // system("modprobe -r zocl");

//  std::this_thread::sleep_for(std::chrono::milliseconds(500));
    // 6) 结束即释放：runner/graph/attrs/inputs/outputs 都会在作用域结束自动清理
    // （inputs/outputs 是 vector< TensorBuffer* >，不需要 delete；runner 释放时其内部资源也释放）
    LOG(INFO) << "[" << name_ << "] job done, resources released.";
  }

 private:
  std::string name_;
  std::thread th_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::condition_variable cv_idle_;
  std::condition_variable cv_done_;

  State state_{State::IDLE};
  Job job_;
  bool stop_{false};
  std::string last_error_;
};

static void set_xclbin_firmware(const std::string& xclbin_path) {
  // 关键点：必须在创建 runner 之前设置
  setenv("XLNX_VART_FIRMWARE", xclbin_path.c_str(), 1);
  LOG(INFO) << "Set XLNX_VART_FIRMWARE=" << xclbin_path;
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv<< " <input_file>\n";
    return 1;
  }
  const std::string hls_input_file = "hls_input.bin";
   const std::string defog_input_file = "defog_input.bin";

  // 你要求的两个 xmodel
  const std::string xmodel_defog = "DeFog.xmodel";
  const std::string xmodel_hls   = "HLS.xmodel";

  // 你要求的两个 xclbin
  const std::string xclbin_dpu3d = "dpu3d.xclbin";
  const std::string xclbin_dpu4k = "dpu4k.xclbin";

  // 两个常驻线程（不退出）
  Worker w0("worker-0(DeFog)");
  Worker w1("worker-1(HLS)");

  auto run_phase = [&](const std::string& xclbin_path) {
    // 1) 切 xclbin（必须保证上一阶段 runner 已经释放；我们是 phase-by-phase 串行，所以满足）
    set_xclbin_firmware(xclbin_path);

    // 2) 提交两个 job：两个线程各一个 runner，各跑 200 次
    Job j0;
    j0.xclbin_path = xclbin_path;
    j0.xmodel_path = xmodel_defog;
    j0.input_file = defog_input_file;
    j0.execute_count = 10;
    j0.golden_file = "defog_golden.bin";

    Job j1;
    j1.xclbin_path = xclbin_path;
    j1.xmodel_path = xmodel_hls;
    j1.input_file = hls_input_file;
    j1.execute_count = 10;
    j1.golden_file = "hls_golden.bin";

    w0.submit(j0);
    w1.submit(j1);

    // 3) 等待两边都完成（确保 runner/buffer 都释放完，才允许切换到下一个 xclbin）
    w0.wait_done();
    w1.wait_done();
    // LOG(INFO) << "[Works completed. now start to remove zocl module";


    LOG(INFO) << "Phase done for xclbin=" << xclbin_path;
  };


for(int i =0 ; i<10000; i++){
  LOG(INFO)<<"==============================ITERATION="<<i<< "times"<<"=======================";
  try {
    // 阶段 1：dpu3d.xclbin
    run_phase(xclbin_dpu3d);
    // sleep(2);
    // 阶段 2：dpu4k.xclbin
    run_phase(xclbin_dpu4k);
    // sleep(2);

  } catch (const std::exception& e) {
    LOG(ERROR) << "Fatal: " << e.what();
  }
}
  // 线程不关闭的意思通常是“流程后面还有 phase/循环”
  // 这里示例 main 结束会析构 worker 并 join。
  // 如果你要常驻服务，把这里改成循环 run_phase(...) 即可。

  return 0;
}