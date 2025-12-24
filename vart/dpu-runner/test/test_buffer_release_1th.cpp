#include <glog/logging.h>

#include <xir/graph/graph.hpp>
#include <xir/attrs/attrs.hpp>
#include "vart/runner_ext.hpp"

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
#include <unistd.h>

// ===== 你原本的工具函數：find_dpu_subgraph / total_outputs_bytes / read_file_exact
// / tb_linear_ptr / compare_outputs_with_golden_concat / fill_input_from_file
// 這些都可以原封不動保留 =====
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

static std::pair<uint64_t, uint64_t> tb_linear_ptr(vart::TensorBuffer* tb) {
  return tb->data({0, 0, 0, 0});
}

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
// -------------------- Task: worker 只執行，不創建/銷毀 runner --------------------
struct Task {
  vart::Runner* runner = nullptr;  // 非 owning，生命週期由主線程管理
  std::vector<vart::TensorBuffer*> inputs;
  std::vector<vart::TensorBuffer*> outputs;

  std::string input_file;
  int execute_count = 0;

  // golden 已經在主線程讀好，worker 只拿來比對
  std::shared_ptr<std::vector<uint8_t>> golden;
};

// -------------------- Worker: 不再持有 runner/graph/attrs --------------------
class Worker {
 public:
  explicit Worker(std::string name) : name_(std::move(name)) {
    th_ = std::thread([this] { this->loop(); });
  }

  ~Worker() {
    stop();
    if (th_.joinable()) th_.join();
  }

  void submit(Task t) {
    std::unique_lock<std::mutex> lk(mu_);
    cv_idle_.wait(lk, [&]{ return state_ == State::IDLE; });
    task_ = std::move(t);
    state_ = State::HAS_TASK;
    cv_.notify_one();
  }

  void wait_done() {
    std::unique_lock<std::mutex> lk(mu_);
    cv_done_.wait(lk, [&]{ return state_ == State::IDLE; });
    if (!last_error_.empty()) {
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
  enum class State { IDLE, HAS_TASK, RUNNING };

  void loop() {
    for (;;) {
      Task t;
      {
        std::unique_lock<std::mutex> lk(mu_);
        state_ = State::IDLE;
        last_error_.clear();
        cv_idle_.notify_all();
        cv_done_.notify_all();

        cv_.wait(lk, [&]{ return stop_ || state_ == State::HAS_TASK; });
        if (stop_) return;

        t = std::move(task_);
        state_ = State::RUNNING;
      }

      try {
        run_task(t);
      } catch (const std::exception& e) {
        std::lock_guard<std::mutex> lk(mu_);
        last_error_ = e.what();
      } catch (...) {
        std::lock_guard<std::mutex> lk(mu_);
        last_error_ = "unknown error";
      }
    }
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
          {
      std::cerr << *why << "\nPress ENTER to continue..." << std::flush;
      std::cin.clear();
      std::string line;
      std::getline(std::cin, line);
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
  void run_task(const Task& t) {
    CHECK(t.runner != nullptr);
    CHECK(!t.inputs.empty());
    CHECK(t.golden && !t.golden->empty());

    LOG(INFO) << "[" << name_ << "] start task, count=" << t.execute_count;

    // input 填充也可以放 worker，但你說希望 runner 銷毀在主線程，
    // 這裡是否放 worker 不影響；我保留在這裡，保持你原本邏輯。
    fill_input_from_file(t.inputs, t.input_file);

    for (int c = 0; c < t.execute_count; ++c) {
      for (auto* in : t.inputs) {
        in->sync_for_write(0, in->get_tensor()->get_data_size());
      }

      // 建議你接 wait（依你 VART 版本調整）
      // auto jid = t.runner->execute_async(t.inputs, t.outputs);
      // t.runner->wait(jid.first /*or jid*/, -1);
      t.runner->execute_async(t.inputs, t.outputs);

      for (auto* out : t.outputs) {
        out->sync_for_read(0, out->get_tensor()->get_data_size());
      }

      std::string why;
      bool ok = compare_outputs_with_golden_concat(t.outputs, *t.golden, &why);
      if (!ok) {
        LOG(ERROR) << "[" << name_ << "] GOLDEN MISMATCH iter=" << c << " : " << why;
        throw std::runtime_error("golden mismatch");
      }

      if ((c + 1) % 5 == 0) {
        LOG(INFO) << "[" << name_ << "] finished " << (c + 1) << "/" << t.execute_count;
      }
    }

    LOG(INFO) << "[" << name_ << "] task done.";
  }

 private:
  std::string name_;
  std::thread th_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::condition_variable cv_idle_;
  std::condition_variable cv_done_;

  State state_{State::IDLE};
  Task task_;
  bool stop_{false};
  std::string last_error_;
};

static void set_xclbin_firmware(const std::string& xclbin_path) {
  setenv("XLNX_VART_FIRMWARE", xclbin_path.c_str(), 1);
  LOG(INFO) << "Set XLNX_VART_FIRMWARE=" << xclbin_path;
}

// -------------------- 主線程 phase：create/destroy runner 都在這裡 --------------------
static void run_phase_one_model_on_one_worker(
    Worker& w,
    const std::string& xclbin_path,
    const std::string& xmodel_path,
    const std::string& input_file,
    const std::string& golden_file,
    int execute_count) {

  // 1) 先切 xclbin（仍在主線程）
  set_xclbin_firmware(xclbin_path);

  // 2) 主線程建立 graph/runner
  auto graph = xir::Graph::deserialize(xmodel_path);
  auto* dpu_subgraph = find_dpu_subgraph(graph.get());
  CHECK(dpu_subgraph != nullptr) << "No DPU subgraph found in model: " << xmodel_path;

  auto attrs = xir::Attrs::create();
  std::unique_ptr<vart::Runner> runner = vart::RunnerExt::create_runner(dpu_subgraph, attrs.get());
  auto* r = dynamic_cast<vart::RunnerExt*>(runner.get());
  CHECK(r != nullptr) << "Runner is not RunnerExt";

  auto inputs = r->get_inputs();
  auto outputs = r->get_outputs();
  CHECK_EQ(inputs.size(), 1u);

  // 3) 主線程把 golden 讀好（避免 worker 每輪讀檔）
  const size_t golden_bytes = total_outputs_bytes(outputs);
  auto golden_blob = std::make_shared<std::vector<uint8_t>>(
      read_file_exact(golden_file, golden_bytes));

  // 4) 丟 task 給 worker 執行（runner 指標由主線程持有）
  Task t;
  t.runner = runner.get();
  t.inputs = inputs;
  t.outputs = outputs;
  t.input_file = input_file;
  t.execute_count = execute_count;
  t.golden = golden_blob;

  w.submit(std::move(t));
  w.wait_done();

  // 5) 你要的關鍵：在「切換 xclbin 的中間」由主線程明確銷毀 runner
  LOG(INFO) << "[main] destroying runner/graph/attrs in main thread before switching xclbin...";
  runner.reset();
  attrs.reset();
  graph.reset();

  // 這裡可以加一點延遲/同步（視你平台需要）
  // sleep(1);

  LOG(INFO) << "[main] phase done for xclbin=" << xclbin_path;
}

int main(int argc, char* argv[]) {
  const std::string defog_input_file = "defog_input.bin";
  const std::string xmodel_defog = "DeFog.xmodel";
  const std::string xclbin_dpu3d = "dpu3d.xclbin";
  const std::string xclbin_dpu4k = "dpu4k.xclbin";

  Worker w0("worker-0(DeFog)");

  for (int i = 0; i < 10000; ++i) {
    LOG(INFO) << "================ ITERATION=" << i << " ================";
    try {
      run_phase_one_model_on_one_worker(
          w0, xclbin_dpu3d, xmodel_defog, defog_input_file, "defog_golden.bin", 1);

      run_phase_one_model_on_one_worker(
          w0, xclbin_dpu4k, xmodel_defog, defog_input_file, "defog_golden.bin", 1);

    } catch (const std::exception& e) {
      LOG(ERROR) << "Fatal: " << e.what();
    }
  }

  return 0;
} 