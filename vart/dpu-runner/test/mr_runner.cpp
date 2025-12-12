// File: dpu_mr.cpp

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <xir/graph/graph.hpp>
//#include <xir/tensor/tensor.hpp>
#include <vart/runner.hpp>
#include <vart/runner_ext.hpp>
#include <vart/tensor_buffer.hpp>
#include <vart/runner_helper.hpp>
#include <vart/assistant/tensor_buffer_allocator.hpp>
#include <vart/assistant/xrt_bo_tensor_buffer.hpp>
#include <xrt/xrt/xrt_bo.h>
#include <xrt/xrt/xrt_device.h>
#include <cstring>
#include <memory>
#include <vector>
#include <iostream>

namespace py = pybind11;
constexpr size_t TAIL_SIZE = 8192;
constexpr int NUM_BUFFERS = 4; // 输出buffer数量（4帧循环）
//xrt全局变量
static std::shared_ptr<xrt::device> m_device;

// 去雾算法
static std::shared_ptr<xir::Graph> DeHaze_graph = nullptr;
static std::unique_ptr<vart::RunnerExt> DeHaze_runner = nullptr;
static const xir::Tensor* DeHaze_input_tensor = nullptr;
//static const xir::Tensor* DeHaze_input_tensor1 = nullptr;
static const xir::Tensor* DeHaze_output_tensor = nullptr;
static size_t DeHaze_input_size = 0;
static size_t DeHaze_output_size = 0;

static std::vector<std::unique_ptr<xrt::bo>> DeHaze_input_bo_list;
static std::vector<std::unique_ptr<vart::TensorBuffer>> DeHaze_input_tb_list;
static std::vector<std::unique_ptr<xrt::bo>> DeHaze_output_bo_list;
static std::vector<std::unique_ptr<vart::TensorBuffer>> DeHaze_output_tb_list;
static int DeHaze_cur_buf_idx = 0;

//去反光算法
static std::shared_ptr<xir::Graph> DeReflect_graph = nullptr;
static std::unique_ptr<vart::RunnerExt> DeReflect_runner = nullptr;
static const xir::Tensor* DeReflect_input_tensor = nullptr;
static const xir::Tensor* DeReflect_output_tensor = nullptr;
static size_t DeReflect_input_size = 0;
static size_t DeReflect_output_size = 0;

static std::vector<std::unique_ptr<xrt::bo>> DeReflect_input_bo_list;
static std::vector<std::unique_ptr<vart::TensorBuffer>> DeReflect_input_tb_list;
static std::vector<std::unique_ptr<xrt::bo>> DeReflect_output_bo_list;
static std::vector<std::unique_ptr<vart::TensorBuffer>> DeReflect_output_tb_list;
static int DeReflect_cur_buf_idx = 0;

//插帧算法
static std::shared_ptr<xir::Graph> InterFrame_graph = nullptr;
static std::unique_ptr<vart::RunnerExt> InterFrame_runner = nullptr;
static const xir::Tensor* InterFrame_input_tensor0 = nullptr;
static const xir::Tensor* InterFrame_input_tensor1 = nullptr;
static const xir::Tensor* InterFrame_output_tensor = nullptr;
static size_t InterFrame_input_size0 = 0;
static size_t InterFrame_input_size1 = 0;
static size_t InterFrame_output_size = 0;

static std::vector<std::unique_ptr<xrt::bo>> InterFrame_input_bo_list;
static std::vector<std::unique_ptr<vart::TensorBuffer>> InterFrame_input_tb_list;
static std::vector<std::unique_ptr<xrt::bo>> InterFrame_output_bo_list;
static std::vector<std::unique_ptr<vart::TensorBuffer>> InterFrame_output_tb_list;
static int InterFrame_cur_buf_idx = 0;


// 调试日志开关
static bool debug_enabled = true;

void dpu_set_debug(bool enable) {
    debug_enabled = enable;
}

void dpu_log(const std::string& msg) {
    if (debug_enabled) {
        std::cerr << "[DPU_DBG] " << msg << std::endl;
    }
}

//XRT初始化
void Init_xrt_device(void){
    auto xrt_device = std::make_shared<xrt::device>(0);
    m_device = std::move(xrt_device);
}

// 创建 DPU Runner
void Dehaze_dpu_create(const std::string& xmodel_path, int core_id) {
    dpu_log("Creating Dehaze runner with model: " + xmodel_path + ", core_id: " + std::to_string(core_id));
    DeHaze_graph = xir::Graph::deserialize(xmodel_path);
    auto root = DeHaze_graph->get_root_subgraph();
    const xir::Subgraph* dpu_sg = nullptr;
    for (auto sg : root->children_topological_sort()) {
        if (sg->has_attr("device") && sg->get_attr<std::string>("device") == "DPU") {
            dpu_sg = sg;
            break;
        }
    }
    if (!dpu_sg) throw std::runtime_error("No DPU subgraph found");

    auto attrs = xir::Attrs::create();
    attrs->set_attr<size_t>("__device_core_id__", core_id);
    attrs->set_attr<size_t>("__device_id__", 0);
    DeHaze_runner = vart::RunnerExt::create_runner(dpu_sg, attrs.get());

    auto in_tensors = DeHaze_runner->get_input_tensors();
    auto out_tensors = DeHaze_runner->get_output_tensors();
    DeHaze_input_tensor = in_tensors[0];
    DeHaze_input_size = DeHaze_input_tensor->get_data_size();
    DeHaze_output_tensor = out_tensors[0];
    DeHaze_output_size = DeHaze_output_tensor->get_data_size();

    // 创建4帧输出buffer
    DeHaze_input_bo_list.clear();
    DeHaze_input_tb_list.clear();
    DeHaze_output_bo_list.clear();
    DeHaze_output_tb_list.clear();
    for (int i = 0; i < NUM_BUFFERS; ++i) {
        //输入
        auto input_bo = std::make_unique<xrt::bo>(*m_device, DeHaze_input_size, 1);
        auto input_tb = vart::assistant::XrtBoTensorBuffer::create(
		vart::xrt_bo_t{ m_device.get(), input_bo.get() }, DeHaze_input_tensor);
        //输出
        auto output_bo = std::make_unique<xrt::bo>(*m_device, DeHaze_output_size + TAIL_SIZE, 1); // 注意分配空间要包含tail
        auto output_tb = vart::assistant::XrtBoTensorBuffer::create(
		vart::xrt_bo_t{ m_device.get(), output_bo.get() }, DeHaze_output_tensor);

        DeHaze_input_bo_list.emplace_back(std::move(input_bo));
        DeHaze_input_tb_list.emplace_back(std::move(input_tb));
        DeHaze_output_bo_list.emplace_back(std::move(output_bo));
        DeHaze_output_tb_list.emplace_back(std::move(output_tb));
    }
    DeHaze_cur_buf_idx = 0;

    dpu_log("Dehaze Runner created. Input size0: " + std::to_string(DeHaze_input_size)
            + ", Output size: " + std::to_string(DeHaze_output_size)
            + ", Output buffers: " + std::to_string(NUM_BUFFERS));
}

void DeReflect_dpu_create(const std::string& xmodel_path, int core_id) {
    dpu_log("Creating DeReflect runner with model: " + xmodel_path + ", core_id: " + std::to_string(core_id));
    DeReflect_graph = xir::Graph::deserialize(xmodel_path);
    auto root = DeReflect_graph->get_root_subgraph();
    const xir::Subgraph* dpu_sg = nullptr;
    for (auto sg : root->children_topological_sort()) {
        if (sg->has_attr("device") && sg->get_attr<std::string>("device") == "DPU") {
            dpu_sg = sg;
            break;
        }
    }
    if (!dpu_sg) throw std::runtime_error("No DPU subgraph found");

    auto attrs = xir::Attrs::create();
    attrs->set_attr<size_t>("__device_core_id__", core_id);
    attrs->set_attr<size_t>("__device_id__", 0);
    DeReflect_runner = vart::RunnerExt::create_runner(dpu_sg, attrs.get());

    auto in_tensors = DeReflect_runner->get_input_tensors();
    auto out_tensors = DeReflect_runner->get_output_tensors();
    DeReflect_input_tensor = in_tensors[0];
    DeReflect_input_size = DeReflect_input_tensor->get_data_size();
    DeReflect_output_tensor = out_tensors[0];
    DeReflect_output_size = DeReflect_output_tensor->get_data_size();

    // 创建4帧输出buffer
    DeReflect_input_bo_list.clear();
    DeReflect_input_tb_list.clear();
    DeReflect_output_bo_list.clear();
    DeReflect_output_tb_list.clear();
    for (int i = 0; i < NUM_BUFFERS; ++i) {
        //输入
        auto input_bo = std::make_unique<xrt::bo>(*m_device, DeReflect_input_size, 1);
        auto input_tb = vart::assistant::XrtBoTensorBuffer::create(
		vart::xrt_bo_t{ m_device.get(), input_bo.get() }, DeReflect_input_tensor);
        //输出
        auto output_bo = std::make_unique<xrt::bo>(*m_device, DeReflect_output_size + TAIL_SIZE, 1); // 注意分配空间要包含tail
        auto output_tb = vart::assistant::XrtBoTensorBuffer::create(
		vart::xrt_bo_t{ m_device.get(), output_bo.get() }, DeReflect_output_tensor);

        DeReflect_input_bo_list.emplace_back(std::move(input_bo));
        DeReflect_input_tb_list.emplace_back(std::move(input_tb));
        DeReflect_output_bo_list.emplace_back(std::move(output_bo));
        DeReflect_output_tb_list.emplace_back(std::move(output_tb));
    }
    DeReflect_cur_buf_idx = 0;

    dpu_log("DeReflect Runner created. Input size0: " + std::to_string(DeReflect_input_size)
            + ", Output size: " + std::to_string(DeReflect_output_size)
            + ", Output buffers: " + std::to_string(NUM_BUFFERS));
}

void InterFrame_dpu_create(const std::string& xmodel_path, int core_id) {
    dpu_log("Creating InterFrame runner with model: " + xmodel_path + ", core_id: " + std::to_string(core_id));
    InterFrame_graph = xir::Graph::deserialize(xmodel_path);
    auto root = InterFrame_graph->get_root_subgraph();
    const xir::Subgraph* dpu_sg = nullptr;
    for (auto sg : root->children_topological_sort()) {
        if (sg->has_attr("device") && sg->get_attr<std::string>("device") == "DPU") {
            dpu_sg = sg;
            break;
        }
    }
    if (!dpu_sg) throw std::runtime_error("No DPU subgraph found");

    auto attrs = xir::Attrs::create();
    attrs->set_attr<size_t>("__device_core_id__", core_id);
    attrs->set_attr<size_t>("__device_id__", 0);
    InterFrame_runner = vart::RunnerExt::create_runner(dpu_sg, attrs.get());

    auto in_tensors = InterFrame_runner->get_input_tensors();
    auto out_tensors = InterFrame_runner->get_output_tensors();
    if (in_tensors.size() > 1) {
        dpu_log("InterFrame xmodel Run As Dual-Tensor!");
        InterFrame_input_tensor0 = in_tensors[0];
        InterFrame_input_tensor1 = in_tensors[1];
        InterFrame_input_size0 = in_tensors[0]->get_data_size();
        InterFrame_input_size1 = in_tensors[1]->get_data_size();
    }else{
        dpu_log("InterFrame xmodel Run As Single-Tensor!");
        InterFrame_input_tensor0 = in_tensors[0];
        InterFrame_input_tensor1 = nullptr;
        InterFrame_input_size0 = in_tensors[0]->get_data_size();
        InterFrame_input_size1 = 0;
    }
    InterFrame_output_tensor = out_tensors[0];
    InterFrame_output_size = InterFrame_output_tensor->get_data_size();

    // 创建4帧输出buffer
    InterFrame_input_bo_list.clear();
    InterFrame_input_tb_list.clear();
    InterFrame_output_bo_list.clear();
    InterFrame_output_tb_list.clear();
    for (int i = 0; i < NUM_BUFFERS; ++i) {
        //输入
        auto input_bo = std::make_unique<xrt::bo>(*m_device, InterFrame_input_size0, 1);
        auto input_tb = vart::assistant::XrtBoTensorBuffer::create(
		vart::xrt_bo_t{ m_device.get(), input_bo.get() }, InterFrame_input_tensor0);
        //输出
        auto output_bo = std::make_unique<xrt::bo>(*m_device, InterFrame_output_size + TAIL_SIZE, 1); // 注意分配空间要包含tail
        auto output_tb = vart::assistant::XrtBoTensorBuffer::create(
		vart::xrt_bo_t{ m_device.get(), output_bo.get() }, InterFrame_output_tensor);

        InterFrame_input_bo_list.emplace_back(std::move(input_bo));
        InterFrame_input_tb_list.emplace_back(std::move(input_tb));
        InterFrame_output_bo_list.emplace_back(std::move(output_bo));
        InterFrame_output_tb_list.emplace_back(std::move(output_tb));
    } 
    InterFrame_cur_buf_idx = 0;

    dpu_log("InterFrame Runner created. Input size0: " + std::to_string(InterFrame_input_size0)
            + ", Output size: " + std::to_string(InterFrame_output_size)
            + ", Output buffers: " + std::to_string(NUM_BUFFERS));
}

// 释放资源
void Dehaze_dpu_destroy() {
    DeHaze_input_bo_list.clear();
    DeHaze_input_tb_list.clear();
    DeHaze_output_tb_list.clear();
    DeHaze_output_bo_list.clear();
    DeHaze_runner.reset();
    DeHaze_graph.reset();
    DeHaze_input_tensor = nullptr;
    DeHaze_output_tensor = nullptr;
    DeHaze_input_size = DeHaze_output_size = 0;
    DeHaze_cur_buf_idx = 0;
}

void DeReflect_dpu_destroy() {
    DeReflect_input_bo_list.clear();
    DeReflect_input_tb_list.clear();
    DeReflect_output_tb_list.clear();
    DeReflect_output_bo_list.clear();
    DeReflect_runner.reset();
    DeReflect_graph.reset();
    DeReflect_input_tensor = nullptr;
    DeReflect_output_tensor = nullptr;
    DeReflect_input_size = DeReflect_output_size = 0;
    DeReflect_cur_buf_idx = 0;
}

void InterFrame_dpu_destroy() {
    InterFrame_input_bo_list.clear();
    InterFrame_input_tb_list.clear();
    InterFrame_output_tb_list.clear();
    InterFrame_output_bo_list.clear();
    InterFrame_runner.reset();
    InterFrame_graph.reset();
    InterFrame_input_tensor0 = nullptr;
    InterFrame_input_tensor1 = nullptr;
    InterFrame_output_tensor = nullptr;
    InterFrame_input_size0 = InterFrame_input_size1 = InterFrame_output_size = 0;
    InterFrame_cur_buf_idx = 0;
}

// 获取输入&输出 shape
std::vector<int> Dehaze_get_input_shape() {
    if (!DeHaze_input_tensor) throw std::runtime_error("Runner not initialized");
    const auto& shape = DeHaze_input_tensor->get_shape();
    return std::vector<int>(shape.begin(), shape.end());
}

std::vector<int> Dehaze_get_output_shape() {
    if (!DeHaze_output_tensor) throw std::runtime_error("Runner not initialized");
    const auto& shape = DeHaze_output_tensor->get_shape();
    return std::vector<int>(shape.begin(), shape.end());
}

std::vector<int> DeReflect_get_input_shape() {
    if (!DeReflect_input_tensor) throw std::runtime_error("Runner not initialized");
    const auto& shape = DeReflect_input_tensor->get_shape();
    return std::vector<int>(shape.begin(), shape.end());
}

std::vector<int> DeReflect_get_output_shape() {
    if (!DeReflect_output_tensor) throw std::runtime_error("Runner not initialized");
    const auto& shape = DeReflect_output_tensor->get_shape();
    return std::vector<int>(shape.begin(), shape.end());
}

std::vector<int> InterFrame_get_input_shape() {
    if (!InterFrame_input_tensor0) throw std::runtime_error("Runner not initialized");
    const auto& shape = InterFrame_input_tensor0->get_shape();
    return std::vector<int>(shape.begin(), shape.end());
}

std::vector<int> InterFrame_get_output_shape() {
    if (!InterFrame_output_tensor) throw std::runtime_error("Runner not initialized");
    const auto& shape = InterFrame_output_tensor->get_shape();
    return std::vector<int>(shape.begin(), shape.end());
}

// 单输入推理
int64_t Dehaze_run(py::array_t<int8_t> input_buf) {
    if (!DeHaze_runner || DeHaze_input_bo_list.empty() || DeHaze_output_tb_list.empty()) throw std::runtime_error("Runner not initialized");

    // 使用当前的output buffer
    int buf_idx = DeHaze_cur_buf_idx;
    auto& in_tb = DeHaze_input_tb_list[buf_idx];
    auto& in_bo = DeHaze_input_bo_list[buf_idx];
    auto& out_tb = DeHaze_output_tb_list[buf_idx];
    auto& out_bo = DeHaze_output_bo_list[buf_idx];
    auto input_virtaddr = in_bo->map();
    auto input_phyaddr  = in_tb->data_phy({0,0,0,0}).first;
    auto output_virtaddr = out_bo->map();
    auto output_phyaddr = out_tb->data_phy({0,0,0,0}).first;
    std::stringstream ss;
    ss << "inputPhyAddr= 0x" << std::hex << input_phyaddr
       << " OutputPhyAddr= 0x" << std::hex << output_phyaddr;
    dpu_log(ss.str());

    auto info = input_buf.request();
    if ((size_t)info.size < DeHaze_input_size + TAIL_SIZE)
        throw std::runtime_error("Input too small");

    void* input_ptr = info.ptr;
    std::memcpy(input_virtaddr, input_ptr, DeHaze_input_size);
    in_tb->sync_for_write(0, DeHaze_input_size);

    dpu_log("Dehaze Running inference, buffer idx: " + std::to_string(buf_idx));
    std::vector<vart::TensorBuffer*> inputs = {in_tb.get()};
    std::vector<vart::TensorBuffer*> outputs = {out_tb.get()};
    auto job_id = DeHaze_runner->execute_async(inputs, outputs);
    auto status = DeHaze_runner->wait(job_id.first, -1);
    if(status == 0){
        dpu_log("Run singele-input inference success!");
    }else{
        dpu_log("Run singele-input inference Failed!");
    }

    // tail拷贝到output buffer尾部
    out_tb->sync_for_read(0, DeHaze_output_size);
    std::memcpy(static_cast<uint8_t*>(output_virtaddr) + DeHaze_output_size,
                static_cast<uint8_t*>(input_ptr) + DeHaze_input_size,
                TAIL_SIZE);

    uint64_t phy_addr = output_phyaddr;

    dpu_log("Run complete, output buffer idx: " + std::to_string(buf_idx)
            + ", physical address: 0x" + std::to_string(phy_addr));

    DeHaze_cur_buf_idx = (DeHaze_cur_buf_idx + 1) % NUM_BUFFERS;
    return static_cast<int64_t>(phy_addr);
}

int64_t DeReflect_run(py::array_t<int8_t> input_buf) {
    if (!DeReflect_runner || DeReflect_input_bo_list.empty() || DeReflect_output_tb_list.empty()) throw std::runtime_error("Runner not initialized");

    // 使用当前的output buffer
    int buf_idx = DeReflect_cur_buf_idx;
    auto& in_tb = DeReflect_input_tb_list[buf_idx];
    auto& in_bo = DeReflect_input_bo_list[buf_idx];
    auto& out_tb = DeReflect_output_tb_list[buf_idx];
    auto& out_bo = DeReflect_output_bo_list[buf_idx];
    auto input_virtaddr = in_bo->map();
    auto input_phyaddr  = in_tb->data_phy({0,0,0,0}).first;
    auto output_virtaddr = out_bo->map();
    auto output_phyaddr = out_tb->data_phy({0,0,0,0}).first;
    std::stringstream ss;
    ss << "inputPhyAddr= 0x" << std::hex << input_phyaddr
       << " OutputPhyAddr= 0x" << std::hex << output_phyaddr;
    dpu_log(ss.str());

    auto info = input_buf.request();
    if ((size_t)info.size < DeReflect_input_size + TAIL_SIZE)
        throw std::runtime_error("Input too small");

    void* input_ptr = info.ptr;
    std::memcpy(input_virtaddr, input_ptr, DeReflect_input_size);
    in_tb->sync_for_write(0, DeReflect_input_size);

    dpu_log("DeReflect Running inference, buffer idx: " + std::to_string(buf_idx));
    std::vector<vart::TensorBuffer*> inputs = {in_tb.get()};
    std::vector<vart::TensorBuffer*> outputs = {out_tb.get()};
    auto job_id = DeReflect_runner->execute_async(inputs, outputs);
    auto status = DeReflect_runner->wait(job_id.first, -1);
    if(status == 0){
        dpu_log("Run singele-input inference success!");
    }else{
        dpu_log("Run singele-input inference Failed!");
    }

    // tail拷贝到output buffer尾部
    out_tb->sync_for_read(0, DeReflect_output_size);
    std::memcpy(static_cast<uint8_t*>(output_virtaddr) + DeReflect_output_size,
                static_cast<uint8_t*>(input_ptr) + DeReflect_input_size,
                TAIL_SIZE);

    uint64_t phy_addr = output_phyaddr;

    dpu_log("Run complete, output buffer idx: " + std::to_string(buf_idx)
            + ", physical address: 0x" + std::to_string(phy_addr));

    DeReflect_cur_buf_idx = (DeReflect_cur_buf_idx + 1) % NUM_BUFFERS;
    return static_cast<int64_t>(phy_addr);
}

// 单输入推理
int64_t InterFrame_run(py::array_t<int8_t> input_buf) {
    if (!InterFrame_runner || InterFrame_input_bo_list.empty() || InterFrame_output_tb_list.empty()) throw std::runtime_error("Runner not initialized");

    // 使用当前的output buffer
    int buf_idx = InterFrame_cur_buf_idx;
    auto& in_tb = InterFrame_input_tb_list[buf_idx];
    auto& in_bo = InterFrame_input_bo_list[buf_idx];
    auto& out_tb = InterFrame_output_tb_list[buf_idx];
    auto& out_bo = InterFrame_output_bo_list[buf_idx];
    auto input_virtaddr = in_bo->map();
    auto input_phyaddr  = in_tb->data_phy({0,0,0,0}).first;
    auto output_virtaddr = out_bo->map();
    auto output_phyaddr = out_tb->data_phy({0,0,0,0}).first;
    std::stringstream ss;
    ss << "inputPhyAddr= 0x" << std::hex << input_phyaddr
       << " OutputPhyAddr= 0x" << std::hex << output_phyaddr;
    dpu_log(ss.str());

    auto info = input_buf.request();
    if ((size_t)info.size < InterFrame_input_size0 + TAIL_SIZE)
        throw std::runtime_error("Input too small");

    void* input_ptr = info.ptr;
    std::memcpy(input_virtaddr, input_ptr, InterFrame_input_size0);
    in_tb->sync_for_write(0, InterFrame_input_size0);

    dpu_log("InterFrame Running inference, buffer idx: " + std::to_string(buf_idx));
    std::vector<vart::TensorBuffer*> inputs = {in_tb.get()};
    std::vector<vart::TensorBuffer*> outputs = {out_tb.get()};
    auto job_id = InterFrame_runner->execute_async(inputs, outputs);
    auto status = InterFrame_runner->wait(job_id.first, -1);
    if(status == 0){
        dpu_log("Run singele-input inference success!");
    }else{
        dpu_log("Run singele-input inference Failed!");
    }

    // tail拷贝到output buffer尾部
    out_tb->sync_for_read(0, InterFrame_output_size);
    std::memcpy(static_cast<uint8_t*>(output_virtaddr) + InterFrame_output_size,
                static_cast<uint8_t*>(input_ptr) + InterFrame_input_size0,
                TAIL_SIZE);

    uint64_t phy_addr = output_phyaddr;

    dpu_log("Run complete, output buffer idx: " + std::to_string(buf_idx)
            + ", physical address: 0x" + std::to_string(phy_addr));

    InterFrame_cur_buf_idx = (InterFrame_cur_buf_idx + 1) % NUM_BUFFERS;
    return static_cast<int64_t>(phy_addr);
}

// 双输入推理
int64_t InterFrame_run2(py::array_t<int8_t> in0, py::array_t<int8_t> in1) {
    if (!InterFrame_runner || InterFrame_input_bo_list.empty() || InterFrame_output_tb_list.empty()) throw std::runtime_error("Runner not initialized");

    // 使用当前的output buffer
    int buf_idx = InterFrame_cur_buf_idx;
    auto& in_tb0 = InterFrame_input_tb_list[buf_idx];
    auto& in_bo0 = InterFrame_input_bo_list[buf_idx];
    auto& in_tb1 = InterFrame_input_tb_list[(buf_idx+1)%NUM_BUFFERS];
    auto& in_bo1 = InterFrame_input_bo_list[(buf_idx+1)%NUM_BUFFERS];
    auto& out_tb = InterFrame_output_tb_list[buf_idx];
    auto& out_bo = InterFrame_output_bo_list[buf_idx];
    auto input_virtaddr0 = in_bo0->map();
    auto input_phyaddr0  = in_tb0->data_phy({0,0,0,0}).first;
    auto input_virtaddr1 = in_bo1->map();
    auto input_phyaddr1  = in_tb1->data_phy({0,0,0,0}).first;
    auto output_virtaddr = out_bo->map();
    auto output_phyaddr = out_tb->data_phy({0,0,0,0}).first;
    std::stringstream ss;
    ss << "inputPhyAddr0= 0x" << std::hex << input_phyaddr0
       << "inputPhyAddr1= 0x" << std::hex << input_phyaddr1
       << " OutputPhyAddr= 0x" << std::hex << output_phyaddr;
    dpu_log(ss.str());

    auto info0 = in0.request();
    auto info1 = in1.request();

    if ((size_t)info0.size < InterFrame_input_size0 || (size_t)info1.size < InterFrame_input_size1 + TAIL_SIZE)
        throw std::runtime_error("Input buffer too small");

    void* ptr0 = info0.ptr;
    void* ptr1 = info1.ptr;
    std::memcpy(input_virtaddr0, ptr0, InterFrame_input_size0);
    std::memcpy(input_virtaddr1, ptr1, InterFrame_input_size1);
    in_tb0->sync_for_write(0, InterFrame_input_size0);
    in_tb1->sync_for_write(0, InterFrame_input_size1);

    dpu_log("Running dual-input inference");
    std::vector<vart::TensorBuffer*> inputs = {in_tb0.get(), in_tb1.get()};
    std::vector<vart::TensorBuffer*> outputs = {out_tb.get()};
    auto job_id = InterFrame_runner->execute_async(inputs, outputs);
    auto status = InterFrame_runner->wait(job_id.first, -1);
    if(status == 0){
        dpu_log("Run singele-input inference success!");
    }else{
        dpu_log("Run singele-input inference Failed!");
    }


    // tail拷贝到output buffer尾部
    out_tb->sync_for_read(0, InterFrame_output_size);
    std::memcpy(static_cast<uint8_t*>(output_virtaddr) + InterFrame_output_size,
                static_cast<uint8_t*>(ptr1) + InterFrame_input_size0,
                TAIL_SIZE);

    uint64_t phy_addr = output_phyaddr;

    dpu_log("Run complete, output buffer idx: " + std::to_string(buf_idx)
            + ", physical address: 0x" + std::to_string(phy_addr));

    InterFrame_cur_buf_idx = (InterFrame_cur_buf_idx + 1) % NUM_BUFFERS;
    return static_cast<int64_t>(phy_addr);
}

// pybind11 导出接口
PYBIND11_MODULE(dpu_mr, m) {
    m.def("Dehaze_dpu_create", &Dehaze_dpu_create);
    m.def("DeReflect_dpu_create", &DeReflect_dpu_create);
    m.def("InterFrame_dpu_create", &InterFrame_dpu_create);
    m.def("Dehaze_dpu_destroy", &Dehaze_dpu_destroy);
    m.def("DeReflect_dpu_destroy", &DeReflect_dpu_destroy);
    m.def("InterFrame_dpu_destroy", &InterFrame_dpu_destroy);
    m.def("Dehaze_get_input_shape", &Dehaze_get_input_shape);
    m.def("Dehaze_get_output_shape", &Dehaze_get_output_shape);
    m.def("DeReflect_get_input_shape", &DeReflect_get_input_shape);
    m.def("DeReflect_get_output_shape", &DeReflect_get_output_shape);
    m.def("InterFrame_get_input_shape", &InterFrame_get_input_shape);
    m.def("InterFrame_get_output_shape", &InterFrame_get_output_shape);
    m.def("Dehaze_run", &Dehaze_run);
    m.def("DeReflect_run", &DeReflect_run);
    m.def("InterFrame_run", &InterFrame_run);
    m.def("InterFrame_run2", &InterFrame_run2);
    m.def("dpu_set_debug", &dpu_set_debug);
    m.def("Init_xrt_device", &Init_xrt_device);
}
