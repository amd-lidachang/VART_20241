#pragma once

#include <string>
#include <string_view>

#include "InstTable.hpp"

template <DPUVersion T>
struct target_traits {
  static constexpr std::string_view target_name{"unknown_target"};
};

template <>
struct target_traits<DPUVersion::XVDPU> {
  static constexpr std::string_view target_name{"xvdpu"};
};

template <>
struct target_traits<DPUVersion::XV2DPU> {
  static constexpr std::string_view target_name{"xv2dpu"};
};

template <DPUVersion T>
struct Target : public target_traits<T> {
  static std::string data_path(const std::string subpath = "") {
    const std::string path{"./"};
    return path + string(target_traits<T>::target_name) + "_data/" + subpath;
  }
};
