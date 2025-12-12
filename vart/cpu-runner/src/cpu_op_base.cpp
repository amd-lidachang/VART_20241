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

#include "cpu_op_base.hpp"

#include "vart/xir_helper.hpp"
#include "xir/op/op.hpp"

namespace vart {
namespace cpu {

std::atomic<uint64_t> CPUOPBase::subg_ops;
const string CPUOPBase::SUBG_DIFF_SCRIPT = "diff.sh";
const string CPUOPBase::SUBG_DIFF_SCRIPT_HEADER = R"code(
#!/bin/bash

if [ $# -eq 0 ];then
  if [ $GOLDEN_PATH ];then
    GOLDEN_PATH=${GOLDEN_PATH}
  else
    echo "param error! pls set GOLDEN_PATH env or use: ./diff.sh {golden_path}"
    echo "./diff.sh {golden_path}"
    exit 1
  fi
elif [ $# -eq 1 ];then
    GOLDEN_PATH=$1
else
  echo "param error! pls set GOLDEN_PATH env or use: ./diff.sh {golden_path}"
  exit 1
fi

#set -x

if_same=true
has_match_file=false
for file in ./*
do
    filename=$(basename $file )
    golden_file=${GOLDEN_PATH}/${filename}
    if [ ! -f $gloden_file ]
    then
      file_size="$(wc -c <"$file")"
      gloden_file_size="$(wc -c <"$gloden_file")"
      if [ $file_size -ne $gloden_file_size ]
      then
        echo "The size of $filename not match!"
      else
        diff ${file} ${golden_file} 1>diff.txt 2>&1 && result=0 || result=1

        if [ "$result" == 1 ];then
            echo "$filename is diff"
            if_same=false
        else
            has_match_file=true    
        fi
      fi
    #else
     # echo "$golden_file does not exist, please check filename."
    fi
done

if [ "$if_same" = true -a "$has_match_file" = true ]; then
  echo "All same!"
fi

)code";

void CPUOPBase::StaticInit() { subg_ops.store(0); }

CPUOPBase::CPUOPBase(const xir::Subgraph* subg, const xir::Op* op,
                     IMapTBs_t inputs, CPUTBPtr_t output)
    : xir_subg_(subg),
      xir_op_(op),
      output_tensor_(xir_op_->get_output_tensor()),
      inputs_(inputs),
      output_(output),
      data_type_(output_tensor_->get_data_type().type),
      bit_width_(output_tensor_->get_data_type().bit_width) {
  if_signed_ = get_if_signed(data_type_);
  data_min_ = if_signed_ ? -pow(2, bit_width_ - 1) : 0;
  data_max_ = if_signed_ ? pow(2, bit_width_ - 1) - 1 : pow(2, bit_width_) - 1;
}

void CPUOPBase::save() {
  if (CPUCfg::Instance().debug()) {
    output_->save_bin();
  }
}

string CPUOPBase::get_input_list() const {
  string s;
  auto v = vec_input_ops(xir_op_->get_input_ops());

  for (auto i = 0U; i < v.size(); i++) {
    s += v[i]->get_name();
    s += "(" + v[i]->get_type() + ")";
    if (i != v.size() - 1) {
      s += ", ";
    }
  }

  return s;
}

}  // namespace cpu
}  // namespace vart
