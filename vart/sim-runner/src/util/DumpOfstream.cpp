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
#include "DumpOfstream.hpp"
#include "Util.hpp"

namespace dump {
std::map<std::string, std::shared_ptr<DumpOfstreamContext>>
    DumpOfstream_::assembleContextManager_;
number_t DumpOfstreamContext::to_queue_context_No_{0};
std::atomic<number_t> DumpOfstreamContext::atomic_to_emit_context_No_{0};

void DumpOfstreamContext::toQueue(
    std::shared_ptr<DumpOfstreamContext> context_) {
  if (DUMP_WITH_MULTITHREAD) {
    context_->context_No_ = (to_queue_context_No_++);
    std::future<void> fut =
        std::async(std::launch::async, DumpOfstreamContext::toEmit, context_);
    FutureManager_::instance().add(std::move(fut));
  } else {
    context_->toFile();
  }
}

void DumpOfstreamContext::toEmit(
    std::shared_ptr<DumpOfstreamContext> context_) {
  while (atomic_to_emit_context_No_ != context_->context_No_) {
  }
  context_->toFile();
  atomic_to_emit_context_No_++;
}

void DumpOfstreamContext::toFile() {
  assert(!isDumped && "duplicated dump");
  if (buffer_.getSize() > 0) {
    std::ofstream f;
    f.open(dumpFilePath_, openMode_);
    Util::ChkOpen(f, dumpFilePath_);
    f.write(buffer_.data(), buffer_.getSize());
    f.close();
  }
  isDumped = true;
}

bool DumpOfstreamContext::isOverdue() {
  return (0 != lastSaveTime_) &&
         (difftime(time(NULL), lastSaveTime_) > OVER_DUE_LIMIT);
}

bool DumpOfstreamContext::isFullfilled() { return !buffer_.getState(); }

DumpOfstream_& operator<<(DumpOfstream_& dump_os,
                          std::ostream& (*fptr)(std::ostream&)) {
  dump_os.os_ << fptr;
  return dump_os;
}
}  // namespace dump