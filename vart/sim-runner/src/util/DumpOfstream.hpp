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
#ifndef __DUMP_OFSTREAM__
#define __DUMP_OFSTREAM__

#include <atomic>
#include <cassert>
#include <cstring>
#include <fstream>
#include <future>
#include <map>
#include <memory>
#include <queue>
#include <sstream>
#include <thread>

#define DUMP_WITH_CACHE false
#define DUMP_WITH_MULTITHREAD true
#define MAX_THREAD_NUM 10
#define MAX_DUMP_CACHE_SIZE 2000000
#define RESERVED_DUMP_CACHE_SIZE 1000
#define OVER_DUE_LIMIT 5
using number_t = unsigned long long int;

namespace dump {
class FutureManager_ {
 public:
  static FutureManager_& instance() {
    static FutureManager_ futManager_;
    return futManager_;
  }
  FutureManager_() = default;
  ~FutureManager_() {
    while (futs_.size() > 0) {
      futs_.front().wait();
      futs_.pop();
    }
  }
  void add(std::future<void>&& fut) {
    while (futs_.size() > MAX_THREAD_NUM) {
      futs_.front().wait();
      futs_.pop();
    }
    futs_.push(std::move(fut));
  }

 private:
  std::queue<std::future<void>> futs_;
};

class StringBuffer_ {
 public:
  StringBuffer_()
      : capacity_(MAX_DUMP_CACHE_SIZE),
        reserved_(RESERVED_DUMP_CACHE_SIZE),
        buf_(new char[capacity_ + reserved_]),
        size_(0),
        state_(true) {}
  ~StringBuffer_() {
    if (buf_) delete[] buf_;
  }
  StringBuffer_& add(const char* data, int data_size) {
    if (size_ + data_size <= capacity_) {
      memcpy(buf_ + size_, data, data_size);
    } else if ((size_ + data_size > capacity_) &&
               (size_ + data_size <= capacity_ + reserved_)) {
      memcpy(buf_ + size_, data, data_size);
      state_ = false;
    } else {
      char* tmp = buf_;
      buf_ = new char[size_ + data_size];
      reserved_ = size_ + data_size - capacity_;
      memcpy(buf_, tmp, size_);
      memcpy(buf_ + size_, data, data_size);
      state_ = false;
      delete[] tmp;
    }

    size_ += data_size;
    return (*this);
  }
  bool getState() { return state_; }
  char* data() { return buf_; }
  int getSize() { return size_; }
  int size() { return size_; }

 private:
  const int capacity_;
  int reserved_;
  char* buf_;
  int size_;
  bool state_;
};

struct DumpOfstreamContext {
  /*
   * handle in three ways:
   * 1. fullfilled --> DumpOfstream_::~DumpOfstream_
   * 2. overdue    --> DumpOfstream_::~DumpOfstream_
   * 3. released by runtime --> DumpOfstreamContext::~DumpOfstreamContext
   **/
  static number_t to_queue_context_No_;
  static std::atomic<number_t> atomic_to_emit_context_No_;

  std::string dumpFilePath_;
  std::ios_base::openmode openMode_;
  StringBuffer_ buffer_;
  time_t lastSaveTime_;
  bool isFine;
  bool isDumped;
  number_t context_No_;

  DumpOfstreamContext(const std::string& dumpFilePath,
                      std::ios_base::openmode openMode)
      : dumpFilePath_(dumpFilePath),
        openMode_(openMode),
        lastSaveTime_(0),
        isFine(false),
        isDumped(false),
        context_No_(0) {}
  ~DumpOfstreamContext() {
    // for the handle of remained context in assembleContextManager_
    // when it's destroyed by runtime
    if (!isDumped) toFile();
  }
  static void toQueue(std::shared_ptr<DumpOfstreamContext> context_);
  static void toEmit(std::shared_ptr<DumpOfstreamContext> context_);
  void toFile();
  bool isOverdue();
  bool isFullfilled();
};

class DumpOfstream_ {
 public:
  DumpOfstream_() : context_(nullptr), os_("") {}
  ~DumpOfstream_() {
    assert(os_.str().length() == 0 && "the ostream has not been closed");
    for (auto iter = assembleContextManager_.begin();
         iter != assembleContextManager_.end();) {
      if (iter->second->isOverdue() || iter->second->isFullfilled()) {
        // to queue before erasing from assembleContextManager_
        auto context_toQueue = iter->second;
        iter = assembleContextManager_.erase(iter);
        DumpOfstreamContext::toQueue(context_toQueue);
        continue;
      } else {
        ++iter;
      }
    }
  }
  void open(const std::string& dumpFilePath,
            std::ios_base::openmode openMode = std::ios_base::out) {
    if (0 != assembleContextManager_.count(dumpFilePath)) {
      context_ = assembleContextManager_.at(dumpFilePath);
      // 0x21 = std::ios::trunc | std::ios::app
      assert(((0x21 & openMode) == (0x21 & context_->openMode_)) &&
             "invalid openMode");
    } else {
      context_ = std::make_shared<DumpOfstreamContext>(dumpFilePath, openMode);
      assembleContextManager_.insert({dumpFilePath, context_});
    }
  }
  void close() {
    context_->lastSaveTime_ = time(NULL);
    context_->buffer_.add(os_.str().c_str(), os_.str().length());
    os_.str("");
  }
  DumpOfstream_& write(const char* data, uint64_t size) {
    std::string tmp(data, data + size);
    os_ << tmp;
    return (*this);
  }
  const std::shared_ptr<DumpOfstreamContext> getContext() const {
    return context_;
  }
  template <typename T>
  friend DumpOfstream_& operator<<(DumpOfstream_& dump_os, const T& data);
  friend DumpOfstream_& operator<<(DumpOfstream_& dump_os,
                                   std::ostream& (*fptr)(std::ostream&));

 private:
  static std::map<std::string, std::shared_ptr<DumpOfstreamContext>>
      assembleContextManager_;
  std::shared_ptr<DumpOfstreamContext> context_;
  std::stringstream os_;
};

template <typename T>
DumpOfstream_& operator<<(DumpOfstream_& dump_os, const T& data) {
  dump_os.os_ << data;
  return dump_os;
}
}  // namespace dump

#if DUMP_WITH_CACHE
using DumpOfstream = dump::DumpOfstream_;
#else
using DumpOfstream = std::ofstream;
#endif

#endif /* __DUMP_OFSTREAM__ */