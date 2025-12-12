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
#ifndef __DUMPBANK_HPP__
#define __DUMPBANK_HPP__

#include "InstBase.hpp"

template <DPUVersion dv> class DumpBank : public InstBase {
public:
  // real constructor
  explicit DumpBank(int inst_type, int instid, vector<string> &inst_str,
                    vector<uint32_t> &inst_val);
  ~DumpBank();

public:
  // excute load instruction
  void Exec() override final;
  void GenInstStr(int inst_fmt) override final;

private:
  void dump_bank_range();

private:
  string save_name_;
  int save_fmt_;
  int bank_start_;
  int bank_dump_num_;
};

// used for debug model
class DDRblock {
 public:
  static bool status;
  static std::string subgraphName_;
  static std::set<DDRblock> usedBlocks;

 public:
  static bool insert(const std::string& subgraphName,
                     const DDRblock& ddrBlock) {
    if (subgraphName != DDRblock::subgraphName_) {
      assert(status);
      DDRblock::subgraphName_ = subgraphName;
      DDRblock::usedBlocks.clear();
    }
    bool isCrossWithOtherBlock = false;
    for (auto iter = usedBlocks.begin(); iter != usedBlocks.end(); ++iter) {
      if (isCorss(*iter, ddrBlock)) {
        isCrossWithOtherBlock = true;
        std::cout << "two crossed blocks:\n"
                  << iter->toString() << "\n"
                  << ddrBlock.toString() << std::endl;
      }
    }
    if (isCrossWithOtherBlock) {
      status = false;
      std::cout << "layerName: " << subgraphName_ << std::endl;
    }
    usedBlocks.insert(ddrBlock);
    return isCrossWithOtherBlock;
  }

  static bool isCorss(const DDRblock& rgt, const DDRblock& lft) {
    int max_low = std::max(rgt.addr_, lft.addr_);
    int min_high = std::min(rgt.addr_ + rgt.size_, lft.addr_ + lft.size_);

    return (rgt.reg_id_ == lft.reg_id_) && (min_high > max_low);
  }

 public:
  explicit DDRblock(int reg_id, int addr, int size)
      : reg_id_(reg_id), addr_(addr), size_(size) {}

  bool operator<(const DDRblock& rgt) const {
    if (this->reg_id_ < rgt.reg_id_) return true;
    if (this->reg_id_ > rgt.reg_id_) return false;
    if (this->reg_id_ == rgt.reg_id_) {
      return this->addr_ < rgt.addr_;
    }
    return false;
  }

  std::string toString() const {
    return "reg_id: " + std::to_string(reg_id_) + " " +
           "addr: " + std::to_string(addr_) + " " +
           "size: " + std::to_string(size_);
  }

 public:
  int reg_id_;
  int addr_;
  int size_;
};

enum class TOBANK { ENGINE2BANK = 0, DDR2BANK = 1 };

class Bankblock {
 public:
  static bool status_engine2bank;
  static bool status_DDR2bank;
  static std::string subgraphName_;
  static std::set<Bankblock> usedBlocks_engine2bank;
  static std::set<Bankblock> usedBlocks_DDR2bank;

 public:
  static bool insert(const std::string& subgraphName,
                     const Bankblock& bankBlock, TOBANK toBank) {
    if (subgraphName != Bankblock::subgraphName_) {
      assert(status_DDR2bank);
      assert(status_engine2bank);
      Bankblock::subgraphName_ = subgraphName;
      Bankblock::usedBlocks_DDR2bank.clear();
      Bankblock::usedBlocks_engine2bank.clear();
    }

    bool isCrossWithOtherBlock = false;
    if (TOBANK::ENGINE2BANK == toBank) {
      for (auto iter = usedBlocks_engine2bank.begin();
           iter != usedBlocks_engine2bank.end(); ++iter) {
        if (isCorss(*iter, bankBlock)) {
          isCrossWithOtherBlock = true;
          std::cout << "two crossed blocks when engine2bank:\n"
                    << iter->toString() << "\n"
                    << bankBlock.toString() << std::endl;
        }
      }
      if (isCrossWithOtherBlock) {
        status_engine2bank = false;
        std::cout << "layerName: " << subgraphName_ << std::endl;
      }
      usedBlocks_engine2bank.insert(bankBlock);

    } else if (TOBANK::DDR2BANK == toBank) {
      for (auto iter = usedBlocks_DDR2bank.begin();
           iter != usedBlocks_DDR2bank.end(); ++iter) {
        if (isCorss(*iter, bankBlock)) {
          isCrossWithOtherBlock = true;
          std::cout << "two crossed blocks when DDR2bank:\n"
                    << iter->toString() << "\n"
                    << bankBlock.toString() << std::endl;
        }
      }
      if (isCrossWithOtherBlock) {
        status_DDR2bank = false;
        std::cout << "layerName: " << subgraphName_ << std::endl;
      }
      usedBlocks_DDR2bank.insert(bankBlock);

    } else {
      assert("error TOBANK" && false);
    }

    return isCrossWithOtherBlock;
  }

  static bool isCorss(const Bankblock& rgt, const Bankblock& lft) {
    int max_low = std::max(rgt.addr_, lft.addr_);
    int min_high = std::min(rgt.addr_ + rgt.size_, lft.addr_ + lft.size_);

    return (rgt.bank_id_ == lft.bank_id_) && (min_high > max_low);
  }

 public:
  explicit Bankblock(int bank_id, int addr, int size)
      : bank_id_(bank_id), addr_(addr), size_(size) {}

  bool operator<(const Bankblock& rgt) const {
    if (this->bank_id_ < rgt.bank_id_) return true;
    if (this->bank_id_ > rgt.bank_id_) return false;
    if (this->bank_id_ == rgt.bank_id_) {
      return this->addr_ < rgt.addr_;
    }
    return false;
  }

  std::string toString() const {
    return "bank_id: " + std::to_string(bank_id_) + " " +
           "addr: " + std::to_string(addr_) + " " +
           "size: " + std::to_string(size_);
  }

 public:
  int bank_id_;
  int addr_;
  int size_;
};

#endif /* __DUMPBANK_HPP__ */
