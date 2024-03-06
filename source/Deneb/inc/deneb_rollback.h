#pragma once

#include <vector>

namespace deneb {
class RollBack {
 protected:
  int buffer_size_;
  int length_;
  int buffer_pointer_;
  std::vector<std::vector<double>> buffer_;

 public:
  RollBack(){};
  virtual ~RollBack(){};
  virtual void ExecuteRollBack(double* solution, int& iteration) = 0;
  virtual void UpdateBuffer(double* solution) = 0;
  virtual bool Status() = 0;
};
class NoRollBack : public RollBack {
 public:
  NoRollBack(){};
  virtual ~NoRollBack(){};
  virtual void ExecuteRollBack(double* solution, int& iteration){};
  virtual void UpdateBuffer(double* solution){};
  virtual bool Status() { return false; }
};
class ConstantRollBack : public RollBack {
 protected:
  int roll_back_step_;

 public:
  ConstantRollBack();
  virtual ~ConstantRollBack(){};
  virtual void ExecuteRollBack(double* solution, int& iteration);
  virtual void UpdateBuffer(double* solution);
  virtual bool Status() { return true; }
};
}  // namespace deneb