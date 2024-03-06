#pragma once

#include <vector>
#include <string>

namespace deneb {
class TimestepController {
 protected:
  enum class TimestepControl : int { CFL = 0, DT = 1 };

  double timestep_control_value_;

  std::string control_tag_;

  TimestepControl timestep_control_;

 public:
  TimestepController(const std::string& timestep_control);
  TimestepController(const std::string& timestep_control,
                     const std::string& control_tag);
  virtual ~TimestepController(){};

 public:
  inline double GetTimestepControlValue() { return timestep_control_value_; }
  virtual void IntializeTimestepControlValue() = 0;
  void ComputeLocalTimestep(const double* solution, const double ref_value,
                            std::vector<double>& local_timestep);

  double ComputeGlobalTimestep(const double* solution, const double ref_value,
                               std::vector<double>& local_timestep);

 protected:
  virtual void UpdateTimestepControlValue(const double* solution,
                                          const double ref_value) = 0;
};
class TimestepControllerConstant : public TimestepController {
 public:
  TimestepControllerConstant(const std::string& timestep_control);
  TimestepControllerConstant(const std::string& timestep_control,
                             const std::string& control_tag);
  virtual ~TimestepControllerConstant() {}

  virtual void IntializeTimestepControlValue(){};

 protected:
  virtual void UpdateTimestepControlValue(const double* solution,
                                          const double ref_value) {}
};
class TimestepControllerSER : public TimestepController {
 protected:
  double initial_timestep_control_value_;
  double prev_residual_norm_;
  double upper_limit_;
  double lower_limit_;
  double rate_upper_limit_;
  double rate_lower_limit_;
  int update_period_;
  int update_iteration_;
  double current_residual_norm_;

 public:
  TimestepControllerSER(const std::string& timestep_control);
  TimestepControllerSER(const std::string& timestep_control,
                        const std::string& control_tag);
  virtual ~TimestepControllerSER() {}

  virtual void IntializeTimestepControlValue();

 protected:
  virtual void UpdateTimestepControlValue(const double* solution,
                                          const double ref_value);
};
}  // namespace deneb
