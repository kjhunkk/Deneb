#include "deneb_timestep_controller.h"

#include "avocado.h"
#include "deneb_config_macro.h"
#include "deneb_equation.h"

namespace deneb {
TimestepController::TimestepController(const std::string& timestep_control) {
  if (!timestep_control.compare("CFL"))
    timestep_control_ = TimestepControl::CFL;
  else if (!timestep_control.compare("dt"))
    timestep_control_ = TimestepControl::DT;
  else
    ERROR_MESSAGE("Wrong time step control (no-exist): " + timestep_control +
                  "\n");
}
TimestepController::TimestepController(const std::string& timestep_control,
                                       const std::string& control_tag) {
  if (!timestep_control.compare("CFL"))
    timestep_control_ = TimestepControl::CFL;
  else if (!timestep_control.compare("dt"))
    timestep_control_ = TimestepControl::DT;
  else
    ERROR_MESSAGE("Wrong time step control (no-exist): " + timestep_control +
                  "\n");
  control_tag_ = control_tag;
}
void TimestepController::ComputeLocalTimestep(
    const double* solution, const double ref_value,
    std::vector<double>& local_timestep) {
  UpdateTimestepControlValue(solution, ref_value);

  if (timestep_control_ == TimestepControl::DT) {
    for (auto&& timestep : local_timestep) timestep = timestep_control_value_;
    return;
  }

  DENEB_EQUATION->ComputeLocalTimestep(solution, local_timestep);
  avocado::VecScale(static_cast<int>(local_timestep.size()),
                    timestep_control_value_, &local_timestep[0]);
}
double TimestepController::ComputeGlobalTimestep(
    const double* solution, const double ref_value,
    std::vector<double>& local_timestep) {
  UpdateTimestepControlValue(solution, ref_value);
  if (timestep_control_ == TimestepControl::DT) return timestep_control_value_;

  DENEB_EQUATION->ComputeLocalTimestep(solution, local_timestep);
  const double min_dt = AVOCADO_MPI->Reduce(
      *std::min_element(local_timestep.begin(), local_timestep.end()),
      avocado::MPI::Op::MIN);

  if (!std::isnormal(min_dt))
    ERROR_MESSAGE("Wrong time step.\n\tdt = " + std::to_string(min_dt) + "\n");
  return min_dt * timestep_control_value_;
}
TimestepControllerConstant::TimestepControllerConstant(
    const std::string& timestep_control)
    : TimestepController(timestep_control) {
  auto& config = AVOCADO_CONFIG;
  timestep_control_value_ =
      std::stod(config->GetConfigValue(control_tag_ + ".1"));
}
TimestepControllerConstant::TimestepControllerConstant(
    const std::string& timestep_control, const std::string& control_tag)
    : TimestepController(timestep_control, control_tag) {
  auto& config = AVOCADO_CONFIG;
  timestep_control_value_ =
      std::stod(config->GetConfigValue(control_tag_ + ".1"));
}
TimestepControllerSER::TimestepControllerSER(
    const std::string& timestep_control)
    : TimestepController(timestep_control) {
  auto& config = AVOCADO_CONFIG;
  timestep_control_value_ =
      std::stod(config->GetConfigValue(control_tag_ + ".1"));
  initial_timestep_control_value_ = timestep_control_value_;
  upper_limit_ = std::stod(config->GetConfigValue(control_tag_ + ".2"));
  lower_limit_ = std::stod(config->GetConfigValue(control_tag_ + ".3"));
  rate_upper_limit_ = std::stod(config->GetConfigValue(control_tag_ + ".4"));
  rate_lower_limit_ = std::stod(config->GetConfigValue(control_tag_ + ".5"));
  update_period_ = std::stoi(config->GetConfigValue(control_tag_ + ".6"));
  prev_residual_norm_ = -1.0;
  current_residual_norm_ = 0.0;
  update_iteration_ = 0;
}
TimestepControllerSER::TimestepControllerSER(
    const std::string& timestep_control, const std::string& control_tag)
    : TimestepController(timestep_control, control_tag) {
  auto& config = AVOCADO_CONFIG;
  timestep_control_value_ =
      std::stod(config->GetConfigValue(control_tag_ + ".1"));
  initial_timestep_control_value_ = timestep_control_value_;
  upper_limit_ = std::stod(config->GetConfigValue(control_tag_ + ".2"));
  lower_limit_ = std::stod(config->GetConfigValue(control_tag_ + ".3"));
  rate_upper_limit_ = std::stod(config->GetConfigValue(control_tag_ + ".4"));
  rate_lower_limit_ = std::stod(config->GetConfigValue(control_tag_ + ".5"));
  update_period_ = std::stoi(config->GetConfigValue(control_tag_ + ".6"));
  prev_residual_norm_ = -1.0;
  current_residual_norm_ = 0.0;
  update_iteration_ = 0;
}
void TimestepControllerSER::IntializeTimestepControlValue() {
  timestep_control_value_ = initial_timestep_control_value_;
  prev_residual_norm_ = -1.0;
  current_residual_norm_ = 0.0;
  update_iteration_ = 0;
}
void TimestepControllerSER::UpdateTimestepControlValue(const double* solution,
                                                       const double ref_value) {
  current_residual_norm_ += ref_value;
  if (++update_iteration_ == update_period_) {
    if (prev_residual_norm_ > 0) {
      double rate = prev_residual_norm_ / current_residual_norm_;
      if (rate > rate_upper_limit_) {
        timestep_control_value_ *= rate_upper_limit_;
      } else if (rate < rate_lower_limit_) {
        timestep_control_value_ *= rate_lower_limit_;
      } else
        timestep_control_value_ *= rate;
    }
    if (timestep_control_value_ > upper_limit_)
      timestep_control_value_ = upper_limit_;
    if (timestep_control_value_ < lower_limit_)
      timestep_control_value_ = lower_limit_;

    prev_residual_norm_ = current_residual_norm_;
    current_residual_norm_ = 0.0;
    update_iteration_ = 0;
  }
}
}  // namespace deneb