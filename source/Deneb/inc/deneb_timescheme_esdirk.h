#pragma once

#include <vector>

#include "deneb_system_matrix.h"
#include "deneb_timescheme.h"
#include "deneb_timestep_controller.h"

namespace deneb {
// -------------------- ESDIRK ------------------- //
class TimeschemeESDIRK : public Timescheme {
 private:
  Mat sysmat_;
  GMRES solver_;

  Vec temp_rhs_;
  Vec stage_solution_;
  Vec delta_;
  Vec temp_delta_;

  std::vector<Vec> stage_rhs_;
  std::vector<double> base_rhs_;

  std::shared_ptr<TimestepController> pseudo_timestep_controller_;

 protected:
  bool stiffly_accurate_;
  int time_order_;
  int num_stages_;
  int max_newton_iteration_;
  int jacobi_recompute_max_iteration_;
  int shock_capturing_freeze_iteration_;
  double newton_relative_error_tol_;
  double newton_absolute_error_tol_;
  double jacobi_recompute_tol_;

  std::vector<double> coeff_b_;
  std::vector<double> coeff_c_;
  std::vector<std::vector<double>> coeff_a_;

 public:
  TimeschemeESDIRK();
  ~TimeschemeESDIRK();

  virtual void BuildData(void);
  virtual void Marching(void);
};

// ------------------------ ESDIRK3 ------------------------ //
// Jorgensen, 2018
class TimeschemeESDIRK3 : public TimeschemeESDIRK {
 public:
  TimeschemeESDIRK3();
  ~TimeschemeESDIRK3();

  virtual void BuildData(void);
  virtual void Marching(void);
};
// ------------------------ ESDIRK4 ------------------------ //
// Blom, 2004
class TimeschemeESDIRK4 : public TimeschemeESDIRK {
 public:
  TimeschemeESDIRK4();
  ~TimeschemeESDIRK4();

  virtual void BuildData(void);
  virtual void Marching(void);
};
}  // namespace deneb