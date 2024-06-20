#pragma once

#include <vector>

#include "deneb_system_matrix.h"
#include "deneb_timescheme.h"
#include "deneb_timestep_controller.h"

namespace deneb {
// -------------- Strong stability preserving SDIRK -------------- //
// Ferracina and Spijker, 2008
class TimeschemeSDIRK : public Timescheme {
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
  int time_order_;
  int num_stages_;
  int max_newton_iteration_;
  int jacobi_recompute_max_iteration_;
  int shock_capturing_freeze_iteration_;
  double newton_relative_error_tol_;
  double jacobi_recompute_tol_;

  std::vector<double> coeff_b_;
  std::vector<double> coeff_c_;
  std::vector<std::vector<double>> coeff_a_;

 public:
  TimeschemeSDIRK();
  ~TimeschemeSDIRK();

  virtual void BuildData(void);
  virtual void Marching(void);
};

// ------------------------ SDIRK21 ------------------------ //
// 2nd order, 1 stage
class TimeschemeSDIRK21 : public TimeschemeSDIRK {
 public:
  TimeschemeSDIRK21();
  ~TimeschemeSDIRK21();

  virtual void BuildData(void);
  virtual void Marching(void);
};

// ------------------------ SDIRK22 ------------------------ //
// 2nd order, 2 stages
class TimeschemeSDIRK22 : public TimeschemeSDIRK {
 public:
  TimeschemeSDIRK22();
  ~TimeschemeSDIRK22();

  virtual void BuildData(void);
  virtual void Marching(void);
};

// ------------------------ SDIRK32 ------------------------ //
// 3rd order, 2 stages
class TimeschemeSDIRK32 : public TimeschemeSDIRK {
 public:
  TimeschemeSDIRK32();
  ~TimeschemeSDIRK32();

  virtual void BuildData(void);
  virtual void Marching(void);
};

// ------------------------ SDIRK43 ------------------------ //
// 4th order, 3 stages
class TimeschemeSDIRK43 : public TimeschemeSDIRK {
 public:
  TimeschemeSDIRK43();
  ~TimeschemeSDIRK43();

  virtual void BuildData(void);
  virtual void Marching(void);
};
}  // namespace deneb