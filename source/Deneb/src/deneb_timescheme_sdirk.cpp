#include "deneb_timescheme_sdirk.h"

#include <iomanip>
#include <numeric>
#include <sstream>

#include "avocado.h"
#include "deneb_artificial_viscosity.h"
#include "deneb_config_macro.h"
#include "deneb_contour.h"
#include "deneb_data.h"
#include "deneb_equation.h"
#include "deneb_limiter.h"
#include "deneb_pressurefix.h"
#include "deneb_saveload.h"
#include "deneb_utility.h"

namespace deneb {
// -------------------- SDIRK ------------------- //
TimeschemeSDIRK::TimeschemeSDIRK() : Timescheme(true){};
TimeschemeSDIRK::~TimeschemeSDIRK() {
  VecDestroy(&temp_rhs_);
  VecDestroy(&stage_solution_);
  VecDestroy(&delta_);
  VecDestroy(&temp_delta_);
  for (auto vec : stage_rhs_) VecDestroy(&vec);
  stage_rhs_.clear();
};
void TimeschemeSDIRK::BuildData(void) {
  const int& num_states = DENEB_EQUATION->GetNumStates();
  const int& num_bases = DENEB_DATA->GetNumBases();
  const int& num_cells = DENEB_DATA->GetNumCells();
  const int& num_global_cells = DENEB_DATA->GetNumGlobalCells();
  length_ = num_cells * num_states * num_bases;
  const int global_length = num_global_cells * num_states * num_bases;

  VecCreate(MPI_COMM_WORLD, &temp_rhs_);
  VecSetSizes(temp_rhs_, length_, global_length);
  VecSetFromOptions(temp_rhs_);
  VecAssemblyBegin(temp_rhs_);
  VecAssemblyEnd(temp_rhs_);
  VecCreate(MPI_COMM_WORLD, &stage_solution_);
  VecDuplicate(temp_rhs_, &stage_solution_);
  VecCreate(MPI_COMM_WORLD, &delta_);
  VecDuplicate(temp_rhs_, &delta_);
  VecCreate(MPI_COMM_WORLD, &temp_delta_);
  VecDuplicate(temp_rhs_, &temp_delta_);

  base_rhs_.resize(length_);
  stage_rhs_.resize(num_stages_);
  for (int istage = 0; istage < num_stages_; istage++) {
    VecCreate(MPI_COMM_WORLD, &stage_rhs_[istage]);
    VecDuplicate(temp_rhs_, &stage_rhs_[istage]);
  }

  local_timestep_.resize(num_cells);
  computing_cost_ = 0.0;

  InitializeSystemMatrix(sysmat_);
  solver_.Initialize(sysmat_);

  auto& config = AVOCADO_CONFIG;
  const std::string& roll_back_mechanism =
      config->GetConfigValue(ROLL_BACK_MECHANISM);
  if (!roll_back_mechanism.compare("None")) {
    roll_back_ = std::make_shared<NoRollBack>();
  } else if (!roll_back_mechanism.compare("Constant")) {
    roll_back_ = std::make_shared<ConstantRollBack>();
  } else
    ERROR_MESSAGE(
        "Wrong roll back mechanism (no-exist): " + roll_back_mechanism + "\n");
  newton_relative_error_tol_ =
      std::stod(config->GetConfigValue(NEWTON_RELATIVE_ERROR_TOL));
  newton_absolute_error_tol_ =
      std::stod(config->GetConfigValue(NEWTON_ABSOLUTE_ERROR_TOL));
  max_newton_iteration_ =
      std::stoi(config->GetConfigValue(NEWTON_MAX_ITERATION));

  const std::string pseudo_timestep_control_tag = "PseudoTimestep";
  const std::string& pseudo_timestep_control =
      config->GetConfigValue(pseudo_timestep_control_tag + ".0");
  pseudo_timestep_controller_ = std::make_shared<TimestepControllerSER>(
      pseudo_timestep_control, pseudo_timestep_control_tag);

  jacobi_recompute_max_iteration_ =
      std::stoi(config->GetConfigValue(JACOBIAN_RECOMPUTE_ITERATION));
  jacobi_recompute_tol_ =
      std::stod(config->GetConfigValue(JACOBIAN_RECOMPUTE_TOL));
  shock_capturing_freeze_iteration_ =
      std::stoi(config->GetConfigValue(SHOCK_CAPTURING_FREEZE_ITERATION));
  if (shock_capturing_freeze_iteration_ < 0)
    shock_capturing_freeze_iteration_ = max_newton_iteration_;

  InitSolution();
}
// delta control
void TimeschemeSDIRK::Marching(void) {
  DENEB_LIMITER->Limiting(&solution_[0]);
  DENEB_PRESSUREFIX->Execute(&solution_[0]);
  DENEB_ARTIFICIAL_VISCOSITY->ComputeArtificialViscosity(&solution_[0], 0.0);
  auto& config = AVOCADO_CONFIG;
  const std::string& dir = config->GetConfigValue(RETURN_DIR);
  DENEB_CONTOUR->FaceGrid(dir + RETURN_POST_DIR + "Face/Grid" +
                          std::to_string(iteration_) + ".plt");
  DENEB_CONTOUR->CellGrid(dir + RETURN_POST_DIR + "Grid" +
                          std::to_string(iteration_) + ".plt");
  DENEB_CONTOUR->FaceSolution(
      dir + RETURN_POST_DIR + "Face/Iter" + std::to_string(iteration_) + ".plt",
      &solution_[0], GetCurrentTime());
  DENEB_CONTOUR->CellSolution(
      dir + RETURN_POST_DIR + "Iter" + std::to_string(iteration_) + ".plt",
      &solution_[0], GetCurrentTime());

  const int& num_states = DENEB_EQUATION->GetNumStates();
  const int& num_bases = DENEB_DATA->GetNumBases();
  const int& num_cells = DENEB_DATA->GetNumCells();
  const auto& mat_index = DENEB_DATA->GetMatIndex();
  const int sb = num_states * num_bases;
  std::vector<double> block(sb * sb);

  bool is_stop, is_post, is_save;
  const double* solution = &solution_[0];
  double *temp_rhs_ptr, *stage_solution_ptr, *stage_rhs_ptr;
  double relative_delta_norm = 1.0;
  double delta_norm = 1.0;
  std::vector<double> pseudo_local_timestep(num_cells, 0.0);
  std::vector<double> pseudo_local_timestep_inv(num_cells, 0.0);

  // for debug
  std::string filename =
      dir + RETURN_POST_DIR + "history" + std::to_string(iteration_) + ".dat";
  std::ofstream history;
  if (MYRANK == MASTER_NODE) {
    history.open(filename);
    history << "variables = \"time\", \"ComT\", "
               "\"rho_N2\", \"rho_O2\", \"rho_NO\", \"rho_N\", \"rho_O\", "
               "\"T_tr\", \"T_eev\", \"rho\", \"p\", \"e\", \"e_eev\"\n";
    history.close();
  }
  // end for debug

  while (iteration_ < max_iteration_) {
    START_TIMER();
    // Computing time step
    unsigned __int64 time_step =
        ConvertTime(ComputeGlobalTimestep(solution, local_timestep_));
    is_stop = stop_.CheckTimeFinish(current_time_, time_step);
    is_post = post_.CheckTimeEvent(current_time_, time_step);
    is_save = save_.CheckTimeEvent(current_time_, time_step);

    const double t = GetCurrentTime();
    const double dt = ConvertTime(time_step);
    const double dt_inv = 1.0 / dt;

    // Updating solution
    DENEB_EQUATION->PreProcess(solution);

    VecGetArray(stage_solution_, &stage_solution_ptr);
    cblas_dcopy(length_, solution, 1, stage_solution_ptr, 1);
    VecRestoreArray(stage_solution_, &stage_solution_ptr);

    // Stage 1~
    DENEB_EQUATION->ComputeSystemMatrix(solution, sysmat_,
                                        t + coeff_c_[0] * dt);
    MatShift(sysmat_, dt_inv / coeff_a_[0][0]);
    double pseudo_dt_inv = 0.0;
    double jacobi_recompute_iteration = 0;
    std::array<int, 2> sum_sub_iteration = {0, 0};
    int sum_newton_iteration = 0;
    for (int istage = 0; istage < num_stages_; istage++) {
      // RHS formulation
      cblas_daxpby(length_, dt_inv / coeff_a_[istage][istage], solution, 1, 0.0,
                   &base_rhs_[0], 1);
      for (int jstage = 0; jstage < istage; jstage++) {
        VecGetArray(stage_rhs_[jstage], &stage_rhs_ptr);
        cblas_daxpy(length_,
                    -coeff_a_[istage][jstage] / coeff_a_[istage][istage],
                    stage_rhs_ptr, 1, &base_rhs_[0], 1);
        VecRestoreArray(stage_rhs_[jstage], &stage_rhs_ptr);
      }

      // Global time step : Initialize pseudo time step
      MatShift(sysmat_, -pseudo_dt_inv);
      pseudo_dt_inv = 0.0;

      // Local time step : Initialize pseudo time step
      // for (int icell = 0; icell < num_cells; icell++) {
      //  memset(&block[0], 0, sb * sb * sizeof(double));

      //  for (int i = 0; i < sb * sb; i += (sb + 1))
      //    block[i] = -pseudo_local_timestep_inv[icell];
      //  pseudo_local_timestep_inv[icell] = 0.0;

      //  MatSetValuesBlocked(sysmat_, 1, &mat_index[icell], 1,
      //  &mat_index[icell],
      //                      &block[0], ADD_VALUES);
      //}
      // MatAssemblyBegin(sysmat_, MAT_FINAL_ASSEMBLY);
      // MatAssemblyEnd(sysmat_, MAT_FINAL_ASSEMBLY);

      // Newton iteration
      relative_delta_norm = 1.0;
      int newton_iteration = 0;
      delta_norm = 1.0;
      double initial_delta_norm = 0.0;
      double convergence_rate = 0.0;
      double prev_delta_norm = 0.0;
      VecSet(temp_delta_, 0.0);
      pseudo_timestep_controller_->IntializeTimestepControlValue();
      while (newton_iteration < max_newton_iteration_) {
        // Stage RHS formulation
        VecGetArray(stage_rhs_[istage], &stage_rhs_ptr);
        VecGetArray(temp_rhs_, &temp_rhs_ptr);
        VecGetArray(stage_solution_, &stage_solution_ptr);
        DENEB_EQUATION->ComputeRHS(stage_solution_ptr, stage_rhs_ptr,
                                   t + coeff_c_[istage] * dt);
        cblas_dcopy(length_, stage_rhs_ptr, 1, temp_rhs_ptr, 1);
        cblas_daxpby(length_, -dt_inv / coeff_a_[istage][istage],
                     stage_solution_ptr, 1, -1.0, temp_rhs_ptr, 1);
        cblas_daxpy(length_, 1.0, &base_rhs_[0], 1, temp_rhs_ptr, 1);
        VecRestoreArray(stage_rhs_[istage], &stage_rhs_ptr);
        VecRestoreArray(temp_rhs_, &temp_rhs_ptr);

        // Jacobian matrix recompute
        if (jacobi_recompute_iteration > jacobi_recompute_max_iteration_ ||
            convergence_rate > jacobi_recompute_tol_) {
          jacobi_recompute_iteration = 0;
          DENEB_EQUATION->ComputeSystemMatrix(stage_solution_ptr, sysmat_,
                                              t + coeff_c_[istage] * dt);

          MatShift(sysmat_, dt_inv / coeff_a_[istage][istage]);

          // Global pseudo time step
          pseudo_dt_inv = 0.0;

          // Local pseudo time step
          // memset(&pseudo_local_timestep_inv[0], 0, num_cells *
          // sizeof(double));
        }

        // Global pseudo time step
        const double pseudo_dt =
            pseudo_timestep_controller_->ComputeGlobalTimestep(
                stage_solution_ptr, relative_delta_norm, local_timestep_);
        VecRestoreArray(stage_solution_, &stage_solution_ptr);
        MatShift(sysmat_, -pseudo_dt_inv + 1.0 / pseudo_dt);
        pseudo_dt_inv = 1.0 / pseudo_dt;

        // Local pseudo time step
        // VecGetArray(stage_solution_, &stage_solution_ptr);
        // pseudo_timestep_controller_->ComputeLocalTimestep(
        //    stage_solution_ptr, residual_norm / initial_residual_norm,
        //    pseudo_local_timestep);
        // VecRestoreArray(stage_solution_, &stage_solution_ptr);
        // for (int icell = 0; icell < num_cells; icell++) {
        //  memset(&block[0], 0, sb * sb * sizeof(double));
        //  const double dt_factor = 1.0 / pseudo_local_timestep[icell];

        //  const double temp_dt_inv =
        //      -pseudo_local_timestep_inv[icell] + dt_factor;
        //  for (int i = 0; i < sb * sb; i += (sb + 1)) block[i] =
        //  temp_dt_inv; pseudo_local_timestep_inv[icell] = dt_factor;

        //  MatSetValuesBlocked(sysmat_, 1, &mat_index[icell], 1,
        //                      &mat_index[icell], &block[0], ADD_VALUES);
        //}
        // MatAssemblyBegin(sysmat_, MAT_FINAL_ASSEMBLY);
        // MatAssemblyEnd(sysmat_, MAT_FINAL_ASSEMBLY);

        // Newton solve
        int sub_iteration = solver_.Solve(sysmat_, temp_rhs_, delta_);
        int converged = solver_.GetConvergedReason();  // Check convergence
        if (converged < 0) {

          MASTER_MESSAGE("GMRES diverged! : KSPConvergedReason = " +
                         std::to_string(converged) + "\n");
          if (converged == -3) {
            MASTER_MESSAGE(
                "KSPConvergedReason = KSP_DIVERGED_ITS - Ran out of "
                "iterations before any convergence criteria was reached\n");
            MASTER_MESSAGE("Trying again with double search direction...");
            solver_.DoubleSearchDirection();
            sub_iteration = solver_.Solve(sysmat_, temp_rhs_, delta_);
            solver_.ResetSearchDirection();
            converged = solver_.GetConvergedReason();  // Check convergence
            if (converged < 0) {
              MASTER_MESSAGE("Fail\n");
              is_stop = true;
              break;
            } else {
              MASTER_MESSAGE("Success\n");
            }
          } else {
            is_stop = true;
            break;
          }
        }
        sum_sub_iteration[0]++;
        sum_sub_iteration[1] += sub_iteration;
        VecAYPX(temp_delta_, -1.0, delta_);
        VecCopy(delta_, temp_delta_);

        newton_iteration++;
        jacobi_recompute_iteration++;
        // Solution update
        VecAXPY(stage_solution_, 1.0, delta_);

        prev_delta_norm = delta_norm;
        VecNorm(delta_, NORM_2, &delta_norm);
        if (newton_iteration == 1) {
          initial_delta_norm = delta_norm;
          convergence_rate = 1.0;
        } else
          convergence_rate = delta_norm / prev_delta_norm;
        relative_delta_norm = delta_norm / initial_delta_norm;

        // printmessage
        {
          std::stringstream ss;
          if (newton_iteration > 1) {
            ss << std::setw(54);
            ss << std::scientific << std::setprecision(6);
          } else {
            ss << std::setw(38);
            ss << " |  Stage=" << istage + 1;
            ss << std::scientific << std::setprecision(6);
          }
          ss << " |  Pseudo CFL="
             << pseudo_timestep_controller_->GetTimestepControlValue();
          ss << " |  Newton subiter=" << newton_iteration;
          ss << " |  GMRES subiter=" << sub_iteration;
          ss << " |  rel.delta=" << std::scientific << std::setprecision(3)
             << relative_delta_norm;
          ss << " |  abs.delta=" << std::scientific << std::setprecision(3)
             << delta_norm;
          MASTER_MESSAGE(ss.str() + "\n");
        }

        if (relative_delta_norm < newton_relative_error_tol_) break;
        if (delta_norm < newton_absolute_error_tol_) break;
      }
      sum_newton_iteration += newton_iteration;
    }

    // Solution update
    for (int istage = 0; istage < num_stages_; istage++) {
      VecGetArray(stage_rhs_[istage], &stage_rhs_ptr);
      cblas_daxpy(length_, -dt * coeff_b_[istage], stage_rhs_ptr, 1,
                  &solution_[0], 1);
      VecRestoreArray(stage_rhs_[istage], &stage_rhs_ptr);
    }

    DENEB_LIMITER->Limiting(&solution_[0]);
    DENEB_PRESSUREFIX->Execute(&solution_[0]);
    DENEB_ARTIFICIAL_VISCOSITY->ComputeArtificialViscosity(&solution_[0], 0.0);

    // Updating time and iteration
    current_time_ += time_step;
    iteration_++;

    // Measuring computing cost
    const double cost = STOP_TIMER();
    computing_cost_ += cost;

    // Printing message
    {
      std::stringstream ss;
      ss << "Iter=" << std::setw(2) << iteration_;
      ss << " |  PhyT=" << std::scientific << std::setprecision(6)
         << GetCurrentTime();
      ss << " |  ComT=" << std::fixed << std::setprecision(2)
         << computing_cost_;
      ss << " |  ComT/iter=" << std::scientific << std::setprecision(4) << cost;
      ss << " |  Newton avg subiter="
         << static_cast<double>(sum_newton_iteration) /
                static_cast<double>(num_stages_ - 1);
      ss << " |  GMRES avg subiter="
         << static_cast<double>(sum_sub_iteration[1]) /
                static_cast<double>(sum_sub_iteration[0]);
      ss << " |  rel.delta=" << std::scientific << std::setprecision(3)
         << relative_delta_norm;
      ss << " |  abs.delta=" << std::scientific << std::setprecision(3)
         << delta_norm;
      MASTER_MESSAGE(ss.str() + "\n");
    }

    // for debug
    // print log
    if (MYRANK == MASTER_NODE) {
      std::vector<double> point_solution;
      DENEB_EQUATION->GetPointPostSolution(&solution_[0], point_solution);

      history.open(filename, std::ios::app);
      history << std::scientific << std::setprecision(5);
      history << GetCurrentTime() << "\t";
      history << computing_cost_ << "\t";
      history << point_solution[0] << "\t"; // N2
      history << point_solution[1] << "\t"; // O2
      history << point_solution[2] << "\t"; // NO
      history << point_solution[3] << "\t"; // N
      history << point_solution[4] << "\t"; // O
      history << point_solution[5] << "\t"; // T_tr
      history << point_solution[6] << "\t"; // T_eev
      history << point_solution[7] << "\n"; // rho
      history << point_solution[8] << "\n";  // p
      history << point_solution[9] << "\n";  // e
      history << point_solution[10] << "\n";  // e_eev
      history.close();
    }
    // end for debug

    // Checking interruption
    is_stop = is_stop || stop_.CheckIterationFinish(iteration_);
    is_post = is_post || post_.CheckIterationEvent(iteration_);
    is_save = is_save || save_.CheckIterationEvent(iteration_);

    if (is_post) {
      DENEB_CONTOUR->FaceSolution(dir + RETURN_POST_DIR + "Face/Iter" +
                                      std::to_string(iteration_) + ".plt",
                                  &solution_[0], GetCurrentTime());
      DENEB_CONTOUR->CellSolution(
          dir + RETURN_POST_DIR + "Iter" + std::to_string(iteration_) + ".plt",
          &solution_[0], GetCurrentTime());
    }
    if (is_save) {
      SaveLoad::SaveData data;
      data.iteration_ = iteration_;
      data.strandid_ = DENEB_CONTOUR->GetStrandID();
      data.time_ = current_time_;
      DENEB_SAVELOAD->Save(
          dir + RETURN_SAVE_DIR + "Iter" + std::to_string(iteration_) + ".SAVE",
          &solution_[0], data);
    }
    if (is_stop) break;
  }
  MASTER_MESSAGE("Computing cost: " + std::to_string(computing_cost_) + "\n");
  DENEB_CONTOUR->FaceSolution(
      dir + RETURN_POST_DIR + "Face/Iter" + std::to_string(iteration_) + ".plt",
      &solution_[0], GetCurrentTime());
  DENEB_CONTOUR->CellSolution(
      dir + RETURN_POST_DIR + "Iter" + std::to_string(iteration_) + ".plt",
      &solution_[0], GetCurrentTime());

  // Compute error
  std::vector<double> exact_solution(length_);
  DENEB_EQUATION->ComputeInitialSolution(&exact_solution[0], 0.0);
  DENEB_UTILITY->ComputeError(dir + RETURN_POST_DIR + "Utility/Error.dat",
                              &solution_[0], &exact_solution[0]);
}
// ------------------------ SDIRK21 ------------------------ //
// 2nd order, 1 stage
TimeschemeSDIRK21::TimeschemeSDIRK21() : TimeschemeSDIRK() {
  MASTER_MESSAGE(avocado::GetTitle("TimeschemeSDIRK21"));
  MASTER_MESSAGE("Implicit = " + std::string(is_implicit_ ? "true" : "false") +
                 "\n");

  // time order and the number of stages
  time_order_ = 2;
  num_stages_ = 1;

  coeff_a_.resize(num_stages_);
  for (int istage = 0; istage < num_stages_; istage++)
    coeff_a_[istage].resize(num_stages_, 0.0);
  coeff_b_.resize(num_stages_, 0.0);
  coeff_c_.resize(num_stages_, 0.0);

  coeff_a_[0][0] = 0.5;
  coeff_b_[0] = 1.0;
}
TimeschemeSDIRK21::~TimeschemeSDIRK21(){};
void TimeschemeSDIRK21::BuildData(void) {
  MASTER_MESSAGE(avocado::GetTitle("TimeschemeSDIRK21::BuildData()"));
  TimeschemeSDIRK::BuildData();
}
void TimeschemeSDIRK21::Marching(void) {
  MASTER_MESSAGE(avocado::GetTitle("TimeschemeSDIRK21::Marching()"));
  TimeschemeSDIRK::Marching();
}
// ------------------------ SDIRK22 ------------------------ //
// 2nd order, 2 stage
TimeschemeSDIRK22::TimeschemeSDIRK22() : TimeschemeSDIRK() {
  MASTER_MESSAGE(avocado::GetTitle("TimeschemeSDIRK22"));
  MASTER_MESSAGE("Implicit = " + std::string(is_implicit_ ? "true" : "false") +
                 "\n");

  // time order and the number of stages
  time_order_ = 2;
  num_stages_ = 2;

  coeff_a_.resize(num_stages_);
  for (int istage = 0; istage < num_stages_; istage++)
    coeff_a_[istage].resize(num_stages_, 0.0);
  coeff_b_.resize(num_stages_, 0.0);
  coeff_c_.resize(num_stages_, 0.0);

  coeff_a_[0][0] = 0.25;
  coeff_a_[1][0] = 0.5;
  coeff_a_[1][1] = 0.25;
  coeff_b_[0] = 0.5;
  coeff_b_[1] = 0.5;
}
TimeschemeSDIRK22::~TimeschemeSDIRK22(){};
void TimeschemeSDIRK22::BuildData(void) {
  MASTER_MESSAGE(avocado::GetTitle("TimeschemeSDIRK22::BuildData()"));
  TimeschemeSDIRK::BuildData();
}
void TimeschemeSDIRK22::Marching(void) {
  MASTER_MESSAGE(avocado::GetTitle("TimeschemeSDIRK22::Marching()"));
  TimeschemeSDIRK::Marching();
}
// ------------------------ SDIRK32 ------------------------ //
// 3rd order, 2 stage
TimeschemeSDIRK32::TimeschemeSDIRK32() : TimeschemeSDIRK() {
  MASTER_MESSAGE(avocado::GetTitle("TimeschemeSDIRK32"));
  MASTER_MESSAGE("Implicit = " + std::string(is_implicit_ ? "true" : "false") +
                 "\n");

  // time order and the number of stages
  time_order_ = 3;
  num_stages_ = 2;

  coeff_a_.resize(num_stages_);
  for (int istage = 0; istage < num_stages_; istage++)
    coeff_a_[istage].resize(num_stages_, 0.0);
  coeff_b_.resize(num_stages_, 0.0);
  coeff_c_.resize(num_stages_, 0.0);

  coeff_a_[0][0] = (3.0 - std::sqrt(3.0)) / 6.0;
  coeff_a_[1][0] = 1.0 / std::sqrt(3.0);
  coeff_a_[1][1] = (3.0 - std::sqrt(3.0)) / 6.0;
  coeff_b_[0] = 0.5;
  coeff_b_[1] = 0.5;
}
TimeschemeSDIRK32::~TimeschemeSDIRK32(){};
void TimeschemeSDIRK32::BuildData(void) {
  MASTER_MESSAGE(avocado::GetTitle("TimeschemeSDIRK32::BuildData()"));
  TimeschemeSDIRK::BuildData();
}
void TimeschemeSDIRK32::Marching(void) {
  MASTER_MESSAGE(avocado::GetTitle("TimeschemeSDIRK32::Marching()"));
  TimeschemeSDIRK::Marching();
}
// ------------------------ SDIRK43 ------------------------ //
// 4th order, 3 stage
TimeschemeSDIRK43::TimeschemeSDIRK43() : TimeschemeSDIRK() {
  MASTER_MESSAGE(avocado::GetTitle("TimeschemeSDIRK43"));
  MASTER_MESSAGE("Implicit = " + std::string(is_implicit_ ? "true" : "false") +
                 "\n");

  // time order and the number of stages
  time_order_ = 4;
  num_stages_ = 3;

  coeff_a_.resize(num_stages_);
  for (int istage = 0; istage < num_stages_; istage++)
    coeff_a_[istage].resize(num_stages_, 0.0);
  coeff_b_.resize(num_stages_, 0.0);
  coeff_c_.resize(num_stages_, 0.0);

  const double xi = 0.128886400515;

  coeff_a_[0][0] = xi;
  coeff_a_[1][0] = 0.5 - xi;
  coeff_a_[1][1] = xi;
  coeff_a_[2][0] = 2.0 * xi;
  coeff_a_[2][1] = 1.0 - 4.0 * xi;
  coeff_a_[2][2] = xi;
  const double deno = 3.0 * std::pow(2.0 * xi - 1.0, 2.0);
  coeff_b_[0] = 1.0 / (2.0 * deno);
  coeff_b_[1] = 2.0 * (6.0 * xi * xi - 6.0 * xi + 1.0) / deno;  
  coeff_b_[2] = 1.0 / (2.0 * deno);
}
TimeschemeSDIRK43::~TimeschemeSDIRK43(){};
void TimeschemeSDIRK43::BuildData(void) {
  MASTER_MESSAGE(avocado::GetTitle("TimeschemeSDIRK43::BuildData()"));
  TimeschemeSDIRK::BuildData();
}
void TimeschemeSDIRK43::Marching(void) {
  MASTER_MESSAGE(avocado::GetTitle("TimeschemeSDIRK43::Marching()"));
  TimeschemeSDIRK::Marching();
}
}  // namespace deneb