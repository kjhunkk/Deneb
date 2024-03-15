#pragma once

#include <cmath>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "deneb_equation.h"

// 2-D compressible Navier-Stokes equations
// for thermo-chemical non-equilibrium flow with 2-temperature model

namespace Neq2T {
class Mixture;
}

namespace deneb {
class ProblemNS2DNeq2T;
class BoundaryNS2DNeq2T;

// ------------------------------- Constants ------------------------------- //
class ConstantsNS2DNeq2T {
 protected:
  static constexpr int D_ = 2;
  static int S_;
  static int DS_;
  static int SS_;
  static int DSS_;
  static int DDSS_;
  static constexpr double c23_ = 2.0 / 3.0;
  static constexpr double radius_eps_ = 1.0E-8;
  static int ax_;  // axi-symmetric -> need check

  static std::shared_ptr<Neq2T::Mixture> mixture_;
  static int ns_;

  static double T_eev_min_;

  int max_num_points_;
  int max_num_cell_points_;
  int max_num_face_points_;
  int max_num_bdry_points_;

 public:
  ConstantsNS2DNeq2T();
  virtual ~ConstantsNS2DNeq2T(){};

  inline int GetMaxNumBdryPoints() const { return max_num_bdry_points_; };

 protected:
  // sol: rho_1, ..., rho_ns, u, v, T_tr, T_eev
};

// ------------------------------- Equation -------------------------------- //
class EquationNS2DNeq2T : public Equation, public ConstantsNS2DNeq2T {
 private:
  std::shared_ptr<ProblemNS2DNeq2T> problem_;
  void (EquationNS2DNeq2T::*compute_numflux_)(const int num_points,
                                              std::vector<double>& flux,
                                              FACE_INPUTS);
  void (EquationNS2DNeq2T::*compute_numflux_jacobi_)(const int num_points,
                                                     FACE_JACOBI_OUTPUTS,
                                                     FACE_INPUTS);

  std::unordered_map<int, std::shared_ptr<BoundaryNS2DNeq2T>>
      boundary_registry_;
  std::vector<std::shared_ptr<BoundaryNS2DNeq2T>> boundaries_;

 public:
  EquationNS2DNeq2T(bool axis = false);
  virtual ~EquationNS2DNeq2T();

  virtual inline bool GetMassMatrixFlag(void) const { return true; }

  virtual void RegistBoundary(const std::vector<int>& bdry_tag);
  virtual void BuildData(void);
  virtual void PreProcess(const double* solution) { return; };
  virtual void ComputeRHS(const double* solution, double* rhs, const double t);
  virtual bool ComputeRHSdt(const double* solution, double* rhs_dt,
                            const double t) {
    return false;
  };
  virtual void ComputeSystemMatrix(const double* solution, Mat& sysmat,
                                   const double t);
  virtual void SystemMatrixShift(const double* solution, Mat& sysmat,
                                 const double dt, const double t);
  virtual void SystemMatrixShift(const double* solution, Mat& sysmat,
                                 const std::vector<double>& local_dt,
                                 const double t);
  virtual void GetCellPostSolution(const int icell, const int num_points,
                                   const std::vector<double>& solution,
                                   const std::vector<double>& solution_grad,
                                   std::vector<double>& post_solution);
  virtual void GetFacePostSolution(const int num_points,
                                   const std::vector<double>& solution,
                                   const std::vector<double>& solution_grad,
                                   const std::vector<double>& normal,
                                   std::vector<double>& post_solution);
  virtual void ComputeInitialSolution(double* solution, const double t);
  virtual void ComputeLocalTimestep(const double* solution,
                                    std::vector<double>& local_timestep);
  virtual void ComputePressureCoefficient(
      std::vector<double>& pressure_coefficient, const int num_points,
      const std::vector<double>& solution){};  // need check
  virtual void ComputeViscousStress(
      std::vector<double>& viscous_stress, const int num_points,
      const std::vector<double>& solution,
      const std::vector<double>& solution_grad){};  // need check
  virtual bool IsContact(const int& icell,
                         const std::vector<int>& neighbor_cells,
                         const double* solution,
                         const double* total_solution) const {
    return false;
  };  // need check
  virtual double ComputeMaxCharacteristicSpeed(
      const double* input_solution) const;  // need check
  virtual const std::vector<double>& ComputePressureFixValues(
      const double* input_solution) {
    return std::vector<double>();
  };  // need check

  void SolutionLimit(double* solution, double* delta){};  // necessary?

  void ComputeComFlux(const int num_points, std::vector<double>& flux,
                      const int icell, const std::vector<double>& owner_u,
                      const std::vector<double>& owner_div_u);
  void ComputeComFluxJacobi(const int num_points,
                            std::vector<double>& flux_jacobi,
                            std::vector<double>& flux_grad_jacobi,
                            const int icell, const std::vector<double>& owner_u,
                            const std::vector<double>& owner_div_u);
  // source : ps
  void ComputeSource(const int num_points, std::vector<double>& source,
                     const int icell, const std::vector<double>& owner_u,
                     const std::vector<double>& owner_div_u);
  void ComputeSourceJacobi(const int num_points,
                           std::vector<double>& source_jacobi, const int icell,
                           const std::vector<double>& owner_u,
                           const std::vector<double>& owner_div_u);
  void ComputeSolutionJacobi(const int num_points, std::vector<double>& jacobi,
                             const int icell,
                             const std::vector<double>& owner_u);
  inline void ComputeNumFlux(const int num_points, std::vector<double>& flux,
                             FACE_INPUTS) {
    (this->*compute_numflux_)(num_points, flux, owner_cell, neighbor_cell,
                              owner_u, owner_div_u, neighbor_u, neighbor_div_u,
                              normal);
  };
  inline void ComputeNumFluxJacobi(const int num_points, FACE_JACOBI_OUTPUTS,
                                   FACE_INPUTS) {
    (this->*compute_numflux_jacobi_)(
        num_points, flux_owner_jacobi, flux_neighbor_jacobi,
        flux_owner_grad_jacobi, flux_neighbor_grad_jacobi, owner_cell,
        neighbor_cell, owner_u, owner_div_u, neighbor_u, neighbor_div_u,
        normal);
  };

  DEFINE_FLUX(LLF);
};

class BoundaryNS2DNeq2T : public ConstantsNS2DNeq2T {
 public:
  static std::shared_ptr<BoundaryNS2DNeq2T> GetBoundary(
      const std::string& type, const int bdry_tag, EquationNS2DNeq2T* equation);

 protected:
  int bdry_tag_;
  EquationNS2DNeq2T* equation_;

 public:
  BoundaryNS2DNeq2T(const int bdry_tag, EquationNS2DNeq2T* equation)
      : bdry_tag_(bdry_tag), equation_(equation){};
  virtual ~BoundaryNS2DNeq2T(){};

  virtual void ComputeBdrySolution(
      const int num_points, std::vector<double>& bdry_u,
      std::vector<double>& bdry_div_u, const std::vector<double>& owner_u,
      const std::vector<double>& owner_div_u, const std::vector<double>& normal,
      const std::vector<double>& coords, const double& time) = 0;
  virtual void ComputeBdryFlux(const int num_points, std::vector<double>& flux,
                               FACE_INPUTS, const std::vector<double>& coords,
                               const double& time) = 0;
  virtual void ComputeBdrySolutionJacobi(const int num_points,
                                         double* bdry_u_jacobi,
                                         const std::vector<double>& owner_u,
                                         const std::vector<double>& owner_div_u,
                                         const std::vector<double>& normal,
                                         const std::vector<double>& coords,
                                         const double& time) = 0;
  virtual void ComputeBdryFluxJacobi(const int num_points,
                                     std::vector<double>& flux_jacobi,
                                     std::vector<double>& flux_grad_jacobi,
                                     FACE_INPUTS,
                                     const std::vector<double>& coords,
                                     const double& time) = 0;
};
// Boundary Name = SlilpWall
// Dependency: -
class SlipWallNS2DNeq2T : public BoundaryNS2DNeq2T {
 public:
  SlipWallNS2DNeq2T(const int bdry_tag, EquationNS2DNeq2T* equation);
  virtual ~SlipWallNS2DNeq2T(){};

  BOUNDARY_METHODS;
};

// Boundary Name = IsothermalWall
// Dependency: Twall
class IsothermalWallNS2DNeq2T : public BoundaryNS2DNeq2T {
 private:
  double Twall_;

 public:
  IsothermalWallNS2DNeq2T(const int bdry_tag, EquationNS2DNeq2T* equation);
  virtual ~IsothermalWallNS2DNeq2T(){};

  BOUNDARY_METHODS;
};

// Boundary Name = SupersonicInflow
// Dependency: rho_1, ..., rho_ns, u, v, T_tr, T_eev
class SupersonicInflowBdryNS2DNeq2T : public BoundaryNS2DNeq2T {
 private:
  std::vector<double> d_;
  double mixture_d_;
  double u_;
  double v_;
  double T_tr_;
  double T_eev_;
  double mixture_p_;
  double du_;
  double dv_;
  double dE_;
  double de_eev_;

 public:
  SupersonicInflowBdryNS2DNeq2T(const int bdry_tag,
                                EquationNS2DNeq2T* equation);
  virtual ~SupersonicInflowBdryNS2DNeq2T(){};

  BOUNDARY_METHODS;
};

// Boundary Name = SupersonicOutflow
// Dependency: -
class SupersonicOutflowBdryNS2DNeq2T : public BoundaryNS2DNeq2T {
 public:
  SupersonicOutflowBdryNS2DNeq2T(const int bdry_tag,
                                 EquationNS2DNeq2T* equation);
  virtual ~SupersonicOutflowBdryNS2DNeq2T(){};

  BOUNDARY_METHODS;
};

// Boundary Name = CatalyticWall
// Dependency: Twall, gamma_N, gamma_O, maxiter (optional)
class CatalyticWallNS2DNeq2T : public BoundaryNS2DNeq2T {
 private:
  bool isN_;
  int idxN_;
  int idxN2_;
  double KwN_;

  bool isO_;
  int idxO_;
  int idxO2_;
  double KwO_;

  double gamma_N_;
  double gamma_O_;
  double Twall_;
  int maxiter_;

  std::vector<double> Dwall_;
  std::vector<double> Ywall_;

 public:
  CatalyticWallNS2DNeq2T(const int bdry_tag, EquationNS2DNeq2T* equation);
  virtual ~CatalyticWallNS2DNeq2T(){};

  BOUNDARY_METHODS;
};

// Boundary Name = SuperCatalyticWall
// Dependency: Twall, rho_1, ..., rho_ns
class SuperCatalyticWallNS2DNeq2T : public BoundaryNS2DNeq2T {
 private:
  double Twall_;
  std::vector<double> Ywall_;

 public:
  SuperCatalyticWallNS2DNeq2T(const int bdry_tag, EquationNS2DNeq2T* equation);
  virtual ~SuperCatalyticWallNS2DNeq2T(){};

  BOUNDARY_METHODS;
};

// -------------------------------- Problem -------------------------------- //
class ProblemNS2DNeq2T : public ConstantsNS2DNeq2T {
 public:
  static std::shared_ptr<ProblemNS2DNeq2T> GetProblem(const std::string& name);

  ProblemNS2DNeq2T() : ConstantsNS2DNeq2T(){};
  virtual ~ProblemNS2DNeq2T(){};

  virtual void Problem(const int num_points, std::vector<double>& solutions,
                       const std::vector<double>& coord,
                       const double time = 0.0) const = 0;
};

// Problem = Freestream
// ProblemInput = rho_1, ..., rho_ns, u, v, T_tr, T_eev
class FreeStreamNS2DNeq2T : public ProblemNS2DNeq2T {
 private:
  std::vector<double> d_;
  double u_;
  double v_;
  double T_tr_;
  double T_eev_;

 public:
  FreeStreamNS2DNeq2T();
  virtual ~FreeStreamNS2DNeq2T(){};

  virtual void Problem(const int num_points, std::vector<double>& solutions,
                       const std::vector<double>& coord,
                       const double time = 0.0) const;
};
}  // namespace deneb
