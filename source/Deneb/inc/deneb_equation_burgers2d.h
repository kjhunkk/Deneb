#pragma once

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "deneb_equation.h"

// 2-D burgers equation

namespace deneb {
class ProblemBurgers2D;
class BoundaryBurgers2D;

// ------------------------------- Constants ------------------------------- //
class ConstantsBurgers2D {
 protected:
  static constexpr int D_ = 2;
  static constexpr int S_ = 1;
  static constexpr int DS_ = D_ * S_;
  static constexpr int SS_ = S_ * S_;
  static constexpr int DSS_ = DS_ * S_;
  static constexpr int DDSS_ = D_ * DSS_;

  int max_num_points_;
  int max_num_cell_points_;
  int max_num_face_points_;
  int max_num_bdry_points_;

 public:
  ConstantsBurgers2D(){};
  virtual ~ConstantsBurgers2D(){};

  inline int GetMaxNumBdryPoints() const { return max_num_bdry_points_; };
};

// ------------------------------- Equation -------------------------------- //
class EquationBurgers2D : public Equation, public ConstantsBurgers2D {
 private:
  std::shared_ptr<ProblemBurgers2D> problem_;
  void (EquationBurgers2D::*compute_numflux_)(const int num_points,
                                              std::vector<double>& flux,
                                              FACE_INPUTS);
  void (EquationBurgers2D::*compute_numflux_jacobi_)(const int num_points,
                                                     FACE_JACOBI_OUTPUTS,
                                                     FACE_INPUTS);

  std::unordered_map<int, std::shared_ptr<BoundaryBurgers2D>>
      boundary_registry_;
  std::vector<std::shared_ptr<BoundaryBurgers2D>> boundaries_;

 public:
  EquationBurgers2D();
  virtual ~EquationBurgers2D();

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
      const std::vector<double>& solution){};
  virtual void ComputeViscousStress(std::vector<double>& viscous_stress,
                                    const int num_points,
                                    const std::vector<double>& solution,
                                    const std::vector<double>& solution_grad){};
  virtual bool IsContact(const int& icell,
                         const std::vector<int>& neighbor_cells,
                         const double* solution,
                         const double* total_solution) const {
    return false;
  };
  virtual double ComputeMaxCharacteristicSpeed(
      const double* input_solution) const {
    return std::sqrt(2) * input_solution[0];
  };
  virtual const std::vector<double>& ComputePressureFixValues(
      const double* input_solution) {
    return std::vector<double>();
  };

  void ComputeComFlux(const int num_points, std::vector<double>& flux,
                      const int icell, const std::vector<double>& owner_u,
                      const std::vector<double>& owner_div_u);
  void ComputeComFluxJacobi(const int num_points,
                            std::vector<double>& flux_jacobi,
                            std::vector<double>& flux_grad_jacobi,
                            const int icell, const std::vector<double>& owner_u,
                            const std::vector<double>& owner_div_u);
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

// ------------------------------- Boundary -------------------------------- //
class BoundaryBurgers2D : public ConstantsBurgers2D {
 public:
  static std::shared_ptr<BoundaryBurgers2D> GetBoundary(
      const std::string& type, const int bdry_tag, EquationBurgers2D* equation);

 protected:
  int bdry_tag_;
  EquationBurgers2D* equation_;

 public:
  BoundaryBurgers2D(const int bdry_tag, EquationBurgers2D* equation)
      : bdry_tag_(bdry_tag), equation_(equation){};
  virtual ~BoundaryBurgers2D(){};

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

// -------------------------------- Problem -------------------------------- //
class ProblemBurgers2D : public ConstantsBurgers2D {
 public:
  static std::shared_ptr<ProblemBurgers2D> GetProblem(const std::string& name);

  ProblemBurgers2D() : ConstantsBurgers2D(){};
  virtual ~ProblemBurgers2D(){};

  virtual void Problem(const int num_points, std::vector<double>& solutions,
                       const std::vector<double>& coord,
                       const double time = 0.0) const = 0;
};
// Problem = DoubleSine
// ProblemInput = -
class DoubleSineBurgers2D : public ProblemBurgers2D {
 private:
  std::vector<double> wave_number_;

 public:
  DoubleSineBurgers2D();
  virtual ~DoubleSineBurgers2D(){};

  virtual void Problem(const int num_points, std::vector<double>& solutions,
                       const std::vector<double>& coord,
                       const double time = 0.0) const;
};
}  // namespace deneb