#pragma once

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#define DENEB_ARTIFICIAL_VISCOSITY_NAME artificial_viscosity_global_ptr
#define DENEB_ARTIFICIAL_VISCOSITY deneb::DENEB_ARTIFICIAL_VISCOSITY_NAME
#define DENEB_ARTIFICIAL_VISCOSITY_INITIALIZE(name) \
  DENEB_ARTIFICIAL_VISCOSITY =                      \
      deneb::ArtificialViscosity::GetArtificialViscosity(name)
#define DENEB_ARTIFICIAL_VISCOSITY_FINALIZE() DENEB_ARTIFICIAL_VISCOSITY.reset()

namespace avocado {
class Communicate;
}

namespace deneb {
class ArtificialViscosity {
 protected:
  std::vector<double> artificial_viscosity_;

 public:
  static std::shared_ptr<ArtificialViscosity> GetArtificialViscosity(
      const std::string& name);

  ArtificialViscosity(){};
  virtual ~ArtificialViscosity(){};

  virtual void BuildData(void) = 0;
  virtual void ComputeArtificialViscosity(const double* solution,
                                          const double dt) = 0;
  virtual void ComputeArtificialViscosity(
      const double* solution, const std::vector<double>& local_dt) = 0;

  virtual void GetArtificialViscosityValues(const double* basis_values,
                                            const int num_points,
                                            double* artificial_viscosities){};

  virtual double GetArtificialViscosityValue(const int icell,
                                             const int ipoint) const = 0;
  const std::vector<double>& GetArtificialViscosity() const {
    return artificial_viscosity_;
  };
};
extern std::shared_ptr<ArtificialViscosity> DENEB_ARTIFICIAL_VISCOSITY_NAME;

class NoArtificialViscosity : public ArtificialViscosity {
 public:
  NoArtificialViscosity();
  virtual ~NoArtificialViscosity(){};

  virtual void BuildData(void);
  virtual void ComputeArtificialViscosity(const double* solution,
                                          const double dt){};
  virtual void ComputeArtificialViscosity(
      const double* solution, const std::vector<double>& local_dt){};

  virtual double GetArtificialViscosityValue(const int icell,
                                             const int ipoint) const {
    return 0.0;
  };
};

// ArtificialViscosity = Peclet, kappa
class LaplacianP0 : public ArtificialViscosity {
 protected:
  int target_state_;
  int num_bases_m1_;  // num basis of P(n-1)
  double Peclet_;
  double kappa_;
  double S0_;
  double dLmax_;

  std::shared_ptr<avocado::Communicate> communicate_;

 public:
  LaplacianP0();
  virtual ~LaplacianP0(){};

  virtual void BuildData(void);
  virtual void ComputeArtificialViscosity(const double* solution,
                                          const double dt);
  virtual void ComputeArtificialViscosity(const double* solution,
                                          const std::vector<double>& local_dt) {
    ComputeArtificialViscosity(solution, 0.0);
  };

  virtual double GetArtificialViscosityValue(const int icell,
                                             const int ipoint) const {
    if (icell >= 0)
      return artificial_viscosity_[icell];
    else
      return 0.0;
  };

 protected:
  double SmoothnessIndicator(const double* solution);

  double MaxArtificialViscosity(const double* solution,
                                const double cell_volumes,
                                const double cell_basis_value);
};


// Shock-capturing PID (SPID) = P gain, I gain, D gain, S gain
class SPID : public ArtificialViscosity {
 protected:
  enum class NODETYPE : char { NORMAL = 0, BOUNDARY = 1 };

  int target_state_;
  // SPID parameters
  double Pgain_;
  double Igain_;
  double Dgain_;
  double Sgain_;
  std::vector<double> cell_MLP_error_;
  std::vector<double> cell_BD_error_;
  std::vector<double> cell_error0_;
  std::vector<double> cell_error1_;
  std::vector<double> cell_integ_MLP_error_;  // MLP integrator
  std::vector<double> cell_integ_BD_error_;   // BD integrator
  // Ducros parameters
  bool Ducros_switch_;
  int max_num_cell_points_;
  const double ducros_nonzero_eps_ = 1.0e-12;  // Ducros nonzero epsilon
  std::vector<double> ducros_sensor_;
  std::vector<std::vector<double>> quad_basis_value_;
  std::vector<std::vector<double>> quad_basis_grad_value_;
  std::vector<std::vector<double>> quad_weights_;
  // hMLP parameters
  std::vector<NODETYPE> nodetypes_;
  std::vector<int> num_bases_list_;
  std::vector<double> foreign_solution_;
  std::vector<double> cell_average_;
  std::vector<double> vertex_min_;
  std::vector<double> vertex_max_;
  std::vector<double> foreign_cell_basis_value_;
  std::vector<std::vector<int>> node_cells_;
  std::vector<std::vector<int>> node_vertices_;
  std::vector<std::vector<double>> cell_vertex_basis_value_;
  // hMLP_BD parameters
  const double BD_nonzero_eps_ = 1.0e-12;  // Ducros nonzero epsilon
  std::vector<double> face_characteristic_length_;
  std::vector<double> cell_face_characteristic_length_;
  std::vector<std::vector<int>> cell_cells_;
  std::vector<std::vector<int>> cell_faces_;
  // Anti-windup parameters
  std::vector<double> smooth_sensor_;
  std::vector<double> integral_decay_rate_;

  std::shared_ptr<avocado::Communicate> communicate_AV_;
  std::shared_ptr<avocado::Communicate> communicate_MLP_;

 public:
  SPID();
  virtual ~SPID(){};

  virtual void BuildData(void);
  virtual void ComputeArtificialViscosity(const double* solution,
                                          const double dt);
  virtual void ComputeArtificialViscosity(const double* solution,
                                          const std::vector<double>& local_dt);
  virtual double GetArtificialViscosityValue(const int icell,
                                             const int ipoint) const {
    if (icell >= 0)
      return artificial_viscosity_[icell];
    else
      return 0.0;
  };

 protected:
  inline double ReLU(double x) { return std::max(x, 0.0); }

  void BuildData_hMLPBD();
  void ConstructCellCells();
  void ConstructNodeCells();
  void VertexMinMax(const double* solution);
  void ComputeMlpError(const double* solution);
  void ComputeBdError(const double* solution);
  void ComputeDucros(const double* solution);
  void ComputeSmoothSensor(const double* solution);
  double ComputeDilatation(const double* solution, const double* solution_grad);
  double ComputeVorticityMagnitude(const double* solution, const double* solution_grad);
};
}  // namespace deneb