#include "deneb_equation_ns2dneq2t_nondim.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <unordered_set>

#include "avocado.h"
#include "deneb_artificial_viscosity.h"
#include "deneb_config_macro.h"
#include "deneb_data.h"
#include "deneb_timescheme.h"
#include "neq2t.h"

#define GET_NORMAL_PD(data)     \
  ind = D_ * ipoint;            \
  const auto& nx = data[ind++]; \
  const auto& ny = data[ind]

#define GET_SOLUTION_PS(tag, data)                                    \
  ind = S_ * ipoint;                                                  \
  const auto* d##tag = &data[ind];                                    \
  ind += ns_;                                                         \
  const auto& u##tag = data[ind++];                                   \
  const auto& v##tag = data[ind++];                                   \
  const auto& T_tr##tag = data[ind++];                                \
  const auto& T_eev##tag = data[ind];                                 \
  for (int i = 0; i < ns_; i++) dim_d##tag[i] = d##tag[i] * rho_ref_; \
  const auto& dim_T_tr##tag = T_tr##tag * T_ref_;                     \
  const auto& dim_T_eev##tag = T_eev##tag * T_ref_  \

#define GET_SOLUTION_GRAD_PDS(tag, data) \
  ind = DS_ * ipoint;                    \
  const auto* dx##tag = &data[ind];      \
  ind += ns_;                            \
  const auto& ux##tag = data[ind++];     \
  const auto& vx##tag = data[ind++];     \
  const auto& Ttrx##tag = data[ind++];   \
  const auto& Teevx##tag = data[ind++];  \
  const auto* dy##tag = &data[ind];      \
  ind += ns_;                            \
  const auto& uy##tag = data[ind++];     \
  const auto& vy##tag = data[ind++];     \
  const auto& Ttry##tag = data[ind++];   \
  const auto& Teevy##tag = data[ind]


#define GET_CONS_SOLUTION_GRAD_PDS(tag, data)  \
  ind = DS_ * ipoint;                     \
  ind += ns_;                             \
  const auto& dux##tag = data[ind++];     \
  const auto& dvx##tag = data[ind++];     \
  const auto& dEx##tag = data[ind++];     \
  const auto& de_eevx##tag = data[ind++]; \
  ind += ns_;                             \
  const auto& duy##tag = data[ind++];     \
  const auto& dvy##tag = data[ind++];     \
  const auto& dEy##tag = data[ind++];     \
  const auto& de_eevy##tag = data[ind]

namespace deneb {
// ------------------------------- Constants ------------------------------- //
int ConstantsNS2DNeq2Tnondim::S_ = 0;
int ConstantsNS2DNeq2Tnondim::DS_ = 0;
int ConstantsNS2DNeq2Tnondim::SS_ = 0;
int ConstantsNS2DNeq2Tnondim::DSS_ = 0;
int ConstantsNS2DNeq2Tnondim::DDSS_ = 0;
int ConstantsNS2DNeq2Tnondim::ax_ = 0;
std::shared_ptr<Neq2T::Mixture> ConstantsNS2DNeq2Tnondim::mixture_ = NULL;
int ConstantsNS2DNeq2Tnondim::ns_ = 0;
double ConstantsNS2DNeq2Tnondim::T_eev_min_ = 0.0;
double ConstantsNS2DNeq2Tnondim::L_ref_ = 0.0;
double ConstantsNS2DNeq2Tnondim::rho_ref_ = 0.0;
double ConstantsNS2DNeq2Tnondim::v_ref_ = 0.0;
double ConstantsNS2DNeq2Tnondim::e_ref_ = 0.0;
double ConstantsNS2DNeq2Tnondim::T_ref_ = 0.0;
double ConstantsNS2DNeq2Tnondim::p_ref_ = 0.0;
double ConstantsNS2DNeq2Tnondim::D_ref_ = 0.0;
double ConstantsNS2DNeq2Tnondim::mu_ref_ = 0.0;
double ConstantsNS2DNeq2Tnondim::k_ref_ = 0.0;

ConstantsNS2DNeq2Tnondim::ConstantsNS2DNeq2Tnondim() {
  if (mixture_ == NULL) {
    auto& config = AVOCADO_CONFIG;
    const std::string mixture_filename = config->GetConfigValue("MixtureFile");
    mixture_ = std::make_shared<Neq2T::Mixture>();
    mixture_->InitMixture(mixture_filename, "mixture", MYRANK == MASTER_NODE);
    ns_ = mixture_->GetNumSpecies();
    S_ = ns_ + D_ + 2;
    DS_ = D_ * S_;
    SS_ = S_ * S_;
    DSS_ = D_ * SS_;
    DDSS_ = D_ * DSS_;

    if (config->IsConfigValue("MinimumTemperature"))
      T_eev_min_ = std::stod(config->GetConfigValue("MinimumTemperature"));
    else
      T_eev_min_ = 0;

    // for debug
    L_ref_ = std::stod(config->GetConfigValue("L_ref"));
    rho_ref_ = std::stod(config->GetConfigValue("rho_ref"));
    T_ref_ = std::stod(config->GetConfigValue("T_ref"));
    v_ref_ = std::stod(config->GetConfigValue("v_ref"));
    p_ref_ = rho_ref_ * v_ref_ * v_ref_;
    e_ref_ = v_ref_ * v_ref_;
    mu_ref_ = rho_ref_ * v_ref_ * L_ref_;
    k_ref_ = rho_ref_ * v_ref_ * e_ref_ * L_ref_ / T_ref_;
    D_ref_ = v_ref_ * L_ref_;
  }

  max_num_points_ = 0;
  max_num_cell_points_ = 0;
  max_num_face_points_ = 0;
  max_num_bdry_points_ = 0;
}
void ConstantsNS2DNeq2Tnondim::gradpri2gradsol(const int num_points,
  const double* sol_jacobi,
  const double* grad_pri,
  double* grad_sol) const {
  for (int ipoint = 0; ipoint < num_points; ipoint++) {
    gemvAx(1.0, &sol_jacobi[ipoint * SS_], S_, &grad_pri[ipoint * S_ * 2], 1,
      0.0, &grad_sol[ipoint * S_ * 2], 1, S_, S_);
    gemvAx(1.0, &sol_jacobi[ipoint * SS_], S_, &grad_pri[ipoint * S_ * 2 + S_],
      1, 0.0, &grad_sol[ipoint * S_ * 2 + S_], 1, S_, S_);
  }
}
// ------------------------------- Equation -------------------------------- //
EquationNS2DNeq2Tnondim::EquationNS2DNeq2Tnondim(bool axis)
    : ConstantsNS2DNeq2Tnondim(), Equation(D_, S_, true) {
  if (axis) {
    ax_ = 1;
    MASTER_MESSAGE(
        avocado::GetTitle("EquationNS2DNeq2Tnondim (Axi-symmetric)"));
  } else {
    MASTER_MESSAGE(avocado::GetTitle("EquationNS2DNeq2Tnondim"));
  }
  num_species_ = ns_;

  MASTER_MESSAGE("Dimension = " + std::to_string(D_) + "\n");
  MASTER_MESSAGE("Number of state variables = " + std::to_string(S_) + "\n");
  MASTER_MESSAGE("Number of species = " + std::to_string(num_species_) + "\n");
  MASTER_MESSAGE(
      "Source term = " + std::string(source_term_ ? "true" : "false") + "\n");

  num_states_ = S_;
  dimension_ = D_;

  auto& config = AVOCADO_CONFIG;
  problem_ =
      ProblemNS2DNeq2Tnondim::GetProblem(config->GetConfigValue(PROBLEM));
  const std::string& numflux = config->GetConfigValue(CONVECTIVE_FLUX);
  if (!numflux.compare("LLF")) {
    ASSIGN_FLUX(EquationNS2DNeq2Tnondim, LLF);
  } else
    ERROR_MESSAGE("Wrong numerical flux (no-exist):" + numflux + "\n");
  MASTER_MESSAGE("Problem: " + config->GetConfigValue(PROBLEM) + "\n");
  MASTER_MESSAGE("Convective flux: " + numflux + "\n");
}
EquationNS2DNeq2Tnondim::~EquationNS2DNeq2Tnondim() {
  problem_.reset();
  boundaries_.clear();
  boundary_registry_.clear();
}
void EquationNS2DNeq2Tnondim::RegistBoundary(const std::vector<int>& bdry_tag) {
  auto& config = AVOCADO_CONFIG;
  std::vector<int> all_bdry_tag;
  {
    std::unordered_set<int> temp(bdry_tag.begin(), bdry_tag.end());
    std::vector<int> bdry_tag_new(temp.begin(), temp.end());
    temp.clear();

    std::vector<std::vector<int>> send_data(NDOMAIN, bdry_tag_new);
    bdry_tag_new.clear();
    std::vector<std::vector<int>> recv_data;
    AVOCADO_MPI->CommunicateData(send_data, recv_data);
    send_data.clear();

    for (auto&& data : recv_data) temp.insert(data.begin(), data.end());
    recv_data.clear();
    all_bdry_tag = std::vector<int>(temp.begin(), temp.end());
  }

  for (auto&& tag : all_bdry_tag) {
    const std::string& bdry_type = config->GetConfigValue(BDRY_TYPE(tag));
    if (boundary_registry_.find(tag) == boundary_registry_.end())
      boundary_registry_[tag] =
          BoundaryNS2DNeq2Tnondim::GetBoundary(bdry_type, tag, this);
  }
}
void EquationNS2DNeq2Tnondim::BuildData(void) {
  MASTER_MESSAGE(avocado::GetTitle("EquationNS2DNeq2Tnondim::BuildData()"));

  const int& num_bdries = DENEB_DATA->GetNumBdries();
  boundaries_.resize(num_bdries, nullptr);
  {
    const std::vector<int>& bdry_tag = DENEB_DATA->GetBdryTag();
    RegistBoundary(bdry_tag);
    for (int ibdry = 0; ibdry < num_bdries; ibdry++)
      boundaries_[ibdry] = boundary_registry_[bdry_tag[ibdry]];
  }

  const int& num_bases = DENEB_DATA->GetNumBases();

  max_num_points_ = 0;
  max_num_cell_points_ = 0;
  max_num_face_points_ = 0;
  max_num_bdry_points_ = 0;
  const std::vector<int>& num_cell_points = DENEB_DATA->GetNumCellPoints();
  const std::vector<int>& num_face_points = DENEB_DATA->GetNumFacePoints();
  const std::vector<int>& num_bdry_points = DENEB_DATA->GetNumBdryPoints();
  if (num_cell_points.size() != 0)
    max_num_cell_points_ =
        *std::max_element(num_cell_points.begin(), num_cell_points.end());
  if (num_face_points.size() != 0)
    max_num_face_points_ =
        *std::max_element(num_face_points.begin(), num_face_points.end());
  if (num_bdry_points.size() != 0)
    max_num_bdry_points_ =
        *std::max_element(num_bdry_points.begin(), num_bdry_points.end());
  max_num_face_points_ = std::max(max_num_bdry_points_, max_num_face_points_);
  max_num_points_ = std::max(max_num_cell_points_, max_num_face_points_);

  const int& order = DENEB_DATA->GetOrder();
  const int& num_cells = DENEB_DATA->GetNumCells();
  static const auto& cell_volumes = DENEB_DATA->GetCellVolumes();
  static const auto& cell_proj_volumes = DENEB_DATA->GetCellProjVolumes();
  dt_auxiliary_.resize(num_cells);
  for (int icell = 0; icell < num_cells; icell++)
    dt_auxiliary_[icell] =
        D_ * static_cast<double>(2 * order + 1) / cell_volumes[icell] *
        avocado::VecInnerProd(D_, &cell_proj_volumes[icell * D_],
                              &cell_proj_volumes[icell * D_]);
  auxiliary_solution_.resize(num_cells * D_ * S_ * num_bases);
  pressure_fix_values_.resize(2);  // check

  const int& num_outer_cells = DENEB_DATA->GetNumOuterCells();
  outer_solution_.resize(
      std::max((num_outer_cells - num_cells) * S_ * num_bases, 1));
  communicate_ = std::make_shared<avocado::Communicate>(
      S_ * num_bases, DENEB_DATA->GetOuterSendCellList(),
      DENEB_DATA->GetOuterRecvCellList());

  auto& config = AVOCADO_CONFIG;
  const auto& bdry_owner_cell = DENEB_DATA->GetBdryOwnerCell();
  const auto& bdry_tag = DENEB_DATA->GetBdryTag();
  for (int ibdry = 0; ibdry < num_bdries; ibdry++) {
    const int owner_cell = bdry_owner_cell[ibdry];
    const std::string& bdry_type = config->GetConfigValue(BDRY_TYPE(bdry_tag[ibdry]));
    if (bdry_type.find("Wall") != std::string::npos)
      wall_cells_.push_back(owner_cell);
  }

  std::vector<std::string> variable_names;
  for (auto& sp : mixture_->GetSpecies())
    variable_names.push_back("rho_" + sp->GetSymbol());
  variable_names.push_back("U");
  variable_names.push_back("V");
  variable_names.push_back("T_tr");
  variable_names.push_back("T_eev");

  cell_variable_names_ = variable_names;
  cell_variable_names_.push_back("rho");
  cell_variable_names_.push_back("e");
  cell_variable_names_.push_back("e_eev");
  cell_variable_names_.push_back("p");
  cell_variable_names_.push_back("SoundSpeed");
  cell_variable_names_.push_back("MachNumber");
  cell_variable_names_.push_back("mu");
  cell_variable_names_.push_back("k_tr");
  cell_variable_names_.push_back("k_eev");
  for (auto& sp : mixture_->GetSpecies())
    cell_variable_names_.push_back("Y_" + sp->GetSymbol());
  for (auto& sp : mixture_->GetSpecies())
    cell_variable_names_.push_back("w_" + sp->GetSymbol());
  cell_variable_names_.push_back("w_eev");

  face_variable_names_ = cell_variable_names_;
  face_variable_names_.push_back("dTtr/dn");
  face_variable_names_.push_back("dTeev/dn");

  cell_variable_names_.push_back("AV"); // for debug

  SYNCRO();
  {
    std::string message = "Contour variables (cell): ";
    for (auto&& name : cell_variable_names_) message += (name + ", ");
    message.pop_back();
    message.pop_back();
    MASTER_MESSAGE(message + "\n");
    message = "Contour variables (face): ";
    for (auto&& name : face_variable_names_) message += (name + ", ");
    message.pop_back();
    message.pop_back();
    MASTER_MESSAGE(message + "\n");
  }
}
void EquationNS2DNeq2Tnondim::GetCellPostSolution(
    const int icell, const int num_points, const std::vector<double>& solution,
    const std::vector<double>& solution_grad,
    std::vector<double>& post_solution) {
  int ind = 0;
  double arvis = 0.0;
   if (icell >= 0) arvis =
    DENEB_ARTIFICIAL_VISCOSITY->GetArtificialViscosityValue(icell, 0); // for debug

  std::vector<double> rho(ns_, 0.0);
  for (int ipoint = 0; ipoint < num_points; ipoint++) {
    const double* sol = &solution[ipoint * S_];

    for (int i = 0; i < ns_; i++) {
      rho[i] = sol[i] * rho_ref_;
      post_solution[ind++] = rho[i];
    }
    const double u = sol[ns_] * v_ref_;
    const double v = sol[ns_ + 1] * v_ref_;
    const double T_tr = sol[ns_ + 2] * T_ref_;
    const double T_eev = sol[ns_ + 3] * T_ref_;
    post_solution[ind++] = u;
    post_solution[ind++] = v;
    post_solution[ind++] = T_tr;
    post_solution[ind++] = T_eev;

    const auto& species = mixture_->GetSpecies();
    mixture_->SetDensity(&rho[0]);
    const double* Y = mixture_->GetMassFraction();
    const double d = mixture_->GetTotalDensity();
    const double d_inv = 1.0 / d;

    double e = 0.0;
    double e_eev = 0.0;
    for (int i = 0; i < ns_; i++) {
      const auto e_sp = species[i]->GetInternalEnergy(T_tr, T_eev);
      const auto e_eev_sp = species[i]->GetElectronicVibrationEnergy(T_eev);
      e += e_sp * Y[i];
      e_eev += e_eev_sp * Y[i];
    }

    const double beta = mixture_->GetBeta(T_tr);
    const double p = mixture_->GetPressure(T_tr, T_eev);
    const double a = std::sqrt((1.0 + beta) * p * d_inv);
    const double Ma = std::sqrt(u * u + v * v) / a;
    const double mu = mixture_->GetViscosity(T_tr, T_eev);
    const double k_tr = mixture_->GetTransRotationConductivity(T_tr, T_eev);
    const double k_eev =
        mixture_->GetElectronicVibrationConductivity(T_tr, T_eev);

    post_solution[ind++] = d;
    post_solution[ind++] = e;
    post_solution[ind++] = e_eev;
    post_solution[ind++] = p;
    post_solution[ind++] = a;
    post_solution[ind++] = Ma;
    post_solution[ind++] = mu;
    post_solution[ind++] = k_tr;
    post_solution[ind++] = k_eev;
    for (int i = 0; i < ns_; i++) post_solution[ind++] = Y[i];

    const double* rate = mixture_->GetSpeciesReactionRate(T_tr, T_eev);
    for (int i = 0; i < ns_; i++) post_solution[ind++] = rate[i];

    const double ct = mixture_->GetReactionTransferRate(T_tr, T_eev);
    const double vt = mixture_->GetVTTransferRate(T_tr, T_eev);
    const double et = mixture_->GetETTransferRate(T_tr, T_eev);
    post_solution[ind++] = ct + vt + et;

    post_solution[ind++] = arvis; // for debug
  }
}
void EquationNS2DNeq2Tnondim::GetFacePostSolution(
    const int num_points, const std::vector<double>& solution,
    const std::vector<double>& solution_grad, const std::vector<double>& normal,
    std::vector<double>& post_solution) {
  int ind = 0;
  std::vector<double> rho(ns_, 0.0);
  for (int ipoint = 0; ipoint < num_points; ipoint++) {
    const double* sol = &solution[ipoint * S_];
    const double* solgrad = &solution_grad[ipoint * DS_];

    for (int i = 0; i < ns_; i++) {
      rho[i] = sol[i] * rho_ref_;
      post_solution[ind++] = rho[i];
    }
    const double u = sol[ns_] * v_ref_;
    const double v = sol[ns_ + 1] * v_ref_;
    const double T_tr = sol[ns_ + 2] * T_ref_;
    const double T_eev = sol[ns_ + 3] * T_ref_;
    const double Ttrx = solgrad[ns_ + 2] * T_ref_ / L_ref_;
    const double Teevx = solgrad[ns_ + 3] * T_ref_ / L_ref_;
    const double Ttry = solgrad[2 * ns_ + 6] * T_ref_ / L_ref_;
    const double Teevy = solgrad[2 * ns_ + 7] * T_ref_ / L_ref_;
    const double nx = normal[ipoint * D_];
    const double ny = normal[ipoint * D_ + 1];

    post_solution[ind++] = u;
    post_solution[ind++] = v;
    post_solution[ind++] = T_tr;
    post_solution[ind++] = T_eev;

    const auto& species = mixture_->GetSpecies();
    mixture_->SetDensity(&rho[0]);
    const double* Y = mixture_->GetMassFraction();
    const double d = mixture_->GetTotalDensity();
    const double d_inv = 1.0 / d;

    double e = 0.0;
    double e_eev = 0.0;
    for (int i = 0; i < ns_; i++) {
      const auto e_sp = species[i]->GetInternalEnergy(T_tr, T_eev);
      const auto e_eev_sp = species[i]->GetElectronicVibrationEnergy(T_eev);
      e += e_sp * Y[i];
      e_eev += e_eev_sp * Y[i];
    }

    const double beta = mixture_->GetBeta(T_tr);
    const double p = mixture_->GetPressure(T_tr, T_eev);
    const double a = std::sqrt((1.0 + beta) * p * d_inv);
    const double Ma = std::sqrt(u * u + v * v) / a;
    const double mu = mixture_->GetViscosity(T_tr, T_eev);
    const double k_tr = mixture_->GetTransRotationConductivity(T_tr, T_eev);
    const double k_eev =
      mixture_->GetElectronicVibrationConductivity(T_tr, T_eev);

    post_solution[ind++] = d;
    post_solution[ind++] = e;
    post_solution[ind++] = e_eev;
    post_solution[ind++] = p;
    post_solution[ind++] = a;
    post_solution[ind++] = Ma;
    post_solution[ind++] = mu;
    post_solution[ind++] = k_tr;
    post_solution[ind++] = k_eev;
    for (int i = 0; i < ns_; i++) post_solution[ind++] = Y[i];

    const double* rate = mixture_->GetSpeciesReactionRate(T_tr, T_eev);
    for (int i = 0; i < ns_; i++) post_solution[ind++] = rate[i];

    const double ct = mixture_->GetReactionTransferRate(T_tr, T_eev);
    const double vt = mixture_->GetVTTransferRate(T_tr, T_eev);
    const double et = mixture_->GetETTransferRate(T_tr, T_eev);
    post_solution[ind++] = ct + vt + et;

    post_solution[ind++] = -(Ttrx * nx + Ttry * ny);
    post_solution[ind++] = -(Teevx * nx + Teevy * ny);
  }
}
void EquationNS2DNeq2Tnondim::ComputeInitialSolution(double* solution,
                                                     const double t) {
  const int& order = DENEB_DATA->GetOrder();
  const int& num_bases = DENEB_DATA->GetNumBases();
  const int& num_cells = DENEB_DATA->GetNumCells();
  static const int sb = S_ * num_bases;

  memset(solution, 0, num_cells * sb * sizeof(double));

  std::vector<double> quad_points;
  std::vector<double> quad_weights;
  std::vector<double> initial_values;
  std::vector<double> basis_values;

  for (int icell = 0; icell < num_cells; icell++) {
    DENEB_DATA->GetCellQuadrature(icell, 2 * order, quad_points, quad_weights);
    DENEB_DATA->GetCellBasisValues(icell, quad_points, basis_values);

    const int num_points = static_cast<int>(quad_weights.size());
    initial_values.resize(num_points * S_);
    problem_->Problem(num_points, initial_values, quad_points, t);

    // solution reconstruction
    for (int istate = 0; istate < S_; istate++)
      for (int ibasis = 0; ibasis < num_bases; ibasis++)
        for (int ipoint = 0; ipoint < num_points; ipoint++)
          solution[icell * sb + istate * num_bases + ibasis] +=
              initial_values[ipoint * S_ + istate] *
              basis_values[ipoint * num_bases + ibasis] * quad_weights[ipoint];
  }
}
void EquationNS2DNeq2Tnondim::ComputeLocalTimestep(
    const double* solution, std::vector<double>& local_timestep) {
  static const int& order = DENEB_DATA->GetOrder();
  static const int& num_bases = DENEB_DATA->GetNumBases();
  static const int sb = S_ * num_bases;

  static const int& num_cells = DENEB_DATA->GetNumCells();
  static const auto& cell_volumes = DENEB_DATA->GetCellVolumes();
  static const auto& cell_proj_volumes = DENEB_DATA->GetCellProjVolumes();
  static const auto& cell_basis_value = DENEB_DATA->GetCellBasisValue();

  static std::vector<double> d(ns_);
  static std::vector<double> dim_d(ns_);
  for (int icell = 0; icell < num_cells; icell++) {
    const double& Vx = cell_proj_volumes[icell * D_];
    const double& Vy = cell_proj_volumes[icell * D_ + 1];
    const double* sol = &solution[icell * sb];

    for (int i = 0; i < ns_; i++) {
      d[i] = sol[i * num_bases] * cell_basis_value[icell][0];
      dim_d[i] = d[i] * rho_ref_;
    }    
    const double& u = sol[ns_ * num_bases] * cell_basis_value[icell][0];
    const double& v = sol[(ns_ + 1) * num_bases] * cell_basis_value[icell][0];
    const double& T_tr =
        sol[(ns_ + 2) * num_bases] * cell_basis_value[icell][0];
    const double& T_eev =
        sol[(ns_ + 3) * num_bases] * cell_basis_value[icell][0];
    const double dim_T_tr = T_tr * T_ref_;
    const double dim_T_eev = T_eev * T_ref_;

    mixture_->SetDensity(&dim_d[0]);
    const double mixture_d = mixture_->GetTotalDensity() / rho_ref_;

    const double beta = mixture_->GetBeta(dim_T_tr);
    const double mixture_p = mixture_->GetPressure(dim_T_tr, dim_T_eev) / p_ref_;
    const double a = std::sqrt((1.0 + beta) * mixture_p / mixture_d);

    const double mu = mixture_->GetViscosity(dim_T_tr, dim_T_eev) / mu_ref_;
    const double max_visradii = 4.0 / 3.0 * mu;
    const double arvis =
        DENEB_ARTIFICIAL_VISCOSITY->GetArtificialViscosityValue(icell, 0);
    const double factor =
        (max_visradii / mixture_d + arvis) * dt_auxiliary_[icell];

    local_timestep[icell] =
        cell_volumes[icell] /
        ((std::fabs(u) + a) * Vx + (std::fabs(v) + a) * Vy + factor);

    if (!std::isnormal(local_timestep[icell]) || local_timestep[icell] <= 0.0)
      ERROR_MESSAGE("Abnormal time step detected! : local_timestep[ " +
        std::to_string(icell) +
        "] = " + std::to_string(local_timestep[icell]) + "\n");
  }
  avocado::VecScale(num_cells, 1.0 / static_cast<double>(2 * order + 1),
                    &local_timestep[0]);
}
double EquationNS2DNeq2Tnondim::ComputeMaxCharacteristicSpeed(
    const double* input_solution) const {
  const double* rho = input_solution;
  const double& u = input_solution[ns_];
  const double& v = input_solution[ns_ + 1];
  const double& T_tr = input_solution[ns_ + 2];
  const double& T_eev = input_solution[ns_ + 3];

  std::vector<double> dim_rho(ns_, 0.0);
  for (int i = 0; i < ns_; i++) dim_rho[i] = rho[i] * rho_ref_;
  const double dim_T_tr = T_tr * T_ref_;
  const double dim_T_eev = T_eev * T_ref_;

  mixture_->SetDensity(&dim_rho[0]);
  const double d = mixture_->GetTotalDensity() / rho_ref_;

  const double beta = mixture_->GetBeta(dim_T_tr);
  const double p =
      mixture_->GetPressure(dim_T_tr, dim_T_eev) / p_ref_;
  const double a = std::sqrt((1.0 + beta) * p / d);
  const double V = std::sqrt(u * u + v * v);

  return V + a;
}
void EquationNS2DNeq2Tnondim::ComputeRHS(const double* solution, double* rhs,
                                         const double t) {
  static const int& num_bases = DENEB_DATA->GetNumBases();
  static const int& num_cells = DENEB_DATA->GetNumCells();
  static const int sb = S_ * num_bases;
  static const int dsb = DS_ * num_bases;

  static std::vector<double> owner_solution(S_ * max_num_points_);
  static std::vector<double> owner_solution_grad(DS_ * max_num_points_);
  static std::vector<double> flux(DS_ * max_num_points_);
  static std::vector<double> neighbor_solution(S_ * max_num_face_points_);
  static std::vector<double> neighbor_solution_grad(DS_ * max_num_face_points_);
  static std::vector<double> solution_difference(S_ * max_num_face_points_);
  static std::vector<double> source(S_ * max_num_points_);
  static std::vector<double> local_auxiliary(dsb);

  communicate_->CommunicateBegin(solution);

  memset(&rhs[0], 0, num_cells * sb * sizeof(double));
  memset(&auxiliary_solution_[0], 0, num_cells * dsb * sizeof(double));

  // inner face sweep
  static const int& num_inner_faces = DENEB_DATA->GetNumInnerFaces();
  static const auto& num_face_points = DENEB_DATA->GetNumFacePoints();
  static const auto& face_owner_cell = DENEB_DATA->GetFaceOwnerCell();
  static const auto& face_neighbor_cell = DENEB_DATA->GetFaceNeighborCell();
  static const auto& face_normals = DENEB_DATA->GetFaceNormals();
  static const auto& face_points = DENEB_DATA->GetFacePoints();
  static const auto& face_owner_basis_value =
      DENEB_DATA->GetFaceOwnerBasisValue();
  static const auto& face_neighbor_basis_value =
      DENEB_DATA->GetFaceNeighborBasisValue();
  static const auto& face_owner_basis_grad_value =
      DENEB_DATA->GetFaceOwnerBasisGradValue();
  static const auto& face_neighbor_basis_grad_value =
      DENEB_DATA->GetFaceNeighborBasisGradValue();
  static const auto& face_owner_coefficients =
      DENEB_DATA->GetFaceOwnerCoefficients();
  static const auto& face_neighbor_coefficients =
      DENEB_DATA->GetFaceNeighborCoefficients();

  for (int iface = 0; iface < num_inner_faces; iface++) {
    const int& num_points = num_face_points[iface];
    const int& owner_cell = face_owner_cell[iface];
    const int& neighbor_cell = face_neighbor_cell[iface];

    avocado::Kernel0::f4(&solution[owner_cell * sb],
                         &face_owner_basis_value[iface][0], &owner_solution[0],
                         S_, num_bases, num_points, 1.0, 0.0);
    avocado::Kernel0::f4(
        &solution[neighbor_cell * sb], &face_neighbor_basis_value[iface][0],
        &neighbor_solution[0], S_, num_bases, num_points, 1.0, 0.0);

    vdSub(num_points * S_, &neighbor_solution[0], &owner_solution[0],
          &solution_difference[0]);

    avocado::Kernel1::f59(&face_owner_coefficients[iface][0],
                          &solution_difference[0], &local_auxiliary[0], D_,
                          num_bases, num_points, S_, 0.5, 0.0);
    avocado::Kernel1::f42(&face_owner_basis_grad_value[iface][0],
                          &solution[owner_cell * sb], &owner_solution_grad[0],
                          D_, num_points, num_bases, S_, 1.0, 0.0);
    avocado::Kernel1::f18(
        &local_auxiliary[0], &face_owner_basis_value[iface][0],
        &owner_solution_grad[0], D_, S_, num_bases, num_points, 1.0, 1.0);
    cblas_daxpy(dsb, 1.0, &local_auxiliary[0], 1,
                &auxiliary_solution_[owner_cell * dsb], 1);

    avocado::Kernel1::f59(&face_neighbor_coefficients[iface][0],
                          &solution_difference[0], &local_auxiliary[0], D_,
                          num_bases, num_points, S_, 0.5, 0.0);
    avocado::Kernel1::f42(&face_neighbor_basis_grad_value[iface][0],
                          &solution[neighbor_cell * sb],
                          &neighbor_solution_grad[0], D_, num_points, num_bases,
                          S_, 1.0, 0.0);
    avocado::Kernel1::f18(
        &local_auxiliary[0], &face_neighbor_basis_value[iface][0],
        &neighbor_solution_grad[0], D_, S_, num_bases, num_points, 1.0, 1.0);
    cblas_daxpy(dsb, 1.0, &local_auxiliary[0], 1,
                &auxiliary_solution_[neighbor_cell * dsb], 1);

    ComputeNumFlux(num_points, flux, owner_cell, neighbor_cell, owner_solution,
                   owner_solution_grad, neighbor_solution,
                   neighbor_solution_grad, face_normals[iface], face_points[iface]);

    avocado::Kernel2::f67(&flux[0], &face_owner_coefficients[iface][0],
                          &rhs[owner_cell * sb], S_, D_, num_points, num_bases,
                          1.0, 1.0);
    avocado::Kernel2::f67(&flux[0], &face_neighbor_coefficients[iface][0],
                          &rhs[neighbor_cell * sb], S_, D_, num_points,
                          num_bases, -1.0, 1.0);
  }

  // bdry sweep
  static const int& num_bdries = DENEB_DATA->GetNumBdries();
  static const auto& num_bdry_points = DENEB_DATA->GetNumBdryPoints();
  static const auto& bdry_owner_cell = DENEB_DATA->GetBdryOwnerCell();
  static const auto& bdry_normals = DENEB_DATA->GetBdryNormals();
  static const auto& bdry_points = DENEB_DATA->GetBdryPoints();
  static const auto& bdry_owner_basis_value =
      DENEB_DATA->GetBdryOwnerBasisValue();
  static const auto& bdry_owner_basis_grad_value =
      DENEB_DATA->GetBdryOwnerBasisGradValue();
  static const auto& bdry_owner_coefficients =
      DENEB_DATA->GetBdryOwnerCoefficients();
  for (int ibdry = 0; ibdry < num_bdries; ibdry++) {
    const int& num_points = num_bdry_points[ibdry];
    const int& owner_cell = bdry_owner_cell[ibdry];

    avocado::Kernel0::f4(&solution[owner_cell * sb],
                         &bdry_owner_basis_value[ibdry][0], &owner_solution[0],
                         S_, num_bases, num_points, 1.0, 0.0);
    boundaries_[ibdry]->ComputeBdrySolution(
        num_points, neighbor_solution, neighbor_solution_grad, owner_solution,
        owner_solution_grad, bdry_normals[ibdry], bdry_points[ibdry], t);

    vdSub(num_points * S_, &neighbor_solution[0], &owner_solution[0],
          &solution_difference[0]);

    //avocado::Kernel1::f59(&bdry_owner_coefficients[ibdry][0],
    //                      &solution_difference[0], &local_auxiliary[0], D_,
    //                      num_bases, num_points, S_, 1.0, 0.0);
    avocado::Kernel1::f59(&bdry_owner_coefficients[ibdry][0],
                          &solution_difference[0], &local_auxiliary[0], D_,
                          num_bases, num_points, S_, 0.5, 0.0);
    avocado::Kernel1::f42(&bdry_owner_basis_grad_value[ibdry][0],
                          &solution[owner_cell * sb], &owner_solution_grad[0],
                          D_, num_points, num_bases, S_, 1.0, 0.0);
    avocado::Kernel1::f18(
        &local_auxiliary[0], &bdry_owner_basis_value[ibdry][0],
        &owner_solution_grad[0], D_, S_, num_bases, num_points, 1.0, 1.0);
    cblas_daxpy(dsb, 1.0, &local_auxiliary[0], 1,
                &auxiliary_solution_[owner_cell * dsb], 1);

    boundaries_[ibdry]->ComputeBdryFlux(
        num_points, flux, owner_cell, -1, owner_solution, owner_solution_grad,
        neighbor_solution, neighbor_solution_grad, bdry_normals[ibdry],
        bdry_points[ibdry], t);

    avocado::Kernel2::f67(&flux[0], &bdry_owner_coefficients[ibdry][0],
                          &rhs[owner_cell * sb], S_, D_, num_points, num_bases,
                          1.0, 1.0);
  }

  communicate_->CommunicateEnd(&outer_solution_[0]);

  // outer face sweep
  static const int& num_faces = DENEB_DATA->GetNumFaces();
  for (int iface = num_inner_faces; iface < num_faces; iface++) {
    const int& num_points = num_face_points[iface];
    const int& owner_cell = face_owner_cell[iface];
    const int& neighbor_cell = face_neighbor_cell[iface];

    avocado::Kernel0::f4(&solution[owner_cell * sb],
                         &face_owner_basis_value[iface][0], &owner_solution[0],
                         S_, num_bases, num_points, 1.0, 0.0);
    avocado::Kernel0::f4(&outer_solution_[neighbor_cell * sb],
                         &face_neighbor_basis_value[iface][0],
                         &neighbor_solution[0], S_, num_bases, num_points, 1.0,
                         0.0);

    vdSub(num_points * S_, &neighbor_solution[0], &owner_solution[0],
          &solution_difference[0]);

    avocado::Kernel1::f59(&face_owner_coefficients[iface][0],
                          &solution_difference[0], &local_auxiliary[0], D_,
                          num_bases, num_points, S_, 0.5, 0.0);
    avocado::Kernel1::f42(&face_owner_basis_grad_value[iface][0],
                          &solution[owner_cell * sb], &owner_solution_grad[0],
                          D_, num_points, num_bases, S_, 1.0, 0.0);
    avocado::Kernel1::f18(
        &local_auxiliary[0], &face_owner_basis_value[iface][0],
        &owner_solution_grad[0], D_, S_, num_bases, num_points, 1.0, 1.0);
    cblas_daxpy(dsb, 1.0, &local_auxiliary[0], 1,
                &auxiliary_solution_[owner_cell * dsb], 1);

    avocado::Kernel1::f59(&face_neighbor_coefficients[iface][0],
                          &solution_difference[0], &local_auxiliary[0], D_,
                          num_bases, num_points, S_, 0.5, 0.0);
    avocado::Kernel1::f42(&face_neighbor_basis_grad_value[iface][0],
                          &outer_solution_[neighbor_cell * sb],
                          &neighbor_solution_grad[0], D_, num_points, num_bases,
                          S_, 1.0, 0.0);
    avocado::Kernel1::f18(
        &local_auxiliary[0], &face_neighbor_basis_value[iface][0],
        &neighbor_solution_grad[0], D_, S_, num_bases, num_points, 1.0, 1.0);

    ComputeNumFlux(num_points, flux, owner_cell, neighbor_cell + num_cells,
                   owner_solution, owner_solution_grad, neighbor_solution,
                   neighbor_solution_grad, face_normals[iface], face_points[iface]);

    avocado::Kernel2::f67(&flux[0], &face_owner_coefficients[iface][0],
                          &rhs[owner_cell * sb], S_, D_, num_points, num_bases,
                          1.0, 1.0);
  }

  // cell sweep
  static const auto& num_cell_points = DENEB_DATA->GetNumCellPoints();
  static const auto& cell_basis_value = DENEB_DATA->GetCellBasisValue();
  static const auto& cell_basis_grad_value =
      DENEB_DATA->GetCellBasisGradValue();
  static const auto& cell_coefficients = DENEB_DATA->GetCellCoefficients();
  static const auto& cell_source_coefficients =
      DENEB_DATA->GetCellSourceCoefficients();
  for (int icell = 0; icell < num_cells; icell++) {
    const int& num_points = num_cell_points[icell];

    avocado::Kernel0::f4(&solution[icell * sb], &cell_basis_value[icell][0],
                         &owner_solution[0], S_, num_bases, num_points, 1.0,
                         0.0);
    avocado::Kernel1::f42(&cell_basis_grad_value[icell][0],
                          &solution[icell * sb], &owner_solution_grad[0], D_,
                          num_points, num_bases, S_, 1.0, 0.0);
    avocado::Kernel1::f18(&auxiliary_solution_[icell * dsb],
                          &cell_basis_value[icell][0], &owner_solution_grad[0],
                          D_, S_, num_bases, num_points, 1.0, 1.0);

    ComputeComFlux(num_points, flux, icell, owner_solution,
                   owner_solution_grad);

    avocado::Kernel2::f67(&flux[0], &cell_coefficients[icell][0],
                          &rhs[icell * sb], S_, D_, num_points, num_bases, -1.0,
                          1.0);

    ComputeSource(num_points, source, icell, owner_solution,
                  owner_solution_grad);

    gemmATB(-1.0, &source[0], &cell_source_coefficients[icell][0], 1.0,
            &rhs[icell * sb], num_points, S_, num_bases);
  }
}
void EquationNS2DNeq2Tnondim::ComputeSystemMatrix(const double* solution,
                                                  Mat& sysmat, const double t) {
  static const int& num_bases = DENEB_DATA->GetNumBases();
  static const int& num_cells = DENEB_DATA->GetNumCells();
  static const int& num_bdries = DENEB_DATA->GetNumBdries();

  static const int db = D_ * num_bases;
  static const int sb = S_ * num_bases;
  static const int dsb = DS_ * num_bases;
  static const int dbb = db * num_bases;
  static const int ssb = SS_ * num_bases;
  static const int ssp = SS_ * max_num_bdry_points_;
  static const int dssb = DSS_ * num_bases;
  static const int ssbb = ssb * num_bases;

  static const std::vector<double> identity(S_, 1.0);

  static std::vector<double> owner_solution(S_ * max_num_points_);
  static std::vector<double> owner_solution_grad(DS_ * max_num_points_);
  static std::vector<double> flux_owner_jacobi(DSS_ * max_num_points_);
  static std::vector<double> flux_owner_grad_jacobi(DDSS_ * max_num_points_);
  static std::vector<double> owner_a(db * max_num_points_);
  static std::vector<double> owner_b(db * max_num_points_);
  static std::vector<double> largeB(dssb * max_num_points_);
  static std::vector<double> flux_derivative(dssb * max_num_points_);

  static std::vector<double> neighbor_solution(S_ * max_num_face_points_);
  static std::vector<double> neighbor_solution_grad(DS_ * max_num_face_points_);
  static std::vector<double> flux_neighbor_jacobi(DSS_ * max_num_face_points_);
  static std::vector<double> flux_neighbor_grad_jacobi(DDSS_ *
                                                       max_num_face_points_);
  static std::vector<double> solution_difference(S_ * max_num_face_points_);
  static std::vector<double> alpha(D_ * max_num_points_ * max_num_face_points_);
  static std::vector<double> largeA(std::max(ssb * max_num_face_points_, dbb));
  static std::vector<double> neighbor_a(db * max_num_face_points_);
  static std::vector<double> neighbor_b(db * max_num_face_points_);
  static std::vector<double> local_auxiliary(dsb);
  static std::vector<double> block(ssbb);
  static std::vector<double> bdry_solution_jacobi(num_bdries * ssp);
  static std::vector<double> source_jacobi(SS_ * max_num_points_);

  communicate_->CommunicateBegin(solution);

  MatZeroEntries(sysmat);
  memset(&auxiliary_solution_[0], 0, num_cells * dsb * sizeof(double));

  // inner face sweep
  static const auto& mat_index = DENEB_DATA->GetMatIndex();
  static const int& num_inner_faces = DENEB_DATA->GetNumInnerFaces();
  static const auto& num_face_points = DENEB_DATA->GetNumFacePoints();
  static const auto& face_owner_cell = DENEB_DATA->GetFaceOwnerCell();
  static const auto& face_neighbor_cell = DENEB_DATA->GetFaceNeighborCell();
  static const auto& face_normals = DENEB_DATA->GetFaceNormals();
  static const auto& face_points = DENEB_DATA->GetFacePoints();
  static const auto& face_owner_basis_value =
      DENEB_DATA->GetFaceOwnerBasisValue();
  static const auto& face_neighbor_basis_value =
      DENEB_DATA->GetFaceNeighborBasisValue();
  static const auto& face_owner_basis_grad_value =
      DENEB_DATA->GetFaceOwnerBasisGradValue();
  static const auto& face_neighbor_basis_grad_value =
      DENEB_DATA->GetFaceNeighborBasisGradValue();
  static const auto& face_owner_coefficients =
      DENEB_DATA->GetFaceOwnerCoefficients();
  static const auto& face_neighbor_coefficients =
      DENEB_DATA->GetFaceNeighborCoefficients();
  for (int iface = 0; iface < num_inner_faces; iface++) {
    const int& num_points = num_face_points[iface];
    const int& owner_cell = face_owner_cell[iface];
    const int& neighbor_cell = face_neighbor_cell[iface];

    avocado::Kernel0::f4(&solution[owner_cell * sb],
                         &face_owner_basis_value[iface][0], &owner_solution[0],
                         S_, num_bases, num_points, 1.0, 0.0);
    avocado::Kernel0::f4(
        &solution[neighbor_cell * sb], &face_neighbor_basis_value[iface][0],
        &neighbor_solution[0], S_, num_bases, num_points, 1.0, 0.0);

    vdSub(num_points * S_, &neighbor_solution[0], &owner_solution[0],
          &solution_difference[0]);

    avocado::Kernel1::f59(&face_owner_coefficients[iface][0],
                          &solution_difference[0], &local_auxiliary[0], D_,
                          num_bases, num_points, S_, 0.5, 0.0);
    avocado::Kernel1::f42(&face_owner_basis_grad_value[iface][0],
                          &solution[owner_cell * sb], &owner_solution_grad[0],
                          D_, num_points, num_bases, S_, 1.0, 0.0);
    avocado::Kernel1::f18(
        &local_auxiliary[0], &face_owner_basis_value[iface][0],
        &owner_solution_grad[0], D_, S_, num_bases, num_points, 1.0, 1.0);
    cblas_daxpy(dsb, 1.0, &local_auxiliary[0], 1,
                &auxiliary_solution_[owner_cell * dsb], 1);

    avocado::Kernel1::f59(&face_neighbor_coefficients[iface][0],
                          &solution_difference[0], &local_auxiliary[0], D_,
                          num_bases, num_points, S_, 0.5, 0.0);
    avocado::Kernel1::f42(&face_neighbor_basis_grad_value[iface][0],
                          &solution[neighbor_cell * sb],
                          &neighbor_solution_grad[0], D_, num_points, num_bases,
                          S_, 1.0, 0.0);
    avocado::Kernel1::f18(
        &local_auxiliary[0], &face_neighbor_basis_value[iface][0],
        &neighbor_solution_grad[0], D_, S_, num_bases, num_points, 1.0, 1.0);
    cblas_daxpy(dsb, 1.0, &local_auxiliary[0], 1,
                &auxiliary_solution_[neighbor_cell * dsb], 1);

    ComputeNumFluxJacobi(num_points, flux_owner_jacobi, flux_neighbor_jacobi,
      flux_owner_grad_jacobi, flux_neighbor_grad_jacobi,
      owner_cell, neighbor_cell, owner_solution,
      owner_solution_grad, neighbor_solution,
      neighbor_solution_grad, face_normals[iface], face_points[iface]);

    avocado::Kernel1::f47(&face_owner_coefficients[iface][0],
                          &face_owner_basis_value[iface][0], &alpha[0], D_,
                          num_points, num_bases, num_points, 1.0, 0.0);
    cblas_dcopy(db * num_points, &face_owner_basis_grad_value[iface][0], 1,
                &owner_a[0], 1);
    avocado::Kernel1::f50(&alpha[0], &face_owner_basis_value[iface][0],
                          &owner_a[0], D_, num_points, num_points, num_bases,
                          -0.5, 1.0);
    avocado::Kernel1::f50(&alpha[0], &face_neighbor_basis_value[iface][0],
                          &neighbor_b[0], D_, num_points, num_points, num_bases,
                          0.5, 0.0);

    avocado::Kernel1::f47(&face_neighbor_coefficients[iface][0],
                          &face_neighbor_basis_value[iface][0], &alpha[0], D_,
                          num_points, num_bases, num_points, 1.0, 0.0);
    cblas_dcopy(db * num_points, &face_neighbor_basis_grad_value[iface][0], 1,
                &neighbor_a[0], 1);
    avocado::Kernel1::f50(&alpha[0], &face_neighbor_basis_value[iface][0],
                          &neighbor_a[0], D_, num_points, num_points, num_bases,
                          0.5, 1.0);
    avocado::Kernel1::f50(&alpha[0], &face_owner_basis_value[iface][0],
                          &owner_b[0], D_, num_points, num_points, num_bases,
                          -0.5, 0.0);

    for (int ipoint = 0; ipoint < num_points; ipoint++) {
      gemmAB(1.0, &flux_owner_grad_jacobi[ipoint * DDSS_],
             &owner_a[ipoint * db], 0.0, &flux_derivative[ipoint * dssb], DSS_,
             D_, num_bases);
      gemmAB(1.0, &flux_neighbor_grad_jacobi[ipoint * DDSS_],
             &owner_b[ipoint * db], 1.0, &flux_derivative[ipoint * dssb], DSS_,
             D_, num_bases);
      cblas_dger(CBLAS_LAYOUT::CblasRowMajor, DSS_, num_bases, 1.0,
                 &flux_owner_jacobi[ipoint * DSS_], 1,
                 &face_owner_basis_value[iface][ipoint * num_bases], 1,
                 &flux_derivative[ipoint * dssb], num_bases);
    }
    avocado::Kernel1::f59(&flux_derivative[0],
                          &face_owner_coefficients[iface][0], &block[0], S_, sb,
                          num_points * D_, num_bases, 1.0, 0.0);
    MatSetValuesBlocked(sysmat, 1, &mat_index[owner_cell], 1,
                        &mat_index[owner_cell], &block[0], ADD_VALUES);
    avocado::Kernel1::f59(&flux_derivative[0],
                          &face_neighbor_coefficients[iface][0], &block[0], S_,
                          sb, num_points * D_, num_bases, -1.0, 0.0);
    MatSetValuesBlocked(sysmat, 1, &mat_index[neighbor_cell], 1,
                        &mat_index[owner_cell], &block[0], ADD_VALUES);

    for (int ipoint = 0; ipoint < num_points; ipoint++) {
      gemmAB(1.0, &flux_owner_grad_jacobi[ipoint * DDSS_],
             &neighbor_b[ipoint * db], 0.0, &flux_derivative[ipoint * dssb],
             DSS_, D_, num_bases);
      gemmAB(1.0, &flux_neighbor_grad_jacobi[ipoint * DDSS_],
             &neighbor_a[ipoint * db], 1.0, &flux_derivative[ipoint * dssb],
             DSS_, D_, num_bases);
      cblas_dger(CBLAS_LAYOUT::CblasRowMajor, DSS_, num_bases, 1.0,
                 &flux_neighbor_jacobi[ipoint * DSS_], 1,
                 &face_neighbor_basis_value[iface][ipoint * num_bases], 1,
                 &flux_derivative[ipoint * dssb], num_bases);
    }
    avocado::Kernel1::f59(&flux_derivative[0],
                          &face_owner_coefficients[iface][0], &block[0], S_, sb,
                          num_points * D_, num_bases, 1.0, 0.0);
    MatSetValuesBlocked(sysmat, 1, &mat_index[owner_cell], 1,
                        &mat_index[neighbor_cell], &block[0], ADD_VALUES);
    avocado::Kernel1::f59(&flux_derivative[0],
                          &face_neighbor_coefficients[iface][0], &block[0], S_,
                          sb, num_points * D_, num_bases, -1.0, 0.0);
    MatSetValuesBlocked(sysmat, 1, &mat_index[neighbor_cell], 1,
                        &mat_index[neighbor_cell], &block[0], ADD_VALUES);
  }

  // bdry sweep
  static const auto& num_bdry_points = DENEB_DATA->GetNumBdryPoints();
  static const auto& bdry_owner_cell = DENEB_DATA->GetBdryOwnerCell();
  static const auto& bdry_normals = DENEB_DATA->GetBdryNormals();
  static const auto& bdry_points = DENEB_DATA->GetBdryPoints();
  static const auto& bdry_owner_basis_value =
      DENEB_DATA->GetBdryOwnerBasisValue();
  static const auto& bdry_owner_basis_grad_value =
      DENEB_DATA->GetBdryOwnerBasisGradValue();
  static const auto& bdry_owner_coefficients =
      DENEB_DATA->GetBdryOwnerCoefficients();
  for (int ibdry = 0; ibdry < num_bdries; ibdry++) {
    const int& num_points = num_bdry_points[ibdry];
    const int& owner_cell = bdry_owner_cell[ibdry];

    avocado::Kernel0::f4(&solution[owner_cell * sb],
                         &bdry_owner_basis_value[ibdry][0], &owner_solution[0],
                         S_, num_bases, num_points, 1.0, 0.0);
    boundaries_[ibdry]->ComputeBdrySolution(
        num_points, neighbor_solution, neighbor_solution_grad, owner_solution,
        owner_solution_grad, bdry_normals[ibdry], bdry_points[ibdry], t);

    vdSub(num_points * S_, &neighbor_solution[0], &owner_solution[0],
          &solution_difference[0]);

    //avocado::Kernel1::f59(&bdry_owner_coefficients[ibdry][0],
    //                      &solution_difference[0], &local_auxiliary[0], D_,
    //                      num_bases, num_points, S_, 1.0, 0.0);
    avocado::Kernel1::f59(&bdry_owner_coefficients[ibdry][0],
                          &solution_difference[0], &local_auxiliary[0], D_,
                          num_bases, num_points, S_, 0.5, 0.0);
    avocado::Kernel1::f42(&bdry_owner_basis_grad_value[ibdry][0],
                          &solution[owner_cell * sb], &owner_solution_grad[0],
                          D_, num_points, num_bases, S_, 1.0, 0.0);
    avocado::Kernel1::f18(
        &local_auxiliary[0], &bdry_owner_basis_value[ibdry][0],
        &owner_solution_grad[0], D_, S_, num_bases, num_points, 1.0, 1.0);
    cblas_daxpy(dsb, 1.0, &local_auxiliary[0], 1,
                &auxiliary_solution_[owner_cell * dsb], 1);

    double* solution_jacobi = &bdry_solution_jacobi[ibdry * ssp];
    boundaries_[ibdry]->ComputeBdrySolutionJacobi(
        num_points, solution_jacobi, owner_solution, owner_solution_grad,
        bdry_normals[ibdry], bdry_points[ibdry], t);
    boundaries_[ibdry]->ComputeBdryFluxJacobi(
        num_points, flux_owner_jacobi, flux_owner_grad_jacobi, owner_cell, -1,
        owner_solution, owner_solution_grad, neighbor_solution,
        neighbor_solution_grad, bdry_normals[ibdry], bdry_points[ibdry], t);

    memset(&largeA[0], 0, num_points * ssb * sizeof(double));
    for (int ipoint = 0; ipoint < num_points; ipoint++) {
      cblas_daxpy(S_, -1.0, &identity[0], 1, &solution_jacobi[ipoint * SS_],
                  S_ + 1);
      cblas_dger(CBLAS_LAYOUT::CblasRowMajor, SS_, num_bases, 1.0,
                 &solution_jacobi[ipoint * SS_], 1,
                 &bdry_owner_basis_value[ibdry][ipoint * num_bases], 1,
                 &largeA[ipoint * ssb], num_bases);
    }
    avocado::Kernel1::f47(&bdry_owner_coefficients[ibdry][0],
                          &bdry_owner_basis_value[ibdry][0], &alpha[0], D_,
                          num_points, num_bases, num_points, 1.0, 0.0);
    avocado::Kernel1::f21(&alpha[0], &largeA[0], &largeB[0], num_points, D_,
                          num_points, ssb, 1.0, 0.0);

    for (int ipoint = 0; ipoint < num_points; ipoint++) {
      avocado::Kernel2::f13(
          &flux_owner_grad_jacobi[ipoint * DDSS_], &largeB[ipoint * dssb],
          &flux_derivative[ipoint * dssb], DS_, D_, S_, sb, 1.0, 0.0);
      gemmAB(1.0, &flux_owner_grad_jacobi[ipoint * DDSS_],
             &bdry_owner_basis_grad_value[ibdry][ipoint * db], 1.0,
             &flux_derivative[ipoint * dssb], DSS_, D_, num_bases);
      cblas_dger(CBLAS_LAYOUT::CblasRowMajor, DSS_, num_bases, 1.0,
                 &flux_owner_jacobi[ipoint * DSS_], 1,
                 &bdry_owner_basis_value[ibdry][ipoint * num_bases], 1,
                 &flux_derivative[ipoint * dssb], num_bases);
    }
    avocado::Kernel1::f59(&flux_derivative[0],
                          &bdry_owner_coefficients[ibdry][0], &block[0], S_, sb,
                          num_points * D_, num_bases, 1.0, 0.0);
    MatSetValuesBlocked(sysmat, 1, &mat_index[owner_cell], 1,
                        &mat_index[owner_cell], &block[0], ADD_VALUES);
  }

  communicate_->CommunicateEnd(&outer_solution_[0]);

  // outer face sweep
  static const int& num_faces = DENEB_DATA->GetNumFaces();
  for (int iface = num_inner_faces; iface < num_faces; iface++) {
    const int& num_points = num_face_points[iface];
    const int& owner_cell = face_owner_cell[iface];
    const int& neighbor_cell = face_neighbor_cell[iface];

    avocado::Kernel0::f4(&solution[owner_cell * sb],
                         &face_owner_basis_value[iface][0], &owner_solution[0],
                         S_, num_bases, num_points, 1.0, 0.0);
    avocado::Kernel0::f4(&outer_solution_[neighbor_cell * sb],
                         &face_neighbor_basis_value[iface][0],
                         &neighbor_solution[0], S_, num_bases, num_points, 1.0,
                         0.0);

    vdSub(num_points * S_, &neighbor_solution[0], &owner_solution[0],
          &solution_difference[0]);

    avocado::Kernel1::f59(&face_owner_coefficients[iface][0],
                          &solution_difference[0], &local_auxiliary[0], D_,
                          num_bases, num_points, S_, 0.5, 0.0);
    avocado::Kernel1::f42(&face_owner_basis_grad_value[iface][0],
                          &solution[owner_cell * sb], &owner_solution_grad[0],
                          D_, num_points, num_bases, S_, 1.0, 0.0);
    avocado::Kernel1::f18(
        &local_auxiliary[0], &face_owner_basis_value[iface][0],
        &owner_solution_grad[0], D_, S_, num_bases, num_points, 1.0, 1.0);
    cblas_daxpy(dsb, 1.0, &local_auxiliary[0], 1,
                &auxiliary_solution_[owner_cell * dsb], 1);

    avocado::Kernel1::f59(&face_neighbor_coefficients[iface][0],
                          &solution_difference[0], &local_auxiliary[0], D_,
                          num_bases, num_points, S_, 0.5, 0.0);
    avocado::Kernel1::f42(&face_neighbor_basis_grad_value[iface][0],
                          &outer_solution_[neighbor_cell * sb],
                          &neighbor_solution_grad[0], D_, num_points, num_bases,
                          S_, 1.0, 0.0);
    avocado::Kernel1::f18(
        &local_auxiliary[0], &face_neighbor_basis_value[iface][0],
        &neighbor_solution_grad[0], D_, S_, num_bases, num_points, 1.0, 1.0);

    ComputeNumFluxJacobi(num_points, flux_owner_jacobi, flux_neighbor_jacobi,
                         flux_owner_grad_jacobi, flux_neighbor_grad_jacobi,
                         owner_cell, neighbor_cell + num_cells, owner_solution,
                         owner_solution_grad, neighbor_solution,
                         neighbor_solution_grad, face_normals[iface], face_points[iface]);

    avocado::Kernel1::f47(&face_owner_coefficients[iface][0],
                          &face_owner_basis_value[iface][0], &alpha[0], D_,
                          num_points, num_bases, num_points, 1.0, 0.0);
    cblas_dcopy(db * num_points, &face_owner_basis_grad_value[iface][0], 1,
                &owner_a[0], 1);
    avocado::Kernel1::f50(&alpha[0], &face_owner_basis_value[iface][0],
                          &owner_a[0], D_, num_points, num_points, num_bases,
                          -0.5, 1.0);
    avocado::Kernel1::f50(&alpha[0], &face_neighbor_basis_value[iface][0],
                          &neighbor_b[0], D_, num_points, num_points, num_bases,
                          0.5, 0.0);

    avocado::Kernel1::f47(&face_neighbor_coefficients[iface][0],
                          &face_neighbor_basis_value[iface][0], &alpha[0], D_,
                          num_points, num_bases, num_points, 1.0, 0.0);
    cblas_dcopy(db * num_points, &face_neighbor_basis_grad_value[iface][0], 1,
                &neighbor_a[0], 1);
    avocado::Kernel1::f50(&alpha[0], &face_neighbor_basis_value[iface][0],
                          &neighbor_a[0], D_, num_points, num_points, num_bases,
                          0.5, 1.0);
    avocado::Kernel1::f50(&alpha[0], &face_owner_basis_value[iface][0],
                          &owner_b[0], D_, num_points, num_points, num_bases,
                          -0.5, 0.0);

    for (int ipoint = 0; ipoint < num_points; ipoint++) {
      gemmAB(1.0, &flux_owner_grad_jacobi[ipoint * DDSS_],
             &owner_a[ipoint * db], 0.0, &flux_derivative[ipoint * dssb], DSS_,
             D_, num_bases);
      gemmAB(1.0, &flux_neighbor_grad_jacobi[ipoint * DDSS_],
             &owner_b[ipoint * db], 1.0, &flux_derivative[ipoint * dssb], DSS_,
             D_, num_bases);
      cblas_dger(CBLAS_LAYOUT::CblasRowMajor, DSS_, num_bases, 1.0,
                 &flux_owner_jacobi[ipoint * DSS_], 1,
                 &face_owner_basis_value[iface][ipoint * num_bases], 1,
                 &flux_derivative[ipoint * dssb], num_bases);
    }
    avocado::Kernel1::f59(&flux_derivative[0],
                          &face_owner_coefficients[iface][0], &block[0], S_, sb,
                          num_points * D_, num_bases, 1.0, 0.0);
    MatSetValuesBlocked(sysmat, 1, &mat_index[owner_cell], 1,
                        &mat_index[owner_cell], &block[0], ADD_VALUES);

    for (int ipoint = 0; ipoint < num_points; ipoint++) {
      gemmAB(1.0, &flux_owner_grad_jacobi[ipoint * DDSS_],
             &neighbor_b[ipoint * db], 0.0, &flux_derivative[ipoint * dssb],
             DSS_, D_, num_bases);
      gemmAB(1.0, &flux_neighbor_grad_jacobi[ipoint * DDSS_],
             &neighbor_a[ipoint * db], 1.0, &flux_derivative[ipoint * dssb],
             DSS_, D_, num_bases);
      cblas_dger(CBLAS_LAYOUT::CblasRowMajor, DSS_, num_bases, 1.0,
                 &flux_neighbor_jacobi[ipoint * DSS_], 1,
                 &face_neighbor_basis_value[iface][ipoint * num_bases], 1,
                 &flux_derivative[ipoint * dssb], num_bases);
    }
    avocado::Kernel1::f59(&flux_derivative[0],
                          &face_owner_coefficients[iface][0], &block[0], S_, sb,
                          num_points * D_, num_bases, 1.0, 0.0);
    MatSetValuesBlocked(sysmat, 1, &mat_index[owner_cell], 1,
                        &mat_index[neighbor_cell + num_cells], &block[0],
                        ADD_VALUES);
  }

  // cell sweep
  static const auto& num_cell_points = DENEB_DATA->GetNumCellPoints();
  static const auto& cell_basis_value = DENEB_DATA->GetCellBasisValue();
  static const auto& cell_basis_grad_value =
      DENEB_DATA->GetCellBasisGradValue();
  static const auto& cell_coefficients = DENEB_DATA->GetCellCoefficients();
  static const auto& cell_source_coefficients =
      DENEB_DATA->GetCellSourceCoefficients();
  static const auto& subface_ptr = DENEB_DATA->GetSubfacePtr();
  static const auto& subface_sign = DENEB_DATA->GetSubfaceSign();
  static const auto& subface_neighbor_cells =
      DENEB_DATA->GetSubfaceNeighborCell();
  static const auto& subface_num_points = DENEB_DATA->GetSubfaceNumPoints();
  static const auto& subface_owner_coefficients =
      DENEB_DATA->GetSubfaceOwnerCoefficients();
  static const auto& subface_owner_basis_value =
      DENEB_DATA->GetSubfaceOwnerBasisValue();
  static const auto& subface_neighbor_basis_value =
      DENEB_DATA->GetSubfaceNeighborBasisValue();
  static const auto& subbdry_ptr = DENEB_DATA->GetSubbdryPtr();
  static const auto& subbdry_ind = DENEB_DATA->GetSubbdryInd();
  for (int icell = 0; icell < num_cells; icell++) {
    const int& num_points = num_cell_points[icell];

    avocado::Kernel0::f4(&solution[icell * sb], &cell_basis_value[icell][0],
                         &owner_solution[0], S_, num_bases, num_points, 1.0,
                         0.0);
    avocado::Kernel1::f42(&cell_basis_grad_value[icell][0],
                          &solution[icell * sb], &owner_solution_grad[0], D_,
                          num_points, num_bases, S_, 1.0, 0.0);
    avocado::Kernel1::f18(&auxiliary_solution_[icell * dsb],
                          &cell_basis_value[icell][0], &owner_solution_grad[0],
                          D_, S_, num_bases, num_points, 1.0, 1.0);

    ComputeComFluxJacobi(num_points, flux_owner_jacobi, flux_owner_grad_jacobi,
                         icell, owner_solution, owner_solution_grad);

    memset(&largeA[0], 0, dbb * sizeof(double));
    for (int index = subface_ptr[icell]; index < subface_ptr[icell + 1];
         index++) {
      const int& num_subface_points = subface_num_points[index];
      const int& neighbor_cell = subface_neighbor_cells[index];
      const double& sign = subface_sign[index];

      avocado::Kernel1::f61(
          subface_owner_coefficients[index], subface_owner_basis_value[index],
          &largeA[0], D_, num_bases, num_subface_points, num_bases, -sign, 1.0);
      avocado::Kernel1::f47(
          subface_owner_coefficients[index], &cell_basis_value[icell][0],
          &alpha[0], D_, num_subface_points, num_bases, num_points, sign, 0.0);
      avocado::Kernel1::f50(&alpha[0], subface_neighbor_basis_value[index],
                            &owner_b[0], D_, num_points, num_subface_points,
                            num_bases, 0.5, 0.0);

      for (int ipoint = 0; ipoint < num_points; ipoint++)
        gemmAB(1.0, &flux_owner_grad_jacobi[ipoint * DDSS_],
               &owner_b[ipoint * db], 0.0, &flux_derivative[ipoint * dssb],
               DSS_, D_, num_bases);
      avocado::Kernel1::f59(&flux_derivative[0], &cell_coefficients[icell][0],
                            &block[0], S_, sb, num_points * D_, num_bases, -1.0,
                            0.0);
      MatSetValuesBlocked(sysmat, 1, &mat_index[icell], 1,
                          &mat_index[neighbor_cell], &block[0], ADD_VALUES);
    }
    cblas_dcopy(db * num_points, &cell_basis_grad_value[icell][0], 1,
                &owner_a[0], 1);
    avocado::Kernel1::f46(&largeA[0], &cell_basis_value[icell][0], &owner_a[0],
                          D_, num_bases, num_bases, num_points, 0.5, 1.0);

    memset(&largeB[0], 0, num_points * dssb * sizeof(double));
    for (int index = subbdry_ptr[icell]; index < subbdry_ptr[icell + 1];
         index++) {
      const int& ibdry = subbdry_ind[index];
      const int& num_subbdry_points = num_bdry_points[ibdry];
      const double* solution_jacobi = &bdry_solution_jacobi[ibdry * ssp];

      avocado::Kernel1::f47(
          &bdry_owner_coefficients[ibdry][0], &cell_basis_value[icell][0],
          &alpha[0], D_, num_subbdry_points, num_bases, num_points, 1.0, 0.0);
      memset(&largeA[0], 0, num_subbdry_points * ssb * sizeof(double));
      for (int ipoint = 0; ipoint < num_subbdry_points; ipoint++)
        cblas_dger(CBLAS_LAYOUT::CblasRowMajor, SS_, num_bases, 1.0,
                   &solution_jacobi[ipoint * SS_], 1,
                   &bdry_owner_basis_value[ibdry][ipoint * num_bases], 1,
                   &largeA[ipoint * ssb], num_bases);
      avocado::Kernel1::f21(&alpha[0], &largeA[0], &largeB[0], num_points, D_,
                            num_subbdry_points, ssb, 1.0, 1.0);
    }

    for (int ipoint = 0; ipoint < num_points; ipoint++) {
      avocado::Kernel2::f13(
          &flux_owner_grad_jacobi[ipoint * DDSS_], &largeB[ipoint * dssb],
          &flux_derivative[ipoint * dssb], DS_, D_, S_, sb, 1.0, 0.0);
      gemmAB(1.0, &flux_owner_grad_jacobi[ipoint * DDSS_],
             &owner_a[ipoint * db], 1.0, &flux_derivative[ipoint * dssb], DSS_,
             D_, num_bases);
      cblas_dger(CBLAS_LAYOUT::CblasRowMajor, DSS_, num_bases, 1.0,
                 &flux_owner_jacobi[ipoint * DSS_], 1,
                 &cell_basis_value[icell][ipoint * num_bases], 1,
                 &flux_derivative[ipoint * dssb], num_bases);
    }
    avocado::Kernel1::f59(&flux_derivative[0], &cell_coefficients[icell][0],
                          &block[0], S_, sb, num_points * D_, num_bases, -1.0,
                          0.0);

    ComputeSourceJacobi(num_points, source_jacobi, icell, owner_solution,
                        owner_solution_grad);

    // Need optimization
    for (int ipoint = 0; ipoint < num_points; ipoint++)
      for (int istate = 0; istate < S_; istate++)
        for (int ibasis = 0; ibasis < num_bases; ibasis++)
          for (int jstate = 0; jstate < S_; jstate++)
            for (int jbasis = 0; jbasis < num_bases; jbasis++) {
              const int irow = ibasis + num_bases * istate;
              const int icolumn = jbasis + num_bases * jstate;

              block[irow * sb + icolumn] -=
                  cell_source_coefficients[icell][ipoint * num_bases + jbasis] *
                  cell_basis_value[icell][ipoint * num_bases + ibasis] *
                  source_jacobi[ipoint * SS_ + istate * S_ + jstate];
            }

    MatSetValuesBlocked(sysmat, 1, &mat_index[icell], 1, &mat_index[icell],
                        &block[0], ADD_VALUES);
  }

  MatAssemblyBegin(sysmat, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(sysmat, MAT_FINAL_ASSEMBLY);
}
void EquationNS2DNeq2Tnondim::SolutionLimit(double* solution) {
  static const int& num_bases = DENEB_DATA->GetNumBases();
  static const int& num_cells = DENEB_DATA->GetNumCells();
  static const int& order = DENEB_DATA->GetOrder();
  static const int sb = S_ * num_bases;
  static const int dsb = DS_ * num_bases;

  static std::vector<double> owner_u(S_ * max_num_points_);
  static std::vector<double> neighbor_u(S_ * max_num_points_);
  static const int& num_inner_faces = DENEB_DATA->GetNumInnerFaces();
  static const auto& num_face_points = DENEB_DATA->GetNumFacePoints();
  static const auto& face_owner_cell = DENEB_DATA->GetFaceOwnerCell();
  static const auto& face_neighbor_cell = DENEB_DATA->GetFaceNeighborCell();
  static const auto& face_owner_basis_value =
      DENEB_DATA->GetFaceOwnerBasisValue();
  static const auto& face_neighbor_basis_value =
      DENEB_DATA->GetFaceNeighborBasisValue();

  int ind = 0;
  for (int iface = 0; iface < num_inner_faces; iface++) {
    const int& num_points = num_face_points[iface];
    const int& owner_cell = face_owner_cell[iface];
    const int& neighbor_cell = face_neighbor_cell[iface];

    avocado::Kernel0::f4(&solution[owner_cell * sb],
                         &face_owner_basis_value[iface][0], &owner_u[0], S_,
                         num_bases, num_points, 1.0, 0.0);
    avocado::Kernel0::f4(&solution[neighbor_cell * sb],
                         &face_neighbor_basis_value[iface][0], &neighbor_u[0],
                         S_, num_bases, num_points, 1.0, 0.0);

    bool limit_flag_o = false;
    bool limit_flag_n = false;
    for (int ipoint = 0; ipoint < num_points; ipoint++) {
      ind = S_ * ipoint;
      const auto* d_o = &owner_u[ind];
      const auto& T_tr_o = owner_u[ind + ns_ + 2];
      const auto& T_eev_o = owner_u[ind + ns_ + 3];

      const auto* d_n = &neighbor_u[ind];
      const auto& T_tr_n = neighbor_u[ind + ns_ + 2];
      const auto& T_eev_n = neighbor_u[ind + ns_ + 3];

      for (int i = 0; i < ns_; i++) {
        if (d_o[i] < 0.0) limit_flag_o = true;
        if (d_n[i] < 0.0) limit_flag_n = true;
      }

      if (T_tr_o < 10 / T_ref_ || T_eev_o < 10 / T_ref_) limit_flag_o = true;
      if (T_tr_n < 10 / T_ref_ || T_eev_n < 10 / T_ref_) limit_flag_n = true;
    }

    if (limit_flag_o) {
      for (int istate = 0; istate < S_; istate++)
        memset(&solution[owner_cell * sb + istate * num_bases + 1], 0,
               sizeof(double) * (num_bases - 1));
    }
    if (limit_flag_n) {
      for (int istate = 0; istate < S_; istate++)
        memset(&solution[neighbor_cell * sb + istate * num_bases + 1], 0,
               sizeof(double) * (num_bases - 1));
    }
  }

  static const int& num_faces = DENEB_DATA->GetNumFaces();
  for (int iface = num_inner_faces; iface < num_faces; iface++) {
    const int& num_points = num_face_points[iface];
    const int& owner_cell = face_owner_cell[iface];

    avocado::Kernel0::f4(&solution[owner_cell * sb],
                         &face_owner_basis_value[iface][0], &owner_u[0], S_,
                         num_bases, num_points, 1.0, 0.0);

    bool limit_flag = false;
    for (int ipoint = 0; ipoint < num_points; ipoint++) {
      ind = S_ * ipoint;
      const auto* d = &owner_u[ind];
      const auto& T_tr = owner_u[ind + ns_ + 2];
      const auto& T_eev = owner_u[ind + ns_ + 3];

      for (int i = 0; i < ns_; i++)
        if (d[i] < 0.0) limit_flag = true;

      if (T_tr < 10 / T_ref_ || T_eev < 10 / T_ref_) limit_flag = true;
    }

    if (limit_flag) {
      for (int istate = 0; istate < S_; istate++)
        memset(&solution[owner_cell * sb + istate * num_bases + 1], 0,
               sizeof(double) * (num_bases - 1));
    }
  }

  static const int& num_bdries = DENEB_DATA->GetNumBdries();
  static const auto& num_bdry_points = DENEB_DATA->GetNumBdryPoints();
  static const auto& bdry_owner_cell = DENEB_DATA->GetBdryOwnerCell();
  static const auto& bdry_points = DENEB_DATA->GetBdryPoints();
  static const auto& bdry_owner_basis_value =
      DENEB_DATA->GetBdryOwnerBasisValue();
  for (int ibdry = 0; ibdry < num_bdries; ibdry++) {
    const int& num_points = num_bdry_points[ibdry];
    const int& owner_cell = bdry_owner_cell[ibdry];

    avocado::Kernel0::f4(&solution[owner_cell * sb],
                         &bdry_owner_basis_value[ibdry][0], &owner_u[0], S_,
                         num_bases, num_points, 1.0, 0.0);

    bool limit_flag = false;
    for (int ipoint = 0; ipoint < num_points; ipoint++) {
      ind = S_ * ipoint;
      const auto* d = &owner_u[ind];
      const auto& T_tr = owner_u[ind + ns_ + 2];
      const auto& T_eev = owner_u[ind + ns_ + 3];

      for (int i = 0; i < ns_; i++)
        if (d[i] < 0.0) limit_flag = true;

      if (T_tr < 10 / T_ref_ || T_eev < 10 / T_ref_) limit_flag = true;

      if (limit_flag) {
        for (int istate = 0; istate < S_; istate++)
          memset(&solution[owner_cell * sb + istate * num_bases + 1], 0,
                 sizeof(double) * (num_bases - 1));
      }
    }
  }

  // cell sweep
  static const auto& num_cell_points = DENEB_DATA->GetNumCellPoints();
  static const auto& cell_basis_value = DENEB_DATA->GetCellBasisValue();
  static std::vector<double> sol0(S_);
  static std::vector<double> pri0(S_);
  for (int icell = 0; icell < num_cells; icell++) {
    const int& num_points = num_cell_points[icell];

    avocado::Kernel0::f4(&solution[icell * sb], &cell_basis_value[icell][0],
                         &owner_u[0], S_, num_bases, num_points, 1.0, 0.0);

    bool limit_flag = false;
    for (int ipoint = 0; ipoint < num_points; ipoint++) {
      ind = S_ * ipoint;
      const auto* d = &owner_u[ind];
      const auto& T_tr = owner_u[ind + ns_ + 2];
      const auto& T_eev = owner_u[ind + ns_ + 3];

      for (int i = 0; i < ns_; i++)
        if (d[i] < 0.0) limit_flag = true;

      if (T_tr < 10 / T_ref_ || T_eev < 10 / T_ref_) limit_flag = true;

      if (limit_flag) {
        for (int istate = 0; istate < S_; istate++)
          memset(&solution[icell * sb + istate * num_bases + 1], 0,
                 sizeof(double) * (num_bases - 1));
      }
    }

    double* sol = &solution[icell * sb];

    for (int i = 0; i < ns_; i++) {
      sol0[i] = sol[i * num_bases] * cell_basis_value[icell][0];
      if (sol0[i] < 0.0) sol[i * num_bases] = 0.0;
    }
    sol0[ns_ + 2] = sol[(ns_ + 2) * num_bases] * cell_basis_value[icell][0];
    sol0[ns_ + 3] = sol[(ns_ + 3) * num_bases] * cell_basis_value[icell][0];

    if (sol0[ns_ + 2] < 10 / T_ref_) {
      sol0[ns_ + 2] = 10 / T_ref_;
      sol[(ns_ + 2) * num_bases] = sol0[ns_ + 2] / cell_basis_value[icell][0];
    }
    if (sol0[ns_ + 3] < 10 / T_ref_) {
      sol0[ns_ + 3] = 10 / T_ref_;
      sol[(ns_ + 3) * num_bases] = sol0[ns_ + 3] / cell_basis_value[icell][0];
    }
  }

  if (order > 1) {
    int num_bases1 = 0;
    if (dimension_ == 2)
      num_bases1 = 3;
    else if (dimension_ == 3)
      num_bases1 = 4;
    else ERROR_MESSAGE("Wrong dimension");

    for (int icell = 0; icell < wall_cells_.size(); icell++) {
      int wall_cell_index = wall_cells_[icell];
      for (int istate = 0; istate < S_; istate++) {
        cblas_dscal(num_bases - num_bases1, 0.0, &solution[wall_cell_index * sb + istate * num_bases + num_bases1], 1);
      }
    }
  }
}
void EquationNS2DNeq2Tnondim::SystemMatrixShift(const double* solution,
                                                Mat& sysmat, const double dt,
                                                const double t) {
  static const auto& num_cell_points = DENEB_DATA->GetNumCellPoints();
  static const auto& cell_basis_value = DENEB_DATA->GetCellBasisValue();
  static const int& num_cells = DENEB_DATA->GetNumCells();
  static const int& num_bases = DENEB_DATA->GetNumBases();
  static const auto& mat_index = DENEB_DATA->GetMatIndex();
  static const auto& cell_mass_coefficients =
      DENEB_DATA->GetCellMassCoefficients();
  static const auto& cell_points = DENEB_DATA->GetCellPoints();
  static const int sb = num_states_ * num_bases;
  static std::vector<double> block(sb * sb);
  static std::vector<double> owner_solution(S_ * max_num_points_);
  static std::vector<double> solution_jacobi(SS_ * max_num_points_);
  

  const double dt_factor = 1.0 / dt;
  for (int icell = 0; icell < num_cells; icell++) {
    const int& num_points = num_cell_points[icell];
    memset(&block[0], 0, sb * sb * sizeof(double));

    avocado::Kernel0::f4(&solution[icell * sb], &cell_basis_value[icell][0],
                         &owner_solution[0], S_, num_bases, num_points, 1.0,
                         0.0);
    ComputeSolutionJacobi(num_points, solution_jacobi, owner_solution);
      
    // Need optimization
    for (int ipoint = 0; ipoint < num_points; ipoint++)
    {
      const double fy = static_cast<double>(ax_) * (cell_points[icell][ipoint * D_ + 1] - 1.0) + 1.0;
      for (int istate = 0; istate < S_; istate++)
        for (int ibasis = 0; ibasis < num_bases; ibasis++)
          for (int jstate = 0; jstate < S_; jstate++)
            for (int jbasis = 0; jbasis < num_bases; jbasis++) {
              const int irow = ibasis + num_bases * istate;
              const int icolumn = jbasis + num_bases * jstate;

              block[irow * sb + icolumn] +=
                cell_mass_coefficients[icell][ipoint * num_bases * num_bases +
                ibasis * num_bases + jbasis] *
                solution_jacobi[ipoint * SS_ + istate * S_ + jstate] * fy;
            }
    }

    avocado::VecScale(sb * sb, dt_factor, &block[0]);

    MatSetValuesBlocked(sysmat, 1, &mat_index[icell], 1, &mat_index[icell],
                        &block[0], ADD_VALUES);
  }
  MatAssemblyBegin(sysmat, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(sysmat, MAT_FINAL_ASSEMBLY);
}
void EquationNS2DNeq2Tnondim::SystemMatrixShift(
    const double* solution, Mat& sysmat, const std::vector<double>& local_dt,
    const double t) {
  static const auto& num_cell_points = DENEB_DATA->GetNumCellPoints();
  static const auto& cell_basis_value = DENEB_DATA->GetCellBasisValue();
  static const int& num_cells = DENEB_DATA->GetNumCells();
  static const int& num_bases = DENEB_DATA->GetNumBases();
  static const auto& mat_index = DENEB_DATA->GetMatIndex();
  static const auto& cell_mass_coefficients =
      DENEB_DATA->GetCellMassCoefficients();
  static const auto& cell_points = DENEB_DATA->GetCellPoints();
  static const int sb = num_states_ * num_bases;
  static std::vector<double> block(sb * sb);
  static std::vector<double> owner_solution(S_ * max_num_points_);
  static std::vector<double> solution_jacobi(SS_ * max_num_points_);

  for (int icell = 0; icell < num_cells; icell++) {
    const int& num_points = num_cell_points[icell];
    memset(&block[0], 0, sb * sb * sizeof(double));
    const double dt_factor = 1.0 / local_dt[icell];

    avocado::Kernel0::f4(&solution[icell * sb], &cell_basis_value[icell][0],
                         &owner_solution[0], S_, num_bases, num_points, 1.0,
                         0.0);
    ComputeSolutionJacobi(num_points, solution_jacobi, owner_solution);

    // Need optimization
    for (int ipoint = 0; ipoint < num_points; ipoint++) {
      const double fy = static_cast<double>(ax_) * (cell_points[icell][ipoint * D_ + 1] - 1.0) + 1.0;
      for (int istate = 0; istate < S_; istate++)
        for (int ibasis = 0; ibasis < num_bases; ibasis++)
          for (int jstate = 0; jstate < S_; jstate++)
            for (int jbasis = 0; jbasis < num_bases; jbasis++) {
              const int irow = ibasis + num_bases * istate;
              const int icolumn = jbasis + num_bases * jstate;

              block[irow * sb + icolumn] +=
                cell_mass_coefficients[icell][ipoint * num_bases * num_bases + ibasis * num_bases + jbasis] *
                solution_jacobi[ipoint * SS_ + istate * S_ + jstate] * fy;
            }
    }

    avocado::VecScale(sb * sb, dt_factor, &block[0]);

    MatSetValuesBlocked(sysmat, 1, &mat_index[icell], 1, &mat_index[icell],
                        &block[0], ADD_VALUES);
  }
  MatAssemblyBegin(sysmat, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(sysmat, MAT_FINAL_ASSEMBLY);
}
void EquationNS2DNeq2Tnondim::ComputeComFlux(
    const int num_points, std::vector<double>& flux, const int icell,
    const std::vector<double>& owner_u,
    const std::vector<double>& owner_div_u) {
  static std::vector<double> Ds(ns_, 0.0);
  static std::vector<double> Isx(ns_, 0.0);
  static std::vector<double> Isy(ns_, 0.0);
  static std::vector<double> hs(ns_, 0.0);
  static std::vector<double> diffflux(2 * ns_, 0.0);
  std::vector<double> sol_jacobi(num_points * SS_, 0.0);
  std::vector<double> owner_div_sol(num_points * S_ * 2, 0.0);

  const auto& species = mixture_->GetSpecies();
  const auto& hidx = mixture_->GetHeavyParticleIdx();
  const auto& eidx = mixture_->GetElectronIndex();
  const auto& cell_points = DENEB_DATA->GetCellPoints();

  std::vector<double> dim_d(ns_);
  ComputeSolutionJacobi(num_points, sol_jacobi, owner_u);
  gradpri2gradsol(num_points, &sol_jacobi[0], &owner_div_u[0], &owner_div_sol[0]);

  int ind = 0;
  double arvis = 0.0;
  for (int ipoint = 0; ipoint < num_points; ipoint++) {
    GET_SOLUTION_PS(, owner_u);
    GET_SOLUTION_GRAD_PDS(, owner_div_u);
    GET_CONS_SOLUTION_GRAD_PDS(, owner_div_sol);
    arvis =
        DENEB_ARTIFICIAL_VISCOSITY->GetArtificialViscosityValue(icell, ipoint);
    const double y = cell_points[icell][ipoint * D_ + 1];

    // Convective flux    
    mixture_->SetDensity(&dim_d[0]);
    const double mixture_d = mixture_->GetTotalDensity() / rho_ref_;
    const double mixture_p = mixture_->GetPressure(dim_T_tr, dim_T_eev) / p_ref_;
    const double beta = mixture_->GetBeta(dim_T_tr);
    const double a = std::sqrt((1.0 + beta) * mixture_p / mixture_d);
    double de = 0.0;
    double de_eev = 0.0;
    for (int i = 0; i < ns_; i++) {
      const double e = species[i]->GetInternalEnergy(dim_T_tr, dim_T_eev) / e_ref_;
      const double e_eev =
          species[i]->GetElectronicVibrationEnergy(dim_T_eev) / e_ref_;
      de += d[i] * e;
      de_eev += d[i] * e_eev;
    }
    const double du = mixture_d * u;
    const double dv = mixture_d * v;
    const double dE = de + 0.5 * (du * u + dv * v);

    // Viscous flux
    const double mu = mixture_->GetViscosity(dim_T_tr, dim_T_eev) / mu_ref_;
    const double k_tr =
        mixture_->GetTransRotationConductivity(dim_T_tr, dim_T_eev) / k_ref_;
    const double k_eev =
        mixture_->GetElectronicVibrationConductivity(dim_T_tr, dim_T_eev) / k_ref_;

    const double* Y = mixture_->GetMassFraction();
    double mixture_dx = 0.0;
    double mixture_dy = 0.0;
    for (int i = 0; i < ns_; i++) {
      mixture_dx += dx[i];
      mixture_dy += dy[i];
    }

    mixture_->GetDiffusivity(&Ds[0], dim_T_tr, dim_T_eev);
    for (int i = 0; i < ns_; i++) {
      Ds[i] /= D_ref_;
      Isx[i] = -Ds[i] * (dx[i] - Y[i] * mixture_dx);
      Isy[i] = -Ds[i] * (dy[i] - Y[i] * mixture_dy);
    }

    double Ix_sum = 0.0;
    double Iy_sum = 0.0;
    for (const int& idx : hidx) {
      Ix_sum += Isx[idx];
      Iy_sum += Isy[idx];
    }

    for (const int& idx : hidx) {
      diffflux[idx] = -Isx[idx] + Y[idx] * Ix_sum;
      diffflux[idx + ns_] = -Isy[idx] + Y[idx] * Iy_sum;
    }

    if (eidx >= 0) {
      double sum_x = 0.0;
      double sum_y = 0.0;
      for (const int& idx : hidx) {
        const int charge = species[idx]->GetCharge();
        const double Mw = species[idx]->GetMolecularWeight();
        sum_x += diffflux[idx] * charge / Mw;
        sum_y += diffflux[idx + ns_] * charge / Mw;
      }
      const double Mw_e = species[eidx]->GetMolecularWeight();
      diffflux[eidx] = Mw_e * sum_x;
      diffflux[eidx + ns_] = Mw_e * sum_y;
    }

    const auto tau_ax = (std::abs(y) < 1.0e-12) ? 0.0 : static_cast<double>(ax_) * v / y;
    const auto txx = c23_ * mu * (2.0 * ux - vy - tau_ax);
    const auto txy = mu * (uy + vx);
    const auto tyy = c23_ * mu * (2.0 * vy - ux - tau_ax);

    double Jh_x = 0.0;
    double Jh_y = 0.0;
    double Je_eev_x = 0.0;
    double Je_eev_y = 0.0;
    for (int i = 0; i < ns_; i++) {
      const double h = species[i]->GetEnthalpy(dim_T_tr, dim_T_eev) / e_ref_;
      const double e_eev = species[i]->GetElectronicVibrationEnergy(dim_T_eev) / e_ref_;
      Jh_x += diffflux[i] * h;
      Je_eev_x += diffflux[i] * e_eev;
      Jh_y += diffflux[i + ns_] * h;
      Je_eev_y += diffflux[i + ns_] * e_eev;
    }
    const double q_tr_x = k_tr * Ttrx;
    const double q_tr_y = k_tr * Ttry;
    const double q_eev_x = k_eev * Teevx;
    const double q_eev_y = k_eev * Teevy;

    ind = DS_ * ipoint;
    for (int i = 0; i < ns_; i++)
      flux[ind++] = d[i] * u - diffflux[i] - arvis * dx[i];
    flux[ind++] = du * u + mixture_p - txx - arvis * dux;
    flux[ind++] = du * v - txy - arvis * dvx;
    flux[ind++] = (dE + mixture_p) * u - txx * u - txy * v - q_tr_x - q_eev_x - Jh_x - arvis * dEx;
    flux[ind++] = de_eev * u - q_eev_x - Je_eev_x - arvis * de_eevx;
    for (int i = 0; i < ns_; i++)
      flux[ind++] = d[i] * v - diffflux[i + ns_] - arvis * dy[i];
    flux[ind++] = dv * u - txy - arvis * duy;
    flux[ind++] = dv * v + mixture_p - tyy - arvis * dvy;
    flux[ind++] = (dE + mixture_p) * v - txy * u - tyy * v - q_tr_y - q_eev_y - Jh_y - arvis * dEy;
    flux[ind] = de_eev * v - q_eev_y - Je_eev_y - arvis * de_eevy;

    const double fy = static_cast<double>(ax_) * (cell_points[icell][ipoint * D_ + 1] - 1.0) + 1.0;
    cblas_dscal(DS_, fy, &flux[DS_ * ipoint], 1);
  }
}
void EquationNS2DNeq2Tnondim::ComputeComFluxJacobi(
    const int num_points, std::vector<double>& flux_jacobi,
    std::vector<double>& flux_grad_jacobi, const int icell,
    const std::vector<double>& owner_u,
    const std::vector<double>& owner_div_u) {
  // flux_jacobi(ds1s2) = F(ds1) over U(s2)
  // flux_grad_jacobi(d1s1 s2d2) = F(d1s1) over gradU(d2s2)
  static std::vector<double> Ds(ns_, 0.0);
  static std::vector<double> Isx(ns_, 0.0);
  static std::vector<double> Isy(ns_, 0.0);
  static std::vector<double> hs(ns_, 0.0);
  static std::vector<aDual> flux1(DS_, aDual(S_));
  static std::vector<aDual> flux2(DS_, aDual(DS_));
  std::vector<double> sol_jacobi(num_points * SS_, 0.0);
  std::vector<double> owner_div_sol(num_points * S_ * 2, 0.0);

  int ind = 0;
  double arvis = 0.0;

  const auto& species = mixture_->GetSpecies();
  const auto& hidx = mixture_->GetHeavyParticleIdx();
  const auto& eidx = mixture_->GetElectronIndex();
  const auto& cell_points = DENEB_DATA->GetCellPoints();

  // flux jacobi
  std::vector<double> dim_d(ns_);
  ComputeSolutionJacobi(num_points, sol_jacobi, owner_u);
  gradpri2gradsol(num_points, &sol_jacobi[0], &owner_div_u[0], &owner_div_sol[0]);
  for (int ipoint = 0; ipoint < num_points; ipoint++) {
    const double y = cell_points[icell][ipoint * D_ + 1];
    const double fy = static_cast<double>(ax_) * (cell_points[icell][ipoint * D_ + 1] - 1.0) + 1.0;
    {
      // ps
      ind = S_ * ipoint;
      std::vector<aDual> d(ns_, aDual(S_));
      for (int i = 0; i < ns_; i++) {
        dim_d[i] = owner_u[ind] * rho_ref_;
        d[i] = aDual(S_, owner_u[ind++], i);
      }
      aDual u(S_, owner_u[ind++], ns_);
      aDual v(S_, owner_u[ind++], ns_ + 1);
      const double& T_tr = owner_u[ind++];
      const double& T_eev = owner_u[ind];
      const double dim_T_tr = T_tr * T_ref_;
      const double dim_T_eev = T_eev * T_ref_;

      GET_SOLUTION_GRAD_PDS(, owner_div_u);
      GET_CONS_SOLUTION_GRAD_PDS(, owner_div_sol);
      arvis = DENEB_ARTIFICIAL_VISCOSITY->GetArtificialViscosityValue(icell,
                                                                      ipoint);

      // Convective flux
      mixture_->SetDensity(&dim_d[0]);
      aDual mixture_d(S_, mixture_->GetTotalDensity() / rho_ref_);
      for (int i = 0; i < ns_; i++) mixture_d.df[i] = 1.0;

      aDual mixture_p(S_);
      mixture_p.f =
          mixture_->GetPressureJacobian(dim_T_tr, dim_T_eev, &mixture_p.df[0]);
      std::swap(mixture_p.df[ns_], mixture_p.df[ns_ + 2]);
      std::swap(mixture_p.df[ns_ + 1], mixture_p.df[ns_ + 3]);
      mixture_p.f /= p_ref_;
      for (int i = 0; i < ns_; i++)
        mixture_p.df[i] *= rho_ref_ / p_ref_;
      mixture_p.df[ns_ + 2] *= T_ref_ / p_ref_;
      mixture_p.df[ns_ + 3] *= T_ref_ / p_ref_;

      aDual beta(S_);
      beta.f = mixture_->GetBetaJacobian(dim_T_tr, &beta.df[0]);
      for (int i = 0; i < ns_; i++) beta.df[i] *= rho_ref_;
      beta.df[ns_ + 2] *= T_ref_;
      beta.df[ns_ + 3] *= T_ref_;

      const auto a = std::sqrt((1.0 + beta) * mixture_p / mixture_d);
      aDual de(S_);
      aDual de_eev(S_);
      for (int i = 0; i < ns_; i++) {
        const auto e = species[i]->GetInternalEnergy(dim_T_tr, dim_T_eev) / e_ref_;
        const auto e_eev = species[i]->GetElectronicVibrationEnergy(dim_T_eev) / e_ref_;
        const auto Cv_tr = species[i]->GetTransRotationSpecificHeat(dim_T_tr) * T_ref_ / e_ref_;
        const auto Cv_eev =
            species[i]->GetElectronicVibrationSpecificHeat(dim_T_eev) * T_ref_ /
            e_ref_;

        de.f += d[i].f * e;
        de.df[i] = e;
        de.df[ns_ + 2] += d[i].f * Cv_tr;
        de.df[ns_ + 3] += d[i].f * Cv_eev;

        de_eev.f += d[i].f * e_eev;
        de_eev.df[i] = e_eev;
        de_eev.df[ns_ + 3] += d[i].f * Cv_eev;
      }
      const auto du = mixture_d * u;
      const auto dv = mixture_d * v;
      const auto dE = de + 0.5 * (du * u + dv * v);

      // Viscous flux
      const double mu = mixture_->GetViscosity(dim_T_tr, dim_T_eev) / mu_ref_;
      const double k_tr =
          mixture_->GetTransRotationConductivity(dim_T_tr, dim_T_eev) / k_ref_;
      const double k_eev =
          mixture_->GetElectronicVibrationConductivity(dim_T_tr, dim_T_eev) /
          k_ref_;

      std::vector<aDual> Y(ns_, aDual(S_));
      for (int i = 0; i < ns_; i++) Y[i] = d[i] / mixture_d;

      double mixture_dx = 0.0;
      double mixture_dy = 0.0;
      for (int i = 0; i < ns_; i++) {
        mixture_dx += dx[i];
        mixture_dy += dy[i];
      }

      std::vector<aDual> Isx(ns_, aDual(S_));
      std::vector<aDual> Isy(ns_, aDual(S_));
      mixture_->GetDiffusivity(&Ds[0], dim_T_tr, dim_T_eev);
      for (int i = 0; i < ns_; i++) {
        Ds[i] /= D_ref_;
        Isx[i] = -Ds[i] * (dx[i] - Y[i] * mixture_dx);
        Isy[i] = -Ds[i] * (dy[i] - Y[i] * mixture_dy);
      }

      aDual Ix_sum(S_);
      aDual Iy_sum(S_);
      for (const int& idx : hidx) {
        Ix_sum = Ix_sum + Isx[idx];
        Iy_sum = Iy_sum + Isy[idx];
      }

      std::vector<aDual> diffflux(2 * ns_, aDual(S_));
      for (const int& idx : hidx) {
        diffflux[idx] = -Isx[idx] + Y[idx] * Ix_sum;
        diffflux[idx + ns_] = -Isy[idx] + Y[idx] * Iy_sum;
      }
      if (eidx >= 0) {
        aDual sum_x(S_);
        aDual sum_y(S_);
        for (const int& idx : hidx) {
          const int charge = species[idx]->GetCharge();
          const double Mw = species[idx]->GetMolecularWeight();
          sum_x = sum_x + diffflux[idx] * charge / Mw;
          sum_y = sum_y + diffflux[idx + ns_] * charge / Mw;
        }
        const double Mw_e = species[eidx]->GetMolecularWeight();
        diffflux[eidx] = Mw_e * sum_x;
        diffflux[eidx + ns_] = Mw_e * sum_y;
      }

      const auto tau_ax = (std::abs(y) < 1.0e-12) ? aDual(S_, 0.0) : static_cast<double>(ax_) * v / y;
      const auto txx = c23_ * mu * (2.0 * ux - vy - tau_ax);
      const auto txy = mu * (uy + vx);
      const auto tyy = c23_ * mu * (2.0 * vy - ux - tau_ax);

      aDual Jh_x(S_);
      aDual Jh_y(S_);
      aDual Je_eev_x(S_);
      aDual Je_eev_y(S_);
      for (int i = 0; i < ns_; i++) {
        const double h = species[i]->GetEnthalpy(dim_T_tr, dim_T_eev) / e_ref_;
        const double e_eev =
            species[i]->GetElectronicVibrationEnergy(dim_T_eev) / e_ref_;
        const auto Cv_tr = species[i]->GetTransRotationSpecificHeat(dim_T_tr) *
                           T_ref_ / e_ref_;
        const auto Cv_eev =
            species[i]->GetElectronicVibrationSpecificHeat(dim_T_eev) * T_ref_ /
            e_ref_;
        const double R = species[i]->GetSpecificGasConstant() * T_ref_ / e_ref_;

        const auto Cp_tr = Cv_tr + (i == eidx) ? 0.0 : R;
        const auto Cp_eev = Cv_eev + (i == eidx) ? R : 0.0;

        Jh_x = Jh_x + diffflux[i] * h;
        Jh_x.df[ns_ + 2] += diffflux[i].f * Cp_tr;
        Jh_x.df[ns_ + 3] += diffflux[i].f * Cp_eev;
        Jh_y = Jh_y + diffflux[i + ns_] * h;
        Jh_y.df[ns_ + 2] += diffflux[i + ns_].f * Cp_tr;
        Jh_y.df[ns_ + 3] += diffflux[i + ns_].f * Cp_eev;

        Je_eev_x = Je_eev_x + diffflux[i] * e_eev;
        Je_eev_x.df[ns_ + 3] += diffflux[i].f * Cv_eev;
        Je_eev_y = Je_eev_y + diffflux[i + ns_] * e_eev;
        Je_eev_y.df[ns_ + 3] += diffflux[i + ns_].f * Cv_eev;
      }

      const aDual q_tr_x(S_, k_tr * Ttrx);
      const aDual q_tr_y(S_, k_tr * Ttry);
      const aDual q_eev_x(S_, k_eev * Teevx);
      const aDual q_eev_y(S_, k_eev * Teevy);

      ind = 0;
      for (int i = 0; i < ns_; i++)
        flux1[ind++] = d[i] * u - diffflux[i] - arvis * dx[i];
      flux1[ind++] = du * u + mixture_p - txx - arvis * dux;
      flux1[ind++] = du * v - txy - arvis * dvx;
      flux1[ind++] = (dE + mixture_p) * u - txx * u - txy * v - q_tr_x - q_eev_x - Jh_x - arvis * dEx;
      flux1[ind++] = de_eev * u - q_eev_x - Je_eev_x - arvis * de_eevx;
      for (int i = 0; i < ns_; i++)
        flux1[ind++] = d[i] * v - diffflux[i + ns_] - arvis * dy[i];
      flux1[ind++] = dv * u - txy - arvis * duy;
      flux1[ind++] = dv * v + mixture_p - tyy - arvis * dvy;
      flux1[ind++] = (dE + mixture_p) * v - txy * u - tyy * v - q_tr_y - q_eev_y - Jh_y - arvis * dEy;
      flux1[ind] = de_eev * v - q_eev_y - Je_eev_y - arvis * de_eevy;

      // pdss
      ind = DSS_ * ipoint;
      for (int ds = 0; ds < DS_; ds++)
        for (int istate = 0; istate < S_; istate++)
          flux_jacobi[ind++] = flux1[ds].df[istate];

      cblas_dscal(DSS_, fy, &flux_jacobi[DSS_ * ipoint], 1);
    }
    {
      GET_SOLUTION_PS(, owner_u);
      // pds over psd
      ind = DS_ * ipoint;
      std::vector<aDual> dx(ns_, aDual(DS_));
      for (int i = 0; i < ns_; i++)
        dx[i] = aDual(DS_, owner_div_u[ind++], 2 * i);
      const aDual ux(DS_, owner_div_u[ind++], 2 * ns_);
      const aDual vx(DS_, owner_div_u[ind++], 2 * (ns_ + 1));
      const aDual Ttrx(DS_, owner_div_u[ind++], 2 * (ns_ + 2));
      const aDual Teevx(DS_, owner_div_u[ind++], 2 * (ns_ + 3));
      std::vector<aDual> dy(ns_, aDual(DS_));
      for (int i = 0; i < ns_; i++)
        dy[i] = aDual(DS_, owner_div_u[ind++], 2 * i + 1);
      const aDual uy(DS_, owner_div_u[ind++], 2 * ns_ + 1);
      const aDual vy(DS_, owner_div_u[ind++], 2 * (ns_ + 1) + 1);
      const aDual Ttry(DS_, owner_div_u[ind++], 2 * (ns_ + 2) + 1);
      const aDual Teevy(DS_, owner_div_u[ind], 2 * (ns_ + 3) + 1);

      // Viscous flux
      mixture_->SetDensity(&dim_d[0]);
      const double mu = mixture_->GetViscosity(dim_T_tr, dim_T_eev) / mu_ref_;
      const double k_tr =
          mixture_->GetTransRotationConductivity(dim_T_tr, dim_T_eev) / k_ref_;
      const double k_eev =
          mixture_->GetElectronicVibrationConductivity(dim_T_tr, dim_T_eev) /
          k_ref_;

      const double* Y = mixture_->GetMassFraction();
      aDual mixture_dx(DS_);
      aDual mixture_dy(DS_);
      for (int i = 0; i < ns_; i++) {
        mixture_dx = mixture_dx + dx[i];
        mixture_dy = mixture_dy + dy[i];
      }

      std::vector<aDual> Isx(ns_, aDual(DS_));
      std::vector<aDual> Isy(ns_, aDual(DS_));
      mixture_->GetDiffusivity(&Ds[0], dim_T_tr, dim_T_eev);
      for (int i = 0; i < ns_; i++) {
        Ds[i] /= D_ref_;
        Isx[i] = -Ds[i] * (dx[i] - Y[i] * mixture_dx);
        Isy[i] = -Ds[i] * (dy[i] - Y[i] * mixture_dy);
      }

      aDual Ix_sum(DS_);
      aDual Iy_sum(DS_);
      for (const int& idx : hidx) {
        Ix_sum = Ix_sum + Isx[idx];
        Iy_sum = Iy_sum + Isy[idx];
      }

      std::vector<aDual> diffflux(2 * ns_, aDual(DS_));

      for (const int& idx : hidx) {
        diffflux[idx] = -Isx[idx] + Y[idx] * Ix_sum;
        diffflux[idx + ns_] = -Isy[idx] + Y[idx] * Iy_sum;
      }
      if (eidx >= 0) {
        aDual sum_x(DS_);
        aDual sum_y(DS_);
        for (const int& idx : hidx) {
          const int charge = species[idx]->GetCharge();
          const double Mw = species[idx]->GetMolecularWeight();
          sum_x = sum_x + diffflux[idx] * charge / Mw;
          sum_y = sum_y + diffflux[idx + ns_] * charge / Mw;
        }
        const double Mw_e = species[eidx]->GetMolecularWeight();
        diffflux[eidx] = Mw_e * sum_x;
        diffflux[eidx + ns_] = Mw_e * sum_y;
      }

      const auto tau_ax = (std::abs(y) < 1.0e-12) ? aDual(DS_, 0.0) : aDual(DS_,  static_cast<double>(ax_) * v / y);
      const auto txx = c23_ * mu * (2.0 * ux - vy - tau_ax);
      const auto txy = mu * (uy + vx);
      const auto tyy = c23_ * mu * (2.0 * vy - ux - tau_ax);

      aDual Jh_x(DS_);
      aDual Jh_y(DS_);
      aDual Je_eev_x(DS_);
      aDual Je_eev_y(DS_);

      for (int i = 0; i < ns_; i++) {
        const double h = species[i]->GetEnthalpy(dim_T_tr, dim_T_eev) / e_ref_;
        const double e_eev =
            species[i]->GetElectronicVibrationEnergy(dim_T_eev) / e_ref_;

        Jh_x = Jh_x + diffflux[i] * h;
        Jh_y = Jh_y + diffflux[i + ns_] * h;
        Je_eev_x = Je_eev_x + diffflux[i] * e_eev;
        Je_eev_y = Je_eev_y + diffflux[i + ns_] * e_eev;
      }

      const auto q_tr_x = k_tr * Ttrx;
      const auto q_tr_y = k_tr * Ttry;
      const auto q_eev_x = k_eev * Teevx;
      const auto q_eev_y = k_eev * Teevy;

      ind = 0;
      for (int i = 0; i < ns_; i++) flux2[ind++] = -diffflux[i];
      flux2[ind++] = -txx;
      flux2[ind++] = -txy;
      flux2[ind++] = -txx * u - txy * v - q_tr_x - q_eev_x - Jh_x;
      flux2[ind++] = -q_eev_x - Je_eev_x;
      for (int i = 0; i < ns_; i++) flux2[ind++] = -diffflux[i + ns_];
      flux2[ind++] = -txy;
      flux2[ind++] = -tyy;
      flux2[ind++] = -txy * u - tyy * v - q_tr_y - q_eev_y - Jh_y;
      flux2[ind] = -q_eev_y - Je_eev_y;

      // pdssd
      ind = DDSS_ * ipoint;
      for (int ds = 0; ds < DS_; ds++)
        for (int sd = 0; sd < DS_; sd++)
          flux_grad_jacobi[ind++] = flux2[ds].df[sd];

      for (int istate = 0; istate < S_; istate++)
        for (int jstate = 0; jstate < S_; jstate++)
          for (int idim = 0; idim < D_; idim++) {
            const int irow = idim * S_ + istate;
            const int icolumn = jstate * D_ + idim;
            flux_grad_jacobi[DDSS_ * ipoint + irow * DS_ + icolumn] -= arvis * sol_jacobi[ipoint * SS_ + istate * S_ + jstate];
          }

      cblas_dscal(DDSS_, fy, &flux_grad_jacobi[DDSS_ * ipoint], 1);
    }
  }
}
void EquationNS2DNeq2Tnondim::ComputeSource(
    const int num_points, std::vector<double>& source, const int icell,
    const std::vector<double>& owner_u,
    const std::vector<double>& owner_div_u) {
  const auto& cell_points = DENEB_DATA->GetCellPoints();

  int ind = 0;
  const double inv_rhov = 1.0 / (rho_ref_ * v_ref_);
  const double inv_rhoev = 1.0 / (rho_ref_ * e_ref_ * v_ref_);
  std::vector<double> dim_d(ns_);

  bool axi_flag = false;
  if (ax_ == 1) axi_flag = true;
  for (int ipoint = 0; ipoint < num_points; ipoint++) {
    GET_SOLUTION_PS(, owner_u);
    GET_SOLUTION_GRAD_PDS(, owner_div_u);
    const double y = cell_points[icell][ipoint * D_ + 1];
    const double fy = static_cast<double>(ax_) * (y - 1.0) + 1.0;

    ind = S_ * ipoint;
    mixture_->SetDensity(&dim_d[0]);
    const double* rate = mixture_->GetSpeciesReactionRate(dim_T_tr, dim_T_eev);
    for (int i = 0; i < ns_; i++) source[ind++] = rate[i] * L_ref_ * inv_rhov * fy;

    const double ct = mixture_->GetReactionTransferRate(dim_T_tr, dim_T_eev);
    const double vt = mixture_->GetVTTransferRate(dim_T_tr, dim_T_eev);
    const double et = mixture_->GetETTransferRate(dim_T_tr, dim_T_eev);

    double p = 0.0;
    double mu = 0.0;
    double ttt = 0.0;
    if (axi_flag) {
      p = mixture_->GetPressure(dim_T_tr, dim_T_eev) / p_ref_;
      mu = mixture_->GetViscosity(dim_T_tr, dim_T_eev) / mu_ref_;
      ttt = c23_ * mu * (2.0 * v / y - ux - vy);
    }

    source[ind++] = 0.0;
    source[ind++] = static_cast<double>(ax_) * (p - ttt);
    source[ind++] = 0.0;
    source[ind] = (ct + vt + et) * L_ref_ * inv_rhoev * fy;
  }
}
void EquationNS2DNeq2Tnondim::ComputeSourceJacobi(
    const int num_points, std::vector<double>& source_jacobi, const int icell,
    const std::vector<double>& owner_u,
    const std::vector<double>& owner_div_u) {
  const auto& cell_points = DENEB_DATA->GetCellPoints();

  static std::vector<double> rate_jac(ns_ * (ns_ + 2), 0.0);
  static std::vector<double> transfer_jac(ns_ + 2, 0.0);

  std::vector<double> dim_d(ns_);

  memset(&source_jacobi[0], 0, num_points * SS_ * sizeof(double));

  const double t_ref = L_ref_ / v_ref_;
  const double inv_rho_ref = 1.0 / rho_ref_;
  const double inv_e_ref = 1.0 / e_ref_;
  int ind = 0;
  bool axi_flag = false;
  if (ax_ == 1) axi_flag = true;
  for (int ipoint = 0; ipoint < num_points; ipoint++) {
    GET_SOLUTION_PS(, owner_u);
    GET_SOLUTION_GRAD_PDS(, owner_div_u);

    double* source_pri_jacobi = &source_jacobi[SS_ * ipoint];

    mixture_->SetDensity(&dim_d[0]);
    mixture_->GetSpeciesReactionRateJacobian(dim_T_tr, dim_T_eev, &rate_jac[0]);
    for (int i = 0; i < ns_; i++) {
      for (int j = 0; j < ns_; j++)
        source_pri_jacobi[i * S_ + j] = rate_jac[i * (ns_ + 2) + j] * t_ref;
      source_pri_jacobi[i * S_ + (ns_ + 2)] =
        rate_jac[i * (ns_ + 2) + ns_] * t_ref * T_ref_ * inv_rho_ref;
      source_pri_jacobi[i * S_ + (ns_ + 3)] =
        rate_jac[i * (ns_ + 2) + (ns_ + 1)] * t_ref * T_ref_ * inv_rho_ref;
    }

    mixture_->GetReactionTransferRateJacobian(dim_T_tr, dim_T_eev,
      &transfer_jac[0]);
    for (int j = 0; j < ns_; j++)
      source_pri_jacobi[(ns_ + 3) * S_ + j] =
      transfer_jac[j] * t_ref * inv_e_ref;
    source_pri_jacobi[(ns_ + 3) * S_ + (ns_ + 2)] =
      transfer_jac[ns_] * t_ref * T_ref_ * inv_e_ref;
    source_pri_jacobi[(ns_ + 3) * S_ + (ns_ + 3)] =
      transfer_jac[ns_ + 1] * t_ref * T_ref_ * inv_e_ref;

    mixture_->GetVTTransferRateJacobian(dim_T_tr, dim_T_eev, &transfer_jac[0]);
    for (int j = 0; j < ns_; j++)
      source_pri_jacobi[(ns_ + 3) * S_ + j] +=
      transfer_jac[j] * t_ref * inv_e_ref;
    source_pri_jacobi[(ns_ + 3) * S_ + (ns_ + 2)] +=
      transfer_jac[ns_] * t_ref * T_ref_ * inv_e_ref;
    source_pri_jacobi[(ns_ + 3) * S_ + (ns_ + 3)] +=
      transfer_jac[ns_ + 1] * t_ref * T_ref_ * inv_e_ref;

    if (axi_flag) {
      const double y = cell_points[icell][ipoint * D_ + 1];
      const double fy = static_cast<double>(ax_) * (y - 1.0) + 1.0;
      cblas_dscal(SS_, fy, source_pri_jacobi, 1);

      aDual vaxi(S_, v, ns_ + 1);
      aDual mixture_p(S_);
      mixture_p.f = mixture_->GetPressureJacobian(dim_T_tr, dim_T_eev,
        &mixture_p.df[0]);
      std::swap(mixture_p.df[ns_], mixture_p.df[ns_ + 2]);
      std::swap(mixture_p.df[ns_ + 1], mixture_p.df[ns_ + 3]);
      mixture_p.f /= p_ref_;
      for (int i = 0; i < ns_; i++) mixture_p.df[i] *= rho_ref_ / p_ref_;
      mixture_p.df[ns_ + 2] *= T_ref_ / p_ref_;
      mixture_p.df[ns_ + 3] *= T_ref_ / p_ref_;

      const auto mu = mixture_->GetViscosity(dim_T_tr, dim_T_eev) / mu_ref_;
      const auto ttt = c23_ * mu * (2.0 * vaxi / y - ux - vy);
      const auto source_axi = static_cast<double>(ax_) * (mixture_p - ttt);

      for (int j = 0; j < S_; j++)
        source_pri_jacobi[(ns_ + 1) * S_ + j] += source_axi.df[j];
    }
    // Where is ett?
  }
}
void EquationNS2DNeq2Tnondim::ComputeSolutionJacobi(
  const int num_points, std::vector<double>& jacobi,
  const std::vector<double>& owner_u) {
  const auto& species = mixture_->GetSpecies();
  std::vector<aDual> values(S_, aDual(S_));
  std::vector<double> dim_d(ns_);
  for (int ipoint = 0; ipoint < num_points; ipoint++) {
    const double* sol = &owner_u[ipoint * S_];
    std::vector<aDual> d(ns_, aDual(S_));
    for (int i = 0; i < ns_; i++) {
      dim_d[i] = sol[i] * rho_ref_;
      d[i] = aDual(S_, sol[i], i);
    }
    aDual u(S_, sol[ns_], ns_);
    aDual v(S_, sol[ns_ + 1], ns_ + 1);
    const auto& T_tr = sol[ns_ + 2];
    const auto& T_eev = sol[ns_ + 3];
    const double dim_T_tr = T_tr * T_ref_;
    const double dim_T_eev = T_eev * T_ref_;

    aDual mixture_d(S_);
    for (int i = 0; i < ns_; i++) mixture_d = mixture_d + d[i];

    for (int i = 0; i < ns_; i++) values[i] = d[i];
    values[ns_] = mixture_d * u;
    values[ns_ + 1] = mixture_d * v;

    aDual de(S_, 0.0);
    aDual de_eev(S_, 0.0);

    mixture_->SetDensity(&dim_d[0]);
    for (int i = 0; i < ns_; i++) {
      const auto e = species[i]->GetInternalEnergy(dim_T_tr, dim_T_eev) / e_ref_;
      const auto e_eev =
          species[i]->GetElectronicVibrationEnergy(dim_T_eev) / e_ref_;
      const auto Cv_tr = species[i]->GetTransRotationSpecificHeat(dim_T_tr) * T_ref_ / e_ref_;
      const auto Cv_eev =
          species[i]->GetElectronicVibrationSpecificHeat(dim_T_eev) * T_ref_ /
          e_ref_;

      de.f += d[i].f * e;
      de.df[i] = e;
      de.df[ns_ + 2] += d[i].f * Cv_tr;
      de.df[ns_ + 3] += d[i].f * Cv_eev;

      de_eev.f += d[i].f * e_eev;
      de_eev.df[i] = e_eev;
      de_eev.df[ns_ + 3] += d[i].f * Cv_eev;
    }
    if (dim_T_eev < 250.0 / T_ref_) {
      double sum = 0.0;
      for (int i = 0; i < ns_; i++) {
        const auto Cv_eev =
            species[i]->GetElectronicVibrationSpecificHeat(dim_T_eev + 10.0) * T_ref_ / e_ref_;
        sum += d[i].f * Cv_eev;
      }
      de_eev.df[ns_ + 3] = sum;
      de.df[ns_ + 3] = sum;
    }
    values[ns_ + 2] = de + 0.5 * mixture_d * (u * u + v * v);
    values[ns_ + 3] = de_eev;

    int ind = SS_ * ipoint;
    for (int i = 0; i < S_; i++)
      for (int j = 0; j < S_; j++) jacobi[ind++] = values[i].df[j];
  }
}
void EquationNS2DNeq2Tnondim::ComputeNumFluxLLF(const int num_points,
                                                std::vector<double>& flux,
                                                FACE_INPUTS, const std::vector<double>& coords) {
  const auto& species = mixture_->GetSpecies();
  const auto& hidx = mixture_->GetHeavyParticleIdx();
  const auto& eidx = mixture_->GetElectronIndex();
  const auto& face_points = DENEB_DATA->GetFacePoints();

  static std::vector<double> Ds(ns_, 0.0);
  static std::vector<double> Isx(ns_, 0.0);
  static std::vector<double> Isy(ns_, 0.0);
  static std::vector<double> hs(ns_, 0.0);
  static std::vector<double> diffflux_o(2 * ns_, 0.0);
  static std::vector<double> diffflux_n(2 * ns_, 0.0);
  std::vector<double> owner_sol_jacobi(num_points * SS_, 0.0);
  std::vector<double> neighbor_sol_jacobi(num_points * SS_, 0.0);
  std::vector<double> owner_div_sol(num_points * S_ * 2, 0.0);
  std::vector<double> neighbor_div_sol(num_points * S_ * 2, 0.0);

  int ind = 0;
  double arvis_o = 0.0;
  double arvis_n = 0.0;
  std::vector<double> dim_d_o(ns_);
  std::vector<double> dim_d_n(ns_);
  ComputeSolutionJacobi(num_points, owner_sol_jacobi, owner_u);
  ComputeSolutionJacobi(num_points, neighbor_sol_jacobi, neighbor_u);
  gradpri2gradsol(num_points, &owner_sol_jacobi[0], &owner_div_u[0], &owner_div_sol[0]);
  gradpri2gradsol(num_points, &neighbor_sol_jacobi[0], &neighbor_div_u[0], &neighbor_div_sol[0]);
  for (int ipoint = 0; ipoint < num_points; ipoint++) {
    GET_NORMAL_PD(normal);

    GET_SOLUTION_PS(_o, owner_u);
    GET_SOLUTION_PS(_n, neighbor_u);

    GET_SOLUTION_GRAD_PDS(_o, owner_div_u);
    GET_SOLUTION_GRAD_PDS(_n, neighbor_div_u);

    GET_CONS_SOLUTION_GRAD_PDS(_o, owner_div_sol);
    GET_CONS_SOLUTION_GRAD_PDS(_n, neighbor_div_sol);

    arvis_o = DENEB_ARTIFICIAL_VISCOSITY->GetArtificialViscosityValue(
        owner_cell, ipoint);
    arvis_n = DENEB_ARTIFICIAL_VISCOSITY->GetArtificialViscosityValue(
        neighbor_cell, ipoint);

    const double y = coords[ipoint * D_ + 1];

    // Owner Flux
    // Convective flux
    mixture_->SetDensity(&dim_d_o[0]);
    const double mixture_d_o = mixture_->GetTotalDensity() / rho_ref_;
    const double mixture_p_o = mixture_->GetPressure(dim_T_tr_o, dim_T_eev_o) / p_ref_;
    const double beta_o = mixture_->GetBeta(dim_T_tr_o);
    const double a_o = std::sqrt((1.0 + beta_o) * mixture_p_o / mixture_d_o);
    const double V_o = u_o * nx + v_o * ny;
    double de_o = 0.0;
    double de_eev_o = 0.0;
    for (int i = 0; i < ns_; i++) {
      const double e_o = species[i]->GetInternalEnergy(dim_T_tr_o, dim_T_eev_o) / e_ref_;
      const double e_eev_o =
          species[i]->GetElectronicVibrationEnergy(dim_T_eev_o) / e_ref_;
      de_o += d_o[i] * e_o;
      de_eev_o += d_o[i] * e_eev_o;
    }
    const double du_o = mixture_d_o * u_o;
    const double dv_o = mixture_d_o * v_o;
    const double dE_o = de_o + 0.5 * (du_o * u_o + dv_o * v_o);

    // Viscous flux
    const double mu_o = mixture_->GetViscosity(dim_T_tr_o, dim_T_eev_o) / mu_ref_;
    const double k_tr_o =
        mixture_->GetTransRotationConductivity(dim_T_tr_o, dim_T_eev_o) / k_ref_;
    const double k_eev_o =
        mixture_->GetElectronicVibrationConductivity(dim_T_tr_o, dim_T_eev_o) / k_ref_;

    const double* Y_o = mixture_->GetMassFraction();
    double mixture_dx_o = 0.0;
    double mixture_dy_o = 0.0;
    for (int i = 0; i < ns_; i++) {
      mixture_dx_o += dx_o[i];
      mixture_dy_o += dy_o[i];
    }

    mixture_->GetDiffusivity(&Ds[0], dim_T_tr_o, dim_T_eev_o);
    for (int i = 0; i < ns_; i++) {
      Ds[i] /= D_ref_;
      Isx[i] = -Ds[i] * (dx_o[i] - Y_o[i] * mixture_dx_o);
      Isy[i] = -Ds[i] * (dy_o[i] - Y_o[i] * mixture_dy_o);
    }

    double Ix_sum = 0.0;
    double Iy_sum = 0.0;
    for (const int& idx : hidx) {
      Ix_sum += Isx[idx];
      Iy_sum += Isy[idx];
    }

    for (const int& idx : hidx) {
      diffflux_o[idx] = -Isx[idx] + Y_o[idx] * Ix_sum;
      diffflux_o[idx + ns_] = -Isy[idx] + Y_o[idx] * Iy_sum;
    }

    if (eidx >= 0) {
      double sum_x = 0.0;
      double sum_y = 0.0;
      for (const int& idx : hidx) {
        const int charge = species[idx]->GetCharge();
        const double Mw = species[idx]->GetMolecularWeight();
        sum_x += diffflux_o[idx] * charge / Mw;
        sum_y += diffflux_o[idx + ns_] * charge / Mw;
      }
      const double Mw_e = species[eidx]->GetMolecularWeight();
      diffflux_o[eidx] = Mw_e * sum_x;
      diffflux_o[eidx + ns_] = Mw_e * sum_y;
    }

    auto tau_ax_o = (std::abs(y) < 1.0e-12) ? 0.0 : static_cast<double>(ax_) * v_o / y;
    const auto txx_o = c23_ * mu_o * (2.0 * ux_o - vy_o - tau_ax_o);
    const auto txy_o = mu_o * (uy_o + vx_o);
    const auto tyy_o = c23_ * mu_o * (2.0 * vy_o - ux_o - tau_ax_o);

    double Jh_x_o = 0.0;
    double Jh_y_o = 0.0;
    double Je_eev_x_o = 0.0;
    double Je_eev_y_o = 0.0;
    for (int i = 0; i < ns_; i++) {
      const double h = species[i]->GetEnthalpy(dim_T_tr_o, dim_T_eev_o) / e_ref_;
      const double e_eev =
          species[i]->GetElectronicVibrationEnergy(dim_T_eev_o) / e_ref_;
      Jh_x_o += diffflux_o[i] * h;
      Je_eev_x_o += diffflux_o[i] * e_eev;
      Jh_y_o += diffflux_o[i + ns_] * h;
      Je_eev_y_o += diffflux_o[i + ns_] * e_eev;
    }
    const double q_tr_x_o = k_tr_o * Ttrx_o;
    const double q_tr_y_o = k_tr_o * Ttry_o;
    const double q_eev_x_o = k_eev_o * Teevx_o;
    const double q_eev_y_o = k_eev_o * Teevy_o;

    // Neighbor flux
    // Convective flux
    mixture_->SetDensity(&dim_d_n[0]);
    const double mixture_d_n = mixture_->GetTotalDensity() / rho_ref_;
    const double mixture_p_n = mixture_->GetPressure(dim_T_tr_n, dim_T_eev_n) / p_ref_;
    const double beta_n = mixture_->GetBeta(dim_T_tr_n);
    const double a_n = std::sqrt((1.0 + beta_n) * mixture_p_n / mixture_d_n);
    const double V_n = u_n * nx + v_n * ny;
    double de_n = 0.0;
    double de_eev_n = 0.0;
    for (int i = 0; i < ns_; i++) {
      const double e_n = species[i]->GetInternalEnergy(dim_T_tr_n, dim_T_eev_n) / e_ref_;
      const double e_eev_n = species[i]->GetElectronicVibrationEnergy(dim_T_eev_n) / e_ref_;
      de_n += d_n[i] * e_n;
      de_eev_n += d_n[i] * e_eev_n;
    }
    const double du_n = mixture_d_n * u_n;
    const double dv_n = mixture_d_n * v_n;
    const double dE_n = de_n + 0.5 * (du_n * u_n + dv_n * v_n);

    // Viscous flux
    const double mu_n = mixture_->GetViscosity(dim_T_tr_n, dim_T_eev_n) / mu_ref_;
    const double k_tr_n =
        mixture_->GetTransRotationConductivity(dim_T_tr_n, dim_T_eev_n) / k_ref_;
    const double k_eev_n =
        mixture_->GetElectronicVibrationConductivity(dim_T_tr_n, dim_T_eev_n) / k_ref_;

    const double* Y_n = mixture_->GetMassFraction();
    double mixture_dx_n = 0.0;
    double mixture_dy_n = 0.0;
    for (int i = 0; i < ns_; i++) {
      mixture_dx_n += dx_n[i];
      mixture_dy_n += dy_n[i];
    }

    mixture_->GetDiffusivity(&Ds[0], dim_T_tr_n, dim_T_eev_n);
    for (int i = 0; i < ns_; i++) {
      Ds[i] /= D_ref_;
      Isx[i] = -Ds[i] * (dx_n[i] - Y_n[i] * mixture_dx_n);
      Isy[i] = -Ds[i] * (dy_n[i] - Y_n[i] * mixture_dy_n);
    }

    Ix_sum = 0.0;
    Iy_sum = 0.0;
    for (const int& idx : hidx) {
      Ix_sum += Isx[idx];
      Iy_sum += Isy[idx];
    }

    for (const int& idx : hidx) {
      diffflux_n[idx] = -Isx[idx] + Y_n[idx] * Ix_sum;
      diffflux_n[idx + ns_] = -Isy[idx] + Y_n[idx] * Iy_sum;
    }

    if (eidx >= 0) {
      double sum_x = 0.0;
      double sum_y = 0.0;
      for (const int& idx : hidx) {
        const int charge = species[idx]->GetCharge();
        const double Mw = species[idx]->GetMolecularWeight();
        sum_x += diffflux_n[idx] * charge / Mw;
        sum_y += diffflux_n[idx + ns_] * charge / Mw;
      }
      const double Mw_e = species[eidx]->GetMolecularWeight();
      diffflux_n[eidx] = Mw_e * sum_x;
      diffflux_n[eidx + ns_] = Mw_e * sum_y;
    }

    const auto tau_ax_n = (std::abs(y) < 1.0e-12) ? 0.0 : static_cast<double>(ax_) * v_n / y;
    const auto txx_n = c23_ * mu_n * (2.0 * ux_n - vy_n - tau_ax_n);
    const auto txy_n = mu_n * (uy_n + vx_n);
    const auto tyy_n = c23_ * mu_n * (2.0 * vy_n - ux_n - tau_ax_n);

    double Jh_x_n = 0.0;
    double Jh_y_n = 0.0;
    double Je_eev_x_n = 0.0;
    double Je_eev_y_n = 0.0;
    for (int i = 0; i < ns_; i++) {
      const double h = species[i]->GetEnthalpy(dim_T_tr_n, dim_T_eev_n) / e_ref_;
      const double e_eev =
          species[i]->GetElectronicVibrationEnergy(dim_T_eev_n) / e_ref_;
      Jh_x_n += diffflux_n[i] * h;
      Je_eev_x_n += diffflux_n[i] * e_eev;
      Jh_y_n += diffflux_n[i + ns_] * h;
      Je_eev_y_n += diffflux_n[i + ns_] * e_eev;
    }
    const double q_tr_x_n = k_tr_n * Ttrx_n;
    const double q_tr_y_n = k_tr_n * Ttry_n;
    const double q_eev_x_n = k_eev_n * Teevx_n;
    const double q_eev_y_n = k_eev_n * Teevy_n;

    const double r_max = std::max(std::abs(V_o) + a_o, std::abs(V_n) + a_n);
    std::vector<double> diff(ns_);
    for (int i = 0; i < ns_; i++) diff[i] = r_max * (d_n[i] - d_o[i]);
    const double diff0 = r_max * (du_n - du_o);
    const double diff1 = r_max * (dv_n - dv_o);
    const double diff2 = r_max * (dE_n - dE_o);
    const double diff3 = r_max * (de_eev_n - de_eev_o);

    ind = DS_ * ipoint;
    for (int i = 0; i < ns_; i++)
      flux[ind++] =
      0.5 * (d_o[i] * u_o + d_n[i] * u_n - diffflux_o[i] - diffflux_n[i] -
        arvis_o * dx_o[i] - arvis_n * dx_n[i] - nx * diff[i]);
    flux[ind++] =
      0.5 * (du_o * u_o + mixture_p_o + du_n * u_n + mixture_p_n - txx_o -
        txx_n - arvis_o * dux_o - arvis_n * dux_n - nx * diff0);
    flux[ind++] = 0.5 * (du_o * v_o + du_n * v_n - txy_o - txy_n -
      arvis_o * dvx_o - arvis_n * dvx_n - nx * diff1);
    flux[ind++] =
      0.5 * ((dE_o + mixture_p_o) * u_o + (dE_n + mixture_p_n) * u_n -
        txx_o * u_o - txy_o * v_o - txx_n * u_n - txy_n * v_n -
        q_tr_x_o - q_eev_x_o - Jh_x_o - q_tr_x_n - q_eev_x_n - Jh_x_n -
        arvis_o * dEx_o - arvis_n * dEx_n - nx * diff2);
    flux[ind++] = 0.5 * (de_eev_o * u_o + de_eev_n * u_n - q_eev_x_o -
      Je_eev_x_o - q_eev_x_n - Je_eev_x_n -
      arvis_o * de_eevx_o - arvis_n * de_eevx_n - nx * diff3);
    for (int i = 0; i < ns_; i++)
      flux[ind++] = 0.5 * (d_o[i] * v_o + d_n[i] * v_n - diffflux_o[i + ns_] -
        diffflux_n[i + ns_] - arvis_o * dy_o[i] -
        arvis_n * dy_n[i] - ny * diff[i]);
    flux[ind++] = 0.5 * (dv_o * u_o + dv_n * u_n - txy_o - txy_n -
      arvis_o * duy_o - arvis_n * duy_n - ny * diff0);
    flux[ind++] =
      0.5 * (dv_o * v_o + mixture_p_o + dv_n * v_n + mixture_p_n - tyy_o -
        tyy_n - arvis_o * dvy_o - arvis_n * dvy_n - ny * diff1);
    flux[ind++] =
      0.5 * ((dE_o + mixture_p_o) * v_o + (dE_n + mixture_p_n) * v_n -
        txy_o * u_o - tyy_o * v_o - txy_n * u_n - tyy_n * v_n -
        q_tr_y_o - q_eev_y_o - Jh_y_o - q_tr_y_n - q_eev_y_n - Jh_y_n -
        arvis_o * dEy_o - arvis_n * dEy_n - ny * diff2);
    flux[ind] = 0.5 * (de_eev_o * v_o + de_eev_n * v_n - q_eev_y_o -
      Je_eev_y_o - q_eev_y_n - Je_eev_y_n - arvis_o * de_eevy_o -
      arvis_n * de_eevy_n - ny * diff3);

    const double fy = static_cast<double>(ax_) * (y - 1.0) + 1.0;
    cblas_dscal(DS_, fy, &flux[DS_ * ipoint], 1);
  }
}
void EquationNS2DNeq2Tnondim::ComputeNumFluxJacobiLLF(const int num_points,
                                                      FACE_JACOBI_OUTPUTS,
                                                      FACE_INPUTS, const std::vector<double>& coords) {
  // flux_jacobi(ds1s2) = F(ds1) over U(s2)
  // flux_grad_jacobi(d1s1 s2d2) = F(d1s1) over gradU(d2s2)
  const auto& species = mixture_->GetSpecies();
  const auto& hidx = mixture_->GetHeavyParticleIdx();
  const auto& eidx = mixture_->GetElectronIndex();
  const auto& face_points = DENEB_DATA->GetFacePoints();

  static std::vector<double> Ds(ns_, 0.0);
  static std::vector<double> Isx(ns_, 0.0);
  static std::vector<double> Isy(ns_, 0.0);
  static std::vector<double> hs(ns_, 0.0);

  static std::vector<aDual> flux1(DS_, aDual(S_));
  static std::vector<aDual> flux2(DS_, aDual(DS_));

  std::vector<double> owner_sol_jacobi(num_points * SS_, 0.0);
  std::vector<double> neighbor_sol_jacobi(num_points * SS_, 0.0);
  std::vector<double> owner_div_sol(num_points * S_ * 2, 0.0);
  std::vector<double> neighbor_div_sol(num_points * S_ * 2, 0.0);

  int ind = 0;
  double arvis_o = 0.0;
  double arvis_n = 0.0;

  std::vector<double> dim_d_o(ns_);
  std::vector<double> dim_d_n(ns_);
  ComputeSolutionJacobi(num_points, owner_sol_jacobi, owner_u);
  ComputeSolutionJacobi(num_points, neighbor_sol_jacobi, neighbor_u);
  gradpri2gradsol(num_points, &owner_sol_jacobi[0], &owner_div_u[0], &owner_div_sol[0]);
  gradpri2gradsol(num_points, &neighbor_sol_jacobi[0], &neighbor_div_u[0], &neighbor_div_sol[0]);
  // owner jacobi
  for (int ipoint = 0; ipoint < num_points; ipoint++) {
    GET_NORMAL_PD(normal);
    arvis_o = DENEB_ARTIFICIAL_VISCOSITY->GetArtificialViscosityValue(
        owner_cell, ipoint);
    arvis_n = DENEB_ARTIFICIAL_VISCOSITY->GetArtificialViscosityValue(
        neighbor_cell, ipoint);
    const double y = coords[ipoint * D_ + 1];
    const double fy = static_cast<double>(ax_) * (y - 1.0) + 1.0;
    {
      GET_SOLUTION_PS(_n, neighbor_u);
      GET_SOLUTION_GRAD_PDS(_n, neighbor_div_u);
      GET_CONS_SOLUTION_GRAD_PDS(_n, neighbor_div_sol);

      // Neighbor convective flux
      mixture_->SetDensity(&dim_d_n[0]);
      const double mixture_d_n = mixture_->GetTotalDensity() / rho_ref_;
      const double mixture_p_n = mixture_->GetPressure(dim_T_tr_n, dim_T_eev_n) / p_ref_;
      const double beta_n = mixture_->GetBeta(dim_T_tr_n);
      const double a_n = std::sqrt((1.0 + beta_n) * mixture_p_n / mixture_d_n);
      const double V_n = u_n * nx + v_n * ny;
      double de_n = 0.0;
      double de_eev_n = 0.0;
      for (int i = 0; i < ns_; i++) {
        const double e_n =
            species[i]->GetInternalEnergy(dim_T_tr_n, dim_T_eev_n) / e_ref_;
        const double e_eev_n =
            species[i]->GetElectronicVibrationEnergy(dim_T_eev_n) / e_ref_;
        de_n += d_n[i] * e_n;
        de_eev_n += d_n[i] * e_eev_n;
      }
      const double du_n = mixture_d_n * u_n;
      const double dv_n = mixture_d_n * v_n;
      const double dE_n = de_n + 0.5 * (du_n * u_n + dv_n * v_n);

      // Neighbor viscous flux
      const double mu_n = mixture_->GetViscosity(dim_T_tr_n, dim_T_eev_n) / mu_ref_;
      const double k_tr_n =
          mixture_->GetTransRotationConductivity(dim_T_tr_n, dim_T_eev_n) / k_ref_;
      const double k_eev_n =
          mixture_->GetElectronicVibrationConductivity(dim_T_tr_n, dim_T_eev_n) / k_ref_;

      const double* Y_n = mixture_->GetMassFraction();
      double mixture_dx_n = 0.0;
      double mixture_dy_n = 0.0;
      for (int i = 0; i < ns_; i++) {
        mixture_dx_n += dx_n[i];
        mixture_dy_n += dy_n[i];
      }

      mixture_->GetDiffusivity(&Ds[0], dim_T_tr_n, dim_T_eev_n);
      for (int i = 0; i < ns_; i++) {
        Ds[i] /= D_ref_;
        Isx[i] = -Ds[i] * (dx_n[i] - Y_n[i] * mixture_dx_n);
        Isy[i] = -Ds[i] * (dy_n[i] - Y_n[i] * mixture_dy_n);
      }

      double Ix_sum = 0.0;
      double Iy_sum = 0.0;
      for (const int& idx : hidx) {
        Ix_sum += Isx[idx];
        Iy_sum += Isy[idx];
      }

      std::vector<double> diffflux_n(2 * ns_, 0.0);

      for (const int& idx : hidx) {
        diffflux_n[idx] = -Isx[idx] + Y_n[idx] * Ix_sum;
        diffflux_n[idx + ns_] = -Isy[idx] + Y_n[idx] * Iy_sum;
      }

      if (eidx >= 0) {
        double sum_x = 0.0;
        double sum_y = 0.0;
        for (const int& idx : hidx) {
          const int charge = species[idx]->GetCharge();
          const double Mw = species[idx]->GetMolecularWeight();
          sum_x += diffflux_n[idx] * charge / Mw;
          sum_y += diffflux_n[idx + ns_] * charge / Mw;
        }
        const double Mw_e = species[eidx]->GetMolecularWeight();
        diffflux_n[eidx] = Mw_e * sum_x;
        diffflux_n[eidx + ns_] = Mw_e * sum_y;
      }

      const auto tau_ax_n = (std::abs(y) < 1.0e-12) ? 0.0 : static_cast<double>(ax_) * v_n / y;
      const auto txx_n = c23_ * mu_n * (2.0 * ux_n - vy_n - tau_ax_n);
      const auto txy_n = mu_n * (uy_n + vx_n);
      const auto tyy_n = c23_ * mu_n * (2.0 * vy_n - ux_n - tau_ax_n);

      double Jh_x_n = 0.0;
      double Jh_y_n = 0.0;
      double Je_eev_x_n = 0.0;
      double Je_eev_y_n = 0.0;
      for (int i = 0; i < ns_; i++) {
        const double h = species[i]->GetEnthalpy(dim_T_tr_n, dim_T_eev_n) / e_ref_;
        const double e_eev = species[i]->GetElectronicVibrationEnergy(dim_T_eev_n) / e_ref_;
        Jh_x_n += diffflux_n[i] * h;
        Je_eev_x_n += diffflux_n[i] * e_eev;
        Jh_y_n += diffflux_n[i + ns_] * h;
        Je_eev_y_n += diffflux_n[i + ns_] * e_eev;
      }
      const double q_tr_x_n = k_tr_n * Ttrx_n;
      const double q_tr_y_n = k_tr_n * Ttry_n;
      const double q_eev_x_n = k_eev_n * Teevx_n;
      const double q_eev_y_n = k_eev_n * Teevy_n;
      {
        // flux owner jacobi
        // ps
        ind = S_ * ipoint;
        std::vector<aDual> d_o(ns_, aDual(S_));
        for (int i = 0; i < ns_; i++) {
          dim_d_o[i] = owner_u[ind] * rho_ref_;
          d_o[i] = aDual(S_, owner_u[ind++], i);
        }
        aDual u_o(S_, owner_u[ind++], ns_);
        aDual v_o(S_, owner_u[ind++], ns_ + 1);
        const double& T_tr_o = owner_u[ind++];
        const double& T_eev_o = owner_u[ind];
        const double dim_T_tr_o = T_tr_o * T_ref_;
        const double dim_T_eev_o = T_eev_o * T_ref_;

        GET_SOLUTION_GRAD_PDS(_o, owner_div_u);
        GET_CONS_SOLUTION_GRAD_PDS(_o, owner_div_sol);

        // Owner convective flux
        mixture_->SetDensity(&dim_d_o[0]);
        aDual mixture_d_o(S_, mixture_->GetTotalDensity() / rho_ref_);
        for (int i = 0; i < ns_; i++) mixture_d_o.df[i] = 1.0;

        aDual mixture_p_o(S_);
        mixture_p_o.f = mixture_->GetPressureJacobian(dim_T_tr_o, dim_T_eev_o,
                                                      &mixture_p_o.df[0]);
        std::swap(mixture_p_o.df[ns_], mixture_p_o.df[ns_ + 2]);
        std::swap(mixture_p_o.df[ns_ + 1], mixture_p_o.df[ns_ + 3]);
        mixture_p_o.f /= p_ref_;
        for (int i = 0; i < ns_; i++) mixture_p_o.df[i] *= rho_ref_ / p_ref_;
        mixture_p_o.df[ns_ + 2] *= T_ref_ / p_ref_;
        mixture_p_o.df[ns_ + 3] *= T_ref_ / p_ref_;

        aDual beta_o(S_);
        beta_o.f = mixture_->GetBetaJacobian(dim_T_tr_o, &beta_o.df[0]);
        for (int i = 0; i < ns_; i++) beta_o.df[i] *= rho_ref_;
        beta_o.df[ns_ + 2] *= T_ref_;
        beta_o.df[ns_ + 3] *= T_ref_;

        const auto a_o = std::sqrt((1.0 + beta_o) * mixture_p_o / mixture_d_o);
        const auto V_o = u_o * nx + v_o * ny;
        aDual de_o(S_);
        aDual de_eev_o(S_);
        for (int i = 0; i < ns_; i++) {
          const auto e = species[i]->GetInternalEnergy(dim_T_tr_o, dim_T_eev_o) / e_ref_;
          const auto e_eev =
              species[i]->GetElectronicVibrationEnergy(dim_T_eev_o) / e_ref_;
          const auto Cv_tr =
              species[i]->GetTransRotationSpecificHeat(dim_T_tr_o) * T_ref_ /
              e_ref_;
          const auto Cv_eev =
              species[i]->GetElectronicVibrationSpecificHeat(dim_T_eev_o) *
              T_ref_ / e_ref_;

          de_o.f += d_o[i].f * e;
          de_o.df[i] = e;
          de_o.df[ns_ + 2] += d_o[i].f * Cv_tr;
          de_o.df[ns_ + 3] += d_o[i].f * Cv_eev;

          de_eev_o.f += d_o[i].f * e_eev;
          de_eev_o.df[i] = e_eev;
          de_eev_o.df[ns_ + 3] += d_o[i].f * Cv_eev;
        }
        const auto du_o = mixture_d_o * u_o;
        const auto dv_o = mixture_d_o * v_o;
        const auto dE_o = de_o + 0.5 * (du_o * u_o + dv_o * v_o);

        // Viscous flux
        const auto mu_o =
            mixture_->GetViscosity(dim_T_tr_o, dim_T_eev_o) / mu_ref_;
        const auto k_tr_o =
            mixture_->GetTransRotationConductivity(dim_T_tr_o, dim_T_eev_o) /
            k_ref_;
        const auto k_eev_o = mixture_->GetElectronicVibrationConductivity(
                                 dim_T_tr_o, dim_T_eev_o) /
                             k_ref_;

        std::vector<aDual> Y_o(ns_, aDual(S_));
        for (int i = 0; i < ns_; i++) Y_o[i] = d_o[i] / mixture_d_o;

        double mixture_dx_o = 0.0;
        double mixture_dy_o = 0.0;
        for (int i = 0; i < ns_; i++) {
          mixture_dx_o += dx_o[i];
          mixture_dy_o += dy_o[i];
        }

        std::vector<aDual> Isx_o(ns_, aDual(S_));
        std::vector<aDual> Isy_o(ns_, aDual(S_));
        mixture_->GetDiffusivity(&Ds[0], dim_T_tr_o, dim_T_eev_o);
        for (int i = 0; i < ns_; i++) {
          Ds[i] /= D_ref_;
          Isx_o[i] = -Ds[i] * (dx_o[i] - Y_o[i] * mixture_dx_o);
          Isy_o[i] = -Ds[i] * (dy_o[i] - Y_o[i] * mixture_dy_o);
        }

        aDual Ix_sum_o(S_);
        aDual Iy_sum_o(S_);
        for (const int& idx : hidx) {
          Ix_sum_o = Ix_sum_o + Isx_o[idx];
          Iy_sum_o = Iy_sum_o + Isy_o[idx];
        }

        std::vector<aDual> diffflux_o(2 * ns_, aDual(S_));
        for (const int& idx : hidx) {
          diffflux_o[idx] = -Isx_o[idx] + Y_o[idx] * Ix_sum_o;
          diffflux_o[idx + ns_] = -Isy_o[idx] + Y_o[idx] * Iy_sum_o;
        }
        if (eidx >= 0) {
          aDual sum_x(S_);
          aDual sum_y(S_);
          for (const int& idx : hidx) {
            const int charge = species[idx]->GetCharge();
            const double Mw = species[idx]->GetMolecularWeight();
            sum_x = sum_x + diffflux_o[idx] * charge / Mw;
            sum_y = sum_y + diffflux_o[idx + ns_] * charge / Mw;
          }
          const double Mw_e = species[eidx]->GetMolecularWeight();
          diffflux_o[eidx] = Mw_e * sum_x;
          diffflux_o[eidx + ns_] = Mw_e * sum_y;
        }

        const auto tau_ax_o = (std::abs(y) < 1.0e-12) ? aDual(S_, 0.0) : static_cast<double>(ax_)* v_o / y;
        const auto txx_o = c23_ * mu_o * (2.0 * ux_o - vy_o - tau_ax_o);
        const auto txy_o = mu_o * (uy_o + vx_o);
        const auto tyy_o = c23_ * mu_o * (2.0 * vy_o - ux_o - tau_ax_o);

        aDual Jh_x_o(S_);
        aDual Jh_y_o(S_);
        aDual Je_eev_x_o(S_);
        aDual Je_eev_y_o(S_);
        for (int i = 0; i < ns_; i++) {
          const double h =
              species[i]->GetEnthalpy(dim_T_tr_o, dim_T_eev_o) / e_ref_;
          const double e_eev =
              species[i]->GetElectronicVibrationEnergy(dim_T_eev_o) / e_ref_;
          const auto Cv_tr =
              species[i]->GetTransRotationSpecificHeat(dim_T_tr_o) * T_ref_ /
              e_ref_;
          const auto Cv_eev =
              species[i]->GetElectronicVibrationSpecificHeat(dim_T_eev_o) *
              T_ref_ / e_ref_;
          const double R =
              species[i]->GetSpecificGasConstant() * T_ref_ / e_ref_;

          const auto Cp_tr = Cv_tr + (i == eidx) ? 0.0 : R;
          const auto Cp_eev = Cv_eev + (i == eidx) ? R : 0.0;

          Jh_x_o = Jh_x_o + diffflux_o[i] * h;
          Jh_x_o.df[ns_ + 2] += diffflux_o[i].f * Cp_tr;
          Jh_x_o.df[ns_ + 3] += diffflux_o[i].f * Cp_eev;
          Jh_y_o = Jh_y_o + diffflux_o[i + ns_] * h;
          Jh_y_o.df[ns_ + 2] += diffflux_o[i + ns_].f * Cp_tr;
          Jh_y_o.df[ns_ + 3] += diffflux_o[i + ns_].f * Cp_eev;

          Je_eev_x_o = Je_eev_x_o + diffflux_o[i] * e_eev;
          Je_eev_x_o.df[ns_ + 3] += diffflux_o[i].f * Cv_eev;
          Je_eev_y_o = Je_eev_y_o + diffflux_o[i + ns_] * e_eev;
          Je_eev_y_o.df[ns_ + 3] += diffflux_o[i + ns_].f * Cv_eev;
        }

        const aDual q_tr_x_o(S_, k_tr_o * Ttrx_o);
        const aDual q_tr_y_o(S_, k_tr_o * Ttry_o);
        const aDual q_eev_x_o(S_, k_eev_o * Teevx_o);
        const aDual q_eev_y_o(S_, k_eev_o * Teevy_o);

        const auto r_max = std::max(std::abs(V_o) + a_o, std::abs(V_n) + a_n);
        std::vector<aDual> diff(ns_, aDual(S_));
        for (int i = 0; i < ns_; i++) diff[i] = r_max * (d_n[i] - d_o[i]);
        const auto diff0 = r_max * (du_n - du_o);
        const auto diff1 = r_max * (dv_n - dv_o);
        const auto diff2 = r_max * (dE_n - dE_o);
        const auto diff3 = r_max * (de_eev_n - de_eev_o);

        ind = 0;
        for (int i = 0; i < ns_; i++)
          flux1[ind++] = 0.5 * (d_o[i] * u_o + d_n[i] * u_n - diffflux_o[i] -
            diffflux_n[i] - arvis_o * dx_o[i] -
            arvis_n * dx_n[i] - nx * diff[i]);
        flux1[ind++] =
          0.5 * (du_o * u_o + mixture_p_o + du_n * u_n + mixture_p_n - txx_o -
            txx_n - arvis_o * dux_o - arvis_n * dux_n - nx * diff0);
        flux1[ind++] = 0.5 * (du_o * v_o + du_n * v_n - txy_o - txy_n -
          arvis_o * dvx_o - arvis_n * dvx_n - nx * diff1);
        flux1[ind++] =
          0.5 * ((dE_o + mixture_p_o) * u_o + (dE_n + mixture_p_n) * u_n -
            txx_o * u_o - txy_o * v_o - txx_n * u_n - txy_n * v_n -
            q_tr_x_o - q_eev_x_o - Jh_x_o - q_tr_x_n - q_eev_x_n -
            Jh_x_n - arvis_o * dEx_o - arvis_n * dEx_n - nx * diff2);
        flux1[ind++] =
          0.5 * (de_eev_o * u_o + de_eev_n * u_n - q_eev_x_o - Je_eev_x_o -
            q_eev_x_n - Je_eev_x_n - arvis_o * de_eevx_o -
            arvis_n * de_eevx_n - nx * diff3);
        for (int i = 0; i < ns_; i++)
          flux1[ind++] =
          0.5 * (d_o[i] * v_o + d_n[i] * v_n - diffflux_o[i + ns_] -
            diffflux_n[i + ns_] - arvis_o * dy_o[i] -
            arvis_n * dy_n[i] - ny * diff[i]);
        flux1[ind++] = 0.5 * (dv_o * u_o + dv_n * u_n - txy_o - txy_n -
          arvis_o * duy_o - arvis_n * duy_n - ny * diff0);
        flux1[ind++] =
          0.5 * (dv_o * v_o + mixture_p_o + dv_n * v_n + mixture_p_n - tyy_o -
            tyy_n - arvis_o * dvy_o - arvis_n * dvy_n - ny * diff1);
        flux1[ind++] =
          0.5 * ((dE_o + mixture_p_o) * v_o + (dE_n + mixture_p_n) * v_n -
            txy_o * u_o - tyy_o * v_o - txy_n * u_n - tyy_n * v_n -
            q_tr_y_o - q_eev_y_o - Jh_y_o - q_tr_y_n - q_eev_y_n -
            Jh_y_n - arvis_o * dEy_o - arvis_n * dEy_n - ny * diff2);
        flux1[ind] = 0.5 * (de_eev_o * v_o + de_eev_n * v_n - q_eev_y_o -
          Je_eev_y_o - q_eev_y_n - Je_eev_y_n -
          arvis_o * de_eevy_o - arvis_n * de_eevy_n - ny * diff3);

        // pdss
        ind = DSS_ * ipoint;
        for (int ds = 0; ds < DS_; ds++)
          for (int istate = 0; istate < S_; istate++)
            flux_owner_jacobi[ind++] = flux1[ds].df[istate];

        cblas_dscal(DSS_, fy, &flux_owner_jacobi[DSS_ * ipoint], 1);
      }
      {
        // flux owner grad jacobi
        GET_SOLUTION_PS(_o, owner_u);

        // pds over psd
        ind = DS_ * ipoint;
        std::vector<aDual> dx_o(ns_, aDual(DS_));
        for (int i = 0; i < ns_; i++)
          dx_o[i] = aDual(DS_, owner_div_u[ind++], 2 * i);
        const aDual ux_o(DS_, owner_div_u[ind++], 2 * ns_);
        const aDual vx_o(DS_, owner_div_u[ind++], 2 * (ns_ + 1));
        const aDual Ttrx_o(DS_, owner_div_u[ind++], 2 * (ns_ + 2));
        const aDual Teevx_o(DS_, owner_div_u[ind++], 2 * (ns_ + 3));
        std::vector<aDual> dy_o(ns_, aDual(DS_));
        for (int i = 0; i < ns_; i++)
          dy_o[i] = aDual(DS_, owner_div_u[ind++], 2 * i + 1);
        const aDual uy_o(DS_, owner_div_u[ind++], 2 * ns_ + 1);
        const aDual vy_o(DS_, owner_div_u[ind++], 2 * (ns_ + 1) + 1);
        const aDual Ttry_o(DS_, owner_div_u[ind++], 2 * (ns_ + 2) + 1);
        const aDual Teevy_o(DS_, owner_div_u[ind], 2 * (ns_ + 3) + 1);

        // Owner viscous flux
        mixture_->SetDensity(&dim_d_o[0]);
        const auto mu_o = mixture_->GetViscosity(dim_T_tr_o, dim_T_eev_o) / mu_ref_;
        const auto k_tr_o =
            mixture_->GetTransRotationConductivity(dim_T_tr_o, dim_T_eev_o) / k_ref_;
        const auto k_eev_o = mixture_->GetElectronicVibrationConductivity(
            dim_T_tr_o, dim_T_eev_o) / k_ref_;

        const double* Y_o = mixture_->GetMassFraction();
        aDual mixture_dx_o(DS_);
        aDual mixture_dy_o(DS_);
        for (int i = 0; i < ns_; i++) {
          mixture_dx_o = mixture_dx_o + dx_o[i];
          mixture_dy_o = mixture_dy_o + dy_o[i];
        }

        std::vector<aDual> Isx_o(ns_, aDual(DS_));
        std::vector<aDual> Isy_o(ns_, aDual(DS_));
        mixture_->GetDiffusivity(&Ds[0], dim_T_tr_o, dim_T_eev_o);
        for (int i = 0; i < ns_; i++) {
          Ds[i] /= D_ref_;
          Isx_o[i] = -Ds[i] * (dx_o[i] - Y_o[i] * mixture_dx_o);
          Isy_o[i] = -Ds[i] * (dy_o[i] - Y_o[i] * mixture_dy_o);
        }

        aDual Ix_sum_o(DS_);
        aDual Iy_sum_o(DS_);
        for (const int& idx : hidx) {
          Ix_sum_o = Ix_sum_o + Isx_o[idx];
          Iy_sum_o = Iy_sum_o + Isy_o[idx];
        }

        std::vector<aDual> diffflux_o(2 * ns_, aDual(DS_));

        for (const int& idx : hidx) {
          diffflux_o[idx] = -Isx_o[idx] + Y_o[idx] * Ix_sum_o;
          diffflux_o[idx + ns_] = -Isy_o[idx] + Y_o[idx] * Iy_sum_o;
        }
        if (eidx >= 0) {
          aDual sum_x(DS_);
          aDual sum_y(DS_);
          for (const int& idx : hidx) {
            const int charge = species[idx]->GetCharge();
            const double Mw = species[idx]->GetMolecularWeight();
            sum_x = sum_x + diffflux_o[idx] * charge / Mw;
            sum_y = sum_y + diffflux_o[idx + ns_] * charge / Mw;
          }
          const double Mw_e = species[eidx]->GetMolecularWeight();
          diffflux_o[eidx] = Mw_e * sum_x;
          diffflux_o[eidx + ns_] = Mw_e * sum_y;
        }

        const auto tau_ax_o = (std::abs(y) < 1.0e-12) ? aDual(DS_, 0.0) : aDual(DS_, static_cast<double>(ax_) * v_o / y);
        const auto txx_o = c23_ * mu_o * (2.0 * ux_o - vy_o - tau_ax_o);
        const auto txy_o = mu_o * (uy_o + vx_o);
        const auto tyy_o = c23_ * mu_o * (2.0 * vy_o - ux_o - tau_ax_o);

        aDual Jh_x_o(DS_);
        aDual Jh_y_o(DS_);
        aDual Je_eev_x_o(DS_);
        aDual Je_eev_y_o(DS_);

        for (int i = 0; i < ns_; i++) {
          const double h = species[i]->GetEnthalpy(dim_T_tr_o, dim_T_eev_o) / e_ref_;
          const double e_eev =
              species[i]->GetElectronicVibrationEnergy(dim_T_eev_o) / e_ref_;

          Jh_x_o = Jh_x_o + diffflux_o[i] * h;
          Jh_y_o = Jh_y_o + diffflux_o[i + ns_] * h;
          Je_eev_x_o = Je_eev_x_o + diffflux_o[i] * e_eev;
          Je_eev_y_o = Je_eev_y_o + diffflux_o[i + ns_] * e_eev;
        }

        const auto q_tr_x_o = k_tr_o * Ttrx_o;
        const auto q_tr_y_o = k_tr_o * Ttry_o;
        const auto q_eev_x_o = k_eev_o * Teevx_o;
        const auto q_eev_y_o = k_eev_o * Teevy_o;

        ind = 0;
        for (int i = 0; i < ns_; i++)
          flux2[ind++] = -0.5 * (diffflux_o[i] + diffflux_n[i]);
        flux2[ind++] = -0.5 * (txx_o + txx_n);
        flux2[ind++] = -0.5 * (txy_o + txy_n);
        flux2[ind++] = -0.5 * (txx_o * u_o + txy_o * v_o + txx_n * u_n +
          txy_n * v_n + q_tr_x_o + q_eev_x_o + Jh_x_o +
          q_tr_x_n + q_eev_x_n + Jh_x_n);
        flux2[ind++] = -0.5 * (q_eev_x_o + Je_eev_x_o + q_eev_x_n + Je_eev_x_n);
        for (int i = 0; i < ns_; i++)
          flux2[ind++] = -0.5 * (diffflux_o[i + ns_] + diffflux_n[i + ns_]);
        flux2[ind++] = -0.5 * (txy_o + txy_n);
        flux2[ind++] = -0.5 * (tyy_o + tyy_n);
        flux2[ind++] = -0.5 * (txy_o * u_o + tyy_o * v_o + txy_n * u_n +
          tyy_n * v_n + q_tr_y_o + q_eev_y_o + Jh_y_o +
          q_tr_y_n + q_eev_y_n + Jh_y_n);
        flux2[ind] = -0.5 * (q_eev_y_o + Je_eev_y_o + q_eev_y_n + Je_eev_y_n);

        // pdssd
        ind = DDSS_ * ipoint;
        for (int ds = 0; ds < DS_; ds++)
          for (int sd = 0; sd < DS_; sd++)
            flux_owner_grad_jacobi[ind++] = flux2[ds].df[sd];

        for (int istate = 0; istate < S_; istate++)
          for (int jstate = 0; jstate < S_; jstate++)
            for (int idim = 0; idim < D_; idim++) {
              const int irow = idim * S_ + istate;
              const int icolumn = jstate * D_ + idim;
              flux_owner_grad_jacobi[DDSS_ * ipoint + irow * DS_ + icolumn] -=
                0.5 * arvis_o * owner_sol_jacobi[ipoint * SS_ + istate * S_ + jstate];
            }

        cblas_dscal(DDSS_, fy, &flux_owner_grad_jacobi[DDSS_ * ipoint], 1);
      }
    }
    {
      // flux neighbor jacobi & flux neighbor grad jacobi
      GET_SOLUTION_PS(_o, owner_u);
      GET_SOLUTION_GRAD_PDS(_o, owner_div_u);
      GET_CONS_SOLUTION_GRAD_PDS(_o, owner_div_sol);

      // Owner convective flux
      mixture_->SetDensity(&dim_d_o[0]);
      const double mixture_d_o = mixture_->GetTotalDensity() / rho_ref_;
      const double mixture_p_o = mixture_->GetPressure(dim_T_tr_o, dim_T_eev_o) / p_ref_;
      const double beta_o = mixture_->GetBeta(dim_T_tr_o);
      const double a_o = std::sqrt((1.0 + beta_o) * mixture_p_o / mixture_d_o);
      const double V_o = u_o * nx + v_o * ny;
      double de_o = 0.0;
      double de_eev_o = 0.0;
      for (int i = 0; i < ns_; i++) {
        const double e_o =
            species[i]->GetInternalEnergy(dim_T_tr_o, dim_T_eev_o) / e_ref_;
        const double e_eev_o =
            species[i]->GetElectronicVibrationEnergy(dim_T_eev_o) / e_ref_;
        de_o += d_o[i] * e_o;
        de_eev_o += d_o[i] * e_eev_o;
      }
      const double du_o = mixture_d_o * u_o;
      const double dv_o = mixture_d_o * v_o;
      const double dE_o = de_o + 0.5 * (du_o * u_o + dv_o * v_o);

      // Owner viscous flux
      const double mu_o = mixture_->GetViscosity(dim_T_tr_o, dim_T_eev_o) / mu_ref_;
      const double k_tr_o =
          mixture_->GetTransRotationConductivity(dim_T_tr_o, dim_T_eev_o) / k_ref_;
      const double k_eev_o =
          mixture_->GetElectronicVibrationConductivity(dim_T_tr_o, dim_T_eev_o) / k_ref_;

      const double* Y_o = mixture_->GetMassFraction();
      double mixture_dx_o = 0.0;
      double mixture_dy_o = 0.0;
      for (int i = 0; i < ns_; i++) {
        mixture_dx_o += dx_o[i];
        mixture_dy_o += dy_o[i];
      }

      mixture_->GetDiffusivity(&Ds[0], dim_T_tr_o, dim_T_eev_o);
      for (int i = 0; i < ns_; i++) {
        Ds[i] /= D_ref_;
        Isx[i] = -Ds[i] * (dx_o[i] - Y_o[i] * mixture_dx_o);
        Isy[i] = -Ds[i] * (dy_o[i] - Y_o[i] * mixture_dy_o);
      }

      double Ix_sum = 0.0;
      double Iy_sum = 0.0;
      for (const int& idx : hidx) {
        Ix_sum += Isx[idx];
        Iy_sum += Isy[idx];
      }

      std::vector<double> diffflux_o(2 * ns_, 0.0);

      for (const int& idx : hidx) {
        diffflux_o[idx] = -Isx[idx] + Y_o[idx] * Ix_sum;
        diffflux_o[idx + ns_] = -Isy[idx] + Y_o[idx] * Iy_sum;
      }

      if (eidx >= 0) {
        double sum_x = 0.0;
        double sum_y = 0.0;
        for (const int& idx : hidx) {
          const int charge = species[idx]->GetCharge();
          const double Mw = species[idx]->GetMolecularWeight();
          sum_x += diffflux_o[idx] * charge / Mw;
          sum_y += diffflux_o[idx + ns_] * charge / Mw;
        }
        const double Mw_e = species[eidx]->GetMolecularWeight();
        diffflux_o[eidx] = Mw_e * sum_x;
        diffflux_o[eidx + ns_] = Mw_e * sum_y;
      }

      const auto tau_ax_o = (std::abs(y) < 1.0e-12) ? 0.0 : static_cast<double>(ax_) * v_o / y;
      const auto txx_o = c23_ * mu_o * (2.0 * ux_o - vy_o - tau_ax_o);
      const auto txy_o = mu_o * (uy_o + vx_o);
      const auto tyy_o = c23_ * mu_o * (2.0 * vy_o - ux_o - tau_ax_o);

      double Jh_x_o = 0.0;
      double Jh_y_o = 0.0;
      double Je_eev_x_o = 0.0;
      double Je_eev_y_o = 0.0;
      for (int i = 0; i < ns_; i++) {
        const double h = species[i]->GetEnthalpy(dim_T_tr_o, dim_T_eev_o) / e_ref_;
        const double e_eev =
            species[i]->GetElectronicVibrationEnergy(dim_T_eev_o) / e_ref_;
        Jh_x_o += diffflux_o[i] * h;
        Je_eev_x_o += diffflux_o[i] * e_eev;
        Jh_y_o += diffflux_o[i + ns_] * h;
        Je_eev_y_o += diffflux_o[i + ns_] * e_eev;
      }
      const double q_tr_x_o = k_tr_o * Ttrx_o;
      const double q_tr_y_o = k_tr_o * Ttry_o;
      const double q_eev_x_o = k_eev_o * Teevx_o;
      const double q_eev_y_o = k_eev_o * Teevy_o;
      {
        // flux neighbor jacobi
        // ps
        ind = S_ * ipoint;
        std::vector<aDual> d_n(ns_, aDual(S_));        
        for (int i = 0; i < ns_; i++) {
          dim_d_n[i] = neighbor_u[ind] * rho_ref_;
          d_n[i] = aDual(S_, neighbor_u[ind++], i);
        }
        aDual u_n(S_, neighbor_u[ind++], ns_);
        aDual v_n(S_, neighbor_u[ind++], ns_ + 1);
        const double& T_tr_n = neighbor_u[ind++];
        const double& T_eev_n = neighbor_u[ind];
        const double dim_T_tr_n = T_tr_n * T_ref_;
        const double dim_T_eev_n = T_eev_n * T_ref_;

        GET_SOLUTION_GRAD_PDS(_n, neighbor_div_u);
        GET_CONS_SOLUTION_GRAD_PDS(_n, neighbor_div_sol);

        // Owner convective flux
        mixture_->SetDensity(&dim_d_n[0]);
        aDual mixture_d_n(S_, mixture_->GetTotalDensity() / rho_ref_);
        for (int i = 0; i < ns_; i++) mixture_d_n.df[i] = 1.0;

        aDual mixture_p_n(S_);
        mixture_p_n.f =
            mixture_->GetPressureJacobian(dim_T_tr_n, dim_T_eev_n, &mixture_p_n.df[0]);
        std::swap(mixture_p_n.df[ns_], mixture_p_n.df[ns_ + 2]);
        std::swap(mixture_p_n.df[ns_ + 1], mixture_p_n.df[ns_ + 3]);
        mixture_p_n.f /= p_ref_;
        for (int i = 0; i < ns_; i++) mixture_p_n.df[i] *= rho_ref_ / p_ref_;
        mixture_p_n.df[ns_ + 2] *= T_ref_ / p_ref_;
        mixture_p_n.df[ns_ + 3] *= T_ref_ / p_ref_;

        aDual beta_n(S_);
        beta_n.f = mixture_->GetBetaJacobian(dim_T_tr_n, &beta_n.df[0]);
        for (int i = 0; i < ns_; i++) beta_n.df[i] *= rho_ref_;
        beta_n.df[ns_ + 2] *= T_ref_;
        beta_n.df[ns_ + 3] *= T_ref_;

        const auto a_n = std::sqrt((1.0 + beta_n) * mixture_p_n / mixture_d_n);
        const auto V_n = u_n * nx + v_n * ny;
        aDual de_n(S_);
        aDual de_eev_n(S_);
        for (int i = 0; i < ns_; i++) {
          const auto e = species[i]->GetInternalEnergy(dim_T_tr_n, dim_T_eev_n) / e_ref_;
          const auto e_eev =
              species[i]->GetElectronicVibrationEnergy(dim_T_eev_n) / e_ref_;
          const auto Cv_tr =
              species[i]->GetTransRotationSpecificHeat(dim_T_tr_n) * T_ref_ /
              e_ref_;
          const auto Cv_eev =
              species[i]->GetElectronicVibrationSpecificHeat(dim_T_eev_n) *
              T_ref_ / e_ref_;

          de_n.f += d_n[i].f * e;
          de_n.df[i] = e;
          de_n.df[ns_ + 2] += d_n[i].f * Cv_tr;
          de_n.df[ns_ + 3] += d_n[i].f * Cv_eev;

          de_eev_n.f += d_n[i].f * e_eev;
          de_eev_n.df[i] = e_eev;
          de_eev_n.df[ns_ + 3] += d_n[i].f * Cv_eev;
        }
        const auto du_n = mixture_d_n * u_n;
        const auto dv_n = mixture_d_n * v_n;
        const auto dE_n = de_n + 0.5 * (du_n * u_n + dv_n * v_n);

        // Viscous flux
        const auto mu_n = mixture_->GetViscosity(dim_T_tr_n, dim_T_eev_n) / mu_ref_;
        const auto k_tr_n =
            mixture_->GetTransRotationConductivity(dim_T_tr_n, dim_T_eev_n) / k_ref_;
        const auto k_eev_n = mixture_->GetElectronicVibrationConductivity(
            dim_T_tr_n, dim_T_eev_n) / k_ref_;

        std::vector<aDual> Y_n(ns_, aDual(S_));
        for (int i = 0; i < ns_; i++) Y_n[i] = d_n[i] / mixture_d_n;

        double mixture_dx_n = 0.0;
        double mixture_dy_n = 0.0;
        for (int i = 0; i < ns_; i++) {
          mixture_dx_n += dx_n[i];
          mixture_dy_n += dy_n[i];
        }

        std::vector<aDual> Isx_n(ns_, aDual(S_));
        std::vector<aDual> Isy_n(ns_, aDual(S_));
        mixture_->GetDiffusivity(&Ds[0], dim_T_tr_n, dim_T_eev_n);
        for (int i = 0; i < ns_; i++) {
          Ds[i] /= D_ref_;
          Isx_n[i] = -Ds[i] * (dx_n[i] - Y_n[i] * mixture_dx_n);
          Isy_n[i] = -Ds[i] * (dy_n[i] - Y_n[i] * mixture_dy_n);
        }

        aDual Ix_sum_n(S_);
        aDual Iy_sum_n(S_);
        for (const int& idx : hidx) {
          Ix_sum_n = Ix_sum_n + Isx_n[idx];
          Iy_sum_n = Iy_sum_n + Isy_n[idx];
        }

        std::vector<aDual> diffflux_n(2 * ns_, aDual(S_));
        for (const int& idx : hidx) {
          diffflux_n[idx] = -Isx_n[idx] + Y_n[idx] * Ix_sum_n;
          diffflux_n[idx + ns_] = -Isy_n[idx] + Y_n[idx] * Iy_sum_n;
        }
        if (eidx >= 0) {
          aDual sum_x(S_);
          aDual sum_y(S_);
          for (const int& idx : hidx) {
            const int charge = species[idx]->GetCharge();
            const double Mw = species[idx]->GetMolecularWeight();
            sum_x = sum_x + diffflux_n[idx] * charge / Mw;
            sum_y = sum_y + diffflux_n[idx + ns_] * charge / Mw;
          }
          const double Mw_e = species[eidx]->GetMolecularWeight();
          diffflux_n[eidx] = Mw_e * sum_x;
          diffflux_n[eidx + ns_] = Mw_e * sum_y;
        }

        const auto tau_ax_n = (std::abs(y) < 1.0e-12) ? aDual(S_, 0.0) : static_cast<double>(ax_) * v_n / y;
        const auto txx_n = c23_ * mu_n * (2.0 * ux_n - vy_n);
        const auto txy_n = mu_n * (uy_n + vx_n);
        const auto tyy_n = c23_ * mu_n * (2.0 * vy_n - ux_n);

        aDual Jh_x_n(S_);
        aDual Jh_y_n(S_);
        aDual Je_eev_x_n(S_);
        aDual Je_eev_y_n(S_);
        for (int i = 0; i < ns_; i++) {
          const double h =
              species[i]->GetEnthalpy(dim_T_tr_n, dim_T_eev_n) / e_ref_;
          const double e_eev =
              species[i]->GetElectronicVibrationEnergy(dim_T_eev_n) / e_ref_;
          const auto Cv_tr =
              species[i]->GetTransRotationSpecificHeat(dim_T_tr_n) * T_ref_ /
              e_ref_;
          const auto Cv_eev =
              species[i]->GetElectronicVibrationSpecificHeat(dim_T_eev_n) *
              T_ref_ / e_ref_;
          const double R =
              species[i]->GetSpecificGasConstant() * T_ref_ / e_ref_;

          const auto Cp_tr = Cv_tr + (i == eidx) ? 0.0 : R;
          const auto Cp_eev = Cv_eev + (i == eidx) ? R : 0.0;

          Jh_x_n = Jh_x_n + diffflux_n[i] * h;
          Jh_x_n.df[ns_ + 2] += diffflux_n[i].f * Cp_tr;
          Jh_x_n.df[ns_ + 3] += diffflux_n[i].f * Cp_eev;
          Jh_y_n = Jh_y_n + diffflux_n[i + ns_] * h;
          Jh_y_n.df[ns_ + 2] += diffflux_n[i + ns_].f * Cp_tr;
          Jh_y_n.df[ns_ + 3] += diffflux_n[i + ns_].f * Cp_eev;

          Je_eev_x_n = Je_eev_x_n + diffflux_n[i] * e_eev;
          Je_eev_x_n.df[ns_ + 3] += diffflux_n[i].f * Cv_eev;
          Je_eev_y_n = Je_eev_y_n + diffflux_n[i + ns_] * e_eev;
          Je_eev_y_n.df[ns_ + 3] += diffflux_n[i + ns_].f * Cv_eev;
        }

        const aDual q_tr_x_n(S_, k_tr_n * Ttrx_n);
        const aDual q_tr_y_n(S_, k_tr_n * Ttry_n);
        const aDual q_eev_x_n(S_, k_eev_n * Teevx_n);
        const aDual q_eev_y_n(S_, k_eev_n * Teevy_n);

        const auto r_max = std::max(std::abs(V_o) + a_o, std::abs(V_n) + a_n);
        std::vector<aDual> diff(ns_, aDual(S_));
        for (int i = 0; i < ns_; i++) diff[i] = r_max * (d_n[i] - d_o[i]);
        const auto diff0 = r_max * (du_n - du_o);
        const auto diff1 = r_max * (dv_n - dv_o);
        const auto diff2 = r_max * (dE_n - dE_o);
        const auto diff3 = r_max * (de_eev_n - de_eev_o);

        ind = 0;
        for (int i = 0; i < ns_; i++)
          flux1[ind++] = 0.5 * (d_o[i] * u_o + d_n[i] * u_n - diffflux_o[i] -
            diffflux_n[i] - arvis_o * dx_o[i] -
            arvis_n * dx_n[i] - nx * diff[i]);
        flux1[ind++] = 0.5 * (du_o * u_o + mixture_p_o + du_n * u_n +
          mixture_p_n - txx_o - txx_n - arvis_o * dux_o -
          arvis_n * dux_n - nx * diff0);
        flux1[ind++] = 0.5 * (du_o * v_o + du_n * v_n - txy_o - txy_n -
          arvis_o * dvx_o - arvis_n * dvx_n - nx * diff1);
        flux1[ind++] =
          0.5 * ((dE_o + mixture_p_o) * u_o + (dE_n + mixture_p_n) * u_n -
            txx_o * u_o - txy_o * v_o - txx_n * u_n - txy_n * v_n -
            q_tr_x_o - q_eev_x_o - Jh_x_o - q_tr_x_n - q_eev_x_n -
            Jh_x_n - arvis_o * dEx_o - arvis_n * dEx_n - nx * diff2);
        flux1[ind++] =
          0.5 * (de_eev_o * u_o + de_eev_n * u_n - q_eev_x_o - Je_eev_x_o -
            q_eev_x_n - Je_eev_x_n - arvis_o * de_eevx_o -
            arvis_n * de_eevx_n - nx * diff3);
        for (int i = 0; i < ns_; i++)
          flux1[ind++] =
          0.5 * (d_o[i] * v_o + d_n[i] * v_n - diffflux_o[i + ns_] -
            diffflux_n[i + ns_] - arvis_o * dy_o[i] -
            arvis_n * dy_n[i] - ny * diff[i]);
        flux1[ind++] = 0.5 * (dv_o * u_o + dv_n * u_n - txy_o - txy_n -
          arvis_o * duy_o - arvis_n * duy_n - ny * diff0);
        flux1[ind++] = 0.5 * (dv_o * v_o + mixture_p_o + dv_n * v_n +
          mixture_p_n - tyy_o - tyy_n - arvis_o * dvy_o -
          arvis_n * dvy_n - ny * diff1);
        flux1[ind++] =
          0.5 * ((dE_o + mixture_p_o) * v_o + (dE_n + mixture_p_n) * v_n -
            txy_o * u_o - tyy_o * v_o - txy_n * u_n - tyy_n * v_n -
            q_tr_y_o - q_eev_y_o - Jh_y_o - q_tr_y_n - q_eev_y_n -
            Jh_y_n - arvis_o * dEy_o - arvis_n * dEy_n - ny * diff2);
        flux1[ind] =
          0.5 * (de_eev_o * v_o + de_eev_n * v_n - q_eev_y_o - Je_eev_y_o -
            q_eev_y_n - Je_eev_y_n - arvis_o * de_eevy_o -
            arvis_n * de_eevy_n - ny * diff3);

        // pdss
        ind = DSS_ * ipoint;
        for (int ds = 0; ds < DS_; ds++)
          for (int istate = 0; istate < S_; istate++)
            flux_neighbor_jacobi[ind++] = flux1[ds].df[istate];

        cblas_dscal(DSS_, fy, &flux_neighbor_jacobi[DSS_ * ipoint], 1);
      }
      {
        // flux neighbor grad jacobi
        GET_SOLUTION_PS(_n, neighbor_u);

        // pds over psd
        ind = DS_ * ipoint;
        std::vector<aDual> dx_n(ns_, aDual(DS_));
        for (int i = 0; i < ns_; i++)
          dx_n[i] = aDual(DS_, neighbor_div_u[ind++], 2 * i);
        const aDual ux_n(DS_, neighbor_div_u[ind++], 2 * ns_);
        const aDual vx_n(DS_, neighbor_div_u[ind++], 2 * (ns_ + 1));
        const aDual Ttrx_n(DS_, neighbor_div_u[ind++], 2 * (ns_ + 2));
        const aDual Teevx_n(DS_, neighbor_div_u[ind++], 2 * (ns_ + 3));
        std::vector<aDual> dy_n(ns_, aDual(DS_));
        for (int i = 0; i < ns_; i++)
          dy_n[i] = aDual(DS_, neighbor_div_u[ind++], 2 * i + 1);
        const aDual uy_n(DS_, neighbor_div_u[ind++], 2 * ns_ + 1);
        const aDual vy_n(DS_, neighbor_div_u[ind++], 2 * (ns_ + 1) + 1);
        const aDual Ttry_n(DS_, neighbor_div_u[ind++], 2 * (ns_ + 2) + 1);
        const aDual Teevy_n(DS_, neighbor_div_u[ind], 2 * (ns_ + 3) + 1);

        // Owner viscous flux
        mixture_->SetDensity(&dim_d_n[0]);
        const auto mu_n = mixture_->GetViscosity(dim_T_tr_n, dim_T_eev_n) / mu_ref_;
        const auto k_tr_n =
            mixture_->GetTransRotationConductivity(dim_T_tr_n, dim_T_eev_n) / k_ref_;
        const auto k_eev_n = mixture_->GetElectronicVibrationConductivity(
            dim_T_tr_n, dim_T_eev_n) / k_ref_;

        const double* Y_n = mixture_->GetMassFraction();
        aDual mixture_dx_n(DS_);
        aDual mixture_dy_n(DS_);
        for (int i = 0; i < ns_; i++) {
          mixture_dx_n = mixture_dx_n + dx_n[i];
          mixture_dy_n = mixture_dy_n + dy_n[i];
        }

        std::vector<aDual> Isx_n(ns_, aDual(DS_));
        std::vector<aDual> Isy_n(ns_, aDual(DS_));
        mixture_->GetDiffusivity(&Ds[0], dim_T_tr_n, dim_T_eev_n);
        for (int i = 0; i < ns_; i++) {
          Ds[i] /= D_ref_;
          Isx_n[i] = -Ds[i] * (dx_n[i] - Y_n[i] * mixture_dx_n);
          Isy_n[i] = -Ds[i] * (dy_n[i] - Y_n[i] * mixture_dy_n);
        }

        aDual Ix_sum_n(DS_);
        aDual Iy_sum_n(DS_);
        for (const int& idx : hidx) {
          Ix_sum_n = Ix_sum_n + Isx_n[idx];
          Iy_sum_n = Iy_sum_n + Isy_n[idx];
        }

        std::vector<aDual> diffflux_n(2 * ns_, aDual(DS_));
        for (const int& idx : hidx) {
          diffflux_n[idx] = -Isx_n[idx] + Y_n[idx] * Ix_sum_n;
          diffflux_n[idx + ns_] = -Isy_n[idx] + Y_n[idx] * Iy_sum_n;
        }
        if (eidx >= 0) {
          aDual sum_x(DS_);
          aDual sum_y(DS_);
          for (const int& idx : hidx) {
            const int charge = species[idx]->GetCharge();
            const double Mw = species[idx]->GetMolecularWeight();
            sum_x = sum_x + diffflux_n[idx] * charge / Mw;
            sum_y = sum_y + diffflux_n[idx + ns_] * charge / Mw;
          }
          const double Mw_e = species[eidx]->GetMolecularWeight();
          diffflux_n[eidx] = Mw_e * sum_x;
          diffflux_n[eidx + ns_] = Mw_e * sum_y;
        }

        const auto tau_ax_n = (std::abs(y) < 1.0e-12) ? aDual(DS_, 0.0) : aDual(DS_,  static_cast<double>(ax_) * v_n / y);
        const auto txx_n = c23_ * mu_n * (2.0 * ux_n - vy_n - tau_ax_n);
        const auto txy_n = mu_n * (uy_n + vx_n);
        const auto tyy_n = c23_ * mu_n * (2.0 * vy_n - ux_n - tau_ax_n);

        aDual Jh_x_n(DS_);
        aDual Jh_y_n(DS_);
        aDual Je_eev_x_n(DS_);
        aDual Je_eev_y_n(DS_);

        for (int i = 0; i < ns_; i++) {
          const double h = species[i]->GetEnthalpy(dim_T_tr_n, dim_T_eev_n) / e_ref_;
          const double e_eev =
              species[i]->GetElectronicVibrationEnergy(dim_T_eev_n) / e_ref_;

          Jh_x_n = Jh_x_n + diffflux_n[i] * h;
          Jh_y_n = Jh_y_n + diffflux_n[i + ns_] * h;
          Je_eev_x_n = Je_eev_x_n + diffflux_n[i] * e_eev;
          Je_eev_y_n = Je_eev_y_n + diffflux_n[i + ns_] * e_eev;
        }

        const auto q_tr_x_n = k_tr_n * Ttrx_n;
        const auto q_tr_y_n = k_tr_n * Ttry_n;
        const auto q_eev_x_n = k_eev_n * Teevx_n;
        const auto q_eev_y_n = k_eev_n * Teevy_n;

        ind = 0;
        for (int i = 0; i < ns_; i++)
          flux2[ind++] = -0.5 * (diffflux_o[i] + diffflux_n[i]);
        flux2[ind++] = -0.5 * (txx_o + txx_n);
        flux2[ind++] = -0.5 * (txy_o + txy_n);
        flux2[ind++] = -0.5 * (txx_o * u_o + txy_o * v_o + txx_n * u_n +
          txy_n * v_n + q_tr_x_o + q_eev_x_o + Jh_x_o +
          q_tr_x_n + q_eev_x_n + Jh_x_n);
        flux2[ind++] =
          -0.5 * (q_eev_x_o + Je_eev_x_o + q_eev_x_n + Je_eev_x_n);
        for (int i = 0; i < ns_; i++)
          flux2[ind++] = -0.5 * (diffflux_o[i + ns_] + diffflux_n[i + ns_]);
        flux2[ind++] = -0.5 * (txy_o + txy_n);
        flux2[ind++] = -0.5 * (tyy_o + tyy_n);
        flux2[ind++] = -0.5 * (txy_o * u_o + tyy_o * v_o + txy_n * u_n +
          tyy_n * v_n + q_tr_y_o + q_eev_y_o + Jh_y_o +
          q_tr_y_n + q_eev_y_n + Jh_y_n);
        flux2[ind] = -0.5 * (q_eev_y_o + Je_eev_y_o + q_eev_y_n + Je_eev_y_n);

        // pdssd
        ind = DDSS_ * ipoint;
        for (int ds = 0; ds < DS_; ds++)
          for (int sd = 0; sd < DS_; sd++)
            flux_neighbor_grad_jacobi[ind++] = flux2[ds].df[sd];

        for (int istate = 0; istate < S_; istate++)
          for (int jstate = 0; jstate < S_; jstate++)
            for (int idim = 0; idim < D_; idim++) {
              const int irow = idim * S_ + istate;
              const int icolumn = jstate * D_ + idim;
              flux_neighbor_grad_jacobi[DDSS_ * ipoint + irow * DS_ + icolumn] -=
                0.5 * arvis_n * neighbor_sol_jacobi[ipoint * SS_ + istate * S_ + jstate];
            }

        cblas_dscal(DDSS_, fy, &flux_neighbor_grad_jacobi[DDSS_ * ipoint], 1);
      }
    }
  }
}

// ------------------------------- Boundary -------------------------------- //
std::shared_ptr<BoundaryNS2DNeq2Tnondim> BoundaryNS2DNeq2Tnondim::GetBoundary(
    const std::string& type, const int bdry_tag,
    EquationNS2DNeq2Tnondim* equation) {
  if (!type.compare("SlipWall"))
    return std::make_shared<SlipWallNS2DNeq2Tnondim>(bdry_tag, equation);
  if (!type.compare("AxiSymmetry"))
    return std::make_shared<AxiSymmetryNS2DNeq2Tnondim>(bdry_tag, equation);
  else if (!type.compare("IsothermalWall"))
    return std::make_shared<IsothermalWallNS2DNeq2Tnondim>(bdry_tag, equation);
  else if (!type.compare("LimitingIsothermalWall"))
    return std::make_shared<LimitingIsothermalWallNS2DNeq2Tnondim>(bdry_tag, equation);
  else if (!type.compare("SupersonicInflow"))
    return std::make_shared<SupersonicInflowBdryNS2DNeq2Tnondim>(bdry_tag,
                                                                 equation);
  else if (!type.compare("SupersonicOutflow"))
    return std::make_shared<SupersonicOutflowBdryNS2DNeq2Tnondim>(bdry_tag,
                                                                  equation);
  else if (!type.compare("CatalyticWall"))
    return std::make_shared<CatalyticWallNS2DNeq2Tnondim>(bdry_tag, equation);
  else if (!type.compare("SuperCatalyticWall"))
    return std::make_shared<SuperCatalyticWallNS2DNeq2Tnondim>(bdry_tag,
                                                               equation);
  ERROR_MESSAGE("Wrong boundary condition (no-exist):" + type + "\n");
  return nullptr;
}
// Boundary Name = SlilpWall
// Dependency: -
SlipWallNS2DNeq2Tnondim::SlipWallNS2DNeq2Tnondim(
    const int bdry_tag, EquationNS2DNeq2Tnondim* equation)
    : BoundaryNS2DNeq2Tnondim(bdry_tag, equation) {
  auto& config = AVOCADO_CONFIG;
  scale_ = std::stod(config->GetConfigValue(BDRY_INPUT_I(bdry_tag, 0)));

  MASTER_MESSAGE("SlipWall (tag=" + std::to_string(bdry_tag) +
                 ")\n\tScale factor = " + std::to_string(scale_) + "\n");

  if (scale_ > 1.0 || scale_ < 0.0)
    ERROR_MESSAGE("Wrong scale factor imposed : " + std::to_string(scale_));
}
void SlipWallNS2DNeq2Tnondim::ComputeBdrySolution(
    const int num_points, std::vector<double>& bdry_u,
    std::vector<double>& bdry_div_u, const std::vector<double>& owner_u,
    const std::vector<double>& owner_div_u, const std::vector<double>& normal,
    const std::vector<double>& coords, const double& time) {
  int ind = 0;
  std::vector<double> dim_d(ns_);
  for (int ipoint = 0; ipoint < num_points; ipoint++) {
    GET_NORMAL_PD(normal);

    // ps
    GET_SOLUTION_PS(, owner_u);
    const double V = u * nx + v * ny;

    const double u_new = u - scale_ * nx * V;
    const double v_new = v - scale_ * ny * V;

    // ps
    ind = S_ * ipoint;
    for (int i = 0; i < ns_; i++) bdry_u[ind++] = d[i];
    bdry_u[ind++] = u_new;
    bdry_u[ind++] = v_new;
    bdry_u[ind++] = T_tr;
    bdry_u[ind] = T_eev;
  }
  // Don't touch bdry_div_u.
}
void SlipWallNS2DNeq2Tnondim::ComputeBdryFlux(const int num_points,
                                              std::vector<double>& flux,
                                              FACE_INPUTS,
                                              const std::vector<double>& coords,
                                              const double& time) {
  //static std::vector<double> Ds(ns_, 0.0);
  //static std::vector<double> Isx(ns_, 0.0);
  //static std::vector<double> Isy(ns_, 0.0);
  //static std::vector<double> hs(ns_, 0.0);
  //static std::vector<double> diffflux(2 * ns_, 0.0);

  //const auto& species = mixture_->GetSpecies();
  //const auto& hidx = mixture_->GetHeavyParticleIdx();
  //const auto& eidx = mixture_->GetElectronIndex();

  //int ind = 0;
  //std::vector<double> dim_d(ns_);
  //for (int ipoint = 0; ipoint < num_points; ipoint++) {
  //  GET_NORMAL_PD(normal);

  //  GET_SOLUTION_PS(, owner_u);

  //  const auto V = u * nx + v * ny;
  //  const auto u_new = u - nx * V;
  //  const auto v_new = v - ny * V;

  //  // Convective flux
  //  mixture_->SetDensity(&dim_d[0]);
  //  const double mixture_d = mixture_->GetTotalDensity() / rho_ref_;
  //  const double mixture_p = mixture_->GetPressure(dim_T_tr, dim_T_eev) / p_ref_;

  //  double de = 0.0;
  //  double de_eev = 0.0;
  //  for (int i = 0; i < ns_; i++) {
  //    const double e =
  //        species[i]->GetInternalEnergy(dim_T_tr, dim_T_eev) / e_ref_;
  //    const double e_eev =
  //        species[i]->GetElectronicVibrationEnergy(dim_T_eev) / e_ref_;
  //    de += d[i] * e;
  //    de_eev += d[i] * e_eev;
  //  }
  //  const auto du_new = mixture_d * u_new;
  //  const auto dv_new = mixture_d * v_new;
  //  const auto dE = de + 0.5 * (du_new * u_new + dv_new * v_new);

  //  ind = DS_ * ipoint;
  //  for (int i = 0; i < ns_; i++) flux[ind++] = d[i] * u_new;
  //  flux[ind++] = du_new * u_new + mixture_p;
  //  flux[ind++] = du_new * v_new;
  //  flux[ind++] = (dE + mixture_p) * u_new;
  //  flux[ind++] = de_eev * u_new;
  //  for (int i = 0; i < ns_; i++) flux[ind++] = d[i] * v_new;
  //  flux[ind++] = dv_new * u_new;
  //  flux[ind++] = dv_new * v_new + mixture_p;
  //  flux[ind++] = (dE + mixture_p) * v_new;
  //  flux[ind] = de_eev * v_new;
  //}
  equation_->ComputeNumFlux(num_points, flux, owner_cell, -1, owner_u,
                            owner_div_u, neighbor_u, owner_div_u,
                            normal, coords);  // for debug
}
void SlipWallNS2DNeq2Tnondim::ComputeBdrySolutionJacobi(
    const int num_points, double* bdry_u_jacobi,
    const std::vector<double>& owner_u, const std::vector<double>& owner_div_u,
    const std::vector<double>& normal, const std::vector<double>& coords,
    const double& time) {
  static std::vector<aDual> bdry_sol(S_, aDual(S_));
  int ind = 0;
  memset(bdry_u_jacobi, 0, num_points * SS_ * sizeof(double));

  for (int ipoint = 0; ipoint < num_points; ipoint++) {
    GET_NORMAL_PD(normal);

    // ps
    ind = S_ * ipoint;
    std::vector<aDual> d(ns_, aDual(S_));
    for (int i = 0; i < ns_; i++) d[i] = aDual(S_, owner_u[ind++], i);
    aDual u(S_, owner_u[ind++], ns_);
    aDual v(S_, owner_u[ind++], ns_ + 1);
    aDual T_tr(S_, owner_u[ind++], ns_ + 2);
    aDual T_eev(S_, owner_u[ind], ns_ + 3);

    const auto V = u * nx + v * ny;
    const auto u_new = u - scale_ * nx * V;
    const auto v_new = v - scale_ * ny * V;

    for (int i = 0; i < ns_; i++) bdry_sol[i] = d[i];
    bdry_sol[ns_] = u_new;
    bdry_sol[ns_ + 1] = v_new;
    bdry_sol[ns_ + 2] = T_tr;
    bdry_sol[ns_ + 3] = T_eev;

    ind = SS_ * ipoint;
    for (int i = 0; i < S_; i++)
      for (int j = 0; j < S_; j++) bdry_u_jacobi[ind++] = bdry_sol[i].df[j];
  }
}
void SlipWallNS2DNeq2Tnondim::ComputeBdryFluxJacobi(
    const int num_points, std::vector<double>& flux_jacobi,
    std::vector<double>& flux_grad_jacobi, FACE_INPUTS,
    const std::vector<double>& coords, const double& time) {
  // flux_jacobi(ds1s2) = F(ds1) over U(s2)
  // flux_grad_jacobi(d1s1 s2d2) = F(d1s1) over gradU(d2s2)
  //int ind = 0;
  //static std::vector<double> Ds(ns_, 0.0);
  //static std::vector<double> Isx(ns_, 0.0);
  //static std::vector<double> Isy(ns_, 0.0);
  //static std::vector<double> hs(ns_, 0.0);
  //static std::vector<aDual> flux1(DS_, aDual(S_));
  //static std::vector<aDual> flux2(DS_, aDual(DS_));

  //const auto& species = mixture_->GetSpecies();
  //const auto& hidx = mixture_->GetHeavyParticleIdx();
  //const auto& eidx = mixture_->GetElectronIndex();
  //std::vector<double> dim_d(ns_);
  //for (int ipoint = 0; ipoint < num_points; ipoint++) {
  //  {
  //    GET_NORMAL_PD(normal);

  //    // ps
  //    ind = S_ * ipoint;
  //    std::vector<aDual> d(ns_, aDual(S_));
  //    for (int i = 0; i < ns_; i++) {
  //      dim_d[i] = owner_u[ind] * rho_ref_;
  //      d[i] = aDual(S_, owner_u[ind++], i);
  //    }
  //    aDual u(S_, owner_u[ind++], ns_);
  //    aDual v(S_, owner_u[ind++], ns_ + 1);
  //    aDual T_tr(S_, owner_u[ind++], ns_ + 2);
  //    aDual T_eev(S_, owner_u[ind], ns_ + 3);
  //    const auto dim_T_tr = T_tr * T_ref_;
  //    const auto dim_T_eev = T_eev * T_ref_;

  //    GET_SOLUTION_GRAD_PDS(, owner_div_u);

  //    // Convective flux
  //    mixture_->SetDensity(&dim_d[0]);
  //    aDual mixture_d(S_, mixture_->GetTotalDensity() / rho_ref_);
  //    for (int i = 0; i < ns_; i++) mixture_d.df[i] = 1.0;

  //    aDual mixture_p(S_);
  //    mixture_p.f =
  //        mixture_->GetPressureJacobian(dim_T_tr.f, dim_T_eev.f, &mixture_p.df[0]);
  //    std::swap(mixture_p.df[ns_], mixture_p.df[ns_ + 2]);
  //    std::swap(mixture_p.df[ns_ + 1], mixture_p.df[ns_ + 3]);
  //    mixture_p.f /= p_ref_;
  //    for (int i = 0; i < ns_; i++) mixture_p.df[i] *= rho_ref_ / p_ref_;
  //    mixture_p.df[ns_ + 2] *= T_ref_ / p_ref_;
  //    mixture_p.df[ns_ + 3] *= T_ref_ / p_ref_;

  //    const auto V = u * nx + v * ny;
  //    const auto u_new = u - nx * V;
  //    const auto v_new = v - ny * V;

  //    aDual de(S_);
  //    aDual de_eev(S_);
  //    for (int i = 0; i < ns_; i++) {
  //      const auto e =
  //          species[i]->GetInternalEnergy(dim_T_tr.f, dim_T_eev.f) / e_ref_;
  //      const auto e_eev =
  //          species[i]->GetElectronicVibrationEnergy(dim_T_eev.f) / e_ref_;
  //      const auto Cv_tr = species[i]->GetTransRotationSpecificHeat(dim_T_tr.f) *
  //                         T_ref_ / e_ref_;
  //      const auto Cv_eev =
  //          species[i]->GetElectronicVibrationSpecificHeat(dim_T_eev.f) * T_ref_ /
  //          e_ref_;

  //      de.f += d[i].f * e;
  //      de.df[i] = e;
  //      de.df[ns_ + 2] += d[i].f * Cv_tr;
  //      de.df[ns_ + 3] += d[i].f * Cv_eev;

  //      de_eev.f += d[i].f * e_eev;
  //      de_eev.df[i] = e_eev;
  //      de_eev.df[ns_ + 3] += d[i].f * Cv_eev;
  //    }

  //    const auto du_new = mixture_d * u_new;
  //    const auto dv_new = mixture_d * v_new;
  //    const auto dE = de + 0.5 * (du_new * u_new + dv_new * v_new);

  //    ind = 0;
  //    for (int i = 0; i < ns_; i++) flux1[ind++] = d[i] * u_new;
  //    flux1[ind++] = du_new * u_new + mixture_p;
  //    flux1[ind++] = du_new * v_new;
  //    flux1[ind++] = (dE + mixture_p) * u_new;
  //    flux1[ind++] = de_eev * u_new;
  //    for (int i = 0; i < ns_; i++) flux1[ind++] = d[i] * v_new;
  //    flux1[ind++] = dv_new * u_new;
  //    flux1[ind++] = dv_new * v_new + mixture_p;
  //    flux1[ind++] = (dE + mixture_p) * v_new;
  //    flux1[ind] = de_eev * v_new;

  //    // pdss
  //    ind = DSS_ * ipoint;
  //    for (int ds = 0; ds < DS_; ds++)
  //      for (int istate = 0; istate < S_; istate++)
  //        flux_jacobi[ind++] = flux1[ds].df[istate];
  //  }
  //}
  //memset(&flux_grad_jacobi[0], 0, num_points * DDSS_ * sizeof(double));
  // for debug
  static const int max_num_bdry_points = equation_->GetMaxNumBdryPoints();
  static std::vector<double> flux_neighbor_jacobi(max_num_bdry_points * DSS_);
  static std::vector<double> flux_neighbor_grad_jacobi(max_num_bdry_points *
                                                       DDSS_);
  static std::vector<double> bdry_u_jacobi(max_num_bdry_points * SS_);

  equation_->ComputeNumFluxJacobi(num_points, flux_jacobi, flux_neighbor_jacobi,
                                  flux_grad_jacobi, flux_neighbor_grad_jacobi,
                                  owner_cell, -1, owner_u, owner_div_u,
                                  neighbor_u, owner_div_u, normal, coords);
  ComputeBdrySolutionJacobi(num_points, &bdry_u_jacobi[0], owner_u, owner_div_u,
                            normal, coords, time);

  for (int ipoint = 0; ipoint < num_points; ipoint++)
    gemmAB(1.0, &flux_neighbor_jacobi[ipoint * DSS_],
           &bdry_u_jacobi[ipoint * SS_], 1.0, &flux_jacobi[ipoint * DSS_], DS_,
           S_, S_);
}
// Boundary Name = SlilpWall
// Dependency: -
AxiSymmetryNS2DNeq2Tnondim::AxiSymmetryNS2DNeq2Tnondim(
  const int bdry_tag, EquationNS2DNeq2Tnondim* equation)
  : BoundaryNS2DNeq2Tnondim(bdry_tag, equation) {
  MASTER_MESSAGE("AxiSymmetry (tag=" + std::to_string(bdry_tag) +
    ")\n");
}
void AxiSymmetryNS2DNeq2Tnondim::ComputeBdrySolution(
  const int num_points, std::vector<double>& bdry_u,
  std::vector<double>& bdry_div_u, const std::vector<double>& owner_u,
  const std::vector<double>& owner_div_u, const std::vector<double>& normal,
  const std::vector<double>& coords, const double& time) {
  int ind = 0;
  std::vector<double> dim_d(ns_);
  for (int ipoint = 0; ipoint < num_points; ipoint++) {
    GET_NORMAL_PD(normal);

    // ps
    GET_SOLUTION_PS(, owner_u);

    const double u_new = u;
    const double v_new = 0.0;

    // ps
    ind = S_ * ipoint;
    for (int i = 0; i < ns_; i++) bdry_u[ind++] = d[i];
    bdry_u[ind++] = u_new;
    bdry_u[ind++] = v_new;
    bdry_u[ind++] = T_tr;
    bdry_u[ind] = T_eev;
  }
  // Don't touch bdry_div_u.
}
void AxiSymmetryNS2DNeq2Tnondim::ComputeBdryFlux(const int num_points,
  std::vector<double>& flux,
  FACE_INPUTS,
  const std::vector<double>& coords,
  const double& time) {
  static std::vector<double> Ds(ns_, 0.0);
  static std::vector<double> Isx(ns_, 0.0);
  static std::vector<double> Isy(ns_, 0.0);
  static std::vector<double> hs(ns_, 0.0);
  static std::vector<double> diffflux(2 * ns_, 0.0);

  const auto& species = mixture_->GetSpecies();
  const auto& hidx = mixture_->GetHeavyParticleIdx();
  const auto& eidx = mixture_->GetElectronIndex();

  int ind = 0;
  std::vector<double> dim_d(ns_);
  for (int ipoint = 0; ipoint < num_points; ipoint++) {
    GET_NORMAL_PD(normal);

    GET_SOLUTION_PS(, owner_u);

    const auto u_new = u;
    const auto v_new = 0.0;

    // Convective flux
    mixture_->SetDensity(&dim_d[0]);
    const double mixture_d = mixture_->GetTotalDensity() / rho_ref_;
    const double mixture_p = mixture_->GetPressure(dim_T_tr, dim_T_eev) / p_ref_;

    double de = 0.0;
    double de_eev = 0.0;
    for (int i = 0; i < ns_; i++) {
      const double e =
          species[i]->GetInternalEnergy(dim_T_tr, dim_T_eev) / e_ref_;
      const double e_eev =
          species[i]->GetElectronicVibrationEnergy(dim_T_eev) / e_ref_;
      de += d[i] * e;
      de_eev += d[i] * e_eev;
    }
    const auto du_new = mixture_d * u_new;
    const auto dv_new = mixture_d * v_new;
    const auto dE = de + 0.5 * (du_new * u_new + dv_new * v_new);

    ind = DS_ * ipoint;
    for (int i = 0; i < ns_; i++) flux[ind++] = d[i] * u_new;
    flux[ind++] = du_new * u_new + mixture_p;
    flux[ind++] = du_new * v_new;
    flux[ind++] = (dE + mixture_p) * u_new;
    flux[ind++] = de_eev * u_new;
    for (int i = 0; i < ns_; i++) flux[ind++] = d[i] * v_new;
    flux[ind++] = dv_new * u_new;
    flux[ind++] = dv_new * v_new + mixture_p;
    flux[ind++] = (dE + mixture_p) * v_new;
    flux[ind] = de_eev * v_new;
  }
}
void AxiSymmetryNS2DNeq2Tnondim::ComputeBdrySolutionJacobi(
  const int num_points, double* bdry_u_jacobi,
  const std::vector<double>& owner_u, const std::vector<double>& owner_div_u,
  const std::vector<double>& normal, const std::vector<double>& coords,
  const double& time) {
  static std::vector<aDual> bdry_sol(S_, aDual(S_));
  int ind = 0;
  memset(bdry_u_jacobi, 0, num_points * SS_ * sizeof(double));

  for (int ipoint = 0; ipoint < num_points; ipoint++) {
    GET_NORMAL_PD(normal);

    // ps
    ind = S_ * ipoint;
    std::vector<aDual> d(ns_, aDual(S_));
    for (int i = 0; i < ns_; i++) d[i] = aDual(S_, owner_u[ind++], i);
    aDual u(S_, owner_u[ind++], ns_);
    aDual v(S_, owner_u[ind++], ns_ + 1);
    aDual T_tr(S_, owner_u[ind++], ns_ + 2);
    aDual T_eev(S_, owner_u[ind], ns_ + 3);

    const auto u_new = u;
    const auto v_new = aDual(S_, 0.0);

    for (int i = 0; i < ns_; i++) bdry_sol[i] = d[i];
    bdry_sol[ns_] = u_new;
    bdry_sol[ns_ + 1] = v_new;
    bdry_sol[ns_ + 2] = T_tr;
    bdry_sol[ns_ + 3] = T_eev;

    ind = SS_ * ipoint;
    for (int i = 0; i < S_; i++)
      for (int j = 0; j < S_; j++) bdry_u_jacobi[ind++] = bdry_sol[i].df[j];
  }
}
void AxiSymmetryNS2DNeq2Tnondim::ComputeBdryFluxJacobi(
  const int num_points, std::vector<double>& flux_jacobi,
  std::vector<double>& flux_grad_jacobi, FACE_INPUTS,
  const std::vector<double>& coords, const double& time) {
  // flux_jacobi(ds1s2) = F(ds1) over U(s2)
  // flux_grad_jacobi(d1s1 s2d2) = F(d1s1) over gradU(d2s2)
  int ind = 0;
  static std::vector<double> Ds(ns_, 0.0);
  static std::vector<double> Isx(ns_, 0.0);
  static std::vector<double> Isy(ns_, 0.0);
  static std::vector<double> hs(ns_, 0.0);
  static std::vector<aDual> flux1(DS_, aDual(S_));
  static std::vector<aDual> flux2(DS_, aDual(DS_));

  const auto& species = mixture_->GetSpecies();
  const auto& hidx = mixture_->GetHeavyParticleIdx();
  const auto& eidx = mixture_->GetElectronIndex();
  std::vector<double> dim_d(ns_);
  for (int ipoint = 0; ipoint < num_points; ipoint++) {
    {
      GET_NORMAL_PD(normal);

      // ps
      ind = S_ * ipoint;
      std::vector<aDual> d(ns_, aDual(S_));
      for (int i = 0; i < ns_; i++) {
        dim_d[i] = owner_u[ind] * rho_ref_;
        d[i] = aDual(S_, owner_u[ind++], i);
      }
      aDual u(S_, owner_u[ind++], ns_);
      aDual v(S_, owner_u[ind++], ns_ + 1);
      aDual T_tr(S_, owner_u[ind++], ns_ + 2);
      aDual T_eev(S_, owner_u[ind], ns_ + 3);
      const auto dim_T_tr = T_tr * T_ref_;
      const auto dim_T_eev = T_eev * T_ref_;

      GET_SOLUTION_GRAD_PDS(, owner_div_u);

      // Convective flux
      mixture_->SetDensity(&dim_d[0]);
      aDual mixture_d(S_, mixture_->GetTotalDensity() / rho_ref_);
      for (int i = 0; i < ns_; i++) mixture_d.df[i] = 1.0;

      aDual mixture_p(S_);
      mixture_p.f =
          mixture_->GetPressureJacobian(dim_T_tr.f, dim_T_eev.f, &mixture_p.df[0]);
      std::swap(mixture_p.df[ns_], mixture_p.df[ns_ + 2]);
      std::swap(mixture_p.df[ns_ + 1], mixture_p.df[ns_ + 3]);
      mixture_p.f /= p_ref_;
      for (int i = 0; i < ns_; i++) mixture_p.df[i] *= rho_ref_ / p_ref_;
      mixture_p.df[ns_ + 2] *= T_ref_ / p_ref_;
      mixture_p.df[ns_ + 3] *= T_ref_ / p_ref_;

      const auto u_new = u;
      const auto v_new = aDual(S_, 0.0);

      aDual de(S_);
      aDual de_eev(S_);
      for (int i = 0; i < ns_; i++) {
        const auto e =
            species[i]->GetInternalEnergy(dim_T_tr.f, dim_T_eev.f) / e_ref_;
        const auto e_eev =
            species[i]->GetElectronicVibrationEnergy(dim_T_eev.f) / e_ref_;
        const auto Cv_tr = species[i]->GetTransRotationSpecificHeat(dim_T_tr.f) *
                           T_ref_ / e_ref_;
        const auto Cv_eev =
            species[i]->GetElectronicVibrationSpecificHeat(dim_T_eev.f) * T_ref_ /
            e_ref_;

        de.f += d[i].f * e;
        de.df[i] = e;
        de.df[ns_ + 2] += d[i].f * Cv_tr;
        de.df[ns_ + 3] += d[i].f * Cv_eev;

        de_eev.f += d[i].f * e_eev;
        de_eev.df[i] = e_eev;
        de_eev.df[ns_ + 3] += d[i].f * Cv_eev;
      }

      const auto du_new = mixture_d * u_new;
      const auto dv_new = mixture_d * v_new;
      const auto dE = de + 0.5 * (du_new * u_new + dv_new * v_new);

      ind = 0;
      for (int i = 0; i < ns_; i++) flux1[ind++] = d[i] * u_new;
      flux1[ind++] = du_new * u_new + mixture_p;
      flux1[ind++] = du_new * v_new;
      flux1[ind++] = (dE + mixture_p) * u_new;
      flux1[ind++] = de_eev * u_new;
      for (int i = 0; i < ns_; i++) flux1[ind++] = d[i] * v_new;
      flux1[ind++] = dv_new * u_new;
      flux1[ind++] = dv_new * v_new + mixture_p;
      flux1[ind++] = (dE + mixture_p) * v_new;
      flux1[ind] = de_eev * v_new;

      // pdss
      ind = DSS_ * ipoint;
      for (int ds = 0; ds < DS_; ds++)
        for (int istate = 0; istate < S_; istate++)
          flux_jacobi[ind++] = flux1[ds].df[istate];
    }
  }
  memset(&flux_grad_jacobi[0], 0, num_points * DDSS_ * sizeof(double));
}
// Boundary Name = IsothermalWall
// Dependency: Twall
IsothermalWallNS2DNeq2Tnondim::IsothermalWallNS2DNeq2Tnondim(
    const int bdry_tag, EquationNS2DNeq2Tnondim* equation)
    : BoundaryNS2DNeq2Tnondim(bdry_tag, equation) {
  auto& config = AVOCADO_CONFIG;
  scale_ = std::stod(config->GetConfigValue(BDRY_INPUT_I(bdry_tag, 0)));
  Twall_ =
      std::stod(config->GetConfigValue(BDRY_INPUT_I(bdry_tag, 1))) / T_ref_;

  MASTER_MESSAGE("IsothermalWall (tag=" + std::to_string(bdry_tag) +
                 ")\n\tTwall = " + std::to_string(Twall_) +
                 "\n\tScale factor = " + std::to_string(scale_) + "\n");

  if (scale_ > 1.0 || scale_ < 0.0)
    ERROR_MESSAGE("Wrong scale factor imposed : " + std::to_string(scale_));
}
void IsothermalWallNS2DNeq2Tnondim::ComputeBdrySolution(
    const int num_points, std::vector<double>& bdry_u,
    std::vector<double>& bdry_div_u, const std::vector<double>& owner_u,
    const std::vector<double>& owner_div_u, const std::vector<double>& normal,
    const std::vector<double>& coords, const double& time) {
  const auto& species = mixture_->GetSpecies();
  int ind = 0;
  std::vector<double> dim_d(ns_);
  for (int ipoint = 0; ipoint < num_points; ipoint++) {
    GET_NORMAL_PD(normal);

    // ps
    GET_SOLUTION_PS(, owner_u);

    //mixture_->SetDensity(&dim_d[0]);
    //const double mixture_d = mixture_->GetTotalDensity() / rho_ref_;
    //const double mixture_p =
    //    mixture_->GetPressure(dim_T_tr, dim_T_eev) / p_ref_;

    //double sum = 0.0;
    //const double* Y = mixture_->GetMassFraction();
    //for (int i = 0; i < ns_; i++) {
    //  const double R = species[i]->GetSpecificGasConstant() * T_ref_ / e_ref_;
    //  sum = sum + Y[i] * R * Twall_;
    //}
    //const double d_wall = mixture_p / sum;

    // ps
    ind = S_ * ipoint;
    //for (int i = 0; i < ns_; i++) bdry_u[ind++] = Y[i] * d_wall;
    for (int i = 0; i < ns_; i++) bdry_u[ind++] = d[i];
    bdry_u[ind++] = -scale_ * u;
    bdry_u[ind++] = -scale_ * v;
    bdry_u[ind++] = (1.0 - scale_) * (T_tr - Twall_) + Twall_;
    bdry_u[ind] = (1.0 - scale_) * (T_eev - Twall_) + Twall_;
  }
  // Don't touch bdry_div_u.
}
void IsothermalWallNS2DNeq2Tnondim::ComputeBdryFlux(
    const int num_points, std::vector<double>& flux, FACE_INPUTS,
    const std::vector<double>& coords, const double& time) {
  //static std::vector<double> Ds(ns_, 0.0);
  //static std::vector<double> Isx(ns_, 0.0);
  //static std::vector<double> Isy(ns_, 0.0);
  //static std::vector<double> hs(ns_, 0.0);
  //static std::vector<double> diffflux(2 * ns_, 0.0);
  //static std::vector<double> d_new(ns_, 0.0);
  //static double dim_Twall = Twall_ * T_ref_;

  //const auto& species = mixture_->GetSpecies();
  //const auto& hidx = mixture_->GetHeavyParticleIdx();
  //const auto& eidx = mixture_->GetElectronIndex();
  //// need check
  //int ind = 0;
  //std::vector<double> dim_d(ns_);
  //for (int ipoint = 0; ipoint < num_points; ipoint++) {
  //  GET_SOLUTION_PS(, owner_u);

  //  GET_SOLUTION_GRAD_PDS(, owner_div_u);

  //  // Convective flux
  //  mixture_->SetDensity(&dim_d[0]);
  //  const double mixture_d = mixture_->GetTotalDensity() / rho_ref_;
  //  const double mixture_p = mixture_->GetPressure(dim_T_tr, dim_T_eev) / p_ref_;

  //  double sum = 0.0;
  //  const double* Y = mixture_->GetMassFraction();
  //  for (int i = 0; i < ns_; i++) {
  //    const double R = species[i]->GetSpecificGasConstant() * T_ref_ / e_ref_;
  //    sum = sum + Y[i] * R * Twall_;
  //  }
  //  const double d_wall = mixture_p / sum;
  //  for (int i = 0; i < ns_; i++) d_new[i] = Y[i] * d_wall * rho_ref_;
  //  mixture_->SetDensity(&d_new[0]);

  //  // Viscous flux
  //  const double mu = mixture_->GetViscosity(dim_Twall, dim_Twall) / mu_ref_;
  //  const double k_tr =
  //      mixture_->GetTransRotationConductivity(dim_Twall, dim_Twall) / k_ref_;
  //  const double k_eev =
  //      mixture_->GetElectronicVibrationConductivity(dim_Twall, dim_Twall) / k_ref_;

  //  const auto txx = c23_ * mu * (2.0 * ux - vy);
  //  const auto txy = mu * (uy + vx);
  //  const auto tyy = c23_ * mu * (2.0 * vy - ux);

  //  const double q_tr_x = k_tr * Ttrx;
  //  const double q_tr_y = k_tr * Ttry;
  //  const double q_eev_x = k_eev * Teevx;
  //  const double q_eev_y = k_eev * Teevy;

  //  ind = DS_ * ipoint;
  //  for (int i = 0; i < ns_; i++) flux[ind++] = 0.0;
  //  flux[ind++] = mixture_p - txx;
  //  flux[ind++] = -txy;
  //  flux[ind++] = -q_tr_x - q_eev_x;
  //  flux[ind++] = -q_eev_x;
  //  for (int i = 0; i < ns_; i++) flux[ind++] = 0.0;
  //  flux[ind++] = -txy;
  //  flux[ind++] = mixture_p - tyy;
  //  flux[ind++] = -q_tr_y - q_eev_y;
  //  flux[ind] = -q_eev_y;
  //}
  equation_->ComputeNumFlux(num_points, flux, owner_cell, -1, owner_u,
                            owner_div_u, neighbor_u, owner_div_u,
                            normal, coords);  // Weak
}
void IsothermalWallNS2DNeq2Tnondim::ComputeBdrySolutionJacobi(
    const int num_points, double* bdry_u_jacobi,
    const std::vector<double>& owner_u, const std::vector<double>& owner_div_u,
    const std::vector<double>& normal, const std::vector<double>& coords,
    const double& time) {
  static std::vector<aDual> bdry_sol(S_, aDual(S_));
  int ind = 0;
  memset(bdry_u_jacobi, 0, num_points * SS_ * sizeof(double));

  const auto& species = mixture_->GetSpecies();

  std::vector<double> dim_d(ns_);
  for (int ipoint = 0; ipoint < num_points; ipoint++) {
    GET_NORMAL_PD(normal);

    // ps
    ind = S_ * ipoint;
    std::vector<aDual> d(ns_, aDual(S_));
    for (int i = 0; i < ns_; i++) {
      //dim_d[i] = owner_u[ind] * rho_ref_;
      d[i] = aDual(S_, owner_u[ind++], i);
    }
    aDual u(S_, owner_u[ind++], ns_);
    aDual v(S_, owner_u[ind++], ns_ + 1);
    aDual T_tr(S_, owner_u[ind++], ns_ + 2);
    aDual T_eev(S_, owner_u[ind], ns_ + 3);
    //const auto dim_T_tr = T_tr * T_ref_;
    //const auto dim_T_eev = T_eev * T_ref_;

    //mixture_->SetDensity(&dim_d[0]);
    //aDual mixture_d(S_, 0.0);
    //for (int i = 0; i < ns_; i++) mixture_d = mixture_d + d[i];
    //const auto mixture_d_inv = 1.0 / mixture_d;

    //aDual mixture_p(S_);
    //mixture_p.f = mixture_->GetPressureJacobian(dim_T_tr.f, dim_T_eev.f,
    //                                            &mixture_p.df[0]);
    //std::swap(mixture_p.df[ns_], mixture_p.df[ns_ + 2]);
    //std::swap(mixture_p.df[ns_ + 1], mixture_p.df[ns_ + 3]);
    //mixture_p.f /= p_ref_;
    //for (int i = 0; i < ns_; i++) mixture_p.df[i] *= rho_ref_ / p_ref_;
    //mixture_p.df[ns_ + 2] *= T_ref_ / p_ref_;
    //mixture_p.df[ns_ + 3] *= T_ref_ / p_ref_;

    //aDual sum(S_, 0.0);
    //for (int i = 0; i < ns_; i++) {
    //  const double R = species[i]->GetSpecificGasConstant() * T_ref_ / e_ref_;
    //  const auto Y = d[i] * mixture_d_inv;
    //  sum = sum + Y * R * Twall_;
    //}
    //const auto mixture_d_wall = mixture_p / sum;

    //for (int i = 0; i < ns_; i++)
    //  bdry_sol[i] = d[i] * mixture_d_wall * mixture_d_inv;
    for (int i = 0; i < ns_; i++) bdry_sol[i] = d[i]; // weak
    bdry_sol[ns_] = -scale_ * u;
    bdry_sol[ns_ + 1] = -scale_ * v;
    bdry_sol[ns_ + 2] = (1.0 - scale_) * (T_tr - Twall_) + aDual(S_, Twall_);
    bdry_sol[ns_ + 3] = (1.0 - scale_) * (T_eev - Twall_) + aDual(S_, Twall_);

    ind = SS_ * ipoint;
    for (int i = 0; i < S_; i++)
      for (int j = 0; j < S_; j++) bdry_u_jacobi[ind++] = bdry_sol[i].df[j];
  }
}
void IsothermalWallNS2DNeq2Tnondim::ComputeBdryFluxJacobi(
    const int num_points, std::vector<double>& flux_jacobi,
    std::vector<double>& flux_grad_jacobi, FACE_INPUTS,
    const std::vector<double>& coords, const double& time) {
  // flux_jacobi(ds1s2) = F(ds1) over U(s2)
  // flux_grad_jacobi(d1s1 s2d2) = F(d1s1) over gradU(d2s2)

  // Strong
  //static std::vector<double> Ds(ns_, 0.0);
  //static std::vector<double> Isx(ns_, 0.0);
  //static std::vector<double> Isy(ns_, 0.0);
  //static std::vector<double> hs(ns_, 0.0);
  //static std::vector<double> d_new(ns_, 0.0);
  //static std::vector<aDual> flux1(DS_, aDual(S_));
  //static std::vector<aDual> flux2(DS_, aDual(DS_));
  //static double dim_Twall = Twall_ * T_ref_;

  //const auto& species = mixture_->GetSpecies();
  //const auto& hidx = mixture_->GetHeavyParticleIdx();
  //const auto& eidx = mixture_->GetElectronIndex();

  //int ind = 0;
  //std::vector<double> dim_d(ns_);
  //for (int ipoint = 0; ipoint < num_points; ipoint++) {
  //  {
  //    // ps
  //    ind = S_ * ipoint;
  //    std::vector<aDual> d(ns_, aDual(S_));
  //    for (int i = 0; i < ns_; i++) {
  //      dim_d[i] = owner_u[ind] * rho_ref_;
  //      d[i] = aDual(S_, owner_u[ind++], i);
  //    }
  //    aDual u(S_, owner_u[ind++], ns_);
  //    aDual v(S_, owner_u[ind++], ns_ + 1);
  //    aDual T_tr(S_, owner_u[ind++], ns_ + 2);
  //    aDual T_eev(S_, owner_u[ind], ns_ + 3);
  //    const auto dim_T_tr = T_tr * T_ref_;
  //    const auto dim_T_eev = T_eev * T_ref_;

  //    GET_SOLUTION_GRAD_PDS(, owner_div_u);

  //    // Convective flux
  //    mixture_->SetDensity(&dim_d[0]);
  //    aDual mixture_d(S_, 0.0);
  //    for (int i = 0; i < ns_; i++) mixture_d = mixture_d + d[i];
  //    const auto mixture_d_inv = 1.0 / mixture_d;

  //    aDual mixture_p(S_);
  //    mixture_p.f = mixture_->GetPressureJacobian(dim_T_tr.f, dim_T_eev.f,
  //                                                &mixture_p.df[0]);
  //    std::swap(mixture_p.df[ns_], mixture_p.df[ns_ + 2]);
  //    std::swap(mixture_p.df[ns_ + 1], mixture_p.df[ns_ + 3]);
  //    mixture_p.f /= p_ref_;
  //    for (int i = 0; i < ns_; i++) mixture_p.df[i] *= rho_ref_ / p_ref_;
  //    mixture_p.df[ns_ + 2] *= T_ref_ / p_ref_;
  //    mixture_p.df[ns_ + 3] *= T_ref_ / p_ref_;

  //    aDual sum(S_, 0.0);
  //    std::vector<aDual> Y(ns_, aDual(S_));
  //    for (int i = 0; i < ns_; i++) {
  //      const double R = species[i]->GetSpecificGasConstant() * T_ref_ / e_ref_;
  //      Y[i] = d[i] * mixture_d_inv;
  //      sum = sum + Y[i] * R * Twall_;
  //    }
  //    const auto mixture_d_wall = mixture_p / sum;
  //    for (int i = 0; i < ns_; i++) d_new[i] = Y[i].f * mixture_d_wall.f * rho_ref_;
  //    mixture_->SetDensity(&d_new[0]);

  //    // Viscous flux // need check -> this assumes the dwall_ to be independent
  //    // / check useless computations to solution variables
  //    const auto mu = mixture_->GetViscosity(dim_Twall, dim_Twall) / mu_ref_;
  //    const auto k_tr =
  //        mixture_->GetTransRotationConductivity(dim_Twall, dim_Twall) / k_ref_;
  //    const auto k_eev =
  //        mixture_->GetElectronicVibrationConductivity(dim_Twall, dim_Twall) / k_ref_;

  //    const aDual txx(S_, c23_ * mu * (2.0 * ux - vy));
  //    const aDual txy(S_, mu * (uy + vx));
  //    const aDual tyy(S_, c23_ * mu * (2.0 * vy - ux));

  //    const aDual q_tr_x(S_, k_tr * Ttrx);
  //    const aDual q_tr_y(S_, k_tr * Ttry);
  //    const aDual q_eev_x(S_, k_eev * Teevx);
  //    const aDual q_eev_y(S_, k_eev * Teevy);

  //    ind = 0;
  //    for (int i = 0; i < ns_; i++) flux1[ind++] = aDual(S_, 0.0);
  //    flux1[ind++] = mixture_p - txx;
  //    flux1[ind++] = -txy;
  //    flux1[ind++] = -q_tr_x - q_eev_x;
  //    flux1[ind++] = -q_eev_x;
  //    for (int i = 0; i < ns_; i++) flux1[ind++] = aDual(S_, 0.0);
  //    flux1[ind++] = -txy;
  //    flux1[ind++] = mixture_p - tyy;
  //    flux1[ind++] = -q_tr_y - q_eev_y;
  //    flux1[ind] = -q_eev_y;

  //    // pdss
  //    ind = DSS_ * ipoint;
  //    for (int ds = 0; ds < DS_; ds++)
  //      for (int istate = 0; istate < S_; istate++)
  //        flux_jacobi[ind++] = flux1[ds].df[istate];
  //  }
  //  {
  //    GET_SOLUTION_PS(, owner_u);

  //    // pds over psd
  //    ind = DS_ * ipoint;
  //    std::vector<aDual> dx(ns_, aDual(DS_));
  //    for (int i = 0; i < ns_; i++)
  //      dx[i] = aDual(DS_, owner_div_u[ind++], 2 * i);
  //    const aDual ux(DS_, owner_div_u[ind++], 2 * ns_);
  //    const aDual vx(DS_, owner_div_u[ind++], 2 * (ns_ + 1));
  //    const aDual Ttrx(DS_, owner_div_u[ind++], 2 * (ns_ + 2));
  //    const aDual Teevx(DS_, owner_div_u[ind++], 2 * (ns_ + 3));
  //    std::vector<aDual> dy(ns_, aDual(DS_));
  //    for (int i = 0; i < ns_; i++)
  //      dy[i] = aDual(DS_, owner_div_u[ind++], 2 * i + 1);
  //    const aDual uy(DS_, owner_div_u[ind++], 2 * ns_ + 1);
  //    const aDual vy(DS_, owner_div_u[ind++], 2 * (ns_ + 1) + 1);
  //    const aDual Ttry(DS_, owner_div_u[ind++], 2 * (ns_ + 2) + 1);
  //    const aDual Teevy(DS_, owner_div_u[ind], 2 * (ns_ + 3) + 1);

  //    mixture_->SetDensity(&dim_d[0]);
  //    const double mixture_p = mixture_->GetPressure(dim_T_tr, dim_T_eev) / p_ref_;

  //    double sum = 0.0;
  //    const double* Y = mixture_->GetMassFraction();
  //    for (int i = 0; i < ns_; i++) {
  //      const double R = species[i]->GetSpecificGasConstant() * T_ref_ / e_ref_;
  //      sum = sum + Y[i] * R * Twall_;
  //    }
  //    const double d_wall = mixture_p / sum;
  //    for (int i = 0; i < ns_; i++) d_new[i] = Y[i] * d_wall * rho_ref_;
  //    mixture_->SetDensity(&d_new[0]);

  //    // Viscous flux
  //    const auto mu = mixture_->GetViscosity(dim_Twall, dim_Twall) / mu_ref_;
  //    const auto k_tr =
  //        mixture_->GetTransRotationConductivity(dim_Twall, dim_Twall) / k_ref_;
  //    const auto k_eev =
  //        mixture_->GetElectronicVibrationConductivity(dim_Twall, dim_Twall) / k_ref_;

  //    const aDual txx = c23_ * mu * (2.0 * ux - vy);
  //    const aDual txy = mu * (uy + vx);
  //    const aDual tyy = c23_ * mu * (2.0 * vy - ux);

  //    const aDual q_tr_x = k_tr * Ttrx;
  //    const aDual q_tr_y = k_tr * Ttry;
  //    const aDual q_eev_x = k_eev * Teevx;
  //    const aDual q_eev_y = k_eev * Teevy;

  //    ind = 0;
  //    for (int i = 0; i < ns_; i++) flux2[ind++] = aDual(DS_, 0.0);
  //    flux2[ind++] = -txx;
  //    flux2[ind++] = -txy;
  //    flux2[ind++] = -q_tr_x - q_eev_x;
  //    flux2[ind++] = -q_eev_x;
  //    for (int i = 0; i < ns_; i++) flux2[ind++] = aDual(DS_, 0.0);
  //    flux2[ind++] = -txy;
  //    flux2[ind++] = -tyy;
  //    flux2[ind++] = -q_tr_y - q_eev_y;
  //    flux2[ind] = -q_eev_y;

  //    // pdssd
  //    ind = DDSS_ * ipoint;
  //    for (int ds = 0; ds < DS_; ds++)
  //      for (int sd = 0; sd < DS_; sd++)
  //        flux_grad_jacobi[ind++] = flux2[ds].df[sd];
  //  }
  //}
  // Weak
  static const int max_num_bdry_points = equation_->GetMaxNumBdryPoints();
  static std::vector<double> flux_neighbor_jacobi(max_num_bdry_points * DSS_);
  static std::vector<double> flux_neighbor_grad_jacobi(max_num_bdry_points *
                                                       DDSS_);
  static std::vector<double> bdry_u_jacobi(max_num_bdry_points * SS_);

  equation_->ComputeNumFluxJacobi(num_points, flux_jacobi, flux_neighbor_jacobi,
                                  flux_grad_jacobi, flux_neighbor_grad_jacobi,
                                  owner_cell, -1, owner_u, owner_div_u,
                                  neighbor_u, owner_div_u, normal, coords);
  ComputeBdrySolutionJacobi(num_points, &bdry_u_jacobi[0], owner_u, owner_div_u,
                            normal, coords, time);

  for (int ipoint = 0; ipoint < num_points; ipoint++)
    gemmAB(1.0, &flux_neighbor_jacobi[ipoint * DSS_],
           &bdry_u_jacobi[ipoint * SS_], 1.0, &flux_jacobi[ipoint * DSS_], DS_,
           S_, S_);
}
// Boundary Name = LimitingIsothermalWall
// Dependency: Twall
LimitingIsothermalWallNS2DNeq2Tnondim::LimitingIsothermalWallNS2DNeq2Tnondim(
  const int bdry_tag, EquationNS2DNeq2Tnondim* equation)
  : BoundaryNS2DNeq2Tnondim(bdry_tag, equation) {
  auto& config = AVOCADO_CONFIG;
  scale_ = std::stod(config->GetConfigValue(BDRY_INPUT_I(bdry_tag, 0)));
  Twall_ =
    std::stod(config->GetConfigValue(BDRY_INPUT_I(bdry_tag, 1))) / T_ref_;

  MASTER_MESSAGE("LimitingIsothermalWall (tag=" + std::to_string(bdry_tag) +
    ")\n\tTwall = " + std::to_string(Twall_) +
    "\n\tScale factor = " + std::to_string(scale_) + "\n");

  if (scale_ > 1.0 || scale_ < 0.0)
    ERROR_MESSAGE("Wrong scale factor imposed : " + std::to_string(scale_));
}
void LimitingIsothermalWallNS2DNeq2Tnondim::ComputeBdrySolution(
  const int num_points, std::vector<double>& bdry_u,
  std::vector<double>& bdry_div_u, const std::vector<double>& owner_u,
  const std::vector<double>& owner_div_u, const std::vector<double>& normal,
  const std::vector<double>& coords, const double& time) {
  const auto& species = mixture_->GetSpecies();
  int ind = 0;
  std::vector<double> dim_d(ns_);
  for (int ipoint = 0; ipoint < num_points; ipoint++) {
    GET_NORMAL_PD(normal);

    // ps
    GET_SOLUTION_PS(, owner_u);

    // ps
    ind = S_ * ipoint;
    for (int i = 0; i < ns_; i++) bdry_u[ind++] = d[i];
    bdry_u[ind++] = -u;
    bdry_u[ind++] = -v;
    bdry_u[ind++] = Twall_;
    bdry_u[ind] = Twall_;
  }
  // Don't touch bdry_div_u.
}
void LimitingIsothermalWallNS2DNeq2Tnondim::ComputeBdryFlux(
  const int num_points, std::vector<double>& flux, FACE_INPUTS,
  const std::vector<double>& coords, const double& time) {
  static std::vector<double> Ds(ns_, 0.0);
  static std::vector<double> Isx(ns_, 0.0);
  static std::vector<double> Isy(ns_, 0.0);
  static std::vector<double> hs(ns_, 0.0);
  static std::vector<double> diffflux(2 * ns_, 0.0);
  static std::vector<double> d_new(ns_, 0.0);
  static double dim_Twall = Twall_ * T_ref_;

  const auto& species = mixture_->GetSpecies();
  const auto& hidx = mixture_->GetHeavyParticleIdx();
  const auto& eidx = mixture_->GetElectronIndex();
  // need check
  int ind = 0;
  std::vector<double> dim_d(ns_);
  for (int ipoint = 0; ipoint < num_points; ipoint++) {
    GET_SOLUTION_PS(, owner_u);

    GET_SOLUTION_GRAD_PDS(, owner_div_u);

    // Convective flux
    mixture_->SetDensity(&dim_d[0]);
    const double mixture_d = mixture_->GetTotalDensity() / rho_ref_;
    const double mixture_p = mixture_->GetPressure(dim_T_tr, dim_T_eev) / p_ref_;

    double sum = 0.0;
    const double* Y = mixture_->GetMassFraction();
    for (int i = 0; i < ns_; i++) {
      const double R = species[i]->GetSpecificGasConstant() * T_ref_ / e_ref_;
      sum = sum + Y[i] * R * Twall_;
    }
    const double d_wall = mixture_p / sum;
    for (int i = 0; i < ns_; i++) d_new[i] = Y[i] * d_wall * rho_ref_;
    mixture_->SetDensity(&d_new[0]);

    // Viscous flux
    const double mu = mixture_->GetViscosity(dim_Twall, dim_Twall) / mu_ref_;
    const double k_tr =
        mixture_->GetTransRotationConductivity(dim_Twall, dim_Twall) / k_ref_;
    const double k_eev =
        mixture_->GetElectronicVibrationConductivity(dim_Twall, dim_Twall) / k_ref_;

    const auto txx = c23_ * mu * (2.0 * ux - vy);
    const auto txy = mu * (uy + vx);
    const auto tyy = c23_ * mu * (2.0 * vy - ux);

    const double q_tr_x = k_tr * Ttrx;
    const double q_tr_y = k_tr * Ttry;
    const double q_eev_x = k_eev * Teevx;
    const double q_eev_y = k_eev * Teevy;

    ind = DS_ * ipoint;
    for (int i = 0; i < ns_; i++) flux[ind++] = 0.0;
    flux[ind++] = mixture_p - scale_ * txx;
    flux[ind++] = -scale_ * txy;
    flux[ind++] = -scale_ * q_tr_x - scale_ * q_eev_x;
    flux[ind++] = -scale_ * q_eev_x;
    for (int i = 0; i < ns_; i++) flux[ind++] = 0.0;
    flux[ind++] = -scale_ * txy;
    flux[ind++] = mixture_p - scale_ * tyy;
    flux[ind++] = -scale_ * q_tr_y - scale_ * q_eev_y;
    flux[ind] = -scale_ * q_eev_y;
  }
}
void LimitingIsothermalWallNS2DNeq2Tnondim::ComputeBdrySolutionJacobi(
  const int num_points, double* bdry_u_jacobi,
  const std::vector<double>& owner_u, const std::vector<double>& owner_div_u,
  const std::vector<double>& normal, const std::vector<double>& coords,
  const double& time) {
  static std::vector<aDual> bdry_sol(S_, aDual(S_));
  int ind = 0;
  memset(bdry_u_jacobi, 0, num_points * SS_ * sizeof(double));

  const auto& species = mixture_->GetSpecies();

  std::vector<double> dim_d(ns_);
  for (int ipoint = 0; ipoint < num_points; ipoint++) {
    GET_NORMAL_PD(normal);

    // ps
    ind = S_ * ipoint;
    std::vector<aDual> d(ns_, aDual(S_));
    for (int i = 0; i < ns_; i++) {
      d[i] = aDual(S_, owner_u[ind++], i);
    }
    aDual u(S_, owner_u[ind++], ns_);
    aDual v(S_, owner_u[ind++], ns_ + 1);
    aDual T_tr(S_, owner_u[ind++], ns_ + 2);
    aDual T_eev(S_, owner_u[ind], ns_ + 3);

    for (int i = 0; i < ns_; i++) bdry_sol[i] = d[i];
    bdry_sol[ns_] = -u;
    bdry_sol[ns_ + 1] = -v;
    bdry_sol[ns_ + 2] = aDual(S_, Twall_);
    bdry_sol[ns_ + 3] = aDual(S_, Twall_);

    ind = SS_ * ipoint;
    for (int i = 0; i < S_; i++)
      for (int j = 0; j < S_; j++) bdry_u_jacobi[ind++] = bdry_sol[i].df[j];
  }
}
void LimitingIsothermalWallNS2DNeq2Tnondim::ComputeBdryFluxJacobi(
  const int num_points, std::vector<double>& flux_jacobi,
  std::vector<double>& flux_grad_jacobi, FACE_INPUTS,
  const std::vector<double>& coords, const double& time) {
  // flux_jacobi(ds1s2) = F(ds1) over U(s2)
  // flux_grad_jacobi(d1s1 s2d2) = F(d1s1) over gradU(d2s2)
     
  // Strong
  static std::vector<double> Ds(ns_, 0.0);
  static std::vector<double> Isx(ns_, 0.0);
  static std::vector<double> Isy(ns_, 0.0);
  static std::vector<double> hs(ns_, 0.0);
  static std::vector<double> d_new(ns_, 0.0);
  static std::vector<aDual> flux1(DS_, aDual(S_));
  static std::vector<aDual> flux2(DS_, aDual(DS_));
  static double dim_Twall = Twall_ * T_ref_;

  const auto& species = mixture_->GetSpecies();
  const auto& hidx = mixture_->GetHeavyParticleIdx();
  const auto& eidx = mixture_->GetElectronIndex();

  int ind = 0;
  std::vector<double> dim_d(ns_);
  for (int ipoint = 0; ipoint < num_points; ipoint++) {
    {
      // ps
      ind = S_ * ipoint;
      std::vector<aDual> d(ns_, aDual(S_));
      for (int i = 0; i < ns_; i++) {
        dim_d[i] = owner_u[ind] * rho_ref_;
        d[i] = aDual(S_, owner_u[ind++], i);
      }
      aDual u(S_, owner_u[ind++], ns_);
      aDual v(S_, owner_u[ind++], ns_ + 1);
      aDual T_tr(S_, owner_u[ind++], ns_ + 2);
      aDual T_eev(S_, owner_u[ind], ns_ + 3);
      const auto dim_T_tr = T_tr * T_ref_;
      const auto dim_T_eev = T_eev * T_ref_;

      GET_SOLUTION_GRAD_PDS(, owner_div_u);

      // Convective flux
      mixture_->SetDensity(&dim_d[0]);
      aDual mixture_d(S_, 0.0);
      for (int i = 0; i < ns_; i++) mixture_d = mixture_d + d[i];
      const auto mixture_d_inv = 1.0 / mixture_d;

      aDual mixture_p(S_);
      mixture_p.f = mixture_->GetPressureJacobian(dim_T_tr.f, dim_T_eev.f,
                                                  &mixture_p.df[0]);
      std::swap(mixture_p.df[ns_], mixture_p.df[ns_ + 2]);
      std::swap(mixture_p.df[ns_ + 1], mixture_p.df[ns_ + 3]);
      mixture_p.f /= p_ref_;
      for (int i = 0; i < ns_; i++) mixture_p.df[i] *= rho_ref_ / p_ref_;
      mixture_p.df[ns_ + 2] *= T_ref_ / p_ref_;
      mixture_p.df[ns_ + 3] *= T_ref_ / p_ref_;

      aDual sum(S_, 0.0);
      std::vector<aDual> Y(ns_, aDual(S_));
      for (int i = 0; i < ns_; i++) {
        const double R = species[i]->GetSpecificGasConstant() * T_ref_ / e_ref_;
        Y[i] = d[i] * mixture_d_inv;
        sum = sum + Y[i] * R * Twall_;
      }
      const auto mixture_d_wall = mixture_p / sum;
      for (int i = 0; i < ns_; i++) d_new[i] = Y[i].f * mixture_d_wall.f * rho_ref_;
      mixture_->SetDensity(&d_new[0]);

      // Viscous flux // need check -> this assumes the dwall_ to be independent
      // / check useless computations to solution variables
      const auto mu = mixture_->GetViscosity(dim_Twall, dim_Twall) / mu_ref_;
      const auto k_tr =
          mixture_->GetTransRotationConductivity(dim_Twall, dim_Twall) / k_ref_;
      const auto k_eev =
          mixture_->GetElectronicVibrationConductivity(dim_Twall, dim_Twall) / k_ref_;

      const aDual txx(S_, c23_ * mu * (2.0 * ux - vy));
      const aDual txy(S_, mu * (uy + vx));
      const aDual tyy(S_, c23_ * mu * (2.0 * vy - ux));

      const aDual q_tr_x(S_, k_tr * Ttrx);
      const aDual q_tr_y(S_, k_tr * Ttry);
      const aDual q_eev_x(S_, k_eev * Teevx);
      const aDual q_eev_y(S_, k_eev * Teevy);

      ind = 0;
      for (int i = 0; i < ns_; i++) flux1[ind++] = aDual(S_, 0.0);
      flux1[ind++] = mixture_p - scale_ * txx;
      flux1[ind++] = -scale_ * txy;
      flux1[ind++] = -scale_ * q_tr_x - scale_ * q_eev_x;
      flux1[ind++] = -scale_ * q_eev_x;
      for (int i = 0; i < ns_; i++) flux1[ind++] = aDual(S_, 0.0);
      flux1[ind++] = -scale_ * txy;
      flux1[ind++] = mixture_p - scale_ * tyy;
      flux1[ind++] = -scale_ * q_tr_y - scale_ * q_eev_y;
      flux1[ind] = -scale_ * q_eev_y;

      // pdss
      ind = DSS_ * ipoint;
      for (int ds = 0; ds < DS_; ds++)
        for (int istate = 0; istate < S_; istate++)
          flux_jacobi[ind++] = flux1[ds].df[istate];
    }
    {
      GET_SOLUTION_PS(, owner_u);

      // pds over psd
      ind = DS_ * ipoint;
      std::vector<aDual> dx(ns_, aDual(DS_));
      for (int i = 0; i < ns_; i++)
        dx[i] = aDual(DS_, owner_div_u[ind++], 2 * i);
      const aDual ux(DS_, owner_div_u[ind++], 2 * ns_);
      const aDual vx(DS_, owner_div_u[ind++], 2 * (ns_ + 1));
      const aDual Ttrx(DS_, owner_div_u[ind++], 2 * (ns_ + 2));
      const aDual Teevx(DS_, owner_div_u[ind++], 2 * (ns_ + 3));
      std::vector<aDual> dy(ns_, aDual(DS_));
      for (int i = 0; i < ns_; i++)
        dy[i] = aDual(DS_, owner_div_u[ind++], 2 * i + 1);
      const aDual uy(DS_, owner_div_u[ind++], 2 * ns_ + 1);
      const aDual vy(DS_, owner_div_u[ind++], 2 * (ns_ + 1) + 1);
      const aDual Ttry(DS_, owner_div_u[ind++], 2 * (ns_ + 2) + 1);
      const aDual Teevy(DS_, owner_div_u[ind], 2 * (ns_ + 3) + 1);

      mixture_->SetDensity(&dim_d[0]);
      const double mixture_p = mixture_->GetPressure(dim_T_tr, dim_T_eev) / p_ref_;

      double sum = 0.0;
      const double* Y = mixture_->GetMassFraction();
      for (int i = 0; i < ns_; i++) {
        const double R = species[i]->GetSpecificGasConstant() * T_ref_ / e_ref_;
        sum = sum + Y[i] * R * Twall_;
      }
      const double d_wall = mixture_p / sum;
      for (int i = 0; i < ns_; i++) d_new[i] = Y[i] * d_wall * rho_ref_;
      mixture_->SetDensity(&d_new[0]);

      // Viscous flux
      const auto mu = mixture_->GetViscosity(dim_Twall, dim_Twall) / mu_ref_;
      const auto k_tr =
          mixture_->GetTransRotationConductivity(dim_Twall, dim_Twall) / k_ref_;
      const auto k_eev =
          mixture_->GetElectronicVibrationConductivity(dim_Twall, dim_Twall) / k_ref_;

      const aDual txx = c23_ * mu * (2.0 * ux - vy);
      const aDual txy = mu * (uy + vx);
      const aDual tyy = c23_ * mu * (2.0 * vy - ux);

      const aDual q_tr_x = k_tr * Ttrx;
      const aDual q_tr_y = k_tr * Ttry;
      const aDual q_eev_x = k_eev * Teevx;
      const aDual q_eev_y = k_eev * Teevy;

      ind = 0;
      for (int i = 0; i < ns_; i++) flux2[ind++] = aDual(DS_, 0.0);
      flux2[ind++] = -scale_ * txx;
      flux2[ind++] = -scale_ * txy;
      flux2[ind++] = -scale_ * q_tr_x - scale_ * q_eev_x;
      flux2[ind++] = -scale_ * q_eev_x;
      for (int i = 0; i < ns_; i++) flux2[ind++] = aDual(DS_, 0.0);
      flux2[ind++] = -scale_ * txy;
      flux2[ind++] = -scale_ * tyy;
      flux2[ind++] = -scale_ * q_tr_y - scale_ * q_eev_y;
      flux2[ind] = -scale_ * q_eev_y;

      // pdssd
      ind = DDSS_ * ipoint;
      for (int ds = 0; ds < DS_; ds++)
        for (int sd = 0; sd < DS_; sd++)
          flux_grad_jacobi[ind++] = flux2[ds].df[sd];
    }
  }
}
// Boundary Name = SupersonicInflow
// Dependency: rho_1, ..., rho_ns, u, v, T_tr, T_eev
SupersonicInflowBdryNS2DNeq2Tnondim::SupersonicInflowBdryNS2DNeq2Tnondim(
    const int bdry_tag, EquationNS2DNeq2Tnondim* equation)
    : BoundaryNS2DNeq2Tnondim(bdry_tag, equation) {
  MASTER_MESSAGE("SupersonicInflow boundary (tag=" + std::to_string(bdry_tag) +
                 ")\n");

  auto& config = AVOCADO_CONFIG;
  d_.resize(ns_, 0.0);
  std::vector<double> dim_d(ns_, 0.0);
  for (int i = 0; i < ns_; i++) {
    dim_d[i] = std::stod(config->GetConfigValue(BDRY_INPUT_I(bdry_tag, i)));
    d_[i] = dim_d[i] / rho_ref_;
  }
  u_ = std::stod(config->GetConfigValue(BDRY_INPUT_I(bdry_tag, ns_))) / v_ref_;
  v_ = std::stod(config->GetConfigValue(BDRY_INPUT_I(bdry_tag, ns_ + 1))) / v_ref_;
  T_tr_ = std::stod(config->GetConfigValue(BDRY_INPUT_I(bdry_tag, ns_ + 2))) / T_ref_;
  T_eev_ = std::stod(config->GetConfigValue(BDRY_INPUT_I(bdry_tag, ns_ + 3))) / T_ref_;
  const double dim_T_tr = T_tr_ * T_ref_;
  const double dim_T_eev = T_eev_ * T_ref_;
    
  const auto& species = mixture_->GetSpecies();
  mixture_->SetDensity(&dim_d[0]);
  mixture_p_ = mixture_->GetPressure(dim_T_tr, dim_T_eev) / p_ref_;
  mixture_d_ = mixture_->GetTotalDensity() / rho_ref_;
  const double beta = mixture_->GetBeta(dim_T_tr);
  const double a = std::sqrt((1.0 + beta) * mixture_p_ / mixture_d_);
  const double Ma = std::sqrt(u_ * u_ + v_ * v_) / a;
  du_ = mixture_d_ * u_;
  dv_ = mixture_d_ * v_;
  double de = 0.0;
  de_eev_ = 0.0;
  for (int i = 0; i < ns_; i++) {
    const double e = species[i]->GetInternalEnergy(dim_T_tr, dim_T_eev) / e_ref_;
    const double e_eev = species[i]->GetElectronicVibrationEnergy(dim_T_eev) / e_ref_;
    de += d_[i] * e;
    de_eev_ += d_[i] * e_eev;
  }
  dE_ = de + 0.5 * (du_ * u_ + dv_ * v_);

  MASTER_MESSAGE("SupersonicInflow (tag=" + std::to_string(bdry_tag) +
                 ")"
                 "\n\tdensity = " +
                 std::to_string(mixture_d_ * rho_ref_) +
                 " kg/m3"
                 "\n\tx-velocity = " +
                 std::to_string(u_ * v_ref_) +
                 " m/s"
                 "\n\ty-velocity = " +
                 std::to_string(v_ * v_ref_) +
                 " m/s"
                 "\n\ttrans-rotational temperature = " +
                 std::to_string(dim_T_tr) +
                 " K"
                 "\n\telectron-electronic-vibration temperature = " +
                 std::to_string(dim_T_eev) +
                 " K"
                 "\n\tpressure = " +
                 std::to_string(mixture_p_ * p_ref_) +
                 " Pa"
                 "\n\tMach number = " +
                 std::to_string(Ma) + "\n");
}
void SupersonicInflowBdryNS2DNeq2Tnondim::ComputeBdrySolution(
    const int num_points, std::vector<double>& bdry_u,
    std::vector<double>& bdry_div_u, const std::vector<double>& owner_u,
    const std::vector<double>& owner_div_u, const std::vector<double>& normal,
    const std::vector<double>& coords, const double& time) {
  int ind = 0;
  for (int ipoint = 0; ipoint < num_points; ipoint++) {
    // ps
    ind = S_ * ipoint;
    for (int i = 0; i < ns_; i++) bdry_u[ind++] = d_[i];
    bdry_u[ind++] = u_;
    bdry_u[ind++] = v_;
    bdry_u[ind++] = T_tr_;
    bdry_u[ind] = T_eev_;
  }
  // Don't touch bdry_div_u.
}
void SupersonicInflowBdryNS2DNeq2Tnondim::ComputeBdryFlux(
    const int num_points, std::vector<double>& flux, FACE_INPUTS,
    const std::vector<double>& coords, const double& time) {
  int ind = 0;
  // convection only
  for (int ipoint = 0; ipoint < num_points; ipoint++) {
    ind = DS_ * ipoint;
    for (int i = 0; i < ns_; i++) flux[ind++] = d_[i] * u_;
    flux[ind++] = du_ * u_ + mixture_p_;
    flux[ind++] = du_ * v_;
    flux[ind++] = (dE_ + mixture_p_) * u_;
    flux[ind++] = de_eev_ * u_;
    for (int i = 0; i < ns_; i++) flux[ind++] = d_[i] * v_;
    flux[ind++] = dv_ * u_;
    flux[ind++] = dv_ * v_ + mixture_p_;
    flux[ind++] = (dE_ + mixture_p_) * v_;
    flux[ind] = de_eev_ * v_;
  }
}
void SupersonicInflowBdryNS2DNeq2Tnondim::ComputeBdrySolutionJacobi(
    const int num_points, double* bdry_u_jacobi,
    const std::vector<double>& owner_u, const std::vector<double>& owner_div_u,
    const std::vector<double>& normal, const std::vector<double>& coords,
    const double& time) {
  memset(bdry_u_jacobi, 0, num_points * SS_ * sizeof(double));
}
void SupersonicInflowBdryNS2DNeq2Tnondim::ComputeBdryFluxJacobi(
    const int num_points, std::vector<double>& flux_jacobi,
    std::vector<double>& flux_grad_jacobi, FACE_INPUTS,
    const std::vector<double>& coords, const double& time) {
  memset(&flux_jacobi[0], 0, num_points * DSS_ * sizeof(double));
  memset(&flux_grad_jacobi[0], 0, num_points * DDSS_ * sizeof(double));
}
// Boundary = SupersonicOutflow
// BdryInput() = -
SupersonicOutflowBdryNS2DNeq2Tnondim::SupersonicOutflowBdryNS2DNeq2Tnondim(
    const int bdry_tag, EquationNS2DNeq2Tnondim* equation)
    : BoundaryNS2DNeq2Tnondim(bdry_tag, equation) {
  MASTER_MESSAGE("SupersonicOutflow boundary (tag=" + std::to_string(bdry_tag) +
                 ")\n");
}
void SupersonicOutflowBdryNS2DNeq2Tnondim::ComputeBdrySolution(
    const int num_points, std::vector<double>& bdry_u,
    std::vector<double>& bdry_div_u, const std::vector<double>& owner_u,
    const std::vector<double>& owner_div_u, const std::vector<double>& normal,
    const std::vector<double>& coords, const double& time) {
  int ind = 0;
  for (int ipoint = 0; ipoint < num_points; ipoint++) {
    // ps
    ind = S_ * ipoint;
    for (int i = 0; i < S_; i++) bdry_u[ind + i] = owner_u[ind + i];
  }
}
void SupersonicOutflowBdryNS2DNeq2Tnondim::ComputeBdryFlux(
    const int num_points, std::vector<double>& flux, FACE_INPUTS,
    const std::vector<double>& coords, const double& time) {
  static std::vector<double> Ds(ns_, 0.0);
  static std::vector<double> Isx(ns_, 0.0);
  static std::vector<double> Isy(ns_, 0.0);
  static std::vector<double> hs(ns_, 0.0);
  static std::vector<double> diffflux(2 * ns_, 0.0);

  const auto& species = mixture_->GetSpecies();
  const auto& hidx = mixture_->GetHeavyParticleIdx();
  const auto& eidx = mixture_->GetElectronIndex();

  int ind = 0;
  std::vector<double> dim_d(ns_);
  for (int ipoint = 0; ipoint < num_points; ipoint++) {
    GET_SOLUTION_PS(, owner_u);
    GET_SOLUTION_GRAD_PDS(, owner_div_u);

    // Convective flux
    mixture_->SetDensity(&dim_d[0]);
    const double mixture_d = mixture_->GetTotalDensity() / rho_ref_;
    const double mixture_p = mixture_->GetPressure(dim_T_tr, dim_T_eev) / p_ref_;
    const double beta = mixture_->GetBeta(dim_T_tr);
    const double a = std::sqrt((1.0 + beta) * mixture_p / mixture_d);
    double de = 0.0;
    double de_eev = 0.0;
    for (int i = 0; i < ns_; i++) {
      const double e = species[i]->GetInternalEnergy(dim_T_tr, dim_T_eev) / e_ref_;
      const double e_eev = species[i]->GetElectronicVibrationEnergy(dim_T_eev) / e_ref_;
      de += d[i] * e;
      de_eev += d[i] * e_eev;
    }
    const double du = mixture_d * u;
    const double dv = mixture_d * v;
    const double dE = de + 0.5 * (du * u + dv * v);

    // Viscous flux
    const double mu = mixture_->GetViscosity(dim_T_tr, dim_T_eev) / mu_ref_;
    const double k_tr =
        mixture_->GetTransRotationConductivity(dim_T_tr, dim_T_eev) / k_ref_;
    const double k_eev =
        mixture_->GetElectronicVibrationConductivity(dim_T_tr, dim_T_eev) / k_ref_;

    const double* Y = mixture_->GetMassFraction();
    double mixture_dx = 0.0;
    double mixture_dy = 0.0;
    for (int i = 0; i < ns_; i++) {
      mixture_dx += dx[i];
      mixture_dy += dy[i];
    }

    mixture_->GetDiffusivity(&Ds[0], dim_T_tr, dim_T_eev);
    for (int i = 0; i < ns_; i++) {
      Ds[i] /= D_ref_;
      Isx[i] = -Ds[i] * (dx[i] - Y[i] * mixture_dx);
      Isy[i] = -Ds[i] * (dy[i] - Y[i] * mixture_dy);
    }

    double Ix_sum = 0.0;
    double Iy_sum = 0.0;
    for (const int& idx : hidx) {
      Ix_sum += Isx[idx];
      Iy_sum += Isy[idx];
    }

    for (const int& idx : hidx) {
      diffflux[idx] = -Isx[idx] + Y[idx] * Ix_sum;
      diffflux[idx + ns_] = -Isy[idx] + Y[idx] * Iy_sum;
    }

    if (eidx >= 0) {
      double sum_x = 0.0;
      double sum_y = 0.0;
      for (const int& idx : hidx) {
        const int charge = species[idx]->GetCharge();
        const double Mw = species[idx]->GetMolecularWeight();
        sum_x += diffflux[idx] * charge / Mw;
        sum_y += diffflux[idx + ns_] * charge / Mw;
      }
      const double Mw_e = species[eidx]->GetMolecularWeight();
      diffflux[eidx] = Mw_e * sum_x;
      diffflux[eidx + ns_] = Mw_e * sum_y;
    }

    const auto txx = c23_ * mu * (2.0 * ux - vy);
    const auto txy = mu * (uy + vx);
    const auto tyy = c23_ * mu * (2.0 * vy - ux);

    double Jh_x = 0.0;
    double Jh_y = 0.0;
    double Je_eev_x = 0.0;
    double Je_eev_y = 0.0;
    for (int i = 0; i < ns_; i++) {
      const double h = species[i]->GetEnthalpy(dim_T_tr, dim_T_eev) / e_ref_;
      const double e_eev = species[i]->GetElectronicVibrationEnergy(dim_T_eev) / e_ref_;
      Jh_x += diffflux[i] * h;
      Je_eev_x += diffflux[i] * e_eev;
      Jh_y += diffflux[i + ns_] * h;
      Je_eev_y += diffflux[i + ns_] * e_eev;
    }
    const double q_tr_x = k_tr * Ttrx;
    const double q_tr_y = k_tr * Ttry;
    const double q_eev_x = k_eev * Teevx;
    const double q_eev_y = k_eev * Teevy;

    ind = DS_ * ipoint;
    for (int i = 0; i < ns_; i++) flux[ind++] = d[i] * u - diffflux[i];
    flux[ind++] = du * u + mixture_p - txx;
    flux[ind++] = du * v - txy;
    flux[ind++] =
        (dE + mixture_p) * u - txx * u - txy * v - q_tr_x - q_eev_x - Jh_x;
    flux[ind++] = de_eev * u - q_eev_x - Je_eev_x;
    for (int i = 0; i < ns_; i++) flux[ind++] = d[i] * v - diffflux[i + ns_];
    flux[ind++] = dv * u - txy;
    flux[ind++] = dv * v + mixture_p - tyy;
    flux[ind++] =
        (dE + mixture_p) * v - txy * u - tyy * v - q_tr_y - q_eev_y - Jh_y;
    flux[ind] = de_eev * v - q_eev_y - Je_eev_y;
  }
}
void SupersonicOutflowBdryNS2DNeq2Tnondim::ComputeBdrySolutionJacobi(
    const int num_points, double* bdry_u_jacobi,
    const std::vector<double>& owner_u, const std::vector<double>& owner_div_u,
    const std::vector<double>& normal, const std::vector<double>& coords,
    const double& time) {
  int ind = 0;
  memset(&bdry_u_jacobi[ind], 0, num_points * SS_ * sizeof(double));
  for (int ipoint = 0; ipoint < num_points; ipoint++) {
    // ps
    ind = SS_ * ipoint;
    for (int i = 0; i < SS_; i += (S_ + 1)) bdry_u_jacobi[ind + i] = 1.0;
  }
}
void SupersonicOutflowBdryNS2DNeq2Tnondim::ComputeBdryFluxJacobi(
    const int num_points, std::vector<double>& flux_jacobi,
    std::vector<double>& flux_grad_jacobi, FACE_INPUTS,
    const std::vector<double>& coords, const double& time) {
  static std::vector<double> Ds(ns_, 0.0);
  static std::vector<double> Isx(ns_, 0.0);
  static std::vector<double> Isy(ns_, 0.0);
  static std::vector<double> hs(ns_, 0.0);
  static std::vector<aDual> flux1(DS_, aDual(S_));
  static std::vector<aDual> flux2(DS_, aDual(DS_));

  int ind = 0;

  const auto& species = mixture_->GetSpecies();
  const auto& hidx = mixture_->GetHeavyParticleIdx();
  const auto& eidx = mixture_->GetElectronIndex();

  // flux jacobi
  std::vector<double> dim_d(ns_);
  for (int ipoint = 0; ipoint < num_points; ipoint++) {
    {
      // ps
      ind = S_ * ipoint;
      std::vector<aDual> d(ns_, aDual(S_));
      for (int i = 0; i < ns_; i++) {
        dim_d[i] = owner_u[ind] * rho_ref_;
        d[i] = aDual(S_, owner_u[ind++], i);
      }
      aDual u(S_, owner_u[ind++], ns_);
      aDual v(S_, owner_u[ind++], ns_ + 1);
      aDual T_tr(S_, owner_u[ind++], ns_ + 2);
      aDual T_eev(S_, owner_u[ind], ns_ + 3);
      const auto dim_T_tr = T_tr * T_ref_;
      const auto dim_T_eev = T_eev * T_ref_;

      GET_SOLUTION_GRAD_PDS(, owner_div_u);

      // Convective flux
      mixture_->SetDensity(&dim_d[0]);
      aDual mixture_d(S_, mixture_->GetTotalDensity() / rho_ref_);
      for (int i = 0; i < ns_; i++) mixture_d.df[i] = 1.0;

      aDual mixture_p(S_);
      mixture_p.f = mixture_->GetPressureJacobian(dim_T_tr.f, dim_T_eev.f,
                                                  &mixture_p.df[0]);
      std::swap(mixture_p.df[ns_], mixture_p.df[ns_ + 2]);
      std::swap(mixture_p.df[ns_ + 1], mixture_p.df[ns_ + 3]);
      mixture_p.f /= p_ref_;
      for (int i = 0; i < ns_; i++) mixture_p.df[i] *= rho_ref_ / p_ref_;
      mixture_p.df[ns_ + 2] *= T_ref_ / p_ref_;
      mixture_p.df[ns_ + 3] *= T_ref_ / p_ref_;

      aDual beta(S_);
      beta.f = mixture_->GetBetaJacobian(dim_T_tr.f, &beta.df[0]);
      for (int i = 0; i < ns_; i++) beta.df[i] *= rho_ref_;
      beta.df[ns_ + 2] *= T_ref_;
      beta.df[ns_ + 3] *= T_ref_;

      const auto a = std::sqrt((1.0 + beta) * mixture_p / mixture_d);
      aDual de(S_);
      aDual de_eev(S_);
      for (int i = 0; i < ns_; i++) {
        const auto e =
            species[i]->GetInternalEnergy(dim_T_tr.f, dim_T_eev.f) / e_ref_;
        const auto e_eev =
            species[i]->GetElectronicVibrationEnergy(dim_T_eev.f) / e_ref_;
        const auto Cv_tr =
            species[i]->GetTransRotationSpecificHeat(dim_T_tr.f) *
                           T_ref_ / e_ref_;
        const auto Cv_eev =
            species[i]->GetElectronicVibrationSpecificHeat(dim_T_eev.f) *
            T_ref_ /
            e_ref_;

        de.f += d[i].f * e;
        de.df[i] = e;
        de.df[ns_ + 2] += d[i].f * Cv_tr;
        de.df[ns_ + 3] += d[i].f * Cv_eev;

        de_eev.f += d[i].f * e_eev;
        de_eev.df[i] = e_eev;
        de_eev.df[ns_ + 3] += d[i].f * Cv_eev;
      }
      const auto du = mixture_d * u;
      const auto dv = mixture_d * v;
      const auto dE = de + 0.5 * (du * u + dv * v);

      // Viscous flux
      const double mu =
          mixture_->GetViscosity(dim_T_tr.f, dim_T_eev.f) / mu_ref_;
      const double k_tr =
          mixture_->GetTransRotationConductivity(dim_T_tr.f, dim_T_eev.f) /
          k_ref_;
      const double k_eev = mixture_->GetElectronicVibrationConductivity(
                               dim_T_tr.f, dim_T_eev.f) /
          k_ref_;

      std::vector<aDual> Y(ns_, aDual(S_));
      for (int i = 0; i < ns_; i++) Y[i] = d[i] / mixture_d;

      double mixture_dx = 0.0;
      double mixture_dy = 0.0;
      for (int i = 0; i < ns_; i++) {
        mixture_dx += dx[i];
        mixture_dy += dy[i];
      }

      std::vector<aDual> Isx(ns_, aDual(S_));
      std::vector<aDual> Isy(ns_, aDual(S_));
      mixture_->GetDiffusivity(&Ds[0], dim_T_tr.f, dim_T_eev.f);
      for (int i = 0; i < ns_; i++) {
        Ds[i] /= D_ref_;
        Isx[i] = -Ds[i] * (dx[i] - Y[i] * mixture_dx);
        Isy[i] = -Ds[i] * (dy[i] - Y[i] * mixture_dy);
      }

      aDual Ix_sum(S_);
      aDual Iy_sum(S_);
      for (const int& idx : hidx) {
        Ix_sum = Ix_sum + Isx[idx];
        Iy_sum = Iy_sum + Isy[idx];
      }

      std::vector<aDual> diffflux(2 * ns_, aDual(S_));
      for (const int& idx : hidx) {
        diffflux[idx] = -Isx[idx] + Y[idx] * Ix_sum;
        diffflux[idx + ns_] = -Isy[idx] + Y[idx] * Iy_sum;
      }
      if (eidx >= 0) {
        aDual sum_x(S_);
        aDual sum_y(S_);
        for (const int& idx : hidx) {
          const int charge = species[idx]->GetCharge();
          const double Mw = species[idx]->GetMolecularWeight();
          sum_x = sum_x + diffflux[idx] * charge / Mw;
          sum_y = sum_y + diffflux[idx + ns_] * charge / Mw;
        }
        const double Mw_e = species[eidx]->GetMolecularWeight();
        diffflux[eidx] = Mw_e * sum_x;
        diffflux[eidx + ns_] = Mw_e * sum_y;
      }

      const auto txx = c23_ * mu * (2.0 * ux - vy);
      const auto txy = mu * (uy + vx);
      const auto tyy = c23_ * mu * (2.0 * vy - ux);

      aDual Jh_x(S_);
      aDual Jh_y(S_);
      aDual Je_eev_x(S_);
      aDual Je_eev_y(S_);
      for (int i = 0; i < ns_; i++) {
        const double h =
            species[i]->GetEnthalpy(dim_T_tr.f, dim_T_eev.f) / e_ref_;
        const double e_eev =
            species[i]->GetElectronicVibrationEnergy(dim_T_eev.f) / e_ref_;
        const auto Cv_tr =
            species[i]->GetTransRotationSpecificHeat(dim_T_tr.f) *
                           T_ref_ / e_ref_;
        const auto Cv_eev =
            species[i]->GetElectronicVibrationSpecificHeat(dim_T_eev.f) *
            T_ref_ /
            e_ref_;
        const double R = species[i]->GetSpecificGasConstant() * T_ref_ / e_ref_;

        const auto Cp_tr = Cv_tr + (i == eidx) ? 0.0 : R;
        const auto Cp_eev = Cv_eev + (i == eidx) ? R : 0.0;

        Jh_x = Jh_x + diffflux[i] * h;
        Jh_x.df[ns_ + 2] += diffflux[i].f * Cp_tr;
        Jh_x.df[ns_ + 3] += diffflux[i].f * Cp_eev;
        Jh_y = Jh_y + diffflux[i + ns_] * h;
        Jh_y.df[ns_ + 2] += diffflux[i + ns_].f * Cp_tr;
        Jh_y.df[ns_ + 3] += diffflux[i + ns_].f * Cp_eev;

        Je_eev_x = Je_eev_x + diffflux[i] * e_eev;
        Je_eev_x.df[ns_ + 3] += diffflux[i].f * Cv_eev;
        Je_eev_y = Je_eev_y + diffflux[i + ns_] * e_eev;
        Je_eev_y.df[ns_ + 3] += diffflux[i + ns_].f * Cv_eev;
      }

      const aDual q_tr_x(S_, k_tr * Ttrx);
      const aDual q_tr_y(S_, k_tr * Ttry);
      const aDual q_eev_x(S_, k_eev * Teevx);
      const aDual q_eev_y(S_, k_eev * Teevy);

      ind = 0;
      for (int i = 0; i < ns_; i++) flux1[ind++] = d[i] * u - diffflux[i];
      flux1[ind++] = du * u + mixture_p - txx;
      flux1[ind++] = du * v - txy;
      flux1[ind++] =
          (dE + mixture_p) * u - txx * u - txy * v - q_tr_x - q_eev_x - Jh_x;
      flux1[ind++] = de_eev * u - q_eev_x - Je_eev_x;
      for (int i = 0; i < ns_; i++) flux1[ind++] = d[i] * v - diffflux[i + ns_];
      flux1[ind++] = dv * u - txy;
      flux1[ind++] = dv * v + mixture_p - tyy;
      flux1[ind++] =
          (dE + mixture_p) * v - txy * u - tyy * v - q_tr_y - q_eev_y - Jh_y;
      flux1[ind] = de_eev * v - q_eev_y - Je_eev_y;

      // pdss
      ind = DSS_ * ipoint;
      for (int ds = 0; ds < DS_; ds++)
        for (int istate = 0; istate < S_; istate++)
          flux_jacobi[ind++] = flux1[ds].df[istate];
    }
    {
      GET_SOLUTION_PS(, owner_u);
      // pds over psd
      ind = DS_ * ipoint;
      std::vector<aDual> dx(ns_, aDual(DS_));
      for (int i = 0; i < ns_; i++)
        dx[i] = aDual(DS_, owner_div_u[ind++], 2 * i);
      const aDual ux(DS_, owner_div_u[ind++], 2 * ns_);
      const aDual vx(DS_, owner_div_u[ind++], 2 * (ns_ + 1));
      const aDual Ttrx(DS_, owner_div_u[ind++], 2 * (ns_ + 2));
      const aDual Teevx(DS_, owner_div_u[ind++], 2 * (ns_ + 3));
      std::vector<aDual> dy(ns_, aDual(DS_));
      for (int i = 0; i < ns_; i++)
        dy[i] = aDual(DS_, owner_div_u[ind++], 2 * i + 1);
      const aDual uy(DS_, owner_div_u[ind++], 2 * ns_ + 1);
      const aDual vy(DS_, owner_div_u[ind++], 2 * (ns_ + 1) + 1);
      const aDual Ttry(DS_, owner_div_u[ind++], 2 * (ns_ + 2) + 1);
      const aDual Teevy(DS_, owner_div_u[ind], 2 * (ns_ + 3) + 1);

      // Viscous flux
      mixture_->SetDensity(&dim_d[0]);
      const double mu = mixture_->GetViscosity(dim_T_tr, dim_T_eev) / mu_ref_;
      const double k_tr =
          mixture_->GetTransRotationConductivity(dim_T_tr, dim_T_eev) / k_ref_;
      const double k_eev =
          mixture_->GetElectronicVibrationConductivity(dim_T_tr, dim_T_eev) /
          k_ref_;

      const double* Y = mixture_->GetMassFraction();
      aDual mixture_dx(DS_);
      aDual mixture_dy(DS_);
      for (int i = 0; i < ns_; i++) {
        mixture_dx = mixture_dx + dx[i];
        mixture_dy = mixture_dy + dy[i];
      }

      std::vector<aDual> Isx(ns_, aDual(DS_));
      std::vector<aDual> Isy(ns_, aDual(DS_));
      mixture_->GetDiffusivity(&Ds[0], dim_T_tr, dim_T_eev);
      for (int i = 0; i < ns_; i++) {
        Ds[i] /= D_ref_;
        Isx[i] = -Ds[i] * (dx[i] - Y[i] * mixture_dx);
        Isy[i] = -Ds[i] * (dy[i] - Y[i] * mixture_dy);
      }

      aDual Ix_sum(DS_);
      aDual Iy_sum(DS_);
      for (const int& idx : hidx) {
        Ix_sum = Ix_sum + Isx[idx];
        Iy_sum = Iy_sum + Isy[idx];
      }

      std::vector<aDual> diffflux(2 * ns_, aDual(DS_));
      for (const int& idx : hidx) {
        diffflux[idx] = -Isx[idx] + Y[idx] * Ix_sum;
        diffflux[idx + ns_] = -Isy[idx] + Y[idx] * Iy_sum;
      }
      if (eidx >= 0) {
        aDual sum_x(DS_);
        aDual sum_y(DS_);
        for (const int& idx : hidx) {
          const int charge = species[idx]->GetCharge();
          const double Mw = species[idx]->GetMolecularWeight();
          sum_x = sum_x + diffflux[idx] * charge / Mw;
          sum_y = sum_y + diffflux[idx + ns_] * charge / Mw;
        }
        const double Mw_e = species[eidx]->GetMolecularWeight();
        diffflux[eidx] = Mw_e * sum_x;
        diffflux[eidx + ns_] = Mw_e * sum_y;
      }

      const auto txx = c23_ * mu * (2.0 * ux - vy);
      const auto txy = mu * (uy + vx);
      const auto tyy = c23_ * mu * (2.0 * vy - ux);

      aDual Jh_x(DS_);
      aDual Jh_y(DS_);
      aDual Je_eev_x(DS_);
      aDual Je_eev_y(DS_);

      for (int i = 0; i < ns_; i++) {
        const double h = species[i]->GetEnthalpy(dim_T_tr, dim_T_eev) / e_ref_;
        const double e_eev =
            species[i]->GetElectronicVibrationEnergy(dim_T_eev) / e_ref_;

        Jh_x = Jh_x + diffflux[i] * h;
        Jh_y = Jh_y + diffflux[i + ns_] * h;
        Je_eev_x = Je_eev_x + diffflux[i] * e_eev;
        Je_eev_y = Je_eev_y + diffflux[i + ns_] * e_eev;
      }

      const auto q_tr_x = k_tr * Ttrx;
      const auto q_tr_y = k_tr * Ttry;
      const auto q_eev_x = k_eev * Teevx;
      const auto q_eev_y = k_eev * Teevy;

      ind = 0;
      for (int i = 0; i < ns_; i++) flux2[ind++] = -diffflux[i];
      flux2[ind++] = -txx;
      flux2[ind++] = -txy;
      flux2[ind++] = -txx * u - txy * v - q_tr_x - q_eev_x - Jh_x;
      flux2[ind++] = -q_eev_x - Je_eev_x;
      for (int i = 0; i < ns_; i++) flux2[ind++] = -diffflux[i + ns_];
      flux2[ind++] = -txy;
      flux2[ind++] = -tyy;
      flux2[ind++] = -txy * u - tyy * v - q_tr_y - q_eev_y - Jh_y;
      flux2[ind] = -q_eev_y - Je_eev_y;

      // pdssd
      ind = DDSS_ * ipoint;
      for (int ds = 0; ds < DS_; ds++)
        for (int sd = 0; sd < DS_; sd++)
          flux_grad_jacobi[ind++] = flux2[ds].df[sd];
    }
  }
}
// Boundary Name = CatalyticWall
// BdryInput() = Twall, gamma_N, gamma_O, maxiter (optional)
CatalyticWallNS2DNeq2Tnondim::CatalyticWallNS2DNeq2Tnondim(
    const int bdry_tag, EquationNS2DNeq2Tnondim* equation)
    : BoundaryNS2DNeq2Tnondim(bdry_tag, equation) {
  MASTER_MESSAGE("CatalyticWall (tag=" + std::to_string(bdry_tag) + ")\n");
  ERROR_MESSAGE("CatalyticWall is not supported in this version!!!");
}
void CatalyticWallNS2DNeq2Tnondim::ComputeBdrySolution(
    const int num_points, std::vector<double>& bdry_u,
    std::vector<double>& bdry_div_u, const std::vector<double>& owner_u,
    const std::vector<double>& owner_div_u, const std::vector<double>& normal,
    const std::vector<double>& coords, const double& time) {}
void CatalyticWallNS2DNeq2Tnondim::ComputeBdryFlux(
    const int num_points, std::vector<double>& flux, FACE_INPUTS,
    const std::vector<double>& coords, const double& time) {}
void CatalyticWallNS2DNeq2Tnondim::ComputeBdrySolutionJacobi(
    const int num_points, double* bdry_u_jacobi,
    const std::vector<double>& owner_u, const std::vector<double>& owner_div_u,
    const std::vector<double>& normal, const std::vector<double>& coords,
    const double& time) {}
void CatalyticWallNS2DNeq2Tnondim::ComputeBdryFluxJacobi(
    const int num_points, std::vector<double>& flux_jacobi,
    std::vector<double>& flux_grad_jacobi, FACE_INPUTS,
    const std::vector<double>& coords, const double& time) {}
// Boundary Name = SuperCatalyticWall
// BdryInput() = Twall, rho_1, ..., rho_ns
SuperCatalyticWallNS2DNeq2Tnondim::SuperCatalyticWallNS2DNeq2Tnondim(
    const int bdry_tag, EquationNS2DNeq2Tnondim* equation)
    : BoundaryNS2DNeq2Tnondim(bdry_tag, equation) {
  MASTER_MESSAGE("SuperCatalyticWall (tag=" + std::to_string(bdry_tag) + ")\n");
  ERROR_MESSAGE("SuperCatalyticWall is not supported in this version!!!");
}
void SuperCatalyticWallNS2DNeq2Tnondim::ComputeBdrySolution(
    const int num_points, std::vector<double>& bdry_u,
    std::vector<double>& bdry_div_u, const std::vector<double>& owner_u,
    const std::vector<double>& owner_div_u, const std::vector<double>& normal,
    const std::vector<double>& coords, const double& time) {}
void SuperCatalyticWallNS2DNeq2Tnondim::ComputeBdryFlux(
    const int num_points, std::vector<double>& flux, FACE_INPUTS,
    const std::vector<double>& coords, const double& time) {}
void SuperCatalyticWallNS2DNeq2Tnondim::ComputeBdrySolutionJacobi(
    const int num_points, double* bdry_u_jacobi,
    const std::vector<double>& owner_u, const std::vector<double>& owner_div_u,
    const std::vector<double>& normal, const std::vector<double>& coords,
    const double& time) {}
void SuperCatalyticWallNS2DNeq2Tnondim::ComputeBdryFluxJacobi(
    const int num_points, std::vector<double>& flux_jacobi,
    std::vector<double>& flux_grad_jacobi, FACE_INPUTS,
    const std::vector<double>& coords, const double& time) {}
// -------------------------------- Problem -------------------------------- //
std::shared_ptr<ProblemNS2DNeq2Tnondim> ProblemNS2DNeq2Tnondim::GetProblem(
    const std::string& name) {
  if (!name.compare("FreeStream"))
    return std::make_shared<FreeStreamNS2DNeq2Tnondim>();
  else if (!name.compare("DoubleSine"))
    return std::make_shared<DoubleSineNS2DNeq2Tnondim>();
  else if (!name.compare("Test"))
    return std::make_shared<TestNS2DNeq2Tnondim>();
  else
    ERROR_MESSAGE("Wrong problem (no-exist):" + name + "\n");
  return nullptr;
}
// Problem = FreeStream
// ProblemInput = rho_1, ..., rho_ns, u, v, T_tr, T_eev
FreeStreamNS2DNeq2Tnondim::FreeStreamNS2DNeq2Tnondim() {
  MASTER_MESSAGE("FreeStream problem\n");

  auto& config = AVOCADO_CONFIG;
  d_.resize(ns_, 0.0);
  for (int i = 0; i < ns_; i++)
    d_[i] = std::stod(config->GetConfigValue(PROBLEM_INPUT_I(i)));
  u_ = std::stod(config->GetConfigValue(PROBLEM_INPUT_I(ns_)));
  v_ = std::stod(config->GetConfigValue(PROBLEM_INPUT_I(ns_ + 1)));
  T_tr_ = std::stod(config->GetConfigValue(PROBLEM_INPUT_I(ns_ + 2)));
  T_eev_ = std::stod(config->GetConfigValue(PROBLEM_INPUT_I(ns_ + 3)));

  mixture_->SetDensity(&d_[0]);
  const double mixture_d = mixture_->GetTotalDensity();
  const double mixture_p = mixture_->GetPressure(T_tr_, T_eev_);

  std::stringstream str;
  str << "\tInput:\n";
  str << "\t\tdensity = " << mixture_d << " kg/m3\n";
  str << "\t\tx-velocity = " << u_ << " m/s\n";
  str << "\t\ty-velocity = " << v_ << " m/s\n";
  str << "\t\ttrans-rotational temperature = " << T_tr_ << " K\n";
  str << "\t\telectron-electronic-vibration temperature = " << T_eev_ << " K\n";
  str << "\t\tpressure = " << mixture_p << " Pa\n";
  MASTER_MESSAGE(str.str());

  for (int i = 0; i < ns_; i++) d_[i] /= rho_ref_;
  u_ /= v_ref_;
  v_ /= v_ref_;
  T_tr_ /= T_ref_;
  T_eev_ /= T_ref_;
}
void FreeStreamNS2DNeq2Tnondim::Problem(const int num_points,
                                        std::vector<double>& solutions,
                                        const std::vector<double>& coord,
                                        const double time) const {
  int ind = 0;
  for (int ipoint = 0; ipoint < num_points; ipoint++) {
    for (int i = 0; i < ns_; i++) solutions[ind++] = d_[i];
    solutions[ind++] = u_;
    solutions[ind++] = v_;
    solutions[ind++] = T_tr_;
    solutions[ind++] = T_eev_;
  }
}
// Problem = DoubleSine
// ProblemInput = rho_1, ..., rho_ns, u, v, T_tr, T_eev
DoubleSineNS2DNeq2Tnondim::DoubleSineNS2DNeq2Tnondim()
    : wave_number_({1.0, 1.0}) {
  MASTER_MESSAGE("DoubleSine problem\n");

  auto& config = AVOCADO_CONFIG;
  d_.resize(ns_, 0.0);
  for (int i = 0; i < ns_; i++)
    d_[i] = std::stod(config->GetConfigValue(PROBLEM_INPUT_I(i)));
  u_ = std::stod(config->GetConfigValue(PROBLEM_INPUT_I(ns_)));
  v_ = std::stod(config->GetConfigValue(PROBLEM_INPUT_I(ns_ + 1)));
  T_tr_ = std::stod(config->GetConfigValue(PROBLEM_INPUT_I(ns_ + 2)));
  T_eev_ = std::stod(config->GetConfigValue(PROBLEM_INPUT_I(ns_ + 3)));

  mixture_->SetDensity(&d_[0]);
  const double mixture_d = mixture_->GetTotalDensity();
  p_ = mixture_->GetPressure(T_tr_, T_eev_);

  std::stringstream str;
  str << "\tInput:\n";
  str << "\t\tdensity = " << mixture_d << " kg/m3\n";
  str << "\t\tx-velocity = " << u_ << " m/s\n";
  str << "\t\ty-velocity = " << v_ << " m/s\n";
  str << "\t\ttrans-rotational temperature = " << T_tr_ << " K\n";
  str << "\t\telectron-electronic-vibration temperature = " << T_eev_ << " K\n";
  str << "\t\tpressure = " << p_ << " Pa\n";
  MASTER_MESSAGE(str.str());

  for (int i = 0; i < ns_; i++) d_[i] /= rho_ref_;
  u_ /= v_ref_;
  v_ /= v_ref_;
  T_tr_ /= T_ref_;
  T_eev_ /= T_ref_;
  p_ /= p_ref_;
}
void DoubleSineNS2DNeq2Tnondim::Problem(const int num_points,
                                  std::vector<double>& solutions,
                                  const std::vector<double>& coord,
                                  const double time) const {
  int ind = 0;

  std::vector<double> values(S_);

  const auto& species = mixture_->GetSpecies();
  double R1 = species[0]->GetSpecificGasConstant();
  values[1] = 1.0;
  values[2] = 1.0;
  values[3] = 1.0;

  for (int ipoint = 0; ipoint < num_points; ipoint++) {
    const double& x = coord[ipoint * D_];
    const double& y = coord[ipoint * D_ + 1];

    values[0] = 1.0 + 0.2 * std::sin(2.0 * M_PI * (x - values[1] * time)) *
                          std::sin(2.0 * M_PI * (y - values[2] * time));
    ind = S_ * ipoint;
    solutions[ind] = values[0];
    ind += ns_;
    for (int idim = 0; idim < D_; idim++)
      solutions[ind++] = values[idim + 1];
    solutions[ind++] = v_ref_ * v_ref_ / values[0] / R1 / T_ref_;
    solutions[ind] = values[3];
  }
}
// Problem = Test
// ProblemInput = rho_1, ..., rho_ns, u, v, T_tr, T_eev
TestNS2DNeq2Tnondim::TestNS2DNeq2Tnondim() : wave_number_({1.0, 1.0}) {
  MASTER_MESSAGE("Test problem: DoubleSine\n");

  auto& config = AVOCADO_CONFIG;
  d_.resize(ns_, 0.0);
  for (int i = 0; i < ns_; i++)
    d_[i] = std::stod(config->GetConfigValue(PROBLEM_INPUT_I(i)));
  u_ = std::stod(config->GetConfigValue(PROBLEM_INPUT_I(ns_)));
  v_ = std::stod(config->GetConfigValue(PROBLEM_INPUT_I(ns_ + 1)));
  T_tr_ = std::stod(config->GetConfigValue(PROBLEM_INPUT_I(ns_ + 2)));
  T_eev_ = std::stod(config->GetConfigValue(PROBLEM_INPUT_I(ns_ + 3)));

  mixture_->SetDensity(&d_[0]);
  const double mixture_d = mixture_->GetTotalDensity();
  p_ = mixture_->GetPressure(T_tr_, T_eev_);

  std::stringstream str;
  str << "\tInput:\n";
  str << "\t\tdensity = " << mixture_d << " kg/m3\n";
  str << "\t\tx-velocity = " << u_ << " m/s\n";
  str << "\t\ty-velocity = " << v_ << " m/s\n";
  str << "\t\ttrans-rotational temperature = " << T_tr_ << " K\n";
  str << "\t\telectron-electronic-vibration temperature = " << T_eev_ << " K\n";
  str << "\t\tpressure = " << p_ << " Pa\n";
  MASTER_MESSAGE(str.str());

  for (int i = 0; i < ns_; i++) d_[i] /= rho_ref_;
  u_ /= v_ref_;
  v_ /= v_ref_;
  T_tr_ /= T_ref_;
  T_eev_ /= T_ref_;
  p_ /= p_ref_;
}
void TestNS2DNeq2Tnondim::Problem(const int num_points,
                                  std::vector<double>& solutions,
                                  const std::vector<double>& coord,
                                  const double time) const {
  int ind = 0;

  std::vector<double> values(S_);

  const auto& species = mixture_->GetSpecies();
  double R1 = species[0]->GetSpecificGasConstant();
  values[1] = 1.0;
  values[2] = 1.0;
  values[3] = 1.0;

  for (int ipoint = 0; ipoint < num_points; ipoint++) {
    const double& x = coord[ipoint * D_];
    const double& y = coord[ipoint * D_ + 1];

    values[0] = 1.0 + 0.2 * std::sin(2.0 * M_PI * (x - values[1] * time)) *
                          std::sin(2.0 * M_PI * (y - values[2] * time));
    ind = S_ * ipoint;
    solutions[ind] = values[0];
    ind += ns_;
    for (int idim = 0; idim < D_; idim++) solutions[ind++] = values[idim + 1];
    solutions[ind++] = v_ref_ * v_ref_ / values[0] / R1 / T_ref_;
    solutions[ind] = values[3];
  }
}
}  // namespace deneb