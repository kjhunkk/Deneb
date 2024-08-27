#include "deneb_equation_ns2d_primitive.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <unordered_set>

#include "avocado.h"
#include "deneb_artificial_viscosity.h"
#include "deneb_config_macro.h"
#include "deneb_data.h"
#include "deneb_timescheme.h"

#define GET_NORMAL_PD(data)       \
  ind = D_ * ipoint;              \
  const double& nx = data[ind++]; \
  const double& ny = data[ind]

#define GET_SOLUTION_PS(tag, data)     \
  ind = S_ * ipoint;                   \
  const double& d##tag = data[ind++];  \
  const double& u##tag = data[ind++]; \
  const double& v##tag = data[ind++]; \
  const double& T##tag = data[ind]

#define GET_SOLUTION_GRAD_PDS(tag, data) \
  ind = DS_ * ipoint;                    \
  const double& dx##tag = data[ind++];   \
  const double& ux##tag = data[ind++];  \
  const double& vx##tag = data[ind++];  \
  const double& Tx##tag = data[ind++];  \
  const double& dy##tag = data[ind++];   \
  const double& uy##tag = data[ind++];  \
  const double& vy##tag = data[ind++];  \
  const double& Ty##tag = data[ind]

#define COMPUTE_FLUX_BASICS(tag)                                     \
  const auto d_inv##tag = 1.0 / d##tag;                              \
  const auto du##tag = d##tag * u##tag;                              \
  const auto dv##tag = d##tag * v##tag;                              \
  const auto txx##tag = c23_ * (2.0 * ux##tag - vy##tag);            \
  const auto txy##tag = uy##tag + vx##tag;                           \
  const auto tyy##tag = c23_ * (2.0 * vy##tag - ux##tag);            \
  const auto e##tag = r_inv_ * rm1_inv_ * T##tag;                         \
  const auto dE##tag =                                               \
      d##tag * (e##tag + 0.5 * (u##tag * u##tag + v##tag * v##tag)); \
  const auto p##tag = r_inv_ * d##tag * T##tag                                 

#define COMPUTE_VOLUME_FLUX_PDS(flux, icell, ipoint)                          \
  ind = DS_ * ipoint;                                                         \
  AVcoeff =                                                                   \
      DENEB_ARTIFICIAL_VISCOSITY->GetArtificialViscosityValue(icell, ipoint); \
  flux[ind++] = d * u;                                                        \
  flux[ind++] = d * u * u + p;                                                \
  flux[ind++] = d * u * v;                                                    \
  flux[ind++] = (dE + p) * u;                                                 \
  flux[ind++] = dv;                                                           \
  flux[ind++] = dv * u;                                                       \
  flux[ind++] = dv * v + p;                                                   \
  flux[ind] = (dE + p) * v

#define COMPUTE_NUMERICAL_FLUX_PDS(flux, owner_cell, neighbor_cell, ipoint)   \
  ind = DS_ * ipoint;                                                         \
  AVcoeff_o = DENEB_ARTIFICIAL_VISCOSITY->GetArtificialViscosityValue(        \
      owner_cell, ipoint);                                                    \
  AVcoeff_n = DENEB_ARTIFICIAL_VISCOSITY->GetArtificialViscosityValue(        \
      neighbor_cell, ipoint);                                                 \
  flux[ind++] = 0.5 * (du_o + du_n - nx * diff1);                             \
  flux[ind++] = 0.5 * (du_o * u_o + p_o + du_n * u_n + p_n - nx * diff2);     \
  flux[ind++] = 0.5 * (du_o * v_o + du_n * v_n - nx * diff3);                 \
  flux[ind++] = 0.5 * ((dE_o + p_o) * u_o + (dE_n + p_n) * u_n - nx * diff4); \
  flux[ind++] = 0.5 * (dv_o + dv_n - ny * diff1);                             \
  flux[ind++] = 0.5 * (dv_o * u_o + dv_n * u_n - ny * diff2);                 \
  flux[ind++] = 0.5 * (dv_o * v_o + p_o + dv_n * v_n + p_n - ny * diff3);     \
  flux[ind] = 0.5 * ((dE_o + p_o) * v_o + (dE_n + p_n) * v_n - ny * diff4)

namespace deneb {
// ------------------------------- Constants ------------------------------- //
ConstantsNS2DPri::ConstantsNS2DPri()
    : r_(std::stod(AVOCADO_CONFIG->GetConfigValue(HEAT_CAPACITY_RATIO))),
      r_inv_(1.0 / r_),
      rm1_(r_ - 1.0),
      rm1_inv_(1.0 / rm1_) {
  auto& config = AVOCADO_CONFIG;
  Re_ = std::stod(config->GetConfigValue(REYNOLDS_NUMBER));
  Ma_ = std::stod(config->GetConfigValue(MACH_NUMBER));
  Pr_ = std::stod(config->GetConfigValue(PRANDTL_NUMBER));
  alpha_ = Ma_ / Re_;
  beta_ = r_ / Pr_ * alpha_;
  if (Re_ < 0.0) {
    alpha_ = 0.0;
    beta_ = 0.0;
  }
}
double ConstantsNS2DPri::ComputePressure(const double* sol) const {
  return r_inv_ * sol[0] * sol[S_ - 1];
}
double ConstantsNS2DPri::ComputeTotalEnergy(const double* pri) const {
  return rm1_inv_ * pri[S_ - 1] +
         0.5 * pri[0] * avocado::VecInnerProd(D_, &pri[1], &pri[1]);
}

// ------------------------------- Equation -------------------------------- //
EquationNS2DPri::EquationNS2DPri()
    : ConstantsNS2DPri(), Equation(D_, S_, true) {
  MASTER_MESSAGE(avocado::GetTitle("EquationNS2DPri"));
  MASTER_MESSAGE("Dimension = " + std::to_string(D_) + "\n");
  MASTER_MESSAGE("Number of state variables = " + std::to_string(S_) + "\n");
  MASTER_MESSAGE(
      "Source term = " + std::string(source_term_ ? "true" : "false") + "\n");
  num_species_ = 1;

  auto& config = AVOCADO_CONFIG;
  problem_ = ProblemNS2DPri::GetProblem(config->GetConfigValue(PROBLEM));
  const std::string& numflux = config->GetConfigValue(CONVECTIVE_FLUX);
  if (!numflux.compare("LLF")) {
    ASSIGN_FLUX(EquationNS2DPri, LLF);
  } else if (!numflux.compare("Roe")) {
    ASSIGN_FLUX(EquationNS2DPri, Roe);
  } else
    ERROR_MESSAGE("Wrong numerical flux (no-exist):" + numflux + "\n");
  MASTER_MESSAGE("Problem: " + config->GetConfigValue(PROBLEM) + "\n");
  MASTER_MESSAGE("Convective flux: " + numflux + "\n");
}
EquationNS2DPri::~EquationNS2DPri() {
  problem_.reset();
  boundaries_.clear();
  boundary_registry_.clear();
}

void EquationNS2DPri::RegistBoundary(const std::vector<int>& bdry_tag) {
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
          BoundaryNS2DPri::GetBoundary(bdry_type, tag, this);
  }
}
void EquationNS2DPri::BuildData(void) {
  MASTER_MESSAGE(avocado::GetTitle("EquationNS2DPri::BuildData()"));

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
  pressure_fix_values_.resize(2);

  const int& num_outer_cells = DENEB_DATA->GetNumOuterCells();
  outer_solution_.resize(
      std::max((num_outer_cells - num_cells) * S_ * num_bases, 1));
  communicate_ = std::make_shared<avocado::Communicate>(
      S_ * num_bases, DENEB_DATA->GetOuterSendCellList(),
      DENEB_DATA->GetOuterRecvCellList());

  cell_variable_names_ = {"rho", "U", "V", "T", "P"};
  face_variable_names_ = {"rho", "U", "V", "T", "P"};

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
void EquationNS2DPri::GetCellPostSolution(
    const int icell, const int num_points, const std::vector<double>& solution,
    const std::vector<double>& solution_grad,
    std::vector<double>& post_solution) {
  int ind = 0;
  for (int ipoint = 0; ipoint < num_points; ipoint++) {
    for (int istate = 0; istate < S_; istate++)
      post_solution[ind++] = solution[ipoint * S_ + istate];
    post_solution[ind++] = ComputePressure(&solution[ipoint * S_]);
  }
}
void EquationNS2DPri::GetFacePostSolution(
    const int num_points, const std::vector<double>& solution,
    const std::vector<double>& solution_grad, const std::vector<double>& normal,
    std::vector<double>& post_solution) {
  int ind = 0;
  for (int ipoint = 0; ipoint < num_points; ipoint++) {
    for (int istate = 0; istate < S_; istate++)
      post_solution[ind++] = solution[ipoint * S_ + istate];
    post_solution[ind++] = ComputePressure(&solution[ipoint * S_]);
  }
}
void EquationNS2DPri::ComputeInitialSolution(double* solution, const double t) {
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
void EquationNS2DPri::ComputeLocalTimestep(
    const double* solution, std::vector<double>& local_timestep) {
  static const int& order = DENEB_DATA->GetOrder();
  static const int& num_bases = DENEB_DATA->GetNumBases();
  static const int sb = S_ * num_bases;

  static const int& num_cells = DENEB_DATA->GetNumCells();
  static const auto& cell_volumes = DENEB_DATA->GetCellVolumes();
  static const auto& cell_proj_volumes = DENEB_DATA->GetCellProjVolumes();
  static const auto& cell_basis_value = DENEB_DATA->GetCellBasisValue();
  static const double max_visradii = std::max(4.0 / 3.0 * alpha_, beta_);
  for (int icell = 0; icell < num_cells; icell++) {
    const double& Vx = cell_proj_volumes[icell * D_];
    const double& Vy = cell_proj_volumes[icell * D_ + 1];
    const double d = solution[icell * sb + 0 * num_bases] *
                                cell_basis_value[icell][0];
    const double u =
        solution[icell * sb + 1 * num_bases] * cell_basis_value[icell][0];
    const double v =
        solution[icell * sb + 2 * num_bases] * cell_basis_value[icell][0];
    const double T =
        solution[icell * sb + 3 * num_bases] * cell_basis_value[icell][0];
    const double e = r_inv_ * rm1_inv_ * T;
    const double p = rm1_ * d * e;
    const double a = std::sqrt(r_ * p / d);

    const double AVcoeff =
        DENEB_ARTIFICIAL_VISCOSITY->GetArtificialViscosityValue(icell, 0);

    local_timestep[icell] =
        cell_volumes[icell] /
        ((std::abs(u) + a) * Vx + (std::abs(v) + a) * Vy +
         (max_visradii / d + AVcoeff) * dt_auxiliary_[icell]);
  }
  avocado::VecScale(num_cells, 1.0 / static_cast<double>(2 * order + 1),
                    &local_timestep[0]);
}
bool EquationNS2DPri::IsContact(const int& icell,
                                const std::vector<int>& neighbor_cells,
                                const double* solution,
                                const double* total_solution) const {
  static const int& num_bases = DENEB_DATA->GetNumBases();
  static const int sb = S_ * num_bases;

  static const int& num_cells = DENEB_DATA->GetNumCells();
  static const auto& volume = DENEB_DATA->GetCellVolumes();

  std::vector<double> sol(S_);
  cblas_daxpby(S_, 1.0 / std::sqrt(volume[icell]), &solution[icell * sb],
               num_bases, 0.0, &sol[0], 1);
  const double owner_p = ComputePressure(&sol[0]);

  for (auto&& cell : neighbor_cells) {
    const double* target = (cell < num_cells)
                               ? &solution[cell * sb]
                               : &total_solution[(cell - num_cells) * sb];
    cblas_daxpby(S_, 1.0 / std::sqrt(volume[cell]), target, num_bases, 0.0,
                 &sol[0], 1);
    const double neighbor_p = ComputePressure(&sol[0]);

    if (std::abs(owner_p - neighbor_p) > 1.0E-2 * owner_p) return false;
  }
  return true;
}
double EquationNS2DPri::ComputeMaxCharacteristicSpeed(
    const double* input_solution) const {
  const double d_inv = 1.0 / input_solution[0];
  const double p = ComputePressure(&input_solution[0]);
  const double a = std::sqrt(r_ * p * d_inv);
  const double V = std::sqrt(avocado::VecInnerProd(D_, &input_solution[1],
                                                   &input_solution[1])) *
                   d_inv;
  return V + a;
}
const std::vector<double>& EquationNS2DPri::ComputePressureFixValues(
    const double* input_solution) {
  pressure_fix_values_[0] = input_solution[0];
  pressure_fix_values_[1] = ComputePressure(&input_solution[0]);
  return pressure_fix_values_;
}
void EquationNS2DPri::ComputeRHS(const double* solution, double* rhs,
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
                   neighbor_solution_grad, face_normals[iface]);

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

    avocado::Kernel1::f59(&bdry_owner_coefficients[ibdry][0],
                          &solution_difference[0], &local_auxiliary[0], D_,
                          num_bases, num_points, S_, 1.0, 0.0);
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
                   neighbor_solution_grad, face_normals[iface]);

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
  }
}
void EquationNS2DPri::ComputeSystemMatrix(const double* solution, Mat& sysmat,
                                          const double t) {
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

  communicate_->CommunicateBegin(solution);

  MatZeroEntries(sysmat);
  return; // for debug
  memset(&auxiliary_solution_[0], 0, num_cells * dsb * sizeof(double));

  // inner face sweep
  static const auto& mat_index = DENEB_DATA->GetMatIndex();
  static const int& num_inner_faces = DENEB_DATA->GetNumInnerFaces();
  static const auto& num_face_points = DENEB_DATA->GetNumFacePoints();
  static const auto& face_owner_cell = DENEB_DATA->GetFaceOwnerCell();
  static const auto& face_neighbor_cell = DENEB_DATA->GetFaceNeighborCell();
  static const auto& face_normals = DENEB_DATA->GetFaceNormals();
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
                         neighbor_solution_grad, face_normals[iface]);

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

    avocado::Kernel1::f59(&bdry_owner_coefficients[ibdry][0],
                          &solution_difference[0], &local_auxiliary[0], D_,
                          num_bases, num_points, S_, 1.0, 0.0);
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
                         neighbor_solution_grad, face_normals[iface]);

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
    MatSetValuesBlocked(sysmat, 1, &mat_index[icell], 1, &mat_index[icell],
                        &block[0], ADD_VALUES);
  }

  MatAssemblyBegin(sysmat, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(sysmat, MAT_FINAL_ASSEMBLY);
}
void EquationNS2DPri::SystemMatrixShift(const double* solution, Mat& sysmat,
                                          const double dt, const double t) {
  static const auto& num_cell_points = DENEB_DATA->GetNumCellPoints();
  static const auto& cell_basis_value = DENEB_DATA->GetCellBasisValue();
  static const int& num_cells = DENEB_DATA->GetNumCells();
  static const int& num_bases = DENEB_DATA->GetNumBases();
  static const auto& mat_index = DENEB_DATA->GetMatIndex();
  static const auto& cell_mass_coefficients =
    DENEB_DATA->GetCellMassCoefficients();
  static const int sb = num_states_ * num_bases;
  static std::vector<double> block(sb * sb);
  static std::vector<double> owner_solution(S_ * max_num_points_);
  static std::vector<double> solution_jacobi(SS_ * max_num_points_);

  const double dt_factor = 1.0 / dt;
  MatShift(sysmat, 1.0 / dt);  // for debug
  for (int icell = 0; icell < num_cells; icell++) {
    const int& num_points = num_cell_points[icell];
    memset(&block[0], 0, sb * sb * sizeof(double));

    avocado::Kernel0::f4(&solution[icell * sb], &cell_basis_value[icell][0],
                         &owner_solution[0], S_, num_bases, num_points, 1.0,
                         0.0);
    ComputeSolutionJacobi(num_points, solution_jacobi, owner_solution);

    // Need optimization
    for (int ipoint = 0; ipoint < num_points; ipoint++)
      for (int istate = 0; istate < S_; istate++)
        for (int ibasis = 0; ibasis < num_bases; ibasis++)
          for (int jstate = 0; jstate < S_; jstate++)
            for (int jbasis = 0; jbasis < num_bases; jbasis++) {
              const int irow = ibasis + num_bases * istate;
              const int icolumn = jbasis + num_bases * jstate;

              block[irow * sb + icolumn] +=
                cell_mass_coefficients[icell][ipoint * num_bases * num_bases +
                ibasis * num_bases + jbasis] *
                solution_jacobi[ipoint * SS_ + istate * S_ + jstate];
            }

    avocado::VecScale(sb * sb, dt_factor, &block[0]);

    MatSetValuesBlocked(sysmat, 1, &mat_index[icell], 1, &mat_index[icell],
                        &block[0], ADD_VALUES);
  }
  MatAssemblyBegin(sysmat, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(sysmat, MAT_FINAL_ASSEMBLY);
}
void EquationNS2DPri::SystemMatrixShift(const double* solution, Mat& sysmat,
                                          const std::vector<double>& local_dt,
                                          const double t) {
  static const auto& num_cell_points = DENEB_DATA->GetNumCellPoints();
  static const auto& cell_basis_value = DENEB_DATA->GetCellBasisValue();
  static const int& num_cells = DENEB_DATA->GetNumCells();
  static const int& num_bases = DENEB_DATA->GetNumBases();
  static const auto& mat_index = DENEB_DATA->GetMatIndex();
  static const auto& cell_mass_coefficients =
    DENEB_DATA->GetCellMassCoefficients();
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
    for (int ipoint = 0; ipoint < num_points; ipoint++)
      for (int istate = 0; istate < S_; istate++)
        for (int ibasis = 0; ibasis < num_bases; ibasis++)
          for (int jstate = 0; jstate < S_; jstate++)
            for (int jbasis = 0; jbasis < num_bases; jbasis++) {
              const int irow = ibasis + num_bases * istate;
              const int icolumn = jbasis + num_bases * jstate;

              block[irow * sb + icolumn] +=
                cell_mass_coefficients[icell][ipoint * num_bases * num_bases +
                ibasis * num_bases + jbasis] *
                solution_jacobi[ipoint * SS_ + istate * S_ + jstate];
            }

    avocado::VecScale(sb * sb, dt_factor, &block[0]);

    MatSetValuesBlocked(sysmat, 1, &mat_index[icell], 1, &mat_index[icell],
                        &block[0], ADD_VALUES);
  }
  MatAssemblyBegin(sysmat, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(sysmat, MAT_FINAL_ASSEMBLY);
}
void EquationNS2DPri::ComputeComFlux(const int num_points,
                                     std::vector<double>& flux, const int icell,
                                     const std::vector<double>& owner_u,
                                     const std::vector<double>& owner_div_u) {
  int ind = 0;
  double AVcoeff = 0.0;
  for (int ipoint = 0; ipoint < num_points; ipoint++) {
    GET_SOLUTION_PS(, owner_u);

    GET_SOLUTION_GRAD_PDS(, owner_div_u);

    COMPUTE_FLUX_BASICS();

    COMPUTE_VOLUME_FLUX_PDS(flux, icell, ipoint);
  }
}
void EquationNS2DPri::ComputeComFluxJacobi(
    const int num_points, std::vector<double>& flux_jacobi,
    std::vector<double>& flux_grad_jacobi, const int icell,
    const std::vector<double>& owner_u,
    const std::vector<double>& owner_div_u) {
  // flux_jacobi(ds1s2) = F(ds1) over U(s2)
  // flux_grad_jacobi(d1s1 s2d2) = F(d1s1) over gradU(d2s2)
  int ind = 0;
  double AVcoeff = 0.0;
  static std::vector<Dual<S_>> flux1(DS_);
  static std::vector<Dual<DS_>> flux2(DS_);
  for (int ipoint = 0; ipoint < num_points; ipoint++) {
    {
      // ps
      ind = S_ * ipoint;
      const Dual<S_> d(owner_u[ind++], 0);
      const Dual<S_> u(owner_u[ind++], 1);
      const Dual<S_> v(owner_u[ind++], 2);
      const Dual<S_> T(owner_u[ind], 3);

      GET_SOLUTION_GRAD_PDS(, owner_div_u);

      COMPUTE_FLUX_BASICS();

      COMPUTE_VOLUME_FLUX_PDS(flux1, icell, 0);

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
      const Dual<DS_> dx(owner_div_u[ind++], 0);
      const Dual<DS_> ux(owner_div_u[ind++], 2);
      const Dual<DS_> vx(owner_div_u[ind++], 4);
      const Dual<DS_> Tx(owner_div_u[ind++], 6);
      const Dual<DS_> dy(owner_div_u[ind++], 1);
      const Dual<DS_> uy(owner_div_u[ind++], 3);
      const Dual<DS_> vy(owner_div_u[ind++], 5);
      const Dual<DS_> Ty(owner_div_u[ind], 7);

      COMPUTE_FLUX_BASICS();

      COMPUTE_VOLUME_FLUX_PDS(flux2, icell, 0);

      // pdssd
      ind = DDSS_ * ipoint;
      for (int ds = 0; ds < DS_; ds++)
        for (int sd = 0; sd < DS_; sd++)
          flux_grad_jacobi[ind++] = flux2[ds].df[sd];
    }
  }
}
void EquationNS2DPri::ComputeSolutionJacobi(
  const int num_points, std::vector<double>& jacobi, const std::vector<double>& owner_u) {
  std::vector<aDual> values(S_, aDual(S_));
  for (int ipoint = 0; ipoint < num_points; ipoint++) {
    const double* sol = &owner_u[ipoint * S_];

    aDual d(S_, sol[0], 0);
    aDual u(S_, sol[1], 1);
    aDual v(S_, sol[2], 2);
    aDual T(S_, sol[3], 3);

    values[0] = d;
    values[1] = d * u;
    values[2] = d * v;
    values[3] = d * (r_inv_ * rm1_inv_ * T + 0.5 * (u * u + v * v));

    int ind = SS_ * ipoint;
    for (int i = 0; i < S_; i++)
      for (int j = 0; j < S_; j++) jacobi[ind++] = values[i].df[j];
  }
}
void EquationNS2DPri::ComputeNumFluxLLF(const int num_points,
                                        std::vector<double>& flux,
                                        FACE_INPUTS) {
  int ind = 0;
  double AVcoeff_o = 0.0;
  double AVcoeff_n = 0.0;
  for (int ipoint = 0; ipoint < num_points; ipoint++) {
    GET_NORMAL_PD(normal);

    GET_SOLUTION_PS(_o, owner_u);
    GET_SOLUTION_PS(_n, neighbor_u);

    GET_SOLUTION_GRAD_PDS(_o, owner_div_u);
    GET_SOLUTION_GRAD_PDS(_n, neighbor_div_u);

    COMPUTE_FLUX_BASICS(_o);
    const double V_o = u_o * nx + v_o * ny;
    const double a_o = std::sqrt(r_ * p_o * d_inv_o);

    COMPUTE_FLUX_BASICS(_n);
    const double V_n = u_n * nx + v_n * ny;
    const double a_n = std::sqrt(r_ * p_n * d_inv_n);
    const double r_max = std::max(std::abs(V_o) + a_o, std::abs(V_n) + a_n);

    const auto diff1 = r_max * (d_n - d_o);
    const auto diff2 = r_max * (du_n - du_o);
    const auto diff3 = r_max * (dv_n - dv_o);
    const auto diff4 = r_max * (dE_n - dE_o);

    COMPUTE_NUMERICAL_FLUX_PDS(flux, owner_cell, neighbor_cell, ipoint);
  }
}
void EquationNS2DPri::ComputeNumFluxJacobiLLF(const int num_points,
                                              FACE_JACOBI_OUTPUTS,
                                              FACE_INPUTS) {
  // flux_jacobi(ds1s2) = F(ds1) over U(s2)
  // flux_grad_jacobi(d1s1 s2d2) = F(d1s1) over gradU(d2s2)
  int ind = 0;
  double AVcoeff_o = 0.0;
  double AVcoeff_n = 0.0;
  static std::vector<Dual<S_>> flux1(DS_);
  static std::vector<Dual<DS_>> flux2(DS_);
  for (int ipoint = 0; ipoint < num_points; ipoint++) {
    GET_NORMAL_PD(normal);
    {
      GET_SOLUTION_PS(_n, neighbor_u);

      GET_SOLUTION_GRAD_PDS(_n, neighbor_div_u);

      COMPUTE_FLUX_BASICS(_n);
      const double V_n = u_n * nx + v_n * ny;
      const double a_n = std::sqrt(r_ * p_n * d_inv_n);
      {
        // ps
        ind = S_ * ipoint;
        const Dual<S_> d_o(owner_u[ind++], 0);
        const Dual<S_> u_o(owner_u[ind++], 1);
        const Dual<S_> v_o(owner_u[ind++], 2);
        const Dual<S_> T_o(owner_u[ind], 3);

        GET_SOLUTION_GRAD_PDS(_o, owner_div_u);

        COMPUTE_FLUX_BASICS(_o);
        const auto V_o = u_o * nx + v_o * ny;
        const auto a_o = std::sqrt(r_ * p_o * d_inv_o);
        const auto r_max = std::max(std::abs(V_o) + a_o, std::abs(V_n) + a_n);

        const auto diff1 = r_max * (d_n - d_o);
        const auto diff2 = r_max * (du_n - du_o);
        const auto diff3 = r_max * (dv_n - dv_o);
        const auto diff4 = r_max * (dE_n - dE_o);

        COMPUTE_NUMERICAL_FLUX_PDS(flux1, owner_cell, neighbor_cell, 0);

        // pdss
        ind = DSS_ * ipoint;
        for (int ds = 0; ds < DS_; ds++)
          for (int istate = 0; istate < S_; istate++)
            flux_owner_jacobi[ind++] = flux1[ds].df[istate];
      }
      {
        GET_SOLUTION_PS(_o, owner_u);

        // pds over psd
        ind = DS_ * ipoint;
        const Dual<DS_> dx_o(owner_div_u[ind++], 0);
        const Dual<DS_> ux_o(owner_div_u[ind++], 2);
        const Dual<DS_> vx_o(owner_div_u[ind++], 4);
        const Dual<DS_> Tx_o(owner_div_u[ind++], 6);
        const Dual<DS_> dy_o(owner_div_u[ind++], 1);
        const Dual<DS_> uy_o(owner_div_u[ind++], 3);
        const Dual<DS_> vy_o(owner_div_u[ind++], 5);
        const Dual<DS_> Ty_o(owner_div_u[ind], 7);

        COMPUTE_FLUX_BASICS(_o);
        const auto V_o = u_o * nx + v_o * ny;
        const auto a_o = std::sqrt(r_ * p_o * d_inv_o);
        const auto r_max = std::max(std::abs(V_o) + a_o, std::abs(V_n) + a_n);

        const auto diff1 = r_max * (d_n - d_o);
        const auto diff2 = r_max * (du_n - du_o);
        const auto diff3 = r_max * (dv_n - dv_o);
        const auto diff4 = r_max * (dE_n - dE_o);

        COMPUTE_NUMERICAL_FLUX_PDS(flux2, owner_cell, neighbor_cell, 0);

        // pdssd
        ind = DDSS_ * ipoint;
        for (int ds = 0; ds < DS_; ds++)
          for (int sd = 0; sd < DS_; sd++)
            flux_owner_grad_jacobi[ind++] = flux2[ds].df[sd];
      }
    }
    {
      GET_SOLUTION_PS(_o, owner_u);

      GET_SOLUTION_GRAD_PDS(_o, owner_div_u);

      COMPUTE_FLUX_BASICS(_o);
      const double V_o = u_o * nx + v_o * ny;
      const double a_o = std::sqrt(r_ * p_o * d_inv_o);
      {
        // ps
        ind = S_ * ipoint;
        const Dual<S_> d_n(neighbor_u[ind++], 0);
        const Dual<S_> u_n(neighbor_u[ind++], 1);
        const Dual<S_> v_n(neighbor_u[ind++], 2);
        const Dual<S_> T_n(neighbor_u[ind], 3);

        GET_SOLUTION_GRAD_PDS(_n, neighbor_div_u);

        COMPUTE_FLUX_BASICS(_n);
        const auto V_n = u_n * nx + v_n * ny;
        const auto a_n = std::sqrt(r_ * p_n * d_inv_n);
        const auto r_max = std::max(std::abs(V_o) + a_o, std::abs(V_n) + a_n);

        const auto diff1 = r_max * (d_n - d_o);
        const auto diff2 = r_max * (du_n - du_o);
        const auto diff3 = r_max * (dv_n - dv_o);
        const auto diff4 = r_max * (dE_n - dE_o);

        COMPUTE_NUMERICAL_FLUX_PDS(flux1, owner_cell, neighbor_cell, 0);

        // pdss
        ind = DSS_ * ipoint;
        for (int ds = 0; ds < DS_; ds++)
          for (int istate = 0; istate < S_; istate++)
            flux_neighbor_jacobi[ind++] = flux1[ds].df[istate];
      }
      {
        GET_SOLUTION_PS(_n, neighbor_u);

        // pds over psd
        ind = DS_ * ipoint;
        const Dual<DS_> dx_n(neighbor_div_u[ind++], 0);
        const Dual<DS_> ux_n(neighbor_div_u[ind++], 2);
        const Dual<DS_> vx_n(neighbor_div_u[ind++], 4);
        const Dual<DS_> Tx_n(neighbor_div_u[ind++], 6);
        const Dual<DS_> dy_n(neighbor_div_u[ind++], 1);
        const Dual<DS_> uy_n(neighbor_div_u[ind++], 3);
        const Dual<DS_> vy_n(neighbor_div_u[ind++], 5);
        const Dual<DS_> Ty_n(neighbor_div_u[ind], 7);

        COMPUTE_FLUX_BASICS(_n);
        const auto V_n = u_n * nx + v_n * ny;
        const auto a_n = std::sqrt(r_ * p_n * d_inv_n);
        const auto r_max = std::max(std::abs(V_o) + a_o, std::abs(V_n) + a_n);

        const auto diff1 = r_max * (d_n - d_o);
        const auto diff2 = r_max * (du_n - du_o);
        const auto diff3 = r_max * (dv_n - dv_o);
        const auto diff4 = r_max * (dE_n - dE_o);

        COMPUTE_NUMERICAL_FLUX_PDS(flux2, owner_cell, neighbor_cell, 0);

        // pdssd
        ind = DDSS_ * ipoint;
        for (int ds = 0; ds < DS_; ds++)
          for (int sd = 0; sd < DS_; sd++)
            flux_neighbor_grad_jacobi[ind++] = flux2[ds].df[sd];
      }
    }
  }
}
void EquationNS2DPri::ComputeNumFluxRoe(const int num_points,
                                        std::vector<double>& flux,
                                        FACE_INPUTS) {
  int ind = 0;
  double AVcoeff_o = 0.0;
  double AVcoeff_n = 0.0;
  for (int ipoint = 0; ipoint < num_points; ipoint++) {
    GET_NORMAL_PD(normal);

    GET_SOLUTION_PS(_o, owner_u);
    GET_SOLUTION_PS(_n, neighbor_u);

    GET_SOLUTION_GRAD_PDS(_o, owner_div_u);
    GET_SOLUTION_GRAD_PDS(_n, neighbor_div_u);

    COMPUTE_FLUX_BASICS(_o);
    const double V_o = u_o * nx + v_o * ny;

    COMPUTE_FLUX_BASICS(_n);
    const double V_n = u_n * nx + v_n * ny;

    const double dsqrt_o = std::sqrt(d_o);
    const double dsqrt_n = std::sqrt(d_n);
    const double dsqrt_inv_o = 1.0 / dsqrt_o;
    const double dsqrt_inv_n = 1.0 / dsqrt_n;

    const double Roe_d = dsqrt_o * dsqrt_n;
    const double Roe_u =
        (du_o * dsqrt_inv_o + du_n * dsqrt_inv_n) / (dsqrt_o + dsqrt_n);
    const double Roe_v =
        (dv_o * dsqrt_inv_o + dv_n * dsqrt_inv_n) / (dsqrt_o + dsqrt_n);
    const double Roe_h =
        ((dE_o + p_o) * dsqrt_inv_o + (dE_n + p_n) * dsqrt_inv_n) /
        (dsqrt_o + dsqrt_n);
    const double Roe_a =
        std::sqrt(rm1_ * (Roe_h - 0.5 * (Roe_u * Roe_u + Roe_v * Roe_v)));
    const double Roe_V = (V_o * dsqrt_o + V_n * dsqrt_n) / (dsqrt_o + dsqrt_n);

    const double inc_d = d_n - d_o;
    const double inc_u = u_n - u_o;
    const double inc_v = v_n - v_o;
    const double inc_p = p_n - p_o;
    const double inc_V = V_n - V_o;

    const double temp1 = std::abs(Roe_V) * (inc_d - inc_p / (Roe_a * Roe_a));
    const double temp2 = std::abs(Roe_V) * Roe_d;
    const double temp3 = std::abs(Roe_V + Roe_a) *
                         (inc_p + Roe_d * Roe_a * inc_V) * 0.5 /
                         (Roe_a * Roe_a);
    const double temp4 = std::abs(Roe_V - Roe_a) *
                         (inc_p - Roe_d * Roe_a * inc_V) * 0.5 /
                         (Roe_a * Roe_a);

    const auto diff1 = temp1 + temp3 + temp4;
    const auto diff2 = (temp1 + temp3 + temp4) * Roe_u +
                       (temp3 - temp4) * nx * Roe_a +
                       temp2 * (inc_u - nx * inc_V);
    const auto diff3 = (temp1 + temp3 + temp4) * Roe_v +
                       (temp3 - temp4) * ny * Roe_a +
                       temp2 * (inc_v - ny * inc_V);
    const auto diff4 = 0.5 * temp1 * (Roe_u * Roe_u + Roe_v * Roe_v) +
                       temp2 * (Roe_u * inc_u + Roe_v * inc_v - Roe_V * inc_V) +
                       Roe_h * (temp3 + temp4) +
                       Roe_a * Roe_V * (temp3 - temp4);

    COMPUTE_NUMERICAL_FLUX_PDS(flux, owner_cell, neighbor_cell, ipoint);
  }
}
void EquationNS2DPri::ComputeNumFluxJacobiRoe(const int num_points,
                                              FACE_JACOBI_OUTPUTS,
                                              FACE_INPUTS) {
  // flux_jacobi(ds1s2) = F(ds1) over U(s2)
  // flux_grad_jacobi(d1s1 s2d2) = F(d1s1) over gradU(d2s2)
  int ind = 0;
  double AVcoeff_o = 0.0;
  double AVcoeff_n = 0.0;
  static std::vector<Dual<S_>> flux1(DS_);
  static std::vector<Dual<DS_>> flux2(DS_);
  for (int ipoint = 0; ipoint < num_points; ipoint++) {
    GET_NORMAL_PD(normal);
    {
      GET_SOLUTION_PS(_n, neighbor_u);

      GET_SOLUTION_GRAD_PDS(_n, neighbor_div_u);

      COMPUTE_FLUX_BASICS(_n);
      const double V_n = u_n * nx + v_n * ny;
      const double dsqrt_n = std::sqrt(d_n);
      const double dsqrt_inv_n = 1.0 / dsqrt_n;
      {
        // ps
        ind = S_ * ipoint;
        const Dual<S_> d_o(owner_u[ind++], 0);
        const Dual<S_> u_o(owner_u[ind++], 1);
        const Dual<S_> v_o(owner_u[ind++], 2);
        const Dual<S_> T_o(owner_u[ind], 3);

        GET_SOLUTION_GRAD_PDS(_o, owner_div_u);

        COMPUTE_FLUX_BASICS(_o);
        const auto V_o = u_o * nx + v_o * ny;
        const auto dsqrt_o = std::sqrt(d_o);
        const auto dsqrt_inv_o = 1.0 / dsqrt_o;

        const auto Roe_d = dsqrt_o * dsqrt_n;
        const auto Roe_u =
            (du_o * dsqrt_inv_o + du_n * dsqrt_inv_n) / (dsqrt_o + dsqrt_n);
        const auto Roe_v =
            (dv_o * dsqrt_inv_o + dv_n * dsqrt_inv_n) / (dsqrt_o + dsqrt_n);
        const auto Roe_h =
            ((dE_o + p_o) * dsqrt_inv_o + (dE_n + p_n) * dsqrt_inv_n) /
            (dsqrt_o + dsqrt_n);
        const auto Roe_a =
            std::sqrt(rm1_ * (Roe_h - 0.5 * (Roe_u * Roe_u + Roe_v * Roe_v)));
        const auto Roe_V =
            (V_o * dsqrt_o + V_n * dsqrt_n) / (dsqrt_o + dsqrt_n);

        const auto inc_d = d_n - d_o;
        const auto inc_u = u_n - u_o;
        const auto inc_v = v_n - v_o;
        const auto inc_p = p_n - p_o;
        const auto inc_V = V_n - V_o;

        const auto temp1 = std::abs(Roe_V) * (inc_d - inc_p / (Roe_a * Roe_a));
        const auto temp2 = std::abs(Roe_V) * Roe_d;
        const auto temp3 = std::abs(Roe_V + Roe_a) *
                           (inc_p + Roe_d * Roe_a * inc_V) * 0.5 /
                           (Roe_a * Roe_a);
        const auto temp4 = std::abs(Roe_V - Roe_a) *
                           (inc_p - Roe_d * Roe_a * inc_V) * 0.5 /
                           (Roe_a * Roe_a);

        const auto diff1 = temp1 + temp3 + temp4;
        const auto diff2 = (temp1 + temp3 + temp4) * Roe_u +
                           (temp3 - temp4) * nx * Roe_a +
                           temp2 * (inc_u - nx * inc_V);
        const auto diff3 = (temp1 + temp3 + temp4) * Roe_v +
                           (temp3 - temp4) * ny * Roe_a +
                           temp2 * (inc_v - ny * inc_V);
        const auto diff4 =
            0.5 * temp1 * (Roe_u * Roe_u + Roe_v * Roe_v) +
            temp2 * (Roe_u * inc_u + Roe_v * inc_v - Roe_V * inc_V) +
            Roe_h * (temp3 + temp4) + Roe_a * Roe_V * (temp3 - temp4);

        COMPUTE_NUMERICAL_FLUX_PDS(flux1, owner_cell, neighbor_cell, 0);

        // pdss
        ind = DSS_ * ipoint;
        for (int ds = 0; ds < DS_; ds++)
          for (int istate = 0; istate < S_; istate++)
            flux_owner_jacobi[ind++] = flux1[ds].df[istate];
      }
      {
        GET_SOLUTION_PS(_o, owner_u);

        // pds over psd
        ind = DS_ * ipoint;
        const Dual<DS_> dx_o(owner_div_u[ind++], 0);
        const Dual<DS_> ux_o(owner_div_u[ind++], 2);
        const Dual<DS_> vx_o(owner_div_u[ind++], 4);
        const Dual<DS_> Tx_o(owner_div_u[ind++], 6);
        const Dual<DS_> dy_o(owner_div_u[ind++], 1);
        const Dual<DS_> uy_o(owner_div_u[ind++], 3);
        const Dual<DS_> vy_o(owner_div_u[ind++], 5);
        const Dual<DS_> Ty_o(owner_div_u[ind], 7);

        COMPUTE_FLUX_BASICS(_o);
        const auto V_o = u_o * nx + v_o * ny;
        const auto dsqrt_o = std::sqrt(d_o);
        const auto dsqrt_inv_o = 1.0 / dsqrt_o;

        const auto Roe_d = dsqrt_o * dsqrt_n;
        const auto Roe_u =
            (du_o * dsqrt_inv_o + du_n * dsqrt_inv_n) / (dsqrt_o + dsqrt_n);
        const auto Roe_v =
            (dv_o * dsqrt_inv_o + dv_n * dsqrt_inv_n) / (dsqrt_o + dsqrt_n);
        const auto Roe_h =
            ((dE_o + p_o) * dsqrt_inv_o + (dE_n + p_n) * dsqrt_inv_n) /
            (dsqrt_o + dsqrt_n);
        const auto Roe_a =
            std::sqrt(rm1_ * (Roe_h - 0.5 * (Roe_u * Roe_u + Roe_v * Roe_v)));
        const auto Roe_V =
            (V_o * dsqrt_o + V_n * dsqrt_n) / (dsqrt_o + dsqrt_n);

        const auto inc_d = d_n - d_o;
        const auto inc_u = u_n - u_o;
        const auto inc_v = v_n - v_o;
        const auto inc_p = p_n - p_o;
        const auto inc_V = V_n - V_o;

        const auto temp1 = std::abs(Roe_V) * (inc_d - inc_p / (Roe_a * Roe_a));
        const auto temp2 = std::abs(Roe_V) * Roe_d;
        const auto temp3 = std::abs(Roe_V + Roe_a) *
                           (inc_p + Roe_d * Roe_a * inc_V) * 0.5 /
                           (Roe_a * Roe_a);
        const auto temp4 = std::abs(Roe_V - Roe_a) *
                           (inc_p - Roe_d * Roe_a * inc_V) * 0.5 /
                           (Roe_a * Roe_a);

        const auto diff1 = temp1 + temp3 + temp4;
        const auto diff2 = (temp1 + temp3 + temp4) * Roe_u +
                           (temp3 - temp4) * nx * Roe_a +
                           temp2 * (inc_u - nx * inc_V);
        const auto diff3 = (temp1 + temp3 + temp4) * Roe_v +
                           (temp3 - temp4) * ny * Roe_a +
                           temp2 * (inc_v - ny * inc_V);
        const auto diff4 =
            0.5 * temp1 * (Roe_u * Roe_u + Roe_v * Roe_v) +
            temp2 * (Roe_u * inc_u + Roe_v * inc_v - Roe_V * inc_V) +
            Roe_h * (temp3 + temp4) + Roe_a * Roe_V * (temp3 - temp4);

        COMPUTE_NUMERICAL_FLUX_PDS(flux2, owner_cell, neighbor_cell, 0);

        // pdssd
        ind = DDSS_ * ipoint;
        for (int ds = 0; ds < DS_; ds++)
          for (int sd = 0; sd < DS_; sd++)
            flux_owner_grad_jacobi[ind++] = flux2[ds].df[sd];
      }
    }
    {
      GET_SOLUTION_PS(_o, owner_u);

      GET_SOLUTION_GRAD_PDS(_o, owner_div_u);

      COMPUTE_FLUX_BASICS(_o);
      const double V_o = u_o * nx + v_o * ny;
      const double dsqrt_o = std::sqrt(d_o);
      const double dsqrt_inv_o = 1.0 / dsqrt_o;
      {
        // ps
        ind = S_ * ipoint;
        const Dual<S_> d_n(neighbor_u[ind++], 0);
        const Dual<S_> u_n(neighbor_u[ind++], 1);
        const Dual<S_> v_n(neighbor_u[ind++], 2);
        const Dual<S_> T_n(neighbor_u[ind], 3);

        GET_SOLUTION_GRAD_PDS(_n, neighbor_div_u);

        COMPUTE_FLUX_BASICS(_n);
        const auto V_n = u_n * nx + v_n * ny;
        const auto dsqrt_n = std::sqrt(d_n);
        const auto dsqrt_inv_n = 1.0 / dsqrt_n;

        const auto Roe_d = dsqrt_o * dsqrt_n;
        const auto Roe_u =
            (du_o * dsqrt_inv_o + du_n * dsqrt_inv_n) / (dsqrt_o + dsqrt_n);
        const auto Roe_v =
            (dv_o * dsqrt_inv_o + dv_n * dsqrt_inv_n) / (dsqrt_o + dsqrt_n);
        const auto Roe_h =
            ((dE_o + p_o) * dsqrt_inv_o + (dE_n + p_n) * dsqrt_inv_n) /
            (dsqrt_o + dsqrt_n);
        const auto Roe_a =
            std::sqrt(rm1_ * (Roe_h - 0.5 * (Roe_u * Roe_u + Roe_v * Roe_v)));
        const auto Roe_V =
            (V_o * dsqrt_o + V_n * dsqrt_n) / (dsqrt_o + dsqrt_n);

        const auto inc_d = d_n - d_o;
        const auto inc_u = u_n - u_o;
        const auto inc_v = v_n - v_o;
        const auto inc_p = p_n - p_o;
        const auto inc_V = V_n - V_o;

        const auto temp1 = std::abs(Roe_V) * (inc_d - inc_p / (Roe_a * Roe_a));
        const auto temp2 = std::abs(Roe_V) * Roe_d;
        const auto temp3 = std::abs(Roe_V + Roe_a) *
                           (inc_p + Roe_d * Roe_a * inc_V) * 0.5 /
                           (Roe_a * Roe_a);
        const auto temp4 = std::abs(Roe_V - Roe_a) *
                           (inc_p - Roe_d * Roe_a * inc_V) * 0.5 /
                           (Roe_a * Roe_a);

        const auto diff1 = temp1 + temp3 + temp4;
        const auto diff2 = (temp1 + temp3 + temp4) * Roe_u +
                           (temp3 - temp4) * nx * Roe_a +
                           temp2 * (inc_u - nx * inc_V);
        const auto diff3 = (temp1 + temp3 + temp4) * Roe_v +
                           (temp3 - temp4) * ny * Roe_a +
                           temp2 * (inc_v - ny * inc_V);
        const auto diff4 =
            0.5 * temp1 * (Roe_u * Roe_u + Roe_v * Roe_v) +
            temp2 * (Roe_u * inc_u + Roe_v * inc_v - Roe_V * inc_V) +
            Roe_h * (temp3 + temp4) + Roe_a * Roe_V * (temp3 - temp4);

        COMPUTE_NUMERICAL_FLUX_PDS(flux1, owner_cell, neighbor_cell, 0);

        // pdss
        ind = DSS_ * ipoint;
        for (int ds = 0; ds < DS_; ds++)
          for (int istate = 0; istate < S_; istate++)
            flux_neighbor_jacobi[ind++] = flux1[ds].df[istate];
      }
      {
        GET_SOLUTION_PS(_n, neighbor_u);

        // pds over psd
        ind = DS_ * ipoint;
        const Dual<DS_> dx_n(neighbor_div_u[ind++], 0);
        const Dual<DS_> ux_n(neighbor_div_u[ind++], 2);
        const Dual<DS_> vx_n(neighbor_div_u[ind++], 4);
        const Dual<DS_> Tx_n(neighbor_div_u[ind++], 6);
        const Dual<DS_> dy_n(neighbor_div_u[ind++], 1);
        const Dual<DS_> uy_n(neighbor_div_u[ind++], 3);
        const Dual<DS_> vy_n(neighbor_div_u[ind++], 5);
        const Dual<DS_> Ty_n(neighbor_div_u[ind], 7);

        COMPUTE_FLUX_BASICS(_n);
        const auto V_n = u_n * nx + v_n * ny;
        const auto dsqrt_n = std::sqrt(d_n);
        const auto dsqrt_inv_n = 1.0 / dsqrt_n;

        const auto Roe_d = dsqrt_o * dsqrt_n;
        const auto Roe_u =
            (du_o * dsqrt_inv_o + du_n * dsqrt_inv_n) / (dsqrt_o + dsqrt_n);
        const auto Roe_v =
            (dv_o * dsqrt_inv_o + dv_n * dsqrt_inv_n) / (dsqrt_o + dsqrt_n);
        const auto Roe_h =
            ((dE_o + p_o) * dsqrt_inv_o + (dE_n + p_n) * dsqrt_inv_n) /
            (dsqrt_o + dsqrt_n);
        const auto Roe_a =
            std::sqrt(rm1_ * (Roe_h - 0.5 * (Roe_u * Roe_u + Roe_v * Roe_v)));
        const auto Roe_V =
            (V_o * dsqrt_o + V_n * dsqrt_n) / (dsqrt_o + dsqrt_n);

        const auto inc_d = d_n - d_o;
        const auto inc_u = u_n - u_o;
        const auto inc_v = v_n - v_o;
        const auto inc_p = p_n - p_o;
        const auto inc_V = V_n - V_o;

        const auto temp1 = std::abs(Roe_V) * (inc_d - inc_p / (Roe_a * Roe_a));
        const auto temp2 = std::abs(Roe_V) * Roe_d;
        const auto temp3 = std::abs(Roe_V + Roe_a) *
                           (inc_p + Roe_d * Roe_a * inc_V) * 0.5 /
                           (Roe_a * Roe_a);
        const auto temp4 = std::abs(Roe_V - Roe_a) *
                           (inc_p - Roe_d * Roe_a * inc_V) * 0.5 /
                           (Roe_a * Roe_a);

        const auto diff1 = temp1 + temp3 + temp4;
        const auto diff2 = (temp1 + temp3 + temp4) * Roe_u +
                           (temp3 - temp4) * nx * Roe_a +
                           temp2 * (inc_u - nx * inc_V);
        const auto diff3 = (temp1 + temp3 + temp4) * Roe_v +
                           (temp3 - temp4) * ny * Roe_a +
                           temp2 * (inc_v - ny * inc_V);
        const auto diff4 =
            0.5 * temp1 * (Roe_u * Roe_u + Roe_v * Roe_v) +
            temp2 * (Roe_u * inc_u + Roe_v * inc_v - Roe_V * inc_V) +
            Roe_h * (temp3 + temp4) + Roe_a * Roe_V * (temp3 - temp4);

        COMPUTE_NUMERICAL_FLUX_PDS(flux2, owner_cell, neighbor_cell, 0);

        // pdssd
        ind = DDSS_ * ipoint;
        for (int ds = 0; ds < DS_; ds++)
          for (int sd = 0; sd < DS_; sd++)
            flux_neighbor_grad_jacobi[ind++] = flux2[ds].df[sd];
      }
    }
  }
}

// ------------------------------- Boundary -------------------------------- //
std::shared_ptr<BoundaryNS2DPri> BoundaryNS2DPri::GetBoundary(
    const std::string& type, const int bdry_tag, EquationNS2DPri* equation) {
  if (!type.compare("AdiabaticWall"))
    return std::make_shared<AdiabaticWallNS2DPri>(bdry_tag, equation);
  else if (!type.compare("SlipWall"))
    return std::make_shared<SlipWallNS2DPri>(bdry_tag, equation);
  else if (!type.compare("SupersonicInflow"))
    return std::make_shared<SupersonicInflowBdryNS2DPri>(bdry_tag, equation);
  ERROR_MESSAGE("Wrong boundary condition (no-exist):" + type + "\n");
  return nullptr;
}
// Boundary = AdiabaticWall
// Dependency: -
AdiabaticWallNS2DPri::AdiabaticWallNS2DPri(const int bdry_tag,
                                           EquationNS2DPri* equation)
    : BoundaryNS2DPri(bdry_tag, equation) {
  MASTER_MESSAGE("AdiabaticWall (tag=" + std::to_string(bdry_tag) + ")\n");
}
void AdiabaticWallNS2DPri::ComputeBdrySolution(
    const int num_points, std::vector<double>& bdry_u,
    std::vector<double>& bdry_div_u, const std::vector<double>& owner_u,
    const std::vector<double>& owner_div_u, const std::vector<double>& normal,
    const std::vector<double>& coords, const double& time) {
  int ind = 0;
  for (int ipoint = 0; ipoint < num_points; ipoint++) {
    GET_SOLUTION_PS(, owner_u);

    // ps
    ind = S_ * ipoint;
    bdry_u[ind++] = d;
    bdry_u[ind++] = 0.0;
    bdry_u[ind++] = 0.0;
    bdry_u[ind] = T;
  }
  // Don't touch bdry_div_u.
}
void AdiabaticWallNS2DPri::ComputeBdryFlux(const int num_points,
                                           std::vector<double>& flux,
                                           FACE_INPUTS,
                                           const std::vector<double>& coords,
                                           const double& time) {
  int ind = 0;
  double AVcoeff = 0.0;
  for (int ipoint = 0; ipoint < num_points; ipoint++) {
    GET_SOLUTION_PS(, owner_u);

    GET_SOLUTION_GRAD_PDS(, owner_div_u);

    COMPUTE_FLUX_BASICS();

    // pds
    ind = DS_ * ipoint;
    AVcoeff = DENEB_ARTIFICIAL_VISCOSITY->GetArtificialViscosityValue(
        owner_cell, ipoint);
    flux[ind++] = 0.0;
    flux[ind++] = p - alpha_ * txx;
    flux[ind++] = -alpha_ * txy;
    flux[ind++] = 0.0;

    flux[ind++] = 0.0;
    flux[ind++] = -alpha_ * txy;
    flux[ind++] = p - alpha_ * tyy;
    flux[ind] = 0.0;
  }
}
void AdiabaticWallNS2DPri::ComputeBdrySolutionJacobi(
    const int num_points, double* bdry_u_jacobi,
    const std::vector<double>& owner_u, const std::vector<double>& owner_div_u,
    const std::vector<double>& normal, const std::vector<double>& coords,
    const double& time) {
  int ind = 0;
  memset(bdry_u_jacobi, 0, num_points * SS_ * sizeof(double));
  for (int ipoint = 0; ipoint < num_points; ipoint++) {
    GET_SOLUTION_PS(, owner_u);

    bdry_u_jacobi[SS_ * ipoint] = 1.0;
    bdry_u_jacobi[SS_ * ipoint + SS_ - 1] = 1.0;
  }
}
void AdiabaticWallNS2DPri::ComputeBdryFluxJacobi(
    const int num_points, std::vector<double>& flux_jacobi,
    std::vector<double>& flux_grad_jacobi, FACE_INPUTS,
    const std::vector<double>& coords, const double& time) {
  // flux_jacobi(ds1s2) = F(ds1) over U(s2)
  // flux_grad_jacobi(d1s1 s2d2) = F(d1s1) over gradU(d2s2)
  int ind = 0;
  double AVcoeff = 0.0;
  std::vector<Dual<S_>> flux1(DS_);
  std::vector<Dual<DS_>> flux2(DS_);
  for (int ipoint = 0; ipoint < num_points; ipoint++) {
    AVcoeff = DENEB_ARTIFICIAL_VISCOSITY->GetArtificialViscosityValue(
        owner_cell, ipoint);
    {
      // ps
      ind = S_ * ipoint;
      const Dual<S_> d(owner_u[ind++], 0);
      const Dual<S_> u(owner_u[ind++], 1);
      const Dual<S_> v(owner_u[ind++], 2);
      const Dual<S_> T(owner_u[ind], 3);

      GET_SOLUTION_GRAD_PDS(, owner_div_u);

      COMPUTE_FLUX_BASICS();

      // pds
      flux1[0] = 0.0;
      flux1[1] = p - alpha_ * txx;
      flux1[2] = -alpha_ * txy;
      flux1[3] = 0.0;
      flux1[4] = 0.0;
      flux1[5] = -alpha_ * txy;
      flux1[6] = p - alpha_ * tyy;
      flux1[7] = 0.0;

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
      const Dual<DS_> dx(owner_div_u[ind++], 0);
      const Dual<DS_> ux(owner_div_u[ind++], 2);
      const Dual<DS_> vx(owner_div_u[ind++], 4);
      const Dual<DS_> Tx(owner_div_u[ind++], 6);
      const Dual<DS_> dy(owner_div_u[ind++], 1);
      const Dual<DS_> uy(owner_div_u[ind++], 3);
      const Dual<DS_> vy(owner_div_u[ind++], 5);
      const Dual<DS_> Ty(owner_div_u[ind], 7);

      COMPUTE_FLUX_BASICS();

      // pds
      flux2[0] = 0.0;
      flux2[1] = p - alpha_ * txx;
      flux2[2] = -alpha_ * txy;
      flux2[3] = 0.0;
      flux2[4] = 0.0;
      flux2[5] = -alpha_ * txy;
      flux2[6] = p - alpha_ * tyy;
      flux2[7] = 0.0;

      // pdssd
      ind = DDSS_ * ipoint;
      for (int ds = 0; ds < DS_; ds++)
        for (int sd = 0; sd < DS_; sd++)
          flux_grad_jacobi[ind++] = flux2[ds].df[sd];
    }
  }
}
// Boundary = SlipWall
// Dependency: -
SlipWallNS2DPri::SlipWallNS2DPri(const int bdry_tag, EquationNS2DPri* equation)
    : BoundaryNS2DPri(bdry_tag, equation) {
  MASTER_MESSAGE("SlipWall (tag=" + std::to_string(bdry_tag) + ")\n");
}
void SlipWallNS2DPri::ComputeBdrySolution(
    const int num_points, std::vector<double>& bdry_u,
    std::vector<double>& bdry_div_u, const std::vector<double>& owner_u,
    const std::vector<double>& owner_div_u, const std::vector<double>& normal,
    const std::vector<double>& coords, const double& time) {
  int ind = 0;
  for (int ipoint = 0; ipoint < num_points; ipoint++) {
    GET_NORMAL_PD(normal);

    // ps
    GET_SOLUTION_PS(, owner_u);
    const double V = u * nx + v * ny;

    // ps
    ind = S_ * ipoint;
    bdry_u[ind++] = d;
    bdry_u[ind++] = u - V * nx;
    bdry_u[ind++] = v - V * ny;
    bdry_u[ind] = T;
  }
  // Don't touch bdry_div_u.
}
void SlipWallNS2DPri::ComputeBdryFlux(const int num_points,
                                      std::vector<double>& flux, FACE_INPUTS,
                                      const std::vector<double>& coords,
                                      const double& time) {
  int ind = 0;
  double AVcoeff = 0.0;
  for (int ipoint = 0; ipoint < num_points; ipoint++) {
    GET_SOLUTION_PS(, owner_u);

    GET_SOLUTION_GRAD_PDS(, owner_div_u);

    COMPUTE_FLUX_BASICS();

    // pds
    ind = DS_ * ipoint;
    AVcoeff = DENEB_ARTIFICIAL_VISCOSITY->GetArtificialViscosityValue(
        owner_cell, ipoint);
    flux[ind++] = 0.0;
    flux[ind++] = p;
    flux[ind++] = 0.0;
    flux[ind++] = 0.0;

    flux[ind++] = 0.0;
    flux[ind++] = 0.0;
    flux[ind++] = p;
    flux[ind] = 0.0;
  }
}
void SlipWallNS2DPri::ComputeBdrySolutionJacobi(
    const int num_points, double* bdry_u_jacobi,
    const std::vector<double>& owner_u, const std::vector<double>& owner_div_u,
    const std::vector<double>& normal, const std::vector<double>& coords,
    const double& time) {
  int ind = 0;
  memset(bdry_u_jacobi, 0, num_points * SS_ * sizeof(double));
  for (int ipoint = 0; ipoint < num_points; ipoint++) {
    GET_NORMAL_PD(normal);

    // ps
    GET_SOLUTION_PS(, owner_u);
    const double V = (u * nx + v * ny);

    ind = SS_ * ipoint;
    bdry_u_jacobi[ind++] = 1.0;
    ind += S_;
    bdry_u_jacobi[ind++] = 1.0 - nx * nx;
    bdry_u_jacobi[ind++] = -nx * ny;
    ind += S_ - 2;
    bdry_u_jacobi[ind++] = -nx * ny;
    bdry_u_jacobi[ind++] = 1.0 - ny * ny;
    ind += S_;
    bdry_u_jacobi[ind] = 1.0;
  }
}
void SlipWallNS2DPri::ComputeBdryFluxJacobi(
    const int num_points, std::vector<double>& flux_jacobi,
    std::vector<double>& flux_grad_jacobi, FACE_INPUTS,
    const std::vector<double>& coords, const double& time) {
  // flux_jacobi(ds1s2) = F(ds1) over U(s2)
  // flux_grad_jacobi(d1s1 s2d2) = F(d1s1) over gradU(d2s2)
  int ind = 0;
  double AVcoeff = 0.0;
  std::vector<Dual<S_>> flux1(DS_);
  std::vector<Dual<DS_>> flux2(DS_);
  for (int ipoint = 0; ipoint < num_points; ipoint++) {
    AVcoeff = DENEB_ARTIFICIAL_VISCOSITY->GetArtificialViscosityValue(
        owner_cell, ipoint);
    {
      // ps
      ind = S_ * ipoint;
      const Dual<S_> d(owner_u[ind++], 0);
      const Dual<S_> u(owner_u[ind++], 1);
      const Dual<S_> v(owner_u[ind++], 2);
      const Dual<S_> T(owner_u[ind], 3);

      GET_SOLUTION_GRAD_PDS(, owner_div_u);

      COMPUTE_FLUX_BASICS();

      // pds
      flux1[0] = 0.0;
      flux1[1] = p;
      flux1[2] = 0.0;
      flux1[3] = 0.0;
      flux1[4] = 0.0;
      flux1[5] = 0.0;
      flux1[6] = p;
      flux1[7] = 0.0;

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
      const Dual<DS_> dx(owner_div_u[ind++], 0);
      const Dual<DS_> ux(owner_div_u[ind++], 2);
      const Dual<DS_> vx(owner_div_u[ind++], 4);
      const Dual<DS_> Tx(owner_div_u[ind++], 6);
      const Dual<DS_> dy(owner_div_u[ind++], 1);
      const Dual<DS_> uy(owner_div_u[ind++], 3);
      const Dual<DS_> vy(owner_div_u[ind++], 5);
      const Dual<DS_> Ty(owner_div_u[ind], 7);

      COMPUTE_FLUX_BASICS();

      // pds
      flux2[0] = 0.0;
      flux2[1] = p;
      flux2[2] = 0.0;
      flux2[3] = 0.0;
      flux2[4] = 0.0;
      flux2[5] = 0.0;
      flux2[6] = p;
      flux2[7] = 0.0;

      // pdssd
      ind = DDSS_ * ipoint;
      for (int ds = 0; ds < DS_; ds++)
        for (int sd = 0; sd < DS_; sd++)
          flux_grad_jacobi[ind++] = flux2[ds].df[sd];
    }
  }
}

// Boundary = SupersonicInflow
// Dependency: BdryInput(num)
// BdryInput(num) = rho, p/pinf, AoA
SupersonicInflowBdryNS2DPri::SupersonicInflowBdryNS2DPri(
    const int bdry_tag, EquationNS2DPri* equation)
    : BoundaryNS2DPri(bdry_tag, equation) {
  MASTER_MESSAGE("SupersonicInflow boundary (tag=" + std::to_string(bdry_tag) +
                 ")\n");

  auto& config = AVOCADO_CONFIG;
  Ma_ = std::stod(config->GetConfigValue(BDRY_INPUT_I(bdry_tag, 0)));
  rho_ = std::stod(config->GetConfigValue(BDRY_INPUT_I(bdry_tag, 1)));
  p_over_pinf_ = std::stod(config->GetConfigValue(BDRY_INPUT_I(bdry_tag, 2)));
  AoA_ = std::stod(config->GetConfigValue(BDRY_INPUT_I(bdry_tag, 3)));

  MASTER_MESSAGE("SupersonicInflow (tag=" + std::to_string(bdry_tag) +
                 ")\n\tMa = " + std::to_string(Ma_) +
                 ")\n\trho = " + std::to_string(rho_) +
                 ")\n\tp/pinf = " + std::to_string(p_over_pinf_) +
                 "\n\tAOA = " + std::to_string(AoA_) + "\n");

  uf_ = Ma_ * std::cos(AoA_ * pi_ / 180.0);
  vf_ = Ma_ * std::sin(AoA_ * pi_ / 180.0);
}
void SupersonicInflowBdryNS2DPri::ComputeBdrySolution(
    const int num_points, std::vector<double>& bdry_u,
    std::vector<double>& bdry_div_u, const std::vector<double>& owner_u,
    const std::vector<double>& owner_div_u, const std::vector<double>& normal,
    const std::vector<double>& coords, const double& time) {
  static const double p = p_over_pinf_ * r_inv_;
  int ind = 0;
  for (int ipoint = 0; ipoint < num_points; ipoint++) {
    // ps
    ind = S_ * ipoint;
    bdry_u[ind++] = rho_;
    bdry_u[ind++] = uf_;
    bdry_u[ind++] = vf_;
    bdry_u[ind] = r_ * p / rho_;
  }
}
void SupersonicInflowBdryNS2DPri::ComputeBdryFlux(
    const int num_points, std::vector<double>& flux, FACE_INPUTS,
    const std::vector<double>& coords, const double& time) {
  const double pb = p_over_pinf_ * r_inv_;
  const double dEb = pb * rm1_inv_ + 0.5 * rho_ * (uf_ * uf_ + vf_ * vf_);
  int ind = 0;
  for (int ipoint = 0; ipoint < num_points; ipoint++) {
    // pds
    ind = DS_ * ipoint;
    flux[ind++] = rho_ * uf_;
    flux[ind++] = rho_ * uf_ * uf_ + pb;
    flux[ind++] = rho_ * uf_ * vf_;
    flux[ind++] = (dEb + pb) * uf_;

    flux[ind++] = rho_ * vf_;
    flux[ind++] = rho_ * vf_ * uf_;
    flux[ind++] = rho_ * vf_ * vf_ + pb;
    flux[ind] = (dEb + pb) * vf_;
  }
}
void SupersonicInflowBdryNS2DPri::ComputeBdrySolutionJacobi(
    const int num_points, double* bdry_u_jacobi,
    const std::vector<double>& owner_u, const std::vector<double>& owner_div_u,
    const std::vector<double>& normal, const std::vector<double>& coords,
    const double& time) {
  memset(bdry_u_jacobi, 0, num_points * SS_ * sizeof(double));
}
void SupersonicInflowBdryNS2DPri::ComputeBdryFluxJacobi(
    const int num_points, std::vector<double>& flux_jacobi,
    std::vector<double>& flux_grad_jacobi, FACE_INPUTS,
    const std::vector<double>& coords, const double& time) {
  static const int max_num_bdry_points = equation_->GetMaxNumBdryPoints();
  static std::vector<double> flux_neighbor_jacobi(max_num_bdry_points * DSS_);
  static std::vector<double> flux_neighbor_grad_jacobi(max_num_bdry_points *
                                                       DDSS_);
  static std::vector<double> bdry_u_jacobi(max_num_bdry_points * SS_);

  equation_->ComputeNumFluxJacobi(num_points, flux_jacobi, flux_neighbor_jacobi,
                                  flux_grad_jacobi, flux_neighbor_grad_jacobi,
                                  owner_cell, -1, owner_u, owner_div_u,
                                  neighbor_u, neighbor_div_u, normal);
  ComputeBdrySolutionJacobi(num_points, &bdry_u_jacobi[0], owner_u, owner_div_u,
                            normal, coords, time);

  for (int ipoint = 0; ipoint < num_points; ipoint++)
    gemmAB(1.0, &flux_neighbor_jacobi[ipoint * DSS_],
           &bdry_u_jacobi[ipoint * SS_], 1.0, &flux_jacobi[ipoint * DSS_], DS_,
           S_, S_);
}

// -------------------------------- Problem -------------------------------- //
std::shared_ptr<ProblemNS2DPri> ProblemNS2DPri::GetProblem(
    const std::string& name) {
  if (!name.compare("FreeStream"))
    return std::make_shared<FreeStreamNS2DPri>();
  else if (!name.compare("DoubleSine"))
    return std::make_shared<DoubleSineNS2DPri>();
  else
    ERROR_MESSAGE("Wrong problem (no-exist):" + name + "\n");
  return nullptr;
}
// Problem = FreeStream
// ProblemInput = Ma, AOA
FreeStreamNS2DPri::FreeStreamNS2DPri() {
  MASTER_MESSAGE("FreeStream problem\n");

  auto& config = AVOCADO_CONFIG;
  Ma_ = std::stod(config->GetConfigValue(PROBLEM_INPUT_I(0)));
  AoA_ = std::stod(config->GetConfigValue(PROBLEM_INPUT_I(1)));
  uf_ = Ma_ * std::cos(AoA_ * pi_ / 180.0);
  vf_ = Ma_ * std::sin(AoA_ * pi_ / 180.0);

  std::stringstream str;
  str << "\tInput:\n";
  str << "\t\tMach number = " << Ma_ << "\n";
  str << "\t\tIncident angle = " << AoA_ << " (deg)\n";
  MASTER_MESSAGE(str.str());
}
void FreeStreamNS2DPri::Problem(const int num_points,
                                std::vector<double>& solutions,
                                const std::vector<double>& coord,
                                const double time) const {
  int ind = 0;
  for (int ipoint = 0; ipoint < num_points; ipoint++) {
    solutions[ind++] = 1.0;
    solutions[ind++] = uf_;
    solutions[ind++] = vf_;
    solutions[ind++] = 1.0;
  }
}
// Problem = DoubleSine
// ProblemInput = -
DoubleSineNS2DPri::DoubleSineNS2DPri()
    : velocity_({0.5, 0.5}), wave_number_({1.0, 1.0}) {
  MASTER_MESSAGE("DoubleSine problem\n");

  std::stringstream str;
  str << "\tInput (fixed):\n";
  str << "\t\tx-velocity = " << velocity_[0] << "\n";
  str << "\t\ty-velocity = " << velocity_[1] << "\n";
  str << "\t\tx-wavenumber = " << wave_number_[0] << "\n";
  str << "\t\ty-wavenumber = " << wave_number_[1] << "\n";
  MASTER_MESSAGE(str.str());
}
void DoubleSineNS2DPri::Problem(const int num_points,
                                std::vector<double>& solutions,
                                const std::vector<double>& coord,
                                const double time) const {
  int ind = 0;
  std::vector<double> values(S_);
  values[1] = velocity_[0];
  values[2] = velocity_[1];
  values[3] = 1.0;
  for (int ipoint = 0; ipoint < num_points; ipoint++) {
    const double& x = coord[ipoint * D_];
    const double& y = coord[ipoint * D_ + 1];

    values[0] =
        1.0 +
        0.2 *
            std::sin(2.0 * pi_ * wave_number_[0] * (x - velocity_[0] * time)) *
            std::sin(2.0 * pi_ * wave_number_[1] * (y - velocity_[1] * time));

    solutions[ind++] = values[0];
    for (int idim = 0; idim < D_; idim++)
      solutions[ind++] = velocity_[idim];
    solutions[ind++] = r_ / values[0];
  }
}
}  // namespace deneb