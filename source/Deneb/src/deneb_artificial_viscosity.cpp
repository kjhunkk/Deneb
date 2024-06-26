#include "deneb_artificial_viscosity.h"

#include <cstring>
#include <unordered_set>

#include "avocado.h"
#include "deneb_config_macro.h"
#include "deneb_data.h"
#include "deneb_equation.h"
#include "deneb_quadrature.h"

namespace deneb {
std::shared_ptr<ArtificialViscosity> DENEB_ARTIFICIAL_VISCOSITY_NAME = nullptr;
std::shared_ptr<ArtificialViscosity>
ArtificialViscosity::GetArtificialViscosity(const std::string& name) {
  if (!name.compare("None"))
    return std::make_shared<NoArtificialViscosity>();
  else if (!name.compare("LaplacianP0"))
    return std::make_shared<LaplacianP0>();
  else if (!name.compare("LaplacianP0All"))
    return std::make_shared<LaplacianP0All>();
  else if (!name.compare("LaplacianPolyShockFit"))
    return std::make_shared<LaplacianPolyShockFit>();
  else if (!name.compare("LaplacianPolyShockFitWall"))
    return std::make_shared<LaplacianPolyShockFitWall>();
  else if (!name.compare("SPID"))
    return std::make_shared<SPID>();
  else
    ERROR_MESSAGE("Wrong artificial viscosity (no-exist):" + name + "\n");
  return nullptr;
}

NoArtificialViscosity::NoArtificialViscosity() {
  MASTER_MESSAGE(avocado::GetTitle("NoArtificialViscosity"));
}
void NoArtificialViscosity::BuildData(void) {
  MASTER_MESSAGE(avocado::GetTitle("NoArtificialViscosity::BuildData()"));
}

// -------------------- LaplacianP0 ------------------ //
LaplacianP0::LaplacianP0() { MASTER_MESSAGE(avocado::GetTitle("LaplacianP0")); }
void LaplacianP0::BuildData(void) {
  MASTER_MESSAGE(avocado::GetTitle("LaplacianP0::BuildData()"));

  const std::vector<std::string>& variable_names =
      DENEB_EQUATION->GetCellVariableNames();

  auto& config = AVOCADO_CONFIG;
  const std::string& equation = config->GetConfigValue(EQUATION);
  if (!equation.compare("Euler2D"))
    ERROR_MESSAGE("Artificial viscosity is not compatible with " + equation +
                  ", use NS equation with Re<0\n");
  else if (!equation.compare("ScalarAdvection2D"))
    ERROR_MESSAGE("Artificial viscosity is not compatible with " + equation +
                  ", use NS equation with Re<0\n");

  Peclet_ = std::stod(config->GetConfigValue(PECLET));
  kappa_ = std::stod(config->GetConfigValue(KAPPA));

  const int& num_outer_cells = DENEB_DATA->GetNumOuterCells();
  const int& num_cells = DENEB_DATA->GetNumCells();
  artificial_viscosity_.resize(num_outer_cells + 1, 0.0);
  Se_.resize(num_cells, 0.0);

  const int& order = DENEB_DATA->GetOrder();
  if (order == 0) ERROR_MESSAGE("P0 doesn't require artificial viscosity");
  S0_ = -3.0 * std::log10(order);

  const int& dimension = DENEB_EQUATION->GetDimension();
  if (dimension == 2)
    num_bases_m1_ = order * (order + 1) / 2;
  else if (dimension == 3)
    num_bases_m1_ = order * (order + 1) * (order + 2) / 6;
  else
    ERROR_MESSAGE("Dimension error\n");

  std::vector<double> GL_points;
  std::vector<double> GL_weights;
  quadrature::Line_Poly1D(order, GL_points, GL_weights);
  for (int ipoint = 0; ipoint < GL_points.size(); ipoint++)
    GL_points[ipoint] = 0.5 * (GL_points[ipoint] + 1.0);
  dLmax_ = 0.0;
  for (int ipoint = 0; ipoint < GL_points.size() - 1; ipoint++) {
    const double dist = GL_points[ipoint] - GL_points[ipoint + 1];
    if (dLmax_ < dist) dLmax_ = dist;
  }

  const int& num_states = DENEB_EQUATION->GetNumStates();
  const int& num_bases = DENEB_DATA->GetNumBases();
  const int sb = num_states * num_bases;
  communicate_ = std::make_shared<avocado::Communicate>(
      1, DENEB_DATA->GetOuterSendCellList(),
      DENEB_DATA->GetOuterRecvCellList());
}
void LaplacianP0::ComputeArtificialViscosity(const double* solution,
                                             const double dt) {
  static const int& order = DENEB_DATA->GetOrder();
  if (order == 0) return;
  static const int& num_cells = DENEB_DATA->GetNumCells();
  static const int& num_outer_cells = DENEB_DATA->GetNumOuterCells();
  static const int& num_states = DENEB_EQUATION->GetNumStates();
  static const int& num_bases = DENEB_DATA->GetNumBases();
  static const int sb = num_states * num_bases;
  static const std::vector<double>& cell_volumes = DENEB_DATA->GetCellVolumes();
  static const std::vector<std::vector<double>>& cell_basis_value =
      DENEB_DATA->GetCellBasisValue();

  for (int icell = 0; icell < num_cells; icell++) {
    const double E0 = MaxArtificialViscosity(
        &solution[icell * sb], cell_volumes[icell], cell_basis_value[icell][0]);
    Se_[icell] = SmoothnessIndicator(&solution[icell * sb]);

    if (Se_[icell] <= S0_ - kappa_)
      artificial_viscosity_[icell] = 0.0;
    else if (Se_[icell] <= S0_ + kappa_)
      artificial_viscosity_[icell] =
          0.5 * E0 *
          (1.0 + std::sin(M_PI * (Se_[icell] - S0_) / (2.0 * kappa_)));
    else
      artificial_viscosity_[icell] = E0;
  }
  communicate_->CommunicateBegin(&artificial_viscosity_[0]);
  communicate_->CommunicateEnd(&artificial_viscosity_[num_cells]);
}
double LaplacianP0::SmoothnessIndicator(const double* solution) {
  static const int& num_bases = DENEB_DATA->GetNumBases();
  static const int& num_species = DENEB_EQUATION->GetNumSpecies();
  static const int sb = num_bases * num_species;
  double Pn_value = 0.0;  
  double Pn_minus_1_value = 0.0;
  for (int i = 0; i < num_species; i++) {
    Pn_value += avocado::VecInnerProd(num_bases, &solution[i * num_bases],
                                      &solution[i * num_bases]);
    Pn_minus_1_value += avocado::VecInnerProd(
        num_bases_m1_, &solution[i * num_bases], &solution[i * num_bases]);
  }

  const double del_Pn = Pn_value - Pn_minus_1_value;
  if (del_Pn <= 1.0e-8)
    return -1.0e+8;
  else
    return std::log10(del_Pn / Pn_value);
}

double LaplacianP0::MaxArtificialViscosity(const double* solution,
                                           const double cell_volumes,
                                           const double cell_basis_value) {
  static const int& dimension = DENEB_EQUATION->GetDimension();
  static const int& num_states = DENEB_EQUATION->GetNumStates();
  static const int& num_bases = DENEB_DATA->GetNumBases();
  static const int& num_cells = DENEB_DATA->GetNumCells();
  static std::vector<double> input_solutions(num_states);

  for (int istate = 0; istate < num_states; istate++)
    input_solutions[istate] = solution[istate * num_bases] * cell_basis_value;

  const double max_speed =
      DENEB_EQUATION->ComputeMaxCharacteristicSpeed(&input_solutions[0]);
  const double length_scale = std::pow(cell_volumes, 1.0 / dimension);
  return max_speed * length_scale * (2.0 - dLmax_) / Peclet_;
}

// -------------------- LaplacianP0All ------------------ //
LaplacianP0All::LaplacianP0All() {
  MASTER_MESSAGE(avocado::GetTitle("LaplacianP0All"));
}
void LaplacianP0All::BuildData(void) {
  MASTER_MESSAGE(avocado::GetTitle("LaplacianP0All::BuildData()"));

  const std::vector<std::string>& variable_names =
      DENEB_EQUATION->GetCellVariableNames();

  auto& config = AVOCADO_CONFIG;
  const std::string& equation = config->GetConfigValue(EQUATION);
  if (!equation.compare("Euler2D"))
    ERROR_MESSAGE("Artificial viscosity is not compatible with " + equation +
                  ", use NS equation with Re<0\n");
  else if (!equation.compare("ScalarAdvection2D"))
    ERROR_MESSAGE("Artificial viscosity is not compatible with " + equation +
                  ", use NS equation with Re<0\n");

  Peclet_ = std::stod(config->GetConfigValue(PECLET));
  kappa_ = std::stod(config->GetConfigValue(KAPPA));

  const int& num_outer_cells = DENEB_DATA->GetNumOuterCells();
  const int& num_cells = DENEB_DATA->GetNumCells();
  artificial_viscosity_.resize(num_outer_cells + 1, 0.0);
  Se_.resize(num_cells, 0.0);

  const int& order = DENEB_DATA->GetOrder();
  if (order == 0) ERROR_MESSAGE("P0 doesn't require artificial viscosity");
  S0_ = -3.0 * std::log10(order);

  const int& dimension = DENEB_EQUATION->GetDimension();
  if (dimension == 2)
    num_bases_m1_ = order * (order + 1) / 2;
  else if (dimension == 3)
    num_bases_m1_ = order * (order + 1) * (order + 2) / 6;
  else
    ERROR_MESSAGE("Dimension error\n");

  std::vector<double> GL_points;
  std::vector<double> GL_weights;
  quadrature::Line_Poly1D(order, GL_points, GL_weights);
  for (int ipoint = 0; ipoint < GL_points.size(); ipoint++)
    GL_points[ipoint] = 0.5 * (GL_points[ipoint] + 1.0);
  dLmax_ = 0.0;
  for (int ipoint = 0; ipoint < GL_points.size() - 1; ipoint++) {
    const double dist = GL_points[ipoint] - GL_points[ipoint + 1];
    if (dLmax_ < dist) dLmax_ = dist;
  }

  const int& num_states = DENEB_EQUATION->GetNumStates();
  const int& num_bases = DENEB_DATA->GetNumBases();
  const int sb = num_states * num_bases;
  communicate_ = std::make_shared<avocado::Communicate>(
      1, DENEB_DATA->GetOuterSendCellList(),
      DENEB_DATA->GetOuterRecvCellList());
}
void LaplacianP0All::ComputeArtificialViscosity(const double* solution,
                                                const double dt) {
  static const int& order = DENEB_DATA->GetOrder();
  if (order == 0) return;
  static const int& num_cells = DENEB_DATA->GetNumCells();
  static const int& num_outer_cells = DENEB_DATA->GetNumOuterCells();
  static const int& num_states = DENEB_EQUATION->GetNumStates();
  static const int& num_bases = DENEB_DATA->GetNumBases();
  static const int sb = num_states * num_bases;
  static const std::vector<double>& cell_volumes = DENEB_DATA->GetCellVolumes();
  static const std::vector<std::vector<double>>& cell_basis_value =
      DENEB_DATA->GetCellBasisValue();

  for (int icell = 0; icell < num_cells; icell++) {
    const double E0 = MaxArtificialViscosity(
        &solution[icell * sb], cell_volumes[icell], cell_basis_value[icell][0]);
    Se_[icell] = SmoothnessIndicator(&solution[icell * sb]);

    if (Se_[icell] <= S0_ - kappa_)
      artificial_viscosity_[icell] = 0.0;
    else if (Se_[icell] <= S0_ + kappa_)
      artificial_viscosity_[icell] =
          0.5 * E0 *
          (1.0 + std::sin(M_PI * (Se_[icell] - S0_) / (2.0 * kappa_)));
    else
      artificial_viscosity_[icell] = E0;
  }
  communicate_->CommunicateBegin(&artificial_viscosity_[0]);
  communicate_->CommunicateEnd(&artificial_viscosity_[num_cells]);
}
double LaplacianP0All::SmoothnessIndicator(const double* solution) {
  static const int& num_bases = DENEB_DATA->GetNumBases();
  static const int& num_states = DENEB_EQUATION->GetNumStates();
  static const int sb = num_bases * num_states;
  const double Pn_value = avocado::VecInnerProd(sb, &solution[0], &solution[0]);
  double Pn_minus_1_value = 0.0;
  for (int i = 0; i < num_states; i++)
    Pn_minus_1_value += avocado::VecInnerProd(
        num_bases_m1_, &solution[i * num_bases], &solution[i * num_bases]);

  const double del_Pn = Pn_value - Pn_minus_1_value;
  if (del_Pn <= 1.0e-8)
    return -1.0e+8;
  else
    return std::log10(del_Pn / Pn_value);
}

double LaplacianP0All::MaxArtificialViscosity(const double* solution,
                                              const double cell_volumes,
                                              const double cell_basis_value) {
  static const int& dimension = DENEB_EQUATION->GetDimension();
  static const int& num_states = DENEB_EQUATION->GetNumStates();
  static const int& num_bases = DENEB_DATA->GetNumBases();
  static const int& num_cells = DENEB_DATA->GetNumCells();
  static std::vector<double> input_solutions(num_states);

  for (int istate = 0; istate < num_states; istate++)
    input_solutions[istate] = solution[istate * num_bases] * cell_basis_value;

  const double max_speed =
      DENEB_EQUATION->ComputeMaxCharacteristicSpeed(&input_solutions[0]);
  const double length_scale = std::pow(cell_volumes, 1.0 / dimension);
  return max_speed * length_scale * (2.0 - dLmax_) / Peclet_;
}

// -------------------- LaplacianPolyShockFit ------------------ //
LaplacianPolyShockFit::LaplacianPolyShockFit() {
  MASTER_MESSAGE(avocado::GetTitle("LaplacianPolyShockFit"));
}
void LaplacianPolyShockFit::BuildData(void) {
  MASTER_MESSAGE(avocado::GetTitle("LaplacianPolyShockFit::BuildData()"));

  const std::vector<std::string>& variable_names =
      DENEB_EQUATION->GetCellVariableNames();

  auto& config = AVOCADO_CONFIG;
  const std::string& equation = config->GetConfigValue(EQUATION);
  if (!equation.compare("Euler2D"))
    ERROR_MESSAGE("Artificial viscosity is not compatible with " + equation +
                  ", use NS equation with Re<0\n");
  else if (!equation.compare("ScalarAdvection2D"))
    ERROR_MESSAGE("Artificial viscosity is not compatible with " + equation +
                  ", use NS equation with Re<0\n");

  MaxAV_ = std::stod(config->GetConfigValue("ArtificialViscosity.1"));
  S0_ = std::stod(config->GetConfigValue("ArtificialViscosity.2"));
  sigma_ = std::stod(config->GetConfigValue("ArtificialViscosity.3"));
  poly_order_ = std::stoi(config->GetConfigValue("ArtificialViscosity.4"));
  update_period_ = std::stoi(config->GetConfigValue("ArtificialViscosity.5"));
  eps_ = std::stod(config->GetConfigValue("ArtificialViscosity.6"));
  minPts_ = std::stoi(config->GetConfigValue("ArtificialViscosity.7"));

  if (MaxAV_ <= 0.0) {
    ERROR_MESSAGE(
        "LaplacianPolyShockFit: illegal value at MaxAV = " + std::to_string(MaxAV_) + "\n");
  }
  if (sigma_ <= 0.0) {
    ERROR_MESSAGE("LaplacianPolyShockFit: illegal value at sigma = " +
                  std::to_string(sigma_) + "\n");
  }
  if (poly_order_ < 2) {
    ERROR_MESSAGE("LaplacianPolyShockFit: illegal value at poly_order = " +
                  std::to_string(poly_order_) + " (should be > 1)\n");
  }
  if (eps_ <= 0.0) {
    ERROR_MESSAGE("LaplacianPolyShockFit: illegal value at eps = " +
                  std::to_string(eps_) + "\n");
  }
  if (minPts_ < 1) {
    ERROR_MESSAGE("LaplacianPolyShockFit: illegal value at minPts = " +
                  std::to_string(minPts_) + " (should be > 0)\n");
  }
  newton_tol_ = 1.0e-4;
  max_newton_iter_ = 1000;

  const int& num_outer_cells = DENEB_DATA->GetNumOuterCells();
  const int& num_cells = DENEB_DATA->GetNumCells();
  artificial_viscosity_.resize(num_outer_cells + 1, 0.0);
  Se_.resize(num_cells, 0.0);
  distance_from_shock_.resize(num_cells, 0.0);
  shock_poly_coeff_.resize(poly_order_ + 1, 0.0);

  const int& order = DENEB_DATA->GetOrder();
  if (order == 0) ERROR_MESSAGE("P0 doesn't require artificial viscosity");

  const int& dimension = DENEB_EQUATION->GetDimension();
  if (dimension == 2)
    num_bases_m1_ = order * (order + 1) / 2;
  else if (dimension == 3)
    num_bases_m1_ = order * (order + 1) * (order + 2) / 6;
  else
    ERROR_MESSAGE("Dimension error\n");

  std::vector<double> GL_points;
  std::vector<double> GL_weights;
  quadrature::Line_Poly1D(order, GL_points, GL_weights);
  for (int ipoint = 0; ipoint < GL_points.size(); ipoint++)
    GL_points[ipoint] = 0.5 * (GL_points[ipoint] + 1.0);
  dLmax_ = 0.0;
  for (int ipoint = 0; ipoint < GL_points.size() - 1; ipoint++) {
    const double dist = GL_points[ipoint] - GL_points[ipoint + 1];
    if (dLmax_ < dist) dLmax_ = dist;
  }

  const int& num_states = DENEB_EQUATION->GetNumStates();
  const int& num_bases = DENEB_DATA->GetNumBases();
  const int sb = num_states * num_bases;
  communicate_ = std::make_shared<avocado::Communicate>(
      1, DENEB_DATA->GetOuterSendCellList(),
      DENEB_DATA->GetOuterRecvCellList());
}
void LaplacianPolyShockFit::ComputeArtificialViscosity(const double* solution,
                                             const double dt) {
  static const int& order = DENEB_DATA->GetOrder();
  if (order == 0) return;
  static const int& num_cells = DENEB_DATA->GetNumCells();
  static const int& num_states = DENEB_EQUATION->GetNumStates();
  static const int& num_bases = DENEB_DATA->GetNumBases();
  static const int sb = num_states * num_bases;
  static const std::vector<double>& cell_volumes = DENEB_DATA->GetCellVolumes();
  static const std::vector<std::vector<double>>& cell_basis_value =
      DENEB_DATA->GetCellBasisValue();
  static int av_timer = 0;

  if (av_timer-- == 0) {
    SpotSuspect(solution);
    DBSCAN();
    ShockPolyFit();
    ComputeDistanceFromShock();

    av_timer = update_period_;
  }  

  for (int icell = 0; icell < num_cells; icell++) {
    const double E0 = MaxArtificialViscosity(
        &solution[icell * sb], cell_volumes[icell], cell_basis_value[icell][0]);
    artificial_viscosity_[icell] = AVfunction(E0, distance_from_shock_[icell]);
  }
  communicate_->CommunicateBegin(&artificial_viscosity_[0]);
  communicate_->CommunicateEnd(&artificial_viscosity_[num_cells]);
}
double LaplacianPolyShockFit::SmoothnessIndicator(const double* solution) {
  static const int& num_bases = DENEB_DATA->GetNumBases();
  static const int& num_species = DENEB_EQUATION->GetNumSpecies();
  static const int sb = num_bases * num_species;
  double Pn_value = 0.0;
  double Pn_minus_1_value = 0.0;
  for (int i = num_species + 2; i < num_species + 3; i++) {
    Pn_value += avocado::VecInnerProd(num_bases, &solution[i * num_bases],
                                      &solution[i * num_bases]);
    Pn_minus_1_value += avocado::VecInnerProd(
        num_bases_m1_, &solution[i * num_bases], &solution[i * num_bases]);
  }

  const double del_Pn = Pn_value - Pn_minus_1_value;
  if (del_Pn <= 1.0e-8)
    return -1.0e+8;
  else
    return std::log10((Pn_value - Pn_minus_1_value) / Pn_value);
}

double LaplacianPolyShockFit::MaxArtificialViscosity(
    const double* solution,
                                           const double cell_volumes,
                                           const double cell_basis_value) {
  static const int& dimension = DENEB_EQUATION->GetDimension();
  static const int& num_states = DENEB_EQUATION->GetNumStates();
  static const int& num_bases = DENEB_DATA->GetNumBases();
  static const int& num_cells = DENEB_DATA->GetNumCells();
  static std::vector<double> input_solutions(num_states);

  for (int istate = 0; istate < num_states; istate++)
    input_solutions[istate] = solution[istate * num_bases] * cell_basis_value;

  //const double max_speed =
  //    DENEB_EQUATION->ComputeMaxCharacteristicSpeed(&input_solutions[0]);
  const double length_scale = std::pow(cell_volumes, 1.0 / dimension);
  //return max_speed * length_scale * (2.0 - dLmax_) / Peclet_;
  //return length_scale * (2.0 - dLmax_) / Peclet_;
  return MaxAV_;
}

void LaplacianPolyShockFit::SpotSuspect(const double* solution) {
  static const int& num_cells = DENEB_DATA->GetNumCells();
  static const int& num_states = DENEB_EQUATION->GetNumStates();
  static const int& num_bases = DENEB_DATA->GetNumBases();
  static const int& dimension = DENEB_EQUATION->GetDimension();
  static const int sb = num_states * num_bases;

  const std::vector<double>& cell_center_coords =
      DENEB_DATA->GetCellCenterCoords();
  std::vector<double> local_suspect_coords;
  for (int icell = 0; icell < num_cells; icell++) {
    Se_[icell] = SmoothnessIndicator(&solution[icell * sb]);

    // for debug
    bool add_flag = false;
    const double x = cell_center_coords[icell * dimension];
    const double y = cell_center_coords[icell * dimension + 1];
    const double r = std::sqrt(x * x + y * y);
    if (r > 0.51) add_flag = true;
    // end for debug

    if (Se_[icell] > S0_ && add_flag) {      
      for (int idim = 0; idim < dimension; idim++)
        local_suspect_coords.push_back(
            cell_center_coords[icell * dimension + idim]);
    }
  }

  const int local_num_suspect = local_suspect_coords.size() / dimension;
  std::vector<int> num_suspect(NDOMAIN);
  MPI_Allgather(&local_num_suspect, 1, MPI_INT, &num_suspect[0], 1, MPI_INT,
                MPI_COMM_WORLD);

  int total_num_suspect = 0;
  for (int idomain = 0; idomain < NDOMAIN; idomain++) {
    total_num_suspect += num_suspect[idomain];
    num_suspect[idomain] *= dimension;
  }  

  std::vector<double> suspect_coords(total_num_suspect * dimension);
  std::vector<int> displs(NDOMAIN, 0);
  for (int idomain = 1; idomain < NDOMAIN; idomain++)
    displs[idomain] = displs[idomain - 1] + num_suspect[idomain - 1];
  MPI_Allgatherv(&local_suspect_coords[0], num_suspect[MYRANK],
                 MPI_DOUBLE, &suspect_coords[0], &num_suspect[0], &displs[0],
                 MPI_DOUBLE, MPI_COMM_WORLD);

  suspects_.clear();
  suspects_.resize(total_num_suspect);
  std::vector<double> single_coord(dimension);
  for (int i = 0; i < total_num_suspect; i++) {
    for (int idim = 0; idim < dimension; idim++)
      single_coord[idim] = suspect_coords[i * dimension + idim];
    suspects_[i] = Point(single_coord);
  }
}

void LaplacianPolyShockFit::DBSCAN() {
  int clusterId = 0;

  for (Point& p : suspects_) {
    if (p.visited) continue;
    p.visited = true;

    std::vector<Point*> neighbors;

    for (Point& q : suspects_) {
      if (Distance(p, q) < eps_) {
        neighbors.push_back(&q);
      }
    }

    if (neighbors.size() < minPts_) {
      p.cluster = -1;  // Noise point
    } else {
      p.cluster = clusterId;

      for (size_t i = 0; i < neighbors.size(); ++i) {
        Point* current = neighbors[i];
        if (!current->visited) {
          current->visited = true;

          std::vector<Point*> currentNeighbors;

          for (Point& q : suspects_) {
            if (Distance(*current, q) < eps_) {
              currentNeighbors.push_back(&q);
            }
          }

          if (currentNeighbors.size() >= minPts_) {
            neighbors.insert(neighbors.end(), currentNeighbors.begin(),
                             currentNeighbors.end());
          }
        }

        if (current->cluster == -1) {
          current->cluster = clusterId;
        }
      }

      clusterId++;
    }
  }

  std::vector<int> num_cluster(clusterId + 1, 0);
  for (Point& p : suspects_) {
    if (p.cluster != -1) num_cluster[p.cluster]++;
  }
  max_cluster_ = std::max_element(num_cluster.begin(), num_cluster.end()) -
                num_cluster.begin();
}

void LaplacianPolyShockFit::ShockPolyFit() {
  // Only for 2D
  // Polynomial fitting : y = c0 + c1*x + c2*x^2 + c3*x^3 ...
  static const int& dimension = DENEB_EQUATION->GetDimension();
  if (dimension > 2)
    ERROR_MESSAGE("ShockPolyFit: this function is only supported in 2-D\n");

  std::vector<double> xFilteredData, yFilteredData;
  for (const Point& p : suspects_) {
    if (p.cluster == max_cluster_) {
      // Reverse
      yFilteredData.emplace_back(p.coords[0]);
      xFilteredData.emplace_back(p.coords[1]);
      //yFilteredData.emplace_back(p.coords[1]);
      //xFilteredData.emplace_back(p.coords[0]);
    }
  }

  int numData = xFilteredData.size();

  std::vector<std::vector<double>> A(poly_order_ + 1,
                                     std::vector<double>(poly_order_ + 1, 0.0));
  std::vector<double> B(poly_order_ + 1, 0.0);

  // Generate least square linear system
  for (int i = 0; i < numData; ++i) {
    for (int j = 0; j <= poly_order_; ++j) {
      double xPow = pow(xFilteredData[i], j);
      for (int k = 0; k <= poly_order_; ++k) {
        A[j][k] += xPow * pow(xFilteredData[i], k);
      }
      B[j] += yFilteredData[i] * xPow;
    }
  }

  // Gauss elimination
  for (int i = 0; i <= poly_order_; ++i) {
    for (int j = i + 1; j <= poly_order_; ++j) {
      double ratio = A[j][i] / A[i][i];
      for (int k = 0; k <= poly_order_; ++k) {
        A[j][k] -= ratio * A[i][k];
      }
      B[j] -= ratio * B[i];
    }
  }

  // Linear system solve
  for (int i = poly_order_; i >= 0; --i) {
    for (int j = i + 1; j <= poly_order_; ++j) {
      B[i] -= A[i][j] * shock_poly_coeff_[j];
    }
    shock_poly_coeff_[i] = B[i] / A[i][i];
  }
}

void LaplacianPolyShockFit::ComputeDistanceFromShock() {
  static const int& dimension = DENEB_EQUATION->GetDimension();
  if (dimension > 2)
    ERROR_MESSAGE("ShockPolyFit: this function is only supported in 2-D\n");

  const std::vector<double>& cell_center_coords =
      DENEB_DATA->GetCellCenterCoords();
  static const int& num_cells = DENEB_DATA->GetNumCells();

  for (int icell = 0; icell < num_cells; icell++) {
    //const double xp = cell_center_coords[icell * dimension];
    //const double yp = cell_center_coords[icell * dimension + 1];
    // Reverse
    const double xp = cell_center_coords[icell * dimension + 1];
    const double yp = cell_center_coords[icell * dimension];

    // double xn = 0.0;
    double xn = xp;
    double yn = 0.0;
    double delta_x = 0.0;
    bool converge_flag = false;    
    for (int iter = 0; iter < max_newton_iter_; iter++) {
      double fx = shock_poly_coeff_[0] + shock_poly_coeff_[1] * xn;
      double fxdot = shock_poly_coeff_[1];
      double fxdotdot = 0.0;

      for (int idegree = 2; idegree < poly_order_; idegree++) {
        fx += shock_poly_coeff_[idegree] * std::pow(xn, idegree);
        fxdot += static_cast<double>(idegree) * shock_poly_coeff_[idegree] *
                 std::pow(xn, idegree - 1);
        fxdotdot += static_cast<double>(idegree) *
                    static_cast<double>(idegree - 1) *
                    shock_poly_coeff_[idegree] * std::pow(xn, idegree - 2);

      }

      const double gx = xn - xp + (fx - yp) * fxdot;
      const double gxdot = 1.0 + fxdot * fxdot + (fx - yp) * fxdotdot;

      delta_x = - gx / gxdot;
      const double rate =
          0.5 *
          (1.0 + std::cos(4.0 * M_PI *
                          (static_cast<double>(iter) / max_newton_iter_)));
      xn = xn + rate * delta_x;
      if (std::abs(delta_x) <= newton_tol_) {
        converge_flag = true;
        yn = fx;
        break;
      }
    }
    if (converge_flag) {
      distance_from_shock_[icell] =
          std::sqrt(std::pow(xp - xn, 2.0) + std::pow(yp - yn, 2.0));
    } else
      ERROR_MESSAGE("LaplacianPolyShockFit: Newton's method not converged.\n");
  }
}

double LaplacianPolyShockFit::AVfunction(const double E0, const double distance) {
  // Gaussian distribution
  static const double deno = 1.0 / (std::sqrt(2.0 * M_PI) * sigma_);
  static const double inv_sigma = 1.0 / sigma_;
  return E0 * deno * std::exp(-0.5 * std::pow(distance * inv_sigma, 2.0));
}

double LaplacianPolyShockFit::Distance(const Point& point1, const Point& point2) {
  static const int& dimension = DENEB_EQUATION->GetDimension();
  double sum = 0.0;
  for (int idim = 0; idim < dimension; idim++)
    sum += std::pow(point1.coords[idim] - point2.coords[idim], 2.0);
  return std::sqrt(sum);
}

// -------------------- LaplacianPolyShockFitWall ------------------ //
LaplacianPolyShockFitWall::LaplacianPolyShockFitWall() {
  MASTER_MESSAGE(avocado::GetTitle("LaplacianPolyShockFitWall"));
}
void LaplacianPolyShockFitWall::BuildData(void) {
  MASTER_MESSAGE(avocado::GetTitle("LaplacianPolyShockFitWall::BuildData()"));
  LaplacianPolyShockFit::BuildData();

  auto& config = AVOCADO_CONFIG;
  wallAV_ = std::stod(config->GetConfigValue("ArtificialViscosity.8"));

  if (wallAV_ <= 0.0) {
    ERROR_MESSAGE(
      "LaplacianPolyShockFitWall: illegal value at wallAV = " + std::to_string(wallAV_) + "\n");
  }

  const int& num_bdries = DENEB_DATA->GetNumBdries();  
  const auto& bdry_owner_cell = DENEB_DATA->GetBdryOwnerCell();
  const auto& bdry_tag = DENEB_DATA->GetBdryTag();
  for (int ibdry = 0; ibdry < num_bdries; ibdry++) {
    const int owner_cell = bdry_owner_cell[ibdry];
    const std::string& bdry_type = config->GetConfigValue(BDRY_TYPE(bdry_tag[ibdry]));
    if (bdry_type.find("Wall") != std::string::npos)
      wall_cells_.push_back(owner_cell);
  }
}
void LaplacianPolyShockFitWall::ComputeArtificialViscosity(const double* solution,
  const double dt) {
  static const int& order = DENEB_DATA->GetOrder();
  if (order == 0) return;
  static const int& num_cells = DENEB_DATA->GetNumCells();
  static const int& num_states = DENEB_EQUATION->GetNumStates();
  static const int& num_bases = DENEB_DATA->GetNumBases();
  static const int sb = num_states * num_bases;
  static const std::vector<double>& cell_volumes = DENEB_DATA->GetCellVolumes();
  static const std::vector<std::vector<double>>& cell_basis_value =
    DENEB_DATA->GetCellBasisValue();
  static int av_timer = 0;

  if (av_timer-- == 0) {
    SpotSuspect(solution);
    DBSCAN();
    ShockPolyFit();
    ComputeDistanceFromShock();

    av_timer = update_period_;
  }

  for (int icell = 0; icell < num_cells; icell++) {
    const double E0 = MaxArtificialViscosity(
      &solution[icell * sb], cell_volumes[icell], cell_basis_value[icell][0]);
    artificial_viscosity_[icell] = AVfunction(E0, distance_from_shock_[icell]);
  }

  for (int icell = 0; icell < wall_cells_.size(); icell++)
    artificial_viscosity_[wall_cells_[icell]] = wallAV_;

  communicate_->CommunicateBegin(&artificial_viscosity_[0]);
  communicate_->CommunicateEnd(&artificial_viscosity_[num_cells]);
}
// -------------------- SPID ------------------ //
SPID::SPID() { MASTER_MESSAGE(avocado::GetTitle("SPID")); }
void SPID::BuildData(void) {
  MASTER_MESSAGE(avocado::GetTitle("SPID::BuildData()"));

  target_state_ = 0;
  const std::vector<std::string>& variable_names =
      DENEB_EQUATION->GetCellVariableNames();
  MASTER_MESSAGE("Target state: " + variable_names[target_state_] + "\n");

  auto& config = AVOCADO_CONFIG;
  const std::string& equation = config->GetConfigValue(EQUATION);
  if (!equation.compare("Euler2D"))
    ERROR_MESSAGE("Artificial viscosity is not compatible with " + equation +
                  ", use NS equation with Re<0\n");
  if (!equation.compare("Advection2D")) {
    Ducros_switch_ = false;
  } else if (!equation.compare("Burgers2D")) {
    Ducros_switch_ = false;
  } else
    Ducros_switch_ = true;

  Sgain_ = std::stod(config->GetConfigValue(S_GAIN));
  Pgain_ = std::stod(config->GetConfigValue(P_GAIN));
  Igain_ = std::stod(config->GetConfigValue(I_GAIN));
  Dgain_ = std::stod(config->GetConfigValue(D_GAIN));
  MASTER_MESSAGE("Smoothness gain: " + std::to_string(Sgain_) + "\n");
  MASTER_MESSAGE("Proportional gain: " + std::to_string(Pgain_) + "\n");
  MASTER_MESSAGE("Integral gain: " + std::to_string(Igain_) + "\n");
  MASTER_MESSAGE("Derivative gain: " + std::to_string(Dgain_) + "\n");

  std::string temp = config->GetConfigValue(DUCROS_SWITCH);
  if (!temp.compare("off")) {
    Ducros_switch_ = false;
    MASTER_MESSAGE("Ducros switch turned off");
  }
  
  const int& num_outer_cells = DENEB_DATA->GetNumOuterCells();
  artificial_viscosity_.resize(num_outer_cells + 1, 0.0);

  const int& num_cells = DENEB_DATA->GetNumCells();
  cell_MLP_error_.resize(num_cells, 0.0);
  cell_BD_error_.resize(num_cells, 0.0);
  cell_error0_.resize(num_cells, 1.0e8);
  cell_error1_.resize(num_cells, 0.0);
  cell_integ_MLP_error_.resize(num_cells, 0.0);
  cell_integ_BD_error_.resize(num_cells, 0.0);

  const int& order = DENEB_DATA->GetOrder();
  if (order == 0) ERROR_MESSAGE("P0 doesn't require artificial viscosity");

  communicate_AV_ = std::make_shared<avocado::Communicate>(
      1, DENEB_DATA->GetOuterSendCellList(),
      DENEB_DATA->GetOuterRecvCellList());

  // hMLP & hMLP_BD build data
  BuildData_hMLPBD();

  // Ducros sensor build data
  quad_basis_value_.resize(num_cells);
  quad_basis_grad_value_.resize(num_cells);
  quad_weights_.resize(num_cells);
  std::vector<double> quad_points;
  std::vector<int> num_cell_points(num_cells);
  const int& num_bases = DENEB_DATA->GetNumBases();
  const int& dimension = DENEB_EQUATION->GetDimension();
  std::vector<int> pbd = {0, num_bases, dimension};
  for (int icell = 0; icell < num_cells; icell++) {
    DENEB_DATA->GetCellQuadrature(icell, order, quad_points,
                                  quad_weights_[icell]);
    DENEB_DATA->GetCellBasisValues(icell, quad_points,
                                   quad_basis_value_[icell]);
    DENEB_DATA->GetCellBasisGradValues(icell, quad_points,
                                       quad_basis_grad_value_[icell]);
    num_cell_points[icell] = quad_weights_[icell].size();
    pbd[0] = num_cell_points[icell];
    avocado::TensorTranspose(&quad_basis_grad_value_[icell][0], 3, "pbd",
                             &pbd[0], "pdb");
  }
  max_num_cell_points_ = 0;
  if (num_cell_points.size() != 0)
    max_num_cell_points_ =
        *std::max_element(num_cell_points.begin(), num_cell_points.end());

  // Anti-windup build data
  smooth_sensor_.resize(num_cells);
  const std::vector<double>& cell_volumes = DENEB_DATA->GetCellVolumes();
  integral_decay_rate_.resize(num_cells);
  for (int icell = 0; icell < num_cells; icell++) {
    integral_decay_rate_[icell] =
        std::pow(cell_volumes[icell], -1.0 / static_cast<double>(dimension));
  }
}
void SPID::ComputeArtificialViscosity(const double* solution, const double dt) {
  static const int& order = DENEB_DATA->GetOrder();
  if (order == 0) return;
  VertexMinMax(solution);
  ComputeMlpError(solution);
  ComputeBdError(solution);
  ComputeDucros(solution);
  ComputeSmoothSensor(solution);

  const int& num_cells = DENEB_DATA->GetNumCells();
  for (int icell = 0; icell < num_cells; icell++) {
    cell_integ_MLP_error_[icell] =
        ReLU(cell_integ_MLP_error_[icell] +
             (cell_MLP_error_[icell] - integral_decay_rate_[icell] *
                                           smooth_sensor_[icell] *
                                           cell_integ_MLP_error_[icell]) *
                 dt);
    cell_integ_BD_error_[icell] =
        ReLU(cell_integ_BD_error_[icell] +
             (cell_BD_error_[icell] - integral_decay_rate_[icell] *
                                          smooth_sensor_[icell] *
                                          cell_integ_BD_error_[icell]) *
                 dt);

    const double P_term = Pgain_ * cell_error1_[icell];
    const double I_term =
        Igain_ * (cell_integ_MLP_error_[icell] + cell_integ_BD_error_[icell]);
    const double D_term =
        Dgain_ * ReLU(cell_error1_[icell] - cell_error0_[icell]);

    artificial_viscosity_[icell] = ReLU(P_term + I_term + D_term);
    cell_error0_[icell] = cell_error1_[icell];
  }

  communicate_AV_->CommunicateBegin(&artificial_viscosity_[0]);
  communicate_AV_->CommunicateEnd(&artificial_viscosity_[num_cells]);
}
void SPID::ComputeArtificialViscosity(const double* solution,
                                      const std::vector<double>& local_dt) {
  static const int& order = DENEB_DATA->GetOrder();
  if (order == 0) return;
  VertexMinMax(solution);
  ComputeMlpError(solution);
  ComputeBdError(solution);
  ComputeDucros(solution);
  ComputeSmoothSensor(solution);

  const int& num_cells = DENEB_DATA->GetNumCells();
  for (int icell = 0; icell < num_cells; icell++) {
    cell_integ_MLP_error_[icell] =
        ReLU(cell_integ_MLP_error_[icell] +
             (cell_MLP_error_[icell] - integral_decay_rate_[icell] *
                                           smooth_sensor_[icell] *
                                           cell_integ_MLP_error_[icell]) *
                 local_dt[icell]);
    cell_integ_BD_error_[icell] =
        ReLU(cell_integ_BD_error_[icell] +
             (cell_BD_error_[icell] - integral_decay_rate_[icell] *
                                          smooth_sensor_[icell] *
                                          cell_integ_BD_error_[icell]) *
                 local_dt[icell]);

    const double P_term = Pgain_ * cell_error1_[icell];
    const double I_term =
        Igain_ * (cell_integ_MLP_error_[icell] + cell_integ_BD_error_[icell]);
    const double D_term =
        Dgain_ * ReLU(cell_error1_[icell] - cell_error0_[icell]);

    artificial_viscosity_[icell] = ReLU(P_term + I_term + D_term);
    cell_error0_[icell] = cell_error1_[icell];
  }

  communicate_AV_->CommunicateBegin(&artificial_viscosity_[0]);
  communicate_AV_->CommunicateEnd(&artificial_viscosity_[num_cells]);
}
void SPID::BuildData_hMLPBD() {
  SYNCRO();
  START_TIMER_TAG("BuildData_hMLPBD");
  MASTER_MESSAGE(avocado::GetTitle("SPID::BuildData_hMLPBD()"));

  START_TIMER();
  MASTER_MESSAGE("Constructing cell to cells data... ");
  ConstructCellCells();
  SYNCRO();
  MASTER_MESSAGE("Complete. (Time: " + std::to_string(STOP_TIMER()) + "s)\n");

  START_TIMER();
  MASTER_MESSAGE("Constructing node to cells data... ");
  ConstructNodeCells();
  SYNCRO();
  MASTER_MESSAGE("Complete. (Time: " + std::to_string(STOP_TIMER()) + "s)\n");

  START_TIMER();
  MASTER_MESSAGE("Constructing cell to faces data... ");
  {
    const int& num_cells = DENEB_DATA->GetNumCells();
    const int& num_faces = DENEB_DATA->GetNumFaces();
    const int& num_inner_faces = DENEB_DATA->GetNumInnerFaces();
    const auto& face_owner_cell = DENEB_DATA->GetFaceOwnerCell();
    const auto& face_neighbor_cell = DENEB_DATA->GetFaceNeighborCell();
    cell_faces_.resize(num_cells);
    for (int iface = 0; iface < num_faces; iface++)
      cell_faces_[face_owner_cell[iface]].push_back(iface);
    for (int iface = 0; iface < num_inner_faces; iface++)
      cell_faces_[face_neighbor_cell[iface]].push_back(iface);
  }
  SYNCRO();
  MASTER_MESSAGE("Complete. (Time: " + std::to_string(STOP_TIMER()) + "s)\n");

  START_TIMER();
  MASTER_MESSAGE("Constructing hMLP data... ");
  const int& order = DENEB_DATA->GetOrder();
  num_bases_list_.resize(order + 1);
  for (int i = 0; i < order + 1; i++)
    num_bases_list_[i] = DENEB_DATA->GetNumBases(i);

  const int& num_nodes = DENEB_DATA->GetNumNodes();
  const std::vector<int>& cell_node_ind = DENEB_DATA->GetCellNodeInd();
  const std::vector<int>& cell_node_ptr = DENEB_DATA->GetCellNodePtr();
  nodetypes_.resize(num_nodes, NODETYPE::NORMAL);
  const int& num_bdries = DENEB_DATA->GetNumBdries();
  const auto& cell_element = DENEB_DATA->GetCellElement();
  const std::vector<int>& bdry_owner_cell = DENEB_DATA->GetBdryOwnerCell();
  const std::vector<int>& bdry_owner_type = DENEB_DATA->GetBdryOwnerType();
  for (int ibdry = 0; ibdry < num_bdries; ibdry++) {
    const int& owner_cell = bdry_owner_cell[ibdry];
    const int& owner_type = bdry_owner_type[ibdry];

    std::vector<int> bdry_nodes =
        cell_element[owner_cell]->GetFacetypeNodes(1)[owner_type];
    const int& start_index = cell_node_ptr[owner_cell];
    for (auto&& inode : bdry_nodes) inode = cell_node_ind[start_index + inode];
    for (auto&& inode : bdry_nodes) nodetypes_[inode] = NODETYPE::BOUNDARY;
  }

  const int& num_cells = DENEB_DATA->GetNumCells();
  const int& num_total_cells = DENEB_DATA->GetNumTotalCells();
  const int& num_states = DENEB_EQUATION->GetNumStates();
  const int& num_bases = DENEB_DATA->GetNumBases();
  const int sb = num_states * num_bases;
  foreign_solution_.resize(std::max((num_total_cells - num_cells) * sb, 1));
  communicate_MLP_ = std::make_shared<avocado::Communicate>(
      sb, DENEB_DATA->GetTotalSendCellList(),
      DENEB_DATA->GetTotalRecvCellList());

  const int& dimension = DENEB_EQUATION->GetDimension();
  const std::vector<double>& node_coords = DENEB_DATA->GetNodeCoords();
  cell_vertex_basis_value_.resize(num_cells);
  for (int icell = 0; icell < num_cells; icell++) {
    const int num_vertex = cell_node_ptr[icell + 1] - cell_node_ptr[icell];

    std::vector<double> coords;
    for (int v = 0; v < num_vertex; v++) {
      const int& node = cell_node_ind[cell_node_ptr[icell] + v];
      const double* coord = &node_coords[node * dimension];
      for (int idim = 0; idim < dimension; idim++)
        coords.push_back(coord[idim]);
    }
    DENEB_DATA->GetCellBasisValues(icell, coords,
                                   cell_vertex_basis_value_[icell]);
  }

  foreign_cell_basis_value_.resize(std::max(num_total_cells - num_cells, 1));
  {
    std::shared_ptr<avocado::Communicate> communicate =
        std::make_shared<avocado::Communicate>(
            1, DENEB_DATA->GetTotalSendCellList(),
            DENEB_DATA->GetTotalRecvCellList());
    std::vector<double> cell_basis_value(num_cells);
    for (int icell = 0; icell < num_cells; icell++)
      cell_basis_value[icell] = cell_vertex_basis_value_[icell][0];
    communicate->CommunicateBegin(&cell_basis_value[0]);
    communicate->CommunicateEnd(&foreign_cell_basis_value_[0]);
  }

  cell_average_.resize(num_total_cells);
  vertex_min_.resize(num_nodes);
  vertex_max_.resize(num_nodes);
  SYNCRO();
  MASTER_MESSAGE("Complete. (Time: " + std::to_string(STOP_TIMER()) + "s)\n");

  START_TIMER();
  MASTER_MESSAGE("Constructing face quadrature data... ");
  DENEB_DATA->BuildFaceQuadData();
  const int& num_faces = DENEB_DATA->GetNumFaces();
  const std::vector<double>& face_area = DENEB_DATA->GetFaceArea();
  face_characteristic_length_.resize(num_faces);
  cell_face_characteristic_length_.resize(num_cells);
  for (int iface = 0; iface < num_faces; iface++) {
    double length = face_area[iface];
    if (dimension == 3) length = std::sqrt(length);
    face_characteristic_length_[iface] = length;
  }
  for (int icell = 0; icell < num_cells; icell++) {
    const int num_local_faces = cell_faces_[icell].size();
    cell_face_characteristic_length_[icell] = 0.0;
    for (int iface = 0; iface < num_local_faces; iface++)
      cell_face_characteristic_length_[icell] +=
          face_characteristic_length_[iface];
  }
  SYNCRO();
  MASTER_MESSAGE("Complete. (Time: " + std::to_string(STOP_TIMER()) + "s)\n");
}
void SPID::ConstructCellCells() {
  const std::vector<int>& face_owner_cell = DENEB_DATA->GetFaceOwnerCell();
  const std::vector<int>& face_neighbor_cell =
      DENEB_DATA->GetFaceNeighborCell();

  const int& num_cells = DENEB_DATA->GetNumCells();
  const int& num_faces = DENEB_DATA->GetNumFaces();
  const int& num_inner_faces = DENEB_DATA->GetNumInnerFaces();
  std::vector<std::vector<int>> cell_cells(num_cells);
  for (int i = 0; i < num_inner_faces; i++) {
    const int& owner_cell = face_owner_cell[i];
    const int& neighbor_cell = face_neighbor_cell[i];

    cell_cells[owner_cell].push_back(neighbor_cell);
    cell_cells[neighbor_cell].push_back(owner_cell);
  }
  for (int i = num_inner_faces; i < num_faces; i++) {
    const int& owner_cell = face_owner_cell[i];
    const int& neighbor_cell = face_neighbor_cell[i];

    cell_cells[owner_cell].push_back(neighbor_cell + num_cells);
  }

  cell_cells_ = std::move(cell_cells);
}
void SPID::ConstructNodeCells() {
  const std::vector<int>& node_global_index = DENEB_DATA->GetNodeGlobalIndex();
  std::unordered_map<int, int> node_mapping;  // global to local index
  {
    int ind = 0;
    for (auto&& global_index : node_global_index)
      node_mapping[global_index] = ind++;
  }

  const std::unordered_map<int, std::vector<int>>&
      periodic_matching_global_node_index =
          DENEB_DATA->GetPeriodicMatchingGlobalNodeIndex();
  const auto& node_grps = periodic_matching_global_node_index;

  const int& num_nodes = DENEB_DATA->GetNumNodes();
  const int& num_cells = DENEB_DATA->GetNumCells();
  const int& num_total_cells = DENEB_DATA->GetNumTotalCells();
  const std::vector<int>& cell_node_ptr = DENEB_DATA->GetCellNodePtr();
  const std::vector<int>& cell_node_ind = DENEB_DATA->GetCellNodeInd();
  std::vector<std::vector<int>> node_cells;
  {
    std::unordered_map<int, std::unordered_set<int>> node_cells_temp;
    for (int i = 0; i < num_total_cells; i++)
      for (int ptr = cell_node_ptr[i]; ptr < cell_node_ptr[i + 1]; ptr++)
        node_cells_temp[cell_node_ind[ptr]].insert(i);

    for (auto&& iterator : node_grps) {
      const int& mynode = node_mapping.at(iterator.first);

      std::unordered_set<int> new_cells;
      for (auto&& node : iterator.second) {
        const auto& cells = node_cells_temp.find(node_mapping.at(node));
        if (cells == node_cells_temp.end()) continue;
        new_cells.insert(cells->second.begin(), cells->second.end());
      }
      node_cells_temp[mynode].insert(new_cells.begin(), new_cells.end());
    }

    node_cells.resize(num_nodes);
    for (auto&& iterator : node_cells_temp) {
      if (iterator.first >= num_nodes) continue;
      node_cells[iterator.first] =
          std::vector<int>(iterator.second.begin(), iterator.second.end());
    }
  }

  std::vector<std::vector<int>> node_vertices(num_nodes);
  for (int i = 0; i < num_nodes; i++) {
    const auto& matching_nodes = node_grps.find(node_global_index[i]);
    for (auto&& cell : node_cells[i]) {
      for (int ptr = cell_node_ptr[cell]; ptr < cell_node_ptr[cell + 1];
           ptr++) {
        if (cell_node_ind[ptr] == i) {
          node_vertices[i].push_back(ptr - cell_node_ptr[cell]);
          break;
        }
        if (matching_nodes == node_grps.end()) continue;
        const int& node = node_global_index[cell_node_ind[ptr]];
        if (std::find(matching_nodes->second.begin(),
                      matching_nodes->second.end(),
                      node) != matching_nodes->second.end()) {
          node_vertices[i].push_back(ptr - cell_node_ptr[cell]);
          break;
        }
      }
    }
  }

  node_cells_ = std::move(node_cells);
  node_vertices_ = std::move(node_vertices);
}
void SPID::VertexMinMax(const double* solution) {
  static const int& num_cells = DENEB_DATA->GetNumCells();
  static const int& num_states = DENEB_EQUATION->GetNumStates();
  static const int& num_bases = DENEB_DATA->GetNumBases();
  const int sb = num_states * num_bases;

  communicate_MLP_->CommunicateBegin(solution);
  for (int icell = 0; icell < num_cells; icell++)
    cell_average_[icell] = solution[icell * sb + target_state_ * num_bases] *
                           cell_vertex_basis_value_[icell][0];

  static const int& num_total_cells = DENEB_DATA->GetNumTotalCells();
  communicate_MLP_->CommunicateEnd(&foreign_solution_[0]);
  for (int icell = num_cells; icell < num_total_cells; icell++)
    cell_average_[icell] = foreign_solution_[(icell - num_cells) * sb +
                                             target_state_ * num_bases] *
                           foreign_cell_basis_value_[icell - num_cells];

  static const int& num_nodes = DENEB_DATA->GetNumNodes();
  for (int inode = 0; inode < num_nodes; inode++) {
    std::vector<double> avgs;
    for (auto&& icell : node_cells_[inode])
      avgs.push_back(cell_average_[icell]);

    vertex_min_[inode] = *std::min_element(avgs.begin(), avgs.end());
    vertex_max_[inode] = *std::max_element(avgs.begin(), avgs.end());
  }
}

void SPID::ComputeMlpError(const double* solution) {
  static const int& num_states = DENEB_EQUATION->GetNumStates();
  static const int& num_basis = DENEB_DATA->GetNumBases();
  static const int& num_cells = DENEB_DATA->GetNumCells();
  static const int sb = num_states * num_basis;
  static const std::vector<int>& cell_node_ptr = DENEB_DATA->GetCellNodePtr();
  static const std::vector<int>& cell_node_ind = DENEB_DATA->GetCellNodeInd();

  for (int icell = 0; icell < num_cells; icell++) {
    const int num_vertex = cell_node_ptr[icell + 1] - cell_node_ptr[icell];
    const double average_value =
        solution[icell * sb + target_state_ * num_basis] *
        cell_vertex_basis_value_[icell][0];

    double cell_error = 0.0;
    for (int ivertex = 0.0; ivertex < num_vertex; ivertex++) {
      const int node = cell_node_ind[cell_node_ptr[icell] + ivertex];
      if (nodetypes_[node] == NODETYPE::BOUNDARY) continue;

      const double P1var = cblas_ddot(
          num_bases_list_[1], &solution[icell * sb + target_state_ * num_basis],
          1, &cell_vertex_basis_value_[icell][ivertex * num_basis], 1);

      cell_error += std::min(ReLU(P1var - vertex_max_[node]) +
                                 ReLU(vertex_min_[node] - P1var),
                             (vertex_max_[node] - vertex_min_[node])) /
                    vertex_min_[node];
    }
    cell_error /= static_cast<double>(num_vertex);

    cell_error1_[icell] = cell_error;
    cell_MLP_error_[icell] = cell_error;
  }
}

void SPID::ComputeBdError(const double* solution) {
  static const int& num_states = DENEB_EQUATION->GetNumStates();
  static const int& num_bases = DENEB_DATA->GetNumBases();
  static const int& num_cells = DENEB_DATA->GetNumCells();
  static const int sb = num_states * num_bases;

  // hMLP parameters
  static const std::vector<int>& cell_node_ptr = DENEB_DATA->GetCellNodePtr();
  static const std::vector<int>& cell_node_ind = DENEB_DATA->GetCellNodeInd();

  // hMLP_BD parameters
  static const int& num_faces = DENEB_DATA->GetNumFaces();
  static const int& num_inner_faces = DENEB_DATA->GetNumInnerFaces();
  static const std::vector<int>& num_face_quad_points =
      DENEB_DATA->GetNumFaceQuadPoints();
  static const std::vector<int>& face_owner_cell =
      DENEB_DATA->GetFaceOwnerCell();
  static const std::vector<int>& face_neighbor_cell =
      DENEB_DATA->GetFaceNeighborCell();
  static const std::vector<double>& face_area = DENEB_DATA->GetFaceArea();
  static const std::vector<std::vector<double>>& face_owner_basis_value =
      DENEB_DATA->GetFaceQuadOwnerBasisValue();
  static const std::vector<std::vector<double>>& face_neighbor_basis_value =
      DENEB_DATA->GetFaceQuadNeighborBasisValue();
  static const std::vector<std::vector<double>>& face_quad_weights =
      DENEB_DATA->GetFaceQuadWeights();

  // Compute face diffenence
  std::vector<double> face_difference(num_faces);
  for (int iface = 0; iface < num_inner_faces; iface++) {
    const int num_points = num_face_quad_points[iface];
    const int owner_cell = face_owner_cell[iface];
    const int neighbor_cell = face_neighbor_cell[iface];

    double state_sum = 0.0;
    for (int ipoint = 0; ipoint < num_points; ipoint++) {
      const double owner_solution = cblas_ddot(
          num_bases, &solution[owner_cell * sb + target_state_ * num_bases], 1,
          &face_owner_basis_value[iface][ipoint * num_bases], 1);
      const double owner_avg_solution = cell_average_[owner_cell];
      const double neighbor_solution = cblas_ddot(
          num_bases, &solution[neighbor_cell * sb + target_state_ * num_bases],
          1, &face_neighbor_basis_value[iface][ipoint * num_bases], 1);
      const double neighbor_avg_solution = cell_average_[neighbor_cell];

      state_sum += std::abs(owner_solution - neighbor_solution) *
                   face_quad_weights[iface][ipoint] /
                   std::min(owner_avg_solution, neighbor_avg_solution);
    }

    face_difference[iface] = state_sum / face_area[iface];
  }
  for (int iface = num_inner_faces; iface < num_faces; iface++) {
    const int num_points = num_face_quad_points[iface];
    const int owner_cell = face_owner_cell[iface];
    const int neighbor_cell = face_neighbor_cell[iface];

    double state_sum = 0.0;
    for (int ipoint = 0; ipoint < num_points; ipoint++) {
      const double owner_solution = cblas_ddot(
          num_bases, &solution[owner_cell * sb + target_state_ * num_bases], 1,
          &face_owner_basis_value[iface][ipoint * num_bases], 1);
      const double owner_avg_solution = cell_average_[owner_cell];
      const double neighbor_solution = cblas_ddot(
          num_bases,
          &foreign_solution_[neighbor_cell * sb + target_state_ * num_bases], 1,
          &face_neighbor_basis_value[iface][ipoint * num_bases], 1);
      const double neighbor_avg_solution = cell_average_[neighbor_cell + num_cells];

      state_sum += std::abs(owner_solution - neighbor_solution) *
                   face_quad_weights[iface][ipoint] /
                   std::min(owner_avg_solution, neighbor_avg_solution);
    }

    face_difference[iface] = state_sum / face_area[iface];
  }

  // Compute BD error
  for (int icell = 0; icell < num_cells; icell++) {
    const int num_local_faces = cell_faces_[icell].size();
    double cell_BD_error = 0.0;
    for (int iface = 0; iface < num_local_faces; iface++)
      cell_BD_error += face_difference[cell_faces_[icell][iface]];

    cell_BD_error -= cell_face_characteristic_length_[icell];
    cell_BD_error /= static_cast<double>(num_local_faces);

    cell_BD_error_[icell] = ReLU(cell_BD_error);
    cell_error1_[icell] += cell_BD_error_[icell];
  }
}
void SPID::ComputeDucros(const double* solution) {
  static const int& num_states = DENEB_EQUATION->GetNumStates();
  static const int& num_bases = DENEB_DATA->GetNumBases();
  static const int& num_cells = DENEB_DATA->GetNumCells();
  static const int& dimension = DENEB_EQUATION->GetDimension();
  static const int sb = num_states * num_bases;
  static const int sd = num_states * dimension;
  static const std::vector<double>& cell_volumes = DENEB_DATA->GetCellVolumes();
  static std::vector<double> local_solution(num_states * max_num_cell_points_,
                                            0.0);
  static std::vector<double> local_solution_grad(
      num_states * dimension * max_num_cell_points_, 0.0);

  if (!Ducros_switch_) return;
  for (int icell = 0; icell < num_cells; icell++) {
    const int num_points = quad_weights_[icell].size();

    double Ducros_value = 0.0;
    avocado::Kernel0::f4(&solution[icell * sb], &quad_basis_value_[icell][0],
                         &local_solution[0], num_states, num_bases, num_points,
                         1.0, 0.0);
    avocado::Kernel1::f42(&quad_basis_grad_value_[icell][0],
                          &solution[icell * sb], &local_solution_grad[0],
                          dimension, num_points, num_bases, num_states, 1.0,
                          0.0);

    for (int ipoint = 0; ipoint < num_points; ipoint++) {
      const double dilatation =
          ComputeDilatation(&local_solution[ipoint * num_states],
                            &local_solution_grad[ipoint * sd]);
      const double vorticity =
          ComputeVorticityMagnitude(&local_solution[ipoint * num_states],
                                    &local_solution_grad[ipoint * sd]);

      const double dilatation2 = dilatation * dilatation;
      const double vorticity2 = vorticity * vorticity;
      Ducros_value += dilatation2 /
                      (dilatation2 + vorticity2 + ducros_nonzero_eps_) *
                      quad_weights_[icell][ipoint];
    }
    Ducros_value /= cell_volumes[icell];

    cell_error1_[icell] *= Ducros_value;
    cell_MLP_error_[icell] *= Ducros_value;
    cell_BD_error_[icell] *= Ducros_value;
  }
}
void SPID::ComputeSmoothSensor(const double* solution) {
  static const int& order = DENEB_DATA->GetOrder();
  static const int& num_states = DENEB_EQUATION->GetNumStates();
  static const int& num_bases = DENEB_DATA->GetNumBases();
  static const int& num_cells = DENEB_DATA->GetNumCells();
  static const int sb = num_states * num_bases;

  if (order == 0) return;

  for (int icell = 0; icell < num_cells; icell++) {
    const double* solution_ptr_start =
        solution + target_state_ * num_bases + icell * sb;
    const int num_basis_1_ = num_bases_list_[order - 1];
    const double sum0 =
        cblas_ddot(num_basis_1_, solution_ptr_start, 1, solution_ptr_start, 1);
    const double sum1 =
        cblas_ddot(num_bases - num_basis_1_, solution_ptr_start + num_basis_1_,
                   1, solution_ptr_start + num_basis_1_, 1);
    const double sum = sum0 + sum1;
    const double x = sum1 / sum;

    smooth_sensor_[icell] = 1 - std::pow(x, Sgain_);
  }  
}
double SPID::ComputeDilatation(const double* solution, const double* solution_grad) {
  static const int& dimension = DENEB_EQUATION->GetDimension();

  if (dimension == 2) {
    int ind = 0;
    const double d = solution[ind++];
    const double du = solution[ind++];
    const double dv = solution[ind++];

    ind = 0;
    const double dx = solution_grad[ind++];
    const double dux = solution_grad[ind++];
    const double dvx = solution_grad[ind++];
    ind++;
    const double dy = solution_grad[ind++];
    const double duy = solution_grad[ind++];
    const double dvy = solution_grad[ind++];

    const double d_inv = 1.0 / d;
    const double u = du * d_inv;
    const double v = dv * d_inv;
    const double ux = (dux - dx * u) * d_inv;
    const double vy = (dvy - dy * v) * d_inv;

    return ux + vy;
  } else if (dimension == 3) {
    int ind = 0;
    const double d = solution[ind++];
    const double du = solution[ind++];
    const double dv = solution[ind++];
    const double dw = solution[ind++];

    ind = 0;
    const double dx = solution_grad[ind++];
    const double dux = solution_grad[ind++];
    const double dvx = solution_grad[ind++];
    const double dwx = solution_grad[ind++];
    ind++;

    const double dy = solution_grad[ind++];
    const double duy = solution_grad[ind++];
    const double dvy = solution_grad[ind++];
    const double dwy = solution_grad[ind++];
    ind++;

    const double dz = solution_grad[ind++];
    const double duz = solution_grad[ind++];
    const double dvz = solution_grad[ind++];
    const double dwz = solution_grad[ind++];
    ind++;

    const double d_inv = 1.0 / d;
    const double u = du * d_inv;
    const double v = dv * d_inv;
    const double w = dw * d_inv;
    const double ux = (dux - dx * u) * d_inv;
    const double vy = (dvy - dy * v) * d_inv;
    const double wz = (dwz - dz * w) * d_inv;

    return ux + vy + wz;
  } else
    ERROR_MESSAGE("Dimension error\n");
}

double SPID::ComputeVorticityMagnitude(const double* solution,
                                       const double* solution_grad) {
  static const int& dimension = DENEB_EQUATION->GetDimension();

  if (dimension == 2) {
    int ind = 0;
    const double d = solution[ind++];
    const double du = solution[ind++];
    const double dv = solution[ind++];

    ind = 0;
    const double dx = solution_grad[ind++];
    const double dux = solution_grad[ind++];
    const double dvx = solution_grad[ind++];
    ind++;
    const double dy = solution_grad[ind++];
    const double duy = solution_grad[ind++];
    const double dvy = solution_grad[ind++];

    const double d_inv = 1.0 / d;
    const double u = du * d_inv;
    const double v = dv * d_inv;
    const double uy = (duy - dy * u) * d_inv;
    const double vx = (dvx - dx * v) * d_inv;

    return vx - uy;
  } else if (dimension == 3) {
    int ind = 0;
    const double d = solution[ind++];
    const double du = solution[ind++];
    const double dv = solution[ind++];
    const double dw = solution[ind++];

    ind = 0;
    const double dx = solution_grad[ind++];
    const double dux = solution_grad[ind++];
    const double dvx = solution_grad[ind++];
    const double dwx = solution_grad[ind++];
    ind++;

    const double dy = solution_grad[ind++];
    const double duy = solution_grad[ind++];
    const double dvy = solution_grad[ind++];
    const double dwy = solution_grad[ind++];
    ind++;

    const double dz = solution_grad[ind++];
    const double duz = solution_grad[ind++];
    const double dvz = solution_grad[ind++];
    const double dwz = solution_grad[ind++];
    ind++;

    const double d_inv = 1.0 / d;
    const double u = du * d_inv;
    const double v = dv * d_inv;
    const double w = dw * d_inv;
    const double ux = (dux - dx * u) * d_inv;
    const double uy = (duy - dy * u) * d_inv;
    const double uz = (duz - dz * u) * d_inv;
    const double vx = (dvx - dx * v) * d_inv;
    const double vy = (dvy - dy * v) * d_inv;
    const double vz = (dvz - dz * v) * d_inv;
    const double wx = (dwx - dx * w) * d_inv;
    const double wy = (dwy - dy * w) * d_inv;
    const double wz = (dwz - dz * w) * d_inv;

    const double Vorx = wy - vz;
    const double Vory = uz - wx;
    const double Vorz = vx - uy;

    return std::sqrt(Vorx * Vorx + Vory * Vory + Vorz * Vorz);
  } else
    ERROR_MESSAGE("Dimension error\n");
}
}  // namespace deneb