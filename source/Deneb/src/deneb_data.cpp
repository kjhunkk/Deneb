#include "deneb_data.h"

#include <algorithm>
#include <string>

#include "avocado.h"
#include "deneb_DRM.h"
#include "deneb_basis.h"
#include "deneb_config_macro.h"
#include "deneb_equation.h"
#include "deneb_grid_builder.h"
#include "deneb_jacobian.h"
#include "deneb_quadrature.h"
#include "deneb_timescheme.h"

#define MOVE_DATA(data) data = std::move(grid->data)

namespace deneb {
std::shared_ptr<Data> DENEB_DATA_NAME = nullptr;
Data::Data() {
  MASTER_MESSAGE(avocado::GetTitle("Data"));

  auto& config = AVOCADO_CONFIG;
  order_ = std::stoi(config->GetConfigValue(ORDER));
  volume_flux_order_ = std::stoi(config->GetConfigValue(VOLUME_FLUX_ORDER));
  surface_flux_order_ = std::stoi(config->GetConfigValue(SURFACE_FLUX_ORDER));
  MASTER_MESSAGE("Order = " + std::to_string(order_) + "\n");
  MASTER_MESSAGE("Volume flux order = " + std::to_string(volume_flux_order_) +
                 "\n");
  MASTER_MESSAGE("Surface flux order = " + std::to_string(surface_flux_order_) +
                 "\n");
  if (config->IsConfigValue(CHARACTERISTIC_LENGTH))
    lc_ =
        std::stod(config->GetConfigValue(CHARACTERISTIC_LENGTH));
  else lc_ = 1.0;
};
Data::~Data(){};

void Data::BuildData(void) {
  has_source_term_ = DENEB_EQUATION->GetSourceTerm();
  has_mass_matrix_ = DENEB_EQUATION->GetMassMatrixFlag();

  dimension_ = DENEB_EQUATION->GetDimension();
  std::shared_ptr<GridBuilder> grid = std::make_shared<GridBuilder>(dimension_);
  grid->BuildGrid();

  SYNCRO();
  START_TIMER_TAG("BuildData");
  MASTER_MESSAGE(avocado::GetTitle("Data::BuildData()"));

  num_bases_ = GetNumBases(order_);
  MASTER_MESSAGE("Number of bases = " + std::to_string(num_bases_) + "\n");

  // node
  START_TIMER();
  MASTER_MESSAGE("Building node data... ");
  MOVE_DATA(num_global_nodes_);
  MOVE_DATA(num_total_nodes_);
  MOVE_DATA(num_nodes_);
  MOVE_DATA(node_global_index_);
  MOVE_DATA(node_coords_);
  SYNCRO();
  MASTER_MESSAGE("Complete. (Time: " + std::to_string(STOP_TIMER()) + "s)\n");

  // cell
  START_TIMER();
  MASTER_MESSAGE("Building cell data... ");
  MOVE_DATA(num_global_cells_);
  MOVE_DATA(num_total_cells_);
  MOVE_DATA(num_outer_cells_);
  MOVE_DATA(num_cells_);
  MOVE_DATA(cell_global_index_);
  MOVE_DATA(cell_elemtype_);
  MOVE_DATA(cell_order_);
  MOVE_DATA(cell_node_ptr_);
  MOVE_DATA(cell_node_ind_);
  MOVE_DATA(cell_subnode_ptr_);
  MOVE_DATA(cell_subnode_ind_);

  cell_element_.resize(num_total_cells_);
  cell_jacobians_.resize(num_total_cells_);
  cell_basis_.resize(num_total_cells_);
  num_cell_points_.resize(num_cells_);
  cell_volumes_.resize(num_total_cells_, 0.0);
  cell_proj_volumes_.resize(num_cells_ * dimension_, 0.0);
  cell_center_coords_.resize(num_total_cells_ * dimension_, 0.0);
  cell_points_.resize(num_cells_);
  cell_basis_value_.resize(num_cells_);
  cell_basis_grad_value_.resize(num_cells_);
  cell_coefficients_.resize(num_cells_);
  if (has_source_term_) cell_source_coefficients_.resize(num_cells_);
  if (has_mass_matrix_) cell_mass_coefficients_.resize(num_cells_);

  for (int icell = 0; icell < num_total_cells_; icell++) BuildCellData(icell);
  SYNCRO();
  MASTER_MESSAGE("Complete. (Time: " + std::to_string(STOP_TIMER()) + "s)\n");

  // face
  START_TIMER();
  MASTER_MESSAGE("Building face data... ");
  MOVE_DATA(num_faces_);
  MOVE_DATA(num_inner_faces_);
  MOVE_DATA(face_owner_cell_);
  MOVE_DATA(face_neighbor_cell_);
  MOVE_DATA(face_owner_type_);
  MOVE_DATA(face_neighbor_type_);

  num_face_points_.resize(num_faces_);
  face_points_.resize(num_faces_);
  face_normals_.resize(num_faces_);
  face_owner_basis_value_.resize(num_faces_);
  face_neighbor_basis_value_.resize(num_faces_);
  face_owner_basis_grad_value_.resize(num_faces_);
  face_neighbor_basis_grad_value_.resize(num_faces_);
  face_owner_coefficients_.resize(num_faces_);
  face_neighbor_coefficients_.resize(num_faces_);

  for (int iface = 0; iface < num_faces_; iface++) BuildFaceData(iface);
  SYNCRO();
  MASTER_MESSAGE("Complete. (Time: " + std::to_string(STOP_TIMER()) + "s)\n");

  // boundary
  START_TIMER();
  MASTER_MESSAGE("Building boundary data... ");
  MOVE_DATA(num_global_bdries_);
  MOVE_DATA(num_bdries_);
  MOVE_DATA(bdry_global_index_);
  MOVE_DATA(bdry_tag_);
  MOVE_DATA(bdry_owner_cell_);
  MOVE_DATA(bdry_owner_type_);

  num_bdry_points_.resize(num_bdries_);
  bdry_points_.resize(num_bdries_);
  bdry_normals_.resize(num_bdries_);
  bdry_owner_basis_value_.resize(num_bdries_);
  bdry_owner_basis_grad_value_.resize(num_bdries_);
  bdry_owner_coefficients_.resize(num_bdries_);

  for (int ibdry = 0; ibdry < num_bdries_; ibdry++) BuildBdryData(ibdry);
  SYNCRO();
  MASTER_MESSAGE("Complete. (Time: " + std::to_string(STOP_TIMER()) + "s)\n");

  // periodic boundary
  START_TIMER();
  MASTER_MESSAGE("Building periodic boundary data...");
  MOVE_DATA(num_global_peribdries_);
  MOVE_DATA(num_total_peribdries_);
  MOVE_DATA(num_peribdries_);
  MOVE_DATA(num_inner_peribdries_);
  MOVE_DATA(peribdry_global_index_);
  MOVE_DATA(peribdry_tag_);
  MOVE_DATA(peribdry_owner_cell_);
  MOVE_DATA(peribdry_neighbor_cell_);
  MOVE_DATA(peribdry_owner_type_);
  MOVE_DATA(peribdry_neighbor_type_);

  num_peribdry_points_.resize(num_peribdries_);
  peribdry_points_.resize(num_peribdries_);
  peribdry_normals_.resize(num_peribdries_);
  peribdry_owner_basis_value_.resize(num_peribdries_);
  peribdry_neighbor_basis_value_.resize(num_peribdries_);
  peribdry_owner_basis_grad_value_.resize(num_peribdries_);
  peribdry_neighbor_basis_grad_value_.resize(num_peribdries_);
  peribdry_owner_coefficients_.resize(num_peribdries_);
  peribdry_neighbor_coefficients_.resize(num_peribdries_);

  for (int ibdry = 0; ibdry < num_peribdries_; ibdry++)
    BuildPeribdryData(ibdry);
  SYNCRO();
  MASTER_MESSAGE("Complete. (Time: " + std::to_string(STOP_TIMER()) + "s)\n");

  MOVE_DATA(periodic_matching_global_node_index_);

  MOVE_DATA(outer_send_cell_list_);
  MOVE_DATA(outer_recv_cell_list_);
  MOVE_DATA(total_send_cell_list_);
  MOVE_DATA(total_recv_cell_list_);
  grid.reset();

  START_TIMER();
  MASTER_MESSAGE("Merging periodic boundary and face data... ");
  CombinePeribdryToFaceData();
  SYNCRO();
  MASTER_MESSAGE("Complete. (Time: " + std::to_string(STOP_TIMER()) + "s)\n");

  if (DENEB_TIMESCHEME->IsImplicit()) {
    START_TIMER();
    MASTER_MESSAGE("Building data for implicit method...");
    BuildImplicitData();
    SYNCRO();
    MASTER_MESSAGE("Complete. (Time: " + std::to_string(STOP_TIMER()) + "s)\n");
  }

  START_TIMER();
  MASTER_MESSAGE("Transposing data... ");
  // ---------- default index ---------- //
  // face_owner_basis_value_ : pb
  // face_owner_div_basis_value_ : pbd
  // face_owner_coefficients_ : pbd
  // cell_basis_value_ : pb
  // cell_div_basis_value_ : pbd
  // cell_coefficients_ : pbd
  // cell_source_coefficients_ : pb
  // ----------------------------------- //
  std::vector<int> pbd = {0, num_bases_, dimension_};
  for (int iface = 0; iface < num_faces_; iface++) {
    pbd[0] = num_face_points_[iface];
    avocado::TensorTranspose(&face_owner_basis_grad_value_[iface][0], 3, "pbd",
                             &pbd[0], "pdb");
    avocado::TensorTranspose(&face_neighbor_basis_grad_value_[iface][0], 3,
                             "pbd", &pbd[0], "pdb");
    avocado::TensorTranspose(&face_owner_coefficients_[iface][0], 3, "pbd",
                             &pbd[0], "pdb");
    avocado::TensorTranspose(&face_neighbor_coefficients_[iface][0], 3, "pbd",
                             &pbd[0], "pdb");
  }
  for (int icell = 0; icell < num_cells_; icell++) {
    pbd[0] = num_cell_points_[icell];
    avocado::TensorTranspose(&cell_basis_grad_value_[icell][0], 3, "pbd",
                             &pbd[0], "pdb");
    avocado::TensorTranspose(&cell_coefficients_[icell][0], 3, "pbd", &pbd[0],
                             "pdb");
  }
  for (int ibdry = 0; ibdry < num_bdries_; ibdry++) {
    pbd[0] = num_bdry_points_[ibdry];
    avocado::TensorTranspose(&bdry_owner_basis_grad_value_[ibdry][0], 3, "pbd",
                             &pbd[0], "pdb");
    avocado::TensorTranspose(&bdry_owner_coefficients_[ibdry][0], 3, "pbd",
                             &pbd[0], "pdb");
  }
  SYNCRO();
  MASTER_MESSAGE("Complete. (Time: " + std::to_string(STOP_TIMER()) + "s)\n");

  SYNCRO();
  STOP_TIMER_TAG("BuildData");
  MASTER_MESSAGE("Data::BuildData() completes. (Time: " +
                 std::to_string(GET_RECORD_TAG("BuildData")) + "s)\n");
}
void Data::BuildFaceQuadData(void) {
  const int num_pure_faces = num_faces_ - num_peribdries_;
  const int num_pure_inner_faces = num_inner_faces_ - num_inner_peribdries_;
  for (int i = num_inner_faces_; i < num_faces_; i++)
    face_neighbor_cell_[i] += num_cells_;

  const std::vector<int> rules = {0,
                                  0,
                                  num_pure_inner_faces,
                                  num_inner_peribdries_,
                                  num_pure_faces,
                                  num_peribdries_};

  SplitData(face_owner_cell_, peribdry_owner_cell_, rules);
  SplitData(face_neighbor_cell_, peribdry_neighbor_cell_, rules);
  SplitData(face_owner_type_, peribdry_owner_type_, rules);
  SplitData(face_neighbor_type_, peribdry_neighbor_type_, rules);

  num_face_quad_points_.resize(num_pure_faces);
  face_area_.resize(num_pure_faces);
  face_quad_owner_basis_value_.resize(num_pure_faces);
  face_quad_neighbor_basis_value_.resize(num_pure_faces);
  face_quad_weights_.resize(num_pure_faces);
  for (int iface = 0; iface < num_pure_faces; iface++) {
    const int owner_index = face_owner_cell_[iface];
    const int owner_type = face_owner_type_[iface];
    const int neighbor_index = face_neighbor_cell_[iface];

    const std::shared_ptr<Jacobian> owner_cell_jacobian =
        cell_jacobians_[owner_index];
    const Element* owner_cell_element = cell_element_[owner_index];
    const ElemType face_elemtype =
        owner_cell_element->GetFaceElemtype(owner_type);
    const int& elemorder = cell_order_[owner_index];

    int num_flux_bases;
    std::shared_ptr<DRMSurface> drm = DRMSurface::GetDRM(face_elemtype);
    std::shared_ptr<BasisSurface> flux_basis = drm->GetApproximateBasis();
    {
      std::vector<int> face_nodes =
          owner_cell_element->GetFacetypeNodes(1)[owner_type];
      const int& start_index = cell_node_ptr_[owner_index];
      for (auto&& inode : face_nodes)
        inode = cell_node_ind_[start_index + inode];
      const int num_nodes = static_cast<int>(face_nodes.size());

      std::vector<double> coords;
      coords.resize(num_nodes * dimension_);
      avocado::VecCopy(num_nodes, &face_nodes[0], &node_coords_[0], dimension_,
                       &coords[0], dimension_);
      flux_basis->SetTransform(coords);
    }

    std::vector<double> quad_points, quad_weights, quad_normal;
    {
      const int quad_order = order_;

      std::vector<double> ref_points, ref_weights;
      flux_basis->GetBasisPolynomial()->GetSurfaceQuadrature(
          quad_order, face_elemtype, elemorder, ref_points, ref_weights);

      const int num_points = static_cast<int>(ref_weights.size());
      std::vector<double> phy_points(num_points * dimension_);
      quad_points.resize(num_points * dimension_);
      owner_cell_element->TransformToPhyCoords(owner_type, num_points,
                                               &ref_points[0], &phy_points[0]);
      owner_cell_jacobian->TransformToPhyCoords(num_points, &phy_points[0],
                                                &quad_points[0]);

      quad_weights.resize(num_points);
      const double* normal = owner_cell_element->GetFacetypeNormal(owner_type);
      std::vector<double> cofactor(num_points * dimension_ * dimension_);
      owner_cell_jacobian->CalJacobianCofMat(num_points, &phy_points[0],
                                             &cofactor[0]);

      const Element* face_element = DENEB_ELEMENT->GetElement(face_elemtype);
      const double area_ratio =
          owner_cell_element->GetFacetypeArea(owner_type) /
          face_element->GetVolume();
      quad_normal.resize(num_points * dimension_);

      for (int ipoint = 0; ipoint < num_points; ipoint++) {
        gemvAx(1.0, &cofactor[ipoint * dimension_ * dimension_], dimension_,
               normal, 1, 0.0, &quad_normal[ipoint * dimension_], 1, dimension_,
               dimension_);
        quad_weights[ipoint] =
            avocado::VecLength(dimension_, &quad_normal[ipoint * dimension_]) *
            ref_weights[ipoint] * area_ratio;
      }
    }

    const int num_quad_points = static_cast<int>(quad_weights.size());
    double area = 0.0;
    for (auto&& weight : quad_weights) {
      weight = std::abs(weight);
      area += weight;
    }

    num_face_quad_points_[iface] = num_quad_points;
    face_area_[iface] = area;
    face_quad_weights_[iface] = quad_weights;

    {
      std::vector<double> owner_basis_value(num_quad_points * num_bases_);
      std::vector<double> neighbor_basis_value(num_quad_points * num_bases_);

      cell_basis_[owner_index]->GetBasis(num_quad_points, &quad_points[0],
                                        &owner_basis_value[0]);
      cell_basis_[neighbor_index]->GetBasis(num_quad_points, &quad_points[0],
                                          &neighbor_basis_value[0]);

      face_quad_owner_basis_value_[iface] = std::move(owner_basis_value);
      face_quad_neighbor_basis_value_[iface] = std::move(neighbor_basis_value);
    }
  }

  num_peribdry_quad_points_.resize(num_faces_);
  peribdry_area_.resize(num_faces_);
  peribdry_quad_owner_basis_value_.resize(num_faces_);
  peribdry_quad_neighbor_basis_value_.resize(num_faces_);
  peribdry_quad_weights_.resize(num_faces_);
  for (int ibdry = 0; ibdry < num_peribdries_; ibdry++) {
    const int owner_index = peribdry_owner_cell_[ibdry];
    const int owner_type = peribdry_owner_type_[ibdry];
    const int neighbor_index = peribdry_neighbor_cell_[ibdry];
    const int neighbor_type = peribdry_neighbor_type_[ibdry];

    const std::shared_ptr<Jacobian> owner_cell_jacobian =
        cell_jacobians_[owner_index];
    const Element* owner_cell_element = cell_element_[owner_index];
    const Element* neighbor_cell_element = cell_element_[neighbor_index];
    const ElemType bdry_elemtype =
        owner_cell_element->GetFaceElemtype(owner_type);
    const int& elemorder = cell_order_[owner_index];

    std::vector<double> offset(dimension_);  // owner to neighbor
    {
      std::vector<int> owner_bdry_nodes =
          owner_cell_element->GetFacetypeNodes(1)[owner_type];
      const int& owner_start_index = cell_node_ptr_[owner_index];
      for (auto&& inode : owner_bdry_nodes)
        inode = cell_node_ind_[owner_start_index + inode];
      const int num_nodes = static_cast<int>(owner_bdry_nodes.size());

      std::vector<double> owner_coords;
      owner_coords.resize(num_nodes * dimension_);
      avocado::VecCopy(num_nodes, &owner_bdry_nodes[0], &node_coords_[0],
                       dimension_, &owner_coords[0], dimension_);

      std::vector<int> neighbor_bdry_nodes =
          neighbor_cell_element->GetFacetypeNodes(1)[neighbor_type];
      const int& neighbor_start_index = cell_node_ptr_[neighbor_index];
      for (auto&& inode : neighbor_bdry_nodes)
        inode = cell_node_ind_[neighbor_start_index + inode];

      std::vector<double> neighbor_coords;
      neighbor_coords.resize(num_nodes * dimension_);
      avocado::VecCopy(num_nodes, &neighbor_bdry_nodes[0], &node_coords_[0],
                       dimension_, &neighbor_coords[0], dimension_);

      std::vector<double> neighbor_owner(num_nodes * dimension_);
      for (int i = 0, len = num_nodes * dimension_; i < len; i++)
        neighbor_owner[i] = neighbor_coords[i] - owner_coords[i];

      for (int idim = 0; idim < dimension_; idim++)
        offset[idim] =
            avocado::VecAverage(num_nodes, &neighbor_owner[idim], dimension_);
    }

    int num_flux_bases;
    std::shared_ptr<DRMSurface> drm = DRMSurface::GetDRM(bdry_elemtype);
    std::shared_ptr<BasisSurface> flux_basis = drm->GetApproximateBasis();
    {
      std::vector<int> bdry_nodes =
          owner_cell_element->GetFacetypeNodes(1)[owner_type];
      const int& start_index = cell_node_ptr_[owner_index];
      for (auto&& inode : bdry_nodes)
        inode = cell_node_ind_[start_index + inode];
      const int num_nodes = static_cast<int>(bdry_nodes.size());

      std::vector<double> coords;
      coords.resize(num_nodes * dimension_);
      avocado::VecCopy(num_nodes, &bdry_nodes[0], &node_coords_[0], dimension_,
                       &coords[0], dimension_);
      flux_basis->SetTransform(coords);
    }

    std::vector<double> quad_points, quad_weights, quad_normal;
    std::vector<double> neighbor_quad_points;

    const int quad_order = order_;

    std::vector<double> ref_points, ref_weights;
    flux_basis->GetBasisPolynomial()->GetSurfaceQuadrature(
        quad_order, bdry_elemtype, elemorder, ref_points, ref_weights);

    const int num_points = static_cast<int>(ref_weights.size());
    std::vector<double> phy_points(num_points * dimension_);
    quad_points.resize(num_points * dimension_);
    owner_cell_element->TransformToPhyCoords(owner_type, num_points,
                                             &ref_points[0], &phy_points[0]);
    owner_cell_jacobian->TransformToPhyCoords(num_points, &phy_points[0],
                                              &quad_points[0]);

    quad_weights.resize(num_points);
    const double* normal = owner_cell_element->GetFacetypeNormal(owner_type);
    std::vector<double> cofactor(num_points * dimension_ * dimension_);
    owner_cell_jacobian->CalJacobianCofMat(num_points, &phy_points[0],
                                           &cofactor[0]);

    const Element* bdry_element = DENEB_ELEMENT->GetElement(bdry_elemtype);
    const double area_ratio = owner_cell_element->GetFacetypeArea(owner_type) /
                              bdry_element->GetVolume();
    quad_normal.resize(num_points * dimension_);
    for (int ipoint = 0; ipoint < num_points; ipoint++) {
      gemvAx(1.0, &cofactor[ipoint * dimension_ * dimension_], dimension_,
             normal, 1, 0.0, &quad_normal[ipoint * dimension_], 1, dimension_,
             dimension_);
      quad_weights[ipoint] =
          avocado::VecLength(dimension_, &quad_normal[ipoint * dimension_]) *
          ref_weights[ipoint] * area_ratio;
    }

    neighbor_quad_points = quad_points;
    for (int ipoint = 0; ipoint < num_points; ipoint++)
      for (int idim = 0; idim < dimension_; idim++)
        neighbor_quad_points[ipoint * dimension_ + idim] += offset[idim];

    const int num_quad_points = static_cast<int>(quad_weights.size());
    double area = 0.0;
    for (auto&& weight : quad_weights) {
      weight = std::abs(weight);
      area += weight;
    }

    num_peribdry_quad_points_[ibdry] = num_quad_points;
    peribdry_area_[ibdry] = area;
    peribdry_quad_weights_[ibdry] = std::move(quad_weights);

      std::vector<double> owner_basis_value(num_quad_points * num_bases_);
    std::vector<double> neighbor_basis_value(num_quad_points * num_bases_);
    cell_basis_[owner_index]->GetBasis(num_quad_points, &quad_points[0],
                                       &owner_basis_value[0]);
    cell_basis_[neighbor_index]->GetBasis(
        num_quad_points, &neighbor_quad_points[0], &neighbor_basis_value[0]);

    peribdry_quad_owner_basis_value_[ibdry] = std::move(owner_basis_value);
    peribdry_quad_neighbor_basis_value_[ibdry] =
        std::move(neighbor_basis_value);
  }

  CombineData(face_owner_cell_, peribdry_owner_cell_, rules);
  CombineData(face_neighbor_cell_, peribdry_neighbor_cell_, rules);
  CombineData(face_owner_type_, peribdry_owner_type_, rules);
  CombineData(face_neighbor_type_, peribdry_neighbor_type_, rules);

  CombineData(num_face_quad_points_, num_peribdry_quad_points_, rules);
  CombineData(face_area_, peribdry_area_, rules);
  CombineData(face_quad_owner_basis_value_, peribdry_quad_owner_basis_value_,
              rules);
  CombineData(face_quad_neighbor_basis_value_,
              peribdry_quad_neighbor_basis_value_, rules);
  CombineData(face_quad_weights_, peribdry_quad_weights_, rules);

  for (int i = num_inner_faces_; i < num_faces_; i++)
    face_neighbor_cell_[i] -= num_cells_;
}
void Data::BuildCellData(const int icell) {
  const ElemType& elemtype = cell_elemtype_[icell];
  const int& elemorder = cell_order_[icell];
  cell_element_[icell] = DENEB_ELEMENT->GetElement(elemtype);

  cell_jacobians_[icell] = Jacobian::GetJacobian(elemtype);
  {
    std::vector<double> sub_coords;
    const int num_nodes =
        cell_subnode_ptr_[icell + 1] - cell_subnode_ptr_[icell];
    sub_coords.resize(num_nodes * dimension_);
    avocado::VecCopy(num_nodes, &cell_subnode_ind_[cell_subnode_ptr_[icell]],
                     &node_coords_[0], dimension_, &sub_coords[0], dimension_);
    cell_jacobians_[icell]->SetTopology(elemorder, &sub_coords[0]);

    for (int idim = 0; idim < dimension_; idim++) {
      cell_center_coords_[icell * dimension_ + idim] = 0.0;
      for (int inode = 0; inode < num_nodes; inode++) {
        cell_center_coords_[icell * dimension_ + idim] +=
            sub_coords[inode * dimension_ + idim];
      }
      cell_center_coords_[icell * dimension_ + idim] /=
          static_cast<double>(num_nodes);
    }
  }

  double volume = 0.0;
  std::vector<double> coords;
  cell_basis_[icell] = Basis::GetStandardBasis(elemtype);
  {
    const int num_nodes = cell_node_ptr_[icell + 1] - cell_node_ptr_[icell];
    coords.resize(num_nodes * dimension_);
    avocado::VecCopy(num_nodes, &cell_node_ind_[cell_node_ptr_[icell]],
                     &node_coords_[0], dimension_, &coords[0], dimension_);
    cell_basis_[icell]->SetTransform(coords);

    std::vector<double> ref_points, quad_weights;
    cell_basis_[icell]->GetBasisPolynomial()->GetVolumeQuadrature(
        2 * order_, elemtype, elemorder, ref_points, quad_weights);
    const int num_points = static_cast<int>(quad_weights.size());
    std::vector<double> jacobian_det(num_points);
    cell_jacobians_[icell]->CalJacobianDet(num_points, &ref_points[0],
                                           &jacobian_det[0]);
    for (int ipoint = 0; ipoint < num_points; ipoint++)
      quad_weights[ipoint] *= std::abs(jacobian_det[ipoint]);
    std::vector<double> quad_points(num_points * dimension_);
    cell_jacobians_[icell]->TransformToPhyCoords(num_points, &ref_points[0],
                                                 &quad_points[0]);
    cell_basis_[icell]->ComputeConnectingMatrix(order_, quad_points,
                                                quad_weights);


    for (int ipoint = 0; ipoint < num_points; ipoint++)
      volume += quad_weights[ipoint];
  }

  cell_volumes_[icell] = volume;

  if (icell >= num_cells_) return;

  for (auto&& nodes_per_face : cell_element_[icell]->GetFaceNodes()) {
    std::vector<int> face_nodes = nodes_per_face;
    const int& start_index = cell_node_ptr_[icell];
    for (auto&& inode : face_nodes) inode = cell_node_ind_[start_index + inode];
    const int num_nodes = static_cast<int>(face_nodes.size());

    std::vector<double> face_coords(num_nodes * dimension_);
    avocado::VecCopy(num_nodes, &face_nodes[0], &node_coords_[0], dimension_,
                     &face_coords[0], dimension_);
    for (int idim = 0; idim < dimension_; idim++)
      cell_proj_volumes_[icell * dimension_ + idim] +=
          0.5 * ComputeProjVolume(face_coords, idim);
  }

  int num_flux_bases;
  std::shared_ptr<DRMVolume> drm = DRMVolume::GetDRM(elemtype);
  std::shared_ptr<Basis> flux_basis = drm->GetApproximateBasis();
  {
    flux_basis->SetTransform(coords);

    const int P1 = 1;
    std::shared_ptr<Jacobian> jacobian = Jacobian::GetJacobian(elemtype);
    jacobian->SetTopology(P1, &coords[0]);

    std::vector<double> ref_points, quad_weights;
    flux_basis->GetBasisPolynomial()->GetVolumeQuadrature(
        2 * volume_flux_order_, elemtype, elemorder, ref_points, quad_weights);
    const int num_points = static_cast<int>(quad_weights.size());
    std::vector<double> jacobian_det(num_points);
    jacobian->CalJacobianDet(num_points, &ref_points[0], &jacobian_det[0]);
    for (int ipoint = 0; ipoint < num_points; ipoint++)
      quad_weights[ipoint] *= std::abs(jacobian_det[ipoint]);
    std::vector<double> quad_points(num_points * dimension_);
    jacobian->TransformToPhyCoords(num_points, &ref_points[0], &quad_points[0]);
    flux_basis->ComputeConnectingMatrix(volume_flux_order_, quad_points,
                                        quad_weights);
    num_flux_bases = flux_basis->GetNumBases();
  }

  int num_DRM_points;
  std::vector<double> DRM_points;
  std::vector<double> invvan_trans;
  {
    std::vector<double> ref_points;
    drm->GetDRMPoints(num_flux_bases, ref_points);
    num_DRM_points = static_cast<int>(ref_points.size() / dimension_);
    DRM_points.resize(num_DRM_points * dimension_);
    cell_jacobians_[icell]->TransformToPhyCoords(num_DRM_points, &ref_points[0],
                                                 &DRM_points[0]);
    std::vector<double> van_trans(num_DRM_points * num_flux_bases);
    flux_basis->GetBasis(num_DRM_points, &DRM_points[0], &van_trans[0]);
    int num_ranks;
    avocado::MatPseudoInv(&van_trans[0], num_DRM_points, num_flux_bases,
                          num_ranks, lc_ * 1.0E-4);
    if (num_ranks < num_flux_bases) ERROR_MESSAGE("Not full rank.\n");
    invvan_trans = std::move(van_trans);
  }

  num_cell_points_[icell] = num_DRM_points;
  cell_points_[icell] = DRM_points;
  cell_basis_value_[icell].resize(num_DRM_points * num_bases_);
  cell_basis_grad_value_[icell].resize(num_DRM_points * num_bases_ *
                                       dimension_);
  {
    cell_basis_[icell]->GetBasis(num_DRM_points, &DRM_points[0],
                                 &cell_basis_value_[icell][0]);
    cell_basis_[icell]->GetBasisGrad(num_DRM_points, &DRM_points[0],
                                     &cell_basis_grad_value_[icell][0]);
  }

  std::vector<double> quad_points, quad_weights;
  {
    int quad_order = volume_flux_order_ + order_ - 1;
    if (has_source_term_ || has_mass_matrix_) quad_order++;
    if (has_mass_matrix_) quad_order += order_;

    std::vector<double> ref_points;
    flux_basis->GetBasisPolynomial()->GetVolumeQuadrature(
        quad_order, elemtype, elemorder, ref_points, quad_weights);
    const int num_points = static_cast<int>(quad_weights.size());
    std::vector<double> jacobian_det(num_points);
    cell_jacobians_[icell]->CalJacobianDet(num_points, &ref_points[0],
                                           &jacobian_det[0]);
    for (int ipoint = 0; ipoint < num_points; ipoint++)
      quad_weights[ipoint] *= std::abs(jacobian_det[ipoint]);
    quad_points.resize(num_points * dimension_);
    cell_jacobians_[icell]->TransformToPhyCoords(num_points, &ref_points[0],
                                                 &quad_points[0]);
  }

  cell_coefficients_[icell].resize(num_DRM_points * num_bases_ * dimension_);
  if (has_source_term_)
    cell_source_coefficients_[icell].resize(num_DRM_points * num_bases_);
  if (has_mass_matrix_)
    cell_mass_coefficients_[icell].resize(num_DRM_points * num_bases_ *
                                          num_bases_);
  {
    const int num_quad_points = static_cast<int>(quad_weights.size());
    std::vector<double> basis_value(num_quad_points * num_flux_bases);
    flux_basis->GetBasis(num_quad_points, &quad_points[0], &basis_value[0]);

    std::vector<double> nodal_basis_value(num_quad_points * num_DRM_points);
    gemmAB(1.0, &basis_value[0], &invvan_trans[0], 0.0, &nodal_basis_value[0],
           num_quad_points, num_flux_bases, num_DRM_points);

    for (int ipoint = 0; ipoint < num_quad_points; ipoint++)
    {
      cblas_dscal(num_DRM_points, quad_weights[ipoint],
        &nodal_basis_value[ipoint * num_DRM_points], 1);
    }

    std::vector<double> basis_grad_value(num_quad_points * num_bases_ *
                                         dimension_);
    cell_basis_[icell]->GetBasisGrad(num_quad_points, &quad_points[0],
                                     &basis_grad_value[0]);

    gemmATB(1.0, &nodal_basis_value[0], &basis_grad_value[0], 0.0,
            &cell_coefficients_[icell][0], num_quad_points, num_DRM_points,
            num_bases_ * dimension_);

    if (has_source_term_ || has_mass_matrix_) {
      std::vector<double> basis_value(num_quad_points * num_bases_);
      cell_basis_[icell]->GetBasis(num_quad_points, &quad_points[0],
                                   &basis_value[0]);

      gemmATB(1.0, &nodal_basis_value[0], &basis_value[0], 0.0,
              &cell_source_coefficients_[icell][0], num_quad_points,
              num_DRM_points, num_bases_);

      if (has_mass_matrix_) {
        std::vector<double> basis_prod_value(num_quad_points * num_bases_ *
                                        num_bases_, 0.0);

        for (int ipoint = 0; ipoint < num_quad_points; ipoint++) {
          gemmATB(1.0, &basis_value[ipoint * num_bases_],
                  &basis_value[ipoint * num_bases_], 0.0,
                  &basis_prod_value[ipoint * num_bases_ * num_bases_], 1,
                  num_bases_, num_bases_);
        }

        gemmATB(1.0, &nodal_basis_value[0], &basis_prod_value[0], 0.0,
                &cell_mass_coefficients_[icell][0], num_quad_points,
                num_DRM_points, num_bases_ * num_bases_);
      }
    }
  }
}
void Data::BuildFaceData(const int iface) {
  const int& owner_index = face_owner_cell_[iface];
  const int& owner_type = face_owner_type_[iface];
  const int& neighbor_index = face_neighbor_cell_[iface];

  const std::shared_ptr<Jacobian> owner_cell_jacobian =
      cell_jacobians_[owner_index];
  const Element* owner_cell_element = cell_element_[owner_index];
  const ElemType face_elemtype =
      owner_cell_element->GetFaceElemtype(owner_type);
  const int& elemorder = cell_order_[owner_index];

  int num_flux_bases;
  std::shared_ptr<DRMSurface> drm = DRMSurface::GetDRM(face_elemtype);
  std::shared_ptr<BasisSurface> flux_basis = drm->GetApproximateBasis();
  {
    std::vector<int> face_nodes =
        owner_cell_element->GetFacetypeNodes(1)[owner_type];
    const int& start_index = cell_node_ptr_[owner_index];
    for (auto&& inode : face_nodes) inode = cell_node_ind_[start_index + inode];
    const int num_nodes = static_cast<int>(face_nodes.size());

    std::vector<double> coords;
    coords.resize(num_nodes * dimension_);
    avocado::VecCopy(num_nodes, &face_nodes[0], &node_coords_[0], dimension_,
                     &coords[0], dimension_);
    flux_basis->SetTransform(coords);

    std::vector<double> ref_points, ref_weights;
    flux_basis->GetBasisPolynomial()->GetSurfaceQuadrature(
        2 * surface_flux_order_, face_elemtype, elemorder, ref_points,
        ref_weights);
    const int num_points = static_cast<int>(ref_weights.size());
    std::vector<double> phy_points(num_points * dimension_);
    std::vector<double> quad_points(num_points * dimension_);
    owner_cell_element->TransformToPhyCoords(owner_type, num_points,
                                             &ref_points[0], &phy_points[0]);
    owner_cell_jacobian->TransformToPhyCoords(num_points, &phy_points[0],
                                              &quad_points[0]);
    flux_basis->ComputeConnectingMatrix(surface_flux_order_, quad_points,
                                        ref_weights, 1.0E-10);
    num_flux_bases = flux_basis->GetNumBases();
  }

  int num_DRM_points = num_flux_bases;
  std::vector<double> DRM_points;
  std::vector<double> invvan_trans;
  for (int ideg = 0; ideg < 100; ideg++) {
    if (ideg == 10)
      ERROR_MESSAGE(
          "Fail to achieve full-rank DRM points for DRM surface integration."
          "\n\tface number: " +
          std::to_string(iface) +
          "\n\tface type: " + Element::GetElementName(face_elemtype) + "\n");

    std::vector<double> ref_points;
    drm->GetDRMPoints(num_DRM_points, ref_points);
    num_DRM_points = static_cast<int>(ref_points.size()) / (dimension_ - 1);

    std::vector<double> phy_points(num_DRM_points * dimension_);
    DRM_points.clear();
    DRM_points.resize(num_DRM_points * dimension_);
    owner_cell_element->TransformToPhyCoords(owner_type, num_DRM_points,
                                             &ref_points[0], &phy_points[0]);
    owner_cell_jacobian->TransformToPhyCoords(num_DRM_points, &phy_points[0],
                                              &DRM_points[0]);

    std::vector<double> van_trans(num_DRM_points * num_flux_bases);
    flux_basis->GetBasis(num_DRM_points, &DRM_points[0], &van_trans[0]);
    int num_ranks;
    avocado::MatPseudoInv(&van_trans[0], num_DRM_points, num_flux_bases,
                          num_ranks, lc_ * 1.0E-4);
    if (num_ranks < num_flux_bases) {
      num_DRM_points++;
      continue;
    }
    invvan_trans = std::move(van_trans);

    const double* normal = owner_cell_element->GetFacetypeNormal(owner_type);
    std::vector<double> cofactor(num_DRM_points * dimension_ * dimension_);
    owner_cell_jacobian->CalJacobianCofMat(num_DRM_points, &phy_points[0],
                                           &cofactor[0]);
    face_normals_[iface].resize(num_DRM_points * dimension_);
    for (int ipoint = 0; ipoint < num_DRM_points; ipoint++) {
      gemvAx(1.0, &cofactor[ipoint * dimension_ * dimension_], dimension_,
             normal, 1, 0.0, &face_normals_[iface][ipoint * dimension_], 1,
             dimension_, dimension_);
      avocado::VecNormal(dimension_,
                         &face_normals_[iface][ipoint * dimension_]);
    }
    break;
  }

  num_face_points_[iface] = num_DRM_points;
  face_points_[iface] = DRM_points;
  face_owner_basis_value_[iface].resize(num_DRM_points * num_bases_);
  face_owner_basis_grad_value_[iface].resize(num_DRM_points * num_bases_ *
                                             dimension_);
  face_neighbor_basis_value_[iface].resize(num_DRM_points * num_bases_);
  face_neighbor_basis_grad_value_[iface].resize(num_DRM_points * num_bases_ *
                                                dimension_);
  {
    cell_basis_[owner_index]->GetBasis(num_DRM_points, &DRM_points[0],
                                       &face_owner_basis_value_[iface][0]);
    cell_basis_[owner_index]->GetBasisGrad(
        num_DRM_points, &DRM_points[0],
        &face_owner_basis_grad_value_[iface][0]);
    cell_basis_[neighbor_index]->GetBasis(
        num_DRM_points, &DRM_points[0], &face_neighbor_basis_value_[iface][0]);
    cell_basis_[neighbor_index]->GetBasisGrad(
        num_DRM_points, &DRM_points[0],
        &face_neighbor_basis_grad_value_[iface][0]);
  }

  std::vector<double> quad_points, quad_weights, quad_normal;
  {
    const int quad_order = surface_flux_order_ + order_;

    std::vector<double> ref_points, ref_weights;
    flux_basis->GetBasisPolynomial()->GetSurfaceQuadrature(
        quad_order, face_elemtype, elemorder, ref_points, ref_weights);
    const int num_points = static_cast<int>(ref_weights.size());
    std::vector<double> phy_points(num_points * dimension_);
    quad_points.resize(num_points * dimension_);
    owner_cell_element->TransformToPhyCoords(owner_type, num_points,
                                             &ref_points[0], &phy_points[0]);
    owner_cell_jacobian->TransformToPhyCoords(num_points, &phy_points[0],
                                              &quad_points[0]);

    quad_weights.resize(num_points);
    const double* normal = owner_cell_element->GetFacetypeNormal(owner_type);
    std::vector<double> cofactor(num_points * dimension_ * dimension_);
    owner_cell_jacobian->CalJacobianCofMat(num_points, &phy_points[0],
                                           &cofactor[0]);

    const Element* face_element = DENEB_ELEMENT->GetElement(face_elemtype);
    const double area_ratio = owner_cell_element->GetFacetypeArea(owner_type) /
                              face_element->GetVolume();
    quad_normal.resize(num_points * dimension_);
    for (int ipoint = 0; ipoint < num_points; ipoint++) {
      gemvAx(1.0, &cofactor[ipoint * dimension_ * dimension_], dimension_,
             normal, 1, 0.0, &quad_normal[ipoint * dimension_], 1, dimension_,
             dimension_);
      quad_weights[ipoint] =
          avocado::VecLength(dimension_, &quad_normal[ipoint * dimension_]) *
          ref_weights[ipoint] * area_ratio;
      avocado::VecNormal(dimension_, &quad_normal[ipoint * dimension_]);
    }
  }

  face_owner_coefficients_[iface].resize(num_DRM_points * num_bases_ *
                                         dimension_);
  face_neighbor_coefficients_[iface].resize(num_DRM_points * num_bases_ *
                                            dimension_);
  {
    const int num_quad_points = static_cast<int>(quad_weights.size());
    std::vector<double> basis_value(num_quad_points * num_flux_bases);
    flux_basis->GetBasis(num_quad_points, &quad_points[0], &basis_value[0]);

    std::vector<double> nodal_basis_value(num_quad_points * num_DRM_points);
    gemmAB(1.0, &basis_value[0], &invvan_trans[0], 0.0, &nodal_basis_value[0],
           num_quad_points, num_flux_bases, num_DRM_points);

    for (int ipoint = 0; ipoint < num_quad_points; ipoint++)
    {
      cblas_dscal(num_DRM_points, quad_weights[ipoint],
        &nodal_basis_value[ipoint * num_DRM_points], 1);
    }

    std::vector<double> temp(num_quad_points * num_DRM_points * dimension_,
                             0.0);
    for (int ipoint = 0; ipoint < num_quad_points; ipoint++)
      cblas_dger(CBLAS_LAYOUT::CblasRowMajor, num_DRM_points, dimension_, 1.0,
                 &nodal_basis_value[ipoint * num_DRM_points], 1,
                 &quad_normal[ipoint * dimension_], 1,
                 &temp[ipoint * num_DRM_points * dimension_], dimension_);

    std::vector<double> owner_basis_value(num_quad_points * num_bases_);
    std::vector<double> neighbor_basis_value(num_quad_points * num_bases_);
    cell_basis_[owner_index]->GetBasis(num_quad_points, &quad_points[0],
                                       &owner_basis_value[0]);
    cell_basis_[neighbor_index]->GetBasis(num_quad_points, &quad_points[0],
                                          &neighbor_basis_value[0]);

    avocado::Kernel1::f72(
        &temp[0], &owner_basis_value[0], &face_owner_coefficients_[iface][0],
        dimension_, num_DRM_points, num_quad_points, num_bases_, 1.0, 0.0);
    avocado::Kernel1::f72(&temp[0], &neighbor_basis_value[0],
                          &face_neighbor_coefficients_[iface][0], dimension_,
                          num_DRM_points, num_quad_points, num_bases_, 1.0,
                          0.0);
  }
}
void Data::BuildBdryData(const int ibdry) {
  const int& owner_index = bdry_owner_cell_[ibdry];
  const int& owner_type = bdry_owner_type_[ibdry];

  const std::shared_ptr<Jacobian> owner_cell_jacobian =
      cell_jacobians_[owner_index];
  const Element* owner_cell_element = cell_element_[owner_index];
  const ElemType bdry_elemtype =
      owner_cell_element->GetFaceElemtype(owner_type);
  const int& elemorder = cell_order_[owner_index];

  int num_flux_bases;
  std::shared_ptr<DRMSurface> drm = DRMSurface::GetDRM(bdry_elemtype);
  std::shared_ptr<BasisSurface> flux_basis = drm->GetApproximateBasis();
  {
    std::vector<int> bdry_nodes =
        owner_cell_element->GetFacetypeNodes(1)[owner_type];
    const int& start_index = cell_node_ptr_[owner_index];
    for (auto&& inode : bdry_nodes) inode = cell_node_ind_[start_index + inode];
    const int num_nodes = static_cast<int>(bdry_nodes.size());

    std::vector<double> coords;
    coords.resize(num_nodes * dimension_);
    avocado::VecCopy(num_nodes, &bdry_nodes[0], &node_coords_[0], dimension_,
                     &coords[0], dimension_);
    flux_basis->SetTransform(coords);

    std::vector<double> ref_points, ref_weights;
    flux_basis->GetBasisPolynomial()->GetSurfaceQuadrature(
        2 * surface_flux_order_, bdry_elemtype, elemorder, ref_points,
        ref_weights);
    const int num_points = static_cast<int>(ref_weights.size());
    std::vector<double> phy_points(num_points * dimension_);
    std::vector<double> quad_points(num_points * dimension_);
    owner_cell_element->TransformToPhyCoords(owner_type, num_points,
                                             &ref_points[0], &phy_points[0]);
    owner_cell_jacobian->TransformToPhyCoords(num_points, &phy_points[0],
                                              &quad_points[0]);
    flux_basis->ComputeConnectingMatrix(surface_flux_order_, quad_points,
                                        ref_weights, 1.0E-10);
    num_flux_bases = flux_basis->GetNumBases();
  }

  int num_DRM_points = num_flux_bases;
  std::vector<double> DRM_points;
  std::vector<double> invvan_trans;
  for (int ideg = 0; ideg < 100; ideg++) {
    if (ideg == 10)
      ERROR_MESSAGE(
          "Fail to achieve full-rank DRM points for DRM surface integration."
          "\n\tboundary number: " +
          std::to_string(ibdry) +
          "\n\tbdry type: " + Element::GetElementName(bdry_elemtype) + "\n");

    std::vector<double> ref_points;
    drm->GetDRMPoints(num_DRM_points, ref_points);
    num_DRM_points = static_cast<int>(ref_points.size()) / (dimension_ - 1);

    std::vector<double> phy_points(num_DRM_points * dimension_);
    DRM_points.clear();
    DRM_points.resize(num_DRM_points * dimension_);
    owner_cell_element->TransformToPhyCoords(owner_type, num_DRM_points,
                                             &ref_points[0], &phy_points[0]);
    owner_cell_jacobian->TransformToPhyCoords(num_DRM_points, &phy_points[0],
                                              &DRM_points[0]);

    std::vector<double> van_trans(num_DRM_points * num_flux_bases);
    flux_basis->GetBasis(num_DRM_points, &DRM_points[0], &van_trans[0]);
    int num_ranks;
    avocado::MatPseudoInv(&van_trans[0], num_DRM_points, num_flux_bases,
                          num_ranks, lc_ * 1.0E-4);
    if (num_ranks < num_flux_bases) {
      num_DRM_points++;
      continue;
    }
    invvan_trans = std::move(van_trans);

    const double* normal = owner_cell_element->GetFacetypeNormal(owner_type);
    std::vector<double> cofactor(num_DRM_points * dimension_ * dimension_);
    owner_cell_jacobian->CalJacobianCofMat(num_DRM_points, &phy_points[0],
                                           &cofactor[0]);
    bdry_normals_[ibdry].resize(num_DRM_points * dimension_);
    for (int ipoint = 0; ipoint < num_DRM_points; ipoint++) {
      gemvAx(1.0, &cofactor[ipoint * dimension_ * dimension_], dimension_,
             normal, 1, 0.0, &bdry_normals_[ibdry][ipoint * dimension_], 1,
             dimension_, dimension_);
      avocado::VecNormal(dimension_,
                         &bdry_normals_[ibdry][ipoint * dimension_]);
    }
    break;
  }

  num_bdry_points_[ibdry] = num_DRM_points;
  bdry_points_[ibdry] = DRM_points;
  bdry_owner_basis_value_[ibdry].resize(num_DRM_points * num_bases_);
  bdry_owner_basis_grad_value_[ibdry].resize(num_DRM_points * num_bases_ *
                                             dimension_);
  {
    cell_basis_[owner_index]->GetBasis(num_DRM_points, &DRM_points[0],
                                       &bdry_owner_basis_value_[ibdry][0]);
    cell_basis_[owner_index]->GetBasisGrad(
        num_DRM_points, &DRM_points[0],
        &bdry_owner_basis_grad_value_[ibdry][0]);
  }

  std::vector<double> quad_points, quad_weights, quad_normal;
  {
    const int quad_order = surface_flux_order_ + order_;

    std::vector<double> ref_points, ref_weights;
    flux_basis->GetBasisPolynomial()->GetSurfaceQuadrature(
        quad_order, bdry_elemtype, elemorder, ref_points, ref_weights);
    const int num_points = static_cast<int>(ref_weights.size());
    std::vector<double> phy_points(num_points * dimension_);
    quad_points.resize(num_points * dimension_);
    owner_cell_element->TransformToPhyCoords(owner_type, num_points,
                                             &ref_points[0], &phy_points[0]);
    owner_cell_jacobian->TransformToPhyCoords(num_points, &phy_points[0],
                                              &quad_points[0]);

    quad_weights.resize(num_points);
    const double* normal = owner_cell_element->GetFacetypeNormal(owner_type);
    std::vector<double> cofactor(num_points * dimension_ * dimension_);
    owner_cell_jacobian->CalJacobianCofMat(num_points, &phy_points[0],
                                           &cofactor[0]);

    const Element* bdry_element = DENEB_ELEMENT->GetElement(bdry_elemtype);
    const double area_ratio = owner_cell_element->GetFacetypeArea(owner_type) /
                              bdry_element->GetVolume();
    quad_normal.resize(num_points * dimension_);
    for (int ipoint = 0; ipoint < num_points; ipoint++) {
      gemvAx(1.0, &cofactor[ipoint * dimension_ * dimension_], dimension_,
             normal, 1, 0.0, &quad_normal[ipoint * dimension_], 1, dimension_,
             dimension_);
      quad_weights[ipoint] =
          avocado::VecLength(dimension_, &quad_normal[ipoint * dimension_]) *
          ref_weights[ipoint] * area_ratio;
      avocado::VecNormal(dimension_, &quad_normal[ipoint * dimension_]);
    }
  }

  bdry_owner_coefficients_[ibdry].resize(num_DRM_points * num_bases_ *
                                         dimension_);
  {
    const int num_quad_points = static_cast<int>(quad_weights.size());
    std::vector<double> basis_value(num_quad_points * num_flux_bases);
    flux_basis->GetBasis(num_quad_points, &quad_points[0], &basis_value[0]);

    std::vector<double> nodal_basis_value(num_quad_points * num_DRM_points);
    gemmAB(1.0, &basis_value[0], &invvan_trans[0], 0.0, &nodal_basis_value[0],
           num_quad_points, num_flux_bases, num_DRM_points);

    for (int ipoint = 0; ipoint < num_quad_points; ipoint++)
    {
      cblas_dscal(num_DRM_points, quad_weights[ipoint],
        &nodal_basis_value[ipoint * num_DRM_points], 1);
    }

    std::vector<double> temp(num_quad_points * num_DRM_points * dimension_,
                             0.0);
    for (int ipoint = 0; ipoint < num_quad_points; ipoint++)
      cblas_dger(CBLAS_LAYOUT::CblasRowMajor, num_DRM_points, dimension_, 1.0,
                 &nodal_basis_value[ipoint * num_DRM_points], 1,
                 &quad_normal[ipoint * dimension_], 1,
                 &temp[ipoint * num_DRM_points * dimension_], dimension_);

    std::vector<double> owner_basis_value(num_quad_points * num_bases_);
    cell_basis_[owner_index]->GetBasis(num_quad_points, &quad_points[0],
                                       &owner_basis_value[0]);
    avocado::Kernel1::f72(
        &temp[0], &owner_basis_value[0], &bdry_owner_coefficients_[ibdry][0],
        dimension_, num_DRM_points, num_quad_points, num_bases_, 1.0, 0.0);
  }
}
void Data::BuildPeribdryData(const int ibdry) {
  const int& owner_index = peribdry_owner_cell_[ibdry];
  const int& owner_type = peribdry_owner_type_[ibdry];
  const int& neighbor_index = peribdry_neighbor_cell_[ibdry];
  const int& neighbor_type = peribdry_neighbor_type_[ibdry];

  const std::shared_ptr<Jacobian> owner_cell_jacobian =
      cell_jacobians_[owner_index];
  const Element* owner_cell_element = cell_element_[owner_index];
  const Element* neighbor_cell_element = cell_element_[neighbor_index];
  const ElemType bdry_elemtype =
      owner_cell_element->GetFaceElemtype(owner_type);
  const int& elemorder = cell_order_[owner_index];

  std::vector<double> offset(dimension_);  // owner to neighbor
  {
    std::vector<int> owner_bdry_nodes =
        owner_cell_element->GetFacetypeNodes(1)[owner_type];
    const int& owner_start_index = cell_node_ptr_[owner_index];
    for (auto&& inode : owner_bdry_nodes)
      inode = cell_node_ind_[owner_start_index + inode];
    const int num_nodes = static_cast<int>(owner_bdry_nodes.size());

    std::vector<double> owner_coords;
    owner_coords.resize(num_nodes * dimension_);
    avocado::VecCopy(num_nodes, &owner_bdry_nodes[0], &node_coords_[0],
                     dimension_, &owner_coords[0], dimension_);

    std::vector<int> neighbor_bdry_nodes =
        neighbor_cell_element->GetFacetypeNodes(1)[neighbor_type];
    const int& neighbor_start_index = cell_node_ptr_[neighbor_index];
    for (auto&& inode : neighbor_bdry_nodes)
      inode = cell_node_ind_[neighbor_start_index + inode];

    std::vector<double> neighbor_coords;
    neighbor_coords.resize(num_nodes * dimension_);
    avocado::VecCopy(num_nodes, &neighbor_bdry_nodes[0], &node_coords_[0],
                     dimension_, &neighbor_coords[0], dimension_);

    std::vector<double> neighbor_owner(num_nodes * dimension_);
    for (int i = 0, len = num_nodes * dimension_; i < len; i++)
      neighbor_owner[i] = neighbor_coords[i] - owner_coords[i];

    for (int idim = 0; idim < dimension_; idim++)
      offset[idim] =
          avocado::VecAverage(num_nodes, &neighbor_owner[idim], dimension_);
  }

  int num_flux_bases;
  std::shared_ptr<DRMSurface> drm = DRMSurface::GetDRM(bdry_elemtype);
  std::shared_ptr<BasisSurface> flux_basis = drm->GetApproximateBasis();
  {
    std::vector<int> bdry_nodes =
        owner_cell_element->GetFacetypeNodes(1)[owner_type];
    const int& start_index = cell_node_ptr_[owner_index];
    for (auto&& inode : bdry_nodes) inode = cell_node_ind_[start_index + inode];
    const int num_nodes = static_cast<int>(bdry_nodes.size());

    std::vector<double> coords;
    coords.resize(num_nodes * dimension_);
    avocado::VecCopy(num_nodes, &bdry_nodes[0], &node_coords_[0], dimension_,
                     &coords[0], dimension_);
    flux_basis->SetTransform(coords);

    std::vector<double> ref_points, ref_weights;
    flux_basis->GetBasisPolynomial()->GetSurfaceQuadrature(
        2 * surface_flux_order_, bdry_elemtype, elemorder, ref_points,
        ref_weights);
    const int num_points = static_cast<int>(ref_weights.size());
    std::vector<double> phy_points(num_points * dimension_);
    std::vector<double> quad_points(num_points * dimension_);
    owner_cell_element->TransformToPhyCoords(owner_type, num_points,
                                             &ref_points[0], &phy_points[0]);
    owner_cell_jacobian->TransformToPhyCoords(num_points, &phy_points[0],
                                              &quad_points[0]);
    flux_basis->ComputeConnectingMatrix(surface_flux_order_, quad_points,
                                        ref_weights, 1.0E-10);
    num_flux_bases = flux_basis->GetNumBases();
  }

  int num_DRM_points = num_flux_bases;
  std::vector<double> DRM_points;
  std::vector<double> neighbor_DRM_points;
  std::vector<double> invvan_trans;
  for (int ideg = 0; ideg < 100; ideg++) {
    if (ideg == 10)
      ERROR_MESSAGE(
          "Fail to achieve full-rank DRM points for DRM surface integration."
          "\n\tperiodic boundary number: " +
          std::to_string(ibdry) +
          "\n\tbdry type: " + Element::GetElementName(bdry_elemtype) + "\n");

    std::vector<double> ref_points;
    drm->GetDRMPoints(num_DRM_points, ref_points);
    num_DRM_points = static_cast<int>(ref_points.size()) / (dimension_ - 1);

    std::vector<double> phy_points(num_DRM_points * dimension_);
    DRM_points.clear();
    DRM_points.resize(num_DRM_points * dimension_);
    owner_cell_element->TransformToPhyCoords(owner_type, num_DRM_points,
                                             &ref_points[0], &phy_points[0]);
    owner_cell_jacobian->TransformToPhyCoords(num_DRM_points, &phy_points[0],
                                              &DRM_points[0]);

    std::vector<double> van_trans(num_DRM_points * num_flux_bases);
    flux_basis->GetBasis(num_DRM_points, &DRM_points[0], &van_trans[0]);
    int num_ranks;
    avocado::MatPseudoInv(&van_trans[0], num_DRM_points, num_flux_bases,
                          num_ranks, lc_ * 1.0E-4);
    if (num_ranks < num_flux_bases) {
      num_DRM_points++;
      continue;
    }
    invvan_trans = std::move(van_trans);

    const double* normal = owner_cell_element->GetFacetypeNormal(owner_type);
    std::vector<double> cofactor(num_DRM_points * dimension_ * dimension_);
    owner_cell_jacobian->CalJacobianCofMat(num_DRM_points, &phy_points[0],
                                           &cofactor[0]);
    peribdry_normals_[ibdry].resize(num_DRM_points * dimension_);
    for (int ipoint = 0; ipoint < num_DRM_points; ipoint++) {
      gemvAx(1.0, &cofactor[ipoint * dimension_ * dimension_], dimension_,
             normal, 1, 0.0, &peribdry_normals_[ibdry][ipoint * dimension_], 1,
             dimension_, dimension_);
      avocado::VecNormal(dimension_,
                         &peribdry_normals_[ibdry][ipoint * dimension_]);
    }

    neighbor_DRM_points = DRM_points;
    for (int ipoint = 0; ipoint < num_DRM_points; ipoint++)
      for (int idim = 0; idim < dimension_; idim++)
        neighbor_DRM_points[ipoint * dimension_ + idim] += offset[idim];
    break;
  }

  num_peribdry_points_[ibdry] = num_DRM_points;
  peribdry_points_[ibdry] = DRM_points;
  peribdry_owner_basis_value_[ibdry].resize(num_DRM_points * num_bases_);
  peribdry_owner_basis_grad_value_[ibdry].resize(num_DRM_points * num_bases_ *
                                                 dimension_);
  peribdry_neighbor_basis_value_[ibdry].resize(num_DRM_points * num_bases_);
  peribdry_neighbor_basis_grad_value_[ibdry].resize(num_DRM_points *
                                                    num_bases_ * dimension_);
  {
    cell_basis_[owner_index]->GetBasis(num_DRM_points, &DRM_points[0],
                                       &peribdry_owner_basis_value_[ibdry][0]);
    cell_basis_[owner_index]->GetBasisGrad(
        num_DRM_points, &DRM_points[0],
        &peribdry_owner_basis_grad_value_[ibdry][0]);
    cell_basis_[neighbor_index]->GetBasis(
        num_DRM_points, &neighbor_DRM_points[0],
        &peribdry_neighbor_basis_value_[ibdry][0]);
    cell_basis_[neighbor_index]->GetBasisGrad(
        num_DRM_points, &neighbor_DRM_points[0],
        &peribdry_neighbor_basis_grad_value_[ibdry][0]);
  }

  std::vector<double> quad_points, quad_weights, quad_normal;
  std::vector<double> neighbor_quad_points;
  {
    const int quad_order = surface_flux_order_ + order_;

    std::vector<double> ref_points, ref_weights;
    flux_basis->GetBasisPolynomial()->GetSurfaceQuadrature(
        quad_order, bdry_elemtype, elemorder, ref_points, ref_weights);
    const int num_points = static_cast<int>(ref_weights.size());
    std::vector<double> phy_points(num_points * dimension_);
    quad_points.resize(num_points * dimension_);
    owner_cell_element->TransformToPhyCoords(owner_type, num_points,
                                             &ref_points[0], &phy_points[0]);
    owner_cell_jacobian->TransformToPhyCoords(num_points, &phy_points[0],
                                              &quad_points[0]);

    quad_weights.resize(num_points);
    const double* normal = owner_cell_element->GetFacetypeNormal(owner_type);
    std::vector<double> cofactor(num_points * dimension_ * dimension_);
    owner_cell_jacobian->CalJacobianCofMat(num_points, &phy_points[0],
                                           &cofactor[0]);

    const Element* bdry_element = DENEB_ELEMENT->GetElement(bdry_elemtype);
    const double area_ratio = owner_cell_element->GetFacetypeArea(owner_type) /
                              bdry_element->GetVolume();
    quad_normal.resize(num_points * dimension_);
    for (int ipoint = 0; ipoint < num_points; ipoint++) {
      gemvAx(1.0, &cofactor[ipoint * dimension_ * dimension_], dimension_,
             normal, 1, 0.0, &quad_normal[ipoint * dimension_], 1, dimension_,
             dimension_);
      quad_weights[ipoint] =
          avocado::VecLength(dimension_, &quad_normal[ipoint * dimension_]) *
          ref_weights[ipoint] * area_ratio;
      avocado::VecNormal(dimension_, &quad_normal[ipoint * dimension_]);
    }

    neighbor_quad_points = quad_points;
    for (int ipoint = 0; ipoint < num_points; ipoint++)
      for (int idim = 0; idim < dimension_; idim++)
        neighbor_quad_points[ipoint * dimension_ + idim] += offset[idim];
  }

  peribdry_owner_coefficients_[ibdry].resize(num_DRM_points * num_bases_ *
                                             dimension_);
  peribdry_neighbor_coefficients_[ibdry].resize(num_DRM_points * num_bases_ *
                                                dimension_);
  {
    const int num_quad_points = static_cast<int>(quad_weights.size());
    std::vector<double> basis_value(num_quad_points * num_flux_bases);
    flux_basis->GetBasis(num_quad_points, &quad_points[0], &basis_value[0]);

    std::vector<double> nodal_basis_value(num_quad_points * num_DRM_points);
    gemmAB(1.0, &basis_value[0], &invvan_trans[0], 0.0, &nodal_basis_value[0],
           num_quad_points, num_flux_bases, num_DRM_points);

    for (int ipoint = 0; ipoint < num_quad_points; ipoint++)
    {
      cblas_dscal(num_DRM_points, quad_weights[ipoint],
        &nodal_basis_value[ipoint * num_DRM_points], 1);
    }

    std::vector<double> temp(num_quad_points * num_DRM_points * dimension_,
                             0.0);
    for (int ipoint = 0; ipoint < num_quad_points; ipoint++)
      cblas_dger(CBLAS_LAYOUT::CblasRowMajor, num_DRM_points, dimension_, 1.0,
                 &nodal_basis_value[ipoint * num_DRM_points], 1,
                 &quad_normal[ipoint * dimension_], 1,
                 &temp[ipoint * num_DRM_points * dimension_], dimension_);

    std::vector<double> owner_basis_value(num_quad_points * num_bases_);
    std::vector<double> neighbor_basis_value(num_quad_points * num_bases_);
    cell_basis_[owner_index]->GetBasis(num_quad_points, &quad_points[0],
                                       &owner_basis_value[0]);
    cell_basis_[neighbor_index]->GetBasis(
        num_quad_points, &neighbor_quad_points[0], &neighbor_basis_value[0]);

    avocado::Kernel1::f72(&temp[0], &owner_basis_value[0],
                          &peribdry_owner_coefficients_[ibdry][0], dimension_,
                          num_DRM_points, num_quad_points, num_bases_, 1.0,
                          0.0);
    avocado::Kernel1::f72(&temp[0], &neighbor_basis_value[0],
                          &peribdry_neighbor_coefficients_[ibdry][0],
                          dimension_, num_DRM_points, num_quad_points,
                          num_bases_, 1.0, 0.0);
  }
}
void Data::CombinePeribdryToFaceData(void) {
  const std::vector<int> rules = {0,
                                  0,
                                  num_inner_faces_,
                                  num_inner_peribdries_,
                                  num_faces_,
                                  num_peribdries_};
  CombineData(face_owner_cell_, peribdry_owner_cell_, rules);
  CombineData(face_neighbor_cell_, peribdry_neighbor_cell_, rules);
  CombineData(face_owner_type_, peribdry_owner_type_, rules);
  CombineData(face_neighbor_type_, peribdry_neighbor_type_, rules);
  CombineData(num_face_points_, num_peribdry_points_, rules);
  CombineData(face_points_, peribdry_points_, rules);
  CombineData(face_normals_, peribdry_normals_, rules);
  CombineData(face_owner_basis_value_, peribdry_owner_basis_value_, rules);
  CombineData(face_neighbor_basis_value_, peribdry_neighbor_basis_value_,
              rules);
  CombineData(face_owner_basis_grad_value_, peribdry_owner_basis_grad_value_,
              rules);
  CombineData(face_neighbor_basis_grad_value_,
              peribdry_neighbor_basis_grad_value_, rules);
  CombineData(face_owner_coefficients_, peribdry_owner_coefficients_, rules);
  CombineData(face_neighbor_coefficients_, peribdry_neighbor_coefficients_,
              rules);

  num_faces_ += num_peribdries_;
  num_inner_faces_ += num_inner_peribdries_;

  for (int i = num_inner_faces_; i < num_faces_; i++)
    face_neighbor_cell_[i] -= num_cells_;
}
void Data::BuildImplicitData(void) {
  std::vector<std::vector<int>> cell_to_faces(num_cells_);
  for (int iface = 0; iface < num_faces_; iface++)
    cell_to_faces[face_owner_cell_[iface]].push_back(iface);
  for (int iface = 0; iface < num_inner_faces_; iface++)
    cell_to_faces[face_neighbor_cell_[iface]].push_back(iface);

  std::vector<int> subface_ptr;
  std::vector<double> subface_sign;
  std::vector<int> subface_neighbor_cell;
  std::vector<int> subface_num_points;
  std::vector<double*> subface_owner_coefficients;
  std::vector<double*> subface_owner_basis_value;
  std::vector<double*> subface_neighbor_basis_value;
  subface_ptr.push_back(0);

  for (int icell = 0; icell < num_cells_; icell++) {
    for (auto&& iface : cell_to_faces[icell]) {
      bool isowner = true;
      if (iface < num_inner_faces_) {
        const int& owner_cell = face_owner_cell_[iface];
        const int& neighbor_cell = face_neighbor_cell_[iface];
        if (icell == neighbor_cell) isowner = false;
      }
      if (isowner) {
        subface_sign.push_back(1.0);
        if (iface < num_inner_faces_)
          subface_neighbor_cell.push_back(face_neighbor_cell_[iface]);
        else
          subface_neighbor_cell.push_back(face_neighbor_cell_[iface] +
                                          num_cells_);
        subface_num_points.push_back(num_face_points_[iface]);
        subface_owner_coefficients.push_back(
            &face_owner_coefficients_[iface][0]);
        subface_owner_basis_value.push_back(&face_owner_basis_value_[iface][0]);
        subface_neighbor_basis_value.push_back(
            &face_neighbor_basis_value_[iface][0]);
      } else {
        subface_sign.push_back(-1.0);
        subface_neighbor_cell.push_back(face_owner_cell_[iface]);
        subface_num_points.push_back(num_face_points_[iface]);
        subface_owner_coefficients.push_back(
            &face_neighbor_coefficients_[iface][0]);
        subface_owner_basis_value.push_back(
            &face_neighbor_basis_value_[iface][0]);
        subface_neighbor_basis_value.push_back(
            &face_owner_basis_value_[iface][0]);
      }
    }
    subface_ptr.push_back(subface_sign.size());
  }
  cell_to_faces.clear();
  subface_ptr_ = std::move(subface_ptr);
  subface_sign_ = std::move(subface_sign);
  subface_neighbor_cell_ = std::move(subface_neighbor_cell);
  subface_num_points_ = std::move(subface_num_points);
  subface_owner_coefficients_ = std::move(subface_owner_coefficients);
  subface_owner_basis_value_ = std::move(subface_owner_basis_value);
  subface_neighbor_basis_value_ = std::move(subface_neighbor_basis_value);

  std::vector<std::vector<int>> cell_to_bdries(num_cells_);
  for (int ibdry = 0; ibdry < num_bdries_; ibdry++)
    cell_to_bdries[bdry_owner_cell_[ibdry]].push_back(ibdry);

  std::vector<int> subbdry_ptr;
  std::vector<int> subbdry_ind;
  subbdry_ptr.push_back(0);
  for (int icell = 0; icell < num_cells_; icell++) {
    for (auto&& ibdry : cell_to_bdries[icell]) subbdry_ind.push_back(ibdry);
    subbdry_ptr.push_back(subbdry_ind.size());
  }
  cell_to_bdries.clear();

  subbdry_ptr_ = std::move(subbdry_ptr);
  subbdry_ind_ = std::move(subbdry_ind);

  // generate mat_index
  std::vector<int> num_cells(NDOMAIN);
  MPI_Allgather(&num_cells_, 1, MPI_INT, &num_cells[0], 1, MPI_INT,
                MPI_COMM_WORLD);
  std::vector<int> mat_index_offset(NDOMAIN);
  mat_index_offset[0] = 0;
  for (int idomain = 1; idomain < NDOMAIN; idomain++)
    mat_index_offset[idomain] =
        mat_index_offset[idomain - 1] + num_cells[idomain - 1];

  std::vector<int> mat_index(num_outer_cells_);
  for (int icell = 0; icell < num_cells_; icell++)
    mat_index[icell] = icell + mat_index_offset[MYRANK];

  std::vector<std::vector<int>> recv_data;
  AVOCADO_MPI->CommunicateData(outer_send_cell_list_, recv_data);

  for (int idomain = 0; idomain < NDOMAIN; idomain++)
    for (int icell = 0,
             len = static_cast<int>(outer_recv_cell_list_[idomain].size());
         icell < len; icell++)
      mat_index[outer_recv_cell_list_[idomain][icell] + num_cells_] =
          recv_data[idomain][icell] + mat_index_offset[idomain];
  recv_data.clear();
  mat_index_ = move(mat_index);
}
double Data::ComputeProjVolume(const std::vector<double>& coords,
                               const int idim) const {
  int num_points = coords.size() / dimension_;
  if (num_points == 2) {
    switch (idim) {
      case 0:
        return std::abs(coords[dimension_ + 1] - coords[1]);
      case 1:
        return std::abs(coords[dimension_] - coords[0]);
      default:
        return 0.0;
    }
  }
  double sum = 0.0;
  switch (idim) {
    case 0:
      sum += (coords[dimension_ * (num_points - 1) + 1] - coords[1]) *
             (coords[dimension_ * (num_points - 1) + 2] + coords[2]);
      for (int i = 0; i < num_points - 1; i++)
        sum += (coords[dimension_ * i + 1] - coords[dimension_ * (i + 1) + 1]) *
               (coords[dimension_ * i + 2] + coords[dimension_ * (i + 1) + 2]);
      return 0.5 * std::abs(sum);
    case 1:
      sum += (coords[dimension_ * (num_points - 1)] - coords[0]) *
             (coords[dimension_ * (num_points - 1) + 2] + coords[2]);
      for (int i = 0; i < num_points - 1; i++)
        sum += (coords[dimension_ * i] - coords[dimension_ * (i + 1)]) *
               (coords[dimension_ * i + 2] + coords[dimension_ * (i + 1) + 2]);
      return 0.5 * std::abs(sum);
    case 2:
      sum += (coords[dimension_ * (num_points - 1)] - coords[0]) *
             (coords[dimension_ * (num_points - 1) + 1] + coords[1]);
      for (int i = 0; i < num_points - 1; i++)
        sum += (coords[dimension_ * i] - coords[dimension_ * (i + 1)]) *
               (coords[dimension_ * i + 1] + coords[dimension_ * (i + 1) + 1]);
      return 0.5 * std::abs(sum);
  }
  return 0.0;
}

void Data::GetCellQuadrature(const int icell, const int order,
                             std::vector<double>& quad_points,
                             std::vector<double>& quad_weights) const {
  quad_points.clear();
  quad_weights.clear();
  std::vector<double> ref_points;
  cell_basis_[icell]->GetBasisPolynomial()->GetVolumeQuadrature(
      order, cell_elemtype_[icell], cell_order_[icell], ref_points,
      quad_weights);

  const int num_points = static_cast<int>(quad_weights.size());
  std::vector<double> jacobian_det(num_points);
  cell_jacobians_[icell]->CalJacobianDet(num_points, &ref_points[0],
                                         &jacobian_det[0]);
  for (int ipoint = 0; ipoint < num_points; ipoint++)
    quad_weights[ipoint] *= std::abs(jacobian_det[ipoint]);
  quad_points.resize(num_points * dimension_);
  cell_jacobians_[icell]->TransformToPhyCoords(num_points, &ref_points[0],
                                               &quad_points[0]);
}
void Data::GetCellBasisValues(const int icell,
                              const std::vector<double>& coords,
                              std::vector<double>& basis_value) const {
  basis_value.clear();
  const int num_points = static_cast<int>(coords.size()) / dimension_;
  basis_value.resize(num_points * num_bases_);
  cell_basis_[icell]->GetBasis(num_points, &coords[0], &basis_value[0]);
}
void Data::GetCellBasisGradValues(const int icell,
                                  const std::vector<double>& coords,
                                  std::vector<double>& basis_grad_value) const {
  basis_grad_value.clear();
  const int num_points = static_cast<int>(coords.size()) / dimension_;
  basis_grad_value.resize(num_points * num_bases_ * dimension_);
  cell_basis_[icell]->GetBasisGrad(num_points, &coords[0],
                                   &basis_grad_value[0]);
}
void Data::GetCellRefToPhyCoords(const int icell,
                                 const std::vector<double>& ref_coords,
                                 std::vector<double>& phy_coords) const {
  const int num_points = static_cast<int>(ref_coords.size()) / dimension_;
  phy_coords.clear();
  phy_coords.resize(num_points * dimension_);
  cell_jacobians_[icell]->TransformToPhyCoords(num_points, &ref_coords[0],
                                               &phy_coords[0]);
}
void Data::GetFaceCoords(const int icell, const int face_type,
                         std::vector<double>& coords) const {
  std::vector<int> face_nodes =
      cell_element_[icell]->GetFacetypeNodes(cell_order_[icell])[face_type];
  const int& start_index = cell_subnode_ptr_[icell];
  for (auto&& inode : face_nodes)
    inode = cell_subnode_ind_[start_index + inode];
  const int num_nodes = static_cast<int>(face_nodes.size());
  coords.resize(num_nodes * dimension_);
  avocado::VecCopy(num_nodes, &face_nodes[0], &node_coords_[0], dimension_,
                   &coords[0], dimension_);
}
void Data::calWallDistance(const bool finalize, const int num_query_points,
                           std::vector<double>& wall_distance,
                           const std::vector<double>& query_points) {
  START_TIMER();
  MASTER_MESSAGE("Computing wall distance... ");
  static bool initialized = false;
  static int num_wall_bdries = -1;
  static std::vector<std::vector<double>> wall_bdry_coords;
  static std::vector<std::shared_ptr<Jacobian>> wall_bdry_jacobians;
  static int num_wbb = -1;
  static std::vector<std::vector<int>> wbb_to_wbdry;
  static std::vector<double> wbb_center;
  static std::vector<double> wbb_radius;

//  if (initialized == false) {
//    initialized = true;
//    MASTER_MESSAGE("Wall distance module initialize!\n");
//
//    // Wall check
//    std::string boundary_type;
//    std::vector<int> wallbdry_list;
//    for (int ibdry = 0; ibdry < num_bdries_; ibdry++) {
//      CONFIG.getConfigValue("bdry(" + TO_STRING(bdry_type_[ibdry]) + ")", 0,
//                            boundary_type);
//      if (COMPARE(boundary_type, "adiabatic_wall"))
//        wallbdry_list.push_back(ibdry);
//    }
//
//    const int num_local_wallbdries = wallbdry_list.size();
//    std::vector<int> local_wallbdry_orders;
//    std::vector<int> local_wallbdry_elemtypes;
//    std::vector<int> num_nodes_per_local_wallbdry;
//    std::vector<double> local_wallbdry_coords;
//    local_wallbdry_orders.reserve(num_local_wallbdries);
//    local_wallbdry_elemtypes.reserve(num_local_wallbdries);
//    num_nodes_per_local_wallbdry.reserve(num_local_wallbdries);
//    local_wallbdry_coords.reserve(num_local_wallbdries * dimension_ *
//                                  dimension_);
//    for (auto&& ibdry : wallbdry_list) {
//      const int& owner_index = bdry_owner_cell_[ibdry];
//      const int& owner_type = bdry_owner_type_[ibdry];
//
//      std::vector<int> bdry_nodes =
//          cell_element_[owner_index]->getFacetypeNodes(
//              cell_order_[owner_index])[owner_type];
//      const int& start_index = cell_to_subnode_ptr_[owner_index];
//      for (auto&& inode : bdry_nodes)
//        inode = cell_to_subnode_ind_[start_index + inode];
//      const std::vector<double> bdry_sub_coords =
//          getCoords(&bdry_nodes[0], bdry_nodes.size());
//      const ElemType bdry_elemtype =
//          cell_element_[owner_index]->getFaceElemType(owner_type);
//
//      local_wallbdry_orders.push_back(cell_order_[owner_index]);
//      local_wallbdry_elemtypes.push_back(static_cast<int>(bdry_elemtype));
//      num_nodes_per_local_wallbdry.push_back(bdry_nodes.size());
//      for (auto&& coord : bdry_sub_coords)
//        local_wallbdry_coords.push_back(coord);
//    }
//
//    std::vector<std::vector<int>> send_data1;
//    std::vector<std::vector<int>> recv_data1;
//
//    std::vector<int> wallbdry_orders;
//    std::vector<int> wallbdry_elemtypes;
//    std::vector<int> num_nodes_per_wallbdry;
//    std::vector<double> wallbdry_coords;
//
//    send_data1.resize(NDOMAIN);
//    for (int idomain = 0; idomain < NDOMAIN; idomain++)
//      send_data1[idomain] = local_wallbdry_orders;
//    local_wallbdry_orders.clear();
//    Header::dataCommunication(send_data1, recv_data1);
//    send_data1.clear();
//
//    for (int idomain = 0; idomain < NDOMAIN; idomain++)
//      for (auto&& data : recv_data1[idomain]) wallbdry_orders.push_back(data);
//    recv_data1.clear();
//
//    send_data1.resize(NDOMAIN);
//    for (int idomain = 0; idomain < NDOMAIN; idomain++)
//      send_data1[idomain] = local_wallbdry_elemtypes;
//    local_wallbdry_elemtypes.clear();
//    Header::dataCommunication(send_data1, recv_data1);
//    send_data1.clear();
//
//    for (int idomain = 0; idomain < NDOMAIN; idomain++)
//      for (auto&& data : recv_data1[idomain])
//        wallbdry_elemtypes.push_back(data);
//    recv_data1.clear();
//
//    send_data1.resize(NDOMAIN);
//    for (int idomain = 0; idomain < NDOMAIN; idomain++)
//      send_data1[idomain] = num_nodes_per_local_wallbdry;
//    num_nodes_per_local_wallbdry.clear();
//    Header::dataCommunication(send_data1, recv_data1);
//    send_data1.clear();
//
//    for (int idomain = 0; idomain < NDOMAIN; idomain++)
//      for (auto&& data : recv_data1[idomain])
//        num_nodes_per_wallbdry.push_back(data);
//    recv_data1.clear();
//
//    std::vector<std::vector<double>> send_data2;
//    std::vector<std::vector<double>> recv_data2;
//
//    send_data2.resize(NDOMAIN);
//    for (int idomain = 0; idomain < NDOMAIN; idomain++)
//      send_data2[idomain] = local_wallbdry_coords;
//    local_wallbdry_coords.clear();
//    Header::dataCommunication(send_data2, recv_data2);
//    send_data2.clear();
//
//    for (int idomain = 0; idomain < NDOMAIN; idomain++)
//      for (auto&& data : recv_data2[idomain]) wallbdry_coords.push_back(data);
//    recv_data2.clear();
//
//    num_wall_bdries = wallbdry_elemtypes.size();
//    wall_bdry_coords.resize(num_wall_bdries);
//    wall_bdry_jacobians.resize(num_wall_bdries);
//    int start_ind = 0;
//    for (int ibdry = 0; ibdry < num_wall_bdries; ibdry++) {
//      std::shared_ptr<Jacobian> bdry_jacobian =
//          JacobianFactory::getFaceJacobian(
//              static_cast<ElemType>(wallbdry_elemtypes[ibdry]),
//              wallbdry_orders[ibdry]);
//
//      wall_bdry_coords[ibdry].assign(
//          wallbdry_coords.begin() + start_ind,
//          wallbdry_coords.begin() + start_ind +
//              dimension_ * num_nodes_per_wallbdry[ibdry]);
//      start_ind += (dimension_ * num_nodes_per_wallbdry[ibdry]);
//      bdry_jacobian->setCoords(wall_bdry_coords[ibdry]);
//      bdry_jacobian->setTopology();
//
//      wall_bdry_jacobians[ibdry] = bdry_jacobian;
//    }
//    wallbdry_orders.clear();
//    wallbdry_elemtypes.clear();
//    num_nodes_per_wallbdry.clear();
//    wallbdry_coords.clear();
//
//    // wall bounding box
//    std::vector<double> min_wcoord(dimension_, 1.0e+10);
//    std::vector<double> max_wcoord(dimension_, -1.0e+10);
//    for (auto&& coords : wall_bdry_coords) {
//      for (int inode = 0; inode < coords.size() / dimension_; inode++) {
//        for (int idim = 0; idim < dimension_; idim++) {
//          min_wcoord[idim] =
//              std::min(min_wcoord[idim], coords[dimension_ * inode + idim]);
//          max_wcoord[idim] =
//              std::max(max_wcoord[idim], coords[dimension_ * inode + idim]);
//        }
//      }
//    }
//
//    const int nwpart = 20;  // input
//    std::vector<int> nwwet(dimension_);
//    for (int idim = 0; idim < dimension_; idim++)
//      nwwet[idim] = std::pow(nwpart, idim);
//    std::vector<double> hw(dimension_);
//    for (int idim = 0; idim < dimension_; idim++)
//      hw[idim] =
//          (max_wcoord[idim] - min_wcoord[idim]) / static_cast<double>(nwpart);
//
//    // wall bounding box (local)
//    wbb_to_wbdry.resize(std::pow(nwpart, dimension_));
//    for (int ibdry = 0; ibdry < num_wall_bdries; ibdry++) {
//      const std::vector<double>& coords = wall_bdry_coords[ibdry];
//      int wbb_tag = 0;
//      for (int idim = 0; idim < dimension_; idim++) {
//        int loc = std::floor((coords[idim] - min_wcoord[idim]) / hw[idim]);
//        if (loc == nwpart) loc--;
//        wbb_tag += loc * nwwet[idim];
//      }
//      wbb_to_wbdry[wbb_tag].push_back(ibdry);
//    }
//
//    std::vector<std::vector<int>> temp;
//    for (auto&& wbdry_list : wbb_to_wbdry)
//      if (wbdry_list.size() != 0) temp.push_back(wbdry_list);
//    wbb_to_wbdry = move(temp);
//    num_wbb = wbb_to_wbdry.size();
//
//    min_wcoord.clear();
//    max_wcoord.clear();
//    hw.clear();
//    min_wcoord.resize(dimension_ * num_wbb, 1.0e+100);
//    max_wcoord.resize(dimension_ * num_wbb, -1.0e+100);
//    hw.resize(dimension_ * num_wbb);
//    wbb_center.resize(dimension_ * num_wbb);
//    wbb_radius.resize(num_wbb);
//    for (int iwbb = 0; iwbb < num_wbb; iwbb++) {
//      const std::vector<int>& wbdry_list = wbb_to_wbdry[iwbb];
//      for (auto&& ibdry : wbdry_list) {
//        const std::vector<double> coords = wall_bdry_coords[ibdry];
//        for (int inode = 0; inode < coords.size() / dimension_; inode++) {
//          for (int idim = 0; idim < dimension_; idim++) {
//            min_wcoord[iwbb * dimension_ + idim] =
//                std::min(min_wcoord[iwbb * dimension_ + idim],
//                         coords[dimension_ * inode + idim]);
//            max_wcoord[iwbb * dimension_ + idim] =
//                std::max(max_wcoord[iwbb * dimension_ + idim],
//                         coords[dimension_ * inode + idim]);
//          }
//        }
//      }
//      for (int idim = 0; idim < dimension_; idim++) {
//        hw[iwbb * dimension_ + idim] = max_wcoord[iwbb * dimension_ + idim] -
//                                       min_wcoord[iwbb * dimension_ + idim];
//        wbb_center[iwbb * dimension_ + idim] =
//            0.5 * (max_wcoord[iwbb * dimension_ + idim] +
//                   min_wcoord[iwbb * dimension_ + idim]);
//      }
//      wbb_radius[iwbb] = std::sqrt(calDist2(&max_wcoord[iwbb * dimension_],
//                                            &min_wcoord[iwbb * dimension_])) *
//                         0.5;
//    }
//
//    // for debug
//    /*if (MYRANK() == MASTER_NODE) {
//            const int npost = 20;
//            std::vector<double> coords(npost*dimension_);
//            std::vector<double> xstart(dimension_);
//            std::vector<double> xend(dimension_);
//            std::ofstream fout("wbb" + TO_STRING(MYRANK()) + ".dat");
//            for (int iwbb = 0; iwbb < num_wbb; iwbb++) {
//                    if (dimension_ == 2) {
//                            const double& x1 = min_wcoord[iwbb*dimension_ + 0];
//                            const double& y1 = min_wcoord[iwbb*dimension_ + 1];
//                            const double& x2 = max_wcoord[iwbb*dimension_ + 0];
//                            const double& y2 = max_wcoord[iwbb*dimension_ + 1];
//                            const double& c1 = wbb_center[iwbb*dimension_ + 0];
//                            const double& c2 = wbb_center[iwbb*dimension_ + 1];
//                            const double& r = wbb_radius[iwbb];
//
//                            for (int iline = 0; iline < 6; iline++) {
//                                    switch (iline) {
//                                    case 0:	xstart = { x1, y1 }; xend = { x1,
//    y2 }; break; case 1:	xstart = { x1, y1 }; xend = { x2, y1 }; break;
//                                    case 2:	xstart = { x1, y2 }; xend = { x2,
//    y2 }; break; case 3:	xstart = { x2, y1 }; xend = { x2, y2 }; break;
//                                    case 4:	xstart = { c1, c2 }; xend = { c1
//    + r, c2 }; break; case 5:	xstart = { c1, c2 }; xend = { c1, c2 + r };
//    break;
//                                    }
//                                    makeLine(&coords[0], &xstart[0], &xend[0],
//    npost); for (int i = 0; i < npost; i++) { for (int idim = 0; idim <
//    dimension_; idim++) fout << coords[i*dimension_ + idim] << "\t"; fout <<
//    iwbb << std::endl;
//                                    }
//                            }
//                    }
//                    if (dimension_ == 3) {
//                            const double& x1 = min_wcoord[iwbb*dimension_ + 0];
//                            const double& y1 = min_wcoord[iwbb*dimension_ + 1];
//                            const double& z1 = min_wcoord[iwbb*dimension_ + 2];
//                            const double& x2 = max_wcoord[iwbb*dimension_ + 0];
//                            const double& y2 = max_wcoord[iwbb*dimension_ + 1];
//                            const double& z2 = max_wcoord[iwbb*dimension_ + 2];
//                            const double& c1 = wbb_center[iwbb*dimension_ + 0];
//                            const double& c2 = wbb_center[iwbb*dimension_ + 1];
//                            const double& c3 = wbb_center[iwbb*dimension_ + 2];
//                            const double& r = wbb_radius[iwbb];
//
//                            for (int iline = 0; iline < 15; iline++) {
//                                    switch (iline) {
//                                    case 0:	xstart = { x1, y1, z1 }; xend = {
//    x1, y2, z1 }; break;
//                                    case 1:	xstart = { x1, y1, z1 }; xend = {
//    x2, y1, z1 }; break;
//                                    case 2:	xstart = { x1, y2, z1 }; xend = {
//    x2, y2, z1 }; break;
//                                    case 3:	xstart = { x2, y1, z1 }; xend = {
//    x2, y2, z1 }; break;
//                                    case 4:	xstart = { x1, y1, z2 }; xend = {
//    x1, y2, z2 }; break;
//                                    case 5:	xstart = { x1, y1, z2 }; xend = {
//    x2, y1, z2 }; break;
//                                    case 6:	xstart = { x1, y2, z2 }; xend = {
//    x2, y2, z2 }; break;
//                                    case 7:	xstart = { x2, y1, z2 }; xend = {
//    x2, y2, z2 }; break;
//                                    case 8:	xstart = { x1, y1, z1 }; xend = {
//    x1, y1, z2 }; break;
//                                    case 9:	xstart = { x1, y2, z1 }; xend = {
//    x1, y2, z2 }; break; case 10: xstart = { x2, y1, z1 }; xend = { x2, y1, z2
//    }; break; case 11: xstart = { x2, y2, z1 }; xend = { x2, y2, z2 }; break;
//                                    case 12: xstart = { c1, c2, c3 }; xend = {
//    c1 + r, c2, c3 }; break; case 13: xstart = { c1, c2, c3 }; xend = { c1, c2 +
//    r, c3 }; break; case 14: xstart = { c1, c2, c3 }; xend = { c1, c2, c3 + r };
//    break;
//                                    }
//                                    makeLine(&coords[0], &xstart[0], &xend[0],
//    npost); for (int i = 0; i < npost; i++) { for (int idim = 0; idim <
//    dimension_; idim++) fout << coords[i*dimension_ + idim] << "\t"; fout <<
//    iwbb << std::endl;
//                                    }
//                            }
//                    }
//            }
//            fout.close();
//    }*/
//    // end for debug
//  }
//
//  if (num_wall_bdries == 0) {
//    wall_distance.resize(num_query_points, 1.0e+100);
//  } else {
//    // query point search
//    wall_distance.resize(num_query_points);
//    std::vector<double> max_dist_to_wbb(num_wbb);
//    for (int ipoint = 0; ipoint < num_query_points; ipoint++) {
//      const double* coord = &query_points[ipoint * dimension_];
//
//      for (int iwbb = 0; iwbb < num_wbb; iwbb++)
//        max_dist_to_wbb[iwbb] =
//            std::sqrt(calDist2(&wbb_center[iwbb * dimension_], coord)) +
//            wbb_radius[iwbb];
//      const double max_dist =
//          *std::max_element(max_dist_to_wbb.begin(), max_dist_to_wbb.end());
//
//      double min_dist2 = 1.0e+100;
//      std::vector<int> bdry_candidates;
//      for (int iwbb = 0; iwbb < num_wbb; iwbb++) {
//        if (max_dist < max_dist_to_wbb[iwbb] - 2.0 * wbb_radius[iwbb]) continue;
//        for (auto&& ibdry : wbb_to_wbdry[iwbb]) {
//          const std::vector<double>& bdry_coords = wall_bdry_coords[ibdry];
//          const int num_node = bdry_coords.size() / dimension_;
//
//          double min_dist2_local = 1.0e+10;
//          for (int inode = 0; inode < num_node; inode++) {
//            const double dist2 =
//                calDist2(&bdry_coords[inode * dimension_], coord);
//            if (dist2 < min_dist2_local) min_dist2_local = dist2;
//          }
//
//          if (std::abs(min_dist2_local - min_dist2) < 1.0e-8) {
//            bdry_candidates.push_back(ibdry);
//          } else if (min_dist2_local < min_dist2) {
//            min_dist2 = min_dist2_local;
//            bdry_candidates.clear();
//            bdry_candidates.push_back(ibdry);
//          }
//        }
//      }
//
//      for (auto&& ibdry : bdry_candidates) {
//        const double dist2 = calMinimumDist2(wall_bdry_jacobians[ibdry], coord);
//        min_dist2 = std::min(min_dist2, dist2);
//      }
//
//      wall_distance[ipoint] = std::sqrt(min_dist2);
//    }
//  }
//
//  if (finalize == true) {
//    num_wall_bdries = -1;
//    wall_bdry_coords.clear();
//    wall_bdry_jacobians.clear();
//    num_wbb = -1;
//    wbb_to_wbdry.clear();
//    wbb_center.clear();
//    wbb_radius.clear();
//    initialized = false;
//    MASTER_MESSAGE("Wall distance module finalize!\n");
//  }
//
//  SYNCRO();
//  MASTER_MESSAGE("Complete. (Time: " + std::to_string(STOP_TIMER()) + "s)\n");
//  int num_global_query_points;
//  MPI_Allreduce(&num_query_points, &num_global_query_points, 1, MPI_INT,
//                MPI_SUM, MPI_COMM_WORLD);
//  MASTER_MESSAGE("The number of query points: " +
//                 std::to_string(num_global_query_points) + "\n");
}
}  // namespace deneb