#include "deneb_utility.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "avocado.h"
#include "deneb_data.h"
#include "deneb_equation.h"

namespace deneb {
std::shared_ptr<Utility> DENEB_UTILITY_NAME = nullptr;
void Utility::ComputeError(
    const std::string& filename, const double* solution, const double* exact_solution) {
  MASTER_MESSAGE(avocado::GetTitle("Utility::ComputeError()"));

  START_TIMER();
  const int& order = DENEB_DATA->GetOrder();
  const int& num_bases = DENEB_DATA->GetNumBases();
  const int& num_cells = DENEB_DATA->GetNumCells();
  const int& num_states = DENEB_EQUATION->GetNumStates();
  const int& dimension = DENEB_EQUATION->GetDimension();
  const int sb = num_states * num_bases;
  const std::vector<double>& cell_volumes = DENEB_DATA->GetCellVolumes();

  double local_volume = 0.0;
  std::vector<std::vector<double>> quad_basis_value(num_cells);
  std::vector<std::vector<double>> quad_weights(num_cells);
  std::vector<std::vector<double>> quad_points(num_cells);
  std::vector<int> num_cell_points(num_cells);
  for (int icell = 0; icell < num_cells; icell++) {
    DENEB_DATA->GetCellQuadrature(icell, order, quad_points[icell],
                                  quad_weights[icell]);
    DENEB_DATA->GetCellBasisValues(icell, quad_points[icell],
                                   quad_basis_value[icell]);
    num_cell_points[icell] = quad_weights[icell].size();
    local_volume += cell_volumes[icell];
  }

  int max_num_cell_points = 0;
  if (num_cell_points.size() != 0)
    max_num_cell_points =
        *std::max_element(num_cell_points.begin(), num_cell_points.end());

  int ind = 0;
  std::vector<double> error(3 * num_states, 0.0);
  std::vector<double> local_simulated_solution(max_num_cell_points *
                                               num_states);
  std::vector<double> local_exact_solution(max_num_cell_points * num_states);
  for (int icell = 0; icell < num_cells; icell++) {
    const int& num_points = num_cell_points[icell];

    avocado::Kernel0::f4(&solution[icell * sb], &quad_basis_value[icell][0],
                         &local_simulated_solution[0], num_states, num_bases,
                         num_points, 1.0, 0.0);
    avocado::Kernel0::f4(&exact_solution[icell * sb], &quad_basis_value[icell][0],
                         &local_exact_solution[0], num_states, num_bases,
                         num_points, 1.0, 0.0);

    for (int ipoint = 0; ipoint < num_points; ipoint++) {
      ind = 0;
      // L1 error
      for (int istate = 0; istate < num_states; istate++) {
        error[ind++] += std::abs(local_simulated_solution[istate] -
                                 local_exact_solution[istate]) *
                        quad_weights[icell][ipoint];
      }

      // L2 error
      for (int istate = 0; istate < num_states; istate++) {
        error[ind++] += std::pow(std::abs(local_simulated_solution[istate] -
                                          local_exact_solution[istate]),
                                 2.0) *
                        quad_weights[icell][ipoint];
      }

      // Linf error
      for (int istate = 0; istate < num_states; istate++) {
        error[ind] = std::max(std::abs(local_simulated_solution[istate] -
                                       local_exact_solution[istate]),
                              error[ind]);
        ind++;
      }
    }
  }
  std::vector<double> global_error(3 * num_states, 0.0);
  MPI_Allreduce(&error[0], &global_error[0], 2 * num_states, MPI_DOUBLE,
                MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&error[2 * num_states], &global_error[2 * num_states],
                num_states, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  double global_volume = 0.0;
  MPI_Allreduce(&local_volume, &global_volume, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  for (int istate = 0; istate < num_states; istate++) {
    global_error[istate] = global_error[istate] / global_volume;
  }
  for (int istate = num_states; istate < 2 * num_states; istate++) {
    global_error[istate] = std::sqrt(global_error[istate] / global_volume);
  }
  const double cost = STOP_TIMER();
  MASTER_MESSAGE("Computing cost: " + std::to_string(cost) + "\n");

  // Printing message
  std::stringstream ss1, ss2, ss3;
  ss1 << "L1 error=" << std::scientific << std::setprecision(8);
  ss1 << global_error[0];
  ss2 << "L2 error=" << std::scientific << std::setprecision(8);
  ss2 << global_error[num_states];
  ss3 << "Linf error=" << std::scientific << std::setprecision(8);
  ss3 << global_error[2 * num_states];
  for (int istate = 1; istate < num_states; istate++) {
    ss1 << " | " << global_error[istate];
    ss2 << " | " << global_error[num_states + istate];
    ss3 << " | " << global_error[2 * num_states + istate];
  }
  MASTER_MESSAGE(ss1.str() + "\n");
  MASTER_MESSAGE(ss2.str() + "\n");
  MASTER_MESSAGE(ss3.str() + "\n");

  if (MYRANK == MASTER_NODE) {
    avocado::MakeDirectory(filename);
    std::ofstream file(filename, std::ios::trunc);
    if (!file.is_open())
      ERROR_MESSAGE("Save file is not opened: " + filename + "\n");
    file.precision(12);
    file << "L1  ";
    for (int i = 0; i < num_states; i++) file << " \t " << global_error[i];
    file << "\nL2  ";
    for (int i = 0; i < num_states; i++)
      file << " \t " << global_error[i + num_states];
    file << "\nLinf";
    for (int i = 0; i < num_states; i++)
      file << " \t " << global_error[i + 2 * num_states];

    file.close();
  }
}
}  // namespace deneb