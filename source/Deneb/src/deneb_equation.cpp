#include "deneb_equation.h"

#include "avocado.h"
#include "deneb_data.h"
#include "deneb_equation_advection2d.h"
#include "deneb_equation_burgers2d.h"
#include "deneb_equation_equilibriumns2d.h"
#include "deneb_equation_euler2d.h"
#include "deneb_equation_euler3d.h"
#include "deneb_equation_glmmhd2d.h"
#include "deneb_equation_ns2d.h"
#include "deneb_equation_ns3d.h"

namespace deneb {
std::shared_ptr<Equation> DENEB_EQUATION_NAME = nullptr;
std::shared_ptr<Equation> Equation::GetEquation(const std::string& name) {
  if (!name.compare("NS2D"))
    return std::make_shared<EquationNS2D>();
  else if (!name.compare("EquilibriumNS2D"))
    return std::make_shared<EquationEquilibriumNS2D>();
  else if (!name.compare("Euler2D"))
    return std::make_shared<EquationEuler2D>();
  else if (!name.compare("Euler3D"))
    return std::make_shared<EquationEuler3D>();
  else if (!name.compare("GLMMHD2D"))
    return std::make_shared<EquationGLMMHD2D>();
  else if (!name.compare("NS3D"))
    return std::make_shared<EquationNS3D>();
  else if (!name.compare("Advection2D"))
    return std::make_shared<EquationAdvection2D>();
  else if (!name.compare("Burgers2D"))
    return std::make_shared<EquationBurgers2D>();
  else
    ERROR_MESSAGE("Wrong equation (no-exist):" + name + "\n");
  return nullptr;
}

Equation::Equation(const int dimension, const int num_states,
                   const bool source_term)
    : dimension_(dimension),
      num_states_(num_states),
      source_term_(source_term) {}

void Equation::SystemMatrixShift(const double* solution, Mat& sysmat,
  const std::vector<double>& local_dt, const double t)
{
  static const int& num_cells = DENEB_DATA->GetNumCells();
  static const int& num_bases = DENEB_DATA->GetNumBases();
  static const auto& mat_index = DENEB_DATA->GetMatIndex();
  static const int sb = num_states_ * num_bases;
  static std::vector<double> block(sb * sb);
  for (int icell = 0; icell < num_cells; icell++) {
    memset(&block[0], 0, sb * sb * sizeof(double));
    const double dt_factor = 1.0 / local_dt[icell];

    for (int i = 0; i < sb * sb; i += (sb + 1)) block[i] = dt_factor;

    MatSetValuesBlocked(sysmat, 1, &mat_index[icell], 1, &mat_index[icell],
      &block[0], ADD_VALUES);
  }
  MatAssemblyBegin(sysmat, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(sysmat, MAT_FINAL_ASSEMBLY);
}
}  // namespace deneb