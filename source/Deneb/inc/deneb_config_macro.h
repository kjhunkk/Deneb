#pragma once

// Basic macro for programming
#define RESOURCE_DIR "ResourceDir"
#define RETURN_DIR "ReturnDir"
#define RETURN_CONFIG_DIR "config/"
#define RETURN_POST_DIR "post/"
#define RETURN_SAVE_DIR "save/"

#define ORDER "Order"
#define VOLUME_FLUX_ORDER "FluxOrder.0"
#define SURFACE_FLUX_ORDER "FluxOrder.1"
#define MASS_MATRIX_ORDER "MassMatrixOrder.0"
#define GRID_FILE_FORMAT "Grid.0"
#define GRID_FILE_PATH "Grid.1"
#define CHARACTERISTIC_LENGTH "CharacteristicLength.0"

#define EQUATION "Equation.0"
#define CONVECTIVE_FLUX "Equation.1"
#define PROBLEM "Problem"

#define TIMESCHEME "Timescheme"
#define TIMESTEP_CONTROL "Timestep.0"
#define TIMESTEP_CONTROL_VALUE "Timestep.1"
#define TIME_RESOLUTION "TimeResolution"
#define GMRES_SEARCH_DIRECTION "GMRES.0"
#define GMRES_ERROR_TOL "GMRES.1"
#define GMRES_MAX_ITERATION "GMRES.2"
#define CFL_INCREASE_INTERVAL "IncreaseCFL.0"
#define CFL_INCREASE_AMOUNT "IncreaseCFL.1"
#define CFL_INCREASE_MAX "IncreaseCFL.2"
#define STEADY_CONVERGENCE_TOL "SteadyConvergenceTol"

#define RESTART "Restart.0"
#define RESTART_PATH "Restart.1"
#define CELL_POST_ORDER "PostOrder.0"
#define FACE_POST_ORDER "PostOrder.1"
#define POST_CONTROL "Post.0"
#define POST_CONTROL_VALUE(num) "Post." + std::to_string(num + 1)

#define SAVE_CONTROL "Save.0"
#define SAVE_CONTROL_VALUE(num) "Save." + std::to_string(num + 1)

#define STOP_CONDITION "Stop.0"
#define STOP_CONTROL_VALUE "Stop.1"
#define MAX_ITERATION "MaxIter"

#define NEWTON_ERROR_TOL "Newton.0"
#define NEWTON_MAX_ITERATION "Newton.1"

#define JACOBIAN_RECOMPUTE_TOL "JacobianRecompute.0"
#define JACOBIAN_RECOMPUTE_ITERATION "JacobianRecompute.1"

#define ROLL_BACK_BUFFER_SIZE "RollBack.0"
#define ROLL_BACK_MECHANISM "RollBack.1"
#define ROLL_BACK_MECHANISM_INPUT_I(i) "RollBack." + std::to_string(i + 2)

#define LIMITER "Limiter"
#define PRESSUREFIX "Pressurefix"

// artificial viscosity
#define ARTIFICIAL_VISCOSITY "ArtificialViscosity"
#define PECLET ARTIFICIAL_VISCOSITY ".1"
#define KAPPA ARTIFICIAL_VISCOSITY ".2"
// shock-capturing PID
#define S_GAIN ARTIFICIAL_VISCOSITY ".1"
#define P_GAIN ARTIFICIAL_VISCOSITY ".2"
#define I_GAIN ARTIFICIAL_VISCOSITY ".3"
#define D_GAIN ARTIFICIAL_VISCOSITY ".4"
#define DUCROS_SWITCH ARTIFICIAL_VISCOSITY ".5"

#define SHOCK_CAPTURING_FREEZE_ITERATION "ShockCapturingFreezeIter"

// gmsh grid generation
#define GEN_GMSH_LEFT(direction) \
  std::string("GenGmsh") + direction + "Options.0"
#define GEN_GMSH_RIGHT(direction) \
  std::string("GenGmsh") + direction + "Options.1"
#define GEN_GMSH_DIVISION(direction) \
  std::string("GenGmsh") + direction + "Options.2"
#define GEN_GMSH_LEFT_TAG(direction) \
  std::string("GenGmsh") + direction + "Options.3"
#define GEN_GMSH_RIGHT_TAG(direction) \
  std::string("GenGmsh") + direction + "Options.4"

// periodic bc
#define MATCHING_BC "Matching"
#define DIRECTION(direction) std::string(1, (direction) + 88)
#define PERIODIC "Periodic"
#define PERIODIC_LEFT(direction) DIRECTION(direction) + PERIODIC ".0"
#define PERIODIC_RIGHT(direction) DIRECTION(direction) + PERIODIC ".1"
#define PERIODIC_OFFSET(direction) DIRECTION(direction) + PERIODIC ".2"

// bc
#define BDRY(tag) "Bdry(" + std::to_string(tag) + ")"
#define BDRY_TYPE(tag) BDRY(tag) ".0"
#define BDRY_NAME(tag) BDRY(tag) ".1"
#define BDRY_INPUT(tag) "BdryInput(" + std::to_string(tag) + ")"
#define BDRY_INPUT_I(tag, i) BDRY_INPUT(tag) + "." + std::to_string(i)

// equation
#define HEAT_CAPACITY_RATIO "Gamma"
#define MACH_NUMBER "Ma"
#define INCIDENT_ANGLE "AOA"
#define SIDESLIP_ANGLE "Sideslip"
#define REYNOLDS_NUMBER "Re"
#define PRANDTL_NUMBER "Pr"
#define REFERENCE_LENGTH "L_ref"
#define REFERENCE_DENSITY "rho_ref"
#define REFERENCE_TEMPERATURE "T_ref"
#define REFERENCE_SOUNDSPEED "a_ref"
#define REFERENCE_VELOCITY "V_ref"
#define REFERENCE_VISCOSITY "mu_ref"
#define REFERENCE_CONDUCTIVITY "k_ref"
#define IDEA_RESOURCE_DIRECTORY "IDEAResourceDir"

// problem
#define PROBLEM_INPUT "ProblemInput"
#define PROBLEM_INPUT_I(i) PROBLEM_INPUT "." + std::to_string(i)

// source
#define SOURCE "Source"