#include "deneb_rollback.h"

#include "avocado.h"
#include "deneb_config_macro.h"
#include "deneb_data.h"
#include "deneb_equation.h"

namespace deneb {
ConstantRollBack::ConstantRollBack() {
  auto& config = AVOCADO_CONFIG;
  const std::string& mechanism = config->GetConfigValue(ROLL_BACK_MECHANISM);

  buffer_size_ = std::stoi(config->GetConfigValue(ROLL_BACK_BUFFER_SIZE));
  buffer_.resize(buffer_size_);
  const int& num_cells = DENEB_DATA->GetNumCells();
  const int& num_states = DENEB_EQUATION->GetNumStates();
  const int& num_bases = DENEB_DATA->GetNumBases();
  length_ = num_cells * num_states * num_bases;
  for (int i = 0; i < buffer_size_; i++) {
    buffer_[i].resize(length_);
  }
  buffer_pointer_ = 0;

  roll_back_step_ =
      std::stoi(config->GetConfigValue(ROLL_BACK_MECHANISM_INPUT_I(0)));
}
void ConstantRollBack::ExecuteRollBack(double* solution, int& iteration) {
  iteration -= roll_back_step_;
  buffer_pointer_ = (buffer_pointer_ - roll_back_step_) % buffer_size_;
  for (int i = 0; i < length_; i++) {
    solution[i] = buffer_[buffer_pointer_][i];
  }
}
void ConstantRollBack::UpdateBuffer(double* solution) {
  for (int i = 0; i < length_; i++) {
    buffer_[buffer_pointer_][i] = solution[i];
  }
  buffer_pointer_ = (buffer_pointer_ + 1) % buffer_size_;
}
}  // namespace deneb