#pragma once

#include <algorithm>
#include <cstring>
#include <cmath>
#include <vector>

struct aDual {
  int dim;
  double f;
  std::vector<double> df;

  aDual();
  aDual(const int dim);
  aDual(const int dim, const double fvar);
  aDual(const int dim, const double fvar, const int dimvar);
  aDual operator-() const;
};

aDual operator+(const aDual& left, const aDual& right);
aDual operator-(const aDual& left, const aDual& right);
aDual operator*(const aDual& left, const aDual& right);
aDual operator/(const aDual& left, const aDual& right);
bool operator<(const aDual& left, const aDual& right);
bool operator>(const aDual& left, const aDual& right);
bool operator<=(const aDual& left, const aDual& right);
bool operator>=(const aDual& left, const aDual& right);
aDual operator+(const double& left, const aDual& right);
aDual operator-(const double& left, const aDual& right);
aDual operator*(const double& left, const aDual& right);
aDual operator/(const double& left, const aDual& right);
bool operator<(const double& left, const aDual& right);
bool operator>(const double& left, const aDual& right);
bool operator<=(const double& left, const aDual& right);
bool operator>=(const double& left, const aDual& right);
aDual operator+(const aDual& left, const double& right);
aDual operator-(const aDual& left, const double& right);
aDual operator*(const aDual& left, const double& right);
aDual operator/(const aDual& left, const double& right);
bool operator<(const aDual& left, const double& right);
bool operator>(const aDual& left, const double& right);
bool operator<=(const aDual& left, const double& right);
bool operator>=(const aDual& left, const double& right);

namespace std {
  aDual abs(const aDual& var);
  aDual fabs(const aDual& var);
  aDual sqrt(const aDual& var);
  aDual cos(const aDual& var);
  aDual sin(const aDual& var);
  aDual max(const aDual& left, const aDual& right);
  aDual max(const double& left, const aDual& right);
  aDual max(const aDual& left, const double& right);
  aDual min(const aDual& left, const aDual& right);
  aDual min(const double& left, const aDual& right);
  aDual min(const aDual& left, const double& right);
  aDual pow(const aDual& var, const double& exp);
  aDual pow(const aDual& var, const int& exp);
  aDual exp(const aDual& var);
  aDual atan(const aDual& var);
  aDual tanh(const aDual& var);
}  // namespace std