#include "avocado_dual_number_alloc.h"

aDual::aDual() : dim(0), f(0.0), df() {}
aDual::aDual(const int dim) : dim(dim), f(0.0), df(dim, 0.0) {}
aDual::aDual(const int dim, const double fvar) : dim(dim), f(fvar), df(dim, 0.0) {}
aDual::aDual(const int dim, const double fvar, const int dimvar)
  : dim(dim), f(fvar), df(dim, 0.0) {
  df[dimvar] = 1.0;
}
aDual aDual::operator-() const {
  aDual var(dim, -f);
  for (int idim = 0; idim < dim; idim++) var.df[idim] = -df[idim];
  return var;
}

aDual operator+(const aDual& left, const aDual& right) {
  aDual var(left.dim, left.f + right.f);
  for (int idim = 0; idim < left.dim; idim++)
    var.df[idim] = left.df[idim] + right.df[idim];
  return var;
}
aDual operator-(const aDual& left, const aDual& right) {
  aDual var(left.dim, left.f - right.f);
  for (int idim = 0; idim < left.dim; idim++)
    var.df[idim] = left.df[idim] - right.df[idim];
  return var;
}
aDual operator*(const aDual& left, const aDual& right) {
  aDual var(left.dim, left.f * right.f);
  for (int idim = 0; idim < left.dim; idim++)
    var.df[idim] = left.f * right.df[idim] + left.df[idim] * right.f;
  return var;
}
aDual operator/(const aDual& left, const aDual& right) {
  const double t1 = 1.0 / right.f;
  aDual var(left.dim, left.f * t1);
  const double t3 = var.f * t1;
  for (int idim = 0; idim < left.dim; idim++)
    var.df[idim] = t1 * left.df[idim] - t3 * right.df[idim];
  return var;
}
bool operator<(const aDual& left, const aDual& right) {
  return (left.f < right.f);
}
bool operator>(const aDual& left, const aDual& right) {
  return (left.f > right.f);
}
bool operator<=(const aDual& left, const aDual& right) {
  return (left.f <= right.f);
}
bool operator>=(const aDual& left, const aDual& right) {
  return (left.f >= right.f);
}
aDual operator+(const double& left, const aDual& right) {
  aDual var = right;
  var.f += left;
  return var;
}
aDual operator-(const double& left, const aDual& right) {
  aDual var = -right;
  var.f += left;
  return var;
}
aDual operator*(const double& left, const aDual& right) {
  aDual var(right.dim, left * right.f);
  for (int idim = 0; idim < right.dim; idim++) var.df[idim] = left * right.df[idim];
  return var;
}
aDual operator/(const double& left, const aDual& right) {
  const double t1 = 1.0 / right.f;
  aDual var(right.dim, left * t1);
  const double t3 = var.f * t1;
  for (int idim = 0; idim < right.dim; idim++) var.df[idim] = -t3 * right.df[idim];
  return var;
}
bool operator<(const double& left, const aDual& right) {
  return (left < right.f);
}
bool operator>(const double& left, const aDual& right) {
  return (left > right.f);
}
bool operator<=(const double& left, const aDual& right) {
  return (left <= right.f);
}
bool operator>=(const double& left, const aDual& right) {
  return (left >= right.f);
}
aDual operator+(const aDual& left, const double& right) {
  aDual var = left;
  var.f += right;
  return var;
}
aDual operator-(const aDual& left, const double& right) {
  aDual var = left;
  var.f -= right;
  return var;
}
aDual operator*(const aDual& left, const double& right) {
  aDual var(left.dim, left.f * right);
  for (int idim = 0; idim < left.dim; idim++) var.df[idim] = left.df[idim] * right;
  return var;
}
aDual operator/(const aDual& left, const double& right) {
  const double t1 = 1.0 / right;
  aDual var(left.dim, left.f * t1);
  for (int idim = 0; idim < left.dim; idim++) var.df[idim] = t1 * left.df[idim];
  return var;
}
bool operator<(const aDual& left, const double& right) {
  return (left.f < right);
}
bool operator>(const aDual& left, const double& right) {
  return (left.f > right);
}
bool operator<=(const aDual& left, const double& right) {
  return (left.f <= right);
}
bool operator>=(const aDual& left, const double& right) {
  return (left.f >= right);
}

template <typename T>
int sgn(T val) {
  return (T(0) < val) - (val < T(0));
}

namespace std {
  aDual abs(const aDual& var) {
    const int sign = sgn(var.f);
    aDual result(var.dim, std::abs(var.f));
    for (int idim = 0; idim < var.dim; idim++) result.df[idim] = sign * var.df[idim];
    return result;
  }
  aDual fabs(const aDual& var) {
    const int sign = sgn(var.f);
    aDual result(var.dim, std::fabs(var.f));
    for (int idim = 0; idim < var.dim; idim++) result.df[idim] = sign * var.df[idim];
    return result;
  }
  aDual sqrt(const aDual& var) {
    aDual result(var.dim, std::sqrt(var.f));
    const double t = 0.5 / result.f;
    for (int idim = 0; idim < var.dim; idim++) result.df[idim] = t * var.df[idim];
    return result;
  }
  aDual cos(const aDual& var) {
    aDual result(var.dim, std::cos(var.f));
    const double t = -std::sin(var.f);
    for (int idim = 0; idim < var.dim; idim++) result.df[idim] = t * var.df[idim];
    return result;
  }
  aDual sin(const aDual& var) {
    aDual result(var.dim, std::sin(var.f));
    const double t = std::cos(var.f);
    for (int idim = 0; idim < var.dim; idim++) result.df[idim] = t * var.df[idim];
    return result;
  }
  aDual max(const aDual& left, const aDual& right) {
    if (left.f > right.f)
      return left;
    else
      return right;
  }
  aDual max(const double& left, const aDual& right) {
    if (left > right.f)
      return aDual(right.dim, left);
    else
      return right;
  }
  aDual max(const aDual& left, const double& right) {
    if (left.f > right)
      return left;
    else
      return aDual(left.dim, right);
  }
  aDual min(const aDual& left, const aDual& right) {
    if (left.f < right.f)
      return left;
    else
      return right;
  }
  aDual min(const double& left, const aDual& right) {
    if (left < right.f)
      return aDual(right.dim, left);
    else
      return right;
  }
  aDual min(const aDual& left, const double& right) {
    if (left.f < right)
      return left;
    else
      return aDual(left.dim, right);
  }
  aDual pow(const aDual& var, const double& exp) {
    aDual result(var.dim, std::pow(var.f, exp));
    const double t = exp * std::pow(var.f, exp - 1.0);
    for (int idim = 0; idim < var.dim; idim++) result.df[idim] = t * var.df[idim];
    return result;
  }
  aDual pow(const aDual& var, const int& exp) {
    aDual result(var.dim, std::pow(var.f, exp));
    const double t = exp * std::pow(var.f, exp - 1.0);
    for (int idim = 0; idim < var.dim; idim++) result.df[idim] = t * var.df[idim];
    return result;
  }
  aDual exp(const aDual& var) {
    const double t = std::exp(var.f);
    aDual result(var.dim, t);
    for (int idim = 0; idim < var.dim; idim++) result.df[idim] = t * var.df[idim];
    return result;
  }
  aDual atan(const aDual& var) {
    const double t = 1.0 / (1.0 + var.f * var.f);
    aDual result(var.dim, std::atan(var.f));
    for (int idim = 0; idim < var.dim; idim++) result.df[idim] = t * var.df[idim];
    return result;
  }
  aDual tanh(const aDual& var) {
    const double t = 1.0 - var.f * var.f;
    aDual result(var.dim, std::tanh(var.f));
    for (int idim = 0; idim < var.dim; idim++) result.df[idim] = t * var.df[idim];
    return result;
  }
}  // namespace std