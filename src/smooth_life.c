#include <float.h>
#include <math.h>
#include <stddef.h>

double clamp2(double x, double min, double max)
{
  if (x < min) x = min;
  if (x > max) x = max;
  return x;
}

void logistic_threshold(double* x, double* x_out, size_t length, double x0, double alpha) {
  for (size_t i = 0; i< length; i++) {
    x_out[i] = 1.0 / (1.0 + exp(-4.0 / alpha * (x[i] - x0)));
  }
}

void hard_threshold(double* x, double* x_out, size_t length, double x0) {
  for (size_t i = 0; i< length; i++) {
    x_out[i] = x[i] > x0 ? 1.0 : 0.0;
  }
}

void linearized_threshold(double* x, double* x_out, size_t length, double x0, double alpha) {
  for (size_t i = 0; i < length; i++) {
    x_out[i] = clamp2((x[i] - x0) / alpha + 0.5, 0.0, 1.0);
  }
}

void logistic_interval(double *x, double *x_out, size_t length, double a,
                       double b, double alpha) {
  for (size_t i = 0; i< length; i++) {
    x_out[i] = 1.0 / (1.0 + exp(-4.0 / alpha * (x[i] - a))) * (1.0 - (1.0 / (1.0 + exp(-4.0 / alpha * (x[i] - b)))));
  }
}

void linearized_interval(double *x, double *x_out, size_t length, double a,
                       double b, double alpha) {
  for (size_t i = 0; i< length; i++) {
    x_out[i] = clamp2((x[i] - a) / alpha + 0.5, 0.0, 1.0) * (1.0 - (clamp2((x[i] - b) / alpha + 0.5, 0.0, 1.0)));
  }
}

void lerp (double *a, double *b, double *t, double *x_out, size_t length) {
  for (size_t i = 0; i < length; i++) {
    x_out[i] = (1.0 - t[i]) * a[i] + t[i] * b[i];
  }
}
