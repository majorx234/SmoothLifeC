#include <stddef.h>
#include <stdbool.h>
#include <stdio.h>
#include <assert.h>
#include "smooth_life.h"

bool test_clamp2() {
  assert(clamp2(5.0, 0.0, 1.0) == 1.0);
  assert(clamp2(-2.3, 0.0, 1.0) == 0.0);
  assert(clamp2(0.6, 0.0, 1.0) == 0.6);
  printf("All clamp tests passed.\n");
  return true;
}

bool test_logistic_threshold() {
  size_t length = 6;
  double input[6] = {1.2, 2.2, -1.2, 0.2, 0.9, 6.4};
  double output[6] = {0};
  double expected[6] = { 0.68997448, 0.81255021, 0.31002552, 0.53328404, 0.64565631, 0.98616561 };
  double x0 = 0.0;
  double alpha = 6.0;
  logistic_threshold(input, output, length, x0, alpha);
  double epsilon;
  for (size_t i = 0; i < length; i++) {
    double error = output[i] - expected[i];
    epsilon += error * error;
  }
  assert( epsilon < 0.00001);
  printf("All logistic_threshold tests passed.\n");
  return true;
}

int main(int argc, char **argv) {
  test_clamp2();
  test_logistic_threshold();
  printf("All tests passed\n");
}

