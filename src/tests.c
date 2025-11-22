#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>
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
  double epsilon = 0.0;
  for (size_t i = 0; i < length; i++) {
    double error = output[i] - expected[i];
    epsilon += error * error;
  }
  assert( epsilon < 0.00001);
  printf("All logistic_threshold tests passed.\n");
  return true;
}

bool test_hard_threshold() {
  size_t length = 6;
  double input[6] = {1.2, 2.2, -1.2, 0.2, 0.9, 6.4};
  double output[6] = {0};
  double expected[6] = { 0.0, 1.0, 0.0, 0.0, 0.0, 1.0 };
  double x0 = 2.0;
  hard_threshold(input, output, length, x0);
  double epsilon = 0.0;
  for (size_t i = 0; i < length; i++) {
    double error = output[i] - expected[i];
    epsilon += error * error;
  }
  assert( epsilon < 0.00001);
  printf("All hard_threshold tests passed.\n");
  return true;
}

bool test_hard_threshold_mul_invth() {
  size_t length = 6;
  double input[6] = {1.2, 2.2, -1.2, 0.2, 0.9, 6.4};
  double output[6] = {0};
  double expected[6] = { 1.0, 1.0, 0.0, 0.0, 0.0, 0.0 };
  double x0 = 1.0;
  double x1 = 3.0;
  hard_threshold_mul_invth(input, output, length, x0, x1);
  double epsilon = 0.0;
  for (size_t i = 0; i < length; i++) {
    double error = output[i] - expected[i];
    epsilon += error * error;
  }
  assert( epsilon < 0.00001);
  printf("All hard_threshold_mul_invth tests passed.\n");
  return true;
}

bool test_hard_threshold_mul_invth_array() {
  size_t length = 6;
  double input[6] = {1.2, 2.2, -1.2, 0.2, 0.9, 6.4};
  double output[6] = {0};
  double expected[6] = { 1.0, 1.0, 0.0, 1.0, 1.0, 1.0 };
  double x0[6] = {1.0, 2.0, 3.0, 0.0, 0.0, 6.0};
  double x1[6] = {3.0, 3.0, 4.0, 4.0, 1.0, 7.0};
  hard_threshold_mul_invth_array(input, output, length, x0, x1);
  double epsilon = 0.0;
  for (size_t i = 0; i < length; i++) {
    double error = output[i] - expected[i];
    epsilon += error * error;
  }
  assert( epsilon < 0.00001);
  printf("All hard_threshold_mul_invth_array tests passed.\n");
  return true;
}

bool test_linearized_threshold() {
  size_t length = 6;
  double input[6] = {1.2, 2.2, -1.2, 0.2, 0.9, 6.4};
  double output[6] = {0};
  double expected[6] = {0.56666667, 0.9, 0.0, 0.23333333, 0.46666667, 1.0 };
  double x0 = 1.0;
  double alpha = 3.0;
  linearized_threshold(input, output, length, x0, alpha);
  double epsilon = 0.0;
  for (size_t i = 0; i < length; i++) {
    double error = output[i] - expected[i];
    epsilon += error * error;
  }
  assert( epsilon < 0.00001);
  printf("All linearized_threshold tests passed.\n");
  return true;
}

bool test_logistic_interval() {
  size_t length = 6;
  double input[6] = {1.2, 2.2, -1.2, 0.2, 0.9, 6.4};
  double output[6] = {0};
  double expected[6] = {0.59864711, 0.91636871, 0.01212843, 0.16798008, 0.45014927, 0.31001919};
  double a = 1.0;
  double b = 6.0;
  double alpha = 2.0;
  logistic_interval(input, output, length, a, b, alpha);
  double epsilon = 0.0;
  for (size_t i = 0; i < length; i++) {
    double error = output[i] - expected[i];
    epsilon += error * error;
  }
  assert( epsilon < 0.00001);
  printf("All logistic_interval tests passed.\n");
  return true;
}

bool test_logistic_interval_array() {
  size_t length = 6;
  double input[6] = {1.2, 2.2, -1.2, 0.2, 0.9, 6.4};
  double output[6] = {0};
  double expected[6] = {0.34974848, 0.30124244, 0.13964908, 0.42667126, 0.30040984, 0.30219773};
  double a[6] = {1.0, 2.0, 3.0, 0.0, 0.0, 6.0};
  double b[6] = {3.0, 3.0, 4.0, 4.0, 1.0, 7.0};
  double alpha = 10.0;
  logistic_interval_array(input, output, length, a, b, alpha);
  double epsilon = 0.0;
  for (size_t i = 0; i < length; i++) {
    double error = output[i] - expected[i];
    epsilon += error * error;
  }
  assert( epsilon < 0.00001);
  printf("All logistic_interval_array tests passed.\n");
  return true;
}

bool test_linearized_interval() {
  size_t length = 6;
  double input[6] = {1.2, 2.2, -1.2, 0.2, 0.9, 6.4};
  double output[6] = {0};
  double expected[6] = {0.6, 1.0, 0.0, 0.1, 0.45, 0.3};
  double a = 1.0;
  double b = 6.0;
  double alpha = 2.0;
  linearized_interval(input, output, length, a, b, alpha);
  double epsilon = 0.0;
  for (size_t i = 0; i < length; i++) {
    double error = output[i] - expected[i];
    epsilon += error * error;
  }
  assert( epsilon < 0.00001);
  printf("All linearized_interval tests passed.\n");
  return true;
}

bool test_linearized_interval_array() {
  size_t length = 6;
  double input[6] = {1.2, 2.2, -1.2, 0.2, 0.9, 6.4};
  double output[6] = {0};
  double expected[6] = {0.6, 0.54, 0.0, 0.6, 0.5225, 0.56};
  double a[6] = {1.0, 2.0, 3.0, 0.0, 0.0, 6.0};
  double b[6] = {3.0, 3.0, 4.0, 4.0, 1.0, 7.0};
  double alpha = 2.0;
  linearized_interval_array(input, output, length, a, b, alpha);
  double epsilon = 0.0;
  for (size_t i = 0; i < length; i++) {
    double error = output[i] - expected[i];
    epsilon += error * error;
  }
  assert( epsilon < 0.00001);
  printf("All linearized_interval_array tests passed.\n");
  return true;
}

bool test_lerp() {
  size_t length = 6;
  double input[6] = {1.2, 2.2, -1.2, 0.2, 0.9, 6.4};
  double output[6] = {0};
  double expected[6] = { 7.0, 12.0, -5.0, 2.0, 5.5, 33.0};
  double a = 1.0;
  double b = 6.0;
  lerp(a, b, input, output, length);
  double epsilon = 0.0;
  for (size_t i = 0; i < length; i++) {
    double error = output[i] - expected[i];
    epsilon += error * error;
  }
  assert( epsilon < 0.00001);
  printf("All lerp tests passed.\n");
  return true;
}

bool test_lerp_array() {
  size_t length = 6;
  double input[6] = {1.2, 2.2, -1.2, 0.2, 0.9, 6.4};
  double output[6] = {0};
  double expected[6] = {3.4,  4.2,  1.8,  0.8,  0.9, 12.4};
  double a[6] = {1.0, 2.0, 3.0, 0.0, 0.0, 6.0};
  double b[6] = {3.0, 3.0, 4.0, 4.0, 1.0, 7.0};
  double alpha = 2.0;
  lerp_array(a, b,input, output, length);
  double epsilon = 0.0;
  for (size_t i = 0; i < length; i++) {
    double error = output[i] - expected[i];
    epsilon += error * error;
  }
  assert( epsilon < 0.00001);
  printf("All lerp_array tests passed.\n");
  return true;
}

bool test_sigmoid_ab() {
  int8_t sigtypes[3] = {0, 1 , 4};
  size_t length = 6;
  double input[6] = {1.2, 2.9, 3.01, 3.8, 0.99, 4.001};
  double output[6] = {0};
  double expected[3][6] = {{1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
                           {1.0, 1.0, 1.0, 1.0, 0.53571429, 0.5},
                           {1.0, 1.0, 1.0, 1.0, 0.53565367, 0.5}};
  double a = 0.989;
  double b = 4.001;
  double N = 0.028;
  for (size_t j = 0; j< sizeof(sigtypes); j++) {
    sigmoid_ab(input, output, length, a, b, N, sigtypes[j]);
    double epsilon = 0.0;
    for (size_t i = 0; i < length; i++) {
      double error = output[i] - expected[j][i];
      epsilon += error * error;
    }
    assert( epsilon < 0.00001);
  }
  printf("All sigmoid_ab tests passed.\n");
  return true;
}

bool test_sigmoid_ab_array() {
  int8_t sigtypes[3] = {0, 1 , 4};
  size_t length = 6;
  double input[6] = {1.2, 2.9, 3.01, 3.8, 0.99, 4.001};
  double output[6] = {0};
  double expected[3][6] = {{1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
                           {1.0, 1.0, 0.85714286, 1.0, 0.85714286, 0.53571429},
                           {1.0, 0.99999938, 0.80667863, 1.0, 0.80667863, 0.53565367}};
  double a[6] = {1.0 , 2.0 , 3.0 , 3.5, 0.5, 4.0};
  double b[6] = {2.0 , 3.0 , 4.0 , 4.0 , 1.0 , 4.3};
  double N = 0.028;
  for (size_t j = 0; j< sizeof(sigtypes); j++) {
    sigmoid_ab_array(input, output, length, a, b, N, sigtypes[j]);
    double epsilon = 0.0;
    for (size_t i = 0; i < length; i++) {
      double error = output[i] - expected[j][i];
      epsilon += error * error;
    }
    assert( epsilon < 0.00001);
  }
  printf("All sigmoid_ab_aray tests passed.\n");
  return true;
}

int main(int argc, char **argv) {
  test_clamp2();
  test_logistic_threshold();
  test_hard_threshold();
  test_hard_threshold_mul_invth();
  test_hard_threshold_mul_invth_array();
  test_linearized_threshold();
  test_logistic_interval();
  test_logistic_interval_array();
  test_linearized_interval();
  test_linearized_interval_array();
  test_lerp();
  test_lerp_array();
  test_sigmoid_ab();
  test_sigmoid_ab_array();
  printf("All tests passed\n");
}

