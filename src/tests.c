#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
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

bool test_logistic_threshold2() {
  size_t length = 6;
  double input[6] = {1.2 , 0.42, 3.01, 3.8 , 0.47, 1.0};
  double output[6] = {0};
  double expected[6] = { 1.0, 0.10184681, 1.0, 1.0, 0.306544, 0.99999875 };
  double x0 = 0.5;
  double alpha = 0.147;
  logistic_threshold(input, output, length, x0, alpha);
  double epsilon = 0.0;
  for (size_t i = 0; i < length; i++) {
    double error = output[i] - expected[i];
    epsilon += error * error;
  }
  assert( epsilon < 0.00001);
  printf("All logistic_threshold2 tests passed.\n");
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

bool test_logistic_interval_array2() {
  size_t length = 6;
  double input[6] = {0.5 , 0.4 , 0.43, 0.45, 0.38, 0.41};
  double output[6] = {0};
  double expected[6] = {0.0003868, 0.0211232, 0.8949993, 0.3286523, 0.7958416, 0.9933070};
  double a[6] = {0.267, 0.27687967, 0.267, 0.267, 0.274628, 0.26700002};
  double b[6] = {0.445, 0.37314776, 0.445, 0.445, 0.38952354, 0.44499987};
  double alpha = 0.028;
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

bool test_lerp1() {
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
  printf("All lerp1 tests passed.\n");
  return true;
}

bool test_lerp2() {
  size_t length = 6;
  double input[6] = {1.0, 0.10184681, 1.0, 1.0, 0.306544, 0.99999875};
  double output[6] = {0};
  double expected[6] = { 0.267, 0.27687967, 0.267, 0.267, 0.274628, 0.26700002};
  double a = 0.278;
  double b = 0.267;
  lerp(a, b, input, output, length);
  double epsilon = 0.0;
  for (size_t i = 0; i < length; i++) {
    double error = output[i] - expected[i];
    epsilon += error * error;
  }
  assert( epsilon < 0.00001);
  printf("All lerp2 tests passed.\n");
  return true;
}

bool test_lerp3() {
  size_t length = 6;
  double input[6] = {1.0, 0.10184681, 1.0, 1.0, 0.306544, 0.99999875};
  double output[6] = {0};
  double expected[6] = { 0.445, 0.37314776, 0.445, 0.445, 0.38952354, 0.44499987};
  double a = 0.365;
  double b = 0.445;
  lerp(a, b, input, output, length);
  double epsilon = 0.0;
  for (size_t i = 0; i < length; i++) {
    double error = output[i] - expected[i];
    epsilon += error * error;
  }
  assert( epsilon < 0.00001);
  printf("All lerp3 tests passed.\n");
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

bool test_sigmoid_mix() {
  int8_t mixtypes[3] = {0, 1 , 4};
  size_t length = 6;
  double input_x[6] = {1.0, 2.0, 3.0, 3.0, 0.0, 4.0};
  double input_y[6] = {2.0, 3.0, 4.0, 4.0, 2.0, 5.0};
  double input_m[6] = {1.2, 2.9, 3.01, 3.8, 0.99, 4.001};
  double output[6] = {0};
  double expected[3][6] = {{2.0, 3.0, 4.0, 4.0, 2.0, 5.0},
                           {2.0, 3.0, 4.0, 4.0, 2.0, 5.0},
                           {1.99999999, 3.0, 4.0, 4.0, 1.99999676, 5.0 }};
  double M = 0.147;
  for (size_t j = 0; j< sizeof(mixtypes); j++) {
    sigmoid_mix(input_x, input_y, input_m, output, length, mixtypes[j], M);
    double epsilon = 0.0;
    for (size_t i = 0; i < length; i++) {
      double error = output[i] - expected[j][i];
      epsilon += error * error;
    }
    assert( epsilon < 0.00001);
  }
  printf("All sigmoid_mix tests passed.\n");
  return true;
}

bool test_sigmoid_mix_point_xy() {
  int8_t mixtypes[3] = {0, 1 , 4};
  size_t length = 6;
  double input_x = 0.0;
  double input_y = 5.0;
  double input_m[6] = {1.2, 2.9, 3.01, 3.8, 0.99, 4.001};
  double output[6] = {0};
  double expected[3][6] = {{5.0, 5.0, 5.0, 5.0, 5.0, 5.0},
                           {5.0, 5.0, 5.0, 5.0, 5.0, 5.0},
                           {4.99999997, 5.0, 5.0, 5.0, 4.9999919, 5.0 }};
  double M = 0.147;
  for (size_t j = 0; j< sizeof(mixtypes); j++) {
    sigmoid_mix_point_xy(input_x, input_y, input_m, output, length, mixtypes[j], M);
    double epsilon = 0.0;
    for (size_t i = 0; i < length; i++) {
      double error = output[i] - expected[j][i];
      epsilon += error * error;
    }
    assert( epsilon < 0.00001);
  }
  printf("All sigmoid_mix_point_xy tests passed.\n");
  return true;
}

bool test_matrix_roll() {
  double mat1[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  double mat1_expected[] = {7, 8, 9, 1, 2, 3, 4, 5, 6};
  int m = 3;
  int n = 3;
  int axis = 0;
  int k = 1;
  matrix_roll(mat1, m, n, k, axis);

  double epsilon = 0.0;
  for (size_t i = 0; i < m*n; i++) {
    double error = mat1[i] - mat1_expected[i];
    epsilon += error * error;
  }
  assert( epsilon < 0.00001);

  double mat2[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  double mat2_expected[] = {3, 1, 2, 6, 4, 5, 9, 7, 8};
  axis = 1;
  k = 1;
  matrix_roll(mat2, m, n, k, axis);
  epsilon = 0.0;
  for (size_t i = 0; i < m*n; i++) {
    double error = mat2[i] - mat2_expected[i];
    epsilon += error * error;
  }
  assert( epsilon < 0.00001);

  // 6*6 matrix
  double mat3[] = {
      1,  2,  3,  4,  5,  6,
      7,  8,  9, 10, 11, 12,
     13, 14, 15, 16, 17, 18,
     19, 20, 21, 22, 23, 24,
     25, 26, 27, 28, 29, 30,
     31, 32, 33, 34, 35, 36};

  double mat3_expected[] = {
     13, 14, 15, 16, 17, 18,
     19, 20, 21, 22, 23, 24,
     25, 26, 27, 28, 29, 30,
     31, 32, 33, 34, 35, 36,
      1,  2,  3,  4,  5,  6,
      7,  8,  9, 10, 11, 12};

  m = 6;
  n = 6;
  axis = 0;
  k = -2;

  matrix_roll(mat3, m, n, k, axis);
  epsilon = 0.0;
  for (size_t i = 0; i < m*n; i++) {
    double error = mat3[i] - mat3_expected[i];
    epsilon += error * error;
  }
  assert( epsilon < 0.00001);

  // 6*6 matrix
  double mat4[] = {
     1.1, 2.2, 3.3, 4.4, 5.5, 6.6,
     7.7, 8.8, 9.9,  10,  11,  12,
     13,   14,  15,  16,  17,  18,
     19,   20,  21,  22,  23,  24,
     25,   26,  27,  28,  29,  30,
     31,   32,  33,  34,  35,  36};

  double mat4_expected[] = {
     6.6, 1.1, 2.2, 3.3, 4.4, 5.5,
      12, 7.7, 8.8, 9.9,  10,  11,
      18,  13,  14,  15,  16,  17,
      24,  19,  20,  21,  22,  23,
      30,  25,  26,  27,  28,  29,
      36,  31,  32,  33,  34,  35,};

  m = 6;
  n = 6;
  axis = 1;
  k = 1;

  matrix_roll(mat4, m, n, k, axis);
  epsilon = 0.0;
  for (size_t i = 0; i < m*n; i++) {
    double error = mat3[i] - mat3_expected[i];
    epsilon += error * error;
  }
  assert( epsilon < 0.00001);

  printf("All matrix roll tests passed.\n");
  return true;
}

bool test_basic_rules() {
  size_t length = 6;
  double aliveness[6] = {0};
  double threshold1[6] = {0};
  double threshold2[6] = {0};
  double new_aliveness[6] = {0};
  double b_thresh[6] = {0};
  double d_thresh[6] = {0};
  double transistion[6] = {0};
  double nextfield[6] = {0};
  double delta[6] = {0};

  AlivenessTemp aliveness_temp = {
    .aliveness = aliveness,
    .threshold1 = threshold1,
    .threshold2 = threshold2,
    .new_aliveness = new_aliveness,
    .b_thresh = b_thresh,
    .d_thresh = d_thresh,
    .transistion = transistion,
    .nextfield = nextfield,
    .delta = delta
  };
  double input_m[6] = {1.2, 0.42, 3.01, 3.8, 0.47, 1.0};
  double input_n[6] = {0.5 , 0.4 , 0.43, 0.45, 0.38, 0.41};
  double output[6] = {0};
  double field[6] = {0};
  double expected[6] = {0.0003868, 0.0211232, 0.8949993, 0.3286523, 0.7958416, 0.9933070};

  BasicRules basic_rules_test;
  basic_rules_new(&basic_rules_test, NULL);
  basic_rules_s(&basic_rules_test, input_n, 6, input_m, 6,
                field, 6 , output, &aliveness_temp);

  double epsilon = 0.0;
  for (size_t i = 0; i < length; i++) {
    double error = output[i] - expected[i];
    epsilon += error * error;
  }
  assert( epsilon < 0.00001);
  printf("All sigmoid_mix_point_xy tests passed.\n");
  return true;
}

void test_antialiased_circle(){
  double output[36] = {0};
  double expected[36] = {
      0.99434716, 0.9298875,  0.5,        0.0701125,  0.5,        0.9298875,
      0.9298875,  0.81968814, 0.35200745, 0.04722551, 0.35200745, 0.81968814,
      0.5,        0.35200745, 0.10513261, 0.01551533, 0.10513261, 0.35200745,
      0.0701125,  0.04722551, 0.01551533, 0.00302703, 0.01551533, 0.04722551,
      0.5,        0.35200745, 0.10513261, 0.01551533, 0.10513261, 0.35200745,
      0.9298875,  0.81968814, 0.35200745, 0.04722551, 0.35200745, 0.81968814};
  unsigned int width  = 6;
  unsigned int height = 6;
  double radius = 2.0;
  antialiased_circle(height, width, radius, output);
  double epsilon = 0.0;
  for (size_t i = 0; i < width*height; i++) {
    double error = output[i] - expected[i];
    epsilon += error * error;
  }
  assert( epsilon < 0.00001);
  printf("All antialiased cicrcle tests passed.\n");
};

int main(int argc, char **argv) {
  test_clamp2();
  test_logistic_threshold();
  test_logistic_threshold2();
  test_hard_threshold();
  test_hard_threshold_mul_invth();
  test_hard_threshold_mul_invth_array();
  test_linearized_threshold();
  test_logistic_interval();
  test_logistic_interval_array();
  test_logistic_interval_array2();
  test_linearized_interval();
  test_linearized_interval_array();
  test_lerp1();
  test_lerp2();
  test_lerp3();
  test_lerp_array();
  test_sigmoid_ab();
  test_sigmoid_ab_array();
  test_sigmoid_mix();
  test_sigmoid_mix_point_xy();
  test_matrix_roll();
  printf("All math function tests passed\n");
  printf("== test Basic Rules Class ==\n");
  test_basic_rules();
  printf("== test antialiased circle ==\n");
  test_antialiased_circle();
}

