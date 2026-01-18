#include <fftw3.h>
#include <stdint.h>
#include <stdio.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "smooth_life.h"

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

void hard_threshold_mul_invth(double* x, double* x_out, size_t length, double x0, double x0_inv) {
  for (size_t i = 0; i< length; i++) {
    x_out[i] = (x[i] > x0 ? 1.0 : 0.0) * 1 -(x[i] > x0_inv ? 1.0 : 0.0);
  }
}

void hard_threshold_mul_invth_array(double* x, double* x_out, size_t length, double* x0, double* x0_inv) {
  for (size_t i = 0; i< length; i++) {
    x_out[i] = (x[i] > x0[i] ? 1.0 : 0.0) * 1 -(x[i] > x0_inv[i] ? 1.0 : 0.0);
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
    x_out[i] = (1.0 / (1.0 + exp(-4.0 / alpha * (x[i] - a)))) * (1.0 - (1.0 / (1.0 + exp(-4.0 / alpha * (x[i] - b)))));
  }
}

void logistic_interval_array(double *x, double *x_out, size_t length, double* a,
                             double* b, double alpha) {
  for (size_t i = 0; i< length; i++) {
    x_out[i] = (1.0 / (1.0 + exp(-4.0 / alpha * (x[i] - a[i])))) * (1.0 - (1.0 / (1.0 + exp(-4.0 / alpha * (x[i] - b[i])))));
  }
}

void linearized_interval(double *x, double *x_out, size_t length, double a,
                         double b, double alpha) {
  for (size_t i = 0; i< length; i++) {
    x_out[i] = clamp2((x[i] - a) / alpha + 0.5, 0.0, 1.0) * (1.0 - (clamp2((x[i] - b) / alpha + 0.5, 0.0, 1.0)));
  }
}

void linearized_interval_array(double *x, double *x_out, size_t length, double* a,
                       double* b, double alpha) {
  for (size_t i = 0; i< length; i++) {
    x_out[i] = clamp2(((x[i] - a[i]) / alpha) + 0.5, 0.0, 1.0) * (1.0 - (clamp2(((x[i] - b[i]) / alpha) + 0.5, 0.0, 1.0)));
  }
}

void lerp(double a, double b, double *t, double *x_out, size_t length) {
  for (size_t i = 0; i < length; i++) {
    x_out[i] = (1.0 - t[i]) * a + t[i] * b;
  }
}

void lerp_array(double *a, double *b, double *t, double *x_out, size_t length) {
  for (size_t i = 0; i < length; i++) {
    x_out[i] = (1.0 - t[i]) * a[i] + t[i] * b[i];
  }
}

void sigmoid_ab(double* x, double* x_out, size_t length, double a, double b, double N, int8_t sigtype) {
  if (sigtype == 0) {
    hard_threshold_mul_invth(x, x_out, length, a, b);
  } else if (sigtype == 1){
    linearized_interval(x, x_out, length, a, b, N);
  } else if (sigtype == 4) {
    logistic_interval(x, x_out, length, a, b, N);
  } else {
    printf("sigtype not implemented");
    exit(-2);
  }
}

void sigmoid_ab_array(double* x, double* x_out, size_t length, double* a, double* b, double N, int8_t sigtype) {
  if (sigtype == 0) {
    hard_threshold_mul_invth_array(x, x_out, length, a, b);
  } else if (sigtype == 1){
    linearized_interval_array(x, x_out, length, a, b, N);
  } else if (sigtype == 4) {
    logistic_interval_array(x, x_out, length, a, b, N);
  } else {
    printf("sigtype not implemented");
    exit(-3);
  }
}

void sigmoid_mix(double* x, double* y, double* m, double* x_out, size_t length, int8_t mixtype, double M) {
  // used x_out as temp array to hold intermediate values
  if (mixtype == 0) {
    hard_threshold(m, x_out, length, 0.5);
  } else if (mixtype == 1){
    linearized_threshold(m, x_out, length, 0.5, M);
  } else if (mixtype == 4) {
    logistic_threshold(m, x_out, length, 0.5, M);
  } else {
    printf("mixtype not implemented");
    exit(-3);
  }
  lerp_array(x, y, x_out, x_out, length);
}

void sigmoid_mix_point_xy(double x, double y, double* m, double* x_out, size_t length, int8_t mixtype, double M) {
  // used x_out as temp array to hold intermediate values
  if (mixtype == 0) {
    hard_threshold(m, x_out, length, 0.5);
  } else if (mixtype == 1){
    linearized_threshold(m, x_out, length, 0.5, M);
  } else if (mixtype == 4) {
    logistic_threshold(m, x_out, length, 0.5, M);
  } else {
    printf("mixtype not implemented");
    exit(-3);
  }
  lerp(x, y, x_out, x_out, length);
}
/* Class helper */
void s (const void * self,
        double* n,
        size_t length_n,
        double* m,
        size_t length_m,
        double* field,
        size_t length_field,
        double* x_out,
        AlivenessTemp* aliveness_temp) {
  const Class * const * cp = self;
  assert(self && * cp && (* cp) -> s);
  (* cp) -> s(self,
              n, length_n,
              m, length_m,
              field, length_field,
              x_out,
              aliveness_temp);
  // TODO: basic implementaion
}

/* BasicRules Class */
void* basic_rules_new(void* _self, va_list * app){
  // BasicRules* basic_rules = (BasicRules*)malloc(sizeof(BasicRules));
  BasicRules* self = (BasicRules*)_self;
  self->b1 = 0.278;
  self->b2 = 0.365;
  self->d1 = 0.267;
  self->d2 = 0.445;
  self->N = 0.028;
  self->M = 0.147;
  // TODO ? class
  // self->class->size = ;

  // TODO: read in cli arguments, change params
  return self;
}

void basic_rules_clear(const void* _self) {
  // TODO reset internal state (no stat in basic rules)
}

void basic_rules_s(const void* _self,
                   double* n,
                   size_t length_n,
                   double* m,
                   size_t length_m,
                   double* field,
                   size_t length_field,
                   double* x_out,
                   AlivenessTemp* aliveness_temp) {
  BasicRules* basic_rules = (BasicRules*)_self;
  logistic_threshold(m, aliveness_temp->aliveness, length_m, 0.5, basic_rules->M);
  lerp(basic_rules->b1,
       basic_rules->d1,
       aliveness_temp->aliveness,
       aliveness_temp->threshold1,
       length_m);
  lerp(basic_rules->b2,
       basic_rules->d2,
       aliveness_temp->aliveness,
       aliveness_temp->threshold2,
       length_m);
  logistic_interval_array(n,
                          aliveness_temp->new_aliveness,
                          length_m,
                          aliveness_temp->threshold1,
                          aliveness_temp->threshold2,
                          basic_rules->N);
  for (int i = 0; i < length_m; i++) {
    x_out[i] = clamp2(aliveness_temp->new_aliveness[i], 0.0, 1.0);
  }
}

static const Class _point = {
  sizeof(BasicRules), basic_rules_new, 0 , basic_rules_s, basic_rules_clear
};
/* ExtensiveRules */

ExtensiveRules* extensive_rules_new(void* _self, va_list * app) {
  basic_rules_new(_self, app);
  ExtensiveRules* self = (ExtensiveRules*)_self;
  self->sigmode = 0;
  self->sigtype = 0;
  self->mixtype = 0;
  self->timestep_mode = 0;
  self->dt = 0.1;
  self->esses[0] = NULL;
  self->esses[1] = NULL;
  self->esses[2] = NULL;
  self->esses_free = NULL;
  self->esses_count = 0;
  return self;
};

void extensive_rules_clear(const void* _self, double** esses) {
  ExtensiveRules* self = (ExtensiveRules*)_self;
  for(size_t i = 0;i<3;i++){
    self->esses[i] = esses[i]; // NULL;
  }
  self->esses_free = esses[3]; //NULL;
  self->esses_count = 0;
}

void extensive_rules_s(const void* _self,
                       double* n,
                       size_t length_n,
                       double* m,
                       size_t length_m,
                       double* field,
                       size_t length_field,
                       double* x_out,
                       AlivenessTemp* aliveness_temp) {
  ExtensiveRules* self = (ExtensiveRules*)_self;
  if (self->sigmode == 1) {
    sigmoid_ab(n,
               aliveness_temp->b_thresh,
               length_n,
               self->_.b1,
               self->_.b2,
               self->_.N,
               self->sigtype);
    sigmoid_ab(n,
               aliveness_temp->d_thresh,
               length_n,
               self->_.d1,
               self->_.d2,
               self->_.N,
               self->sigtype);
    lerp_array(aliveness_temp->b_thresh,
               aliveness_temp->d_thresh,
               m,
               aliveness_temp->transistion,
               length_m);
  } else if (self->sigmode == 2) {
    sigmoid_ab(n,
               aliveness_temp->b_thresh,
               length_n,
               self->_.b1,
               self->_.b2,
               self->_.N,
               self->sigtype);
    sigmoid_ab(n,
               aliveness_temp->d_thresh,
               length_n,
               self->_.d1,
               self->_.d2,
               self->_.N,
               self->sigtype);
    sigmoid_mix(aliveness_temp->b_thresh,
                aliveness_temp->d_thresh,
                m,
                aliveness_temp->transistion,
                length_m,
                self->mixtype,
                self->_.M);
  } else if (self->sigmode == 3){
    lerp(self->_.b1, self->_.d1, m, aliveness_temp->threshold1, length_m);
    lerp(self->_.b2, self->_.d2, m, aliveness_temp->threshold2, length_m);
    sigmoid_ab_array(n, aliveness_temp->transistion, length_n, aliveness_temp->threshold1, aliveness_temp->threshold2, self->_.N, self->sigtype );
  } else if (self->sigmode == 4){
    sigmoid_mix_point_xy(self->_.b1, self->_.d1, m, aliveness_temp->threshold1, length_m, self->mixtype, self->_.M);
    sigmoid_mix_point_xy(self->_.b2, self->_.d2, m, aliveness_temp->threshold1, length_m, self->mixtype, self->_.M);
    sigmoid_ab_array(n, aliveness_temp->transistion, length_n, aliveness_temp->threshold1, aliveness_temp->threshold2, self->_.N, self->sigtype );
  } else {
    printf("sigmod not implemented");
    exit(-4);
  }
  // STEP 2: Integrate based on timestep_mode
  if (self->timestep_mode == 0) {
    for (size_t i = 0; i < length_field; i++) {
      aliveness_temp->nextfield[i] = aliveness_temp->transistion[i];
    }
  } else if(self->timestep_mode == 1) {
    for (size_t i = 0; i < length_field; i++) {
      aliveness_temp->nextfield[i] = field[i] + self->dt * (2.0 * aliveness_temp->transistion[i] - 1);
    }
  } else if(self->timestep_mode == 2) {
    for (size_t i = 0; i < length_field; i++) {
      aliveness_temp->nextfield[i] = field[i] + self->dt * (aliveness_temp->transistion[i] - field[i]);
    }
  } else if(self->timestep_mode == 3) {
    for (size_t i = 0; i < length_field; i++) {
      aliveness_temp->nextfield[i] = m[i] + self->dt * (2.0 * aliveness_temp->transistion[i] - 1);
    }
  } else if(self->timestep_mode == 4) {
    for (size_t i = 0; i < length_field; i++) {
      aliveness_temp->nextfield[i] = m[i] + self->dt * (aliveness_temp->transistion[i] - m[i]);
    }
  } else if(self->timestep_mode == 5) {
    for (size_t i = 0; i < length_field; i++) {
      self->esses_free[i] = aliveness_temp->transistion[i] - m[i];
    }
    double* delta_tmp = aliveness_temp->delta;
    if (self->esses_count == 0) {
      double* tmp = self->esses[0];
      delta_tmp = self->esses_free;
      self->esses[0] = self->esses_free;
      self->esses_free = tmp;
      self->esses_count++;
    } else if (self->esses_count == 1) {
      for (size_t i = 0; i < length_field; i++) {
        delta_tmp[i] = (3.0 * self->esses_free[i] - self->esses[0][i]) / 2.0;
      }
      double* tmp = self->esses[0];
      self->esses[0] = self->esses_free;
      self->esses_free = self->esses[1];
      self->esses[1] = tmp;
      self->esses_count++;
    } else if (self->esses_count == 2) {
      for (size_t i = 0; i < length_field; i++) {
        delta_tmp[i] = (23.0 * self->esses_free[i]
                        - 16.0 * self->esses[0][i]
                        + 5.0 * self->esses[1][i]) / 12.0;
      }
      double* tmp = self->esses[0];
      self->esses[0] = self->esses_free;
      self->esses_free = self->esses[2];
      self->esses[2] = self->esses[1];
      self->esses[1] = tmp;
      self->esses_count++;
    } else { // esses_count ==3
      for (size_t i = 0; i < length_field; i++) {
        delta_tmp[i] = (55.0 * self->esses_free[i]
                        - 59.0 * self->esses[0][i]
                        + 37.0 * self->esses[1][i]
                        - 9.0 * self->esses[2][i]) / 24.0;
      }
      double* tmp = self->esses[0];
      self->esses[0] = self->esses_free;
      self->esses_free = self->esses[2];
      self->esses[2] = self->esses[1];
      self->esses[1] = tmp;
    }
    for (size_t i = 0; i < length_field; i++) {
      aliveness_temp->nextfield[i] = field[i] + self->dt *delta_tmp[i];
    }
  }
  else {
    printf("timestep_mode %d not implemented", s);
    exit(-5);
  }
  for (int i = 0; i < length_m; i++) {
    x_out[i] = clamp2(aliveness_temp->nextfield[i], 0.0, 1.0);
  }
}

ExtensiveRules* smooth_timestep_rules_new(void* _self, va_list * app) {
  extensive_rules_new(_self, app);
  ExtensiveRules* self = (ExtensiveRules*)_self;
  self->sigmode = 2;
  self->sigtype = 1;
  self->mixtype = 0;
  self->timestep_mode = 2;
  self->dt = 0.2;
  self->esses[0] = NULL;
  self->esses[1] = NULL;
  self->esses[2] = NULL;
  self->esses_free = NULL;
  self->esses_count = 0;
  self->_.b1 = 0.254;
  self->_.b2 = 0.312;
  self->_.d1 = 0.340;
  self->_.d2 = 0.518;
  return self;
}


void reverse_array(double* arr,size_t start, size_t end, size_t stride) {
  int left = start, right = end;
  // reverse first n -k elements
  while (left < right) {
    double temp = arr[left*stride];
    arr[left*stride] = arr[right*stride];
    arr[right*stride] = temp;
    left++;
    right--;
  }

}

void array_roll(double *arr, int length, int roll_offset, size_t stride) {
  // numpy convention
  roll_offset = -roll_offset;
  if (length <= 1) return; // No rotation needed for empty or single-element arrays
  if (roll_offset < 0) roll_offset = length + roll_offset;

  roll_offset = roll_offset % length; // Normalize roll_offset to handle cases where roll_offset >= n
  if (roll_offset == 0) return; // No rotation needed

  reverse_array(arr, 0,           roll_offset-1,  stride);
  reverse_array(arr, roll_offset, length-1,       stride);
  reverse_array(arr, 0,           length-1,       stride);
}

void matrix_print(double* mat, size_t m, size_t n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      printf("%f ", mat[i*m+j]);
    }
    printf("\n");
  }
}

void matrix_roll(double* matrix, size_t w, size_t h, size_t roll_offset, bool axis) {
  if (axis == 0) {
    for (size_t i = 0; i < w; i++) {
      array_roll(&matrix[i], w, roll_offset, w);
    }
  } else if (axis == 1) {
    for (size_t i = 0; i < h; i++) {
      array_roll(&matrix[i*w], w, roll_offset, 1);
    }
  } else {
    // TODO
  }
}

void antialiased_circle(unsigned int h,
                        unsigned int w,
                        double radius,
                        double* x_out)
{
  double logres = log2f(h < w ? h : w);
  for(size_t i = 0; i<h; i++) {
    for(size_t j = 0; j<w; j++) {
      double x = j - (w/2.0);
      double y = i - (h/2.0);
      double sqrt_r = sqrt(x*x+y*y);
      double value = 1.0 / (1.0 +  expf(logres * (sqrt_r - radius)));
      x_out[w*i + j] = value;
    }
  }
  matrix_roll(x_out, w, h, (h>>1), 0);
  matrix_roll(x_out, w, h, (w>>1), 1);
  // logistic roll
}

void init_multipliers(Multipliers *self, MultipliersTemp *tmp, int width,
                      int height, double inner_radius, double outer_radius) {
  self->inner_radius = inner_radius;
  self->inner_radius = outer_radius;
  antialiased_circle(height, width, inner_radius, tmp->inner);
  antialiased_circle(height, width, outer_radius, tmp->outer);

  // Build double spatial kernels
  double sum_inner = 0.0;
  double sum_annulus = 0.0;
  for(size_t i = 0; i < height; i++) {
    for (size_t j; j < width; j++) {
      tmp->annulus[i * width + j] =
          tmp->outer[i * width + j] - tmp->inner[i * width + j];
      sum_annulus += tmp->annulus[i * width + j];
      sum_inner += tmp->inner[i * width + j];
    }
  }

  // Normalize each kernel so their sum is 1. This makes them proper averaging filters.
  for(size_t i = 0; i < height; i++) {
    for (size_t j; j < width; j++) {
      tmp->annulus[i * width + j] /= sum_annulus;
      tmp->inner[i * width + j] /= sum_inner;
    }
  }
  fftw_plan plan_inner_M_freq = fftw_plan_dft_r2c_2d(width, height, tmp->inner, self->_M_freq, FFTW_ESTIMATE);
  fftw_plan plan_annulus_N_freq = fftw_plan_dft_r2c_2d(width, height, tmp->annulus, self->_N_freq, FFTW_ESTIMATE);
  // Compute real-to-complex FFTs. Inputs are float32; outputs are complex64.
  fftw_execute(plan_inner_M_freq);
  fftw_execute(plan_annulus_N_freq);

  // set shapes
  self->_M_freq_width = width / 2 + 1;
  self->_M_freq_height = height;
  self->_N_freq_width = width / 2 + 1;
  self->_N_freq_height = height;

  fftw_destroy_plan(plan_inner_M_freq);
  fftw_destroy_plan(plan_annulus_N_freq);
  fftw_free(tmp->inner);
  fftw_free(tmp->annulus);
}

void init_smooth_life(SmootheLife* self, int width, int height) {
  self->width = width;
  self->height = height;
  self->basic_rules = malloc(sizeof(BasicRules));
  basic_rules_new(self->basic_rules,NULL);
  self->multipliers = malloc(sizeof(Multipliers));
  MultipliersTemp multipliers_temp = {0};

  // just tempory, need to be deleted at the end
  multipliers_temp.annulus = malloc(width*height*sizeof(double));
  multipliers_temp.inner = malloc(width*height*sizeof(double));
  multipliers_temp.outer = malloc(width*height*sizeof(double));
  double INNER_RADIUS = 7.0;
  double OUTER_RADIUS = INNER_RADIUS * 3.0;

  init_multipliers(self->multipliers, &multipliers_temp, width, height, INNER_RADIUS, OUTER_RADIUS);
  self->field = malloc(width*height*sizeof(double));
  smoother_life_clear(self);
}

void smoother_life_clear(SmootheLife* self) {
  memset(self->field, 0, self->width*self->height*sizeof(double));
  basic_rules_clear(self->basic_rules);
}
