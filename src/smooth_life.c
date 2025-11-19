#include <stdint.h>
#include <stdio.h>
#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <assert.h>

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

void linearized_threshold_point_x(double x, double* x_out, size_t length, double x0, double alpha) {
  for (size_t i = 0; i < length; i++) {
    x_out[i] = clamp2((x - x0) / alpha + 0.5, 0.0, 1.0);
  }
}

void logistic_interval(double *x, double *x_out, size_t length, double a,
                       double b, double alpha) {
  for (size_t i = 0; i< length; i++) {
    x_out[i] = 1.0 / (1.0 + exp(-4.0 / alpha * (x[i] - a))) * (1.0 - (1.0 / (1.0 + exp(-4.0 / alpha * (x[i] - b)))));
  }
}

void logistic_interval_array(double *x, double *x_out, size_t length, double* a,
                       double* b, double alpha) {
  for (size_t i = 0; i< length; i++) {
    x_out[i] = 1.0 / (1.0 + exp(-4.0 / alpha * (x[i] - a[i]))) * (1.0 - (1.0 / (1.0 + exp(-4.0 / alpha * (x[i] - b[i])))));
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
    x_out[i] = clamp2((x[i] - a[i]) / alpha + 0.5, 0.0, 1.0) * (1.0 - (clamp2((x[i] - b[i]) / alpha + 0.5, 0.0, 1.0)));
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
    linearized_threshold(x, x_out, length, 0.5, M);
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
    linearized_threshold_point_x(x, x_out, length, 0.5, M);
  } else if (mixtype == 4) {
    logistic_threshold(m, x_out, length, 0.5, M);
  } else {
    printf("mixtype not implemented");
    exit(-3);
  }
  lerp(x, y, x_out, x_out, length);
}

typedef struct AlivenessTemp {
  double* aliveness;
  double* threshold1;
  double* threshold2;
  double* new_aliveness;
  double* b_thresh;
  double* d_thresh;
  double* transistion;
  double* nextfield;
  double* delta;
} AlivenessTemp;

typedef struct Class {
  size_t size;
  void * (* ctor) (void * self, va_list * app);
  void * (* dtor) (void * self);
  void (* s) (const void * self,
              double* n,
              size_t length_n,
              double* m,
              size_t length_m,
              double* field,
              size_t length_field,
              double* x_out,
              AlivenessTemp* aliveness_temp);
  void (* clear)(const void * self);
} Class;

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

typedef struct BasicRules {
  const Class* class;
  double b1;
  double b2;
  double d1;
  double d2;
  double N;
  double M;
} BasicRules;

void* basic_rules_new(void* _self, va_list * app){
  // BasicRules* basic_rules = (BasicRules*)malloc(sizeof(BasicRules));
  BasicRules* self = (BasicRules*)_self;
  self->b1 = 0.278;
  self->b2 = 0.365;
  self->d1 = 0.267;
  self->d2 = 0.445;
  self->M = 0.028;
  self->N = 0.147;
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
                          length_m, aliveness_temp->threshold1,
                          aliveness_temp->threshold2,
                          basic_rules->N);
  for (int i = 0; i < length_m; i++) {
    x_out[i] = clamp2(aliveness_temp->new_aliveness[i], 0.0, 1.0);
  }
}

static const Class _point = {
  sizeof(BasicRules), basic_rules_new, 0 , basic_rules_s, basic_rules_clear
};

typedef struct ExtensiveRules {
  BasicRules _;
  uint8_t sigmode;
  uint8_t sigtype;
  uint8_t mixtype;
  uint8_t timestep_mode;
  double dt;
  double *esses[3];
  double *esses_free;
  size_t esses_count;
} ExtensiveRules;

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

void matrix_roll(double* matrix, size_t w, size_t h, size_t roll_offset, bool axis) {
  if(bool) { // y-axis
    
  } else { // x-axis
    
  }
}

void antialiased_circle(unsigned int h,
                        unsigned int w,
                        double radius,
                        double* x_out,
                        size_t length)
{
  double logres = log2(h < w ? h : w);
  for(size_t i = 0; i<h; i++) {
    for(size_t j = 0; w<h; i++) {
      double x = (j - (w/2.0)) * (j - (w/2.0));
      double y = (i - (h/2.0)) * (i - (h/2.0));
      double sqrt_r = sqrt(x+y);
      x_out[i + w*j] = 1.0 / (1.0 + exp(logres * (sqrt_r - radius)));
    }
  }
}

typedef struct SmootheLife {
  size_t with;
  size_t height;
  double shape_h;
  double shape_w;
  BasicRules* basic_rules;
} SmootheLife;
