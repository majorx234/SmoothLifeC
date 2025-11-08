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

typedef struct AlivenessTemp {
  double* aliveness;
  double* threshold1;
  double* threshold2;
  double* new_aliveness;
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
  const BasicRules _;
  uint8_t sigmode;
  uint8_t sigtype;
  uint8_t mixtype;
  uint8_t timestep_mode;
  double dt;
  double *esses[3];
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
  self->esses_count = 0;
  return self;
};

void extensive_rules_clear(const void* _self) {
  ExtensiveRules* self = (ExtensiveRules*)_self;
  for(size_t i = 0;i<3;i++){
    self->esses[i] = NULL;
    self->esses_count = 0;
  }
}

typedef struct SmootheLife {
  size_t with;
  size_t height;
  double shape_h;
  double shape_w;
  BasicRules* basic_rules;
} SmootheLife;
