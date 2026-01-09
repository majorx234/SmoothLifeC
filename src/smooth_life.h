#include <stdint.h>
#include <stddef.h>
#include <fftw3.h>

double clamp2(double x, double min, double max);
void logistic_threshold(double* x, double* x_out, size_t length, double x0, double alpha);
void hard_threshold(double* x, double* x_out, size_t length, double x0);
void hard_threshold(double* x, double* x_out, size_t length, double x0);
void hard_threshold_mul_invth(double* x, double* x_out, size_t length, double x0, double x0_inv);
void hard_threshold_mul_invth_array(double *x, double *x_out, size_t length, double *x0, double *x0_inv);
void linearized_threshold(double* x, double* x_out, size_t length, double x0, double alpha);
void linearized_threshold_point_x(double x, double* x_out, size_t length, double x0, double alpha);
void logistic_interval(double *x, double *x_out, size_t length, double a, double b, double alpha);
void logistic_interval_array(double *x, double *x_out, size_t length, double* a, double* b, double alpha);
void linearized_interval(double *x, double *x_out, size_t length, double a, double b, double alpha);
void linearized_interval_array(double *x, double *x_out, size_t length, double* a, double* b, double alpha);
void lerp(double a, double b, double *t, double *x_out, size_t length);
void lerp_array(double *a, double *b, double *t, double *x_out, size_t length);
void sigmoid_ab(double *x, double *x_out, size_t length, double a, double b, double N, int8_t sigtype);
void sigmoid_ab_array(double* x, double* x_out, size_t length, double* a, double* b, double N, int8_t sigtype);
void sigmoid_mix(double* x, double* y, double* m, double* x_out, size_t length, int8_t mixtype, double M);
void sigmoid_mix_point_xy(double x, double y, double* m, double* x_out, size_t length, int8_t mixtype, double M);

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
        AlivenessTemp* aliveness_temp);

typedef struct BasicRules {
  const Class* class;
  double b1;
  double b2;
  double d1;
  double d2;
  double N;
  double M;
} BasicRules;

void* basic_rules_new(void* _self, va_list * app);
void basic_rules_clear(const void* _self);
void basic_rules_s(const void* _self,
                   double* n,
                   size_t length_n,
                   double* m,
                   size_t length_m,
                   double* field,
                   size_t length_field,
                   double* x_out,
                   AlivenessTemp* aliveness_temp);

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
ExtensiveRules* extensive_rules_new(void* _self, va_list * app);
void extensive_rules_clear(const void* _self, double** esses);
void extensive_rules_s(const void* _self,
                       double* n,
                       size_t length_n,
                       double* m,
                       size_t length_m,
                       double* field,
                       size_t length_field,
                       double* x_out,
                       AlivenessTemp* aliveness_temp);
ExtensiveRules* smooth_timestep_rules_new(void* _self, va_list * app);
void matrix_print(double* mat, size_t m, size_t n);
void matrix_roll(double* matrix, size_t w, size_t h, size_t roll_offset, bool axis);
void antialiased_circle(unsigned int h,
                        unsigned int w,
                        double radius,
                        double* x_out);
typedef struct SmootheLife {
  size_t with;
  size_t height;
  double shape_h;
  double shape_w;
  BasicRules* basic_rules;
} SmootheLife;

typedef struct Multipliers{
  double inner_radius;
  double outer_radius;
  fftw_complex* _M_freq;
  fftw_complex* _N_freq;
  size_t _M_freq_width;
  size_t _M_freq_height;
  size_t _N_freq_width;
  size_t _N_freq_height;
} Multipliers;

typedef struct MultipliersTemp {
  double* inner;
  double* annulus;
  double* outer;
} MultipliersTemp;

void init_multipliers(Multipliers *self, MultipliersTemp *tmp, int width,
                      int height, double inner_radius, double outer_radius);
