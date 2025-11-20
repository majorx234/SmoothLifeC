#include <stdint.h>
#include <stddef.h>

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


