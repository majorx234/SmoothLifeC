#include <stddef.h>
#include <stdio.h>
#include <fftw3.h>

void matrix_print(double* mat, size_t m, size_t n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      printf("%.2f ", mat[i*m+j]);
    }
    printf("\n");
  }
}

void imatrix_print(fftw_complex* mat, size_t m, size_t n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      printf("%.2f+i(%.2f) ", mat[i*m+j][0],mat[i*m+j][i]);
    }
    printf("\n");
  }
}

typedef struct DataTmp{
  double* input;
  fftw_complex *output;
  fftw_plan plan;
} DataTmp;

int main() {
  const size_t width = 6;
  const size_t height = 6;

  DataTmp tmp = {
    .input = (double *)fftw_malloc(sizeof(double) * width * height),
    .output = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * (width/2 + 1) * height),
    .plan = fftw_plan_dft_r2c_2d(width, height, tmp.input, tmp.output, FFTW_ESTIMATE)
  };

  for(size_t i = 0; i < height; i++) {
    for (size_t j = 0; j < width; j++) {
      tmp.input[i * width + j] = ((i*width+i + j) % 2) - 0.5;
    }
  }
  matrix_print(tmp.input, width, height);

  fftw_execute(tmp.plan);

  imatrix_print(tmp.output, (width/2 + 1), height);

  fftw_destroy_plan(tmp.plan);
  fftw_free(tmp.input);
  fftw_free(tmp.output);

  return 0;
}
