#include <stddef.h>
#include <stdio.h>

void print_mat(int* mat, size_t m, size_t n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      printf("%d ", mat[i*m+j]);
    }
    printf("\n");
  }
}

void reverse_array(int* arr,size_t start, size_t end, size_t stride) {
  int left = start, right = end;
  // reverse first n -k elements
  while (left < right) {
    int temp = arr[left*stride];
    arr[left*stride] = arr[right*stride];
    arr[right*stride] = temp;
    left++;
    right--;
  }

}

void roll_array(int *arr, int length, int k, size_t stride) {
  // numpy convention
  k = -k;
  if (length <= 1) return; // No rotation needed for empty or single-element arrays
  if (k < 0) k = length + k;

  k = k % length; // Normalize k to handle cases where k >= n
  if (k == 0) return; // No rotation needed

  // reverse first n -k elements
  reverse_array(arr, 0, k-1,      stride);
  reverse_array(arr, k, length-1, stride);
  reverse_array(arr, 0, length-1, stride);
}

void roll_mat(int* arr, int m, int n, int k, int axis) {
  if (axis == 0) {
    for (size_t i = 0; i < n; i++) {
      roll_array(&arr[i], m, k, 3);
    }
  } else if (axis == 1) {
    for (size_t i = 0; i < n; i++) {
      roll_array(&arr[i*m], m, k, 1);
    }
  } else {
    // TODO
  }
}

// Example usage:
int main() {
  int arr[] = {1, 2, 3, 4, 5, 6, 7};
  int length = sizeof(arr) / sizeof(arr[0]);
  int k = 1;

  roll_array(arr, length, k, 1);

  // Print rotated array
  for (int i = 0; i < length; i++) {
    printf("%d ", arr[i]);
  }
  printf("\n"); // Output: 3 4 5 1 2

  printf("== matrix: ==\n");
  printf("axis = 0\n");
  int mat1[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  int m = 3;
  int n = 3;
  int axis = 0;
  k = 1;
  roll_mat(mat1, m, n, k, 0);
  print_mat(mat1,m,n);

  printf("axis = 1\n");
  int mat2[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  axis = 1;
  k = 1;
  roll_mat(mat2, m, n, k, 1);
  print_mat(mat2,m,n);

  return 0;
}
