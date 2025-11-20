#include <stddef.h>
#include <stdio.h>
#include <assert.h>
#include "smooth_life.h"

int main(int argc, char **argv) {
  printf("tests\n");
}

bool test_clamp2() {
  assert(clamp2(5.0, 0.0, 1.0) == 1.0);
  printf("All clamp tests passed.\n");
}
