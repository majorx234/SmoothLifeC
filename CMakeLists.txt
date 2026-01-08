cmake_minimum_required(VERSION 3.16)
project(SmoothLifeC)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)

find_package(raylib)

set(SMOOTH_LIFE_LIB_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/smooth_life.c
)
add_library(smooth_life)
target_sources(smooth_life
 PRIVATE
  ${SMOOTH_LIFE_LIB_SOURCES}
)
target_include_directories(smooth_life
 PRIVATE
  include
)
target_link_libraries(smooth_life
 PRIVATE
  m
)

set(SMOOTH_LIFE_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/main.c
)
add_executable(smooth_life_app)
target_sources(smooth_life_app
 PRIVATE
  ${SMOOTH_LIFE_SOURCES}
)
target_include_directories(smooth_life_app
 PRIVATE
  include
)
target_link_libraries(smooth_life_app
 PRIVATE
  ${raylib_LIBRARIES}
)

set(ROLL_EXAMPLE_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/roll_example.c
)
add_executable(roll_example)
target_sources(roll_example
 PRIVATE
  ${ROLL_EXAMPLE_SOURCES}
)
target_include_directories(roll_example
 PRIVATE
  include
)

set(FFTW3_EXAMPLE_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/fftw3_example.c
)
add_executable(fftw3_example)
target_sources(fftw3_example
 PRIVATE
  ${FFTW3_EXAMPLE_SOURCES}
)
target_include_directories(fftw3_example
 PRIVATE
  include
)
target_link_libraries(fftw3_example
  PRIVATE
  fftw3
)

# Tests
set(SMOOTH_LIFE_TEST_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/tests.c
)
add_executable(smooth_life_tests)
target_sources(smooth_life_tests
 PRIVATE
  ${SMOOTH_LIFE_TEST_SOURCES}
)
target_include_directories(smooth_life_tests
 PRIVATE
  include
)
target_link_libraries(smooth_life_tests
 PRIVATE
  ${raylib_LIBRARIES}
  smooth_life
)
