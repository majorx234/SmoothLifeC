cmake_minimum_required(VERSION 3.16)
project(SmoothLifeC)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)

find_package(raylib)

set(SMOOTH_LIFE_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/main.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/smooth_life.c
)
add_executable(smooth_life)
target_sources(smooth_life
 PRIVATE
  ${SMOOTH_LIFE_SOURCES}
)
target_include_directories(smooth_life
 PRIVATE
  include
)
target_link_libraries(smooth_life
 PRIVATE
  ${raylib_LIBRARIES}
  m
)
