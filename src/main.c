#include <stdbool.h>
#include <float.h>
#include <math.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdio.h>
#include "raylib.h"
#include "raymath.h"
#include <time.h>

// ENV PART
typedef struct Env {
  float delta_time;
  double previous_time;
  double current_time;
  double update_draw_time;
  double wait_time;
  int target_fps;
  float screen_width;
  float screen_height;
  bool rendering;
  void *params;
  float time;
} Env;

int main(int argc, char **argv) {
  Env env = {0};
  env.target_fps = 60;
  env.screen_width = 800;
  env.screen_height = 600;

  // Initialization
  Color background_color = ColorFromHSV(0, 0, 0.05);
  Color foreground_color = ColorFromHSV(0, 0, 0.95);

  printf("starting smooth life\n");
  SetConfigFlags(FLAG_WINDOW_RESIZABLE);
  InitWindow(env.screen_width, env.screen_height, "sequencer_frontend");
  while (!WindowShouldClose()) {
    BeginDrawing();

    ClearBackground(background_color);


    Rectangle boundary;
    boundary.x = 100;
    boundary.y = 100;
    boundary.width  = 200;
    boundary.height = 350;

    DrawRectangleRec(boundary, foreground_color);
    EndDrawing();

    // fps handling
    env.current_time = GetTime();
    env.update_draw_time = env.current_time - env.previous_time;
    env.time += env.delta_time;
    env.wait_time = (1.0f/(float)env.target_fps) - env.update_draw_time;
    if (env.wait_time > 0.0)
    {
      WaitTime((float)env.wait_time);
      env.current_time = GetTime();
      env.delta_time = (float)(env.current_time - env.previous_time);
    }
    env.previous_time = env.current_time;
  }
  CloseWindow();
  return 0;
}
