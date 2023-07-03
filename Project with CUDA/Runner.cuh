#ifndef RUNNER_H
#define RUNNER_H
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#define NUM_RUNNERS 100
#define MAX_SPEED 5.0f
#define MIN_SPEED 1.0f
#define RACE_DISTANCE 100
#define TIME_INTERVAL 1000

class Runner {
public:
    float speed;
    float position;
    Runner();
    __device__ void updatePosition();
};

Runner::Runner() {

    speed = ((float)rand() / RAND_MAX) * (MAX_SPEED - MIN_SPEED) + MIN_SPEED;
    position = 0.0f;
}

__device__ void Runner::updatePosition() {
    position += speed * (TIME_INTERVAL / 1000.0f);
}

#endif