#ifndef TIMER_H
#define TIMER_H

#ifdef __cplusplus
extern "C" {
#endif

#define CLOCK_MONOTONIC 1

#include <time.h>
#include <sys/time.h>

// Function to get time in milliseconds
double get_time_ms();

#ifdef __cplusplus
}
#endif

#endif // TIMER_H
