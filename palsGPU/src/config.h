#ifndef CONFIG_H_
#define CONFIG_H_

#define OUTPUT_SOLUTION
//#define DEBUG
//#define TIMMING

#define SIMPLE_DELTA
//#define COMPLEX_DELTA
//#define MIXED_DELTA

//#define SINGLE_STEP
//#define MULTI_STEP_GPU
#define MULTI_STEP_CPU

#define MAX_THREAD_COUNT        12

//#define REPORT_EVERY_SECONDS    1
#define REPORT_EVERY_SECONDS    1048576
#define REPORT_EVERY_ITERS      500
//#define REPORT_EVERY_ITERS      1048576

#define PALS_CONVERGENCE        7864318

#define GPU_NO_SHARED_MEMORY

#endif //CONFIG_H_
