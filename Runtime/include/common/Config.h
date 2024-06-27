#ifndef MLINSIGHT_CONFIG_H
#define MLINSIGHT_CONFIG_H
/*
 * This file lists all configurations in this project.
 */

/**
 * Configs for logging.h
 */
//This option should always be true. Otherwise, it is hard to make mlinsight compatible with python programs.
//For example if a program relies on another command for output, then the output from mlsinght will affect other programs
//and cause error.
#define SAVE_LOG_TO_FILE true

//Control knobs for different types of log printing
#define PRINT_INFO_LOG false
#define PRINT_DBG_LOG false
#define PRINT_ERR_LOG true


/**
 * Configs for framework support
 */
//The following macros are controlled in CMakeLists.txt
//#define USE_TORCH USE_TORCH
//#define USE_TENSORFLOW USE_TENSORFLOW

/**
 * Configs for profiling support.
 */
#define MEMORY_PROFILING true
#define PYTORCH_ALLOCATOR_SIMULATION false
#define MEMORY_SNAPSHOT true
#define FLAME_GRAPH true
#define FLAME_GRAPH_SUMMARY true
#define GPU_PROFILING true


#endif
