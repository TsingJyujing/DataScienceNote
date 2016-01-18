#ifndef _PARALLEL_LOGISTIC_KERNEL_
#define _PARALLEL_LOGISTIC_KERNEL_
//#define DEBUG_FLAG
#ifdef DEBUG_FLAG
	#include <stdio.h>
#endif
#include <math.h>
#include <malloc.h>
#include <pthread.h>

extern void logistic_result(
	double *input_matrix,
	double *weight_vec,
	unsigned int size_dim,
	unsigned int size_len,
	double *output_vec);
    
extern void parallel_logistic_gradient(
	double *input_matrix,
	double *label_vec,
	double *weight_vec,
	unsigned int size_dim,
	unsigned int size_len,
	double *gradient_vec,
    int parallel_pool_count,
    int batch_normalization_blocks,
    int batch_normalization_bid);
    
extern void logistic_hessian_matrix_sum(
	double *input_matrix,
	double *weight_vec,
	unsigned int size_dim,
	unsigned int size_len,
	double *hessian_matrix);
    
#endif
