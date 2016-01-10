#define  DLL_EXPORT

#include <stdio.h>
#include <math.h>
#include <malloc.h>

#include "logistic_kernel.h"

void logistic_result(
	double *input_matrix,
	double *weight_vec,
	unsigned int size_dim,
	unsigned int size_len,
	double *output_vec)
{
	unsigned int iter_dim = 0;
	unsigned int iter_len = 0;
	double wx = 1.0f;
	for (iter_len = 0; iter_len<size_len; ++iter_len) {
		wx = 1.0f;
		for (iter_dim = 0; iter_dim<size_dim; ++iter_dim) {
			wx += weight_vec[iter_dim]*
				input_matrix[iter_len*size_dim+iter_dim];
		}
		output_vec[iter_len] = 1/(1+exp(wx));
	}
}

void logistic_gradient(
	double *input_matrix,
	double *lable_vec,
	double *weight_vec,
	unsigned int size_dim,
	unsigned int size_len,
	double *gradient_vec)
{
	unsigned int iter_dim = 0;
	unsigned int iter_len = 0;
	double *output_vec = NULL;
	double productor = 0;
	output_vec = (double *)malloc(sizeof(double)*size_len);
	if (!output_vec) return;
	logistic_result(input_matrix,weight_vec,size_dim,size_len,output_vec);
	for (iter_dim = 0; iter_dim<size_dim; ++iter_dim) {
		gradient_vec[iter_dim] = 0.0f;
	}
	for (iter_len = 0; iter_len<size_len; ++iter_len) {
		productor = output_vec[iter_len]-lable_vec[iter_len];
		for (iter_dim = 0; iter_dim<size_dim; ++iter_dim) {
			gradient_vec[iter_dim] += productor*input_matrix[iter_len*size_dim+iter_dim];
		}
	}
	free(output_vec);
}

void logistic_hessian_matrix_sum(
	double *input_matrix,
	double *weight_vec,
	unsigned int size_dim,
	unsigned int size_len,
	double *hessian_matrix
	)
{
	unsigned int iter_dim = 0;
	unsigned int iter_len = 0;
	unsigned int i,j;
	double wx = 1.0f;
	double *output_vec = NULL;
	output_vec = (double *)malloc(sizeof(double)*size_len);
	if (!output_vec) return;
	logistic_result(input_matrix,weight_vec,size_dim,size_len,output_vec);

	for (i = 0; i<size_dim; ++i){
		for (j = 0; j<=i; ++j){
			double mat_val = 0;
			for (iter_len = 0; iter_len<size_len; ++iter_len){
				mat_val += output_vec[iter_len]*(1-output_vec[iter_len])
					*input_matrix[iter_len*size_dim+i]*input_matrix[iter_len*size_dim+j];
			}
			hessian_matrix[i*size_dim+j] = mat_val;
			hessian_matrix[j*size_dim+i] = mat_val;
		}
	}
	free(output_vec);
}

