#include <stdio.h>
#include <malloc.h>
#include <math.h>

#include "liblogistic.h"

void logistic_result_ut(
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
	double *label_vec,
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
	logistic_result_ut(input_matrix,weight_vec,size_dim,size_len,output_vec);
	for (iter_dim = 0; iter_dim<size_dim; ++iter_dim) {
		gradient_vec[iter_dim] = 0.0f;
	}
	for (iter_len = 0; iter_len<size_len; ++iter_len) {
		productor = output_vec[iter_len]-label_vec[iter_len];
		for (iter_dim = 0; iter_dim<size_dim; ++iter_dim) {
			gradient_vec[iter_dim] += productor*input_matrix[iter_len*size_dim+iter_dim];
		}
	}
    for (iter_dim = 0; iter_dim<size_dim; ++iter_dim) {
        gradient_vec[iter_dim] /= size_len;
    }
	free(output_vec);
}

int main(){
    double input_matrix[10] = {1,1,2,3,5,8,13,21,34,55};
    double label_vec[5] = {1,1,1,0,0};
    double weight_vec[2] = {1,-5};
    unsigned int size_dim = 2;
    unsigned int size_len = 5;
    double gradient_vec[2] = {0.0f};
    int i = 0;
    
    logistic_gradient(
        input_matrix,
        label_vec,
        weight_vec,
        size_dim,
        size_len,
        gradient_vec);
    printf("Gradient:[%lf,%lf]\n",gradient_vec[0],gradient_vec[1]);
    
    parallel_logistic_gradient(
        input_matrix,
        label_vec,
        weight_vec,
        size_dim,
        size_len,
        gradient_vec,
        2,1,0);
    printf("Gradient:[%lf,%lf]\n",gradient_vec[0],gradient_vec[1]);
    printf("Test Terminated normally.\n");
    return 0;
}
