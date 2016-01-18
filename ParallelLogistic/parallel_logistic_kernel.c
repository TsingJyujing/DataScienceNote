#include "parallel_logistic_kernel.h"

typedef struct 
    parallel_logistic_gradient_varin
{
    double *input_matrix;
	double *weight_vec;
    double *label_vec;
	unsigned int size_dim;
	unsigned int size_len;
	double *gradient_vec;
}
    logistic_gradient_varin;

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

void *thread_logistic_gradient(void *argv){
    logistic_gradient_varin *varsin = argv;
    
    double *input_matrix =  varsin->input_matrix;
    double *label_vec =     varsin->label_vec;
    double *weight_vec =    varsin->weight_vec;
    unsigned int size_dim = varsin->size_dim;
    unsigned int size_len = varsin->size_len;
    double *gradient_vec =  varsin->gradient_vec;
    
    unsigned int iter_dim = 0;
	unsigned int iter_len = 0;
	double *output_vec = NULL;
	double productor = 0;
	output_vec = (double *)malloc(sizeof(double)*size_len);
	if (!output_vec) return;
	logistic_result(
        input_matrix,
        weight_vec,
        size_dim,size_len,
        output_vec);
	for (iter_dim = 0; iter_dim<size_dim; ++iter_dim) {
		gradient_vec[iter_dim] = 0.0f;
	}
	for (iter_len = 0; iter_len<size_len; ++iter_len) {
		productor = output_vec[iter_len]-label_vec[iter_len];
		for (iter_dim = 0; iter_dim<size_dim; ++iter_dim) {
			gradient_vec[iter_dim] += productor*
                input_matrix[iter_len*size_dim+iter_dim];
		}
	}
	free( output_vec );
}

void parallel_logistic_gradient(
	double *input_matrix,
	double *label_vec,
	double *weight_vec,
	unsigned int size_dim,
	unsigned int size_len,
	double *gradient_vec,
    int parallel_pool_count,
    int batch_normalization_blocks,
    int batch_normalization_bid
    )
{
    int i = 0, 
        j = 0;
    int rc = 0;
    int data_per_block = size_len/parallel_pool_count;
    unsigned int sum_all = 0;
    logistic_gradient_varin *thread_parameters = NULL;
    double *gradient_result = NULL;
    pthread_t *threads; 
    pthread_attr_t attr;
    void *status;
    
    thread_parameters = (logistic_gradient_varin *)
        malloc(sizeof(logistic_gradient_varin)*parallel_pool_count);
    
    if(!thread_parameters){
        //malloc ram failed.
        #ifdef DEBUG_FLAG
        printf("malloc ram failed while generating parameters.\n");
        #endif
        return;
    }
    
    gradient_result = (double *)
        malloc(sizeof(double)*size_dim*parallel_pool_count);
        
    if(!gradient_result){
        //malloc ram failed.
        #ifdef DEBUG_FLAG
        printf("malloc ram failed while generating gradient buffer.\n");
        #endif
        return;
    }

    threads = (pthread_t *)malloc(sizeof(pthread_t)*parallel_pool_count);
    if (!threads){
        //malloc ram failed.
        #ifdef DEBUG_FLAG
        printf("malloc ram failed while creating thread space.\n");
        #endif
        return;
    }
    
    //Generate parallel pool parameters
    #ifdef DEBUG_FLAG
        printf("Generating parallel pool parameters\n");
    #endif
    for (i = 0; i<parallel_pool_count; ++i){
        int start_index = 0;
        int end_index = 0;
        
        //No batch normalization
        start_index = i*data_per_block;
        end_index = (i+1)*data_per_block - 1;
        if (i==(parallel_pool_count-1)){
            end_index = size_len-1;
        }
        
        if (batch_normalization_blocks>1){
            //Batch normalization with bid
            int data_num_in_block = end_index - start_index + 1;
            int data_per_batch = 
                data_num_in_block / batch_normalization_blocks;
            int start_offset = batch_normalization_bid * data_per_batch;
            int end_offset = 0;
            
            
            if (batch_normalization_bid==(batch_normalization_blocks-1)){
                //Final batch
                end_offset = data_num_in_block - 1;
            }else{
                //Normal batch
                end_offset = (batch_normalization_bid + 1) * data_per_batch - 1;
            }
            start_index += start_offset;
            end_index = start_index + end_offset - start_offset;
        }
        
        thread_parameters[i].input_matrix=
            input_matrix+start_index*size_dim;
        thread_parameters[i].size_dim = size_dim;
        thread_parameters[i].size_len = 
            end_index-start_index+1;
        thread_parameters[i].label_vec = 
            label_vec + start_index;
        thread_parameters[i].gradient_vec = 
            gradient_result + i*size_dim;
        thread_parameters[i].weight_vec = weight_vec;
        #ifdef DEBUG_FLAG
        printf("Start:%u End:%u \n",start_index,end_index);
        #endif
    }
    
    pthread_attr_init(&attr); 
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE); 
    #ifdef DEBUG_FLAG
        printf("Generating threads\n");
    #endif
    for (i = 0; i<parallel_pool_count; ++i){
        rc = pthread_create(
            &threads[i],
            &attr,
            thread_logistic_gradient,
            &thread_parameters[i]
        );
        if (rc) {
            //has some fault while creating thread
            #ifdef DEBUG_FLAG
            printf("error while creating threads.\n");
            #endif
            pthread_exit(NULL);
            return;
        }
    }
    
    for (i = 0; i<parallel_pool_count; ++i){
        rc = pthread_join(threads[i],&status);
        if (rc) {
            //has some fault while creating thread
            #ifdef DEBUG_FLAG
            printf("error while joining threads.\n");
            #endif
            pthread_exit(NULL);
            return;
        }
    }
    #ifdef DEBUG_FLAG
        printf("Threads terminated normally~\n");
    #endif
    for (i = 0; i<parallel_pool_count; ++i){
        sum_all += thread_parameters[i].size_len;
        #ifdef DEBUG_FLAG
        printf("Size of subData:%u \n",thread_parameters[i].size_len);
        #endif
    }
    #ifdef DEBUG_FLAG
        printf("Size of Data:%u \n",sum_all);
    #endif
    for (j = 0; j <size_dim; ++j){
        gradient_vec[j] = 0.0f;
        for (i = 0; i<parallel_pool_count; ++i){
            gradient_vec[j]+=
                gradient_result[i*size_dim+j];
        }
        gradient_vec[j] /= sum_all;
    }
    #ifdef DEBUG_FLAG
        printf("PLKC_Gradient:[%lf,%lf]\n",gradient_vec[0],gradient_vec[1]);
    #endif

    free(thread_parameters);
    #ifdef DEBUG_FLAG
        printf("thread_parameters has free.\n");
    #endif
    free(gradient_result);
    #ifdef DEBUG_FLAG
        printf("gradient_result has free.\n");
    #endif
    free(threads);
    #ifdef DEBUG_FLAG
        printf("All RAM has free.\n");
    #endif
    pthread_attr_destroy(&attr); 
    #ifdef DEBUG_FLAG
        printf("Attr has destroyed successfully.\n");
    #endif
    return;
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
			hessian_matrix[i*size_dim+j] = mat_val/size_len;
			hessian_matrix[j*size_dim+i] = mat_val/size_len;
		}
	}
	free(output_vec);
}
