#ifndef _DLL_BASICUI_H_
#define _DLL_BASICUI_H_
#define DLLMSG_API
#if defined DLL_EXPORT
#define SBUI_API __declspec(dllexport)
#else
#define SBUI_API __declspec(dllimport)
#endif
#ifdef __cplusplus   
extern "C" {
#endif
	SBUI_API void logistic_hessian_matrix_sum(
		double *input_matrix,
		double *weight_vec,
		unsigned int size_dim,
		unsigned int size_len,
		double *hessian_matrix
		);

	SBUI_API void logistic_result(
		double *input_matrix,
		double *weight_vec,
		unsigned int size_dim,
		unsigned int size_len,
		double *output_vec);

	SBUI_API void logistic_gradient(
		double *input_matrix,
		double *lable_vec,
		double *weight_vec,
		unsigned int size_dim,
		unsigned int size_len,
		double *gradient_vec);
#undef  SBUI_API 
#ifdef __cplusplus
}
#endif
#endif