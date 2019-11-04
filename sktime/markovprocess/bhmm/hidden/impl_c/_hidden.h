#ifndef HMM_H_
#define HMM_H_


#define _BHMM_ERR_NO_MEM 2
/*
 API FUNCTIONS
*/

double _forward(
        double *alpha,
        const double *A,
        const double *pobs,
        const double *pi,
        int N, int T);

void _backward(
        double *beta,
        const double *A,
        const double *pobs,
        int N, int T);

void _computeGamma(
        double *gamma,
        const double *alpha,
        const double *beta,
        int T, int N);

void _compute_state_counts(
        double *state_counts,
        const double *gamma,
        int T, int N);

int _compute_transition_counts(
        double *transition_counts,
        const double *A,
        const double *pobs,
        const double *alpha,
        const double *beta,
        int N, int T);

int _compute_viterbi(
        int *path,
        const double *A,
        const double *pobs,
        const double *pi,
        int N, int T);

int _sample_path(
        int *path,
        const double *alpha,
        const double *A,
        const double *pobs,
        const int N, const int T);

/*
 HELPER FUNCTIONS
*/
int argmax(double* v, int N);
int _random_choice(const double* p, const int N);
void _normalize(double* v, const int N);


/*void computeGamma(
		double *gamma,
		const double *alpha,
		const double *beta,
		int T, int N);

void computeXi(
		double *xi,
		const double *A,
		const double *B,
		const short *O,
		const double *alpha,
		const double *beta,
		int N, int M, int T);

void update(
		double *A,
		double *B,
		double *pi,
		const short *O,
		const double *gamma,
		const double *xi,
		int N, int M, int T);

void update_multiple(
		double *A,
		double *B,
		const double *weights,
		const double *nomsA,
		const double *denomsA,
		const double *nomsB,
		const double *denomsB,
		int N, int M, int K);
		
float forward32(
		float *alpha,
		float *scaling,
		const float *A,
		const float *B,
		const float *pi,
		const short *O,
		int N, int M, int T);

float forward_no_scaling32(
		float *alpha,
		const float *A,
		const float *B,
		const float *pi,
		const short *O,
		int N, int M, int T);

void backward32(
		float *beta,
		const float *A,
		const float *B,
		const short *O,
		const float *scaling,
		int N, int M, int T);

void backward_no_scaling32(
        float *beta,
        const float *A,
        const float *B,
        const short *ob,
        int N, int M, int T);

void compute_nomA32(
		float *nomA,
		const float *A,
		const float *B,
		const short *O,
		const float *alpha,
		const float *beta,
		int N, int M, int T);

void compute_denomA32(
		float *denomA,
		const float *gamma,
		int T, int N);

void compute_nomB32(
		float *nomB,
		const float *gamma,
		const short *O,
		int N, int M, int T);

void computeGamma32(
		float *gamma,
		const float *alpha,
		const float *beta,
		int T, int N);

void computeXi32(
		float *xi,
		const float *A,
		const float *B,
		const short *O,
		const float *alpha,
		const float *beta,
		int N, int M, int T);

void update32(
		float *A,
		float *B,
		float *pi,
		const short *O,
		const float *gamma,
		const float *xi,
		int N, int M, int T);

void update_multiple32(
		float *A,
		float *B,
		const float *weights,
		const float *nomsA,
		const float *denomsA,
		const float *nomsB,
		const float *denomsB,
		int N, int M, int K);
		*/
#endif /* HMM_H_ */
