#ifndef HMM_H_
#define HMM_H_


#define _BHMM_ERR_NO_MEM 2
#define _BHMM_ERR_RANDOM_SELECTION 3

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

#endif /* HMM_H_ */
