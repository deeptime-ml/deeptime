void _update_pout(int* obs, double* weights, int T, int N, int M, double* pout)
/* Updates the discrete HMM output probabilities

    Parameters
    ----------
    obs : ptr to int array, size T
        observation sequence amongst M symbols
    weights : ptr to double array, size (T, N)
        probability weights
    sigmas : ptr to double array, size N
        standard deviations
    T : int
        number of time steps
    N : int
        number of hidden states
    M : int
        number of symbols
    pout : ptr to output probability matrix, size (N, M)
        will be updated
*/
{
    int i, j, t, o;

    for (t=0; t<T; t++)
    {
        // current observed symbol
        o = obs[t];
        for (i=0; i<N; i++)
        {
            pout[i*M + o] += weights[t*N + i];
        }
    }
}