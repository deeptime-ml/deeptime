double gaussian(double o, double mu, double sigma);

void _p_o(const double o, const double* mus, const double* sigmas, const int N, double* p);

void _p_obs(const double* o, const double* mus, const double* sigmas, const int N, const int T, double* p);
