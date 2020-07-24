#ifndef _RNGLIB_
#define _RNGLIB_

extern void advance_state ( int k );
extern int antithetic_get (void );
extern void antithetic_memory ( int i, int *value );
extern void antithetic_set (  int value );
extern void cg_get ( int g, int *cg1, int *cg2 );
extern void cg_memory ( int i, int g, int *cg1, int *cg2 );
extern void cg_set ( int g, int cg1, int cg2 );
extern int cgn_get (void);
extern void cgn_memory ( int i, int *g );
extern void cgn_set ( int g );
extern void get_state ( int *cg1, int *cg2 );
extern int i4_uni (void);
extern void ig_get ( int g, int *ig1, int *ig2 );
extern void ig_memory ( int i, int g, int *ig1, int *ig2 );
extern void ig_set ( int g, int ig1, int ig2 );
extern void init_generator ( int t );
extern void initialize (void);
extern int initialized_get (void);
extern void initialized_memory ( int i, int *initialized );
extern void initialized_set (void);
extern void lg_get ( int g, int *lg1, int *lg2 );
extern void lg_memory ( int i, int g, int *lg1, int *lg2 );
extern void lg_set ( int g, int lg1, int lg2 );
extern int multmod ( int a, int s, int m );
extern float r4_uni_01 (void);
extern double r8_uni_01 (void);
extern void set_initial_seed ( int ig1, int ig2 );
extern void set_seed ( int cg1, int cg2 );
extern void timestamp (void);

#endif /* _RNGLIB_ */
