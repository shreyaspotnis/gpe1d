#ifndef PCA_UTILS_H
#define PCA_UTILS_H

#include <stdlib.h>
#include <sys/time.h>

typedef struct timeval pca_time;

/* vector/matrix allocation routines */

double *vector(long n);
double **matrix(long n,long m);
float **fmatrix(long n,long m);
void free_matrix(double **mat);
void free_fmatrix(float **mat);

/* timing routines */

void tick(pca_time *tt);
void tock(pca_time *tt);
double tock_ret(pca_time *tt);
double tocksilent(pca_time *tt);
void tocktag(pca_time *tt,char *msg);

#endif
