#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>
#include "pca_utils.h"


/*--------------------------------------------------------------------------------*/

double *vector(long n)
{
  double *data;
  data=(double *)malloc(sizeof(double)*n);
  assert(data!=NULL);

  return data;

}

/*--------------------------------------------------------------------------------*/

double **matrix(long n,long m)
{
  double *data, **ptrvec;
  data=(double *)malloc(sizeof(double)*n*m);
  assert(data!=NULL);
  ptrvec=(double **)malloc(sizeof(double *)*n);
  assert(ptrvec!=NULL);
  for (int i=0;i<n;i++)
    ptrvec[i]=&data[i*m];

  return ptrvec;
}
/*--------------------------------------------------------------------------------*/

float **fmatrix(long n,long m)
{
  float *data, **ptrvec;
  data=(float *)malloc(sizeof(float)*n*m);
  assert(data!=NULL);
  ptrvec=(float **)malloc(sizeof(float *)*n);
  assert(ptrvec!=NULL);
  for (int i=0;i<n;i++)
    ptrvec[i]=&data[i*m];

  return ptrvec;
}
/*--------------------------------------------------------------------------------*/
void free_matrix(double **mat)
{
  free(mat[0]);
  free(mat);
  return;
}
/*--------------------------------------------------------------------------------*/
void free_fmatrix(float **mat)
{
  free(mat[0]);
  free(mat);
}
/*--------------------------------------------------------------------------------*/

void tick(pca_time *tt)
{
  gettimeofday(tt,NULL);
}
/*--------------------------------------------------------------------------------*/
void tock(pca_time *tt)
{
  pca_time tnow;
  gettimeofday(&tnow,NULL);
  double   dt=(tnow.tv_usec-tt->tv_usec)/1.0e6+(tnow.tv_sec-tt->tv_sec);  
  fprintf(stderr,"Tock registers %14.4e seconds.\n",dt);
  
}
double tock_ret(pca_time *tt)
{
  pca_time tnow;
  gettimeofday(&tnow,NULL);
  double   dt=(tnow.tv_usec-tt->tv_usec)/1.0e6+(tnow.tv_sec-tt->tv_sec);  
  return dt;  
}
/*--------------------------------------------------------------------------------*/
double tocksilent(pca_time *tt)
{
  pca_time tnow;
  gettimeofday(&tnow,NULL);
  return (tnow.tv_usec-tt->tv_usec)/1.0e6+(tnow.tv_sec-tt->tv_sec);  
  
}
/*--------------------------------------------------------------------------------*/
void tocktag(pca_time *tt,char *msg)
{
  pca_time tnow;
  gettimeofday(&tnow,NULL);
  double   dt=(tnow.tv_usec-tt->tv_usec)/1.0e6+(tnow.tv_sec-tt->tv_sec);  
  if (!msg)
    fprintf(stdout,"Tock registers %14.4e seconds.\n",dt);
  else
    fprintf(stdout,"%14.4e: %s\n",dt,msg);
  
}
/*--------------------------------------------------------------------------------*/
