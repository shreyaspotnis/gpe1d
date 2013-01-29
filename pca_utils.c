#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>
#include "pca_utils.h"
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
