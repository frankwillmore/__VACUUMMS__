#include <stdio.h>
#include <time.h>
#include <unistd.h>

#define GIG 1073741824

__device__ char d_data[GIG];
__device__ char *d_c;

float sum;
int i;
char h_data[GIG];
char *h_c;

main(){
  clock_t start, end;
  double elapsed;

  start = clock();

printf("transferring 1GB back and forth...\n");
  for (i=0;i<1;i++){
    cudaMemcpyToSymbol(d_data, h_data, GIG, 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(h_data, d_data, GIG, 0, cudaMemcpyDeviceToHost);
  }
  end = clock();
  elapsed = ((double) (end - start)) / CLOCKS_PER_SEC;
  printf("time elapsed is %f.\n\n", elapsed);

  start = clock();

printf("transferring 1B back and forth...\n");
  for (i=0;i<100000;i++){
    cudaMemcpyToSymbol(d_c, h_c, 1, 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(h_c, d_c, 1, 0, cudaMemcpyDeviceToHost);
  }
  end = clock();
  elapsed = ((double) (end - start)) / CLOCKS_PER_SEC;
  printf("time elapsed is %f.\n\n", elapsed);

  
  
}
