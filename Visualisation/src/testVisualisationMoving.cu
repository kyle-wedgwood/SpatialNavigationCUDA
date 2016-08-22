#include <iostream>
#include <cstdlib>
#include <arrayfire.h>
#include <cuda_runtime.h>
#define pi 3.141592654f
#define WIDTH 1024 // WIDTH of image
#define HEIGHT 1024 // WIDTH of image

using namespace af;

__global__ void FunctionKernel( float* pVal, const float t, const unsigned int N)
{
  unsigned int index = threadIdx.x+blockDim.x*blockIdx.x;
  int x;
  int y;
  float local_val;
  if (index<N*N)
  {
    x = index/N;
    y = index%N;
    local_val = cos(t)*sin(8.0f*pi*x/N)*sin(8.0*pi*y/N);
    local_val++;
    local_val *= 0.5f;
    pVal[index] = local_val;
  }
}

void SaveData( int npts, float *x, char *filename) {
  FILE *fp = fopen(filename,"w");
  for (int i=0;i<npts;i++) {
    fprintf(fp,"%f\n",x[i]);
  }
  fclose(fp);
}

int main( int argc, char* argv[])
{
  const unsigned int N = 1<<10;

  // Allocate memory
  array data( N, N);
  array RGBdata( N, N, 3);

  float* p_dev_val = data.device<float>();

  const unsigned int noThreads = 1024;
  const unsigned int noBlocks = (N*N+noThreads-1)/noThreads;

  printf("** Testing Visualisation using Forge **\n");

  Window window( N, N, "Test Visualisation");
  window.setColorMap(AF_COLORMAP_SPECTRUM);

  float time = 0.0f;
  float dt = 0.1f;
  float endTime = 100.0f;

  // Evaluate the function
  do
  {
    std::cout << "Time = " << time << std::endl;
    FunctionKernel<<<noBlocks,noThreads>>>( p_dev_val, time, N);
    RGBdata = gray2rgb( data);
    window.image(RGBdata);
    time += dt;
  } while ((time<endTime) && (!window.close()));

  // Clean up
  data.unlock();
  cudaFree( p_dev_val);

  return 0;
}
