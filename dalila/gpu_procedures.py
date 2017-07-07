from pycuda import compiler, gpuarray
import pycuda.autoinit


def _non_negative_projection_GPU(x, nn, matrix_type=None):
    if nn == "both" or nn==matrix_type:
      zeros = gpuarray.zeros_like(x)
      return  gpuarray.maximum(zeros, x)
    return x


mod = compiler.SourceModule("""
__global__ void L1Kernel(float *a, float lambda, float *c, int rows, int cols)
{
   int row = blockIdx.y * blockDim.y + threadIdx.y;
   int col = blockIdx.x * blockDim.x + threadIdx.x;

   c[row*cols+col] = (!signbit(a[row*cols+col])*2 - 1)*
                       max(fabs(a[row*cols+col]) - lambda, 0.0);
}
""")
L1prox = mod.get_function("L1Kernel")


mod = compiler.SourceModule("""
__global__ void L2Kernel(float *a, float lambda, float *c, int rows, int cols)
{
   int row = blockIdx.y * blockDim.y + threadIdx.y;
   int col = blockIdx.x * blockDim.x + threadIdx.x;

   int sum = 0;
   if( row < rows  ){
        for (int i = 0; i < cols; i++ )
            sum += a[i+cols*row]*a[i+cols*row];
        float norm = sqrtf(sum);
        if(col < cols){
            c[row*cols+col] = max(1-(lambda/norm), 0.)*a[row*cols+col];
        }
   }
}
""")
L2prox = mod.get_function("L2Kernel")


mod = compiler.SourceModule("""
__global__ void ENKernel(float *a, float lambda1, float lambda2, float *c,
                         int rows, int cols)
{
   int row = blockIdx.y * blockDim.y + threadIdx.y;
   int col = blockIdx.x * blockDim.x + threadIdx.x;

   c[row*cols+col] = (!signbit(a[row*cols+col])*2 - 1)*
                       max(fabs(a[row*cols+col]) - lambda1, 0.0);

   c[row*cols+col] *= (1 + lambda2);
}
""")
ENprox = mod.get_function("ENKernel")
