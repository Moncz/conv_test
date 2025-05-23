#include <stdio.h>
#include "hip/hip_runtime.h"
#include <hip/hip_ext.h>
#include "verfiy.h"
#include "conv2d.h"

using T = _Float16;

__global__ void nchw_to_nhwc_kernel(T *output,
                                    const T *input,
                                    const int n,
                                    const int h,
                                    const int w,
                                    const int c)
{
  const int hw = h * w;
  const int chw = c * hw;
  __shared__ T shbuf[32 * (32 + 1)];
  const int32_t tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int32_t wid = tid / 32;
  const int32_t lid = tid % 32;
  const int32_t ni = blockIdx.z;
  const int32_t ci0 = blockIdx.y * 32;
  const int32_t hwi0 = blockIdx.x * 32;

  const size_t input_idx = ni * chw + (ci0 + wid) * hw + hwi0;
  const T *A = input + input_idx;
  if (hwi0 + lid < hw)
  {
    const int lid_x_33 = lid * 33;
    if ((ci0 + 32) <= c)
    {
      int ci = wid; // between 0 and 7
      for (int cLoopIdx = 0; cLoopIdx < 4; cLoopIdx++)
      {
        shbuf[lid_x_33 + ci] = A[lid];
        A = &A[8 * hw];
        ci += 8;
      }
    }
    else
    {
      for (int ci = wid; ci < 32; ci += 8)
      {
        if ((ci + ci0) < c)
        {
          shbuf[lid_x_33 + ci] = A[lid];
        }
        A = &A[8 * hw];
      }
    }
  }
  __syncthreads();

  const int32_t ciOut = ci0 + lid;
  output = &output[ni * chw + ciOut];
  if (ciOut < c)
  {
    if (hwi0 + 32 < hw)
    {
      int hwI = wid;
      for (int hwLoopIdx = 0; hwLoopIdx < 4; ++hwLoopIdx)
      {
        output[(hwi0 + hwI) * c] = shbuf[(hwI) * 33 + lid];
        hwI += 8;
      }
    }
    else
    {
      for (int hwI = wid; hwI < 32; hwI += 8)
      {
        if (hwi0 + hwI < hw)
        {
          output[(hwi0 + hwI) * c] = shbuf[(hwI) * 33 + lid];
        }
      }
    }
  }
}

template <typename T>
__global__ void nhwc_to_nchw_kernel(T *output,
                                    const T *input,
                                    const int n,
                                    const int h,
                                    const int w,
                                    const int c)
{

  const int hw = h * w;
  const int hwc = hw * c;
  __shared__ T shbuf[32 * (32 + 1)];
  const int32_t tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int32_t wid = tid / 32;
  const int32_t lid = tid % 32;
  const int32_t ni = blockIdx.z;
  const int32_t hwi0 = blockIdx.y * 32;
  const int32_t ci0 = blockIdx.x * 32;

  const size_t input_idx = ni * hwc + (hwi0 + wid) * c + ci0;
  const T *A = input + input_idx;
  if (ci0 + lid < c)
  {
    const int lid_x_33 = lid * 33;
    if ((hwi0 + 32) <= hw)
    {
      int hwi = wid; // between 0 and 7
      for (int cLoopIdx = 0; cLoopIdx < 4; cLoopIdx++)
      {
        shbuf[lid_x_33 + hwi] = A[lid];
        A = &A[8 * c];
        hwi += 8;
      }
    }
    else
    {
      for (int hwi = wid; hwi < 32; hwi += 8)
      {
        if ((hwi + hwi0) < hw)
        {
          shbuf[lid_x_33 + hwi] = A[lid];
        }
        A = &A[8 * c];
      }
    }
  }
  __syncthreads();

  const int32_t hwiOut = hwi0 + lid;
  output = &output[ni * hwc + hwiOut];
  if (hwiOut < hw)
  {
    if (ci0 + 32 < c)
    {
      int cI = wid;
      for (int hwLoopIdx = 0; hwLoopIdx < 4; ++hwLoopIdx)
      {
        output[(ci0 + cI) * hw] = shbuf[(cI) * 33 + lid];
        cI += 8;
      }
    }
    else
    {
      for (int cI = wid; cI < 32; cI += 8)
      {
        if (ci0 + cI < c)
        {
          output[(ci0 + cI) * hw] = shbuf[(cI) * 33 + lid];
        }
      }
    }
  }
}

__global__ void reduce2d(int size, _Float16 *out)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;

  if (x >= size)
  {
    return;
  }
  float sum = 0.f;
  for (int iter = 0; iter < 3; iter++)
  {
    sum += (float)out[iter * size + x];
  }
  out[x] = (_Float16)sum;
}

int main(int argc, char **argv)
{
  int n = atoi(argv[1]);
  int c = atoi(argv[2]);
  int h = atoi(argv[3]);
  int w = atoi(argv[4]);
  int k = atoi(argv[5]);
  int r = atoi(argv[6]);
  int s = atoi(argv[7]);
  int u = atoi(argv[8]);
  int v = atoi(argv[9]);
  int p = atoi(argv[10]);
  int q = atoi(argv[11]);

  int outh = (h - r + 2 * p) / u + 1;
  int outw = (w - s + 2 * q) / v + 1;

  double M = k;
  double N = n * outh * outw;
  double K = c * r * s;
  double temp = n * outh * outw * 1e-9f;
  double flopsPerConv = temp * M * K * 2.0;

  _Float16 *pIn = (_Float16 *)malloc(n * c * h * w * sizeof(_Float16));
  _Float16 *pIn_trans = (_Float16 *)malloc(n * c * h * w * sizeof(_Float16));
  _Float16 *pWeight = (_Float16 *)malloc(k * c * r * s * sizeof(_Float16));
  _Float16 *pWeight_trans = (_Float16 *)malloc(k * c * r * s * sizeof(_Float16));
  _Float16 *pOut = (_Float16 *)malloc(6 * n * k * outh * outw * sizeof(_Float16));
  _Float16 *pOut_trans = (_Float16 *)malloc(6 * n * k * outh * outw * sizeof(_Float16));
  _Float16 *pOut_host = (_Float16 *)malloc(6 * n * k * outh * outw * sizeof(_Float16));

  _Float16 *pIn_device, *pWeight_device, *pOut_device;
  _Float16 *pIn_trans_device, *pWeight_trans_device, *pOut_trans_device;
  hipMalloc((void **)&pIn_device, n * c * h * w * sizeof(_Float16));
  hipMalloc((void **)&pWeight_device, k * c * r * s * sizeof(_Float16));
  hipMalloc((void **)&pOut_device, 6 * n * k * outh * outw * sizeof(_Float16));

  hipMalloc((void **)&pIn_trans_device, n * c * h * w * sizeof(_Float16));
  hipMalloc((void **)&pWeight_trans_device, k * c * r * s * sizeof(_Float16));
  hipMalloc((void **)&pOut_trans_device, 6 * n * k * outh * outw * sizeof(_Float16));

  for (int i = 0; i < n * c * h * w; i++)
  {
    pIn[i] = (rand() % 255) / 255.0;
  }

  for (int i = 0; i < k * c * r * s; i++)
  {
    pWeight[i] = (rand() % 255) / 255.0;
  }

  for (int i = 0; i < 6 * n * k * outh * outw; i++)
  {
    pOut[i] = 0.0;
    pOut_host[i] = 0.0;
  }

  hipMemcpy(pIn_device, pIn, n * c * h * w * sizeof(_Float16), hipMemcpyHostToDevice);
  hipMemcpy(pWeight_device, pWeight, k * c * r * s * sizeof(_Float16), hipMemcpyHostToDevice);
  hipMemcpy(pOut_device, pOut, 6 * n * k * outh * outw * sizeof(_Float16), hipMemcpyHostToDevice);

  /********************step 1*****************************/

  problem_t problem;
  int paramSize;
  kernelInfo_t kernelInfo;

  FastDivmod divmod_rs(r * s);
  FastDivmod divmod_s(s);
  FastDivmod divmod_pq(outh * outw);
  FastDivmod divmod_q(outw);
  FastDivmod divmod_c(c);

  problem.in = pIn_trans_device;
  problem.weight = pWeight_trans_device;
  // problem.out       = pOut_trans_device;
  problem.out = pOut_device;
  problem.n = n;
  problem.c = c;
  problem.h = h;
  problem.w = w;
  problem.k = k;
  problem.r = r;
  problem.s = s;
  problem.u = u;
  problem.v = v;
  problem.p = p;
  problem.q = q;
  problem.divmod_rs = divmod_rs;
  problem.divmod_s = divmod_s;
  problem.divmod_pq = divmod_pq;
  problem.divmod_q = divmod_q;
  problem.divmod_c = divmod_c;

  /********************************** step 2****************************/
  getParamsize(&problem, &paramSize);
  // printf("paramsize:%d\n", paramSize);
  void *param = malloc(paramSize);

  getkernelInfo(&problem, &kernelInfo, param);

  dim3 groups(kernelInfo.blockx, kernelInfo.blocky, kernelInfo.blockz);
  dim3 threads(kernelInfo.threadx, kernelInfo.thready, kernelInfo.threadz);
  int ldsSize = kernelInfo.dynmicLdsSize;

  dim3 block_act(32, 8, 1);
  dim3 grid_act((h * w + 31) / 32, (c + 31) / 32, n);
  dim3 block_flt(32, 8, 1);
  dim3 grid_flt((r * s + 32 - 1) / 32, (c + 32 - 1) / 32, k);
  dim3 block_out(32, 8, 1);
  dim3 grid_out((k + 31) / 32, (outh * outw + 31) / 32, n);
  const int THREAD_NUM = 256;
  dim3 block_reduce(THREAD_NUM, 1, 1);
  dim3 grid_reduce((n * k * outh * outw + THREAD_NUM - 1) / THREAD_NUM, 1, 1);

  /*******************************warm up and get result************************************/
  nchw_to_nhwc_kernel<<<grid_act, block_act>>>(pIn_trans_device, pIn_device, n, h, w, c);
  nchw_to_nhwc_kernel<<<grid_flt, block_flt>>>(pWeight_trans_device, pWeight_device, k, r, s, c);
  hipExtLaunchKernel(kernelInfo.kernelPtr, groups, threads, (void **)&param, ldsSize, 0, 0, 0, 0);
  if (c == 1920)
  {
    reduce2d<<<grid_reduce, block_reduce>>>(n * k * outh * outw, pOut_device);
  }
  hipMemcpy(pOut_host, pOut_device, n * k * outh * outw * sizeof(_Float16), hipMemcpyDeviceToHost);

  /*******************************cost time test************************************/
  hipEvent_t start, stop;
  hipEventCreate(&start);
  hipEventCreate(&stop);
  hipEventRecord(start, 0);
  float time_elapsed = 0.0;

  int iternum = 10;
  for (int i = 0; i < iternum; i++)
  {
    nchw_to_nhwc_kernel<<<grid_act, block_act>>>(pIn_trans_device, pIn_device, n, h, w, c);
    nchw_to_nhwc_kernel<<<grid_flt, block_flt>>>(pWeight_trans_device, pWeight_device, k, r, s, c);
    hipExtLaunchKernel(kernelInfo.kernelPtr, groups, threads, (void **)&param, ldsSize, 0, 0, 0, 0);
    if (c == 1920)
    {
      reduce2d<<<grid_reduce, block_reduce>>>(n * k * outh * outw, pOut_device);
    }
  }
  hipEventRecord(stop, 0);

  hipEventSynchronize(stop);
  hipEventElapsedTime(&time_elapsed, start, stop);
  float timePerConv = time_elapsed / iternum;
  double gflops = flopsPerConv / (timePerConv / 1000.0f);

  // printf("%d\n", n*k*outh*outw);
  // time_elapsed：ms
  printf("time: %f us\n", time_elapsed * 1000 / iternum);
  printf("%f\n", gflops);
  hipEventDestroy(start);
  hipEventDestroy(stop);

  free(param);

  printf("===================start verfiy===================\n");
  conv2dcpu(pIn, pWeight, pOut, n, c, h, w, k, r, s, u, v, p, q);

  int error = 0;
  for (int i = 0; i < n * k * outh * outw; i++)
  {
    float device_out = pOut_host[i];
    if ((fabs(pOut_host[i] - pOut[i])) / pOut_host[i] > 0.01 || isnan(device_out) || isinf(device_out))
    {
      printf("error, postion:%d, gpuvalue:%f, cpuvalue:%f\n", i, (float)pOut_host[i], (float)pOut[i]);
      error++;
      break;
    }
  }
  printf("================finish,error:%d=========================\n", error);

  hipFree(pIn_device);
  hipFree(pWeight_device);
  hipFree(pOut_device);

  free(pIn);
  free(pWeight);
  free(pOut);
  free(pOut_host);

  return 0;
}