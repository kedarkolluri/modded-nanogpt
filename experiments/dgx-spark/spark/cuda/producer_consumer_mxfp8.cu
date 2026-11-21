// SM12x producer/consumer MXFP8 prototype, no TMEM or tcgen05.
// Native fragment mapping: NVIDIA PTX ISA, mma.m16n8k32 and block scaling.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <cfloat>
#include <cmath>

__device__ __forceinline__ unsigned smem_addr(const void* p) {
  return static_cast<unsigned>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void init_bar(uint64_t* p,unsigned count) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(smem_addr(p)), "r"(count) : "memory");
}
__device__ __forceinline__ void signal_bar(uint64_t* p) {
  asm volatile("{ .reg .b64 state; mbarrier.arrive.shared.b64 state, [%0]; }" :: "r"(smem_addr(p)) : "memory");
}
__device__ __forceinline__ void wait_bar(uint64_t* p, unsigned phase) {
  asm volatile("{ .reg .pred done; WAIT: mbarrier.try_wait.parity.shared.b64 done, [%0], %1; @!done bra WAIT; }"
               :: "r"(smem_addr(p)), "r"(phase) : "memory");
}
template<int WARPS> __device__ __forceinline__ void producer_sync() {
  asm volatile("bar.sync 1, %0;" :: "n"(WARPS*32) : "memory");
}
template<int WARPS> __device__ __forceinline__ void consumer_sync() {
  asm volatile("bar.sync 2, %0;" :: "n"(WARPS*32) : "memory");
}
__device__ __forceinline__ int scale_offset(int row,int col,int nc) {
  return ((row/128)*(nc/4)+col/4)*512+(row%32)*16+((row%128)/32)*4+col%4;
}
__device__ __forceinline__ void mma(float* d,unsigned a0,unsigned a1,unsigned a2,unsigned a3,
                                   unsigned b0,unsigned b1,unsigned sa,unsigned sb) {
  asm volatile(
    "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0,0}, %11, {0,0};"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
    : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"r"(sa),"r"(sb));
}

// BM=16; producer warps convert activation rows, consumer warps own N subtiles.
// SERIAL uses the same work/roles but hands the next tile to the producer only
// after current MMA completes. It is a no-overlap control, not the native baseline.
template<int BN,int BK,int PW,int CW,bool SERIAL,bool TRACE>
__global__ void pc_mxfp8(const float* A,const uint8_t* W,const uint8_t* WS,
                         __nv_bfloat16* Y,__nv_bfloat16* AD,
                         int M,int N,int K,uint64_t* events) {
  constexpr int NS=BK/32;
  constexpr int NT=BN/(8*CW);
  __shared__ __align__(16) uint8_t aq[2][16][BK];
  __shared__ uint8_t scales[2][16][NS];
  __shared__ __align__(8) uint64_t ready[2],empty[2];
  const int tid=threadIdx.x,lane=tid%32,warp=tid/32;
  const int m0=blockIdx.x*16,n0=blockIdx.y*BN,tiles=(K+BK-1)/BK;
  const bool sampled=blockIdx.x==0 && blockIdx.y==0;
  if(tid==0) for(int s=0;s<2;s++){init_bar(&ready[s],PW*32);init_bar(&empty[s],CW*32);}
  __syncthreads();
  if(warp<PW) {
    for(int t=0;t<tiles;t++) {
      const int slot=t%2;
      if constexpr(SERIAL) {
        if(t>0) wait_bar(&empty[(t-1)%2],((t-1)/2)%2);
      } else {
        if(t>=2) wait_bar(&empty[slot],((t/2)-1)%2);
      }
      if constexpr(TRACE) {
        producer_sync<PW>();
        if(sampled && tid==0) events[t*7]=clock64();
        producer_sync<PW>();
      }
      for(int task=warp;task<16*NS;task+=PW) {
        const int row=task/NS,ks=task%NS,k=t*BK+ks*32+lane;
        const float x=(m0+row<M && k<K)?A[(m0+row)*K+k]:0.f;
        float maximum=fabsf(x);
        #pragma unroll
        for(int distance=16;distance>0;distance/=2)
          maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,distance));
        const float exponent=ceilf(__log2f(fmaxf(__fdiv_rn(maximum,448.f),FLT_MIN)));
        const float scale=exp2f(exponent);
        const __nv_fp8_e4m3 q(x*exp2f(-exponent));
        aq[slot][row][ks*32+lane]=q.__x;
        if(lane==0) scales[slot][row][ks]=static_cast<uint8_t>(exponent+127.f);
        if(blockIdx.y==0 && m0+row<M && k<K)
          AD[(m0+row)*K+k]=__float2bfloat16_rn(static_cast<float>(q)*scale);
        if constexpr(TRACE) if(task==0) {
          __syncwarp();
          if(sampled && lane==0) events[t*7+6]=clock64();
        }
      }
      if constexpr(TRACE) {
        producer_sync<PW>();
        if(sampled && tid==0) events[t*7+1]=clock64();
      }
      signal_bar(&ready[slot]); // every producer releases its own stores
    }
  } else {
    const int cw=warp-PW,g=lane/4,c=lane%4;
    float acc[NT][4]={};
    const int nc=((K/32+3)/4)*4;
    for(int t=0;t<tiles;t++) {
      const int slot=t%2;
      wait_bar(&ready[slot],(t/2)%2);
      unsigned af[NS][4],sa[NS];
      #pragma unroll
      for(int ks=0;ks<NS;ks++) {
        const int k=ks*32+c*4;
        af[ks][0]=*reinterpret_cast<volatile unsigned*>(&aq[slot][g][k]);
        af[ks][1]=*reinterpret_cast<volatile unsigned*>(&aq[slot][g+8][k]);
        af[ks][2]=*reinterpret_cast<volatile unsigned*>(&aq[slot][g][k+16]);
        af[ks][3]=*reinterpret_cast<volatile unsigned*>(&aq[slot][g+8][k+16]);
        sa[ks]=*reinterpret_cast<volatile uint8_t*>(&scales[slot][g+(c%2)*8][ks]);
      }
      if constexpr(TRACE) consumer_sync<CW>();
      if constexpr(!SERIAL) signal_bar(&empty[slot]);
      if constexpr(TRACE) {
        if(sampled && cw==0 && lane==0) events[t*7+2]=clock64();
        consumer_sync<CW>();
      }
      #pragma unroll
      for(int ks=0;ks<NS;ks++) {
        const int k=t*BK+ks*32+c*4;
        #pragma unroll
        for(int j=0;j<NT;j++) {
          const int n=n0+(cw+j*CW)*8+g;
          const unsigned b0=(n<N && k<K)?*reinterpret_cast<const unsigned*>(W+n*K+k):0;
          const unsigned b1=(n<N && k+16<K)?*reinterpret_cast<const unsigned*>(W+n*K+k+16):0;
          const unsigned sb=(n<N && t*BK+ks*32<K)?WS[scale_offset(n,t*NS+ks,nc)]:127;
          if constexpr(TRACE) if(ks==0 && j==0 && sampled && cw==0 && lane==0)
            events[t*7+4]=clock64();
          mma(acc[j],af[ks][0],af[ks][1],af[ks][2],af[ks][3],b0,b1,sa[ks],sb);
          if constexpr(TRACE) if(ks==NS-1 && j==NT-1) {
            __syncwarp();
            if(sampled && cw==0 && lane==0) events[t*7+5]=clock64();
          }
        }
      }
      if constexpr(TRACE) {
        consumer_sync<CW>();
        if(cw==0 && lane==0) {
          if constexpr(TRACE) if(sampled) events[t*7+3]=clock64();
        }
      }
      if constexpr(SERIAL) signal_bar(&empty[slot]);
    }
    #pragma unroll
    for(int j=0;j<NT;j++) {
      const int n=n0+(cw+j*CW)*8+c*2;
      if(m0+g<M && n<N) Y[(m0+g)*N+n]=__float2bfloat16_rn(acc[j][0]);
      if(m0+g<M && n+1<N) Y[(m0+g)*N+n+1]=__float2bfloat16_rn(acc[j][1]);
      if(m0+g+8<M && n<N) Y[(m0+g+8)*N+n]=__float2bfloat16_rn(acc[j][2]);
      if(m0+g+8<M && n+1<N) Y[(m0+g+8)*N+n+1]=__float2bfloat16_rn(acc[j][3]);
    }
  }
  __syncthreads();
  if(tid==0) for(int s=0;s<2;s++) {
    asm volatile("mbarrier.inval.shared.b64 [%0];"::"r"(smem_addr(&ready[s])):"memory");
    asm volatile("mbarrier.inval.shared.b64 [%0];"::"r"(smem_addr(&empty[s])):"memory");
  }
}

template<int BN,int BK,int PW>
int launch(const float* a,const uint8_t* w,const uint8_t* s,__nv_bfloat16* y,__nv_bfloat16* ad,
           int m,int n,int k,uint64_t* events,bool serial,bool trace,cudaStream_t stream) {
  dim3 grid((m+15)/16,(n+BN-1)/BN);
  constexpr int CW=4;
  if(trace) {
    if(serial) pc_mxfp8<BN,BK,PW,CW,true,true><<<grid,(PW+CW)*32,0,stream>>>(a,w,s,y,ad,m,n,k,events);
    else pc_mxfp8<BN,BK,PW,CW,false,true><<<grid,(PW+CW)*32,0,stream>>>(a,w,s,y,ad,m,n,k,events);
  } else {
    if(serial) pc_mxfp8<BN,BK,PW,CW,true,false><<<grid,(PW+CW)*32,0,stream>>>(a,w,s,y,ad,m,n,k,events);
    else pc_mxfp8<BN,BK,PW,CW,false,false><<<grid,(PW+CW)*32,0,stream>>>(a,w,s,y,ad,m,n,k,events);
  }
  return static_cast<int>(cudaGetLastError());
}

extern "C" int run_pc(const float* a,const uint8_t* w,const uint8_t* s,void* y,void* ad,
                      int m,int n,int k,uint64_t* events,int config,int serial,int trace,void* stream) {
  auto yy=static_cast<__nv_bfloat16*>(y), dd=static_cast<__nv_bfloat16*>(ad);
  auto st=static_cast<cudaStream_t>(stream);
  #define CASE(I,BN,BK,PW) case I:return launch<BN,BK,PW>(a,w,s,yy,dd,m,n,k,events,serial,trace,st)
  switch(config) {
    CASE(0,128,32,4); CASE(1,256,32,4); CASE(2,512,32,4);
    CASE(3,128,64,4); CASE(4,256,64,4); CASE(5,512,64,4);
    CASE(6,256,32,2); CASE(7,512,32,2); CASE(8,256,64,2); CASE(9,512,64,2);
    default:return -1;
  }
  #undef CASE
}
