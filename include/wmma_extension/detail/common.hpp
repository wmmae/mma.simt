#ifndef __WMMAE_SIMT_DETAIL_COMMON__
#define __WMMAE_SIMT_DETAIL_COMMON__
#include <cuda_fp16.h>
namespace mtk {
namespace wmma {
namespace mma_simt {
namespace detail {
template <typename T, int size>
struct __align__(4) __frag_base {
	T x[size];
	enum {num_elements = size};
};

template <class DST_T, class SRC_T>
__device__ inline DST_T cast_to(const SRC_T src) {return static_cast<DST_T>(src);};
template <>
__device__ inline float cast_to<float, half>(const half src) {return __half2float(src);}
template <>
__device__ inline half cast_to<half, float>(const float src) {return __float2half(src);}
} // namespace detail
} // namespace mma_simt
} // namespace wmma
} // namespace mtk
#endif
