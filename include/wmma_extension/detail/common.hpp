#ifndef __WMMAE_SIMT_DETAIL_COMMON__
#define __WMMAE_SIMT_DETAIL_COMMON__
#include <cuda_fp16.h>

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800
namespace nvcuda {
namespace wmma {
namespace precision {
class tf32;
} // namespace precision
} // namespace wmma
} // namespace nvcuda
#endif

namespace mtk {
namespace wmma {
namespace mma_simt {
template <class Use, int m, int n, int k, class T, class Layout = void>
class fragment;

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

template <class Use, int M, int N, int K> struct get_M;
template <int M, int N, int K> struct get_M<nvcuda::wmma::matrix_a   , M, N, K>{static const int value = M;};
template <int M, int N, int K> struct get_M<nvcuda::wmma::matrix_b   , M, N, K>{static const int value = K;};
template <int M, int N, int K> struct get_M<nvcuda::wmma::accumulator, M, N, K>{static const int value = M;};

template <class Use, int M, int N, int K> struct get_N;
template <int M, int N, int K> struct get_N<nvcuda::wmma::matrix_a   , M, N, K>{static const int value = K;};
template <int M, int N, int K> struct get_N<nvcuda::wmma::matrix_b   , M, N, K>{static const int value = N;};
template <int M, int N, int K> struct get_N<nvcuda::wmma::accumulator, M, N, K>{static const int value = N;};

template <class Layout, int col_value, int row_value> struct layout_switch;
template <int col_value, int row_value> struct layout_switch<nvcuda::wmma::col_major, col_value, row_value> {static const int value = col_value;};
template <int col_value, int row_value> struct layout_switch<nvcuda::wmma::row_major, col_value, row_value> {static const int value = row_value;};
} // namespace detail

template <class T, class S, int size>
__device__ inline void fill_fragment(__frag_base<float, size>& f, const S v) {
#pragma unroll
	for (unsigned i = 0; i < f.num_elements; i++)
		f.x[i] = detail::cast_to<T>(v);
}

template <class Use, int M, int N, int K, class T, class Layout>
__device__ inline void fill_zero(mtk::wmma::mma_simt::fragment<Use, M, N, K, T, Layout>& frag) {
	fill_fragment(frag, 0.0f);
}

} // namespace mma_simt
} // namespace wmma
} // namespace mtk
#endif
