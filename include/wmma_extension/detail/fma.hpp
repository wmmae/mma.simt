#ifndef __WMMAE_MMA_SIMT_DETAIL_FMA_HPP__
#define __WMMAE_MMA_SIMT_DETAIL_FMA_HPP__
#include "common.hpp"

namespace mtk {
namespace wmma {
namespace mma_simt {
namespace detail {

template <class A_T, class B_T, class C_T>
__device__ inline C_T fma(const A_T a, const B_T b, const C_T c) {
	const auto fa = cast<float>(a);
	const auto fb = cast<float>(b);
	const auto fc = cast<float>(c);
	return fa * fb + fc;
}

} // namespace detail
} // namespace mma_simt
} // namespace wmma
} // namespace mtk
#endif
