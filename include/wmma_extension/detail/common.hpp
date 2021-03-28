#ifndef __WMMAE_SIMT_DETAIL_COMMON__
#define __WMMAE_SIMT_DETAIL_COMMON__
namespace mtk {
namespace wmma {
namespace mma_simt {
template <typename T, int size>
struct __align__(4) __frag_base {
	T x[size];
	enum {num_elements = size};
};
} // namespace mma_simt
} // namespace wmma
} // namespace mtk
#endif
