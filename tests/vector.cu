#include <iostream>
#include <type_traits>
#include <wmma_extension/mma_simt.hpp>
#include "utils.hpp"

#ifdef WMMAE_USE_NVCUDA_NAMESPACE
namespace f32_namespace = nvcuda;
#else
namespace f32_namespace = mtk;
#endif

__device__ half abs(const half a) {
	if (__half2float(a) < 0) {
		return -a;
	}
	return a;
}

/// Load

template <class Use, int m, int n, int k, class T, class Layout>
__global__ void load_vector_ab_test_kernel(
		const float* const cor_ptr,
		const float* const src_ptr
		) {
	mtk::wmma::mma_simt::fragment<Use, m, n, k, T, Layout> frag, frag_c;
	mtk::wmma::mma_simt::fill_fragment(frag, 0.0f);

	mtk::wmma::mma_simt::load_vector(frag, src_ptr);

	constexpr unsigned mem_m = mtk::wmma::mma_simt::detail::select_value<Use, m, k, m>::value;
	mtk::wmma::mma_simt::load_matrix_sync(frag_c, cor_ptr, mem_m);

	float max_error = 0;
	for (unsigned i = 0; i < frag.num_elements; i++) {
		max_error = max(max_error, abs(mtk::wmma::mma_simt::detail::cast<float>(frag.x[i] - frag_c.x[i])));
	}
	for (unsigned i = 0; i < mtk::test_utils::warp_size; i++) {
		__syncthreads();
		if (i == threadIdx.x) printf("[%u] %e\n", i, max_error);
	}
}

template <int m, int n, int k, class T>
__global__ void load_vector_acc_test_kernel(
		const float* const cor_ptr,
		const float* const src_ptr,
		const nvcuda::wmma::layout_t layout
		) {
	mtk::wmma::mma_simt::fragment<nvcuda::wmma::accumulator, m, n, k, T, void> frag, frag_c;
	mtk::wmma::mma_simt::fill_fragment(frag, 0.0f);

	mtk::wmma::mma_simt::load_vector(frag, src_ptr, layout);

	constexpr unsigned mem_m = m;
	mtk::wmma::mma_simt::load_matrix_sync(frag_c, cor_ptr, mem_m, layout);

	float max_error = 0;
	for (unsigned i = 0; i < frag.num_elements; i++) {
		max_error = max(max_error, abs(mtk::wmma::mma_simt::detail::cast<float>(frag.x[i] - frag_c.x[i])));
	}
	for (unsigned i = 0; i < mtk::test_utils::warp_size; i++) {
		__syncthreads();
		if (i == threadIdx.x) printf("[%u] %e\n", i, max_error);
	}
}

template <class Use, int m, int n, int k, class T, class Layout>
void load_vector_test() {
	std::printf("!-- %s\n", __func__);
	std::printf("Use    : %s\n", mtk::test_utils::to_string<Use>().c_str());
	std::printf("Layout : %s\n", mtk::test_utils::to_string<Layout>().c_str());
	std::printf("Type   : %s\n", mtk::test_utils::to_string<T>().c_str());
	std::printf("Size   : %u, %u, %u\n", m, n, k);
	std::printf("Shape : <%2u,%2u,%2u>\n",
			m,
			n,
			k);
	constexpr unsigned mem_m = mtk::wmma::mma_simt::detail::select_value<Use, m, k, m>::value;
	constexpr unsigned mem_n = mtk::wmma::mma_simt::detail::select_value<Use, k, n, n>::value;

	constexpr auto vec_len = std::is_same<Layout, nvcuda::wmma::col_major>::value ? mem_m : mem_n;

	float* vec_mem;
	float* mat_mem;

	cudaMallocHost(&mat_mem, sizeof(float) * mem_m * mem_n);
	cudaMallocHost(&vec_mem, sizeof(float) * vec_len);

	for (unsigned i = 0; i < vec_len; i++) {
		vec_mem[i] = i;
	}

	for (unsigned i = 0; i < mem_m * mem_n; i++) {
		mat_mem[i] = 0.f;
	}

	for (unsigned i = 0; i < vec_len; i++) {
		mat_mem[i] = vec_mem[i];
	}

	if constexpr (std::is_same<Use, nvcuda::wmma::accumulator>::value) {
		const auto layout = (std::is_same<nvcuda::wmma::col_major, Layout>::value) ? nvcuda::wmma::mem_col_major : nvcuda::wmma::mem_row_major;
		load_vector_acc_test_kernel<m, n, k, T><<<1, mtk::test_utils::warp_size>>>(mat_mem, vec_mem, layout);
	} else {
		load_vector_ab_test_kernel<Use, m, n, k, T, Layout><<<1, mtk::test_utils::warp_size>>>(mat_mem, vec_mem);
	}

	cudaDeviceSynchronize();

	cudaFree(vec_mem);
	cudaFree(mat_mem);
}

/// Store

template <int m, int n, int k, class T>
__global__ void store_vector_acc_test_kernel(
		float* const dst_ptr,
		const float* const src_ptr,
		const nvcuda::wmma::layout_t layout
		) {
	mtk::wmma::mma_simt::fragment<nvcuda::wmma::accumulator, m, n, k, T, void> frag;

	constexpr unsigned mem_m = m;
	mtk::wmma::mma_simt::load_matrix_sync(frag, src_ptr, mem_m, layout);

	mtk::wmma::mma_simt::store_vector(dst_ptr, frag, layout);
}

template <class Use, int m, int n, int k, class T, class Layout>
void store_vector_test() {
	std::printf("!-- %s\n", __func__);
	std::printf("Use    : %s\n", mtk::test_utils::to_string<Use>().c_str());
	std::printf("Layout : %s\n", mtk::test_utils::to_string<Layout>().c_str());
	std::printf("Type   : %s\n", mtk::test_utils::to_string<T>().c_str());
	std::printf("Size   : %u, %u, %u\n", m, n, k);
	std::printf("Shape : <%2u,%2u,%2u>\n",
			m,
			n,
			k);
	constexpr unsigned mem_m = mtk::wmma::mma_simt::detail::select_value<Use, m, k, m>::value;
	constexpr unsigned mem_n = mtk::wmma::mma_simt::detail::select_value<Use, k, n, n>::value;

	constexpr auto vec_len = std::is_same<Layout, nvcuda::wmma::col_major>::value ? mem_m : mem_n;

	float* vec_mem;
	float* res_mem;
	float* mat_mem;

	cudaMallocHost(&mat_mem, sizeof(float) * mem_m * mem_n);
	cudaMallocHost(&vec_mem, sizeof(float) * vec_len);
	cudaMallocHost(&res_mem, sizeof(float) * vec_len);

	for (unsigned i = 0; i < vec_len; i++) {
		vec_mem[i] = i;
	}

	for (unsigned i = 0; i < mem_m * mem_n; i++) {
		mat_mem[i] = 0.f;
	}

	for (unsigned i = 0; i < vec_len; i++) {
		mat_mem[i] = vec_mem[i];
	}

	const auto layout = (std::is_same<nvcuda::wmma::col_major, Layout>::value) ? nvcuda::wmma::mem_col_major : nvcuda::wmma::mem_row_major;
	store_vector_acc_test_kernel<m, n, k, T><<<1, mtk::test_utils::warp_size>>>(mat_mem, mat_mem, layout);

	cudaDeviceSynchronize();

	float max_error = 0.0f;
	for (unsigned i = 0; i < vec_len; i++) {
		const auto diff = mat_mem[i] - vec_mem[i];
		max_error = std::max(max_error, std::abs(diff));
	}
	std::printf("Error  : %e\n", max_error);

	cudaFree(res_mem);
	cudaFree(vec_mem);
	cudaFree(mat_mem);
}

int main() {
	load_vector_test<nvcuda::wmma::matrix_a    , 16, 16, 16, half  , nvcuda::wmma::col_major>();
	load_vector_test<nvcuda::wmma::matrix_b    , 16, 16, 16, half  , nvcuda::wmma::col_major>();
	load_vector_test<nvcuda::wmma::accumulator , 16, 16, 16, half  , nvcuda::wmma::col_major>();
	load_vector_test<nvcuda::wmma::matrix_a    , 16, 16, 16, half  , nvcuda::wmma::row_major>();
	load_vector_test<nvcuda::wmma::matrix_b    , 16, 16, 16, half  , nvcuda::wmma::row_major>();
	load_vector_test<nvcuda::wmma::accumulator , 16, 16, 16, half  , nvcuda::wmma::row_major>();
	store_vector_test<nvcuda::wmma::accumulator, 16, 16, 16, half  , nvcuda::wmma::col_major>();
	store_vector_test<nvcuda::wmma::accumulator, 16, 16, 16, half  , nvcuda::wmma::row_major>();
	load_vector_test<nvcuda::wmma::matrix_a    , 16, 16, 16, float , nvcuda::wmma::col_major>();
	load_vector_test<nvcuda::wmma::matrix_b    , 16, 16, 16, float , nvcuda::wmma::col_major>();
	load_vector_test<nvcuda::wmma::accumulator , 16, 16, 16, float , nvcuda::wmma::col_major>();
	load_vector_test<nvcuda::wmma::matrix_a    , 16, 16, 16, float , nvcuda::wmma::row_major>();
	load_vector_test<nvcuda::wmma::matrix_b    , 16, 16, 16, float , nvcuda::wmma::row_major>();
	load_vector_test<nvcuda::wmma::accumulator , 16, 16, 16, float , nvcuda::wmma::row_major>();
	store_vector_test<nvcuda::wmma::accumulator, 16, 16, 16, float , nvcuda::wmma::col_major>();
	store_vector_test<nvcuda::wmma::accumulator, 16, 16, 16, float , nvcuda::wmma::row_major>();
	load_vector_test<nvcuda::wmma::matrix_a    , 16, 16, 16, double, nvcuda::wmma::col_major>();
	load_vector_test<nvcuda::wmma::matrix_b    , 16, 16, 16, double, nvcuda::wmma::col_major>();
	load_vector_test<nvcuda::wmma::accumulator , 16, 16, 16, double, nvcuda::wmma::col_major>();
	load_vector_test<nvcuda::wmma::matrix_a    , 16, 16, 16, double, nvcuda::wmma::row_major>();
	load_vector_test<nvcuda::wmma::matrix_b    , 16, 16, 16, double, nvcuda::wmma::row_major>();
	load_vector_test<nvcuda::wmma::accumulator , 16, 16, 16, double, nvcuda::wmma::row_major>();
	store_vector_test<nvcuda::wmma::accumulator, 16, 16, 16, double, nvcuda::wmma::col_major>();
	store_vector_test<nvcuda::wmma::accumulator, 16, 16, 16, double, nvcuda::wmma::row_major>();
}
