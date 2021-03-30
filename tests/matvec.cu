#include <iostream>
#include <random>
#include <wmma_extension/mma_simt.hpp>
#include "utils.hpp"

template <class T>
constexpr double error_threshold = 0.0;
template <>
constexpr double error_threshold<half  > = 1e-3;
template <>
constexpr double error_threshold<float > = 1e-5;
template <>
constexpr double error_threshold<double> = 1e-15;

template <class T>
struct smem_t {using type = float;};
template <>
struct smem_t<double> {using type = double;};

template <unsigned N, class T, class MEM_T>
__global__ void matvec_kernel(MEM_T* const y_ptr, const MEM_T* const a_ptr, const MEM_T* const x_ptr) {
	__shared__ MEM_T smem[N * N];
	mtk::test_utils::fill_zero(smem, N * N);

	mtk::wmma::mma_simt::fragment<nvcuda::wmma::matrix_a   , N, N, N, T, nvcuda::wmma::col_major> frag_a;
	mtk::wmma::mma_simt::fragment<nvcuda::wmma::matrix_b   , N, N, N, T, nvcuda::wmma::col_major> frag_x;
	mtk::wmma::mma_simt::fragment<nvcuda::wmma::accumulator, N, N, N, T, void                   > frag_y, frag_z;
	// Load A
	mtk::test_utils::copy_matrix(smem, N, a_ptr, N, N, N);
	mtk::wmma::mma_simt::load_matrix_sync(frag_a, smem, N);

	// Load X
	mtk::test_utils::copy_matrix(smem, N, x_ptr, N, N, 1);
	mtk::wmma::mma_simt::fill_zero(frag_x);
	mtk::wmma::mma_simt::load_vector(frag_x, smem);

	// Init zero
	mtk::wmma::mma_simt::fill_zero(frag_z);

	// mma
	mtk::wmma::mma_simt::mma_sync(frag_y, frag_a, frag_x, frag_z);

	// Store D
	mtk::wmma::mma_simt::store_vector(smem, frag_y, nvcuda::wmma::mem_col_major);
	mtk::test_utils::copy_matrix(y_ptr, N, smem, N, N, 1);
}

template <unsigned N, class T>
void test_matvec() {
	using mem_t = typename smem_t<T>::type;
	mem_t *hX, *hY, *hA;
	cudaMallocHost(&hX, N     * sizeof(mem_t));
	cudaMallocHost(&hY, N     * sizeof(mem_t));
	cudaMallocHost(&hA, N * N * sizeof(mem_t));

	std::mt19937 mt(std::random_device{}());
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

	for (unsigned i = 0; i < N * N; i++) {
			hA[i] = dist(mt);
	}
	for (unsigned i = 0; i < N; i++) {
			hX[i] = dist(mt);
	}
	cudaDeviceSynchronize();

	matvec_kernel<N, T, mem_t><<<1, mtk::test_utils::warp_size>>>(hY, hA, hX);

	cudaDeviceSynchronize();

	double max_error = 0.;
	for (unsigned n = 0; n < N; n++) {
		double cor_d = 0.;
		for (unsigned k = 0; k < N; k++) {
			cor_d += static_cast<double>(hA[k * N + n]) * static_cast<double>(hX[k]);
		}

		max_error = std::max(max_error, std::abs(cor_d - hY[n]));
	}

	std::printf(
			"[Type:%6s, N:%3u] max_error: %e (%6s)\n",
			mtk::test_utils::to_string<T>().c_str(),
			N,
			max_error,
			(max_error < (error_threshold<T> * N) ? "PASSED" : "FAILED")
			);

	cudaFreeHost(hA);
	cudaFreeHost(hX);
	cudaFreeHost(hY);
}

int main() {
	// wmma FP16 test
	test_matvec<16, half  >();
	test_matvec<16, float >();
	test_matvec<16, double>();
}
