#include <iostream>
#include <random>
#include <wmma_extension/mma_simt.hpp>
#include "utils.hpp"

namespace {
template <class T>
constexpr double error_threshold = 0.0;
template <>
constexpr double error_threshold<half > = 1e-3;
template <>
constexpr double error_threshold<float> = 1e-5;
} // noname namespace

template <unsigned N, class T, class A_Layout, class B_Layout>
__global__ void mma_kernel_abcd(float* const d_ptr, const float* const a_ptr, const float* const b_ptr, const float* const c_ptr, const nvcuda::wmma::layout_t cd_layout) {
	constexpr unsigned LD = N;
	__shared__ float smem[N * LD];
	mtk::test_utils::fill_zero(smem, N * LD);

	mtk::wmma::mma_simt::fragment<nvcuda::wmma::matrix_a   , N, N, N, T, A_Layout> frag_a;
	mtk::wmma::mma_simt::fragment<nvcuda::wmma::matrix_b   , N, N, N, T, B_Layout> frag_b;
	mtk::wmma::mma_simt::fragment<nvcuda::wmma::accumulator, N, N, N, T, void    > frag_c, frag_d;
	// Load A
	mtk::test_utils::copy_matrix(smem, LD, a_ptr, N, N, N);
	mtk::wmma::mma_simt::load_matrix_sync(frag_a, smem, LD);

	// Load B
	mtk::test_utils::copy_matrix(smem, LD, b_ptr, N, N, N);
	mtk::wmma::mma_simt::load_matrix_sync(frag_b, smem, LD);

	// Load C
	mtk::test_utils::copy_matrix(smem, LD, c_ptr, N, N, N);
	mtk::wmma::mma_simt::load_matrix_sync(frag_c, smem, LD, cd_layout);

	// Fill D
	mtk::wmma::mma_simt::fill_fragment(frag_d, 0.0f);

	// mma
	mtk::wmma::mma_simt::mma_sync(frag_d, frag_a, frag_b, frag_c);

	// Store D
	mtk::wmma::mma_simt::store_matrix_sync(smem, frag_d, LD, cd_layout);
	mtk::test_utils::copy_matrix(d_ptr, N, smem, LD, N, N);

	// Test for fill_zero
	mtk::wmma::mma_simt::fill_zero(frag_d);
}

template <unsigned N, class T, class A_Layout, class B_Layout>
void test_mma(const nvcuda::wmma::layout_t cd_layout) {
	float *hA, *hB, *hC, *hD;
	cudaMallocHost(&hA, N * N * sizeof(float));
	cudaMallocHost(&hB, N * N * sizeof(float));
	cudaMallocHost(&hC, N * N * sizeof(float));
	cudaMallocHost(&hD, N * N * sizeof(float));

	std::mt19937 mt(std::random_device{}());
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

	for (unsigned i = 0; i < N * N; i++) {
			hA[i] = dist(mt);
			hB[i] = dist(mt);
			hC[i] = dist(mt);
	}
	cudaDeviceSynchronize();

	mma_kernel_abcd<N, T, A_Layout, B_Layout><<<1, mtk::test_utils::warp_size>>>(hD, hA, hB, hC, cd_layout);

	const auto stat = cudaDeviceSynchronize();
	if (stat != cudaSuccess) {
		std::printf("[error] %s\n", cudaGetErrorString(stat));
	}

	double max_error = 0.;
	for (unsigned m = 0; m < N; m++) {
		for (unsigned n = 0; n < N; n++) {
			double cor_d = 0.;
			for (unsigned k = 0; k < N; k++) {
				const auto a_mem_index = std::is_same<A_Layout, nvcuda::wmma::col_major>::value ? (k * N + m) : (m * N + k);
				const auto b_mem_index = std::is_same<B_Layout, nvcuda::wmma::col_major>::value ? (k + n * N) : (n + k * N);
				cor_d += static_cast<double>(hA[a_mem_index]) * static_cast<double>(hB[b_mem_index]);
			}
			const auto c_mem_index = (cd_layout == nvcuda::wmma::mem_col_major) ? (m + n * N) : (n + m * N);
			cor_d += hC[c_mem_index];

			max_error = std::max(max_error, std::abs(cor_d - hD[c_mem_index]));
		}
	}

	std::printf(
			"[Type:%5s, N:%3u, A_Layout:%10s, B_Layout:%10s, C_Layout:%10s, FragShape<%2d,%2d,%2d>] max_error: %e (%6s)\n",
			mtk::test_utils::to_string<T>().c_str(),
			N,
			mtk::test_utils::to_string<A_Layout>().c_str(),
			mtk::test_utils::to_string<B_Layout>().c_str(),
			(cd_layout == nvcuda::wmma::mem_col_major) ? mtk::test_utils::to_string<nvcuda::wmma::col_major>().c_str() : mtk::test_utils::to_string<nvcuda::wmma::row_major>().c_str(),
			N, N, N,
			max_error,
			(max_error < error_threshold<T> ? "PASSED" : "FAILED")
			);

	cudaFreeHost(hA);
	cudaFreeHost(hB);
	cudaFreeHost(hC);
	cudaFreeHost(hD);
}

int main() {
	// wmma FP16 test
	test_mma<16, half , nvcuda::wmma::col_major, nvcuda::wmma::col_major>(nvcuda::wmma::mem_col_major);
	test_mma<16, half , nvcuda::wmma::col_major, nvcuda::wmma::col_major>(nvcuda::wmma::mem_col_major);
	test_mma<16, half , nvcuda::wmma::row_major, nvcuda::wmma::col_major>(nvcuda::wmma::mem_col_major);
	test_mma<16, half , nvcuda::wmma::row_major, nvcuda::wmma::col_major>(nvcuda::wmma::mem_col_major);
	test_mma<16, half , nvcuda::wmma::col_major, nvcuda::wmma::row_major>(nvcuda::wmma::mem_col_major);
	test_mma<16, half , nvcuda::wmma::col_major, nvcuda::wmma::row_major>(nvcuda::wmma::mem_col_major);
	test_mma<16, half , nvcuda::wmma::row_major, nvcuda::wmma::row_major>(nvcuda::wmma::mem_col_major);
	test_mma<16, half , nvcuda::wmma::row_major, nvcuda::wmma::row_major>(nvcuda::wmma::mem_col_major);
	test_mma<16, half , nvcuda::wmma::col_major, nvcuda::wmma::col_major>(nvcuda::wmma::mem_row_major);
	test_mma<16, half , nvcuda::wmma::col_major, nvcuda::wmma::col_major>(nvcuda::wmma::mem_row_major);
	test_mma<16, half , nvcuda::wmma::row_major, nvcuda::wmma::col_major>(nvcuda::wmma::mem_row_major);
	test_mma<16, half , nvcuda::wmma::row_major, nvcuda::wmma::col_major>(nvcuda::wmma::mem_row_major);
	test_mma<16, half , nvcuda::wmma::col_major, nvcuda::wmma::row_major>(nvcuda::wmma::mem_row_major);
	test_mma<16, half , nvcuda::wmma::col_major, nvcuda::wmma::row_major>(nvcuda::wmma::mem_row_major);
	test_mma<16, half , nvcuda::wmma::row_major, nvcuda::wmma::row_major>(nvcuda::wmma::mem_row_major);
	test_mma<16, half , nvcuda::wmma::row_major, nvcuda::wmma::row_major>(nvcuda::wmma::mem_row_major);
	test_mma<16, half , nvcuda::wmma::col_major, nvcuda::wmma::col_major>(nvcuda::wmma::mem_col_major);
	test_mma<16, half , nvcuda::wmma::col_major, nvcuda::wmma::col_major>(nvcuda::wmma::mem_col_major);
	test_mma<16, float, nvcuda::wmma::row_major, nvcuda::wmma::col_major>(nvcuda::wmma::mem_col_major);
	test_mma<16, float, nvcuda::wmma::row_major, nvcuda::wmma::col_major>(nvcuda::wmma::mem_col_major);
	test_mma<16, float, nvcuda::wmma::col_major, nvcuda::wmma::row_major>(nvcuda::wmma::mem_col_major);
	test_mma<16, float, nvcuda::wmma::col_major, nvcuda::wmma::row_major>(nvcuda::wmma::mem_col_major);
	test_mma<16, float, nvcuda::wmma::row_major, nvcuda::wmma::row_major>(nvcuda::wmma::mem_col_major);
	test_mma<16, float, nvcuda::wmma::row_major, nvcuda::wmma::row_major>(nvcuda::wmma::mem_col_major);
	test_mma<16, float, nvcuda::wmma::col_major, nvcuda::wmma::col_major>(nvcuda::wmma::mem_row_major);
	test_mma<16, float, nvcuda::wmma::col_major, nvcuda::wmma::col_major>(nvcuda::wmma::mem_row_major);
	test_mma<16, float, nvcuda::wmma::row_major, nvcuda::wmma::col_major>(nvcuda::wmma::mem_row_major);
	test_mma<16, float, nvcuda::wmma::row_major, nvcuda::wmma::col_major>(nvcuda::wmma::mem_row_major);
	test_mma<16, float, nvcuda::wmma::col_major, nvcuda::wmma::row_major>(nvcuda::wmma::mem_row_major);
	test_mma<16, float, nvcuda::wmma::col_major, nvcuda::wmma::row_major>(nvcuda::wmma::mem_row_major);
	test_mma<16, float, nvcuda::wmma::row_major, nvcuda::wmma::row_major>(nvcuda::wmma::mem_row_major);
	test_mma<16, float, nvcuda::wmma::row_major, nvcuda::wmma::row_major>(nvcuda::wmma::mem_row_major);
}
