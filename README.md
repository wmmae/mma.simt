# MMA emulator on SIMT Core

A library for computing gemm like WMMA API

## Requirements

- CUDA (9.2 or higher)
- C++  (C++11 or later)

## Sample code
```cuda
__global__ void mma_kernel_abcd(float* const d_ptr, const float* const a_ptr, const float* const b_ptr, const float* const c_ptr) {
    constexpr unsigned LD = N;
    __shared__ float smem[N * LD];
    //mtk::test_utils::fill_zero(smem, N * LD);

    mtk::wmma::mma_simt::fragment<nvcuda::wmma::matrix_a   , 16, 16, 16, float, nvcuda::wmma::col_major> frag_a;
    mtk::wmma::mma_simt::fragment<nvcuda::wmma::matrix_b   , 16, 16, 16, float, nvcuda::wmma::col_major> frag_b;
    mtk::wmma::mma_simt::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float, void    > frag_c, frag_d;
    // Load A
    //mtk::test_utils::copy_matrix(smem, LD, a_ptr, N, N, N);
    mtk::wmma::mma_simt::load_matrix_sync(frag_a, smem, LD);

    // Load B
    //mtk::test_utils::copy_matrix(smem, LD, b_ptr, N, N, N);
    mtk::wmma::mma_simt::load_matrix_sync(frag_b, smem, LD);

    // Load C
    //mtk::test_utils::copy_matrix(smem, LD, c_ptr, N, N, N);
    mtk::wmma::mma_simt::load_matrix_sync(frag_c, smem, LD, nvcuda::wmma::mem_col_major);

    // Fill D
    mtk::wmma::mma_simt::fill_fragment(frag_d, 0.0f);

    // mma
    mtk::wmma::mma_simt::mma_sync(frag_d, frag_a, frag_b, frag_c);

    // Store D
    mtk::wmma::mma_simt::store_matrix_sync(smem, frag_d, LD, nvcuda::wmma::mem_col_major);
    //mtk::test_utils::copy_matrix(d_ptr, N, smem, LD, N, N);
}
```

## Supported fragment

| fm | fn | fk | LayoutA | LayoutB | Type             |
| -- | -- | -- | ------- | ------- | ---------------- |
| 16 | 16 | 16 | col/row | col/low | double/float/half|


## Functions
- `mtk::wmma::fill_fragment`
- `mtk::wmma::load_matrix_sync`
- `mtk::wmma::store_matrix_sync`
- `mtk::wmma::mma_sync`

- `mtk::wmma::fill_zero`


## License
MIT
