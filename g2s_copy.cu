#include <iostream>
#include <cuda.h>
#include <cute/tensor.hpp>

static constexpr int row = 512;
static constexpr int col = 512;


template<typename T>
__global__ void initBuffer(T* ptr, int size) {
    int thr_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thr_idx < size) {
        ptr[thr_idx] = static_cast<T>(thr_idx);
    }
}


template<typename T, typename TiledCopy_>
__global__ void g2rCopy(T* ptr) {
    using namespace cute;
    using G2SCopy = TiledCopy_;

    Tensor A = make_tensor(make_gmem_ptr((T *)ptr),
                    make_shape(Int<row>{}, Int<col>{}),
                    make_stride(Int<col>{}, Int<1>{}));

    int idx = threadIdx.x;

    G2SCopy g2s_tiled_copy;
    auto g2s_thr_tiled_copy = g2s_tiled_copy.get_slice(idx);
    auto thr_g = g2s_thr_tiled_copy.partition_S(A);
    auto fragment = make_fragment_like(thr_g);
    // Just copy once
    copy(g2s_tiled_copy, thr_g(_, 0, 0), fragment(_, 0, 0));
#ifdef DEBUG
    printf("Thread ID: %d hold A value %.1f, %.1f, %.1f, %.1f\n",
        idx, float(fragment(0)), float(fragment(1)), float(fragment(2)), float(fragment(3)));
#endif
}


template<typename T, typename TiledCopy_, typename ShmLayout_>
__global__ void g2sCopy(T* ptr) {
    using namespace cute;
    using G2SCopy = TiledCopy_;
    int idx = threadIdx.x;

    extern __shared__ T shm_data[];

    Tensor gA = make_tensor(make_gmem_ptr((T *)ptr),
                make_shape(Int<row>{}, Int<col>{}),
                make_stride(Int<col>{}, Int<1>{})); // (512, 512):(512, 1)

    Tensor sA = make_tensor(make_smem_ptr((T *)shm_data), ShmLayout_{});    // (128, 64)

    G2SCopy g2s_tiled_copy;
    auto g2s_thr_tiled_copy = g2s_tiled_copy.get_slice(idx);
    auto thr_g = g2s_thr_tiled_copy.partition_S(gA);    // ((16, 64), 32, 8)
    auto thr_s = g2s_thr_tiled_copy.partition_D(sA);    // ((16, 64), 8, 1)

    // Just copy once
    copy(g2s_tiled_copy, thr_g(_, 0, 0), thr_s(_, 0, 0));
    cp_async_wait<0>();
    __syncthreads();
#ifdef DEBUG
    if (thread0()) {
        print(thr_g);
        printf("\n");
        print(thr_s);
        printf("\n");
    }
#endif
}


template<typename T, typename TiledCopy_>
__global__ void g2sCopyComplex(T* ptr) {
    using namespace cute;
    using G2SCopy = TiledCopy_;
    int idx = threadIdx.x;

    extern __shared__ T shm_data[];

    Tensor gA = make_tensor(make_gmem_ptr((T *)ptr),
        make_shape(make_shape(Int<2>{}, Int<256>{}), make_shape(Int<4>{}, Int<128>{})),
        make_stride(make_stride(Int<1>{}, Int<8>{}), make_stride(Int<2>{}, Int<2048>{}))
    );
    Tensor sA = make_tensor(make_smem_ptr((T *)shm_data),
        make_shape(make_shape(Int<2>{}, Int<128>{}), make_shape(Int<4>{}, Int<16>{})),
        make_stride(make_stride(Int<1>{}, Int<8>{}), make_stride(Int<2>{}, Int<1024>{}))
    );

#ifdef DEBUG
    if (thread0()) {
        print(gA);
        printf("\n");
        print(sA);
        printf("\n");
    }
#endif

    G2SCopy g2s_tiled_copy;
    auto g2s_thr_tiled_copy = g2s_tiled_copy.get_slice(idx);
    auto thr_g = g2s_thr_tiled_copy.partition_S(gA);    // ((16, 64), 32, 8)
    auto thr_s = g2s_thr_tiled_copy.partition_D(sA);    // ((16, 64), 8, 1)

    // Just copy once
    copy(g2s_tiled_copy, thr_g(_, 0, 0), thr_s(_, 0, 0));
    cp_async_wait<0>();
    __syncthreads();
#ifdef DEBUG
    if (thread0()) {
        print(thr_g);
        printf("\n");
        print(thr_s);
        printf("\n");
    }
#endif
}


template<typename T, typename TiledCopy_>
__global__ void g2sCopyComplexEx(T* ptr) {
    using namespace cute;
    using G2SCopy = TiledCopy_;
    int idx = threadIdx.x;

    extern __shared__ T shm_data[];

    Tensor gA = make_tensor(make_gmem_ptr((T *)ptr),
        make_shape(make_shape(Int<2>{}, Int<256>{}), make_shape(Int<4>{}, Int<128>{})),
        make_stride(make_stride(Int<4>{}, Int<8>{}), make_stride(Int<1>{}, Int<2048>{}))
    );
    Tensor sA = make_tensor(make_smem_ptr((T *)shm_data),
        make_shape(make_shape(Int<2>{}, Int<128>{}), make_shape(Int<4>{}, Int<16>{})),
        make_stride(make_stride(Int<4>{}, Int<8>{}), make_stride(Int<1>{}, Int<1024>{}))
    );

#ifdef DEBUG
    if (thread0()) {
        print(gA);
        printf("\n");
        print(sA);
        printf("\n");
    }
#endif

    G2SCopy g2s_tiled_copy;
    auto g2s_thr_tiled_copy = g2s_tiled_copy.get_slice(idx);
    auto thr_g = g2s_thr_tiled_copy.partition_S(gA);    // ((16, 64), 32, 8)
    auto thr_s = g2s_thr_tiled_copy.partition_D(sA);    // ((16, 64), 8, 1)

    // Just copy once
    copy(g2s_tiled_copy, thr_g(_, 0, 0), thr_s(_, 0, 0));
    cp_async_wait<0>();
    __syncthreads();
#ifdef DEBUG
    if (thread0()) {
        print(thr_g);
        printf("\n");
        print(thr_s);
        printf("\n");
    }
#endif
}


template<typename T>
int initBufferKernelLaunch(T** dev_p) {
    // init buffer
    cudaMalloc((void**)dev_p, row * col * sizeof(T));
    dim3 grid(512, 1, 1);
    dim3 block(512, 1, 1);
    initBuffer<T>
        <<<grid, block>>>(*dev_p, row * col);
    cudaDeviceSynchronize();
    auto err = cudaGetLastError();
    printf("Init done, Error Code: %d, State: %s\n", err, cudaGetErrorString(err));
    return 0;
}


template<typename T>
int g2rCopyKernelLaunch(T* ptr) {
    using namespace cute;
    using g2r_copy_op = UniversalCopy<T>;
    using g2r_copy_traits = Copy_Traits<g2r_copy_op>;
    using g2r_copy_atom = Copy_Atom<g2r_copy_traits, T>;

    using G2RCopyA = decltype(make_tiled_copy(g2r_copy_atom{},
                        Layout<Shape<_16, _8>, Stride<_8, _1>>{},
                        Layout<Shape<_1, _16>>{}));
    dim3 grid(1, 1, 1);
    dim3 block(128, 1, 1);
    
    g2rCopy<T, G2RCopyA><<<grid, block>>>(ptr);

    cudaDeviceSynchronize();
    auto err = cudaGetLastError();
    printf("Copy done, Error Code: %d, State: %s\n", err, cudaGetErrorString(err));
#ifdef DEBUG
    print(G2RCopyA{});
    // using tiled_layout_tv = typename G2RCopyA::TiledLayout_TV;
    // print_layout(tiled_layout_tv{});
#endif
    return 0;
}


template<typename T>
int g2sCopyKernelLaunch(T* ptr) {
    using namespace cute;
    using g2s_copy_op = SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>;
    using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
    using g2s_copy_atom = Copy_Atom<g2s_copy_traits, T>;

    using G2SCopyA = decltype(make_tiled_copy(g2s_copy_atom{},
                        make_layout(make_shape(Int<16>{}, Int<8>{}),
                                    make_stride(Int<8>{}, Int<1>{})),
                        make_layout(make_shape(Int<1>{}, Int<8>{}))));  // Copy Tile: (16, 64)
    
    using SmemLayoutAtom = decltype(composition(Swizzle<3, 3, 3>{},
                            make_layout(make_shape(Int<8>{}, Int<64>{}),
                                        make_stride(Int<64>{}, Int<1>{}))));  // Atom Shape: (8, 64):(64, 1) swizzle(3, 3, 3)
    using SmemLayoutA = decltype(tile_to_shape(SmemLayoutAtom{}, make_shape(Int<128>{}, Int<64>{})));   // (128, 64)

    dim3 grid(1, 1, 1);
    dim3 block(128, 1, 1);

    constexpr int shm_size = cosize(SmemLayoutA{}) * sizeof(T);
    cudaFuncSetAttribute(g2sCopy<T, G2SCopyA, SmemLayoutA>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);

    g2sCopy<T, G2SCopyA, SmemLayoutA><<<grid, block, shm_size>>>(ptr);

    cudaDeviceSynchronize();
    auto err = cudaGetLastError();
    printf("Copy done, Error Code: %d, State: %s\n", err, cudaGetErrorString(err));
#ifdef DEBUG
    print(G2SCopyA{});
#endif
    return 0;
}


template<typename T>
int g2sCopyComplexKernelLaunch(T* ptr) {
    using namespace cute;
    using g2s_copy_op = SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>;
    using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
    using g2s_copy_atom = Copy_Atom<g2s_copy_traits, T>;

    using G2SCopyA = decltype(make_tiled_copy(g2s_copy_atom{},
                        make_layout(make_shape(Int<16>{}, Int<8>{}),
                                    make_stride(Int<8>{}, Int<1>{})),
                        make_layout(make_shape(Int<2>{}, Int<4>{}))));  // Copy Tile: (32, 32)

    dim3 grid(1, 1, 1);
    dim3 block(128, 1, 1);

    constexpr int shm_size = 128 * 64 * sizeof(T);
    cudaFuncSetAttribute(g2sCopyComplex<T, G2SCopyA>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);

    g2sCopyComplex<T, G2SCopyA><<<grid, block, shm_size>>>(ptr);

    cudaDeviceSynchronize();
    auto err = cudaGetLastError();
    printf("Copy done, Error Code: %d, State: %s\n", err, cudaGetErrorString(err));
#ifdef DEBUG
    print(G2SCopyA{});
#endif
    return 0;
}


template<typename T>
int g2sCopyComplexKernelLaunchEx(T* ptr) {
    using namespace cute;
    using g2s_copy_op = SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>;
    using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
    using g2s_copy_atom = Copy_Atom<g2s_copy_traits, T>;

    using G2SCopyA = decltype(make_tiled_copy(g2s_copy_atom{},
                        make_layout(make_shape(Int<16>{}, Int<8>{}),
                                    make_stride(Int<8>{}, Int<1>{})),
                        make_layout(make_shape(Int<2>{}, Int<4>{}))));  // Copy Tile: (32, 32)

    dim3 grid(1, 1, 1);
    dim3 block(128, 1, 1);

    constexpr int shm_size = 128 * 64 * sizeof(T);
    cudaFuncSetAttribute(g2sCopyComplex<T, G2SCopyA>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);

    g2sCopyComplexEx<T, G2SCopyA><<<grid, block, shm_size>>>(ptr);

    cudaDeviceSynchronize();
    auto err = cudaGetLastError();
    printf("Copy done, Error Code: %d, State: %s\n", err, cudaGetErrorString(err));
#ifdef DEBUG
    print(G2SCopyA{});
#endif
    return 0;
}


int main(void) {
    using namespace cute;
    half_t* dev_p = nullptr;
    initBufferKernelLaunch<half_t>(&dev_p);
    if (!dev_p) {
        printf("Init done, but ptr is null.");
        exit(-1);
    }

    int ret = g2rCopyKernelLaunch(dev_p);

    if (ret != 0) {
        printf("g2s copy error.");
        exit(-1);
    }

    if (dev_p) {
        cudaFree(dev_p);
    }
    return 0;
}

