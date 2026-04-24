// dsa_topk_indexer_fp8_h64_d128_topk2048_ps64
// Torch binding. Uses torch::matmul and at::topk for scoring and selection.

#include <cstdint>
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

constexpr int INDEX_HEAD_DIM  = 128;
constexpr int PAGE_SIZE       = 64;
constexpr int PAGE_DATA_BYTES = 8192;
constexpr int PAGE_BYTES      = 8448;
constexpr int TOPK            = 2048;

__device__ __forceinline__ float fp8_e4m3_to_fp32(uint8_t byte) {
    const __nv_fp8_storage_t s  = static_cast<__nv_fp8_storage_t>(byte);
    const __half_raw         hr = __nv_cvt_fp8_to_halfraw(s, __NV_E4M3);
    return __half2float(__half(hr));
}

__global__ void dequant_gather_kernel(
    const uint8_t* __restrict__ k_cache,
    const int32_t* __restrict__ phys_idx,
    float*         __restrict__ K_deq,
    int num_pages_seq)
{
    const int page_id = blockIdx.x;
    const int tok     = blockIdx.y;
    const int lane    = threadIdx.x;
    const int64_t  phys = static_cast<int64_t>(phys_idx[page_id]);
    const uint8_t* pb   = k_cache + phys * PAGE_BYTES;
    const float scale = *reinterpret_cast<const float*>(pb + PAGE_DATA_BYTES + tok * 4);
    const int      k_base = lane * 4;
    const uint32_t packed = *reinterpret_cast<const uint32_t*>(
        pb + tok * INDEX_HEAD_DIM + k_base);
    const float f0 = fp8_e4m3_to_fp32(static_cast<uint8_t>((packed >>  0) & 0xff)) * scale;
    const float f1 = fp8_e4m3_to_fp32(static_cast<uint8_t>((packed >>  8) & 0xff)) * scale;
    const float f2 = fp8_e4m3_to_fp32(static_cast<uint8_t>((packed >> 16) & 0xff)) * scale;
    const float f3 = fp8_e4m3_to_fp32(static_cast<uint8_t>((packed >> 24) & 0xff)) * scale;
    const int out_row = page_id * PAGE_SIZE + tok;
    *reinterpret_cast<float4*>(&K_deq[out_row * INDEX_HEAD_DIM + k_base]) =
        make_float4(f0, f1, f2, f3);
}

__global__ void translate_indices_kernel(
    const int64_t* __restrict__ local_idx,
    const int32_t* __restrict__ phys_idx,
    int32_t*       __restrict__ out_row,
    int actual_topk)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= actual_topk) return;
    const int64_t lid  = local_idx[t];
    const int32_t phys = phys_idx[lid >> 6];
    out_row[t] = phys * PAGE_SIZE + static_cast<int32_t>(lid & 63);
}

void launch_topk_c(
    torch::Tensor q_index_fp8,
    torch::Tensor k_index_cache_fp8,
    torch::Tensor weights,
    torch::Tensor seq_lens,
    torch::Tensor block_table,
    torch::Tensor topk_indices)
{
    const at::cuda::CUDAGuard device_guard(q_index_fp8.device());
    const int B             = static_cast<int>(q_index_fp8.size(0));
    const int max_num_pages = static_cast<int>(block_table.size(1));

    topk_indices.fill_(-1);
    auto q_fp32_all = q_index_fp8.to(torch::kFloat32);
    auto k_cache_u8 = k_index_cache_fp8.view(torch::kUInt8);
    const uint8_t* k_cache_ptr = k_cache_u8.data_ptr<uint8_t>();
    auto sl_cpu = seq_lens.to(torch::kCPU);
    const int32_t* sl_ptr = sl_cpu.data_ptr<int32_t>();

    auto opts = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(q_index_fp8.device());
    auto K_deq_buf = torch::empty({max_num_pages * PAGE_SIZE, INDEX_HEAD_DIM}, opts);
    float* K_deq_base = K_deq_buf.data_ptr<float>();
    const int32_t* bt_base = block_table.data_ptr<int32_t>();
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    for (int b = 0; b < B; ++b) {
        const int sl = sl_ptr[b];
        if (sl <= 0) continue;
        const int num_pages_seq = (sl + PAGE_SIZE - 1) / PAGE_SIZE;
        const int actual_topk   = std::min(TOPK, sl);
        const int32_t* phys_ptr = bt_base + b * max_num_pages;

        dequant_gather_kernel<<<dim3(num_pages_seq, PAGE_SIZE), 32, 0, stream>>>(
            k_cache_ptr, phys_ptr, K_deq_base, num_pages_seq);
        AT_CUDA_CHECK(cudaGetLastError());

        auto K      = K_deq_buf.slice(0, 0, sl);
        auto scores = torch::matmul(q_fp32_all[b], K.transpose(0, 1));
        auto weighted = torch::relu(scores) * weights[b].unsqueeze(-1);
        auto result = at::topk(weighted.sum(0), actual_topk, 0, true, true);
        auto local_idx = std::get<1>(result).contiguous();

        translate_indices_kernel<<<(actual_topk + 255) / 256, 256, 0, stream>>>(
            local_idx.data_ptr<int64_t>(),
            phys_ptr,
            topk_indices[b].data_ptr<int32_t>(),
            actual_topk);
        AT_CUDA_CHECK(cudaGetLastError());
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_topk_c", &launch_topk_c, "DSA TopK FP8 indexer");
}
