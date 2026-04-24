// dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64
// Phase 1 naive torch-binding kernel.
// Mirrors the Python reference in the definition JSON directly via torch ops.
// Correctness-first; no tcgen05, TMA, clusters, or sm_100a-specific paths.

#include <cstdint>
#include <cmath>
#include <limits>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

constexpr int NUM_QO_HEADS = 16;
constexpr int HEAD_DIM_CKV = 512;
constexpr int HEAD_DIM_KPE = 64;
constexpr int PAGE_SIZE    = 64;
constexpr int TOPK         = 2048;

void launch_sparse_attention_c(
    torch::Tensor q_nope,          // [num_tokens, 16, 512] bf16
    torch::Tensor q_pe,            // [num_tokens, 16, 64]  bf16
    torch::Tensor ckv_cache,       // [num_pages, 64, 512]  bf16
    torch::Tensor kpe_cache,       // [num_pages, 64, 64]   bf16
    torch::Tensor sparse_indices,  // [num_tokens, 2048]    int32
    double sm_scale,               // Python float — harness passes scalars raw, not as 0-d tensors
    torch::Tensor output,          // [num_tokens, 16, 512] bf16  (DPS)
    torch::Tensor lse)             // [num_tokens, 16]      fp32  (DPS)
{
    const at::cuda::CUDAGuard device_guard(q_nope.device());
    const int num_tokens   = static_cast<int>(q_nope.size(0));
    const float sm_scale_f = static_cast<float>(sm_scale);
    const float inv_log2   = 1.0f / std::log(2.0f);

    // Flatten paged KV cache: [num_pages, page_size, dim] -> [num_pages * page_size, dim]
    // Cast to fp32 once for numerically-stable naive matmul + softmax.
    auto Kc_all = ckv_cache.reshape({-1, HEAD_DIM_CKV}).to(torch::kFloat32);
    auto Kp_all = kpe_cache.reshape({-1, HEAD_DIM_KPE}).to(torch::kFloat32);

    // Initialise DPS outputs: reference sets output[t]=0 and lse[t]=-inf when the
    // selected-index set for token t is empty.
    output.zero_();
    lse.fill_(-std::numeric_limits<float>::infinity());

    for (int t = 0; t < num_tokens; ++t) {
        auto indices       = sparse_indices[t];             // [topk] int32
        auto valid_mask    = indices.ne(-1);
        auto valid_indices = indices.masked_select(valid_mask);
        if (valid_indices.numel() == 0) continue;

        auto tok_idx = valid_indices.to(torch::kLong);      // index_select needs int64

        auto Kc = Kc_all.index_select(0, tok_idx);          // [num_valid, 512]
        auto Kp = Kp_all.index_select(0, tok_idx);          // [num_valid, 64]
        auto qn = q_nope[t].to(torch::kFloat32);            // [16, 512]
        auto qp = q_pe[t].to(torch::kFloat32);              // [16, 64]

        auto logits = torch::matmul(qn, Kc.transpose(0, 1))
                    + torch::matmul(qp, Kp.transpose(0, 1)); // [16, num_valid]
        auto logits_scaled = logits * sm_scale_f;

        // 2-base LSE, written into the DPS output row for token t.
        // Must use copy_ — in C++ libtorch, `lse[t] = rhs` rebinds the local
        // view handle instead of writing to storage (silent no-op).
        lse[t].copy_(torch::logsumexp(logits_scaled, -1) * inv_log2);

        auto attn = torch::softmax(logits_scaled, -1);       // [16, num_valid]
        auto out  = torch::matmul(attn, Kc);                 // [16, 512]
        output[t].copy_(out.to(torch::kBFloat16));
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_sparse_attention_c", &launch_sparse_attention_c,
          "DSA Sparse Attention (Phase 1 naive torch-op based)");
}
