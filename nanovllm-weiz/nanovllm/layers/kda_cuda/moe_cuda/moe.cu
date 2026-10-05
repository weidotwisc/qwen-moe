// BF16 fused MoE for Qwen3-30B-A3B (EP local kernel), A100 / sm80.
//
// moe_forward(routing_logits[T,128] bf16, hidden_states[T,2048] bf16,
//             W13[LE,1536,2048] bf16, W2[LE,2048,768] bf16, local_expert_offset)
//   -> output[T,2048] bf16   (this rank's partial contribution; sum over ranks = full MoE)
//
// Pipeline (single CUDA stream):
//   1. routing: fp32 softmax(128) -> top-8 -> renormalize (ATen)
//   2. select local (token,slot) pairs, sort by local expert, gather A_perm
//   3. GEMM1 (per-expert cuBLAS bf16, fp32 accum): A_perm @ W13^T -> G1[.,1536]
//   4. fused SiLU-gate: silu(G1[:, :768]) * G1[:, 768:] -> Inter[.,768]
//   5. GEMM2 (per-expert cuBLAS): Inter @ W2^T -> G2[.,2048]
//   6. weighted scatter-add (fp32 accum) -> output
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <vector>

#define CUBLAS_CHECK(x) do { cublasStatus_t st_ = (x); \
  TORCH_CHECK(st_ == CUBLAS_STATUS_SUCCESS, "cuBLAS error ", (int)st_, " (", cublasGetStatusString(st_), \
              ") pending-cuda=", cudaGetErrorString(cudaGetLastError())); } while (0)

constexpr int H = 2048;
constexpr int I = 768;
constexpr int G1N = 1536;  // 2*I

using bf16 = __nv_bfloat16;

// Inter[n,768] = silu(G1[:, :768]) * G1[:, 768:]  (gate-first SwiGLU)
__global__ void silu_gate_kernel(const bf16* __restrict__ g1, bf16* __restrict__ inter, long n) {
  long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
  long total = n * I;
  if (idx >= total) return;
  long row = idx / I, col = idx % I;
  float gate = __bfloat162float(g1[row * G1N + col]);
  float up = __bfloat162float(g1[row * G1N + I + col]);
  float s = gate / (1.0f + __expf(-gate));  // silu
  inter[idx] = __float2bfloat16(s * up);
}

// output[tok] += w * G2[row]   (fp32 accumulator, atomic since a token may hit several local experts)
__global__ void scatter_weighted_kernel(const bf16* __restrict__ g2,
                                        const long* __restrict__ sorted_tok,
                                        const float* __restrict__ sorted_w,
                                        float* __restrict__ out, long n) {
  long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
  long total = n * H;
  if (idx >= total) return;
  long row = idx / H, col = idx % H;
  long tok = sorted_tok[row];
  float v = sorted_w[row] * __bfloat162float(g2[row * H + col]);
  atomicAdd(&out[tok * H + col], v);
}

static inline bf16* bptr(torch::Tensor& t) {
  return reinterpret_cast<bf16*>(t.data_ptr<at::BFloat16>());
}

// C[M,N] (row-major) = A[M,K] @ B[N,K]^T, all bf16, fp32 accumulate.
// col-major cuBLAS: Cc[N,M] = Bc^T @ Ac with Bc(K,N), Ac(K,M).
static void gemm_rowmajor_ABt(cublasHandle_t h, int M, int N, int K,
                              const bf16* Bmat /*[N,K] row-major*/, const bf16* Amat /*[M,K] row-major*/,
                              bf16* Cmat /*[M,N] row-major*/) {
  const float alpha = 1.0f, beta = 0.0f;
  CUBLAS_CHECK(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha,
                            Bmat, CUDA_R_16BF, K,   // op(A)=Bmat^T -> (N x K), lda=K
                            Amat, CUDA_R_16BF, K,   // op(B)=Amat   -> (K x M), ldb=K
                            &beta, Cmat, CUDA_R_16BF, N,  // C (N x M) col-major = [M,N] row-major
                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

torch::Tensor moe_forward(torch::Tensor routing_logits, torch::Tensor hidden_states,
                          torch::Tensor W13, torch::Tensor W2, int64_t local_expert_offset) {
  TORCH_CHECK(hidden_states.is_cuda() && hidden_states.dtype() == torch::kBFloat16, "hidden bf16 cuda");
  TORCH_CHECK(W13.dtype() == torch::kBFloat16 && W2.dtype() == torch::kBFloat16, "weights bf16");
  const int T = hidden_states.size(0);
  const int LE = W13.size(0);
  TORCH_CHECK(hidden_states.size(1) == H && W13.size(1) == G1N && W13.size(2) == H, "W13 shape");
  TORCH_CHECK(W2.size(1) == H && W2.size(2) == I, "W2 shape");

  auto stream = at::cuda::getCurrentCUDAStream();
  auto opt_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(hidden_states.device());
  auto out = torch::zeros({T, H}, opt_f32);  // fp32 accumulator

  // --- routing (fp32 softmax over all E -> top-8 -> renormalize) ---
  auto probs = torch::softmax(routing_logits.to(torch::kFloat32), -1);
  auto topk = torch::topk(probs, 8, -1);
  auto val = std::get<0>(topk);
  auto idx = std::get<1>(topk);                       // [T,8] int64
  val = val / val.sum(-1, true);                      // renormalize
  auto flat_e = idx.reshape(-1);                      // [T*8]
  auto flat_w = val.reshape(-1).to(torch::kFloat32);
  auto flat_tok = torch::arange(T, idx.options()).repeat_interleave(8);

  auto local = (flat_e >= local_expert_offset) & (flat_e < local_expert_offset + LE);
  auto sub_e = (flat_e - local_expert_offset).masked_select(local);   // [n] in [0,LE)
  auto tok = flat_tok.masked_select(local);
  auto wv = flat_w.masked_select(local);
  const long n = sub_e.size(0);
  if (n == 0) return out.to(torch::kBFloat16);

  // group rows by local expert (order within a group is irrelevant to the summed result)
  auto sorted = sub_e.sort(0);
  auto sorted_e = std::get<0>(sorted);
  auto order = std::get<1>(sorted);
  auto sorted_tok = tok.index_select(0, order).contiguous();
  auto sorted_w = wv.index_select(0, order).contiguous();
  auto counts = torch::bincount(sub_e, {}, LE);       // [LE]
  auto offsets_cpu = counts.cumsum(0).to(torch::kCPU).to(torch::kLong);  // exclusive prefix via shift below
  auto counts_cpu = counts.to(torch::kCPU).to(torch::kLong);
  const long* off_end = offsets_cpu.data_ptr<long>();
  const long* cnt = counts_cpu.data_ptr<long>();

  auto A_perm = hidden_states.index_select(0, sorted_tok).contiguous();  // [n,2048] bf16
  auto G1 = torch::empty({n, G1N}, hidden_states.options());
  auto Inter = torch::empty({n, I}, hidden_states.options());
  auto G2 = torch::empty({n, H}, hidden_states.options());

  // NOTE: we create our own cuBLAS handle instead of ATen's. The extension links CUDA 12.4's
  // libcublas.so.12, while torch/ATen loads libcublas.so.13; an ATen handle (v13) passed to our
  // v12 cublasGemmEx returns NOT_INITIALIZED. Own-handle keeps create/call in the same library.
  static cublasHandle_t handle = nullptr;
  if (handle == nullptr) CUBLAS_CHECK(cublasCreate(&handle));
  CUBLAS_CHECK(cublasSetStream(handle, stream));
  CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "sticky cuda before GEMM1");

  // GEMM1 per expert: G1_e[M,1536] = A_perm_e[M,2048] @ W13[e]^T
  bf16* pA = bptr(A_perm);
  bf16* pW13 = bptr(W13);
  bf16* pG1 = bptr(G1);
  for (int e = 0; e < LE; ++e) {
    int M = (int)cnt[e];
    if (M == 0) continue;
    long row0 = off_end[e] - cnt[e];
    gemm_rowmajor_ABt(handle, M, G1N, H, pW13 + (long)e * G1N * H, pA + row0 * H, pG1 + row0 * G1N);
  }

  // fused SiLU-gate
  {
    long total = n * I;
    int threads = 256;
    silu_gate_kernel<<<(total + threads - 1) / threads, threads, 0, stream>>>(pG1, bptr(Inter), n);
  }

  // GEMM2 per expert: G2_e[M,2048] = Inter_e[M,768] @ W2[e]^T
  bf16* pInter = bptr(Inter);
  bf16* pW2 = bptr(W2);
  bf16* pG2 = bptr(G2);
  for (int e = 0; e < LE; ++e) {
    int M = (int)cnt[e];
    if (M == 0) continue;
    long row0 = off_end[e] - cnt[e];
    gemm_rowmajor_ABt(handle, M, H, I, pW2 + (long)e * H * I, pInter + row0 * I, pG2 + row0 * H);
  }

  // weighted scatter-add
  {
    long total = n * H;
    int threads = 256;
    scatter_weighted_kernel<<<(total + threads - 1) / threads, threads, 0, stream>>>(
        pG2, sorted_tok.data_ptr<long>(), sorted_w.data_ptr<float>(), out.data_ptr<float>(), n);
  }

  return out.to(torch::kBFloat16);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("moe_forward", &moe_forward, "Fused MoE forward (BF16, EP local)");
}
