// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Per-head normalization in a CUTLASS matrix epilogue. Adapted from
// https://github.com/Vedaant-J/structured-data-models-inference/blob/01646740a388b388e0617f80b96e2ed5ce40e671/benchmark/inference/kernels/csrc/common/head_rms.cuh
#include <array>
#include <climits>
#include <ATen/autocast_mode.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination.h>

namespace linear_rmsnorm {
using Half = cutlass::half_t;
using Bias = cutlass::epilogue::thread::LinearCombination<Half, 8, float, float>;

struct Op : Bias {
  struct Params : Bias::Params {
    float eps;
    Half const* cosine;
    Half const* sine;
    int sequence;
    int normalized_channels;
    CUTLASS_HOST_DEVICE Params(float epsilon = 1e-6f,
                              Half const* cos = nullptr,
                              Half const* sin = nullptr, int seq = 0, int channels = 0)
        : Bias::Params(1.f, 1.f), eps(epsilon), cosine(cos), sine(sin),
          sequence(seq), normalized_channels(channels) {}
  };
  float eps;
  Half const* cosine;
  Half const* sine;
  int sequence;
  int normalized_channels;
  CUTLASS_HOST_DEVICE Op(Params const& params)
      : Bias(params), eps(params.eps), cosine(params.cosine), sine(params.sine),
        sequence(params.sequence), normalized_channels(params.normalized_channels) {}
};

CUTLASS_DEVICE float add(float a, float b) {
  float result;
  asm("add.rn.f32 %0,%1,%2;" : "=f"(result) : "f"(a), "f"(b));
  return result;
}

template <int Head>
CUTLASS_DEVICE float inverse(float sum, float eps) {
  float mean, result;
  asm("div.full.f32 %0,%1,%2;" : "=f"(mean) : "f"(sum), "f"(float(Head)));
  mean = add(mean, eps);
  asm("rsqrt.approx.ftz.f32 %0,%1;" : "=f"(result) : "f"(mean));
  return result;
}

template <class Base, int Head, bool Rotary, bool Packed>
struct HeadEpilogue : Base {
  using Iterator = typename Base::OutputTileIterator;
  using Fragment = typename Iterator::Fragment;
  using Map = typename Iterator::ThreadMap;
  static constexpr int Lanes = Head / 8;
  static_assert(Iterator::kElementsPerAccess == 8 && Head % 16 == 0 && Lanes <= 32);
  static_assert(Map::Detail::kAccessWidth % Lanes == 0 && Map::Delta::kColumn % Head == 0);
  using Base::Base;

  struct Source : Base::SourceAspectNeeded {
    using Base::SourceAspectNeeded::SourceAspectNeeded;
    int start_row, start_column;

    CUTLASS_DEVICE void load() {
      start_row = this->source_iterator.thread_start_row();
      start_column = this->source_iterator.thread_start_column();
      Base::SourceAspectNeeded::load();
    }

    CUTLASS_DEVICE void apply_output_operator(
        Fragment& out, Op const& op,
        typename Base::SharedLoadIterator::Fragment const& acc) {
      Base::SourceAspectNeeded::apply_output_operator(out, op, acc, this->source_fragment);
      auto* vectors = reinterpret_cast<cutlass::Array<Half, 8>*>(&out);
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Fragment::kElements / 8; ++j) {
        int column = start_column + j % Map::Iterations::kColumn * Map::Delta::kColumn;
        unsigned mask = 0xffffffffu;
        if constexpr (Packed) {
          // Value columns retain the rounded projection. Complete key-head
          // subgroups participate in all rotary and normalization shuffles.
          mask = __ballot_sync(mask, column < op.normalized_channels);
          if (column >= op.normalized_channels) continue;
        }
        if constexpr (Rotary) {
          // Fragment vectors follow column, row, group, cluster order.
          constexpr int Cols = Map::Iterations::kColumn;
          constexpr int Rows = Map::Iterations::kRow;
          constexpr int Groups = Map::Iterations::kGroup;
          int fragment_row = j / Cols;
          int row = start_row + fragment_row % Rows * Map::Delta::kRow
              + fragment_row / Rows % Groups * Map::Delta::kGroup
              + fragment_row / (Rows * Groups) * Map::Delta::kCluster;
          int64_t table = int64_t(row % op.sequence) * (Head / 2);
          using Vector = cutlass::AlignedArray<Half, 8, 16>;
          auto cos = *reinterpret_cast<Vector const*>(op.cosine + table + column % (Head / 2));
          auto sin = *reinterpret_cast<Vector const*>(op.sine + table + column % (Head / 2));
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < 8; ++i) {
            float value = float(vectors[j][i]);
            float partner = __shfl_xor_sync(mask, value, Lanes / 2, Lanes);
            float first = float(Half(__fmul_rn(value, float(cos[i]))));
            float second = float(Half(__fmul_rn(partner, float(sin[i]))));
            vectors[j][i] = Half(add(first, column % Head < Head / 2 ? -second : second));
          }
        }
        float first = 0.f, second = 0.f;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < 4; ++i) {
          float a = float(vectors[j][i]), b = float(vectors[j][i + 4]);
          first = __fmaf_rn(a, a, first);
          second = __fmaf_rn(b, b, second);
        }
        CUTLASS_PRAGMA_UNROLL
        for (int shift = Lanes / 2; shift >= 1; shift >>= 1) {
          first = add(first, __shfl_xor_sync(mask, first, shift, Lanes));
          second = add(second, __shfl_xor_sync(mask, second, shift, Lanes));
        }
        float scale = inverse<Head>(add(first, second), op.eps);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < 8; ++i) {
          vectors[j][i] = Half(__fmul_rn(float(vectors[j][i]), scale));
        }
      }
    }
  };

  CUTLASS_DEVICE void operator()(Op const& op, Iterator destination,
                                 typename Base::AccumulatorTile const& acc,
                                 Iterator bias) {
    Base::operator()(op, destination, acc, Source(bias));
  }
};

template <int Tile>
using Gemm = cutlass::gemm::device::Gemm<
    Half, cutlass::layout::RowMajor, Half, cutlass::layout::ColumnMajor,
    Half, cutlass::layout::RowMajor, float,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, Tile, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<16, 8, 16>, Op,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>, 3, 8, 8,
    false, cutlass::arch::OpMultiplyAdd>;

template <int Tile, int Head, bool Rotary, bool Packed>
using Kernel = cutlass::gemm::kernel::Gemm<
    typename Gemm<Tile>::GemmKernel::Mma,
    HeadEpilogue<typename Gemm<Tile>::GemmKernel::Epilogue, Head, Rotary, Packed>,
    typename Gemm<Tile>::GemmKernel::ThreadblockSwizzle, false>;

template <int Tile, int Head, bool Rotary, bool Packed>
void launch(at::Tensor const& x, at::Tensor const& w, at::Tensor const& bias,
            at::Tensor const& out, float eps,
            at::Tensor const* cosine, at::Tensor const* sine) {
  using G = Gemm<Tile>;
  using K = Kernel<Tile, Head, Rotary, Packed>;
  auto pointer = [](at::Tensor const& tensor) {
    return reinterpret_cast<Half*>(tensor.data_ptr());
  };
  int rows = x.size(0), width = x.size(1), outputs = w.size(0);
  Op::Params epilogue(eps, cosine ? pointer(*cosine) : nullptr,
                      sine ? pointer(*sine) : nullptr,
                      cosine ? int(cosine->size(0)) : 0, width);
  typename G::Arguments args({rows, outputs, width}, {pointer(x), width},
      {pointer(w), width}, {pointer(bias), 0}, {pointer(out), outputs}, epilogue, 1);
  TORCH_CHECK(G::can_implement(args) == cutlass::Status::kSuccess, "unsupported GEMM layout");
  typename G::ThreadblockSwizzle swizzle;
  auto tiled = swizzle.get_tiled_shape(args.problem_size, {128, Tile, 32}, 1);
  typename K::Params params(args.problem_size, tiled, args.ref_A.non_const_ref(),
      args.ref_B.non_const_ref(), args.ref_C.non_const_ref(), args.ref_D, args.epilogue, nullptr);
  constexpr int shared = sizeof(typename K::SharedStorage);
  C10_CUDA_CHECK(cudaFuncSetAttribute(cutlass::Kernel<K>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared));
  cutlass::Kernel<K><<<swizzle.get_grid_shape(tiled), K::kThreadCount, shared,
                       at::cuda::getCurrentCUDAStream()>>>(params);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <bool Packed>
void project(at::Tensor const& x, at::Tensor const& w, at::Tensor const& b,
             at::Tensor const& out, int head, float eps,
             std::optional<at::Tensor> const& cosine,
             std::optional<at::Tensor> const& sine) {
  if (cosine) {
    if (head == 32) launch<256, 32, true, Packed>(x, w, b, out, eps, &*cosine, &*sine);
    else if (head == 64) launch<256, 64, true, Packed>(x, w, b, out, eps, &*cosine, &*sine);
    else launch<256, 128, true, Packed>(x, w, b, out, eps, &*cosine, &*sine);
  } else {
    if (head == 32) launch<256, 32, false, Packed>(x, w, b, out, eps, nullptr, nullptr);
    else if (head == 64) launch<256, 64, false, Packed>(x, w, b, out, eps, nullptr, nullptr);
    else launch<256, 128, false, Packed>(x, w, b, out, eps, nullptr, nullptr);
  }
}

at::Tensor run(at::Tensor const& x, at::Tensor const& weight,
               std::optional<at::Tensor> const& bias, int head, double eps,
               std::optional<at::Tensor> const& cosine,
               std::optional<at::Tensor> const& sine, bool packed_key_value) {
  TORCH_CHECK(x.dim() == 2 && x.is_cuda() && x.scalar_type() == at::kHalf &&
              x.is_contiguous() && x.size(0) >= 32 && x.size(0) < INT_MAX - 128,
              "expected contiguous CUDA FP16 matrix input");
  int width = x.size(1);
  int outputs = (packed_key_value ? 2 : 1) * width;
  TORCH_CHECK((head == 32 || head == 64 || head == 128) && width >= head &&
              width % head == 0 && width < 1024 &&
              weight.sizes() == at::IntArrayRef({outputs, width}) &&
              (!bias || bias->sizes() == at::IntArrayRef({outputs})),
              "unsupported projection dimensions");
  for (auto const* tensor : {&weight, bias ? &*bias : &weight}) {
    TORCH_CHECK(tensor->device() == x.device() && tensor->is_contiguous() &&
                (tensor->scalar_type() == at::kHalf || tensor->scalar_type() == at::kFloat),
                "expected contiguous FP16 or FP32 parameters on the input device");
  }
  TORCH_CHECK(bool(cosine) == bool(sine), "expected both rotary tables");
  if (cosine) {
    TORCH_CHECK(cosine->dim() == 2 && cosine->size(0) > 0 &&
                cosine->size(0) <= x.size(0) && cosine->size(1) == head / 2 &&
                sine->sizes() == cosine->sizes(), "unsupported rotary table dimensions");
    for (auto const* tensor : {&*cosine, &*sine}) {
      TORCH_CHECK(tensor->device() == x.device() && tensor->is_contiguous() &&
                  tensor->scalar_type() == at::kHalf &&
                  reinterpret_cast<uintptr_t>(tensor->data_ptr()) % 16 == 0,
                  "expected aligned CUDA FP16 rotary tables");
    }
  }
  c10::cuda::CUDAGuard guard(x.device());
  // Match native autocast's parameter conversion and cache ownership.
  auto w = at::autocast::cached_cast(at::kHalf, weight, at::kCUDA);
  auto b = bias ? at::autocast::cached_cast(at::kHalf, *bias, at::kCUDA)
                : at::zeros({outputs}, x.options());
  for (auto const* tensor : std::array<at::Tensor const*, 3>{&x, &w, &b}) {
    TORCH_CHECK(reinterpret_cast<uintptr_t>(tensor->data_ptr()) % 16 == 0,
                "expected 16-byte-aligned projection operands");
  }
  auto out = at::empty({x.size(0), outputs}, x.options());
  if (packed_key_value) {
    project<true>(x, w, b, out, head, float(eps), cosine, sine);
  } else {
    project<false>(x, w, b, out, head, float(eps), cosine, sine);
  }
  return out;
}
}  // namespace linear_rmsnorm

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("run", &linear_rmsnorm::run);
}
