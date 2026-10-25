#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstddef>

namespace pcie_ep::host_staged {

inline constexpr float kFP8E4M3Max = 448.0f;
inline constexpr float kFP8ScaleMin = 0x1p-126f;

__device__ inline std::uint8_t quantize_bf16_to_fp8(std::uint16_t value, float invscale) {
    const float input = __uint_as_float(std::uint32_t(value) << 16);
    return __nv_cvt_float_to_fp8(input * invscale, __NV_SATFINITE, __NV_E4M3);
}

__device__ inline std::uint16_t dequantize_fp8_to_bf16(std::uint8_t value, float scale) {
    const __half decoded(__nv_cvt_fp8_to_halfraw(value, __NV_E4M3));
    return __bfloat16_as_ushort(__float2bfloat16_rn(__half2float(decoded) * scale));
}

__device__ inline float fp8_warp_max(float value) {
    for (int offset = 16; offset > 0; offset /= 2)
        value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
    return value;
}

// One 256-thread CTA per token. Host code must skip the launch when T == 0.
// The output allocation and scales live in the runtime, never in this kernel.
__global__ void quantize_transport_payload(const std::uint16_t* input, std::uint8_t* output,
                                         float* scales, int num_tokens, int hidden) {
    const int token = blockIdx.x;
    if (token >= num_tokens)
        return;

    __shared__ float warp_maxima[8];
    __shared__ float scale;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    float maximum = 0.0f;
    for (int column = threadIdx.x; column < hidden; column += blockDim.x) {
        const float value = __uint_as_float(std::uint32_t(input[std::size_t(token) * hidden + column]) << 16);
        maximum = fmaxf(maximum, fabsf(value));
    }
    maximum = fp8_warp_max(maximum);
    if (lane == 0)
        warp_maxima[warp] = maximum;
    __syncthreads();

    if (warp == 0) {
        maximum = lane < 8 ? warp_maxima[lane] : 0.0f;
        maximum = fp8_warp_max(maximum);
        if (lane == 0) {
            // A subnormal scale can overflow its reciprocal and turn a
            // finite zero component into NaN through zero * infinity.
            scale = maximum == 0.0f ? 1.0f : fmaxf(maximum / kFP8E4M3Max, kFP8ScaleMin);
            scales[token] = scale;
        }
    }
    __syncthreads();

    const float invscale = 1.0f / scale;
    for (int column = threadIdx.x; column < hidden; column += blockDim.x) {
        const std::size_t index = std::size_t(token) * hidden + column;
        output[index] = quantize_bf16_to_fp8(input[index], invscale);
    }
}

}  // namespace pcie_ep::host_staged
