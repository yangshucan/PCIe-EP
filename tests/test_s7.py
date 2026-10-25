"""Independent per-token E4M3FN scaling/reference checks (CPU-only)."""

import argparse
import json
import os
from pathlib import Path
import unittest

import torch

from pcie_ep.quantization import (
    dequantize_fp8_reference,
    fp8_roundtrip_reference,
    quantize_fp8_reference,
)
from pcie_ep import HostStagedConfig, HostStagedEPBuffer


def numeric_corner_payload(hidden):
    """Exact BF16 bits: zero, signed subnormals, tiny normals and finite maxima."""
    patterns = (
        torch.tensor(
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0x0001, 0x8001, 0, 0x0002, 0x8002, 0x0007, 0x8007, 0],
                [0x007F, 0x807F, 0x0080, 0x8080, 0x0040, 0x8040, 0, 0x0001],
                [0x0200, 0x8200, 0x0300, 0x8300, 0x0400, 0x8400, 0, 0x0080],
                [0x7F7F, 0xFF7F, 0x7E80, 0xFE80, 0, 0x3F80, 0xBF80, 0x0080],
            ],
            dtype=torch.int32,
        )
        .to(torch.int16)
        .view(torch.bfloat16)
    )
    if hidden % patterns.shape[1]:
        raise ValueError("Numeric test hidden size must be divisible by eight")
    return patterns.repeat(1, hidden // patterns.shape[1])


class FP8ReferenceTests(unittest.TestCase):
    def test_zero_rows_use_scale_one(self):
        x = torch.zeros(3, 7168, dtype=torch.bfloat16)
        codes, scales = quantize_fp8_reference(x)
        self.assertTrue(torch.equal(scales, torch.ones(3)))
        self.assertFalse(torch.count_nonzero(codes))
        self.assertTrue(torch.equal(dequantize_fp8_reference(codes, scales), x))

    def test_scales_are_per_token_and_bound_quantization_error(self):
        generator = torch.Generator().manual_seed(71)
        x = (torch.randn(16, 7168, generator=generator) * torch.logspace(-3, 3, 16).unsqueeze(1)).to(torch.bfloat16)
        codes, scales = quantize_fp8_reference(x)
        self.assertEqual(codes.dtype, torch.uint8)
        self.assertEqual(scales.dtype, torch.float32)
        self.assertTrue(torch.equal(scales, x.float().abs().amax(1) / 448.0))
        decoded = dequantize_fp8_reference(codes, scales)
        error = (decoded.float() - x.float()).abs().amax(1)
        self.assertTrue(torch.all(error <= x.float().abs().amax(1) * 0.041 + 1e-7))

    def test_fp8_max_and_signs_are_representable(self):
        x = torch.tensor([[0, 1, -1, 448, -448]], dtype=torch.bfloat16)
        self.assertTrue(torch.equal(fp8_roundtrip_reference(x), x))

    def test_empty_input_preserves_shapes(self):
        codes, scales = quantize_fp8_reference(torch.empty(0, 128, dtype=torch.bfloat16))
        self.assertEqual(codes.shape, (0, 128))
        self.assertEqual(scales.shape, (0,))

    def test_finite_bf16_extremes_keep_finite_decode_and_bounded_error(self):
        x = numeric_corner_payload(128)
        self.assertTrue(torch.isfinite(x).all())
        codes, scales = quantize_fp8_reference(x)
        decoded = dequantize_fp8_reference(codes, scales)
        self.assertTrue(torch.isfinite(scales).all())
        self.assertTrue(torch.isfinite(scales.reciprocal()).all())
        self.assertTrue(torch.isfinite(decoded).all())
        self.assertTrue(torch.equal(scales[1:4], torch.full((3,), torch.finfo(torch.float32).tiny)))
        self.assertTrue(torch.equal(decoded[0], x[0]))
        self.assertTrue(torch.equal(decoded[1], x[1]))
        # Double precision keeps this independent bound meaningful at both
        # BF16 extremes, including below FP32 normal and near FP32 maximum.
        maximum = x.double().abs().amax(1)
        error = (decoded.double() - x.double()).abs().amax(1)
        minimum_bf16_subnormal = 2.0**-133
        self.assertTrue(torch.all(error <= maximum * 0.07 + minimum_bf16_subnormal))


def distributed_main():
    import torch.distributed as dist
    from test_correctness import gather_cpu, make_inputs, verify_result

    parser = argparse.ArgumentParser()
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--stage", type=int, choices=(7, 8), default=7)
    parser.add_argument("--numa-mode", choices=("direct", "relay"), default="direct")
    parser.add_argument("--only-numeric-corners", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/s7.json"))
    args = parser.parse_args()
    if args.numa_mode == "relay" and args.stage < 8:
        parser.error("NUMA relay requires --stage 8")
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    records = []
    try:
        for stages in (2, 3):
            config = HostStagedConfig(
                stage=args.stage,
                channels=2,
                pipeline_stages=stages,
                warp_specialization=True,
                fuse_receive_expand=True,
                source_fusion=True,
                wire_dtype="fp8_e4m3",
                payload_slots=2,
                descriptor_slots=3,
                numa_mode=args.numa_mode,
            )
            with HostStagedEPBuffer(
                group=dist.group.WORLD,
                num_max_tokens_per_rank=17,
                hidden=7168,
                num_topk=8,
                num_experts=256,
                config=config,
            ) as buffer:
                patterns = (
                    ("numeric_corners",)
                    if args.only_numeric_corners
                    else (
                        "random",
                        "multicast",
                        "local",
                        "none",
                        "partial",
                        "numeric_corners",
                    )
                )
                for pattern in patterns:
                    if pattern == "numeric_corners":
                        payload = numeric_corner_payload(7168)
                        _, indices, weights = make_inputs(
                            rank,
                            world,
                            payload.shape[0],
                            7168,
                            256,
                            8,
                            "multicast",
                        )
                        cpu = (payload, indices, weights)
                    else:
                        cpu = make_inputs(rank, world, 17, 7168, 256, 8, pattern)
                    gathered = gather_cpu(cpu)
                    expected = [
                        (payload if source == rank else fp8_roundtrip_reference(payload), idx, weight)
                        for source, (payload, idx, weight) in enumerate(gathered)
                    ]
                    if pattern == "numeric_corners" and any(
                        not torch.isfinite(payload).all() for payload, _, _ in expected
                    ):
                        raise AssertionError("A finite numeric input produced a nonfinite reference")
                    x, idx, weights = (value.to("cuda") for value in cpu)
                    for expand in (False, True):
                        result = buffer.dispatch(
                            x, idx, weights, do_expand=expand, expert_alignment=8, do_zero_padding=expand
                        )
                        buffer.check_errors(result[3])
                        verify_result(result, expected, rank, world, 256, 8, expand, 8)
                        records.append(
                            {"pipeline_stages": stages, "pattern": pattern, "expand": expand, "status": "passed"}
                        )
        if rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "records": records,
                        "wire_dtype": "fp8_e4m3",
                        "local_bypass": "original BF16",
                        "stage": args.stage,
                        "numa_mode": args.numa_mode,
                    },
                    indent=2,
                )
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    if "--distributed" in os.sys.argv:
        distributed_main()
    else:
        unittest.main()
