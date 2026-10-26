"""Shared transport switches for distributed validation programs."""

import argparse


def add_transport_arguments(parser, stage=6):
    parser.add_argument("--stage", type=int, choices=(6, 7, 8), default=stage)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--warp-specialization", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pipeline-stages", type=int, choices=[2, 3], default=2)
    parser.add_argument("--fuse-receive-expand", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--receive-expand-mode", choices=("scalar", "vectorized", "pipelined"), default="pipelined")
    parser.add_argument("--source-fusion", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--numa-mode", choices=["direct", "relay"], default="direct")
    parser.add_argument("--replica-slots", type=int, default=256)


def transport_options(args):
    return {
        name: getattr(args, name)
        for name in (
            "stage",
            "channels",
            "warp_specialization",
            "pipeline_stages",
            "fuse_receive_expand",
            "receive_expand_mode",
            "source_fusion",
            "numa_mode",
            "replica_slots",
        )
    }
