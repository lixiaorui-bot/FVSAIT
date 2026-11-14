# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# FVSAIT re-implementation: builds the backbone by TASK_MODEL.NAME
#   * VSAIT  -> plain ResNet generator + VSADiscriminator (original)
#   * FVSAIT -> plain ResNet generator + FVSADiscriminator (phase input)

from .discriminators import FVSADiscriminator, VSADiscriminator
from .generators import ResNetGenerator


def build_backbone(cfg):
    """Build backbone generators/discriminators for the task model"""
    # ResNet generator (CUT-style backbone, shared by VSAIT and FVSAIT)
    g_net = ResNetGenerator(
        cfg.TASK_MODEL.IN_CHANNELS,
        cfg.TASK_MODEL.OUT_CHANNELS,
        cfg.TASK_MODEL.RES_CHANNELS,
        cfg.TASK_MODEL.RES_BLOCKS,
        **cfg.TASK_MODEL.get("GENERATOR_KWARGS", {}),
    )

    # discriminator on hyper-vector features
    lsh_cfg = cfg.TASK_MODEL.get("LSH", {})
    d_cls = FVSADiscriminator if cfg.TASK_MODEL.get("NAME") == "FVSAIT" else VSADiscriminator
    d_net = d_cls(
        lsh_cfg.get("out_channels"),
        num_scales=cfg.TASK_MODEL.get("DISCRIMINATOR_SCALES", 1),
        **cfg.TASK_MODEL.get("DISCRIMINATOR_KWARGS", {}),
    )

    return g_net, d_net
