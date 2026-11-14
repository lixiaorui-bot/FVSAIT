# # Copyright (c) Meta Platforms, Inc. and affiliates.
# # All rights reserved.
# #
# # This source code is licensed under the license found in the
# # LICENSE file in the root directory of this source tree.
#
# import os
#
# from fvcore.common.config import CfgNode as _CfgNode
#
# from .logger import get_module_logger
#
# logger = get_module_logger(__name__)
#
#
# BASE_KEY = "_BASE_"
#
#
# def init_cfg_for_merge(cfg, new_cfg):
#     """Initialize config prior to merging new_cfg"""
#     keys = [*cfg.keys(), *new_cfg.keys()]
#     # for each key, set default of cfg
#     for k in keys:
#         if k == BASE_KEY:
#             continue
#         cfg.setdefault(k, new_cfg.get(k))
#         # if k is a node in new_cfg, recursively init
#         if isinstance(new_cfg.get(k), CfgNode):
#             init_cfg_for_merge(cfg.get(k), new_cfg.get(k))
#
#
# class CfgNode(_CfgNode):
#     """
#     Wrapper of `fvcore.common.config.CfgNode` with following updates:
#     1. Use unsafe yaml loading by default.
#        Note that this may lead to arbitrary code execution: you must not
#        load a config file from untrusted sources before manually inspecting
#        the content of the file.
#     2. Support config versioning.
#        When attempting to merge an old config, it will convert the old config automatically.
#     """
#
#     # Note that the default value of allow_unsafe is changed to True
#     def merge_from_file(
#         self,
#         cfg_filename: str,
#         allow_unsafe: bool = True,
#     ) -> None:
#         assert os.path.isfile(cfg_filename), f"{cfg_filename} not found."
#
#         loaded_cfg = self.load_yaml_with_base(cfg_filename, allow_unsafe=allow_unsafe)
#         loaded_cfg = type(self)(loaded_cfg)
#         init_cfg_for_merge(self, loaded_cfg)
#
#         self.merge_from_other_cfg(loaded_cfg)
#
#     def merge_from_dict(self, cfg_dict):
#         """Merge a dict into cfg"""
#         new_cfg = type(self)(cfg_dict)
#         init_cfg_for_merge(self, new_cfg)
#         self.merge_from_other_cfg(new_cfg)
#
#     def dump(self, *args, **kwargs):
#         """
#         Returns:
#             str: a yaml string representation of the config
#         """
#         # to make it show up in docs
#         return super().dump(*args, **kwargs)
#
#
# global_cfg = CfgNode()
#
#
# def get_cfg() -> CfgNode:
#     """
#     Get a copy of the default config.
#     """
#     from .defaults import _C
#
#     return _C.clone()
#
#
# def get_cfg_from_dict(d) -> CfgNode:
#     """
#     Get a copy of the default config, modified by an input dict.
#     """
#     cfg = get_cfg()
#     cfg.merge_from_dict(d)
#
#     return cfg
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import yaml  # 需要导入 yaml 模块

from fvcore.common.config import CfgNode as _CfgNode

from .logger import get_module_logger

logger = get_module_logger(__name__)

BASE_KEY = "_BASE_"


def init_cfg_for_merge(cfg, new_cfg):
    """Initialize config prior to merging new_cfg"""
    keys = [*cfg.keys(), *new_cfg.keys()]
    # for each key, set default of cfg
    for k in keys:
        if k == BASE_KEY:
            continue
        cfg.setdefault(k, new_cfg.get(k))
        # if k is a node in new_cfg, recursively init
        if isinstance(new_cfg.get(k), CfgNode):
            init_cfg_for_merge(cfg.get(k), new_cfg.get(k))


class CfgNode(_CfgNode):
    """
    Wrapper of `fvcore.common.config.CfgNode` with following updates:
    1. Use unsafe yaml loading by default.
       Note that this may lead to arbitrary code execution: you must not
       load a config file from untrusted sources before manually inspecting
       the content of the file.
    2. Support config versioning.
       When attempting to merge an old config, it will convert the old config automatically.
    """

    def load_yaml_with_base(self, cfg_filename, allow_unsafe=True):
        """
        Load a YAML config file with base file support, using UTF-8 encoding.
        """
        with open(cfg_filename, "r", encoding="utf-8") as f:
            loaded_cfg = yaml.safe_load(f) if not allow_unsafe else yaml.load(f, Loader=yaml.Loader)

        # Handle base files recursively
        if BASE_KEY in loaded_cfg:
            base_cfg_file = loaded_cfg[BASE_KEY]
            if base_cfg_file.startswith("~"):
                base_cfg_file = os.path.expanduser(base_cfg_file)
            if not any(base_cfg_file.startswith(d) for d in ["./", "../", "/"]):
                # base file is relative to the current config file
                base_cfg_file = os.path.join(os.path.dirname(cfg_filename), base_cfg_file)
            base_cfg = self.load_yaml_with_base(base_cfg_file, allow_unsafe=allow_unsafe)
            del loaded_cfg[BASE_KEY]

            # Merge base config with current config
            base_cfg = type(self)(base_cfg)
            init_cfg_for_merge(base_cfg, loaded_cfg)
            base_cfg.merge_from_other_cfg(loaded_cfg)
            return base_cfg

        return loaded_cfg

    # Note that the default value of allow_unsafe is changed to True
    def merge_from_file(
            self,
            cfg_filename: str,
            allow_unsafe: bool = True,
    ) -> None:
        assert os.path.isfile(cfg_filename), f"{cfg_filename} not found."

        loaded_cfg = self.load_yaml_with_base(cfg_filename, allow_unsafe=allow_unsafe)
        loaded_cfg = type(self)(loaded_cfg)
        init_cfg_for_merge(self, loaded_cfg)

        self.merge_from_other_cfg(loaded_cfg)

    def merge_from_dict(self, cfg_dict):
        """Merge a dict into cfg"""
        new_cfg = type(self)(cfg_dict)
        init_cfg_for_merge(self, new_cfg)
        self.merge_from_other_cfg(new_cfg)

    def dump(self, *args, **kwargs):
        """
        Returns:
            str: a yaml string representation of the config
        """
        # to make it show up in docs
        return super().dump(*args, **kwargs)


global_cfg = CfgNode()


def get_cfg() -> CfgNode:
    """
    Get a copy of the default config.
    """
    from .defaults import _C

    return _C.clone()


def get_cfg_from_dict(d) -> CfgNode:
    """
    Get a copy of the default config, modified by an input dict.
    """
    cfg = get_cfg()
    cfg.merge_from_dict(d)

    return cfg