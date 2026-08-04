"""Compatibility shim for LLaMAFactory 0.9.4 with TRL 0.24.

LLaMAFactory passes an Accelerator instance to prepare_deepspeed, while the
installed TRL function expects an integer micro-batch size. This shim preserves
normal integer calls and adapts only that incompatible call shape.
"""

from __future__ import annotations

import inspect
import os

import trl.trainer.utils as trainer_utils


_original_prepare_deepspeed = trainer_utils.prepare_deepspeed
_parameters = list(inspect.signature(_original_prepare_deepspeed).parameters)


if len(_parameters) >= 2 and _parameters[1] == "per_device_train_batch_size":

    def _compatible_prepare_deepspeed(model, batch_size_or_accelerator, fp16=False, bf16=False):
        if isinstance(batch_size_or_accelerator, int):
            return _original_prepare_deepspeed(
                model,
                batch_size_or_accelerator,
                fp16=fp16,
                bf16=bf16,
            )

        accelerator = batch_size_or_accelerator
        batch_size = int(os.environ.get("LLAMAFACTORY_REF_BATCH_SIZE", "1"))
        if batch_size <= 0:
            raise ValueError("LLAMAFACTORY_REF_BATCH_SIZE must be positive")
        mixed_precision = getattr(accelerator, "mixed_precision", None)
        return _original_prepare_deepspeed(
            model,
            batch_size,
            fp16=mixed_precision == "fp16",
            bf16=mixed_precision == "bf16",
        )

    trainer_utils.prepare_deepspeed = _compatible_prepare_deepspeed
