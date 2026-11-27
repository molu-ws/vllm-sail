# SPDX-License-Identifier: Apache-2.0
"""Bare-CPU tests for the Qwen4Exp renderer-mode selection helper.

``select_qwen4_mode`` imports nothing from vLLM, so the arch -> tokenizer_mode
default is exercised here on the CPU-only merge gate with plain config stubs.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_sail.renderers.qwen4_mode import (
    QWEN4_EXP_ARCHITECTURES,
    QWEN4_MODE,
    select_qwen4_mode,
)


def _config(tokenizer_mode: str, architectures: list[str]) -> SimpleNamespace:
    return SimpleNamespace(tokenizer_mode=tokenizer_mode, architectures=architectures)


@pytest.mark.parametrize("arch", QWEN4_EXP_ARCHITECTURES)
def test_auto_qwen4_exp_selects_native_mode(arch: str):
    config = _config("auto", [arch])
    select_qwen4_mode(config)
    assert config.tokenizer_mode == QWEN4_MODE


@pytest.mark.parametrize("arch", ["Qwen3MoeForCausalLM", "LlamaForCausalLM"])
def test_auto_other_arch_keeps_auto(arch: str):
    config = _config("auto", [arch])
    select_qwen4_mode(config)
    assert config.tokenizer_mode == "auto"


@pytest.mark.parametrize("mode", ["hf", "mistral", "qwen4"])
def test_explicit_mode_is_respected(mode: str):
    config = _config(mode, ["Qwen4ExpForConditionalGeneration"])
    select_qwen4_mode(config)
    assert config.tokenizer_mode == mode


def test_missing_architectures_keeps_auto():
    config = _config("auto", [])
    select_qwen4_mode(config)
    assert config.tokenizer_mode == "auto"


def test_selection_is_idempotent():
    config = _config("auto", ["Qwen4ExpForCausalLM"])
    select_qwen4_mode(config)
    select_qwen4_mode(config)
    assert config.tokenizer_mode == QWEN4_MODE
