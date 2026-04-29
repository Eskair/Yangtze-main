# -*- coding: utf-8 -*-
"""
全流水线共用的 OpenAI 兼容客户端（默认连本机，可用环境变量改回云上）。

零配置开箱约定：
- **未设置** OPENAI_API_BASE / OPENAI_BASE_URL 时，默认 **`http://127.0.0.1:11434/v1`**（可用 `OPENAI_LOCAL_DEFAULT_BASE` 改默认端口/路径）
- **未设置** OPENAI_MODEL 时，默认 **`llama3.2:latest`**（Ollama 常见 tag；若没有该模型请先 `ollama pull` 或改 `OPENAI_MODEL`）
- **HTTP 超时**：未设置 `LLM_API_TIMEOUT_SECONDS` 时，本机默认 **600s**（CPU 推理首个 chunk 易超过 120s）；云上默认 **120s**。可用环境变量统一覆盖。

改回云端 OpenAI 示例（.env）：
  OPENAI_API_BASE=https://api.openai.com/v1
  OPENAI_API_KEY=sk-...
  OPENAI_MODEL=gpt-4o-mini

注意：各脚本入口使用 `load_dotenv(override=True)`，使项目根 `.env` 优先于 shell 里遗留的 OPENAI_*。
"""
from __future__ import annotations

import os
import sys
import urllib.error
import urllib.request
from typing import Any, Optional

DEFAULT_OPENAI_MODEL = "llama3.2:latest"


def resolve_openai_base_url() -> str:
    """优先 OPENAI_*；否则本机默认 Ollama-compatible base。"""
    u = os.getenv("OPENAI_API_BASE", "").strip() or os.getenv("OPENAI_BASE_URL", "").strip()
    if u:
        return u.rstrip("/")
    return os.getenv(
        "OPENAI_LOCAL_DEFAULT_BASE",
        "http://127.0.0.1:11434/v1",
    ).strip().rstrip("/")


def resolve_openai_api_key() -> str:
    """显式 OPENAI_API_KEY 优先；指向本机且无 key 时用占位密钥。"""
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if key:
        return key
    base = resolve_openai_base_url().lower()
    if "localhost" in base or "127.0.0.1" in base:
        return os.getenv("OPENAI_LOCAL_PLACEHOLDER_KEY", "ollama").strip()
    return ""


def make_openai_client(*, timeout: Optional[float] = None) -> Any:
    from openai import OpenAI

    api_key = resolve_openai_api_key()
    if not api_key:
        raise RuntimeError(
            "未配置 OPENAI_API_KEY。若要使用云端 OpenAI，请在 .env 中设置 OPENAI_API_BASE=https://api.openai.com/v1 "
            "并提供有效密钥；本机默认已指向 localhost，可无真实密钥并使用 OPENAI_LOCAL_PLACEHOLDER_KEY。"
        )
    kw: dict[str, Any] = {"api_key": api_key, "base_url": resolve_openai_base_url()}
    if timeout is None:
        timeout = default_http_timeout_seconds()
    kw["timeout"] = timeout
    return OpenAI(**kw)


def getenv_model(default: Optional[str] = None) -> str:
    """读 OPENAI_MODEL；未配置时用 DEFAULT_OPENAI_MODEL（本机默认）。"""
    d = default if default is not None else DEFAULT_OPENAI_MODEL
    return os.getenv("OPENAI_MODEL", d).strip() or d


def is_local_openai_base(base: str) -> bool:
    b = (base or "").lower()
    return "localhost" in b or "127.0.0.1" in b or b.startswith("http://[::1]")


def default_http_timeout_seconds() -> float:
    """单次请求的客户端超时（秒）。本机 Ollama CPU 推理往往较慢，默认放宽。"""
    raw = os.getenv("LLM_API_TIMEOUT_SECONDS", "").strip()
    if raw:
        return float(raw)
    if is_local_openai_base(resolve_openai_base_url()):
        return 600.0
    return 120.0


def openai_compat_service_root(openai_base: str) -> str:
    """
    将 OpenAI 兼容 base（通常 .../v1）还原为服务根，用于访问 Ollama 自带 /api/*。
    例：http://127.0.0.1:11434/v1 -> http://127.0.0.1:11434
    """
    u = openai_base.rstrip("/")
    if u.endswith("/v1"):
        return u[: -len("/v1")].rstrip("/")
    return u


def looks_like_openai_cloud_model(model_name: str) -> bool:
    """启发式：名称像 OpenAI 云端模型，而本机 Ollama 通常不会用这些 id。"""
    n = (model_name or "").strip().lower()
    if not n:
        return False
    if n.startswith(("gpt-", "o1", "o3", "o4", "text-embedding", "davinci", "babbage")):
        return True
    return False


def warn_if_model_mismatches_local_base(model_name: str, base: str) -> None:
    if not is_local_openai_base(base):
        return
    if looks_like_openai_cloud_model(model_name):
        print(
            "[WARN] 当前 OPENAI_API_BASE 指向本机，但 OPENAI_MODEL 仍像云端名称（"
            + model_name
            + "）。若使用 Ollama，请在 .env 中改为已 ollama pull 的本地模型名，"
            "例如 OPENAI_MODEL=llama3.2:latest",
            file=sys.stderr,
        )


def warn_if_openai_key_while_local_base() -> None:
    """本机推理时仍填了 sk-* 仍可工作，但容易与「仅本地」预期混淆。"""
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key.startswith("sk-"):
        return
    if is_local_openai_base(resolve_openai_base_url()):
        print(
            "[WARN] 已配置 OPENAI_API_BASE 为本机，但 OPENAI_API_KEY 以 sk- 开头（像 OpenAI 云密钥）。"
            "使用本机 Ollama 时可将 OPENAI_API_KEY 留空以使用占位键，或仍保留（部分兼容层会忽略）。",
            file=sys.stderr,
        )


def maybe_probe_local_inference(*, skip: bool = False, timeout: float = 3.0) -> None:
    """
    在发起首次 LLM 请求前调用：若 endpoint 指向本机，则探测服务是否可用。
    可通过环境变量 SKIP_LOCAL_LLM_HEALTH_CHECK=1 或调用参数 skip=True 跳过。
    """
    if skip:
        return
    if os.getenv("SKIP_LOCAL_LLM_HEALTH_CHECK", "").strip().lower() in ("1", "true", "yes", "on"):
        return
    base = resolve_openai_base_url()
    if not is_local_openai_base(base):
        return
    warn_if_openai_key_while_local_base()
    warn_if_model_mismatches_local_base(getenv_model(), base)
    _probe_local_http_or_raise(base, timeout=timeout)


def _probe_local_http_or_raise(openai_base: str, *, timeout: float) -> None:
    """优先探测 Ollama /api/tags；否则探测 OpenAI 兼容 /v1/models。"""
    root = openai_compat_service_root(openai_base)
    candidates: list[str] = []
    if "11434" in openai_base:
        candidates.append(root + "/api/tags")
    candidates.append(openai_base.rstrip("/") + "/models")

    last_err: Optional[BaseException] = None
    for url in candidates:
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_err = e
            continue
    mh = getenv_model()
    pull_name = mh.split(":")[0] if mh and ":" in mh else mh
    if looks_like_openai_cloud_model(mh):
        pull_example = DEFAULT_OPENAI_MODEL.split(":")[0] if ":" in DEFAULT_OPENAI_MODEL else DEFAULT_OPENAI_MODEL
        pull_tip = (
            "当前 OPENAI_MODEL 仍像云端名称；请先在 .env 改为本机可用名称（例如 "
            + DEFAULT_OPENAI_MODEL
            + "），再执行 ollama pull "
            + pull_example
        )
    else:
        pull_tip = "已拉取与本机一致的模型：ollama pull " + (pull_name or "你的模型")

    raise RuntimeError(
        "无法连接本机推理服务。当前 OPENAI_API_BASE="
        + openai_base
        + "。\n"
        "请按顺序检查：\n"
        "  1) 已安装并启动 Ollama（或 LM Studio 等），端口与上述地址一致；\n"
        "  2) Ollama 时可在浏览器访问：http://127.0.0.1:11434/api/tags；\n"
        "  3) "
        + pull_tip
        + "（或改用 ollama list 显示的完整名称）；\n"
        "  4) 若暂时无法起服务，可设 SKIP_LOCAL_LLM_HEALTH_CHECK=1 或脚本参数 --skip-inference-health-check（仍会失败在真实请求时）。\n"
        f"（探测失败原因：{last_err!r}）"
    ) from last_err
