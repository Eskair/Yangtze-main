# -*- coding: utf-8 -*-
"""
通用模型选择器（v5.1 · 兼容新旧 Gemini SDK + 明确 Key 优先级）
- PROVIDER: openai / deepseek / gemini
- 返回: {"client": <client_instance>, "model_name": str, "provider": str}
"""

import os
from dotenv import load_dotenv

load_dotenv()

def get_llm_client():
    provider = (os.getenv("PROVIDER", "openai") or "openai").lower().strip()

    # ========== OpenAI ==========
    if provider in ("openai", "chatgpt"):
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY 未配置")
        client = OpenAI(api_key=api_key)
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        print(f"✅ 已加载 OpenAI 模型：{model_name}")
        return {"client": client, "model_name": model_name, "provider": "openai"}

    # ========== DeepSeek ==========
    elif provider == "deepseek":
        from openai import OpenAI
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY 未配置")
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
        model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        print(f"✅ 已加载 DeepSeek 模型：{model_name}")
        return {"client": client, "model_name": model_name, "provider": "deepseek"}

    # ========== Gemini ==========
    elif provider == "gemini":
        # 统一优先 GEMINI_API_KEY；若仅有 GOOGLE_API_KEY 也做兜底
        api_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY 未配置（未找到 GOOGLE_API_KEY 兜底）")

        # 兼容新旧 SDK：优先使用新 SDK（from google import genai），失败则回退到 google.generativeai
        client = None
        try:
            from google import genai  # 新 SDK
            client = genai.Client(api_key=api_key)
            using_new = True
        except Exception:
            import google.generativeai as genai  # 旧 SDK
            genai.configure(api_key=api_key)
            # 旧 SDK 没有 Client 实例的语义，这里用模块本身作为“客户端”占位
            client = genai
            using_new = False

        model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
        suffix = "（新SDK）" if using_new else "（旧SDK）"
        print(f"✅ 已加载 Gemini 模型：{model_name} {suffix}")
        return {"client": client, "model_name": model_name, "provider": "gemini"}

    else:
        raise ValueError(f"❌ 未知 PROVIDER：{provider}（应为 openai / deepseek / gemini）")
