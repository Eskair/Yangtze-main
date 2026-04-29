# -*- coding: utf-8 -*-
"""
汇总类 Markdown 报告的总篇幅上限（按 Python len，与中文「汉字」计数字符数一致）。
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

# 综述/汇总报告正文总字数上限（可环境变量覆盖）
MAX_AGGREGATED_REPORT_CHARS = int(os.getenv("MAX_AGGREGATED_REPORT_CHARS", "2500"))


def clip_markdown_total_chars(markdown_text: str, max_chars: Optional[int] = None) -> Tuple[str, bool]:
    """
    将整份报告压到不超过 max_chars；超出时在尽量自然的换行处截断并追加说明。
    返回 (处理后文本, 是否发生过截断)。
    """
    if max_chars is None:
        max_chars = MAX_AGGREGATED_REPORT_CHARS
    text = markdown_text or ""
    if max_chars <= 0:
        return "", bool(text.strip())

    tr_note = (
        "\n\n---\n\n"
        "> _以下部分因已达到报告总篇幅上限（"
        + str(max_chars)
        + " 字符）而未予展示；完整分项请参见各数据源文件。_"
    )

    if len(text) <= max_chars:
        return text, False

    budget = max_chars - len(tr_note)
    if budget < 120:
        # 说明本身过长或上限过小：兜底硬截断
        cut = text[:max_chars].rstrip()
        if len(cut) >= 2:
            cut = cut[:-1].rstrip() + "…"
        return cut, True

    head = text[:budget]
    snap = head.rfind("\n\n")
    if snap >= int(budget * 0.45):
        head = head[:snap].rstrip()
    else:
        nl = head.rfind("\n")
        if nl >= int(budget * 0.55):
            head = head[:nl].rstrip()
    return head + tr_note, True
