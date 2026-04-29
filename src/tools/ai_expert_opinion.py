# -*- coding: utf-8 -*-
"""
AI Expert Opinion · v4.1
(dimension-first, QA-grounded, general_insights-aware, with local fallback)
--------------------------------------------------------------------
设计目标：
- 严格基于 post_processing_v2 的 metrics.json + final_payload.json（已选中答案）
- 先对五个维度逐一做“带证据 + 行业通识层”的专家点评，再在本地代码中汇总成总体意见
- 不让 LLM 看到任何具体分数，仅提供“强/中/弱”的文字提示，避免分数泄漏
- 默认经 `local_openai` 连本机 OpenAI 兼容服务（也可用 .env 改回云上）；无法调用时自动退化为“纯本地规则版专家评审”（不依赖 LLM）
- 总体意见采用「一段总括 + 分维度 bullet」形式，更适合给人看
"""

import os
import re
import json
import time
import argparse
from pathlib import Path
from run_context import get_context_proposal_id
from datetime import datetime
from typing import Dict, Any, List, Tuple

import requests
from dotenv import load_dotenv
from local_openai import getenv_model, resolve_openai_api_key, resolve_openai_base_url

from post_processing import _gate_reason_zh

# ----------------- 路径与常量 -----------------
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "src" / "data"
REFINED_ROOT = DATA_DIR / "refined_answers"
EXPERT_DIR = DATA_DIR / "expert_reports"

DIM_ORDER = ["team", "objectives", "strategy", "innovation", "feasibility"]
DIM_LABELS_ZH = {
    "team": "团队与治理",
    "objectives": "项目目标",
    "strategy": "实施路径与战略",
    "innovation": "技术与产品创新",
    "feasibility": "资源与可行性"
}

# ----------------- 环境变量 -----------------
load_dotenv(override=True)
PROVIDER = os.getenv("PROVIDER", "openai").strip().lower()
OPENAI_API_KEY = resolve_openai_api_key()
OPENAI_API_BASE = resolve_openai_base_url()
OPENAI_MODEL = getenv_model()
OPENAI_EXPERT_MODEL = os.getenv("OPENAI_EXPERT_MODEL", "").strip() or OPENAI_MODEL
EXPERT_MAX_TOKENS = int(os.getenv("EXPERT_MAX_TOKENS", "1400"))
EXPERT_MAX_RETRIES = int(os.getenv("EXPERT_MAX_RETRIES", "2"))
TIMEOUT_CONNECT = int(os.getenv("HTTP_TIMEOUT_CONNECT", "12"))
TIMEOUT_READ = int(os.getenv("HTTP_TIMEOUT_READ", "60"))


# ----------------- 小工具 -----------------
def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def read_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def detect_latest_pid() -> str:
    """从 refined_answers 下挑选最近更新且包含 postproc/metrics.json 的 pid"""
    if not REFINED_ROOT.exists():
        return ""
    cands: List[Tuple[str, float]] = []
    for d in REFINED_ROOT.iterdir():
        if not d.is_dir():
            continue
        postproc_dir = d / "postproc"
        if (postproc_dir / "metrics.json").exists() and (postproc_dir / "final_payload.json").exists():
            cands.append((d.name, (postproc_dir / "metrics.json").stat().st_mtime))
    cands.sort(key=lambda x: x[1], reverse=True)
    return cands[0][0] if cands else ""


# ----------------- 维度级打分信号 -> 文字提示 -----------------
def _score_hint(v: float) -> str:
    try:
        v = float(v)
    except Exception:
        return "得分信号不明（信息可能不足）"
    if v >= 0.75:
        return "得分偏高，整体表现较强"
    if v >= 0.62:
        return "得分中上，有明显优势，但仍存在可优化空间"
    if v >= 0.50:
        return "得分中等偏弱，存在若干短板或信息缺口"
    if v >= 0.35:
        return "得分偏低，说明该维度存在明显不足或证据有限"
    return "得分很低，属于明显短板，需要重点关注与补救"


def _align_hint(v: float) -> str:
    try:
        v = float(v)
    except Exception:
        return "跨模型一致性信号不明"
    if v >= 0.8:
        return "多模型之间观点高度一致，结论较稳健"
    if v >= 0.6:
        return "多模型之间观点大致一致，少量差异"
    if v >= 0.4:
        return "多模型之间存在较多分歧，需要谨慎解读"
    return "多模型观点差异较大，该维度结论不稳定"


def _drift_hint(v: float) -> str:
    try:
        v = float(v)
    except Exception:
        return "内容漂移信号不明"
    if v <= 0.18:
        return "回答围绕同一核心，内容漂移较低"
    if v <= 0.30:
        return "回答存在一定漂移，但整体仍围绕同一主题"
    if v <= 0.45:
        return "回答存在明显漂移，需要甄别哪些点是稳定共识"
    return "回答漂移程度较高，该维度存在语义不稳定风险"

def _split_keywords(text: str) -> List[str]:
    """
    非严格分词：把 general_insights 里的句子切成若干关键片段，
    过滤掉太短的 token，用来做“是否在问答中被覆盖”的粗匹配。
    """
    if not text:
        return []
    tokens = re.split(r"[，。,.;；、/\\()（）\s]+", text)
    tokens = [t.strip().lower() for t in tokens if len(t.strip()) >= 3]
    return tokens

def build_dim_inputs(metrics: Dict[str, Any],
                     final_payload: Dict[str, Any],
                     max_qas: int = 6,
                     max_answer_chars: int = 800) -> Dict[str, Any]:
    """
    组装传给 LLM 的维度输入：
      - 不包含任何具体分数，只给“强/中/弱”的文字提示
      - 注入 post_processing_v2 新增的 top_evidence_phrases / general_insights
    """
    dim_inputs: Dict[str, Any] = {}
    dim_metrics = metrics.get("dimensions", {}) or {}
    fp_dims = final_payload.get("dimensions", {}) or {}

    for dim in DIM_ORDER:
        m = dim_metrics.get(dim, {}) or {}
        f = fp_dims.get(dim, {}) or {}
        qas = f.get("qas", []) or []

        # 维度级通识经验层（general_insights）
        dim_general_insights = f.get("general_insights") or []
        # 证据短语（来自 post_processing_v2）
        top_evid_phrases = m.get("top_evidence_phrases") or []
        redlined_samples = m.get("redlined_samples") or []

        # 汇总该维度所有问答内容，用于和 general_insights 做粗匹配
        corpus_parts: List[str] = []
        for qa in qas:
            corpus_parts.append((qa.get("q") or ""))
            corpus_parts.append((qa.get("answer") or ""))
            for c in qa.get("claims") or []:
                corpus_parts.append(c)
            for h in qa.get("evidence_hints") or []:
                corpus_parts.append(h)
        corpus_text = " ".join(corpus_parts).lower()

        # 将维度级 general_insights 分成：已部分覆盖 / 明显缺口
        dim_general_insights_covered: List[str] = []
        dim_general_insights_missing: List[str] = []
        for gi in dim_general_insights:
            if not gi:
                continue
            gi_tokens = _split_keywords(gi)
            # 没有有效 token 的，直接当作“缺口提示”（避免误判为 covered）
            if not gi_tokens:
                dim_general_insights_missing.append(gi)
                continue
            hit = any(tok in corpus_text for tok in gi_tokens)
            if hit:
                dim_general_insights_covered.append(gi)
            else:
                dim_general_insights_missing.append(gi)

        samples = []
        for qa in qas[:max_qas]:
            ans = (qa.get("answer") or "").strip()
            if len(ans) > max_answer_chars:
                ans = ans[:max_answer_chars] + "……"
            samples.append({
                "question": (qa.get("q") or "").strip(),
                "answer": ans,
                "key_claims": (qa.get("claims") or [])[:6],
                "evidence_hints": (qa.get("evidence_hints") or [])[:6],
                "provider": qa.get("provider", ""),
                # 逐问通识经验层（仅作行业基准提示，不代表本项目已实现）
                "general_insights": (qa.get("general_insights") or [])[:6],
            })

        dim_inputs[dim] = {
            "dimension": dim,
            "label_zh": DIM_LABELS_ZH.get(dim, dim),
            "score_hint": _score_hint(m.get("avg")),
            "alignment_hint": _align_hint(m.get("avg_alignment")),
            "drift_hint": _drift_hint(m.get("avg_drift")),
            "metric_strength_phrases": (m.get("strengths") or [])[:6],
            "metric_risk_phrases": (m.get("risks") or [])[:6],
            "metric_top_evidence_phrases": top_evid_phrases[:6],
            "metric_redlined_samples": redlined_samples[:6],
            # 维度级通识拆分：已部分在问答中体现 / 基本缺位
            "dim_general_insights": dim_general_insights[:10],
            "dim_general_insights_covered": dim_general_insights_covered[:10],
            "dim_general_insights_missing": dim_general_insights_missing[:10],
            "qa_samples": samples
        }
    return dim_inputs


# ----------------- OpenAI Chat 调用 -----------------
def call_openai_chat(model: str,
                     system_prompt: str,
                     user_payload: Dict[str, Any],
                     temperature: float = 0.25,
                     max_tokens: int = 2600,
                     seed: int = None,
                     max_retries: int = 3,
                     backoff: float = 1.8) -> Dict[str, Any]:
    """
    只负责“按给定参数调用一次 Chat Completions”，不判断 PROVIDER。
    是否调用由上层根据 .env 配置 & 开关决定。
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY 缺失，请在 .env 中配置。")

    url = OPENAI_API_BASE.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    body: Dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
        ],
        "response_format": {"type": "json_object"}
    }
    if "gpt-5" in (model or "").lower():  # updated 21-April-2026
        body["max_completion_tokens"] = int(max_tokens)  # updated 21-April-2026
    else:
        body["max_tokens"] = int(max_tokens)  # updated 21-April-2026
    if seed is not None:
        body["seed"] = seed

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(
                url,
                headers=headers,
                json=body,
                timeout=(TIMEOUT_CONNECT, TIMEOUT_READ)
            )
            if resp.status_code != 200:
                last_err = f"{resp.status_code} - {resp.text[:400]}"
                body_txt = (resp.text or "").lower()
                if resp.status_code in (401, 429) and (
                    "insufficient_quota" in body_txt
                    or "invalid_api_key" in body_txt
                    or "quota" in body_txt
                ):
                    # 明确额度/密钥问题时快速失败，避免末段重复重试消耗时间。
                    raise RuntimeError(f"OpenAI Chat 快速失败：{last_err}")
                time.sleep(backoff ** attempt)
                continue
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return json.loads(content)
        except Exception as e:
            last_err = str(e)
            time.sleep(backoff ** attempt)
    raise RuntimeError(f"OpenAI Chat 调用失败（已重试 {max_retries} 次）：{last_err}")


# ----------------- Prompt 组装 -----------------
def build_dim_system_prompt() -> str:
    return (
        "你是一名长期参与 AI+技术项目评审的专家顾问。"
        "系统会给出五个维度（team/objectives/strategy/innovation/feasibility）的："
        "· 维度中文含义；· 分数强弱/一致性/漂移的‘文字信号’；"
        "· metrics 抽取的 strengths/risks 短语；"
        "· post_processing 聚合的 top_evidence_phrases（只代表证据方向，不是完整证据）；"
        "· 维度级与逐问级 general_insights（行业通识建议，仅作对比基准，并不代表本项目已达成）；"
        "· dim_general_insights_covered：行业通识中，已在当前问答里部分体现的点；"
        "· dim_general_insights_missing：行业通识中，当前问答几乎未覆盖、但在实际评审中通常被视为重要的信息缺口；"
        "· 该维度的部分问答样本（question/answer/claims/evidence_hints/general_insights）。"
        "你的任务：\n"
        "1）对于每个维度，基于问答内容 + strengths/risks 短语 + 证据方向 + general_insights，"
        "   写出：summary / strengths / concerns / recommendations。\n"
        "   - 在 strengths 中，优先结合 dim_general_insights_covered 与具体问答内容，"
        "     明确指出项目已经在哪些方面达到了行业普遍要求。\n"
        "   - 在 concerns 和 recommendations 中，必须至少点名 1–2 条 dim_general_insights_missing，"
        "     解释这些点在行业内通常为什么重要，以及本项目目前材料中为何体现不足，并给出补齐建议。\n"
        "2）每条 strengths/concerns 必须是‘有因有果’的一句话："
        "   可以使用“从……可以看出……从而说明……”“目前材料显示……这一点是亮点/存在不足……”"
        "   “与行业中成熟做法相比，……”等多种句式，明确指出‘为什么好/为什么有风险’。"
        "   不要所有句子都以“因为……”“由于……”“从……来看”这类相同短语开头。\n"
        "3）可以引用问答中的关键信息，但不要编造不存在的机构名称、注册号、具体数据。"
        "4）严禁出现任何题号（如 Q1/Q2）、分数字符（0.71、71% 等）或内部指标名"
        "   （alignment/coverage/authority/drift/对齐/漂移/覆盖/权威/overall_score/置信度 等）。"
        "5）如果某个维度信息明显不足，可以给出 1–2 条保守结论，例如“当前问答中几乎没有提到……，"
        "   因此该维度信息覆盖不足，需要补充材料”。\n"
        "输出必须是严格 JSON，对每个维度都给出：summary（2–4 句）、strengths（3–5 条）、"
        "concerns（3–5 条）、recommendations（3–5 条）。"
        "除 JSON 键名等程序所需字段外，所有自然语言内容必须使用中文；不得输出整段英文论述。"
    )

def build_dim_user_payload(pid: str,
                           dim_inputs: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "pid": pid,
        "task": "dimension_level_expert_opinion",
        "note": "仅输出 dimensions 字段，其余由系统补充。",
        "dimensions": dim_inputs,
        "output_schema_hint": {
            "type": "object",
            "required": ["dimensions"],
            "properties": {
                "dimensions": {
                    "type": "object",
                    "properties": {
                        dim: {
                            "type": "object",
                            "required": ["summary", "strengths", "concerns", "recommendations"],
                            "properties": {
                                "summary": {"type": "string"},
                                "strengths": {"type": "array", "items": {"type": "string"}},
                                "concerns": {"type": "array", "items": {"type": "string"}},
                                "recommendations": {"type": "array", "items": {"type": "string"}}
                            }
                        } for dim in DIM_ORDER
                    }
                }
            }
        }
    }


# ----------------- 文本清洗 & 聚合 -----------------
FORBID_PATTERNS = [
    r"\bQ\d+\b",
    r"\balign(?:ment)?\b",
    r"\bcoverage\b",
    r"\bauth(?:ority)?\b",
    r"\bdrift\b",
    r"对齐", r"漂移", r"覆盖", r"权威",
    r"\boverall[_ ]?score\b",
    r"\bconfidence\b",
    r"\bjaccard\b",
    r"冲突度",
    r"\d+(\.\d+)?\s*%+",
]


def _strip_metric_parentheticals(s: str) -> str:
    """
    安全删除包含内部指标标签的括号片段，避免出现“(=0.00, =1.00)”残片。
    例如：
      "要点充分（auth=0.00, cover=0.00, align=1.00）" -> "要点充分"
    """
    if not s:
        return ""
    metric_keys = (
        "auth", "authority", "cover", "coverage", "align", "alignment",
        "drift", "confidence", "overall_score", "jaccard"
    )
    key_alt = "|".join(metric_keys)
    # 中文全角括号
    s = re.sub(
        rf"（[^）]*\b(?:{key_alt})\b[^）]*）",
        "",
        s,
        flags=re.IGNORECASE,
    )
    # 英文半角括号
    s = re.sub(
        rf"\([^)]*\b(?:{key_alt})\b[^)]*\)",
        "",
        s,
        flags=re.IGNORECASE,
    )
    return s


def clean_text(s: str) -> str:
    s = (s or "").replace("\u0000", "").strip()
    s = _strip_metric_parentheticals(s)
    for pat in FORBID_PATTERNS:
        s = re.sub(pat, "", s, flags=re.IGNORECASE)
    # 清理删词后可能残留的“=0.00”等碎片
    s = re.sub(r"[（(]?\s*=\s*-?\d+(?:\.\d+)?(?:\s*,\s*=\s*-?\d+(?:\.\d+)?)*\s*[)）]?", "", s)
    # 清理重复标点与孤立括号
    s = re.sub(r"[（(]\s*[)）]", "", s)
    s = re.sub(r"[：:]{2,}", "：", s)
    s = re.sub(r"\s{2,}", " ", s)
    s = s.replace("（）", "").replace("()", "")
    return s.strip()


def clean_list(items: List[str]) -> List[str]:
    out: List[str] = []
    for it in items or []:
        t = clean_text(it)
        if t:
            out.append(t)
    return out


def dedup_soft(items: List[str], thresh: float = 0.85) -> List[str]:
    """非常简单的“字符级 Jaccard”去重，避免几乎一样的句子反复出现。"""
    def to_set(x: str):
        return set((x or "").lower())

    uniq: List[str] = []
    for s in items or []:
        keep = True
        a = to_set(s)
        for t in uniq:
            b = to_set(t)
            if not a or not b:
                continue
            j = len(a & b) / len(a | b)
            if j >= thresh:
                keep = False
                break
        if keep:
            uniq.append(s)
    return uniq


def _shorten_sentence(text: str, max_len: int = 120) -> str:
    """用于总体 summary 中的分维度 bullet：取第一句或截断到 max_len。"""
    text = (text or "").strip()
    if not text:
        return ""
    # 按中英文句号/问号/感叹号切分，取第一句
    parts = re.split(r"[。！？!?.]", text)
    for p in parts:
        p = p.strip()
        if p:
            text = p
            break
    if len(text) > max_len:
        return text[:max_len].rstrip() + "……"
    return text


# ----------------- 仅本地的维度级专家点评（LLM 回退） -----------------
def build_local_dim_blocks(metrics: Dict[str, Any],
                           final_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    当无法调用 LLM 时，基于 metrics + final_payload 直接构造维度点评。
    逻辑尽量简洁可解释，不引入额外“猜测”。
    """
    dims_metrics = metrics.get("dimensions", {}) or {}
    fp_dims = final_payload.get("dimensions", {}) or {}

    dim_blocks: Dict[str, Any] = {}

    for dim in DIM_ORDER:
        m = dims_metrics.get(dim, {}) or {}
        f = fp_dims.get(dim, {}) or {}

        score = float(m.get("avg", 0.0) or 0.0)
        align = float(m.get("avg_alignment", 0.0) or 0.0)
        drift = float(m.get("avg_drift", 0.0) or 0.0)
        strengths_phr = (m.get("strengths") or [])[:5]
        risks_phr = (m.get("risks") or [])[:5]
        dim_gi = (f.get("general_insights") or [])[:8]

        label = DIM_LABELS_ZH.get(dim, dim)

        summary_parts: List[str] = []
        if strengths_phr:
            summary_parts.append(
                f"{label} 维度中，自动评估识别出若干优势点，例如：" +
                "；".join(strengths_phr[:2])
            )
        if risks_phr:
            summary_parts.append(
                "同时也暴露出一些潜在问题或风险，例如：" +
                "；".join(risks_phr[:2])
            )
        if not summary_parts:
            summary_parts.append(
                f"当前问答与自动评估中，关于“{label}”维度的有效信息有限，结论仅供参考，建议补充更详细的事实和量化指标。"
            )
        summary = " ".join(summary_parts)

        # 优势：优先使用 metrics.strengths；若为空，用少量 general_insights 补位
        strengths_out = strengths_phr[:]
        if not strengths_out and dim_gi:
            strengths_out = [
                f"从行业经验看，本维度若能达到以下实践将显著加分：{dim_gi[0]}"
            ]

        # 风险：直接用 metrics.risks
        concerns_out = risks_phr[:]

        # 建议：优先用 general_insights，若为空则给一条通用建议
        recs_out: List[str] = []
        for g in dim_gi:
            recs_out.append(g)
        if risks_phr and not dim_gi:
            recs_out.append(
                "针对上述风险，建议在后续版本中补充更具体的实施计划、里程碑与量化指标，以便评审。"
            )
        if not recs_out:
            recs_out.append(
                f"建议围绕“{label}”维度，系统梳理团队经验、资源保障和实施路径，并结合行业最佳实践补齐信息。"
            )

        dim_blocks[dim] = {
            "score_echo": score,
            "alignment_echo": align,
            "drift_echo": drift,
            "summary": summary,
            "strengths": strengths_out,
            "concerns": concerns_out,
            "recommendations": recs_out
        }

    return dim_blocks


# ----------------- 总体意见（本地） -----------------
def build_overall_from_dims(dim_blocks: Dict[str, Any],
                            metrics_overall: Dict[str, Any],
                            metrics_dims: Dict[str, Any]) -> Dict[str, Any]:
    """
    从各维度点评 + 分数，构造总体意见（不再调用 LLM）。

    优化点：
    - verdict 阈值略放宽，让中间地带更多落在 HOLD，而不是一刀切 NO-GO；
    - 总体 summary 采用“一段总括 + 分维度 bullet”的结构；
    - 配合后续 markdown 渲染，读起来更像人写的评审意见。
    """
    overall_score = float(metrics_overall.get("overall_score", 0.0) or 0.0)
    overall_conf = float(metrics_overall.get("overall_confidence", 0.0) or 0.0)

    quality_gates = metrics_overall.get("quality_gates", {}) or {}
    gate_pass = bool(quality_gates.get("pass", True))

    # 1) 判定 verdict
    def verdict_rule(score: float, conf: float,
                     dims: Dict[str, Any],
                     gate_ok: bool,
                     gate_detail: Dict[str, Any]) -> (str, str):
        inv = float(dims.get("innovation", {}).get("avg", 1.0) or 1.0)
        fea = float(dims.get("feasibility", {}).get("avg", 1.0) or 1.0)

        # 0) 质量闸门优先：数据质量不足时不输出硬性 go/no-go
        if not gate_ok:
            hard = gate_detail.get("fail_reasons") or gate_detail.get("reasons") or []
            reasons = ", ".join(hard) or "quality_gate_failed"
            return (
                "INSUFFICIENT_EVIDENCE",
                f"当前轮次的数据质量闸门未通过（{reasons}），证据充分性不足，不建议直接给出 GO/NO-GO。"
            )

        # ① 明确 GO：得分 + 信心都比较稳
        if score >= 0.62 and conf >= 0.65:
            return "GO", "综合得分与信心度均处于较高区间，关键维度表现扎实，整体风险可控，适合推进。"

        # ② 明确 NO-GO：整体很低，或关键维度严重偏弱
        if score < 0.40 or inv < 0.30 or fea < 0.30:
            return (
                "NO-GO",
                "总体得分或关键维度（尤其是创新/可行性）处于明显偏低区间，"
                "关键信息缺失或短板较多，目前不宜在本轮直接立项，建议补充材料后再行评估。"
            )

        # ③ 剩下全部归为 HOLD：有潜力，但证据/信息不够
        return (
            "HOLD",
            "项目处于中间地带，一方面具备一定亮点和潜力，另一方面在若干关键维度上信息仍不充分，"
            "建议在补充必要材料和澄清关键风险后，再做更明确的 go/no-go 决策。"
        )

    verdict, verdict_reason = verdict_rule(overall_score, overall_conf, metrics_dims, gate_pass, quality_gates)

    # 2) 总体 summary：一段总括 + 分维度 bullet
    # 2.1 总括句随 verdict 变化
    if verdict == "GO":
        head = (
            "整体来看，该项目在当前轮次的综合表现较为扎实：关键假设相对清晰、实施路径具备一定可操作性，"
            "在风险可控的前提下具备推进价值。"
        )
    elif verdict == "HOLD":
        head = (
            "整体来看，该项目在技术和应用场景上体现出一定潜力，但目前关键信息仍有缺口，"
            "更适合作为“待补充材料后再议”的候选项目，而非直接进入大规模投入阶段。"
        )
    elif verdict == "NO-GO":
        head = (
            "整体来看，该项目在技术构想或应用方向上虽有亮点，但现有材料无法支撑稳健的风险收益判断，"
            "短板和不确定性占比较高，当前不宜在本轮直接立项。"
        )
    else:  # INSUFFICIENT_EVIDENCE
        head = (
            "整体来看，本轮结果受数据质量与证据充分性限制，当前证据不足以支持稳健的 go/no-go 结论，"
            "建议先完成关键材料补齐与数据质量修复后再评审。"
        )

    # 2.2 分维度 bullet：从各维度 summary 中提取第一句精简回顾
    dim_snippets: List[str] = []
    for dim in DIM_ORDER:
        blk = dim_blocks.get(dim, {}) or {}
        dim_sum = (blk.get("summary") or "").strip()
        if not dim_sum:
            continue
        label = DIM_LABELS_ZH.get(dim, dim)
        short = _shorten_sentence(dim_sum, max_len=140)
        if not short:
            continue
        dim_snippets.append(f"- {label}：{short}")

    lines: List[str] = [head]
    if dim_snippets:
        lines.append("分维度来看：")
        lines.extend(dim_snippets)

    summary_text = "\n".join(lines)

    # 3) 从各维度 strengths/concerns 中抽取总体 key_strengths/key_risks
    key_strengths: List[str] = []
    key_risks: List[str] = []
    for dim in DIM_ORDER:
        blk = dim_blocks.get(dim, {}) or {}
        label = DIM_LABELS_ZH.get(dim, dim)
        for s in (blk.get("strengths") or [])[:2]:
            key_strengths.append(f"【{label}】{s}")
        for r in (blk.get("concerns") or [])[:2]:
            key_risks.append(f"【{label}】{r}")

    key_strengths = dedup_soft(clean_list(key_strengths))[:6]
    key_risks = dedup_soft(clean_list(key_risks))[:6]

    # 4) 总体 recommendations：从各维度 recommendations 抽样
    recs: List[str] = []
    for dim in DIM_ORDER:
        blk = dim_blocks.get(dim, {}) or {}
        label = DIM_LABELS_ZH.get(dim, dim)
        for r in (blk.get("recommendations") or [])[:2]:
            recs.append(f"【{label}】{r}")
    recs = dedup_soft(clean_list(recs))[:8]

    return {
        "summary": summary_text,
        "overall_score_echo": overall_score,
        "confidence_echo": overall_conf,
        "key_strengths": key_strengths,
        "key_risks": key_risks,
        "recommendations": recs,
        "verdict": verdict,
        "basis": [
            f"结论依据：{verdict_reason}",
            "结论完全基于已选中问答结果与自动评分信号，未引入外部资料。"
        ],
        "quality_gate": quality_gates
    }


# ----------------- Markdown 渲染 -----------------
def _bar(v: float, n: int = 20) -> str:
    try:
        v = float(v)
    except Exception:
        v = 0.0
    v = max(0.0, min(1.0, v))
    k = int(round(v * n))
    return "█" * k + "░" * (n - k)


def render_markdown(opinion: Dict[str, Any]) -> str:
    meta = opinion.get("meta", {}) or {}
    overall = opinion.get("overall_opinion", {}) or {}
    dims = opinion.get("dimensions", {}) or {}
    scoring = opinion.get("scoring_explainer", {}) or {}
    metrics_path = (meta.get("sources") or {}).get("metrics_path", "")

    lines: List[str] = []
    lines.append(f"# AI 专家评审 · {meta.get('pid', '')}")
    lines.append("")
    lines.append(f"- 生成时间：{meta.get('generated_at', '')}")
    lines.append(f"- 模式：{meta.get('mode', '')}")
    lines.append(f"- 模型/引擎：{meta.get('model', '')}（provider={meta.get('provider', '')}）")
    lines.append("")

    # 总体
    lines.append("## 总体意见")
    lines.append(f"- 综合评分（回显）：{overall.get('overall_score_echo', 0.0):.3f}  {_bar(overall.get('overall_score_echo', 0.0))}")
    lines.append(f"- 综合信心度（回显）：{overall.get('confidence_echo', 0.0):.3f}  {_bar(overall.get('confidence_echo', 0.0))}")
    lines.append("")
    if overall.get("summary"):
        lines.append(overall["summary"])
        lines.append("")
    if overall.get("key_strengths"):
        lines.append("**项目优势**")
        for s in overall["key_strengths"]:
            lines.append(f"- {s}")
        lines.append("")
    if overall.get("key_risks"):
        lines.append("**项目不足/潜在风险**")
        for r in overall["key_risks"]:
            lines.append(f"- {r}")
        lines.append("")
    if overall.get("recommendations"):
        lines.append("**总体建议**")
        for r in overall["recommendations"]:
            lines.append(f"- {r}")
        lines.append("")
    qg = overall.get("quality_gate") or {}
    if qg:
        lines.append("**质量闸门回显**")
        hard = qg.get("fail_reasons") or qg.get("reasons") or []
        ok = bool(qg.get("pass")) and not hard
        lines.append(f"- 闸门状态：{'PASS' if ok else 'FAIL'}")
        lines.append(f"- 选中覆盖率：{float(qg.get('selected_coverage_ratio', 0.0)):.1%}")
        lines.append(f"- 一致性（consistency_ratio）：{float(qg.get('consistency_ratio', 0.0)):.3f}")
        lines.append(f"- 解析成功率（估计）：{float(qg.get('parse_success_ratio', 0.0)):.1%}")
        if hard:
            lines.append(
                f"- 闸门失败原因：{', '.join(_gate_reason_zh(x) for x in hard)}"
            )
        ws = qg.get("warnings") or []
        if ws:
            lines.append(
                f"- 质量告警（不否决结论）：{', '.join(_gate_reason_zh(w) for w in ws)}"
            )
        lines.append("")
    if overall.get("verdict"):
        lines.append(f"**总体结论（verdict）**：{overall['verdict']}")
        lines.append("")
    if overall.get("basis"):
        lines.append("**结论依据（系统自动生成）**")
        for b in overall["basis"]:
            lines.append(f"- {b}")
        lines.append("")

    # 维度表
    lines.append("## 各维度评分一览")
    lines.append("")
    lines.append("| 维度 | 分数 |")
    lines.append("|---|---:|")
    for dim in DIM_ORDER:
        blk = dims.get(dim, {}) or {}
        lines.append(f"| {dim} | {blk.get('score_echo', 0.0):.3f} |")
    lines.append("")

    # 分维度详情
    lines.append("## 分维度专家点评")
    lines.append("")
    for dim in DIM_ORDER:
        label = DIM_LABELS_ZH.get(dim, dim)
        blk = dims.get(dim, {}) or {}
        lines.append(f"### {label}（{dim}）")
        lines.append(f"- 评分回显：{blk.get('score_echo', 0.0):.3f}  {_bar(blk.get('score_echo', 0.0))}")
        lines.append("")
        if blk.get("summary"):
            lines.append(blk["summary"])
            lines.append("")
        if blk.get("strengths"):
            lines.append("**优势**")
            for s in blk["strengths"]:
                lines.append(f"- {s}")
            lines.append("")
        if blk.get("concerns"):
            lines.append("**问题/风险**")
            for r in blk["concerns"]:
                lines.append(f"- {r}")
            lines.append("")
        if blk.get("recommendations"):
            lines.append("**改进建议**")
            for r in blk["recommendations"]:
                lines.append(f"- {r}")
            lines.append("")

    # 简单回显评分配置（方便审计）
    if scoring:
        lines.append("## 评分规则回显（来自 post_processing 配置）")
        lines.append(f"- 一致性权重 consistency_weight：{scoring.get('consistency_weight', 0.0):.2f}")
        if scoring.get("dimension_weight"):
            dw = scoring["dimension_weight"]
            order_str = ", ".join([f"{d}:{dw.get(d, 0.0):.2f}" for d in DIM_ORDER if d in dw])
            lines.append(f"- 维度权重：{order_str}")
        lines.append("")

    if metrics_path:
        lines.append("## 溯源")
        lines.append(f"- metrics.json：{metrics_path}")
        lines.append("")

    return "\n".join(lines)


# ----------------- 主流程 -----------------
def main():
    ap = argparse.ArgumentParser(description="Generate AI expert opinion (dimension-first, QA-grounded).")
    ap.add_argument("--pid", type=str, default="", help="提案 ID；缺省则自动选择最新项目")
    ap.add_argument("--model", type=str, default=OPENAI_EXPERT_MODEL, help="OpenAI 模型名（默认使用独立专家模型）")
    ap.add_argument("--dry_run", action="store_true", help="仅生成 prompt，不调用 LLM")
    ap.add_argument("--no_markdown", action="store_true", help="不输出 Markdown，仅 JSON")
    ap.add_argument("--force_local", action="store_true", help="强制使用本地规则版，不调用 LLM（用于调试）")
    args = ap.parse_args()

    pid = args.pid.strip() or get_context_proposal_id() or detect_latest_pid()
    if not pid:
        raise RuntimeError("未检测到可用项目（refined_answers 下无 postproc/metrics.json + final_payload.json）。")

    postproc_dir = REFINED_ROOT / pid / "postproc"
    metrics_path = postproc_dir / "metrics.json"
    payload_path = postproc_dir / "final_payload.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"未找到 metrics.json：{metrics_path}")
    if not payload_path.exists():
        raise FileNotFoundError(f"未找到 final_payload.json：{payload_path}")

    metrics = read_json(metrics_path)
    final_payload = read_json(payload_path)

    dim_inputs = build_dim_inputs(metrics, final_payload)

    out_dir = EXPERT_DIR / pid
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = out_dir / "ai_expert_opinion.prompt.json"
    json_path = out_dir / "ai_expert_opinion.json"
    md_path = out_dir / "ai_expert_opinion.md"

    # 先保存 prompt（即便是本地版，也方便调试看看输入长什么样）
    system_prompt = build_dim_system_prompt()
    user_payload = build_dim_user_payload(pid, dim_inputs)
    write_json(prompt_path, {
        "system": system_prompt,
        "user": user_payload
    })

    # =========== 维度级点评：优先 LLM，失败则回退本地 ===========
    dim_op_raw: Dict[str, Any] = {}
    used_model = ""
    used_mode = ""

    use_llm = (not args.force_local) and (PROVIDER == "openai") and bool(OPENAI_API_KEY)

    if args.dry_run:
        print(f"📝 已导出 prompt -> {prompt_path}")
        return

    if use_llm:
        try:
            resp = call_openai_chat(
                model=args.model,
                system_prompt=system_prompt,
                user_payload=user_payload,
                temperature=0.25,
                max_tokens=max(800, EXPERT_MAX_TOKENS),
                seed=None,
                max_retries=max(1, EXPERT_MAX_RETRIES),
            )
            if "dimensions" not in resp:
                raise RuntimeError("LLM 返回 JSON 中缺少 'dimensions' 字段")
            dim_op_raw = resp["dimensions"] or {}
            used_model = args.model
            used_mode = "llm"
        except Exception as e:
            print(f"⚠️ LLM 生成维度点评失败，将启用纯本地规则回退：{e}")
            dim_op_raw = {}
            used_model = "local_rules"
            used_mode = "local_fallback"
    else:
        used_model = "local_rules"
        used_mode = "local_forced"

    if not dim_op_raw:
        # 纯本地规则版
        dim_op_raw = build_local_dim_blocks(metrics, final_payload)

    # 兜底：保证所有维度都有字段 & 文本清洗/去重 + 收紧条数
    cleaned_dims: Dict[str, Any] = {}
    metrics_dims = metrics.get("dimensions", {}) or {}
    for dim in DIM_ORDER:
        blk = dim_op_raw.get(dim, {}) or {}
        m = metrics_dims.get(dim, {}) or {}

        strengths = dedup_soft(clean_list(blk.get("strengths") or []))[:3]
        concerns = dedup_soft(clean_list(blk.get("concerns") or []))[:3]
        recs = dedup_soft(clean_list(blk.get("recommendations") or []))[:4]
        summary = clean_text(blk.get("summary") or "")

        # 如果维度几乎没有信息，补一条兜底 summary
        if not summary and not strengths and not concerns:
            label = DIM_LABELS_ZH.get(dim, dim)
            summary = (
                f"当前关于“{label}”维度的有效问答与证据信号非常有限，结论不稳定，"
                f"建议项目方在后续版本中补充该维度的核心事实、量化指标与实施细节。"
            )

        cleaned_dims[dim] = {
            "score_echo": float(m.get("avg", 0.0) or 0.0),
            "alignment_echo": float(m.get("avg_alignment", 0.0) or 0.0),
            "drift_echo": float(m.get("avg_drift", 0.0) or 0.0),
            "summary": summary,
            "strengths": strengths,
            "concerns": concerns,
            "recommendations": recs
        }

    overall_block = build_overall_from_dims(
        dim_blocks=cleaned_dims,
        metrics_overall=metrics.get("overall", {}) or {},
        metrics_dims=metrics_dims
    )

    # 拼接最终 JSON
    opinion: Dict[str, Any] = {
        "meta": {
            "pid": pid,
            "generated_at": now_str(),
            "model": used_model,
            "mode": used_mode,
            "provider": PROVIDER,
            "sources": {
                "metrics_path": str(metrics_path),
                "final_payload_path": str(payload_path)
            }
        },
        "overall_opinion": overall_block,
        "dimensions": cleaned_dims,
        "scoring_explainer": {
            # 只回显最关键的几项，便于报告中解释“分是怎么算出来的”
            "consistency_weight": float((metrics.get("config_used") or {}).get("consistency_weight", 0.20) or 0.20),
            "dimension_weight": {
                k: float(v) for k, v in ((metrics.get("config_used") or {}).get("dimension_weight") or {}).items()
            }
        }
    }

    write_json(json_path, opinion)
    if not args.no_markdown:
        md_text = render_markdown(opinion)
        md_path.write_text(md_text, encoding="utf-8")

    print(f"✅ ai_expert_opinion.json -> {json_path}")
    if not args.no_markdown:
        print(f"✅ ai_expert_opinion.md   -> {md_path}")
    print(f"🎯 专家评审生成完成（{used_mode} 模式）。")


if __name__ == "__main__":
    main()
