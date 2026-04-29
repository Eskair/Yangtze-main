# -*- coding: utf-8 -*-
"""
llm_answering.py · Proposal-aware LLM Answering（默认仅用 ChatGPT / OpenAI，无 Web）

答题模型：
- **默认**：`local_openai` 约定的本机端点（`127.0.0.1:11434`，模型 `llama3.2:latest`）；不写 .env 即连 Ollama 风格兼容 API；每条一套结构化回答；改云上请设 OPENAI_*。
- **可选**：在 .env 中设置 ENABLE_DEEPSEEK_IN_ANSWERING=1 且配置 DEEPSEEK_API_KEY 时，额外跑 DeepSeek，
  merged 时每题至多两套候选再由后处理打分择一。
- **全链路离线 / 本机推理**：`extract_facts_by_chunk`、`build_dimensions_from_facts`、`generate_questions`、
  本脚本与 `ai_expert_opinion` 均通过 `local_openai.py` 使用同一套 `OPENAI_API_BASE` / `OPENAI_MODEL`（见该模块说明），
  指向 Ollama 等即可在无公网 API 账单下跑通 canonical 流水线（质量取决于本机模型与硬件）。

新版职责：
- 基于「维度抽取管线」生成的提案事实（dimensions_from_facts）+ 生成的问题集，
  为每个维度的问题生成“强关联该提案”的结构化回答。
- 不再依赖外部 Web 检索；只使用：
    • 提案维度事实（summary/key_points/risks/mitigations/numbers 等）
    • LLM 自身通识做解释，但禁止脑补新实验/新数字/新机构
- 输出结构保持兼容：
    data/refined_answers/{pid}/all_refined_items.json
    data/refined_answers/{pid}/chatgpt_raw.json
    （若启用第二模型）data/refined_answers/{pid}/deepseek_raw.json

本版新增：
- 明确要求回答中包含“行业基准对比”和“常见坑与证据要求”两类内容；
- 这些通识部分必须写成“行业普遍情况/一般建议”，不能写成项目已经达成的事实；
- 新增字段 general_insights：专门装“行业基准 / 常见坑 / 证据要求”这类专家经验层；
- 修复 answer 分点格式问题，避免出现 “1. 1. xxx” 这种双重编号。
"""

import os
import sys
import json
import time
import re
import random
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

from dotenv import load_dotenv
from run_context import get_context_proposal_id
from local_openai import maybe_probe_local_inference
from generate_questions import clip_question_text

# ========== 环境 & 路径 ==========
load_dotenv(override=True)
# 是否在答题流水线中启用 DeepSeek（第二模型）。默认关闭，与「仅 ChatGPT」部署对齐。
ENABLE_DEEPSEEK_IN_ANSWERING = os.getenv("ENABLE_DEEPSEEK_IN_ANSWERING", "0").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)

ROOT = Path(__file__).resolve().parents[1]  # .../src
DATA_DIR = ROOT / "data"
EXTRACTED_DIR = DATA_DIR / "extracted"
PARSED_DIR = DATA_DIR / "parsed"
CONFIG_QS_DEFAULT = DATA_DIR / "config" / "question_sets" / "generated_questions.json"
OUT_REFINED = DATA_DIR / "refined_answers"

DIM_ORDER = ["team", "objectives", "strategy", "innovation", "feasibility"]

# ========== SDK ==========
try:
    from openai import OpenAI as OpenAIClient
except Exception:  # pragma: no cover
    OpenAIClient = None

# ========== 单次综合回答标识 & provider 能力 ==========
# 主流程：每题只调用一次模型；用一条提示同时覆盖综合/风险/落地等侧重点（原为 default/risk/implementation 多分候选）。
COMBINED_VARIANT_ID = "combined"
TEMP_BY_VARIANT = {
    COMBINED_VARIANT_ID: 0.28,
    "default": 0.25,
    "risk": 0.35,
    "implementation": 0.30,
}

PROVIDER_CAPS = {
    # batch_ok 可被 LLM_USE_BATCH 覆盖：默认逐题（JSON 更稳、更易修复）；设为 1 可恢复多题同批。
    "openai": {"json_mode": True, "batch_ok": True},
    "deepseek": {"json_mode": False, "batch_ok": False},
}


def provider_caps(provider: str) -> Dict[str, Any]:
    caps = dict(PROVIDER_CAPS.get(provider, {"json_mode": True, "batch_ok": True}))
    if provider == "openai":
        use_batch = os.getenv("LLM_USE_BATCH", "0").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        caps["batch_ok"] = bool(use_batch)
    return caps


CONF_MIN, CONF_MAX = 0.50, 0.92
MAX_LIST_LEN = 10
# 与 generate_questions 的 MAX_QUESTION_CHARS 默认对齐；override 用环境变量。
MAX_PROMPT_QUESTION_CHARS = int(os.getenv("MAX_PROMPT_QUESTION_CHARS", "60"))
# 每条分点 / claims / evidence_hints / general_insights 各自的字符上限（Python len）。
# 主约束：由答题 LLM 在约此字数内写「语义完整」的一条；程序侧仅在超长时按句读兜底截断（见 _clip_answer_to_max_chars）。
# 可用 MAX_REVIEW_CHARS 或 MAX_ANSWER_CHARS 覆盖。
MAX_REVIEW_CHARS = int(os.getenv("MAX_REVIEW_CHARS") or os.getenv("MAX_ANSWER_CHARS") or "150")
MAX_ANSWER_CHARS = MAX_REVIEW_CHARS


def _clip_answer_to_max_chars(text: str, max_chars: int) -> str:
    """将单行正文压到 max_chars 以内；超长时优先在句读处截断，避免半句话（兜底用，不应替代模型自控篇幅）。"""
    s = (text or "").strip()
    if max_chars <= 0:
        return ""
    if len(s) <= max_chars:
        return s
    cut = s[:max_chars].rstrip()
    for sep in ("。", ".", "？", "?", "！", "!", "；", ";", "，", ",", "\n"):
        i = cut.rfind(sep)
        if i >= max(20, max_chars // 3):
            return cut[: i + 1].strip()
    if len(cut) >= max_chars:
        return cut[: max_chars - 1].rstrip() + "…"
    return cut


def _clip_plain_to_max_chars(text: str, max_chars: int) -> str:
    """claims 等列表的单条：与分点正文共用句读优先截断。"""
    return _clip_answer_to_max_chars(text, max_chars)


def _clip_each_list_item_max_chars(items: List[str], max_chars: int) -> List[str]:
    """claims / hints / insights：每一条单独不得超过 max_chars。"""
    out: List[str] = []
    for s in items or []:
        t = str(s or "").strip()
        if not t:
            continue
        out.append(_clip_plain_to_max_chars(t, max_chars))
    return out


def _clip_numbered_answer_each_bullet(answer_block: str, max_chars: int) -> str:
    """answer：每一行「N. 正文」中【正文部分】单独不得超过 max_chars（行首编号不计入上限）。"""
    if max_chars <= 0:
        return ""
    raw = (answer_block or "").replace("\r\n", "\n").strip()
    if not raw:
        return ""
    out: List[str] = []
    for line in raw.split("\n"):
        s = line.strip()
        if not s:
            continue
        m = re.match(r"^(\d+)([\.\．]\s*)(.*)$", s)
        if m:
            prefix = f"{m.group(1)}{m.group(2)}"
            body = (m.group(3) or "").strip()
            clipped = _clip_answer_to_max_chars(body, max_chars) if body else ""
            out.append(prefix + clipped)
        else:
            out.append(_clip_answer_to_max_chars(s, max_chars))
    return "\n".join(out)


# ========== 工具函数 ==========
def read_json(p: Path) -> Any:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def detect_latest_pid() -> str:
    if not EXTRACTED_DIR.exists():
        return "unknown"
    cands = [d for d in EXTRACTED_DIR.iterdir() if d.is_dir()]
    if not cands:
        return "unknown"
    cands.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return cands[0].name

def _flatten_list_field(block: Dict[str, Any], keys: List[str], limit: int = 12) -> List[str]:
    items: List[str] = []
    for k in keys:
        v = block.get(k)
        if not v:
            continue
        if isinstance(v, str):
            items.extend([x.strip() for x in re.split(r"[;\n]", v) if x.strip()])
        elif isinstance(v, list):
            for x in v:
                if isinstance(x, str):
                    s = x.strip()
                    if s:
                        items.append(s)
    uniq, seen = [], set()
    for x in items:
        key = x.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        uniq.append(key)
        if len(uniq) >= limit:
            break
    return uniq


def _build_dim_context_text(dim: str, block: Dict[str, Any]) -> str:
    """
    把从 facts 管线来的一个维度 block 转成适合放进 prompt 的 context 文本。
    控制长度，优先 summary / key_points / risks / mitigations / numbers。
    """
    if not isinstance(block, dict):
        block = {}

    summary = str(block.get("summary") or "").strip()
    key_points = _flatten_list_field(block, ["key_points", "keypoints", "key_facts", "bullets"], limit=10)
    risks = _flatten_list_field(block, ["risks", "risk_points"], limit=8)
    mitigations = _flatten_list_field(block, ["mitigations", "mitigation_points"], limit=8)
    numbers = _flatten_list_field(block, ["numbers", "key_numbers"], limit=8)

    parts: List[str] = []
    if summary:
        parts.append(f"【{dim} 概览】{summary}")
    if key_points:
        parts.append("【关键要点】" + "；".join(key_points))
    if risks:
        parts.append("【主要风险/不确定性】" + "；".join(risks))
    if mitigations:
        parts.append("【已有缓解措施】" + "；".join(mitigations))
    if numbers:
        parts.append("【关键量化信息】" + "；".join(numbers))

    text = "\n".join(parts)
    if len(text) > 2000:
        text = text[:2000]
    return text


def load_dimension_context(pid: str, dim_file: Optional[Path]) -> Dict[str, str]:
    """
    返回：{dim: context_text}
    若找不到文件或结构异常，所有维度返回空字符串（模型会退化成通识回答）
    """
    ctx: Dict[str, str] = {d: "" for d in DIM_ORDER}
    if dim_file is None or not dim_file.exists():
        return ctx

    try:
        raw = read_json(dim_file)
    except Exception:
        return ctx

    if isinstance(raw, dict) and "dimensions" in raw and isinstance(raw["dimensions"], dict):
        root = raw["dimensions"]
    else:
        root = raw if isinstance(raw, dict) else {}

    for dim in DIM_ORDER:
        block = root.get(dim) or {}
        ctx[dim] = _build_dim_context_text(dim, block)
    return ctx


# ========== 问题集 & 监管提示 ==========
def get_q_list(block: Any) -> List[str]:
    def _clip_q(s: str) -> str:
        return clip_question_text(str(s or "").strip(), MAX_PROMPT_QUESTION_CHARS, "zh")

    if isinstance(block, dict) and isinstance(block.get("questions"), list):
        return [_clip_q(q) for q in block["questions"] if isinstance(q, str) and _clip_q(q)]
    if isinstance(block, list):
        return [_clip_q(q) for q in block if isinstance(q, str) and _clip_q(q)]
    return []


def _load_reg_hints(qs_cfg: Dict[str, Any], dim: str, limit: int = 8) -> List[str]:
    """
    从问题集里抓一点监管/术语 hints，但只作为“可选方向提示”，不强依赖。
    """
    block = qs_cfg.get(dim, {}) or {}
    hints = block.get("search_hints") or block.get("reg_hints") or []
    out: List[str] = []
    if isinstance(hints, list):
        for h in hints:
            if not isinstance(h, str):
                continue
            s = h.strip()
            if not s:
                continue
            out.append(s)
    uniq, seen = [], set()
    for x in out:
        if x in seen:
            continue
        seen.add(x)
        uniq.append(x)
        if len(uniq) >= limit:
            break
    return uniq

# ========== Prompt 相关 ==========
def detect_domain_from_text(text: str) -> str:
    t = (text or "").lower()
    biomed_hits = sum(
        1
        for k in [
            "biomed", "biomedical", "drug", "clinical", "trial", "hospital",
            "patient", "therapy", "fda", "ema", "药", "临床", "患者", "治疗", "医疗"
        ]
        if k in t
    )
    auto_hits = sum(
        1
        for k in [
            "autonomous", "vehicle", "adas", "driving", "lidar", "radar",
            "perception", "自动驾驶", "智驾", "车路协同", "感知", "规控"
        ]
        if k in t
    )
    if auto_hits >= max(2, biomed_hits + 1):
        return "autonomous_systems"
    if biomed_hits >= 2:
        return "biomed"
    return "generic_tech"


def build_system_prompt(domain: str) -> str:
    role_line = {
        "biomed": "你是一名严格的通用技术项目评审专家。",
        "autonomous_systems": "你是一名严格的通用技术项目评审专家。",
        "generic_tech": "你是一名严格的通用技术项目评审专家。",
    }.get(domain, "你是一名严格的通用技术项目评审专家。")
    return (
        role_line
        + "你的任务是：在完整阅读给定的【提案事实】之后，围绕问题进行“与该提案强相关”的专业分析。"
        + "原则："
        + "1）必须优先基于【提案事实】给出结论；"
        + "2）不得发明提案中未出现的新试验、新数据、新机构或具体数字；"
        + "3）当你在【提案事实】中找不到某一类信息时，只能说“当前材料中未看到关于 X 的具体说明”，"
        + "   并优先写成“已有 A/B，但在 C/D 方面细节不足”；严禁使用“完全没有分析”“未进行任何评估”"
        + "   “未提供任何信息”等绝对化表述；"
        + "4）可以使用行业通识解释这些事实的意义，但不得虚构具体注册号/编号/专利号/精确样本量等细节；"
        + "5）回答必须结构化、条理清晰，适合作为专家评审报告的组成部分；"
        + "6）除 JSON 结构所需的英文字段名外，所有自然语言内容必须使用中文；专有名词可保留必要外文缩写；"
        + "7）你需要在回答中补充“行业基准对比”和“类似项目常见的坑与证据要求”等通识内容，"
        + "   但这些通识部分必须明确表述为“行业普遍情况/一般建议”，不得写成项目已经达成的事实；"
        + f"8）主字段 answer：凡采用「1. … / 2. …」分点，每一行中分点正文（不含行首编号）建议在约 {MAX_REVIEW_CHARS} 字内写成一小段**语义完整**的表述（自然收束，勿写到半截就停）；"
        + f"9）字段 claims、evidence_hints、general_insights：数组中每一条字符串同样不超过 {MAX_REVIEW_CHARS} 字，且须各自成句、意思完整；"
        + "   宁可删修饰、合并表述，也不要依赖程序事后截断来「凑字数」。"
    )

def _schema_structured() -> str:
    return f"""
请严格返回 JSON 对象，键名固定（键名保持英文以便程序解析；所有字符串值必须为中文）：
{{
  "answer": "分点主回答（中文；每一行分点正文各自在约 {MAX_REVIEW_CHARS} 字内写完整再收束；行首「1.」编号不计入该上限）",
  "claims": ["每条关键可验证结论：不超过 {MAX_REVIEW_CHARS} 字且语义完整"],
  "evidence_hints": ["每条支撑/核查线索：不超过 {MAX_REVIEW_CHARS} 字且语义完整"],
  "general_insights": ["每条行业通识短句：不超过 {MAX_REVIEW_CHARS} 字且语义完整"],
  "topic_tags": ["维度内的小主题标签（中文）"],
  "confidence": 0.0,
  "caveats": "限制/注意事项（若无可空）"
}}
""".strip()


def _unified_variant_instructions() -> str:
    """
    单次回答中合并原「综合 default / 风险 risk / 落地 implementation」的评审侧重（只输出一套 JSON）。
    """
    mc = MAX_REVIEW_CHARS
    return f"""
单次综合评审视角（请在本题只生成一份结构化回答；不要分拆成多套互不关联的答案）：
- **综合**：结合该维度「当前方案/优势/问题/建议」做整体评价；区分提案事实 vs 行业通识（通识须用「通常/一般而言」等措辞）。
- **风险**：点名不确定性、信息缺口以及潜在合规/技术风险线索；必要时指出「现有材料未见 X」而非绝对否定。
- **落地**：在尊重提案当前状态前提下，简述可执行的下一步或可补充的证据类型；不得假设尚未完成的工作已达成。
- **篇幅**：answer 宜用 2–4 条短分点，使上述三方面在简练篇幅内均被照顾到（无需机械地各占一条）；每条分点正文≤{mc} 字且须**一整句意思完整**（超长时优先压缩措辞，勿写到一半）；
  claims / evidence_hints / general_insights 亦每条≤{mc} 字且各自完整；在 general_insights 末条可附简短免责声明式表述（仍为通识）。
""".strip()

def build_single_prompt(
    dimension: str,
    question: str,
    proposal_context: str,
    reg_hints: List[str],
    variant_id: str,
) -> str:
    reg_txt = "；".join(reg_hints) if reg_hints else "无特别提示"
    return f"""
维度：{dimension}

[提案事实]（只能在这里面引用具体细节；如无相关信息请据实指出）：
{proposal_context or "（该维度的提案事实为空，仅可做通识性分析）"}

[可选监管/术语方向提示]（非必须引用，如与提案无关可忽略）：
{reg_txt}

题目：
{question.strip()}

{_unified_variant_instructions()}

回答要求：
- 语言：中文（专有名词可保留必要外文缩写）；
- **长度（模型自控为主）**：每一行分点正文（不含行首「1.」类编号）与 claims / evidence_hints / general_insights **每一条**均建议在 **{MAX_REVIEW_CHARS}** 字内写**完整语义**（自然结尾）；超长时删套话、合并表述；程序仅在极端超长时按句读兜底截断，不得依赖截断来收尾。\n
- 结构：answer 仅用 **2–4 条**短分点，每条一行；谢绝长篇铺垫。\n
- 内容优先级（最短篇幅）：\n
  1）一两句话概括本项目在该维度最关键的亮点或短板（紧扣[提案事实]）；\n
  2）如需对照行业，各用**一句**通识表述（标明「通常/一般而言」）；\n
  3）细颗粒结论尽量写入 claims，勿在 answer 展开长论述。\n
- 关联度：优先基于[提案事实]；缺信息处只写「需补充」，勿编造。\n
- 若[提案事实]中有部分信息但不完整，写成「已有……但在……仍缺细节」，勿绝对否定。\n
- 通识内容须用「通常/一般而言」等标记；不得写成项目已达成。\n
- general_insights：**2–5 条**短句通识即可；最后一条可为免责声明。\n
- 审慎性：多用「可能/建议/需确认」，避免绝对化。\n
- 仅输出一个 JSON 对象，不得有任何额外说明文字或 Markdown 围栏。

{_schema_structured()}
""".strip()


def build_batch_prompt(
    dimension: str,
    questions: List[str],
    proposal_context: str,
    reg_hints: List[str],
    variant_id: str,
) -> str:
    reg_txt = "；".join(reg_hints) if reg_hints else "无特别提示"
    q_block = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
    return f"""
你将针对同一维度下的多道问题，基于同一份【提案事实】给出与该提案高度相关的回答。
请严格以对象数组 JSON 返回结果，格式为：{{"answers":[<对象1>,<对象2>,...]}}，数组长度必须与题目数一致。

维度：{dimension}

[提案事实]（只能在这里面引用具体细节；如无相关信息请据实指出）：
{proposal_context or "（该维度的提案事实为空，仅可做通识性分析）"}

[可选监管/术语方向提示]（非必须引用，如与提案无关可忽略）：
{reg_txt}

问题列表：
{q_block}

{_unified_variant_instructions()}

统一回答要求：
- **每道题的 answer**：2–4 条短分点；**每一行分点正文（不含编号）各约 {MAX_REVIEW_CHARS} 字内写完整**；简明扼要、切中要害；禁止写到半截。
- 细颗粒展开写入 claims / evidence_hints；各题每一项 claims / evidence_hints / general_insights **各自**不超过 **{MAX_REVIEW_CHARS}** 字且**每条意思完整**；避免在 answer 里长篇铺陈。
- 每道题的 claims：少量可核对短句，优先基于[提案事实]。
- 每道题的 evidence_hints：少量线索；忌空泛套话。
- 每道题的 general_insights：少量短句通识；末条可为免责式总结。\n
- 若[提案事实]中已经给出某一方面的部分信息（例如有市场规模、竞品列举、风险表等），"
"  只能评价为“现有描述在……方面仍不够细化/缺乏定量对比”，不得一概写成“项目未提供市场需求分析/竞争对手评估”等绝对否定；\n
- 不得发明提案中不存在的新实验/新数据/新机构/具体注册号；对未给出的信息，只能以“需补充/需确认”的方式表达；\n
- 仅输出一个 JSON 对象，不得有额外文字或 Markdown 围栏。

{_schema_structured()}
""".strip()


def build_refine_prompt(candidate_obj: Dict[str, Any], proposal_context: str, dimension: str) -> str:
    original = json.dumps(candidate_obj, ensure_ascii=False, indent=2)
    return f"""
请在不引入任何超出【提案事实】的新信息的前提下，对下面的结构化回答做一次快速自我复核：
- **answer**：每一行分点正文（不含编号）各不超过 **{MAX_REVIEW_CHARS}** 字；在限制内保持**句意完整**，删冗余套话；
- **claims、evidence_hints、general_insights**：**每一条**各自不超过 **{MAX_REVIEW_CHARS}** 字且**语义完整**；删冗余条；
- 删改过于武断或缺乏依据的强结论；
- 若某条结论在【提案事实】中找不到依据，请改写为“需要补充的材料/信息”；
- 细节搬迁：过长叙述移到 claims / evidence_hints；answer 只保留短打；
- 检查 general_insights：仅保留简短通识句；\n
- 保持 JSON 结构和字段名完全不变（包括 general_insights）。

维度：{dimension}

[提案事实]：
{proposal_context or "（该维度的提案事实为空，仅可做轻度通识性修正）"}

原候选：
{original}
""".strip()


# ========== LLM 调用基础 ==========
def _with_retry(fn, max_tries: int = 4, base: float = 0.8):
    for i in range(max_tries):
        try:
            return fn()
        except Exception:
            if i == max_tries - 1:
                raise
            time.sleep(base * (2 ** i) + random.random() * 0.2)


ERR_PATTERNS = [
    r"\[?ERROR[:\]]",
    r"\bHTTP\s*4\d{2}\b",
    r"\bHTTP\s*5\d{2}\b",
    r"insufficient[_\s-]?quota",
    r"invalid[_\s-]?api[_\s-]?key",
    r"request\s+timed\s*out",
    r"rate\s*limit",
    r"payment\s*required",
    r"bad gateway",
    r"service unavailable",
    r"connection (?:reset|refused)",
]
_err_re = re.compile("|".join(ERR_PATTERNS), re.I)


def is_error_text(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    return bool(_err_re.search(t))


def _extract_text_from_content(content: Any) -> str:
    """
    兼容不同 SDK/模型返回的 message.content 形态：
    - str
    - list[str | dict]
    - dict
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()

    def _flatten(x: Any) -> List[str]:
        out: List[str] = []
        if x is None:
            return out
        if isinstance(x, str):
            s = x.strip()
            if s:
                out.append(s)
            return out
        if isinstance(x, list):
            for it in x:
                out.extend(_flatten(it))
            return out
        if isinstance(x, dict):
            # 常见字段优先
            for k in ("text", "content", "value"):
                if k in x:
                    out.extend(_flatten(x.get(k)))
            # 兜底：把其它字段也尝试展开
            if not out:
                for v in x.values():
                    out.extend(_flatten(v))
            return out
        s = str(x).strip()
        if s:
            out.append(s)
        return out

    parts = _flatten(content)
    return "\n".join([p for p in parts if p]).strip()


def _chat_completion_json(
    client,
    model: str,
    system_text: str,
    user_text: str,
    max_tokens: int,
    temperature: float,
    force_json: bool,
):
    def _token_limit_kwargs(model_name: str, max_out_tokens: int) -> Dict[str, int]:  # updated 21-April-2026
        if "gpt-5" in (model_name or "").lower():
            return {"max_completion_tokens": int(max_out_tokens)}  # updated 21-April-2026
        return {"max_tokens": int(max_out_tokens)}

    def call(json_mode: bool):
        kwargs = dict(
            model=model,
            messages=[{"role": "system", "content": system_text}, {"role": "user", "content": user_text}],
            temperature=float(temperature),
            **_token_limit_kwargs(model, max_tokens),  # updated 21-April-2026
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        return client.chat.completions.create(**kwargs)

    try:
        if force_json:
            resp = _with_retry(lambda: call(json_mode=True))
            out = _extract_text_from_content(resp.choices[0].message.content)
            if is_error_text(out):
                raise RuntimeError("provider_error_json_mode")
            return out
        else:
            resp = _with_retry(lambda: call(json_mode=False))
            out = _extract_text_from_content(resp.choices[0].message.content)
            if is_error_text(out):
                raise RuntimeError("provider_error_text_mode")
            return out
    except Exception:
        if force_json:
            resp = _with_retry(lambda: call(json_mode=False))
            out = _extract_text_from_content(resp.choices[0].message.content)
            if is_error_text(out):
                raise RuntimeError("provider_error_text_mode")
            return out
        raise


def _safe_parse_json_plus(txt: str) -> Optional[Any]:
    if is_error_text(txt):
        return None

    t = (txt or "").replace("\ufeff", "").strip()
    t = re.sub(r"^\s*```(?:json)?\s*\n?", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\n?\s*```\s*$", "", t, flags=re.IGNORECASE)
    t = t.replace("\xa0", " ")

    try:
        obj = json.loads(t)
        if isinstance(obj, list):
            return {"answers": obj}
        return obj
    except Exception:
        pass

    m = re.search(r"(\{.*\}|\[.*\])", t, re.S)
    if m:
        frag = m.group(1)
        try:
            tmp = json.loads(frag)
            if isinstance(tmp, list):
                return {"answers": tmp}
            return tmp
        except Exception:
            pass

    # 尝试给常见字段名补双引号，包括 general_insights
    cand = re.sub(
        r"(\banswer|claims|evidence_hints|general_insights|topic_tags|confidence|caveats\b)\s*:",
        r'"\1":',
        t,
    )
    try:
        return json.loads(cand)
    except Exception:
        return None


def _repair_to_schema_json(
    client,
    model: str,
    system_text: str,
    raw_text: str,
    max_tokens: int = 900,
) -> Optional[Dict[str, Any]]:
    repair_prompt = (
        "请把下面这段模型输出修复为严格符合 schema 的 JSON。"
        "仅输出 JSON，不要解释，不要 Markdown。\n\n"
        + _schema_structured()
        + "\n\n待修复文本：\n"
        + (raw_text or "")
    )
    try:
        repaired = _chat_completion_json(
            client,
            model,
            system_text,
            repair_prompt,
            max_tokens=max_tokens,
            temperature=0.0,
            force_json=True,
        )
        obj = _safe_parse_json_plus(repaired)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _batch_answers_schema_stub(n: int) -> str:
    return (
        f"必须输出 JSON 对象，顶层键固定为 \"answers\"；\"answers\" 为长度 {n} 的数组。\n"
        "数组元素结构（与单题一致）：\n"
        "{ \"answer\": \"...\", \"claims\": [], \"evidence_hints\": [], "
        "\"general_insights\": [], \"topic_tags\": [], \"confidence\": 0.0, \"caveats\": \"\" }\n"
        "仅输出 JSON，不要 Markdown，不要解释。"
    )


def _repair_to_schema_json_batch(
    client,
    model: str,
    system_text: str,
    raw_text: str,
    n_answers: int,
    max_tokens: int = 1400,
) -> Optional[Any]:
    """将损坏的批量输出修复为 {\"answers\":[...]} 且长度与题目数一致。"""
    repair_prompt = (
        _batch_answers_schema_stub(n_answers)
        + "\n\n待修复文本（可能截断）：\n"
        + (raw_text or "")[:14000]
    )
    try:
        repaired = _chat_completion_json(
            client,
            model,
            system_text,
            repair_prompt,
            max_tokens=max_tokens,
            temperature=0.0,
            force_json=True,
        )
        return _safe_parse_json_plus(repaired)
    except Exception:
        return None


# ========== 模型初始化 ==========
def init_openai():
    if OpenAIClient is None:
        return None, None
    try:
        from local_openai import getenv_model, make_openai_client, resolve_openai_base_url

        client = make_openai_client()
    except Exception:
        return None, None
    model = getenv_model()
    tgt = resolve_openai_base_url() or "(OpenAI 官方)"
    print(f"✅ 已加载问答模型 endpoint={tgt}，model={model}")
    return client, model


def init_deepseek():
    if OpenAIClient is None:
        return None, None
    key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not key:
        return None, None
    base = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1").strip()
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip()
    timeout_s = float(os.getenv("LLM_API_TIMEOUT_SECONDS", "120"))
    client = OpenAIClient(api_key=key, base_url=base, timeout=timeout_s)
    print(f"✅ 已加载 DeepSeek 模型：{model}")
    return client, model


# ========== 答案规范化 & 后处理 ==========
def _norm_str(s: Any) -> str:
    return " ".join(str(s or "").strip().split())


def _uniq_cut(lst: List[Any], k: int = MAX_LIST_LEN) -> List[str]:
    seen, out = set(), []
    for x in lst:
        t = _norm_str(x)
        if not t:
            continue
        key = t.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
        if len(out) >= k:
            break
    return out


RE_DATE = re.compile(r"\b(20\d{2}|19\d{2})([-/.]|年)\d{1,2}([-/\.日]|月)\d{1,2}\b|\b(Q[1-4]\s*-\s*20\d{2})\b", re.I)
RE_MONEY = re.compile(
    r"\b(\$|USD|EUR|CNY|RMB|CAD)\s*\d{2,}(,\d{3})*(\.\d+)?\b|\b\d+(\.\d+)?\s*(million|billion|万|亿)\b",
    re.I,
)
RE_TRIAL = re.compile(r"\bNCT\d{8}\b|\bEUCTR-\d{4}-\d{6}-\d{2}\b", re.I)
RE_DOI = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.I)
RE_PATENT = re.compile(r"\b(US|EP|CN)\d{5,}\b|\bWO\d{7,}\b", re.I)
RE_ISO = re.compile(r"\bISO\s?\d{4,5}(-\d+)?\b", re.I)
RE_STDNUM = re.compile(r"\bEN\s?\d{3,5}\b|\bASTM\s?[A-Z]?\d{2,5}\b", re.I)
RE_ID_ANY = re.compile(r"(注册号|登记号|批准文号|备案号)", re.I)


def _is_redline(text: str) -> bool:
    t = text or ""
    if RE_DATE.search(t):
        return True
    if RE_MONEY.search(t):
        return True
    if RE_TRIAL.search(t):
        return True
    if RE_DOI.search(t):
        return True
    if RE_PATENT.search(t):
        return True
    if RE_ISO.search(t):
        return True
    if RE_STDNUM.search(t):
        return True
    if RE_ID_ANY.search(t):
        return True
    return False


def _scrub_claims_to_hints(claims: List[Any], hints: List[Any]) -> Tuple[List[str], List[str], List[str]]:
    """
    把明显带编号/金额的句子从 claims 挪到 evidence_hints，并记录到 moved_facts。
    不再额外注水模板；只做清洗和重分类。
    """
    new_claims: List[str] = []
    new_hints: List[str] = [str(x).strip() for x in (hints or []) if str(x).strip()]
    moved_facts: List[str] = []

    for c in claims or []:
        s = _norm_str(c)
        if not s:
            continue
        if _is_redline(s):
            new_hints.append(s)
            moved_facts.append(s)
        else:
            new_claims.append(s)

    return _uniq_cut(new_claims, MAX_LIST_LEN), _uniq_cut(new_hints, MAX_LIST_LEN), _uniq_cut(moved_facts, MAX_LIST_LEN)


def _to_bullets(answer: str) -> Tuple[str, int]:
    """
    把任意文本整理成 1. 2. 3. 形式的分点；不过度强制数量，只要 >=1 即可。
    修复双重编号问题：无论是按行还是按句拆分，都会先去掉已有编号/符号再统一加“1. 2. 3.”。
    """
    raw = (answer or "").strip()
    if not raw:
        return "", 0

    text = raw.replace("\r\n", "\n").strip()

    def _normalize_item(line: str) -> str:
        s = line.strip()
        # 去掉 Markdown 项目符号
        s = re.sub(r"^\s*[-*•●▪]+\s*", "", s)
        # 去掉前导数字编号（1. / 1) / 1、 等）
        s = re.sub(r"^\s*\d+[\.\)．、]\s*", "", s)
        # 压缩空白
        s = re.sub(r"\s+", " ", s)
        s = s.strip()
        if not s:
            return ""

        # 丢掉只剩数字或“数字+空格”的内容（例如 "1"、"2"、"1 2" 等）
        s_nospace = s.replace(" ", "")
        if s_nospace.isdigit() and len(s_nospace) <= 4:
            return ""

        return s

    # 先按行拆分（利用 LLM 原始的换行结构）
    lines = [ln for ln in text.split("\n") if ln.strip()]
    items: List[str] = []
    for ln in lines:
        norm = _normalize_item(ln)
        if norm:
            items.append(norm)

    # 如果按行只有 0–1 条，说明可能是整段一坨 -> 再按句号/分号拆一轮
    if len(items) <= 1:
        chunks = re.split(r"[。；;.!?？]\s*", text)
        items = []
        for ch in chunks:
            norm = _normalize_item(ch)
            if norm:
                items.append(norm)

    if not items:
        # 实在拆不出有效内容，就保留原文
        return raw, 0

    # 条数少，便于满足每行分点正文字数上限（finalize 按条截断）
    items = items[:4]

    numbered = [f"{i+1}. {seg}" for i, seg in enumerate(items)]
    return "\n".join(numbered), len(items)

def _calibrate_conf(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        v = 0.65
    if v < CONF_MIN:
        v = CONF_MIN
    if v > CONF_MAX:
        v = CONF_MAX
    return round(v, 2)


def _normalize_candidate_obj(obj: Any) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        obj = {}
    out: Dict[str, Any] = {}

    # ❗ 保留原始换行，不再用 _norm_str 压缩，避免破坏 LLM 已经分好的行
    raw_answer = obj.get("answer", "")

    # ====== 新增：专门处理 list / dict 形式的 answer，避免 json.dumps 成一坨 ======
    if isinstance(raw_answer, list):
        # LLM 有时会给 ["1\n2. ...", "3\n4. ..."] 这种
        pieces: List[str] = []
        for elem in raw_answer:
            if elem is None:
                continue
            # 再兜一层 list（极端情况）
            if isinstance(elem, list):
                for sub in elem:
                    s = str(sub or "").strip()
                    if s:
                        pieces.append(s)
            else:
                s = str(elem or "").strip()
                if s:
                    pieces.append(s)
        raw_answer = "\n".join(pieces)  # 变成多行文本，交给 _to_bullets 按行拆

    elif isinstance(raw_answer, dict):
        # 如果以后 LLM 返回 {"bullets":[...]} 之类，优先拉出里面的 list
        for key in ("bullets", "points", "items"):
            val = raw_answer.get(key)
            if isinstance(val, list):
                pieces = [str(x or "").strip() for x in val if str(x or "").strip()]
                raw_answer = "\n".join(pieces)
                break
        else:
            # 实在没有结构化 list，再兜底 dump 成字符串
            raw_answer = json.dumps(raw_answer, ensure_ascii=False)

    # 其他情况：本来就是字符串/数字，直接转成字符串
    out["answer"] = str(raw_answer or "")

    raw_claims = obj.get("claims", [])
    raw_hints = obj.get("evidence_hints", [])
    raw_tags = obj.get("topic_tags", [])
    raw_gi = obj.get("general_insights", [])

    if isinstance(raw_claims, (str, int, float)):
        raw_claims = [raw_claims]
    if isinstance(raw_hints, (str, int, float)):
        raw_hints = [raw_hints]
    if isinstance(raw_tags, (str, int, float)):
        raw_tags = [raw_tags]
    if isinstance(raw_gi, (str, int, float)):
        raw_gi = [raw_gi]

    out["claims"] = [str(x).strip() for x in (raw_claims or []) if str(x).strip()]
    out["evidence_hints"] = [str(x).strip() for x in (raw_hints or []) if str(x).strip()]
    out["topic_tags"] = [str(x).strip() for x in (raw_tags or []) if str(x).strip()]
    out["general_insights"] = [str(x).strip() for x in (raw_gi or []) if str(x).strip()]
    try:
        out["confidence"] = float(obj.get("confidence", 0.65))
    except Exception:
        out["confidence"] = 0.65
    out["caveats"] = _norm_str(obj.get("caveats", ""))
    return out


def _validate_candidate_dict(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    if not isinstance(obj.get("answer"), str) or not obj.get("answer").strip():
        return False
    if not isinstance(obj.get("claims"), list):
        return False
    if not isinstance(obj.get("evidence_hints"), list):
        return False
    if not isinstance(obj.get("topic_tags"), list):
        return False
    if not isinstance(obj.get("general_insights"), list):
        return False
    return True


def _build_topic_tags(tags: List[Any], dimension: str, answer: str) -> List[str]:
    base = [str(t).lower().strip() for t in (tags or []) if str(t).strip()]
    extra: List[str] = []

    dim_token = dimension.lower().strip()
    if dim_token:
        extra.append(dim_token)

    words = re.findall(r"[A-Za-z]+|[\u4e00-\u9fff]{2,8}", answer)
    freq: Dict[str, int] = {}
    for w in words:
        wl = w.lower()
        if len(wl) < 2:
            continue
        freq[wl] = freq.get(wl, 0) + 1
    common = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:6]
    extra.extend([w for w, _ in common])

    return _uniq_cut(base + extra, MAX_LIST_LEN)


def _quick_score(cand: Dict[str, Any]) -> Dict[str, Any]:
    """
    给后续选优一个简单分数：
    - 分点条数
    - claims 数量
    - confidence
    （general_insights 目前只在下游使用，这里不额外打分，避免逻辑过重）
    """
    bullets = len([ln for ln in str(cand.get("answer", "")).splitlines() if ln.strip()])
    claims_n = len(cand.get("claims") or [])
    conf = float(cand.get("confidence", 0.65))

    score = 0.0
    # 主回答每条分点压短（每条≤上限）时通常仅 2–3 个分点，评分阈值随之放宽
    if bullets >= 2:
        score += 0.35
    elif bullets >= 1:
        score += 0.18
    if claims_n >= 2:
        score += 0.22
    if claims_n >= 3:
        score += 0.08
    score += max(0.0, min(0.20, (conf - CONF_MIN) / (CONF_MAX - CONF_MIN + 1e-6) * 0.20))

    cand["quick_score"] = round(score, 3)
    return cand


def _finalize_candidate(
    obj: Dict[str, Any],
    provider: str,
    model: str,
    variant_id: str,
    sample_id: int,
    dimension: str,
) -> Dict[str, Any]:
    base = _normalize_candidate_obj(obj)
    answer_bullets, n_bullets = _to_bullets(base["answer"])
    base["answer"] = _clip_numbered_answer_each_bullet(answer_bullets, MAX_REVIEW_CHARS)
    n_bullets = len([ln for ln in base["answer"].splitlines() if ln.strip()])

    new_claims, new_hints, moved_facts = _scrub_claims_to_hints(base.get("claims", []), base.get("evidence_hints", []))
    base["facts_redlined"] = moved_facts

    # general_insights 只做去重+截断，不做红线搬移（允许包含“通常需要 NCT 编号”这类行业通识）
    base["general_insights"] = _uniq_cut(base.get("general_insights", []), MAX_LIST_LEN)
    # 各列表字段总字数硬上限（与 answer 同级）
    base["claims"] = _clip_each_list_item_max_chars(new_claims, MAX_REVIEW_CHARS)
    base["evidence_hints"] = _clip_each_list_item_max_chars(new_hints, MAX_REVIEW_CHARS)
    base["general_insights"] = _clip_each_list_item_max_chars(
        base["general_insights"], MAX_REVIEW_CHARS
    )

    base["topic_tags"] = _build_topic_tags(base.get("topic_tags", []), dimension, base["answer"])
    base["confidence"] = _calibrate_conf(base.get("confidence", 0.65))

    if not base.get("caveats"):
        base["caveats"] = "结论需结合原始提案全文与支撑材料进一步核查。"

    base["provider"] = provider
    base["model"] = model
    base["variant_id"] = variant_id
    base["sample_id"] = int(sample_id)
    base["generated_at"] = now_str()

    base["diag"] = {
        "bullet_count": n_bullets,
        "claims_count": len(base["claims"]),
        "hints_count": len(base["evidence_hints"]),
        "general_insights_count": len(base["general_insights"]),
    }

    base = _quick_score(base)
    return base


# ========== 问答主逻辑（batch / single） ==========
def ask_model_batch(
    provider: str,
    client,
    model: str,
    dimension: str,
    q_list: List[str],
    proposal_context: str,
    reg_hints: List[str],
    variant_id: str,
    max_tokens: int,
    system_text: str,
) -> List[Dict[str, Any]]:
    """
    对支持 batch 的模型：同一维度下多题一起问；
    若解析失败或数组长度不对，由上层改走逐题兜底。
    返回：长度 == len(q_list) 的 candidate 列表。
    """
    prompt = build_batch_prompt(dimension, q_list, proposal_context, reg_hints, variant_id)
    try:
        txt = _chat_completion_json(
            client,
            model,
            system_text,
            prompt,
            max_tokens=max_tokens,
            temperature=TEMP_BY_VARIANT.get(variant_id, 0.3),
            force_json=provider_caps(provider)["json_mode"],
        )
    except Exception:
        return []

    low = (txt or "").strip().lower()
    if any(k in low for k in ("incorrect api key", "invalid api key", "rate limit", "quota", "access denied")):
        return []

    obj = _safe_parse_json_plus(txt)
    need_repair = (
        not isinstance(obj, dict)
        or not isinstance((obj or {}).get("answers"), list)
        or len((obj or {}).get("answers") or []) != len(q_list)
    )
    if need_repair and (txt or "").strip() and not is_error_text((txt or "").strip()):
        fixed = _repair_to_schema_json_batch(
            client,
            model,
            system_text,
            txt or "",
            len(q_list),
            max_tokens=min(1800, int(max_tokens) + 500),
        )
        if isinstance(fixed, dict) and isinstance(fixed.get("answers"), list) and len(fixed["answers"]) == len(q_list):
            obj = fixed
    if not isinstance(obj, dict):
        return []
    arr = obj.get("answers")
    if not isinstance(arr, list) or len(arr) != len(q_list):
        return []

    out: List[Dict[str, Any]] = []
    valid_count = 0
    for idx, it in enumerate(arr, 1):
        cand_norm = _normalize_candidate_obj(it)
        if not _validate_candidate_dict(cand_norm):
            out.append({})
            continue
        finalized = _finalize_candidate(
            cand_norm,
            provider=provider,
            model=model,
            variant_id=variant_id,
            sample_id=idx,
            dimension=dimension,
        )
        out.append(finalized)
        valid_count += 1

    # If this batch returned no usable candidates, force caller to switch to single-question fallback.
    if valid_count == 0:
        return []
    return out


def ask_model_single(
    provider: str,
    client,
    model: str,
    dimension: str,
    question: str,
    proposal_context: str,
    reg_hints: List[str],
    variant_id: str,
    max_tokens: int,
    system_text: str,
) -> Dict[str, Any]:
    def _hard_fallback(reason: str, raw_text: str = "") -> Dict[str, Any]:
        # 最终兜底：确保每道题至少有结构化候选，避免全量 candidates 为空。
        ctx_lines = [ln.strip() for ln in (proposal_context or "").splitlines() if ln.strip()]
        ctx_brief = "；".join(ctx_lines[:2]) if ctx_lines else "当前维度材料较少，需结合提案原文补充。"
        base = {
            "answer": (
                "1. 结合现有提案材料，本题可先形成初步判断，但关键证据仍需补充。\n"
                f"2. 当前可引用的维度信息：{ctx_brief}\n"
                "3. 建议优先补充可核验里程碑、量化指标与责任边界，再进行高置信度结论评估。"
            ),
            "claims": [
                "现有材料可支持初步评审，但不足以支撑强结论。",
                "需要补充可核验数据与实施证据以提高结论置信度。",
            ],
            "evidence_hints": [
                "核对该维度在 dimensions_v2.json 的 summary/key_points/risks 字段。",
                "回查提案正文中与本题相关的里程碑、量化指标、合作与责任描述。",
            ],
            "general_insights": [
                "一般而言，评审结论应优先绑定可核验事实与量化指标。",
                "同类项目常见问题是目标定义清晰但验收口径与责任矩阵不足。",
                "以上为行业通识建议，不代表本项目已经满足相关条件。",
            ],
            "topic_tags": [dimension, "fallback"],
            "confidence": 0.55,
            "caveats": f"模型输出降级路径触发：{reason}",
        }
        if raw_text.strip():
            base["evidence_hints"].append("模型原始文本已保留，可用于人工复核。")
        return _finalize_candidate(
            base,
            provider=provider,
            model=model,
            variant_id=variant_id,
            sample_id=1,
            dimension=dimension,
        )

    prompt = build_single_prompt(dimension, question, proposal_context, reg_hints, variant_id)
    try:
        txt = _chat_completion_json(
            client,
            model,
            system_text,
            prompt,
            max_tokens=max_tokens,
            temperature=TEMP_BY_VARIANT.get(variant_id, 0.3),
            force_json=provider_caps(provider)["json_mode"],
        )
    except Exception as e:
        return _hard_fallback(f"model_call_failed:{type(e).__name__}")

    obj = _safe_parse_json_plus(txt)
    if not isinstance(obj, dict):
        cleaned = (txt or "").strip()
        if cleaned and not is_error_text(cleaned):
            repaired_obj = _repair_to_schema_json(client, model, system_text, cleaned)
            if isinstance(repaired_obj, dict):
                obj = repaired_obj
            else:
                return _hard_fallback("non_json_response_quarantined", raw_text=cleaned)
        else:
            return _hard_fallback("parse_failed_or_empty", raw_text=txt or "")

    cand_norm = _normalize_candidate_obj(obj)
    if not _validate_candidate_dict(cand_norm):
        raw_for_repair = (
            json.dumps(obj, ensure_ascii=False)
            if isinstance(obj, dict)
            else (txt or "")
        )
        repaired_obj = _repair_to_schema_json(
            client,
            model,
            system_text,
            raw_for_repair,
            max_tokens=min(900, int(max_tokens)),
        )
        if isinstance(repaired_obj, dict):
            cand_norm = _normalize_candidate_obj(repaired_obj)
        if not _validate_candidate_dict(cand_norm):
            compact_note = (
                "\n\n=== 二次修复约束 ===\n"
                f"answer 每一行分点正文（不含编号）各不超过 {MAX_REVIEW_CHARS} 字且须语义完整；"
                f"claims、evidence_hints、general_insights 每一条各自不超过 {MAX_REVIEW_CHARS} 字且须完整成句；"
                "仅输出合法 JSON。"
            )
            repaired2 = _repair_to_schema_json(
                client,
                model,
                system_text,
                (raw_for_repair or "")[:8000] + compact_note,
                max_tokens=min(700, int(max_tokens)),
            )
            if isinstance(repaired2, dict):
                cand_norm = _normalize_candidate_obj(repaired2)
        if not _validate_candidate_dict(cand_norm):
            return _hard_fallback("schema_validation_failed", raw_text=txt or "")

    return _finalize_candidate(
        cand_norm,
        provider=provider,
        model=model,
        variant_id=variant_id,
        sample_id=1,
        dimension=dimension,
    )


def refine_candidate(
    candidate: Dict[str, Any],
    client,
    model: str,
    dimension: str,
    proposal_context: str,
    provider: str,
    system_text: str,
    max_tokens: int = 600,
) -> Dict[str, Any]:
    try:
        rp = build_refine_prompt(candidate, proposal_context, dimension)
        txt = _chat_completion_json(
            client,
            model,
            system_text,
            rp,
            max_tokens=max_tokens,
            temperature=0.2,
            force_json=provider_caps(provider)["json_mode"],
        )
        obj = _safe_parse_json_plus(txt)
        if isinstance(obj, dict):
            cand_norm = _normalize_candidate_obj(obj)
            if not _validate_candidate_dict(cand_norm):
                return candidate
            return _finalize_candidate(
                cand_norm,
                provider=candidate.get("provider", provider),
                model=candidate.get("model", model),
                variant_id=candidate.get("variant_id", COMBINED_VARIANT_ID),
                sample_id=candidate.get("sample_id", 1),
                dimension=dimension,
            )
        return candidate
    except Exception:
        return candidate


# ========== 维度级问答 ==========
def print_dim_banner(provider_name: str, dim: str, total: int, mode: str):
    bar = "=" * 12
    print(f"\n{bar} [{provider_name}] 维度：{dim} | 题目数：{total} | 模式：{mode} {bar}", flush=True)


def print_q_progress(provider_name: str, dim: str, idx: int, total: int, qtext: str):
    preview = qtext.strip().replace("\n", " ")
    if len(preview) > 80:
        preview = preview[:80] + "..."
    print(f"[{provider_name}] ({dim}) Q{idx}/{total} ▶ {preview}", flush=True)


def chunked(lst: List[Any], n: int):
    for i in range(0, len(lst), n):
        yield i, lst[i : i + n]


def answer_dimension(
    provider: str,
    client,
    model_name: str,
    dim: str,
    q_list: List[str],
    proposal_context: str,
    reg_hints: List[str],
    refine: bool,
    group_size: int,
    max_tokens: int,
    system_text: str,
) -> List[Dict[str, Any]]:
    """
    返回：list[
      {
        "dimension": dim,
        "q_index": idx,
        "question": q,
        "candidates": [candidate_obj, ...]
      }, ...
    ]
    """
    out_items: List[Dict[str, Any]] = []
    provider_name = "ChatGPT" if provider == "openai" else "DeepSeek"

    if not q_list:
        return out_items

    caps = provider_caps(provider)
    supports_batch = bool(caps.get("batch_ok", True))

    mode = "批量+单次综合" if supports_batch else "逐题+单次综合"
    print_dim_banner(provider_name, dim, len(q_list), mode)

    def _maybe_refine(cands_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not refine or not cands_list:
            return cands_list
        c0 = cands_list[0]
        refined = refine_candidate(
            c0,
            client=client,
            model=model_name,
            dimension=dim,
            proposal_context=proposal_context,
            provider=provider,
            system_text=system_text,
            max_tokens=min(700, max_tokens),
        )
        return [refined]

    # 不支持 batch：逐题单次（综合指令）
    if not supports_batch:
        for idx, q in enumerate(q_list, 1):
            print_q_progress(provider_name, dim, idx, len(q_list), q)
            cand = ask_model_single(
                provider=provider,
                client=client,
                model=model_name,
                dimension=dim,
                question=q,
                proposal_context=proposal_context,
                reg_hints=reg_hints,
                variant_id=COMBINED_VARIANT_ID,
                max_tokens=min(900, max_tokens),
                system_text=system_text,
            )
            cands: List[Dict[str, Any]] = []
            if isinstance(cand, dict) and not cand.get("error") and cand.get("answer", "").strip():
                cands.append(cand)
            cands = _maybe_refine(cands)
            for c in cands:
                c["dimension"] = dim
                c["q_index"] = idx

            out_items.append(
                {
                    "dimension": dim,
                    "q_index": idx,
                    "question": q,
                    "candidates": cands,
                }
            )
        return out_items

    # 支持 batch：按 group_size 分批；每题仅一次生成（combined）
    group_size = max(1, min(int(group_size), 4))
    for start_idx, sub_qs in chunked(q_list, group_size):
        batch_tag = f"{start_idx+1}-{start_idx+len(sub_qs)}"
        t0 = time.time()
        arr = ask_model_batch(
            provider=provider,
            client=client,
            model=model_name,
            dimension=dim,
            q_list=sub_qs,
            proposal_context=proposal_context,
            reg_hints=reg_hints,
            variant_id=COMBINED_VARIANT_ID,
            max_tokens=max_tokens,
            system_text=system_text,
        )
        ok = bool(arr) and len(arr) == len(sub_qs)
        print(
            f"[{provider_name}] ({dim}) 小批 {batch_tag} · {COMBINED_VARIANT_ID:<13} "
            f"返回 {len(arr)}/{len(sub_qs)} 条，用时 {time.time()-t0:.1f}s（{'OK' if ok else 'FAIL→逐题'}）"
        )

        if ok:
            for j, q in enumerate(sub_qs, 1):
                global_idx = start_idx + j
                cand = arr[j - 1]
                cands: List[Dict[str, Any]] = []
                if isinstance(cand, dict) and not cand.get("error") and cand.get("answer", "").strip():
                    cands.append(cand)
                if not cands:
                    fb = ask_model_single(
                        provider=provider,
                        client=client,
                        model=model_name,
                        dimension=dim,
                        question=q,
                        proposal_context=proposal_context,
                        reg_hints=reg_hints,
                        variant_id=COMBINED_VARIANT_ID,
                        max_tokens=min(760, max_tokens),
                        system_text=system_text,
                    )
                    if isinstance(fb, dict) and not fb.get("error") and fb.get("answer", "").strip():
                        cands.append(fb)
                cands = _maybe_refine(cands)
                for c in cands:
                    c["dimension"] = dim
                    c["q_index"] = global_idx
                out_items.append(
                    {
                        "dimension": dim,
                        "q_index": global_idx,
                        "question": q,
                        "candidates": cands,
                    }
                )
            continue

        # 批量失败：这一小批改逐题（单次 combined）
        for j, q in enumerate(sub_qs, 1):
            global_idx = start_idx + j
            print_q_progress(provider_name, dim, global_idx, len(q_list), q)
            cand = ask_model_single(
                provider=provider,
                client=client,
                model=model_name,
                dimension=dim,
                question=q,
                proposal_context=proposal_context,
                reg_hints=reg_hints,
                variant_id=COMBINED_VARIANT_ID,
                max_tokens=min(760, max_tokens),
                system_text=system_text,
            )
            cands: List[Dict[str, Any]] = []
            if isinstance(cand, dict) and not cand.get("error") and cand.get("answer", "").strip():
                cands.append(cand)
            cands = _maybe_refine(cands)
            for c in cands:
                c["dimension"] = dim
                c["q_index"] = global_idx
            out_items.append(
                {
                    "dimension": dim,
                    "q_index": global_idx,
                    "question": q,
                    "candidates": cands,
                }
            )

    return out_items


# ========== 多模型合并 ==========
def merge_two_models(chatgpt_items: List[Dict[str, Any]], deepseek_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """按 (dimension, q_index, question) 合并候选；deepseek_items 为空时等价于仅用 ChatGPT 一套。"""
    def key_of(x: Dict[str, Any]):
        return (x["dimension"], x["q_index"], x["question"])

    pool: Dict[Any, Dict[str, Any]] = {}
    for it in chatgpt_items:
        pool[key_of(it)] = {
            "dimension": it["dimension"],
            "q_index": it["q_index"],
            "question": it["question"],
            "candidates": list(it.get("candidates", [])),
        }
    for it in deepseek_items or []:
        k = key_of(it)
        if k in pool:
            pool[k]["candidates"].extend(it.get("candidates", []))
        else:
            pool[k] = {
                "dimension": it["dimension"],
                "q_index": it["q_index"],
                "question": it["question"],
                "candidates": list(it.get("candidates", [])),
            }

    items = list(pool.values())
    items.sort(
        key=lambda x: (
            DIM_ORDER.index(x["dimension"]) if x["dimension"] in DIM_ORDER else 99,
            x["q_index"],
        )
    )
    return items


# ========== CLI ==========
def parse_args():
    ap = argparse.ArgumentParser(
        description="Proposal-aware answering (默认 ChatGPT/OpenAI；可选 DeepSeek —— 见 ENABLE_DEEPSEEK_IN_ANSWERING)"
    )
    ap.add_argument(
        "--proposal-id",
        "--proposal_id",
        dest="proposal_id",
        type=str,
        default="",
        help="指定提案 ID；不填则自动检测 data/extracted 最近目录名",
    )
    ap.add_argument(
        "--qs-file",
        type=str,
        default=str(CONFIG_QS_DEFAULT),
        help="问题集 JSON 路径（通常由 generate_questions.py 生成）",
    )
    ap.add_argument(
        "--dim-file",
        type=str,
        default="",
        help="维度 JSON 路径；默认使用 data/extracted/{pid}/dimensions_v2.json",
    )
    ap.add_argument(
        "--refine",
        type=int,
        default=1,
        help="是否进行轻量自我复核（0/1）",
    )
    ap.add_argument(
        "--max_tokens",
        type=int,
        default=2200,
    )
    ap.add_argument(
        "--group-size",
        type=int,
        default=3,
        help="每批问题数（仅当环境变量 LLM_USE_BATCH=1 启用批量时生效；默认逐题）",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
    )
    ap.add_argument(
        "--skip-inference-health-check",
        action="store_true",
        help="跳过本机推理服务连通性检查（仍可连接失败于首次请求）",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    if args.seed is not None:
        try:
            random.seed(int(args.seed))
        except Exception:
            pass

    pid = args.proposal_id.strip() or get_context_proposal_id() or detect_latest_pid()
    if pid == "unknown":
        print("⚠️ 未检测到 data/extracted 下的提案目录，将使用占位 pid=unknown。")
    print(f"🧩 proposal_id = {pid}")

    qs_path = Path(args.qs_file)
    if not qs_path.exists():
        raise FileNotFoundError(f"未找到问题集：{qs_path}")
    qs_cfg = read_json(qs_path)

    missing = [d for d in DIM_ORDER if not get_q_list(qs_cfg.get(d, []))]
    if missing:
        raise RuntimeError(
            f"问题集缺少维度或该维度题目为空：{missing}（请检查 {qs_path} 或 generate_questions.py 输出）"
        )

    # ========= 维度上下文：固定读取 extracted/{pid}/dimensions_v2.json =========
    if args.dim_file.strip():
        dim_file = Path(args.dim_file.strip())
    else:
        dim_file = EXTRACTED_DIR / pid / "dimensions_v2.json"

    if not dim_file.exists():
        raise FileNotFoundError(
            f"未找到维度文件：{dim_file} ；请先运行 build_dimensions_from_facts.py 生成 dimensions_v2.json"
        )

    print(f"📄 使用维度上下文文件：{dim_file}")
    dim_context_map = load_dimension_context(pid, dim_file)
    domain_probe_text = "\n".join([dim_context_map.get(d, "") for d in DIM_ORDER]).strip()
    domain = detect_domain_from_text(domain_probe_text)
    system_text = build_system_prompt(domain)
    print(f"🧭 已识别提案领域: {domain}")

    reg_hints_map: Dict[str, List[str]] = {}
    for dim in DIM_ORDER:
        reg_hints_map[dim] = _load_reg_hints(qs_cfg, dim, limit=8)

    maybe_probe_local_inference(skip=bool(args.skip_inference_health_check))

    oa_client, oa_model = init_openai()
    ds_client, ds_model = init_deepseek()
    oa_ok = oa_client is not None and oa_model is not None
    ds_ok = ds_client is not None and ds_model is not None
    run_deepseek = bool(ENABLE_DEEPSEEK_IN_ANSWERING and ds_ok)
    if not oa_ok and not run_deepseek:
        raise RuntimeError(
            "未检测到可用推理客户端：默认需要本机 Ollama 等 OpenAI 兼容服务（见端口与 OPENAI_MODEL），"
            "或配置 OPENAI_API_BASE=https://api.openai.com/v1 与 OPENAI_API_KEY；"
            "也可仅启用 ENABLE_DEEPSEEK_IN_ANSWERING=1（需可用的 DeepSeek 配置）。"
        )

    dims = DIM_ORDER[:]
    chatgpt_items_all: List[Dict[str, Any]] = []
    deepseek_items_all: List[Dict[str, Any]] = []

    # ChatGPT
    if oa_client and oa_model:
        print("🧠 ChatGPT 答题中 ...")
        for dim in dims:
            q_list = get_q_list(qs_cfg.get(dim, []))
            ctx = dim_context_map.get(dim, "")
            reg_hints = reg_hints_map.get(dim, [])
            items = answer_dimension(
                provider="openai",
                client=oa_client,
                model_name=oa_model,
                dim=dim,
                q_list=q_list,
                proposal_context=ctx,
                reg_hints=reg_hints,
                refine=bool(args.refine),
                group_size=int(args.group_size),
                max_tokens=int(args.max_tokens),
                system_text=system_text,
            )
            chatgpt_items_all.extend(items)

        out_path = OUT_REFINED / pid / "chatgpt_raw.json"
        write_json(
            out_path,
            {
                "meta": {
                    "model": oa_model,
                    "provider": "openai",
                    "generated_at": now_str(),
                    "pid": pid,
                },
                "items": chatgpt_items_all,
            },
        )
        print(f"✅ ChatGPT 结果 -> {out_path}")
    else:
        print("⚠️ 跳过主推理：init_openai 未就绪（检查本机服务或 OPENAI_API_BASE / OPENAI_MODEL）。")

    # DeepSeek（可选第二模型；默认不跑，避免与「仅 ChatGPT」部署不一致）
    if run_deepseek:
        print("🧠 DeepSeek 答题中（ENABLE_DEEPSEEK_IN_ANSWERING=1）...")
        for dim in dims:
            q_list = get_q_list(qs_cfg.get(dim, []))
            ctx = dim_context_map.get(dim, "")
            reg_hints = reg_hints_map.get(dim, [])
            items = answer_dimension(
                provider="deepseek",
                client=ds_client,
                model_name=ds_model,
                dim=dim,
                q_list=q_list,
                proposal_context=ctx,
                reg_hints=reg_hints,
                refine=bool(args.refine),
                group_size=int(args.group_size),
                max_tokens=int(args.max_tokens),
                system_text=system_text,
            )
            deepseek_items_all.extend(items)

        out_path = OUT_REFINED / pid / "deepseek_raw.json"
        write_json(
            out_path,
            {
                "meta": {
                    "model": ds_model,
                    "provider": "deepseek",
                    "generated_at": now_str(),
                    "pid": pid,
                },
                "items": deepseek_items_all,
            },
        )
        print(f"✅ DeepSeek 结果 -> {out_path}")
    elif ENABLE_DEEPSEEK_IN_ANSWERING and not ds_ok:
        print(
            "⚠️ 跳过 DeepSeek：已开启 ENABLE_DEEPSEEK_IN_ANSWERING 但未正确初始化（检查 DEEPSEEK_API_KEY 等）。"
        )

    merged_items = merge_two_models(chatgpt_items_all, deepseek_items_all)
    merged = {
        "meta": {
            "pid": pid,
            "generated_at": now_str(),
            "schema": "refined_items.v2.proposal_aware_with_general_insights",
            "args": {
                "refine": bool(args.refine),
                "max_tokens": int(args.max_tokens),
                "group_size": int(args.group_size),
            },
            "models": {
                "chatgpt": {"model": oa_model, "provider": "openai"} if chatgpt_items_all else None,
                "deepseek": {"model": ds_model, "provider": "deepseek"} if deepseek_items_all else None,
            },
        },
        "items": merged_items,
    }
    out_path = OUT_REFINED / pid / "all_refined_items.json"
    write_json(out_path, merged)
    print(f"📦 合并结果 -> {out_path}")
    print("🎯 完成。")


if __name__ == "__main__":  # pragma: no cover
    main()
