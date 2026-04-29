# -*- coding: utf-8 -*-
"""
Stage 1 · 块级事实抽取器（extract_facts_by_chunk.py）
----------------------------------------------------
输入：
  - src/data/prepared/<proposal_id>/full_text.txt  （由 prepare_proposal_text.py 生成）

输出：
  - src/data/extracted/<proposal_id>/raw_facts.jsonl  每行一个 JSON fact

职责：
  - 将长文按字符切块（带 overlap）
  - 对每个块，用 LLM 抽取“原子事实列表”
  - 每条事实附带：dimensions[], type, meta(chunk_index, char_range 由代码填充)
  - 后续 Stage 2 再用这些 facts 去构建五个维度的最终 dimensions_v2.json

本版优化要点：
  1）减小 chunk 大小、加大 overlap，提高单块信息覆盖率，避免一个块塞太多信息导致抽不全。
  2）加强 Prompt 中对“五维覆盖”的要求，显式提醒模型不要只抽单一维度的信息。
  3）更激进的 type→dimensions 映射，让跨维度事实被多个维度同时看到，避免某维度信息过于稀薄。
  4）扩充关键词推断逻辑 _infer_dims_from_text，补充 objectives / innovation / feasibility 等隐性表述。
  5）在运行结束时输出五个维度的事实数量分布，并对明显偏少的维度给出警告，便于调参与排错。
  6）新增：对“文本很长但抽取事实过少”的 chunk 自动再跑一轮 dense 模式，强化召回。
"""

import os
import json
import argparse
import re
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from run_context import get_context_proposal_id

load_dotenv(override=True)

from local_openai import getenv_model, make_openai_client, maybe_probe_local_inference

# ========= 路径配置 =========

BASE_DIR = Path(__file__).resolve().parents[2]
PREPARED_DIR = BASE_DIR / "src" / "data" / "prepared"
EXTRACTED_DIR = BASE_DIR / "src" / "data" / "extracted"

OPENAI_MODEL = getenv_model()


def _token_limit_kwargs(max_out_tokens: int) -> Dict[str, int]:  # updated 21-April-2026
    """Use model-compatible output token parameter across model families."""
    if "gpt-5" in OPENAI_MODEL.lower():
        return {"max_completion_tokens": int(max_out_tokens)}  # updated 21-April-2026
    return {"max_tokens": int(max_out_tokens)}

# 全局复用一个 OpenAI 兼容 client（官方 API 与本机推理共用配置），避免每个 chunk 反复初始化
client = make_openai_client()

VALID_DIMENSIONS = ["team", "objectives", "strategy", "innovation", "feasibility"]
VALID_TYPES = [
    "team_member",
    "org_structure",
    "collaboration",
    "resource",
    "pipeline",
    "milestone",
    "market",
    "tech_route",
    "product",
    "ip_asset",
    "evidence",
    "budget_item",
    "funding_source",
    "risk",
    "mitigation",
    "ai_model",
    "clinical_design",
    "regulatory",
    "other",
]

# ⚠️ Prompt：既要防幻觉，又要尽量“捞干净”五维相关信息，并保证多维覆盖
FACT_PROMPT = """
你是一个“事实抽取器”，负责从一小段提案文本中【逐条抽取原子事实】。

⚠️ 非常重要的硬性约束（请逐条遵守）：
1）你只处理当前这一个文本块，不要猜测其他页面或上下文的内容。
2）不允许编造任何文本中没有出现的：机构、人物、公司、医院、国家、城市、疾病名称、药物名、
   技术名称、模型名称、项目名称、市场规模、患者数量、CAGR 或其他具体数字。
3）禁止使用“可能 / likely / 一般认为 / 通常 / 预计 / 大多 / 被认为 / typically / usually / generally /
   probably / potentially / it is believed that”等带有推测性的词语，
   除非这些词本身已经出现在原文中并且你是在忠实复述原文。
4）每条 fact 必须能在原文中找到对应内容，可以轻微改写，但必须保留原文中的关键短语
   （例如技术名称、机构名称、管线名称、疾病名称、模型名称等）。
5）每条 fact 尽量控制在 1–2 句内，不要写成长段落；尽量保持具体、可验证。

【目标领域】
- 这是一个“跨领域技术项目”事实抽取任务，可能属于医疗健康、自动驾驶、智能制造、软件平台等任意领域。
- 你必须以当前文本块为准，不得预设领域，不得把某个行业模板硬套到不相关项目上。

【什么是“原子事实”？】
- 一条 fact 应该尽量只表达“一个相对独立、可以单独复述的事实”；
- 不要把很多主题揉成一条长句；
- 如果一句话里同时提到了“团队配置”和“里程碑时间表”，请拆成两条 facts。
- 如果一句话中包含多个独立的重要信息（例如“团队背景 + 技术路线 + 目标市场”），也应拆成多条 facts。

【高优先级内容（必须尽量完整抽取）】
- 如果文本中出现“项目团队 / 核心成员 / 负责人 / CEO / COO / 联合创始人 / 教授 / 博士”等介绍，
  请为每位成员拆分出多条事实，至少包括：
  - 职务 / 身份 + 所在机构（大学 / 医院 / 公司等）；
  - 主要研究方向或业务领域；
  - 与本项目相关的关键经验或成功案例（如：曾主导某类药物上市、带领团队完成某阶段临床、
    负责过重大合作项目、带来显著销售增长等）。
- 如果文本中出现“市场分析”相关内容（例如市场规模、CAGR、增长率、各区域市场份额、主要竞争对手、
  销售额、客户群体、竞争格局等），请将每一条重要数字和结论拆分成单独的事实，并统一标记 type="market"。
  不要用一句话模糊带过整段市场分析。

【抽取数量与粒度要求】
- 请尽量把长句拆成多条事实，保证每条 fact 聚焦一个主题；
- 如果当前文本块信息比较丰富，请尽量抽取 15–25 条 facts；
- 如果信息很少，可以少于 10 条，甚至为空；但不要因为“懒”而漏掉清晰的事实。
- 事实总数上限为 25 条，请不要超过 25 条。

【维度标签（dimensions 的含义）】
- "team":    任何跟“团队、个人、机构、角色、分工、协同”相关的事实
- "objectives": 项目的总体目标、阶段性目标、KPI、各条管线/子项目的目标等
- "strategy":   技术路线、临床/监管路线、合作/商业策略、市场进入策略、运营模式等
- "innovation": 技术/产品/模式的创新点、与现有方案对比、专利和独特资源、证据优势等
- "feasibility": 资源基础、预算和资金、实施路径、风险与应对、时间规划、支付/报销环境等

⚠️ 维度标注的硬性要求：
- 只要某条 fact 明显与上述任一维度有关，就必须把对应维度写进 dimensions。
- 一条 fact 可以同时属于多个维度（例如 ["team","strategy"]）。
- 每条 fact 的 dimensions 至少要包含 1 个标签；如果你真的无法判断，请使用 ["feasibility"] 兜底。
- 宁可多标维度、也不要让相关的 fact 没有维度标签。

【type 可选值】
- "team_member": 某个成员的身份、机构、研究方向、项目角色
- "org_structure": 团队/公司结构、分工、比例
- "collaboration": 国际合作、校企合作、国内外协同模式
- "resource": 实验室、平台、数据、合作资源等
- "pipeline": 某条管线/子项目的目标、适应症、技术路线
- "milestone": 某个阶段的具体任务/里程碑（含时间信息）
- "market": 市场规模、CAGR、主要国家/客户、市场驱动因素、竞争格局等
- "tech_route": 技术/算法/实验路线（例如 AI 设计流程、LNP 设计流程）
- "product": 产品/平台形态、商业模式等
- "ip_asset": 专利、专有技术、独特数据资源
- "evidence": 动物实验/临床/真实世界数据及其结论
- "budget_item": 某个阶段/用途的预算金额或成本构成
- "funding_source": 资金来源（自有、VC、资助、合作等）
- "risk": 技术/市场/资金/法规/AI 等方面的风险或不确定性
- "mitigation": 上述风险对应的应对措施或缓解策略
- "ai_model": 具体 AI 模型/算法/架构的描述
- "clinical_design": 临床试验/临床验证设计（阶段、入组、终点等）
- "regulatory": 法规路径、伦理审批、注册策略、报销/支付政策等
- "other": 无法归类但又与项目有关的事实

【标注提示（通用规则）】
- 提到市场规模、CAGR、目标国家/客户、竞争者、支付/报销 → type 一般用 "market" 或 "regulatory"
- 提到资金、预算、成本、阶段性投入 → type 一般用 "budget_item" 或 "funding_source"
- 提到风险、不确定性、瓶颈、挑战 → type 用 "risk"
- 提到“如何应对某风险/问题” → type 用 "mitigation"
- 提到具体 AI 模型/算法/架构 → type 用 "ai_model"
- 提到专利号、PCT、核心 IP、独特数据集 → type 用 "ip_asset"
- 如果你不确定 type 应该是什么，可以用 "other"，但尽量根据内容选择一个最接近的类型。

【维度覆盖要求（非常重要）】
- 如果该文本块中同时出现了团队信息、项目目标、技术/商业策略、创新亮点、可行性/风险等多种内容，
  请尽量保证抽取出来的 facts 在这些维度上都有覆盖。
- 不要把所有 fact 都集中在单一维度（例如只抽团队信息），而忽略同一文本块中出现的目标、
  策略、创新或可行性的信息。
- 当你需要“取舍”时，优先保留能代表不同维度、不同主题的事实，而不是在同一个小点上反复细化。

【语言要求】
- 每条 fact 的 text 必须使用中文表述；若原文主要语种不是中文，可译为中文后写入，避免整段以外文为主呈现。

【输出格式要求】
- 必须返回一个 JSON 对象，顶层只有一个键 "facts"
- "facts" 是一个数组，数组里是若干个事实对象
- 每个事实对象必须是这种结构：
  {
    "text": "用自然语言写的一条事实（1–2 句，尽量具体）",
    "dimensions": ["team", "strategy"],   // 从 ["team","objectives","strategy","innovation","feasibility"] 中选一到多个
    "type": "team_member"                 // 从上面给出的 type 列表中选一个
  }
- 不要输出 meta 字段，meta 信息由系统自动补充。
- 如果当前文本块没有任何有用事实，可以返回 {"facts": []}
- 不要输出任何解释文字，不要加注释，只输出 JSON。
现在开始处理我给你的文本块。
"""


def find_latest_prepared_proposal() -> str:
    if not PREPARED_DIR.exists():
        raise FileNotFoundError(f"未找到 prepared 目录: {PREPARED_DIR}")

    candidates = []
    for d in PREPARED_DIR.iterdir():
        if d.is_dir():
            candidates.append((d.stat().st_mtime, d.name))

    if not candidates:
        raise FileNotFoundError(f"prepared 目录下没有任何提案子目录: {PREPARED_DIR}")

    proposal_id = max(candidates, key=lambda x: x[0])[1]
    print(f"[INFO] [auto] 选中最新提案 ID: {proposal_id}")
    return proposal_id


def load_full_text(proposal_id: str) -> str:
    path = PREPARED_DIR / proposal_id / "full_text.txt"
    if not path.exists():
        raise FileNotFoundError(f"full_text.txt 不存在: {path}")
    text = path.read_text(encoding="utf-8")
    print(f"[INFO] 读取 full_text: {path} (长度 {len(text)} 字符)")
    return text


def make_chunks(text: str, max_chars: int = 1800, overlap: int = 400) -> List[Dict[str, Any]]:
    """
    简单按字符切块，带 overlap；不做句子级别切分。
    默认 max_chars=1800, overlap=400，比之前更细、更密，有利于提高抽取覆盖率。
    返回列表，每个元素包含: chunk_text, start, end, index
    """
    chunks = []
    n = len(text)
    if n == 0:
        return chunks

    idx = 0
    chunk_idx = 0
    while idx < n:
        end = min(n, idx + max_chars)
        chunk_text = text[idx:end]
        chunks.append(
            {
                "index": chunk_idx,
                "start": idx,
                "end": end,
                "text": chunk_text,
            }
        )
        chunk_idx += 1
        if end == n:
            break
        idx = max(0, end - overlap)

    print(f"[INFO] 已切分为 {len(chunks)} 个 chunk (max_chars={max_chars}, overlap={overlap})")
    return chunks


def detect_domain_profile(full_text: str) -> str:
    t = (full_text or "").lower()
    biomed_hits = sum(
        1 for k in ["biomedical", "clinical", "patient", "drug", "therapy", "临床", "患者", "药物", "医疗健康"] if k in t
    )
    auto_hits = sum(
        1 for k in ["autonomous", "vehicle", "adas", "lidar", "radar", "perception", "自动驾驶", "智驾", "感知"] if k in t
    )
    if auto_hits >= max(2, biomed_hits + 1):
        return "autonomous_systems"
    if biomed_hits >= 2:
        return "biomed"
    return "generic_tech"


def call_llm_for_chunk(
    chunk_text: str,
    attempt: int = 1,
    dense: bool = False,
    domain_hint: str = "generic_tech",
) -> Dict[str, Any]:
    """
    调用 OpenAI，对单个 chunk 抽取 facts。
    - attempt > 1：用于 JSON 解析失败后的重试（提示“上一次 JSON 不合法”）。
    - dense = True：用于“当前 chunk 文本很长但事实过少”的第二轮密集抽取，会额外要求多抽一些 facts。
    """
    extra_hint_parts = []

    if attempt > 1:
        extra_hint_parts.append(
            "⚠️ 注意：上一次你返回的 JSON 因为太长或不合法导致解析失败。"
            "这一次请严格控制 facts 数量不超过 18 条，并且务必保证 JSON 语法完全正确。"
        )

    if dense:
        extra_hint_parts.append(
            "⚠️ 当前文本块信息非常丰富，你在本次抽取时应尽量覆盖文本中出现的所有与团队、目标、"
            "策略、创新、可行性相关的关键事实。请优先抽取 15–22 条 facts，"
            "并尽量覆盖不同维度和不同主题，不要只聚焦在单一方面。"
        )

    extra_hint = ""
    if extra_hint_parts:
        extra_hint = "\n\n" + "\n".join(extra_hint_parts)

    domain_hint_text = (
        "\n\n【领域识别提示】\n"
        f"- 当前文档推断领域：{domain_hint}\n"
        "- 该提示只用于帮助你理解术语语境；你仍必须完全以当前文本块中的事实为准。"
    )

    messages = [
        {
            "role": "system",
            "content": "你是一个严谨的事实抽取器，只能基于给定文本块抽取原子事实，不得编造。",
        },
        {
            "role": "user",
            "content": FACT_PROMPT
            + extra_hint
            + domain_hint_text
            + "\n\n=== 文本块开始 ===\n"
            + chunk_text.strip(),
        },
    ]

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.0,
        **_token_limit_kwargs(1800),  # updated 21-April-2026
    )

    raw = resp.choices[0].message.content
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        print("[WARN] JSON 解析失败，返回原始内容，便于你排查：")
        print(raw)
        if attempt == 1:
            print("[INFO] 尝试使用收缩版 prompt 重试该 chunk ...")
            return call_llm_for_chunk(chunk_text, attempt=2, dense=dense)
        # 第二次还失败就直接抛出
        raise e

    if not isinstance(data, dict):
        data = {"facts": []}
    if "facts" not in data or not isinstance(data["facts"], list):
        data["facts"] = []

    return data

# ===== 市场类关键词 & 识别函数 =====

_MARKET_CN = [
    "市场", "市场规模", "市场容量", "市场需求", "市场前景", "市场潜力",
    "目标市场", "细分市场", "市场份额", "渗透率",
    "客户", "客户群体", "目标客户", "目标人群",
    "患者群体", "目标患者",
    "销售", "销售额", "销量", "营收", "收入", "收益",
    "定价", "价格", "报销", "支付方", "医保", "保险",
    "商业化", "商业模式", "商业机会",
    "竞争", "竞品", "竞争对手", "竞争格局",
    "CAGR", "增长率"
]

_MARKET_EN = [
    "market", "market size", "market volume", "market demand", "market potential",
    "target market", "segment", "niche",
    "market share", "penetration",
    "customer", "customers", "client", "clients",
    "patient population", "target patients",
    "sales", "revenue", "turnover", "income",
    "pricing", "price", "reimbursement", "payer", "insurance",
    "commercialization", "commercialisation", "business model",
    "competition", "competitive", "competitor", "competitors",
    "cagr", "growth rate"
]


def _looks_like_market_fact(text: str) -> bool:
    """
    判断一条 fact 的文本是否“明显是市场/商业相关”的内容。
    注意：这里只用来纠偏 type，对 risk/mitigation 不覆盖。
    """
    if not text:
        return False
    t = text.lower()

    # 中文关键字
    if any(k in text for k in _MARKET_CN):
        return True

    # 英文关键字
    if any(k in t for k in _MARKET_EN):
        return True

    return False

def _infer_dims_from_text(text: str) -> List[str]:
    """
    当 LLM 没有给出 dimensions 且 type 也无法可靠映射时，
    用中英文关键词做一轮粗略的维度推断，尽量不要丢掉有用信息。
    这里只做“补充”，不会覆盖已有维度。
    """
    t_lower = (text or "").lower()
    t = text or ""

    candidates = set()

    # ---- team ----
    team_keywords = [
        "团队", "小组", "联合体", "合作单位", "合作方", "协作单位",
        "研究者", "研究团队", "临床团队", "项目组",
        "负责人", "项目负责人", "带头人",
        "教授", "副教授", "主任", "专家", "研究员", "博士", "博士后",
        "医院", "大学", "研究所", "中心", "实验室",
        "ceo", "coo", "cto", "cso", "vp", "vice president",
        "founder", "co-founder", "chief executive officer",
    ]
    if any(k in t for k in team_keywords) or any(k in t_lower for k in team_keywords):
        candidates.add("team")

    # ---- objectives ----
    obj_keywords_cn = [
        "目标", "总体目标", "阶段性目标", "里程碑", "阶段性里程碑",
        "计划", "任务", "工作包", "kpi", "终点", "主要终点", "次要终点",
        "本项目旨在", "本项目将", "本项目计划", "预期达到", "希望实现",
    ]
    obj_keywords_en = [
        "aim", "aims to", "aimed to",
        "objective", "objectives", "goal", "goals",
        "milestone", "milestones", "endpoint", "endpoints",
        "is designed to", "seeks to", "intends to", "in order to",
    ]
    if any(k in t for k in obj_keywords_cn) or any(k in t_lower for k in obj_keywords_en):
        candidates.add("objectives")

    # ---- strategy ----
    strat_keywords_cn = [
        "策略", "路径", "路线", "方案", "技术路线", "实施方案",
        "商业模式", "市场进入", "商业化", "推广策略",
        "合作模式", "运营模式", "联合开发", "授权引进",
        "市场", "市场规模", "市场需求", "市场前景", "市场潜力",
        "市场份额", "竞争格局", "竞品", "竞争对手",
    ]

    strat_keywords_en = [
        "strategy", "strategies", "pathway", "roadmap",
        "commercial", "commercialization", "business model",
        "market entry", "go-to-market", "go to market",
        "market", "market size", "market demand", "market potential",
        "market share", "competitive landscape", "competition", "competitor", "competitors",
        "partnership", "licensing", "co-development",
        "development plan", "regulatory strategy",
    ]

    if any(k in t for k in strat_keywords_cn) or any(k in t_lower for k in strat_keywords_en):
        candidates.add("strategy")

    # ---- innovation ----
    inno_keywords_cn = [
        "创新", "创新性", "差异化", "独特", "首创", "领先", "颠覆",
        "新一代", "新型", "原创", "填补空白", "突破性", "首个", "第一例",
    ]
    inno_keywords_en = [
        "novel", "novelty", "innovative", "innovation",
        "differentiated", "differentiation", "unique",
        "first-in-class", "best-in-class", "state-of-the-art",
        "cutting-edge", "breakthrough", "original", "disruptive",
        "fills the gap", "fill the gap",
    ]
    if any(k in t for k in inno_keywords_cn) or any(k in t_lower for k in inno_keywords_en):
        candidates.add("innovation")

    # ---- feasibility ----
    feas_keywords_cn = [
        "可行性", "可行", "可实施", "资源", "平台", "基础设施",
        "预算", "经费", "资金", "成本", "成本负担",
        "风险", "挑战", "瓶颈", "不确定性",
        "时间表", "进度", "周期", "排期",
        "入组难度", "依从性", "工作量", "实施复杂度",
    ]
    feas_keywords_en = [
        "feasibility", "feasible", "resource", "resources", "infrastructure",
        "budget", "funding", "cost", "costs", "cost-effectiveness",
        "risk", "risks", "challenge", "challenges", "bottleneck", "uncertainty",
        "timeline", "schedule", "timeframe",
        "enrollment", "recruitment", "compliance", "adherence", "burden",
    ]
    if any(k in t for k in feas_keywords_cn) or any(k in t_lower for k in feas_keywords_en):
        candidates.add("feasibility")

    # ---- 专门为“市场分析段落”兜底：强行打上 strategy / objectives ----
    market_keywords_cn = [
        "市场", "市场规模", "市场分析", "市场份额", "市场占有率",
        "cagr", "复合年增长率", "销售额", "营收", "收入", "销售收入",
        "增长率", "增长幅度", "客户", "用户", "消费群体",
    ]
    market_keywords_en = [
        "market size", "market", "cagr", "market share", "share of",
        "sales", "revenue", "revenues", "turnover",
        "growth rate", "compound annual growth", "customer", "customers",
        "payer", "payers",
    ]
    if any(k in t for k in market_keywords_cn) or any(k in t_lower for k in market_keywords_en):
        candidates.add("strategy")
        candidates.add("objectives")

    return [d for d in candidates if d in VALID_DIMENSIONS]

def mark_numeric_suspect(fact: Dict[str, Any], chunk_text: str) -> Dict[str, Any]:
    """
    对包含数字的 fact 做简单校验：
    - 把 fact.text 里的数字片段（连续数字，不管是年份/金额）提取出来
    - 如果某个数字完全不出现在 chunk_text 中，则认为这条 fact 存在数字幻觉风险
    - 加 meta.suspect_numeric = True/False
    """
    text = fact.get("text", "") or ""
    nums = re.findall(r"\d+", text)
    if not nums:
        return fact  # 没数字，不管

    chunk_flat = (chunk_text or "").replace(" ", "")
    suspect = False
    for n in nums:
        if n not in chunk_flat:
            suspect = True
            break

    meta = fact.get("meta", {})
    if not isinstance(meta, dict):
        meta = {}
    meta["suspect_numeric"] = suspect
    fact["meta"] = meta
    return fact

def normalize_fact(
    fact: Dict[str, Any],
    proposal_id: str,
    chunk_index: int,
    start: int,
    end: int,
) -> Dict[str, Any]:
    """
    给每条 fact 填上 meta 信息；清洗 dimensions / type。
    同时对维度标签做 type→dimensions 的通用“补标签”映射（支持多维度），
    尽量保证五个维度的信息都不会被漏掉。
    """
    text = fact.get("text", "")
    if not isinstance(text, str):
        text = str(text)

    # 原始维度标签清洗
    dims = fact.get("dimensions", [])
    if not isinstance(dims, list):
        dims = []
    dims_clean = [d for d in dims if isinstance(d, str) and d in VALID_DIMENSIONS]

    type_val = fact.get("type", "other")
    if type_val not in VALID_TYPES:
        type_val = "other"

    # ===== 市场类事实的自动纠偏（在 type→dimensions 映射之前）=====
    # 如果 LLM 没有标成 market，但文本里明显是市场/商业内容，则强制改为 "market"
    # （避免所有市场信息都被丢在 "other" 或 "product" 里）
    if type_val not in ["market", "risk", "mitigation"]:
        if _looks_like_market_fact(text):
            type_val = "market"

    # 先把已有维度放进一个 set，后面按 type / 文本内容补充
    dim_set = set(dims_clean)

    # ===== 1. 通用 type→dimensions 映射（更激进版本，优先保证信息被多个维度看到） =====
    if type_val in ["team_member", "org_structure"]:
        # 团队成员 / 组织结构 → 明确归入 team
        dim_set.add("team")

    elif type_val == "collaboration":
        # 协作既是团队协同，也是合作策略
        dim_set.update(["team", "strategy"])

    elif type_val == "pipeline":
        # 管线：目标 + 路线，既体现 objectives，也体现 strategy
        dim_set.update(["objectives", "strategy"])

    elif type_val == "milestone":
        # 里程碑：目标的阶段性拆分 + 执行可行性
        dim_set.update(["objectives", "feasibility"])

    elif type_val == "clinical_design":
        # 临床设计：目标路径 + 方案策略 + 执行可行性
        dim_set.update(["objectives", "strategy", "feasibility"])

    elif type_val in ["market", "product"]:
        # 市场 / 产品：商业策略 + 落地可行性
        dim_set.update(["objectives", "strategy", "feasibility"])

    elif type_val in ["tech_route", "regulatory"]:
        # 技术路线 / 监管路径：典型 strategy，但对可行性也有影响
        dim_set.update(["strategy", "feasibility"])

    elif type_val == "funding_source":
        # 资金来源：既体现策略布局，也影响可行性
        dim_set.update(["strategy", "feasibility"])

    elif type_val == "ip_asset":
        # IP 资产：创新优势 + 中长期可行性
        dim_set.update(["innovation", "feasibility"])

    elif type_val == "evidence":
        # 证据：创新价值 + 可行性（证据越强，可行性越高）
        dim_set.update(["innovation", "feasibility"])

    elif type_val == "ai_model":
        # AI 模型既是创新亮点，也是策略的一部分
        dim_set.update(["innovation", "strategy"])

    elif type_val in ["resource", "budget_item", "risk", "mitigation"]:
        # 资源、预算、风险与应对 → 可行性
        dim_set.add("feasibility")

    # ===== 2. 如果还是没维度，用文本关键词再判断一轮 =====
    if not dim_set:
        inferred = _infer_dims_from_text(text)
        for d in inferred:
            dim_set.add(d)

    # ===== 3. 最终兜底：还没有，就放到 feasibility，避免彻底丢失 =====
    if not dim_set:
        dim_set.add("feasibility")

    dims_final = [d for d in dim_set if d in VALID_DIMENSIONS]

    meta = fact.get("meta", {})
    if not isinstance(meta, dict):
        meta = {}
    meta.update(
        {
            "proposal_id": proposal_id,
            "chunk_index": chunk_index,
            "char_start": start,
            "char_end": end,
        }
    )

    # ==== 计算 primary_dimension ====
    primary_dim = None
    if dims_final:
        # 简单的优先级规则：尽量按内容来，不行就取第一个
        # 你也可以自己按业务微调优先级
        priority = ["team", "objectives", "strategy", "innovation", "feasibility"]
        # 从 dims_final 里选出优先级最高的那一个
        for d in priority:
            if d in dims_final:
                primary_dim = d
                break
        if primary_dim is None:
            primary_dim = dims_final[0]
    else:
        primary_dim = "feasibility"  # 理论上不会走到这里，因为上面兜底了

    return {
        "text": text.strip(),
        "dimensions": dims_final,
        "type": type_val,
        "primary_dimension": primary_dim,
        "meta": meta,
    }

def run_extract(proposal_id: str, max_chars: int = 1800, overlap: int = 400):
    full_text = load_full_text(proposal_id)
    domain_hint = detect_domain_profile(full_text)
    print(f"[INFO] 识别提案领域: {domain_hint}")
    chunks = make_chunks(full_text, max_chars=max_chars, overlap=overlap)

    out_dir = EXTRACTED_DIR / proposal_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "raw_facts.jsonl"

    total_facts = 0
    # 统计每个维度的 fact 数量，便于 sanity check
    dim_counts = {dim: 0 for dim in VALID_DIMENSIONS}

    with out_path.open("w", encoding="utf-8") as f_out:
        for ch in chunks:
            idx = ch["index"]
            chunk_len = ch["end"] - ch["start"]
            print(f"\n[INFO] 处理 chunk {idx+1}/{len(chunks)} (chars={chunk_len})...")

            # 第一次正常抽取
            data = call_llm_for_chunk(ch["text"], domain_hint=domain_hint)
            facts = data.get("facts", [])
            if not isinstance(facts, list):
                facts = []

            # Hard cap：无论模型返回多少，单 chunk 最多保留 25 条，防止上游超量污染下游。
            if len(facts) > 25:
                print(f"[INFO] chunk {idx+1}: 原始返回 {len(facts)} 条，按 hard-cap 截断为 25 条。")
                facts = facts[:25]

            # 如果 chunk 很长，但抽取的 facts 过少，则尝试 dense 模式重跑一次
            if chunk_len >= 1200 and len(facts) < 5:
                print(
                    f"[INFO] 当前 chunk 文本较长(chars={chunk_len})，但只抽取到 {len(facts)} 条事实，"
                    f"尝试使用 dense 模式重试以提高召回..."
                )
                dense_data = call_llm_for_chunk(ch["text"], dense=True, domain_hint=domain_hint)
                dense_facts = dense_data.get("facts", [])
                if isinstance(dense_facts, list) and len(dense_facts) > len(facts):
                    if len(dense_facts) > 25:
                        print(f"[INFO] chunk {idx+1}: dense 返回 {len(dense_facts)} 条，按 hard-cap 截断为 25 条。")
                        dense_facts = dense_facts[:25]
                    print(
                        f"[INFO] dense 模式抽取到 {len(dense_facts)} 条事实（优于原先的 {len(facts)} 条），"
                        f"采用 dense 结果。"
                    )
                    facts = dense_facts
                else:
                    print(
                        f"[INFO] dense 模式未显著提升（原 {len(facts)} 条，dense={len(dense_facts)} 条），"
                        f"保留原始抽取结果。"
                    )

            normalized_list = []
            for fact in facts:
                if not isinstance(fact, dict):
                    continue

                # 先基于 chunk_text 标记 suspect_numeric
                fact = mark_numeric_suspect(fact, ch["text"])

                norm = normalize_fact(
                    fact,
                    proposal_id=proposal_id,
                    chunk_index=idx,
                    start=ch["start"],
                    end=ch["end"],
                )

                # 过滤掉空文本
                if norm["text"]:
                    normalized_list.append(norm)

            for fact in normalized_list:
                f_out.write(json.dumps(fact, ensure_ascii=False) + "\n")
                total_facts += 1
                # 更新维度计数
                for d in fact.get("dimensions", []):
                    if d in dim_counts:
                        dim_counts[d] += 1

            print(
                f"[INFO] 该 chunk 最终写入 {len(normalized_list)} 条事实，"
                f"目前累计 {total_facts} 条。"
            )

    print(f"\n[OK] 已写出事实文件: {out_path} (总事实数={total_facts})")

    # ===== 全局维度分布检查 =====
    print("\n[SUMMARY] 维度分布统计（基于 raw_facts.jsonl）：")
    for dim in VALID_DIMENSIONS:
        print(f"  - {dim}: {dim_counts[dim]} facts")

    # 简单 sanity check：如果某个维度明显偏少，打个警告（这里只做提示，不终止）
    if total_facts > 0:
        avg = total_facts / len(VALID_DIMENSIONS)
        for dim in VALID_DIMENSIONS:
            if dim_counts[dim] < max(8, 0.25 * avg):
                print(
                    f"[WARN] 维度 {dim} 的事实数仅 {dim_counts[dim]}，"
                    f"显著低于平均值 {avg:.1f}，可能存在抽取不足或映射偏差，"
                    f"建议检查 raw_facts.jsonl 或适当调整 Prompt/映射。"
                )


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1: 按 chunk 抽取原子事实（raw_facts.jsonl）"
    )
    parser.add_argument(
        "--proposal-id",
        "--proposal_id",
        dest="proposal_id",
        required=False,
        help="提案 ID（对应 src/data/prepared/<proposal_id>）",
    )
    parser.add_argument(
        "--max_chars",
        type=int,
        default=1800,
        help="每个 chunk 最大字符数（默认 1800）",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=400,
        help="chunk 之间的字符重叠数（默认 400）",
    )
    parser.add_argument(
        "--skip-inference-health-check",
        action="store_true",
        help="跳过本机推理服务连通性检查（默认在访问模型前探测 Ollama 等是否可用）",
    )
    args = parser.parse_args()

    maybe_probe_local_inference(skip=bool(args.skip_inference_health_check))

    if args.proposal_id:
        pid = args.proposal_id
    else:
        pid = get_context_proposal_id() or find_latest_prepared_proposal()

    run_extract(pid, max_chars=args.max_chars, overlap=args.overlap)


if __name__ == "__main__":
    main()
