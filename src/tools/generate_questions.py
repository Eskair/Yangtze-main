# -*- coding: utf-8 -*-
"""
Stage 3 · 维度定制化问题生成器（generate_questions.py · v3.1）

输入：
  - src/data/extracted/<proposal_id>/dimensions_v2.json

输出（两个）：
  1) 详细版（保留 qid/aspect/answer_type/links_to 等全量信息，方便调试与后续扩展）：
     - src/data/questions/<proposal_id>/generated_questions.json

  2) 简化版（专门给 llm_answering 使用，仅中文问题字符串列表）：
     - src/data/config/question_sets/generated_questions.json
     结构示例：
     {
       "proposal_id": "XXX",
       "generated_at": "...",
       "model": "...",
       "provider": "...",
       "team": {
         "dimension": "team",
         "questions": ["问题1", "问题2", "..."],
         "search_hints": ["由维度摘要/要点/风险等自动抽取的短语"],
         "source_proposal_id": "XXX"
       },
       ...
     }

核心特性（在 v3 基础上的改动）：
  - 问题必须显式“锚定”到该维度的 key_points / risks / mitigations（通过 links_to 索引）。
  - 各维度题型配比随「每维题量」自适应（默认每维 3 题）：少题量时以可审计为先。
  - 问题设计显式基于 payload 内容（不允许脱离 dimensions_v2 瞎飞）。
  - 新增：“信息缺失处理规则”和“平台/中长期视角问题”的约束，减少后续 LLM 回答时的幻觉风险。
"""

import os
import re
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List

from dotenv import load_dotenv
from openai import OpenAI
from run_context import get_context_proposal_id

load_dotenv(override=True)

from local_openai import getenv_model, make_openai_client, maybe_probe_local_inference

# ========== 路径配置 ==========

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "src" / "data"
EXTRACTED_DIR = DATA_DIR / "extracted"
QUESTIONS_DIR = DATA_DIR / "questions"
CONFIG_QS_DIR = DATA_DIR / "config" / "question_sets"

QUESTIONS_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_QS_DIR.mkdir(parents=True, exist_ok=True)

# ========== LLM 配置 ==========

OPENAI_MODEL = getenv_model()
PROVIDER = os.getenv("PROVIDER", "openai").lower()
# 中英各一条的统一字符上限（1 汉字≈1 字符；英文按字符计）。
# 需要更长评审式问题时可在 .env 设 MAX_QUESTION_CHARS=220 等。
MAX_QUESTION_CHARS = int(os.getenv("MAX_QUESTION_CHARS", "30"))


def _token_limit_kwargs(max_out_tokens: int) -> Dict[str, int]:  # updated 21-April-2026
    """Use model-compatible output token parameter across model families."""
    if "gpt-5" in OPENAI_MODEL.lower():
        return {"max_completion_tokens": int(max_out_tokens)}  # updated 21-April-2026
    return {"max_tokens": int(max_out_tokens)}

DIMENSION_NAMES = ["team", "objectives", "strategy", "innovation", "feasibility"]

# 一些“平台/中长期视角”的 aspect id，用于简单 sanity check
PLATFORM_ASPECT_IDS = {
    "platform_and_extensibility",
    "scaling_and_globalization",
}


def _clip_question_text(text: str, lang: str = "zh") -> str:
    s = str(text or "").strip()
    if len(s) <= MAX_QUESTION_CHARS:
        return s
    if lang == "en":
        return s[:MAX_QUESTION_CHARS].rstrip(" ,;:.") + "."
    return s[:MAX_QUESTION_CHARS].rstrip("，。；：;,. ") + "。"

def _looks_like_team_bio_question(q_zh: str, q_en: str) -> bool:
    """简单用关键词判断一个问题是不是在问团队履历 / 经验（仅用于 team 维度兜底）。"""
    q_zh = q_zh or ""
    q_en = (q_en or "").lower()

    kw_zh = [
        "团队", "核心成员", "项目负责人", "负责人", "pi",
        "履历", "背景", "经历", "经验",
        "实施经验", "合规经验", "产业化", "转化", "商业化",
        "带队", "主导", "项目记录", "成功案例",
    ]
    kw_en = [
        "team", "core team", "core member", "leader", "leadership",
        "principal investigator", "pi",
        "background", "track record", "experience", "experiences",
        "implementation", "compliance", "commercialization", "industrial",
    ]

    return any(k in q_zh for k in kw_zh) or any(k in q_en for k in kw_en)


def _looks_like_market_question(q_zh: str, q_en: str) -> bool:
    """简单用关键词判断一个问题是不是在问市场 / 竞争 / 定价（用于 strategy/objectives 兜底）。"""
    q_zh = q_zh or ""
    q_en = (q_en or "").lower()

    kw_zh = [
        "市场", "市场规模", "市场分析", "市场机会",
        "竞争", "竞品", "竞争对手", "竞争格局",
        "客户", "用户群体", "目标人群",
        "cagr", "增长率", "销售", "营收", "定价", "采购",
    ]
    kw_en = [
        "market", "market size", "market analysis", "market opportunity",
        "competitive", "competition", "competitor", "competitors",
        "customer", "customers", "patient population", "target population",
        "cagr", "growth", "sales", "revenue", "pricing", "reimbursement",
    ]

    return any(k in q_zh for k in kw_zh) or any(k in q_en for k in kw_en)

# ========== 维度专属配置 ==========

DIMENSION_CONFIG: Dict[str, Dict[str, Any]] = {
    "team": {
        "min_q": 3,
        "max_q": 3,
        "focus_zh": (
            "关注团队组成、核心负责人履历、工程与合规经验、跨机构合作网络、"
            "以及团队对长期项目执行的稳定性和时间投入。"
        ),
        "aspects": [
            {
                "id": "leadership_experience",
                "desc_zh": "项目负责人 / 核心 PI 的领导经验与往期重大项目执行记录",
            },
            {
                "id": "domain_expertise",
                "desc_zh": "团队在目标业务领域、工程实现与 AI 技术方面的专业深度",
            },
            {
                "id": "implementation_compliance_experience",
                "desc_zh": "团队在实施验证、质量体系与合规落地方面的经验",
            },
            {
                "id": "collaboration_network",
                "desc_zh": "国内外合作机构、产业伙伴网络及其稳定性/互补性",
            },
            {
                "id": "governance_and_decision_making",
                "desc_zh": "项目治理结构、决策机制、利益冲突管理和质量控制体系",
            },
            {
                "id": "team_capacity_and_bandwidth",
                "desc_zh": "团队当前人力负荷、项目并行数量，是否有足够时间和资源投入本项目",
            },
        ],
    },
    "objectives": {
        "min_q": 3,
        "max_q": 3,
        "focus_zh": (
            "关注项目总体目标的清晰度、分阶段里程碑、与核心业务痛点需求的匹配度、"
            "以及可量化的效果指标和可实现性。"
        ),
        "aspects": [
            {
                "id": "overall_goal_clarity",
                "desc_zh": "总体目标是否明确、聚焦，是否避免过度发散（过多不相关子目标）",
            },
            {
                "id": "unmet_need_alignment",
                "desc_zh": "项目针对的关键业务痛点 / 未满足需求的匹配程度与紧迫性",
            },
            {
                "id": "milestones_and_timeline",
                "desc_zh": "各阶段里程碑（0-12 个月、12-36 个月等）的设计是否合理可执行",
            },
            {
                "id": "outcome_and_success_metrics",
                "desc_zh": "目标是否有清晰可量化的成功指标（技术指标、质量指标、商业 KPI 等）",
            },
            {
                "id": "scope_and_prioritization",
                "desc_zh": "项目范围是否过大 / 过多管线并行，是否做了优先级排序与取舍",
            },
            {
                "id": "realism_and_ambition_balance",
                "desc_zh": "目标在雄心和可行性之间的平衡程度（是否过于理想化或过于保守）",
            },
        ],
    },
    "strategy": {
        "min_q": 3,
        "max_q": 3,
        "focus_zh": (
            "关注技术路线设计、实施落地路径、合规策略、市场与商业化路径、"
            "合作伙伴策略和数据资源利用方式。"
        ),
        "aspects": [
            {
                "id": "technical_strategy",
                "desc_zh": "技术路线（包括 AI 模型、递送系统、实验验证路径）的合理性与替代方案",
            },
            {
                "id": "implementation_development_path",
                "desc_zh": "实施落地分阶段（PoC/试点/规模化）的规划与关键假设",
            },
            {
                "id": "compliance_strategy",
                "desc_zh": "针对目标行业与产品形态的合规/标准策略（审计、认证、准入路径等）",
            },
            {
                "id": "commercialization_and_market_entry",
                "desc_zh": "商业化模式、定价策略、市场进入路径（国家/地区/场景选择）",
            },
            {
                "id": "partnership_and_business_model",
                "desc_zh": "与产业伙伴、渠道方、平台公司的合作模式（授权、联合开发、服务等）",
            },
            {
                "id": "data_and_real_world_evidence_strategy",
                "desc_zh": "对真实世界数据、队列、登记系统的利用策略，以及隐私与合规安排",
            },
            {
                "id": "scaling_and_globalization",
                "desc_zh": "项目从早期验证到大规模推广（多中心、跨国）的扩展策略",
            },
        ],
    },
    "innovation": {
        "min_q": 6,
        "max_q": 9,
        "focus_zh": (
            "关注技术/产品相对现有方案的创新性、差异化优势、知识产权布局和现有证据支撑。"
        ),
        "aspects": [
            {
                "id": "novelty_vs_state_of_art",
                "desc_zh": "相对当前国际前沿方案（药物、递送系统、AI 方法等）的真正创新点",
            },
            {
                "id": "differentiation_and_competitive_edge",
                "desc_zh": "与现有同类或替代方案相比的明确优势（疗效、安全性、成本等）",
            },
            {
                "id": "ip_and_protection",
                "desc_zh": "专利/软件著作/数据资产的保护布局，是否足以支撑中长期竞争",
            },
            {
                "id": "evidence_strength_for_innovation",
                "desc_zh": "对创新点的实验/测试/线上运行证据强度（样本量、设计质量、可重复性）",
            },
            {
                "id": "platform_and_extensibility",
                "desc_zh": "是否构成可拓展的平台（可迁移到其他场景/产品线），或只是单点创新",
            },
            {
                "id": "risk_of_obsolescence",
                "desc_zh": "技术在 3-5 年内被替代或快速过时的风险评估",
            },
        ],
    },
    "feasibility": {
        "min_q": 3,
        "max_q": 3,
        "focus_zh": (
            "关注资源与基础设施、资金与预算、实施路径、关键风险和应对措施、"
            "以及落地可行性（包括法规和支付环境）。"
        ),
        "aspects": [
            {
                "id": "resources_and_infrastructure",
                "desc_zh": "实验平台、试点场景、数据平台等资源是否充足且可长期稳定使用",
            },
            {
                "id": "funding_and_budget_planning",
                "desc_zh": "资金来源多样性、预算分配的合理性，以及后续融资/可持续性计划",
            },
            {
                "id": "operational_execution_plan",
                "desc_zh": "项目实施路径（关键任务、时间表、责任人）是否具体清晰",
            },
            {
                "id": "risk_management",
                "desc_zh": "对技术、实施、市场、合规等风险的识别与量化，以及对应缓解措施",
            },
            {
                "id": "compliance_and_procurement_feasibility",
                "desc_zh": "在目标市场完成准入审批与采购/支付流程对接的现实可行性",
            },
            {
                "id": "implementation_barriers",
                "desc_zh": "在真实业务场景中落地的阻力（用户采纳、流程变更、系统集成等）",
            },
            {
                "id": "timeline_and_resource_alignment",
                "desc_zh": "时间表与资源投入是否匹配，是否存在明显的瓶颈期或人手不足期",
            },
        ],
    },
}


# ========== 工具函数 ==========

def find_latest_extracted_proposal_id() -> str:
    if not EXTRACTED_DIR.exists():
        raise FileNotFoundError(f"未找到 extracted 目录: {EXTRACTED_DIR}")

    candidates = []
    for d in EXTRACTED_DIR.iterdir():
        if d.is_dir():
            candidates.append((d.stat().st_mtime, d.name))

    if not candidates:
        raise FileNotFoundError(f"extracted 目录下没有任何子目录: {EXTRACTED_DIR}")

    proposal_id = max(candidates, key=lambda x: x[0])[1]
    print(f"[INFO] [auto] 选中最新提案 ID: {proposal_id}")
    return proposal_id


def load_dimensions(proposal_id: str) -> Dict[str, Any]:
    path = EXTRACTED_DIR / proposal_id / "dimensions_v2.json"
    if not path.exists():
        raise FileNotFoundError(f"dimensions_v2.json 不存在，请先运行 build_dimensions_from_facts.py: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    print(f"[INFO] 读取 dimensions_v2.json 成功: {path}")
    return data


def safe_truncate(text: str, max_len: int = 4000) -> str:
    if not isinstance(text, str):
        text = str(text)
    if len(text) <= max_len:
        return text
    return text[:max_len] + " ...（后续内容已截断，仅供生成问题时参考）"


def build_search_hints_for_dimension(dim_data: Dict[str, Any], max_hints: int = 12) -> List[str]:
    """
    从 dimensions_v2 的 summary / key_points / risks / mitigations 抽取短句，
    供 post_processing 对齐加权与可选检索管线使用（与具体行业无关）。
    """
    if not isinstance(dim_data, dict):
        return []

    chunks: List[str] = []
    summary = str(dim_data.get("summary") or "").strip()
    if summary:
        chunks.append(safe_truncate(summary, 220))

    for kp in (dim_data.get("key_points") or [])[:8]:
        s = str(kp or "").strip()
        if s:
            chunks.append(safe_truncate(s, 180))

    for r in (dim_data.get("risks") or [])[:5]:
        s = str(r or "").strip()
        if s:
            chunks.append(safe_truncate(s, 180))

    for m in (dim_data.get("mitigations") or [])[:5]:
        s = str(m or "").strip()
        if s:
            chunks.append(safe_truncate(s, 180))

    seen = set()
    out: List[str] = []
    for c in chunks:
        key = c.casefold()[:120]
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(c)
        if len(out) >= max_hints:
            break
    return out


def build_dimension_payload(dim_name: str, dim_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    从 dimensions_v2.json 中构造给 LLM 的 payload，并附带 meta 统计信息。
    """
    summary = dim_data.get("summary", "") or ""
    key_points = dim_data.get("key_points", []) or []
    risks = dim_data.get("risks", []) or []
    mitigations = dim_data.get("mitigations", []) or []

    key_points_trunc = [safe_truncate(k, 400) for k in key_points[:12]]
    risks_trunc = [safe_truncate(r, 400) for r in risks[:8]]
    mitigations_trunc = [safe_truncate(m, 400) for m in mitigations[:8]]

    # ✅ 新增：把 risk_coverage 取出来，一起写进 meta
    risk_coverage = dim_data.get("risk_coverage", {})

    payload = {
        "dimension": dim_name,
        "summary": safe_truncate(summary, 1200),
        "key_points": key_points_trunc,
        "risks": risks_trunc,
        "mitigations": mitigations_trunc,
        "meta": {
            "key_points_count": len(key_points),
            "risks_count": len(risks),
            "mitigations_count": len(mitigations),
            "risk_coverage": risk_coverage,   # 👈 新增这一行
        },
    }
    return payload


def get_openai_client() -> OpenAI:
    if PROVIDER != "openai":
        print(f"[WARN] 当前仅实现 openai provider，PROVIDER={PROVIDER} 仍将使用 OpenAI。")
    return make_openai_client()


# ========== Prompt 模板（强化防幻觉 + 平台视角） ==========

QUESTION_PROMPT_TEMPLATE = """
你现在扮演“科技项目评审专家 + 问卷设计顾问”的角色，负责为一个 AI 辅助评审系统设计【某一个维度】的问题集。

【系统背景（简要）】
- 系统会先从提案中抽取五个维度的摘要：team / objectives / strategy / innovation / feasibility。
- 你当前只负责其中一个维度：{dimension_name}。
- 我会给你这个维度的摘要 payload（summary + key_points + risks + mitigations），
  你必须【围绕这些内容出题】，而不是泛泛而谈。

【当前维度】
- 维度名称（英文 key）：{dimension_name}
- 该维度需要重点关注：{dimension_focus_zh}

【该维度的 aspect 配置说明】
- 我会给你一个 JSON 数组 aspects，每个元素形如：
  {{
    "id": "leadership_experience",
    "desc_zh": "项目负责人 / 核心 PI 的领导经验与往期重大项目执行记录"
  }}
- 这些 aspects 是你可以用来“聚焦发问”的子方向。
- 你需要先阅读 payload，判断哪些 aspects 与当前提案此维度的信息最相关，然后在这些方面设计问题。
- 不要求每个 aspect 都出题，但至少应覆盖 3–5 个最关键的 aspects。
- 对于明显偏“平台/中长期扩展”的 aspects（例如 platform_and_extensibility、scaling_and_globalization），
  请至少设计 1 个从中长期或平台化视角出发的问题：
  - 如果 payload 中有相关信息，就围绕这些信息发问；
  - 如果 payload 完全没有相关信息，可以把问题设计为“评估该信息缺失带来的风险”。

【问题设计的内容绑定要求（极其重要）】
1. 你必须显式利用 payload 中的 key_points / risks / mitigations 来设计问题：
   - 至少一半的问题需要能指向一个或多个 key_points；
   - 如果 payload 中存在 risks 条目，至少设计 2 个问题专门围绕这些风险展开；
   - 如果 payload 中存在 mitigations 条目，至少设计 1 个问题评估这些应对措施的充分性。
2. 【实体名硬约束】问题文本中：
  - 严禁出现 payload 中完全没有出现过的具体公司、机构、大学、平台、产品、基金、国家或城市名称；
  - 如果确实需要提到合作方或机构，但 payload 中没有给出具体名字，只能使用
    “某国际科技公司”“某合作机构”“某科研机构”“某平台型公司”等泛指表达，不能自己编造新的名称；
   - 如果 payload 中已经出现了某个实体名称（例如某家公司的正式名称），你可以在问题中以【完全相同的写法】引用它，
     但不得新增其它实体名称，也不得为人物杜撰新的外文姓名。
3. 如果需要讨论“目标客户类型”“市场规模”“疗效提升幅度”等，但 payload 没给精确数字或具体对象：
   - 问题可以要求后续回答者“根据提案中已有信息进行定性分析或区间估计”，
   - 并在问题中明确加入类似措辞：
     “如提案未给出具体数值/名单，请在回答时先说明信息缺失，再分析其可能影响。”

【维度特定要求（team / objectives / strategy）】
- 如果当前维度是 team：
  - 至少设计 2 个高优先级（priority=1 或 2）的“简历驱动”问题，显式围绕提案中对核心成员 / 项目负责人
    的履历、既往项目经验、工程落地 / 产业化推进记录等 key_points 发问。
  - 这类问题应该要求后续回答者基于提案中已有的团队背景信息，综合判断团队推进本项目到规模化落地和商业化的能力。
- 如果当前维度是 strategy：
  - 如果 payload 的 key_points 中出现市场规模、竞争格局、客户或销售等相关内容，
    至少设计 1 个高优先级的“市场驱动”问题，用于评估项目在目标市场中的定位、竞争压力和市场进入 / 商业化策略。
  - 该问题的 links_to.key_points 中，至少要包含一个与市场分析相关的条目。
- 如果当前维度是 objectives：
  - 如果 key_points 中提到了核心痛点人群、目标用户群体、目标市场机会等内容，
    至少设计 1 个问题，把“项目目标与未满足业务需求 / 市场机会的匹配度”作为核心评估点，
    并允许在信息不足时先指出提案中的信息缺口。

【信息缺失时的处理要求】
- 你设计的问题本身要允许“提案信息不足”的情况：
  - 对于依赖具体指标、具体国家/公司/里程碑细节的问题，
    请在问题中显式加入类似提示：
    「若提案未对某一方面给出足够细节，请在回答时先说明信息缺口，再讨论其对评估的影响。」
- 严禁通过问题文本暗示“这些细节一定已经给出”，从而诱导后续回答者编造事实。

【links_to 字段的要求】
- 对于每个问题，请在 links_to 中标出它主要针对 payload 中哪些条目：
  - links_to.key_points: 使用 key_points 数组的下标列表（例如 [0,2]）；
  - links_to.risks: 使用 risks 数组的下标列表；
  - links_to.mitigations: 使用 mitigations 数组的下标列表。
- 如果某个问题主要是针对某个“缺失信息/盲区”，则可以让三个列表都为空。
- 下标从 0 开始，必须是整数。

【问题设计原则】
1. 针对性：
   - 问题必须紧扣当前维度的职责和 aspects，而不是泛泛而谈。
2. 可回答性：
   - 问题应该可以在“通用领域知识 + 维度摘要 payload”的基础上回答。
   - 避免依赖你看不到的隐性细节。
   - 不要在问题中要求回答者给出提案中完全未提到的“精确数值”或“完整公司名单”；
     如确有需要，必须附带前述“信息缺失时的处理提示”。
3. 结构化用途：
   - 问题类型（answer_type）从 ["analysis", "rating", "yes_no", "open"] 中选择：
     - "analysis": 要求给出分析性文字（例如“请分析……的主要优势和不足”）；
     - "rating": 可以在 1-5 分等尺度上打分（例如“在 1-5 分尺度上评价……的成熟度”）；
     - "yes_no": 判断类问题（建议后续附带理由说明）；
     - "open": 开放提问，不强制结构。
   - 同一维度的问题要覆盖不同 aspects，而不是十个问题都问同一个点。
4. 语言与形式（全程中文）：
   - 每个问题只输出中文题干 question_zh；不得生成英文题干或其它语种题干。
   - 问题文本中可以适当提及「该项目」「该团队」「该技术方案」等泛称，不必重复 payload 里的长段原文。

【数量与优先级要求】
- 我会给出推荐问题数量区间（见下文「生成数量提示」）；请将该区间当作硬目标，尽量贴合。
- priority 取值含义：
  - 1：高优先级（核心问题，系统一定会问）；
  - 2：中优先级（建议问）； 
  - 3：可选（在有余量时才问）。
- 若本轮总题数很少（例如每维仅 3 题），建议该维度的问题全部设为 priority=1；题量变多时再分层使用 2 与 3。

【题型配比】
- 「rating / analysis / yes_no / open」的具体数量要求一律以「生成数量提示」中的说明为准；
- 若未另附分项要求，则以 analysis 为主、rating 为辅，并根据题量下限尽量满足打分需要。

【输出 JSON 结构要求】
- 你必须输出一个 JSON 对象，顶层只有一个键 "questions"。
- "questions" 对应一个数组，数组中的每个元素是一个问题对象，结构为：

  {{
    "aspect": "string，aspect 的 id（内部标识，仍为英文 key），例如 leadership_experience",
    "question_zh": "string，仅中文的专家评审问题全文",
    "answer_type": "analysis" | "rating" | "yes_no" | "open",
    "priority": 1 | 2 | 3,
    "links_to": {{
      "key_points": [0, 2],
      "risks": [],
      "mitigations": []
    }}
  }}

- 不要输出任何 JSON 以外的文字，不要解释，不要加注释。
"""

# ========== 调用 LLM 生成问题 ==========
def call_llm_for_dimension_questions(
    client: OpenAI,
    dimension_name: str,
    dim_payload: Dict[str, Any],
    min_q: int,
    max_q: int,
    dim_config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    调用 LLM，为某个维度生成【定制化】问题列表（带 links_to）。
    """

    key_points = dim_payload.get("key_points", []) or []
    kp_cnt = len(key_points)

    # 根据信息量微调区间；题量上限很低（如每维 3 题）时不使用「至少 4 题」这类会与上限冲突的启发式。
    lo_c = int(dim_config["min_q"])
    hi_c = int(dim_config["max_q"])
    if kp_cnt >= 8:
        target_min, target_max = max(min_q, lo_c), min(max_q, hi_c)
    elif 4 <= kp_cnt <= 7:
        target_min, target_max = max(min_q, max(1, lo_c - 1)), min(max_q, hi_c)
    elif kp_cnt > 0:
        if hi_c <= 5:
            target_min, target_max = max(min_q, lo_c), min(max_q, hi_c)
        else:
            target_min, target_max = max(min_q, 4), min(max_q, hi_c - 1)
    else:
        target_min, target_max = min_q, min(max_q, hi_c)

    if target_min > target_max:
        target_min = target_max = min(hi_c, max_q)

    aspects = dim_config.get("aspects", [])
    aspects_str = json.dumps(aspects, ensure_ascii=False, indent=2)

    payload_str = json.dumps(dim_payload, ensure_ascii=False, indent=2)
    focus_zh = dim_config.get("focus_zh", "")

    # 当前维度的内容概览（给模型一个“量级感”）
    overview = dim_payload.get("meta", {})
    overview_str = json.dumps(overview, ensure_ascii=False, indent=2)

    prompt = (
        QUESTION_PROMPT_TEMPLATE
        .replace("{dimension_name}", dimension_name)
        .replace("{dimension_focus_zh}", focus_zh)
    )

    if target_max <= 4:
        _aspect_cov = (
            "- aspects 覆盖面：总题数很少时至少覆盖 **2** 个不同 aspects；信息量允许时再增加覆盖面。\n"
        )
        _ratio_cov = (
            "- 题型：至少 **1** 道 rating、至少 **2** 道 analysis；剩余题位可任选 answer_type。\n"
        )
    else:
        _aspect_cov = (
            "- aspects 覆盖面：尽量覆盖 **3–5** 个最关键 aspects"
            "（含至少 1 个平台/中长期视角 aspects，若存在）。\n"
        )
        _ratio_cov = (
            "- 题型：尽量包含至少 **2** 道 rating 与至少 **3** 道 analysis；信息极少时可按比例下调。\n"
        )

    user_content = (
        prompt
        + "\n\n=== 该维度的 aspects 配置 ===\n"
        + aspects_str
        + "\n\n=== 当前维度内容概览（统计信息）===\n"
        + overview_str
        + "\n\n=== 当前维度的摘要 payload ===\n"
        + payload_str
        + "\n\n=== 生成数量提示 ===\n"
        + f"- 本轮推荐问题数量区间：[{target_min}, {target_max}]，请将 questions 数组长度控制在该区间内（含两端）。\n"
        + _aspect_cov
        + _ratio_cov
        + "- 请严格按前述 JSON 结构输出。"
    )

    base_system_msg = {
        "role": "system",
        "content": (
            "你是一个严谨的通用技术项目评审专家，负责为评审系统设计结构化问题。"
            "默认全程使用中文：输出 JSON 中每条问题的 question_zh 必须为中文，不得输出英文或其它语种题干。"
        ),
    }

    data = None
    last_json_err = None
    for attempt in range(1, 5):
        attempt_user_content = user_content
        if attempt > 1:
            attempt_user_content += (
                "\n\n=== JSON 修复重试要求（updated 21-April-2026）===\n"
                "- 上一次输出 JSON 不完整或语法错误。\n"
                "- 本次请严格输出合法 JSON，且不要输出任何额外文字。\n"
                "- 请适度缩短每条问题文本，避免输出过长导致截断。\n"
                "- 每个维度问题数请控制在推荐区间下限附近。"
            )
        if attempt >= 3:
            attempt_user_content += (
                "\n\n=== 长度硬约束（防止截断）===\n"
                f"- 每条 question_zh 不得超过 {MAX_QUESTION_CHARS} 个字符；\n"
                "- links_to 中的下标列表保持简短；\n"
                "- 不要输出注释或 Markdown。"
            )

        messages = [
            base_system_msg,
            {
                "role": "user",
                "content": attempt_user_content,
            },
        ]

        token_budget = 2200 if attempt <= 2 else 1700
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.2 if attempt >= 2 else 0.3,
            **_token_limit_kwargs(token_budget),  # tightened 25-April-2026
        )

        raw = resp.choices[0].message.content or ""
        try:
            data = json.loads(raw)
            break
        except json.JSONDecodeError as e:
            last_json_err = e
            print(
                f"[WARN] 维度 {dimension_name} 第 {attempt}/4 次 JSON 解析失败：{e}"
            )
            preview = raw[:1200] + ("..." if len(raw) > 1200 else "")
            print("[WARN] 返回内容预览:")
            print(preview)

    if data is None:
        raise last_json_err if last_json_err else RuntimeError(
            f"维度 {dimension_name} 问题生成失败：未获得可解析 JSON。"
        )

    questions = data.get("questions", [])
    if not isinstance(questions, list):
        questions = []

    cleaned: List[Dict[str, Any]] = []
    rating_count = 0
    analysis_count = 0
    linked_to_kp_count = 0
    platform_aspect_used = 0

    for q in questions:
        if not isinstance(q, dict):
            continue

        aspect = str(q.get("aspect", "")).strip()
        q_zh = str(q.get("question_zh", "")).strip()
        answer_type = str(q.get("answer_type", "analysis")).strip().lower()
        priority = q.get("priority", 2)
        links_to = q.get("links_to") or {}

        if not q_zh:
            continue

        # 再加一道硬截断，避免极长句子在后续阶段造成解析和批量答题负担。
        q_zh = _clip_question_text(q_zh, "zh")

        if answer_type not in ["analysis", "rating", "yes_no", "open"]:
            answer_type = "analysis"

        try:
            priority_int = int(priority)
        except Exception:
            priority_int = 2
        if priority_int < 1 or priority_int > 3:
            priority_int = 2

        if not aspect:
            aspect = "general"

        # 清洗 links_to 结构
        if not isinstance(links_to, dict):
            links_to = {}
        kp_idx = links_to.get("key_points", [])
        rk_idx = links_to.get("risks", [])
        mt_idx = links_to.get("mitigations", [])

        def _clean_index_list(v, max_len: int):
            if not isinstance(v, list):
                return []
            cleaned_idx = []
            for x in v:
                try:
                    ix = int(x)
                    if 0 <= ix < max_len:   # 既要非负，又不能越界
                        cleaned_idx.append(ix)
                except Exception:
                    continue
            return cleaned_idx

        kp_idx_clean = _clean_index_list(kp_idx, len(key_points))
        rk_idx_clean = _clean_index_list(
            rk_idx,
            len(dim_payload.get("risks", []) or [])
        )
        mt_idx_clean = _clean_index_list(
            mt_idx,
            len(dim_payload.get("mitigations", []) or [])
        )

        if kp_idx_clean:
            linked_to_kp_count += 1

        if answer_type == "rating":
            rating_count += 1
        if answer_type == "analysis":
            analysis_count += 1

        if aspect in PLATFORM_ASPECT_IDS:
            platform_aspect_used += 1

        cleaned.append(
            {
                "aspect": aspect,
                "question_zh": q_zh,
                "answer_type": answer_type,
                "priority": priority_int,
                "links_to": {
                    "key_points": kp_idx_clean,
                    "risks": rk_idx_clean,
                    "mitigations": mt_idx_clean,
                },
            }
        )

    # ===== 维度特定兜底：确保至少有“简历驱动 / 市场驱动”问题 =====
    if cleaned and key_points:
        # --- team 维度：检查有没有“简历驱动”问题，没有就自动补一题 ---
        if dimension_name == "team":
            has_team_bio_q = any(
                _looks_like_team_bio_question(q.get("question_zh", ""), "")
                for q in cleaned
            )
            if not has_team_bio_q and len(cleaned) < int(dim_config["max_q"]):
                print(f"[INFO] 维度 {dimension_name}: 未检测到明显的简历驱动问题，自动补充 1 题。")
                extra_q_team = {
                    "aspect": "leadership_experience",  # 在 DIMENSION_CONFIG['team']['aspects'] 里已经存在
                    "question_zh": _clip_question_text((
                        "基于提案中对核心成员和项目负责人的教育背景、工程/产业化经验及既往重大项目记录的描述，"
                        "您如何评价该团队在将本项目推进至规模化应用和商业落地方面的整体能力？"
                        "如果提案对关键履历或可验证业绩描述不够具体，请在回答中先指出这一信息缺失，"
                        "并讨论其对评估结果的影响。"
                    ), "zh"),
                    "answer_type": "analysis",
                    "priority": 1,
                    "links_to": {
                        "key_points": list(range(len(key_points))),
                        "risks": [],
                        "mitigations": [],
                    },
                }
                cleaned.insert(0, extra_q_team)

        # --- strategy / objectives 维度：检查有没有“市场驱动”问题 ---
        if dimension_name in ("strategy", "objectives"):
            has_market_q = any(
                _looks_like_market_question(q.get("question_zh", ""), "")
                for q in cleaned
            )
            if not has_market_q and len(cleaned) < int(dim_config["max_q"]):
                print(f"[INFO] 维度 {dimension_name}: 未检测到明显的市场驱动问题，自动补充 1 题。")
                if dimension_name == "strategy":
                    aspect_id = "commercialization_and_market_entry"
                else:
                    aspect_id = "unmet_need_alignment"

                extra_q_market = {
                    "aspect": aspect_id,
                    "question_zh": _clip_question_text((
                        "结合提案中关于目标市场规模、增长率、主要竞争对手或目标客户群体的描述，"
                        "请评估本项目在所瞄准细分市场中的定位、竞争压力以及拟采用的市场进入/商业化策略是否合理。"
                        "如果提案中缺乏具体的市场规模或竞争格局数据，请在回答时先指出这一信息缺失，"
                        "并分析其对评估结论的影响。"
                    ), "zh"),
                    "answer_type": "analysis",
                    "priority": 1,
                    "links_to": {
                        "key_points": list(range(len(key_points))),
                        "risks": [],
                        "mitigations": [],
                    },
                }
                cleaned.insert(0, extra_q_market)

    # ✅ 先根据 target_min / target_max 控制问题数量（只对上限做硬控）
    if cleaned:
        for q in cleaned:
            q["question_zh"] = _clip_question_text(q.get("question_zh", ""), "zh")
            q.pop("question_en", None)
        if len(cleaned) > target_max:
            # 按 priority 排序，优先保留 1，再保留 2，最后 3
            cleaned.sort(key=lambda q: q.get("priority", 2))
            original_len = len(cleaned)
            cleaned = cleaned[:target_max]
            print(
                f"[INFO] 维度 {dimension_name}: LLM 生成 {original_len} 个问题，"
                f"已按 priority 截断为 {len(cleaned)} 个（上限 {target_max}）。"
            )
        elif len(cleaned) < target_min:
            print(
                f"[WARN] 维度 {dimension_name}: 实际只生成 {len(cleaned)} 个问题，"
                f"低于建议下限 {target_min}，如需补强建议后续人工加题。"
            )

    # 简单 sanity check：给你打日志，不强制重试
    if cleaned:
        if len(cleaned) >= 6:
            if rating_count < 2:
                print(
                    f"[WARN] 维度 {dimension_name}: rating 问题只有 {rating_count} 个，"
                    f"可能不利于后续量化打分。"
                )
            if analysis_count < 3:
                print(
                    f"[WARN] 维度 {dimension_name}: analysis 问题只有 {analysis_count} 个，"
                    f"可能不足以支撑深入文字分析。"
                )
        elif len(cleaned) <= 4:
            if rating_count < 1:
                print(
                    f"[WARN] 维度 {dimension_name}: rating 问题为 0 道，"
                    f"若需量化维度建议至少包含 1 道 rating。"
                )
            if analysis_count < 2:
                print(
                    f"[WARN] 维度 {dimension_name}: analysis 仅有 {analysis_count} 道，"
                    f"题量较少时可接受，但可能影响文字审计深度。"
                )
        if linked_to_kp_count < len(cleaned) // 2:
            print(
                f"[WARN] 维度 {dimension_name}: 仅有 {linked_to_kp_count}/{len(cleaned)} "
                f"个问题显式链接到 key_points，建议人工抽检。"
            )
        if dimension_name == "innovation" and platform_aspect_used == 0:
            print(
                f"[WARN] 维度 {dimension_name}: 未检测到使用 platform_and_extensibility 等平台视角 aspect 的问题，"
                f"建议人工检查是否需要补充平台/中长期扩展相关问题。"
            )

    return cleaned

# ========== 主流程 ==========

def run_generate_questions(
    proposal_id: str,
    min_q_per_dim: int = 3,
    max_q_per_dim: int = 3,
    log_questions: bool = False,  # updated 21-April-2026
):
    client = get_openai_client()
    dimensions = load_dimensions(proposal_id)

    all_dim_questions: Dict[str, Any] = {}

    for dim in DIMENSION_NAMES:
        dim_data = dimensions.get(dim, {})
        dim_config = DIMENSION_CONFIG.get(dim)

        if not dim_config:
            print(f"[WARN] 维度 {dim} 没有配置 DIMENSION_CONFIG，将跳过。")
            continue

        print(f"\n[INFO] 开始为维度 {dim} 生成问题 ...")

        dim_payload = build_dimension_payload(dim, dim_data)
        questions = call_llm_for_dimension_questions(
            client=client,
            dimension_name=dim,
            dim_payload=dim_payload,
            min_q=min_q_per_dim,
            max_q=max_q_per_dim,
            dim_config=dim_config,
        )

        dim_qs_with_id = []
        for idx, q in enumerate(questions, start=1):
            qid = f"{dim}_q{idx:02d}"
            item = dict(q)
            item["qid"] = qid
            item["dimension"] = dim
            dim_qs_with_id.append(item)

        print(
            f"[INFO] 维度 {dim} 生成问题数: {len(dim_qs_with_id)} "
            f"(推荐区间: [{dim_config['min_q']}, {dim_config['max_q']}])"
        )

        if log_questions and dim_qs_with_id:  # updated 21-April-2026
            print(f"[DEBUG] 维度 {dim} 生成问题明细：")
            for q in dim_qs_with_id:
                qid = q.get("qid", "")
                aspect = q.get("aspect", "general")
                answer_type = q.get("answer_type", "analysis")
                priority = q.get("priority", 2)
                q_zh = str(q.get("question_zh", "")).strip()
                print(
                    f"  - {qid} | aspect={aspect} | type={answer_type} | priority={priority}\n"
                    f"    Q: {q_zh}"
                )

        all_dim_questions[dim] = {
            "dimension": dim,
            "questions": dim_qs_with_id,
        }

    # ===== 1) 写详细版：src/data/questions/<proposal_id>/generated_questions.json =====
    out_dir = QUESTIONS_DIR / proposal_id
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_out_path = out_dir / "generated_questions.json"

    generated_at_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    detail_output_obj = {
        "proposal_id": proposal_id,
        "generated_at": generated_at_utc,
        "model": OPENAI_MODEL,
        "provider": PROVIDER,
        "dimensions": all_dim_questions,
    }

    detail_out_path.write_text(
        json.dumps(detail_output_obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n[OK] 详细问题集合已生成: {detail_out_path}")

    # ===== 2) 写简化版给 llm_answering 用：src/data/config/question_sets/generated_questions.json =====
    config_out_path = CONFIG_QS_DIR / "generated_questions.json"

    qs_simple: Dict[str, Any] = {}
    for dim in DIMENSION_NAMES:
        dim_block = all_dim_questions.get(dim, {})
        q_items = dim_block.get("questions", []) if isinstance(dim_block, dict) else []
        q_texts = [
            _clip_question_text(q.get("question_zh", ""), "zh")
            for q in q_items
            if isinstance(q, dict) and _clip_question_text(q.get("question_zh", ""), "zh")
        ]
        dim_raw = dimensions.get(dim, {}) if isinstance(dimensions, dict) else {}
        hint_list = build_search_hints_for_dimension(dim_raw, max_hints=12)

        qs_simple[dim] = {
            "dimension": dim,
            "questions": q_texts,      # llm_answering.get_q_list 会直接拿这个
            "search_hints": hint_list,
            "source_proposal_id": proposal_id,
        }

    config_obj = {
        "proposal_id": proposal_id,
        "generated_at": generated_at_utc,
        "model": OPENAI_MODEL,
        "provider": PROVIDER,
        **qs_simple,
    }

    config_out_path.write_text(
        json.dumps(config_obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[OK] 简化问题集合已生成（供 llm_answering 使用）: {config_out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Stage 3: 基于 dimensions_v2.json 为五个维度生成差异化问题集（v3.1, 带 links_to + 防幻觉）"
    )
    parser.add_argument(
        "--proposal_id",
        required=False,
        help="提案 ID（对应 src/data/extracted/<proposal_id>）",
    )
    parser.add_argument(
        "--min_q_per_dim",
        type=int,
        default=3,
        help="每个维度最少的问题数（与 DIMENSION_CONFIG、--max_q_per_dim 共同约束）",
    )
    parser.add_argument(
        "--max_q_per_dim",
        type=int,
        default=3,
        help="每个维度最多的问题数（与 DIMENSION_CONFIG、--min_q_per_dim 共同约束）",
    )
    parser.add_argument(
        "--llm_provider",
        required=False,
        help="LLM 提供商（当前仅支持 openai，默认为 .env 中的 PROVIDER）",
    )
    parser.add_argument(
        "--log_questions",
        action="store_true",
        help="打印每个维度生成的问题明细（qid/aspect/type/priority/question_zh）",
    )
    parser.add_argument(
        "--skip-inference-health-check",
        action="store_true",
        help="跳过本机推理服务连通性检查",
    )

    args = parser.parse_args()

    maybe_probe_local_inference(skip=bool(args.skip_inference_health_check))

    global PROVIDER
    if args.llm_provider:
        PROVIDER = args.llm_provider.lower()

    if args.proposal_id:
        pid = args.proposal_id
    else:
        pid = get_context_proposal_id() or find_latest_extracted_proposal_id()

    run_generate_questions(
        proposal_id=pid,
        min_q_per_dim=args.min_q_per_dim,
        max_q_per_dim=args.max_q_per_dim,
        log_questions=args.log_questions,  # updated 21-April-2026
    )


if __name__ == "__main__":
    main()
