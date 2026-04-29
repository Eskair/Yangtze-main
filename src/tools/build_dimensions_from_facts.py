# -*- coding: utf-8 -*-
"""
Stage 2 · 维度构建器（build_dimensions_from_facts.py）
----------------------------------------------------
输入：
  - src/data/extracted/<proposal_id>/raw_facts.jsonl  （Stage 1 输出）

输出：
  - src/data/extracted/<proposal_id>/dimensions_v2.json      （新的五维度文件）
  - src/data/extracted/<proposal_id>/dimension_facts.json    （按维度分组的 facts，方便后续调试与复用）
  - src/data/parsed/parsed_dimensions.clean.llm.json         （供 llm_answering 使用的全局 parsed 文件）

核心职责：
  - 按 dimensions 标签把 facts 分桶到 team/objectives/strategy/innovation/feasibility
  - 对每个维度单独调用一次 LLM，只基于该维度的事实生成：
      summary / key_points / risks / mitigations
  - 严禁凭空造事实，所有内容必须能在事实列表中找到“影子”
  - 尽量覆盖该维度下出现过的不同 type（team_member/pipeline/market/risk/...）
"""

import os
import json
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from run_context import get_context_proposal_id

load_dotenv(override=True)

from local_openai import getenv_model, make_openai_client, maybe_probe_local_inference

# 假设本文件路径：<project_root>/src/tools/build_dimensions_from_facts.py
BASE_DIR = Path(__file__).resolve().parents[2]
EXTRACTED_DIR = BASE_DIR / "src" / "data" / "extracted"
PARSED_DIR = BASE_DIR / "src" / "data" / "parsed"   # 与 llm_answering 对齐

OPENAI_MODEL = getenv_model()


def _token_limit_kwargs(max_out_tokens: int) -> Dict[str, int]:  # updated 21-April-2026
    """Use model-compatible output token parameter across model families."""
    if "gpt-5" in OPENAI_MODEL.lower():
        return {"max_completion_tokens": int(max_out_tokens)}  # updated 21-April-2026
    return {"max_tokens": int(max_out_tokens)}

DIMENSION_NAMES = ["team", "objectives", "strategy", "innovation", "feasibility"]

# 全局 client 复用（OpenAI 官方或本机兼容端点）
client = make_openai_client()


def _chat_completion_with_retry(**kwargs):
    """
    对偶发超时做轻量重试，避免整批维度构建被单次请求打断。
    可用环境变量控制：
      - LLM_API_MAX_RETRIES（默认 2，即最多 3 次）
      - LLM_API_RETRY_BACKOFF_SECONDS（默认 3.0）
    """
    max_retries = int(os.getenv("LLM_API_MAX_RETRIES", "2"))
    base_backoff = float(os.getenv("LLM_API_RETRY_BACKOFF_SECONDS", "3.0"))
    last_exc: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as e:
            last_exc = e
            is_timeout = (
                "timeout" in str(e).lower()
                or e.__class__.__name__ in ("APITimeoutError", "ReadTimeout", "TimeoutException")
            )
            if (not is_timeout) or attempt >= max_retries:
                raise
            wait_s = base_backoff * (2 ** attempt)
            print(
                f"[WARN] LLM 请求超时，准备重试 {attempt + 1}/{max_retries}，"
                f"等待 {wait_s:.1f}s ..."
            )
            time.sleep(wait_s)

    # 理论上不会走到这里；保留防御式抛错，便于调试
    raise RuntimeError(f"LLM 请求失败（重试后仍失败）：{last_exc}")

# 注意：这里用 {dimension_name} 标记占位，其它所有 { } 都是字面量 JSON 示例
# 后面用 .replace("{dimension_name}", xxx) 而不是 .format()
DIMENSION_PROMPT_TEMPLATE = """
你是一个严谨的项目评审助手，现在要基于【已经抽取好的事实列表】为某一个维度生成结构化摘要。

⚠️ 最重要的硬性约束（请逐条遵守）：
1）你只能使用我给你的事实列表中的信息，不得凭空添加任何新的机构、人名、行业对象、产品名称、技术、模型、
   管线名称、市场数字、患者数量、CAGR 或其他数字。
2）不允许使用“根据常识”“业内通常认为”“一般来说”“通常情况”“可能、大概、似乎”这类
   推测性表达，除非这些话本身就是原始事实的一部分。
3）允许对事实做“合并和压缩”，但：
   - 每一个 key_point、每一条 risk、每一条 mitigation，都应该在某条 fact 文本中能找到对应影子；
   - 不要凭空发明新的风险或对策。
4）你只能总结【当前维度】的内容，可以顺带提及与该维度强相关的信息，
   但不要跑题到其他维度上去写泛泛而谈的评价。

【当前维度】：{dimension_name}

【该维度的代表性内容示例】（只是提醒你关注点，不是让你照抄）：
- team:
  - 核心团队成员、机构、职称、研究方向、项目角色
  - 组织结构、国内外团队协同、分工比例
  - 与其他机构的合作团队构成
- objectives:
  - 总体目标（3–5 年想达到什么）
  - 分阶段里程碑（0–6/6–12/12–18/18–36 个月等）
  - 各条管线/子项目的目标
- strategy:
  - 技术路线（包括 AI/算法/实验路径）
  - 合作策略、商业模式、市场进入方案
  - 运营与国内外协同策略、监管路径
- innovation:
  - 技术/产品/模式的创新点
  - 相比现有方案的优势
  - 关键专利、独特数据/资源、验证证据
- feasibility:
  - 资源基础（实验室、平台、合作方）
  - 预算和资金规划、资金来源
  - 风险矩阵及应对措施（技术、市场、资金、法规、AI 模型等）
  - 时间表、实施路径、入组/依从性等可行性因素

【输入给你的 JSON 结构】（payload）：

payload = {
  "all_facts": [
    {
      "text": "自然语言的一条事实",
      "dimensions": ["team", "strategy"],
      "type": "team_member" 或 "pipeline" 等,
      "meta": { ... }
    },
    ...
  ],
  "risk_facts": [
    // 仅包含 type = "risk" 或 文本中明显包含风险/挑战关键词 的事实
  ],
  "mitigation_facts": [
    // 仅包含 type = "mitigation" 或 文本中明显包含应对/缓解/解决方案关键词 的事实
  ]
}

你需要做两件事：

1) 在心里把 all_facts 归类、去重、合并，找出对【当前维度】最重要、最能代表该维度情况的点。
   - 尽量覆盖当前维度下出现过的不同 type，而不是只盯着一种类型。
2) 按下面格式输出一个 JSON 对象（其中 summary、key_points、risks、mitigations 等所有自然语言字符串必须为中文）：

{
  "summary": "string，2-4 句中文，总结该维度的整体情况（必须是基于 facts 的压缩总结，而不是空泛评价）",
  "key_points": [
    "string，要点 1（尽量具体，能在至少一条 fact 中找到影子）",
    "string，要点 2",
    "... 在事实足够多时，尽量输出 6-10 条互相不重复的要点。"
  ],
  "risks": [
    "string，基于 risk_facts 中的风险信息总结；",
    "如果 risk_facts 为空，可以写：提案中关于该维度的风险信息较少/未详细说明。",
    "不要凭空发明项目没有写明的风险。",
    "..."
  ],
  "mitigations": [
    "string，对应某个风险的应对措施，必须基于 mitigation_facts 或 all_facts 中出现过的应对策略/合作机制；",
    "如果 mitigation_facts 为空且 all_facts 中也没有明显的应对措施，可以写：提案未具体说明该维度风险的应对措施。",
    "同样不要凭空发明解决方案。",
    "..."
  ]
}

【关于 key_points 数量的硬性要求】
- 如果 all_facts 的数量 ≥ 20 条，请尽量输出 6–10 条互相不重复的 key_points；
- 如果 all_facts 的数量在 8–19 条之间，可以输出 5–8 条 key_points；
- 如果 all_facts 很少（≤ 7 条），则按实际情况输出 3–5 条即可。
- 无论哪种情况，key_points 之间尽量覆盖不同类型的信息，而不是重复同一类事实。

【覆盖不同信息类型的要求】
- 请尽量覆盖当前维度下出现过的不同 type 类型，而不是重复同一类信息。
  例如：
  - team: 如果既有团队成员信息（team_member），又有组织结构（org_structure）、协作模式（collaboration），请三类都各写至少 1 条；
  - objectives: 如果既有总体目标（用 pipeline/product/milestone 描述），又有阶段里程碑（milestone），请都覆盖；
  - strategy: 如果既有技术路线（tech_route），又有市场/客户（market）、合作/商业模式（collaboration/funding_source）、监管路径（regulatory），请都覆盖；
  - innovation: 请优先包括创新点（ip_asset/ai_model/tech_route）以及已有证据（evidence）；
  - feasibility: 请尽量包括资源/能力（resource）、预算/资金（budget_item/funding_source）以及风险与应对（risk/mitigation），如有时间表/实施难度也可以提及。

【关于风险与应对的约束】
- 在生成 "risks" 时，请优先基于 risk_facts 中的事实进行归纳；
- 在生成 "mitigations" 时，请优先基于 mitigation_facts，必要时可以参考 all_facts 中明确提到的应对策略；
- 不要凭空假设项目有某种风险或应对，只能总结文本中真实出现过的内容。

补充要求：
- 不要写“根据常识”“业内通常如何如何”这种话，只能说提案文本展示出来的内容。
- 不要输出任何 JSON 以外的文字（不要解释、不要加注释）。
"""


def load_raw_facts(proposal_id: str) -> List[Dict[str, Any]]:
    path = EXTRACTED_DIR / proposal_id / "raw_facts.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"raw_facts.jsonl 不存在，请先运行 extract_facts_by_chunk.py: {path}")

    facts: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                facts.append(obj)
    print(f"[INFO] 读取 facts 数量: {len(facts)} 来自 {path}")
    return facts


def group_facts_by_dimension(facts: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    按维度标签把 facts 分桶到五个维度。
    一条 fact 可能属于多个维度，会出现在多个桶里。
    """
    grouped: Dict[str, List[Dict[str, Any]]] = {dim: [] for dim in DIMENSION_NAMES}

    for fact in facts:
        dims = fact.get("dimensions", [])
        if not isinstance(dims, list):
            continue
        for dim in dims:
            if dim in grouped:
                grouped[dim].append(fact)

    for dim in DIMENSION_NAMES:
        print(f"[INFO] 维度 {dim} 相关事实数: {len(grouped[dim])}")
    return grouped


def sort_facts_for_dimension(dimension_name: str, facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    按维度定义 type 优先级排序，保证更关键信息排在前面。
    这是纯通用规则，不依赖具体提案内容。
    """
    priority_map = {
        "team": [
            # 先看具体成员履历，再看组织结构和协作模式
            "team_member", "org_structure", "collaboration", "resource", "other"
        ],
        "objectives": [
            # 里程碑 + 管线目标仍然是核心，同时补充市场相关目标
            "milestone", "pipeline", "clinical_design", "product", "market", "other"
        ],
        "strategy": [
            # 把 market 放到最前，确保市场分析一定进入 LLM 视野
            "market", "tech_route", "product", "collaboration",
            "funding_source", "regulatory", "other"
        ],
        "innovation": [
            "ip_asset", "evidence", "ai_model", "tech_route", "product", "other"
        ],
        "feasibility": [
            "resource", "budget_item", "funding_source",
            "risk", "mitigation", "regulatory", "other"
        ],
    }
    order = priority_map.get(dimension_name, ["other"])

    def type_rank(t: str) -> int:
        return order.index(t) if t in order else len(order)

    return sorted(
        facts,
        key=lambda f: type_rank(f.get("type", "other"))
    )


def truncate_facts_for_prompt(facts: List[Dict[str, Any]], max_chars: int = 10000) -> List[Dict[str, Any]]:
    """
    为了防止单次 prompt 爆 context，对 facts 做一个简单的字符长度截断。
    按顺序累加 text，超过 max_chars 就停（meta 仍然保留）。
    """
    kept: List[Dict[str, Any]] = []
    total = 0
    for fact in facts:
        t = fact.get("text", "") or ""
        t_len = len(t)
        if total + t_len > max_chars and kept:
            break
        kept.append(fact)
        total += t_len
    return kept


# ===== 辅助：基于文本再兜底识别 risk / mitigation =====

_RISK_CN = ["风险", "挑战", "瓶颈", "不确定性", "不足", "局限", "缺陷", "障碍", "难点"]
_RISK_EN = ["risk", "risks", "challenge", "challenges", "bottleneck", "bottlenecks",
            "uncertainty", "limitation", "limitations", "weakness", "weaknesses",
            "barrier", "barriers", "issue", "issues", "difficulty", "difficulties"]

_MITIG_CN = ["应对", "缓解", "降低", "减少", "解决", "克服", "应对措施", "改进", "优化", "管控"]
_MITIG_EN = ["mitigation", "mitigate", "mitigating", "address", "addresses", "addressing",
             "solve", "solves", "solving", "overcome", "overcoming",
             "reduce", "reduces", "reducing", "decrease", "decreases", "decreasing",
             "improve", "improves", "improving", "optimize", "optimizing", "optimization"]


def _looks_like_risk(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    if any(k in text for k in _RISK_CN):
        return True
    if any(k in t for k in _RISK_EN):
        return True
    return False

def reclassify_risk_mitigation_global(facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    在进入维度构建之前，对所有 facts 做一次全局的 risk/mitigation 纠偏：
    - 如果 type 不是 risk/mitigation，但文本明显是风险/挑战，就改成 risk
    - 如果 type 不是 risk/mitigation，但文本明显是应对/缓解方案，就改成 mitigation
    - 如果本身标成 risk/mitigation 但文本看起来不像风险/应对，则降级为 other
      （必要时在 risk 和 mitigation 之间互换）
    """
    new_facts: List[Dict[str, Any]] = []

    for f in facts:
        t = f.get("type", "other") or "other"
        txt = f.get("text", "") or ""

        # 优先纠错：如果已经标成 risk/mitigation，但文本不符合，就降级/互换
        if t == "risk":
            if not _looks_like_risk(txt):
                # 文本更像应对措施，就改成 mitigation；否则降级成 other
                if _looks_like_mitigation(txt):
                    t = "mitigation"
                else:
                    t = "other"
        elif t == "mitigation":
            if not _looks_like_mitigation(txt):
                # 文本更像风险描述，就改成 risk；否则降级成 other
                if _looks_like_risk(txt):
                    t = "risk"
                else:
                    t = "other"
        else:
            # 如果原始类型既不是 risk 也不是 mitigation，就尝试“升格”
            if _looks_like_risk(txt):
                t = "risk"
            elif _looks_like_mitigation(txt):
                t = "mitigation"

        f["type"] = t
        new_facts.append(f)

    return new_facts

def _looks_like_mitigation(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    if any(k in text for k in _MITIG_CN):
        return True
    if any(k in t for k in _MITIG_EN):
        return True
    return False


def call_llm_for_dimension(dimension_name: str, facts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    对单一维度调用一次 LLM，生成 summary/key_points/risks/mitigations。
    强约束：只能基于 facts，不得脑补。
    """

    # 1) 先按 type 排序，再截断，确保更重要的信息优先被看到
    sorted_facts = sort_facts_for_dimension(dimension_name, facts)
    all_facts_for_prompt = truncate_facts_for_prompt(sorted_facts, max_chars=10000)

    # 2) 识别风险 / 对策 facts：不仅看 type，还看文本关键词
    risk_facts: List[Dict[str, Any]] = []
    mitigation_facts: List[Dict[str, Any]] = []

    for f in all_facts_for_prompt:
        t = f.get("type", "")
        txt = f.get("text", "") or ""

        # 明确标成 risk 的，或者文本看起来是在描述风险/挑战，都归入
        if t == "risk" or _looks_like_risk(txt):
            risk_facts.append(f)

        # 明确标成 mitigation 的，或者文本看起来是在描述应对/解决方案，也归入
        if t == "mitigation" or _looks_like_mitigation(txt):
            mitigation_facts.append(f)

    payload = {
        "all_facts": all_facts_for_prompt,
        "risk_facts": risk_facts,
        "mitigation_facts": mitigation_facts,
    }
    facts_json_str = json.dumps(payload, ensure_ascii=False, indent=2)

    # 用 replace，而不是 format，避免 JSON 里的 { } 被当成占位符
    prompt = DIMENSION_PROMPT_TEMPLATE.replace("{dimension_name}", dimension_name)

    messages = [
        {
            "role": "system",
            "content": "你是一个严谨的项目评审助手，只能基于给定的事实列表进行总结，不得编造。",
        },
        {
            "role": "user",
            "content": prompt + "\n\n=== payload 开始 ===\n" + facts_json_str,
        },
    ]

    resp = _chat_completion_with_retry(
        model=OPENAI_MODEL,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.0,
        **_token_limit_kwargs(2600),  # updated 21-April-2026
    )
    raw = resp.choices[0].message.content

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[WARN] 维度 {dimension_name} JSON 解析失败，原始内容如下：")
        print(raw)
        raise e

    # 兜底：保证四个字段存在
    summary = data.get("summary", "")
    if not isinstance(summary, str):
        summary = ""
    key_points = data.get("key_points", [])
    if not isinstance(key_points, list):
        key_points = []
    risks = data.get("risks", [])
    if not isinstance(risks, list):
        risks = []
    mitigations = data.get("mitigations", [])
    if not isinstance(mitigations, list):
        mitigations = []

    # 轻量兜底：facts 足够多但 key_points 太少，打日志提醒（先不强制重试）
    fact_count = len(all_facts_for_prompt)
    if fact_count >= 20 and len(key_points) < 6:
        print(
            f"[WARN] 维度 {dimension_name}: all_facts={fact_count} 但 key_points 只有 {len(key_points)} 条，"
            f"如有需要可以在此处加重试/补点逻辑。"
        )

    return {
        "summary": summary.strip(),
        "key_points": [str(x).strip() for x in key_points if str(x).strip()],
        "risks": [str(x).strip() for x in risks if str(x).strip()],
        "mitigations": [str(x).strip() for x in mitigations if str(x).strip()],
    }


def run_build(proposal_id: str):
    facts = load_raw_facts(proposal_id)
    # 全局先做一次 risk/mitigation 纠偏
    facts = reclassify_risk_mitigation_global(facts)
    grouped = group_facts_by_dimension(facts)

    # 额外输出一份 dimension_facts.json，方便后续人工检查和调试
    out_dir = EXTRACTED_DIR / proposal_id
    out_dir.mkdir(parents=True, exist_ok=True)
    dim_facts_path = out_dir / "dimension_facts.json"
    dim_facts_path.write_text(
        json.dumps(grouped, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[OK] 已写出按维度分组的 facts: {dim_facts_path}")

    dimensions_result: Dict[str, Dict[str, Any]] = {}

    for dim in DIMENSION_NAMES:
        dim_facts = grouped.get(dim, [])
        print(f"\n[INFO] 开始构建维度 {dim} ...")

        if not dim_facts:
            # 没有任何事实，给一个空壳（显式指出信息缺失）
            dimensions_result[dim] = {
                "summary": f"提案文本中关于 {dim} 维度的明确信息较少，无法做出详细总结。",
                "key_points": [],
                "risks": [f"提案中关于 {dim} 维度的细节信息较少，可能影响评估。"],
                "mitigations": ["提案未具体说明如何补充或缓解该维度信息不足的问题。"],
            }
            print(f"[INFO] 维度 {dim} 无事实，写入占位结果。")
            continue

        data = call_llm_for_dimension(dim, dim_facts)

        # ==== 风险覆盖度标记 ====
        risk_count = len(data.get("risks", []) or [])
        if risk_count == 0:
            level = "low"
            reason = "提案文本中几乎没有显式描述该维度相关的风险，系统无法进行充分的风险细化。"
        elif risk_count <= 2:
            level = "medium"
            reason = "该维度仅有少量风险相关描述，风险分析的粒度有限。"
        else:
            level = "high"
            reason = "该维度在提案中有较为丰富的风险相关描述，可以进行较细致的风险分析。"

        data["risk_coverage"] = {
            "level": level,
            "reason": reason,
            "risk_count": risk_count,
        }

        dimensions_result[dim] = data

        print(
            f"[INFO] 维度 {dim} 完成：summary_len={len(data['summary'])}, "
            f"key_points={len(data['key_points'])}, risks={len(data['risks'])}, "
            f"mitigations={len(data['mitigations'])}, "
            f"risk_coverage={level}"
        )

    # 1) 写入 per-proposal 维度文件
    out_path = out_dir / "dimensions_v2.json"
    out_path.write_text(
        json.dumps(dimensions_result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n[OK] 新版五维度文件已生成: {out_path}")

    # 2) 同时写一份全局 parsed 文件给 llm_answering 用
    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    parsed_path = PARSED_DIR / "parsed_dimensions.clean.llm.json"

    parsed_obj = {
        dim: {
            "summary": data.get("summary", ""),
            "key_points": data.get("key_points", []),
            "risks": data.get("risks", []),
            "mitigations": data.get("mitigations", []),
        }
        for dim, data in dimensions_result.items()
    }

    parsed_path.write_text(
        json.dumps(parsed_obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[OK] 已写出 parsed 维度文件供 llm_answering 使用: {parsed_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Stage 2: 基于 raw_facts.jsonl 构建五个维度的 dimensions_v2.json"
    )
    parser.add_argument(
        "--proposal-id",
        "--proposal_id",
        dest="proposal_id",
        required=False,
        help="提案 ID（对应 src/data/extracted/<proposal_id>）",
    )
    parser.add_argument(
        "--skip-inference-health-check",
        action="store_true",
        help="跳过本机推理服务连通性检查",
    )
    args = parser.parse_args()

    maybe_probe_local_inference(skip=bool(args.skip_inference_health_check))

    if args.proposal_id:
        pid = args.proposal_id
    else:
        pid = get_context_proposal_id()
        if not pid:
            # 默认用 extracted 里最新的一个子目录
            if not EXTRACTED_DIR.exists():
                raise FileNotFoundError(f"未找到 extracted 目录: {EXTRACTED_DIR}")
            candidates = [
                (d.stat().st_mtime, d.name)
                for d in EXTRACTED_DIR.iterdir()
                if d.is_dir()
            ]
            if not candidates:
                raise FileNotFoundError(f"extracted 目录下没有任何子目录: {EXTRACTED_DIR}")
            pid = max(candidates, key=lambda x: x[0])[1]
        print(f"[INFO] [auto/context] 选中提案 ID: {pid}")

    run_build(pid)


if __name__ == "__main__":
    main()
