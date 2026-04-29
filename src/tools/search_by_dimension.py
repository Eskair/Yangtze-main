# -*- coding: utf-8 -*-
"""
阶段 2：智能语义检索器（v2025.12 ProClean · strong-query / early-stop / robust-io / diagnostics）
保持下游兼容：evidence/* 命名/结构不变；新增 *_queries.json（维度投放查询记录）
"""
import os, sys, re, json, time, argparse
from pathlib import Path
from urllib.parse import urlparse
from collections import Counter, defaultdict
from dotenv import load_dotenv

load_dotenv()
CURRENT_DIR = Path(__file__).resolve().parent
SRC_ROOT = CURRENT_DIR.parent
DATA_DIR = SRC_ROOT / "data"
PROPOSAL_DIR = DATA_DIR / "extracted"
EVIDENCE_ROOT = DATA_DIR / "evidence"
CONFIG_DIR = DATA_DIR / "config"
PARSED_DIR = DATA_DIR / "parsed"
EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
if str(SRC_ROOT) not in sys.path: sys.path.insert(0, str(SRC_ROOT))

from backend.utils.model_selector import get_llm_client
from backend.retrievers.web_search import simple_search

llm = get_llm_client()
client = llm["client"]; model_name = llm["model_name"]; provider = llm["provider"]
print(f"💬 QueryGen 使用 {provider.upper()} 模型：{model_name}")
print("ℹ️ 该脚本属于可选的 legacy 检索分支；canonical 流程请使用 src/tools/run_pipeline.py")

# ===== 基本参数 =====
IGNORE_DIMS = {"proposal_id", "generated_time", "chunk_count", "coverage_estimate", "meta", "doc_meta", "run_meta"}
FIRST_ROUND_N = 4
MAX_RESULTS_PER_QUERY = 5
MAX_QUERIES_PER_DIM = 14
SLEEP_BETWEEN_QUERIES = 0.8
MIN_SAVE_EVIDENCE = 1

# 每题“学术域命中”提前停止阈值（维度自定义）
EARLY_STOP_ACADEMIC = {"strategy":4, "objectives":4, "feasibility":4, "innovation":3, "team":3}
# 维度层面的软阈值（累计达到后对后续题适度保守）
DIM_SOFT_EARLY_STOP = {"strategy":10, "objectives":10, "feasibility":10, "innovation":8, "team":8}

ACADEMIC_SITES = [
    "pubmed.ncbi.nlm.nih.gov","pmc.ncbi.nlm.nih.gov","nature.com","sciencedirect.com",
    "who.int","iso.org","ieee.org","oecd.org","nist.gov",
    "thelancet.com","bmj.com","cell.com","springer.com","biorxiv.org","medrxiv.org","arxiv.org","nejm.org",
    "w3.org"
]

DIM_HINTS = {
    "team": ["project lead", "core team", "organization", "track record", "collaboration", "governance"],
    "objectives": ["success metrics", "milestone", "timeline", "target users", "business outcome"],
    "strategy": ["go-to-market", "compliance pathway", "deployment plan", "partnership", "scaling"],
    "innovation": ["technical novelty", "benchmark", "differentiation", "ip", "architecture"],
    "feasibility": ["budget", "resource plan", "operational risk", "implementation", "reliability"]
}

def clean_query(q: str, max_len=280):
    q = re.sub(r"[，。？：；！、]", " ", q or "")
    q = re.sub(r"\s+", " ", q.strip())
    return q[:max_len]

def uniq(seq):
    seen, out = set(), []
    for x in seq:
        sx = str(x or "").strip()
        if not sx: continue
        lx = sx.lower()
        if lx not in seen:
            out.append(sx); seen.add(lx)
    return out


def detect_latest_proposal_id() -> str:
    if not PROPOSAL_DIR.exists():
        return "current_proposal"
    cands = [d for d in PROPOSAL_DIR.iterdir() if d.is_dir()]
    if not cands:
        return "current_proposal"
    cands.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return cands[0].name

def _extract_bracket_block(t: str, lch: str, rch: str):
    s, e = t.find(lch), t.rfind(rch)
    if s != -1 and e != -1 and e > s:
        frag = t[s:e+1]
        return frag.replace("“","\"").replace("”","\"")
    return None

def safe_json_loads(text: str):
    t = (text or "").strip()
    if "```" in t: t = t.replace("```json","").replace("```","").strip()
    for frag in (t, _extract_bracket_block(t,"[","]"), _extract_bracket_block(t,"{","}")):
        if not frag: continue
        try:
            obj = json.loads(frag)
            if isinstance(obj, list): return obj
            if isinstance(obj, dict) and isinstance(obj.get("questions"), list): return obj["questions"]
        except Exception: continue
    return []

def collect_entities_numbers_terms(dim_content: dict):
    ents, nums, key_terms = [], [], []
    try:
        people = dim_content.get("entities", {}).get("people", []) or []
        ents += [p.get("name","") for p in people if isinstance(p, dict)]
        ents += dim_content.get("entities", {}).get("orgs", []) or []
    except Exception: pass
    ents = [e for e in ents if isinstance(e,str) and e.strip()]

    try:
        nums = [str(n.get("value","")) for n in (dim_content.get("numbers") or [])
                if isinstance(n, dict) and n.get("value")]
    except Exception: pass
    nums = [n for n in nums if n]

    try:
        kt = dim_content.get("key_terms") or []
        if isinstance(kt, list): key_terms = [str(k) for k in kt if k]
    except Exception: pass
    return ents[:8], nums[:6], key_terms[:12]

# ---- 回退模板（保留原有）----
FALLBACK_TEMPLATES = {
    "team": [
        '"{PERSON}" ("project lead" OR "engineering lead" OR "research lead") 2018..2025',
        '"{PERSON}" ("track record" OR "previous projects") 2018..2025',
        '"{ORG}" ("team structure" OR "R&D organization") 2018..2025',
        '"{PERSON}" ORCID 2018..2025',
        '"{PERSON}" ("public profile" OR "conference" OR "patent") 2018..2025'
    ],
    "strategy": [
        '"{KEY}" ("compliance guidance" OR "standard") 2019..2025',
        '"{KEY}" ("go-to-market" OR "commercialization plan") 2019..2025',
        '"{KEY}" ("deployment strategy" OR "operating model") 2019..2025',
        '"{KEY}" ("security" OR "safety" OR "risk management") 2019..2025',
        '"{KEY}" ("international expansion" OR "localization") 2019..2025'
    ],
    "objectives": [
        '"{KEY}" ("target metric" OR "success criteria") 2019..2025',
        '"{KEY}" ("milestone" OR "roadmap") 2019..2025',
        '"{KEY}" ("business objective" OR "user outcome") 2019..2025'
    ],
    "innovation": [
        '"{KEY}" ("state of the art" OR "benchmark") 2021..2025',
        '"{KEY}" ("technical comparison" OR "baseline") 2021..2025',
        '"{KEY}" ("patent" OR "IP landscape") 2021..2025',
        '"{KEY}" ("architecture" OR "model performance") 2021..2025'
    ],
    "feasibility": [
        '"{KEY}" ("cost model" OR "budget plan") 2019..2025',
        '"{KEY}" ("implementation risk" OR "execution risk") 2019..2025',
        '"{KEY}" ("operational constraint" OR "resource bottleneck") 2019..2025'
    ]
}

def _inject_fallbacks(dimension: str, entities: list, keywords: list):
    toks = uniq((entities or []) + (keywords or []) + ["lipid nanoparticle","LNP","inhaled","aerosol","nebulizer","pulmonary","mRNA","siRNA"])
    outs = []
    for tpl in FALLBACK_TEMPLATES.get(dimension.lower(), []):
        if "{PERSON}" in tpl and entities:
            outs.append(tpl.replace("{PERSON}", entities[0]))
        elif "{ORG}" in tpl and len(entities) > 1:
            outs.append(tpl.replace("{ORG}", entities[1]))
        elif "{KEY}" in tpl and toks:
            outs.append(tpl.replace("{KEY}", toks[0]))
    return uniq(outs)

# ---- 新增：读取并展开 generated_questions.json 的 query_templates ----
def load_query_templates():
    qset_path = CONFIG_DIR / "question_sets" / "generated_questions.json"
    try:
        raw = json.loads(qset_path.read_text(encoding="utf-8"))
        return raw.get("query_templates", {}) or {}
    except Exception:
        return {}

def expand_templates_for_dim(dimension: str, templates: list, entities: list, key_terms: list, numbers: list, time_hint: str):
    people = [e for e in entities if e]
    orgs = [e for e in entities if e]
    terms = [t for t in key_terms if t]
    nums  = [n for n in numbers if n]

    p_opts = people[:2] or [""]
    o_opts = orgs[:2] or [""]
    t_opts = terms[:3] or [""]
    n_opts = nums[:2]  or [""]

    out = []
    for tpl in templates or []:
        tpl = str(tpl or "")
        for p in p_opts:
            for o in o_opts:
                for t in t_opts:
                    for n in n_opts:
                        q = tpl.replace("{PERSON}", p).replace("{ORG}", o).replace("{TERM}", t).replace("{NUM}", n)
                        q = q.replace("  ", " ").strip()
                        if time_hint and "20" in time_hint and time_hint not in q:
                            q = f'{q} {time_hint}'
                        out.append(clean_query(q))
    return uniq([q for q in out if q])[:MAX_QUERIES_PER_DIM]

def llm_generate_queries(question, context, dimension, hints=None, entities=None, numbers=None, key_terms=None):
    time_hint = "2019..2025" if dimension in ("strategy","objectives","feasibility") else "2021..2025"
    dim_hints = "; ".join(DIM_HINTS.get(dimension.lower(), []))
    hint_text = "; ".join(hints or [])
    ent_text = "; ".join(entities or [])[:240]
    num_text = ", ".join(numbers or [])[:80]
    key_text = "; ".join(key_terms or [])[:240]

    canon = ["AI","machine learning","deployment","benchmark","safety","compliance","scalability","cost"]
    must_have = []
    for t in (key_terms or []) + canon:
        tt = (t or "").lower()
        if any(k in tt for k in ["lnp","lipid nanoparticle","inhal","aerosol","pulmon","nebul","mrna","sirna"]):
            must_have.append(t)
    must_have = uniq(must_have)[:6]
    mh_text = "; ".join(must_have) if must_have else "AI/deployment/safety/compliance"

    prompt = f"""
你是AI通用检索专家。针对“问题+维度+摘要+hint+实体+数字”，生成 {FIRST_ROUND_N} 条高质量检索query。
要求：
- 每条尽量包含：实体/机构/模型名/登记号/关键数字（若有）
- 使用 site: 与时间窗（{time_hint}），中英均可，简洁可直投 Google
- 尽量包含以下主题词至少1个：{mh_text}
- 输出严格的 JSON 数组（字符串列表），仅内容，无解释
问题：{question}
维度：{dimension}
摘要：{context[:900]}
hints：{hint_text}；{dim_hints}
实体：{ent_text}
数字：{num_text}
关键词：{key_text}
"""
    try:
        rsp = client.chat.completions.create(
            model=model_name, messages=[{"role":"user","content":prompt}], temperature=0.35
        )
        content = rsp.choices[0].message.content.strip()
        queries = safe_json_loads(content)
        queries = [clean_query(q["query"] if isinstance(q, dict) and "query" in q else str(q)) for q in queries]
        zh_variants = [f"{q} 技术路线 AI 工程落地 风险与合规 {time_hint}" for q in queries]
        merged = uniq(queries + zh_variants)
        return merged[:MAX_QUERIES_PER_DIM] if merged else [question]
    except Exception as e:
        print(f"⚠️ LLM 生成 Query 失败: {e}")
        return [question]

# ===== 新增：基础 must/should 子句拼接 =====
def build_base_clause(dim_name: str, qcfg: dict, qsets_meta: dict):
    doc_policy = (qsets_meta.get("doc_policy") or {})
    must_terms = list(dict.fromkeys((qcfg.get("search", {}).get("must_terms") or []) + (doc_policy.get("must_terms") or [])))
    should_terms = list(dict.fromkeys((qcfg.get("search", {}).get("should_terms") or []) + (doc_policy.get("should_terms") or [])))

    def qwrap(t):
        t = str(t).strip()
        if not t: return ""
        return f"({t})" if " " in t and not (t.startswith('"') and t.endswith('"')) else t

    must_clause = " ".join(qwrap(t) for t in must_terms if t)
    should_clause = ""
    if should_terms:
        should_clause = " (" + " OR ".join(qwrap(t) for t in should_terms if t) + ")"
    base = (must_clause + should_clause).strip()
    return base, must_terms, should_terms

# ============ 主流程 ============

parser = argparse.ArgumentParser()
parser.add_argument("--fast", action="store_true", help="仅检索每维前2个问题（调试模式）")
parser.add_argument(
    "--proposal-id",
    "--proposal_id",
    dest="proposal_id",
    type=str,
    default="",
    help="可选：显式指定 proposal_id",
)
args = parser.parse_args()

# 1) 读取清洗后的维度（固定路径）
parsed_path = PARSED_DIR / "parsed_dimensions.clean.llm.json"
if not parsed_path.exists():
    print(f"❌ 未找到清洗后的维度文件：{parsed_path}；请先运行 build_dimensions_from_facts.py")
    sys.exit(1)

try:
    dimensions = json.loads(parsed_path.read_text(encoding="utf-8"))
except Exception as e:
    print(f"❌ 读取维度文件失败：{e}"); sys.exit(1)

# 2) 读取问题集（固定路径）
qset_path = CONFIG_DIR / "question_sets" / "generated_questions.json"
if not qset_path.exists():
    print(f"❌ 未找到问题集：{qset_path}"); sys.exit(1)
try:
    question_sets = json.loads(qset_path.read_text(encoding="utf-8"))
except Exception as e:
    print(f"❌ 读取问题集失败：{e}"); sys.exit(1)

# 3) proposal_id 解析优先级：CLI > 环境变量 > extracted 最新目录
proposal_id = (
    (args.proposal_id or "").strip()
    or os.getenv("CURRENT_PROPOSAL_ID", "").strip()
    or detect_latest_proposal_id()
)
os.environ["CURRENT_PROPOSAL_ID"] = proposal_id
print(f"📂 当前提案文件: {proposal_id}")

EVIDENCE_DIR = EVIDENCE_ROOT / proposal_id
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

# 读取顶层 query_templates（问题集里）
QUERY_TEMPLATES_ALL = question_sets.get("query_templates", {}) or {}

global_domain_counter, stats = Counter(), {}
debug_overview = {}

# 供基础子句使用的 meta
qsets_meta = question_sets.get("meta", {}) or {}

for dim, ctx in dimensions.items():
    if dim in IGNORE_DIMS or dim not in question_sets:
        continue

    print(f"\n🔹 开始维度: {dim}")
    t0 = time.time()
    qcfg = question_sets[dim]
    questions = qcfg.get("questions", []) or []
    if not questions:
        print("ℹ️ 该维度无问题，跳过。"); continue
    if args.fast:
        questions = questions[:2]; print("⚙️ Fast：仅检索前2个问题")

    # 收集维度上下文信息
    ents, nums, key_terms = collect_entities_numbers_terms(ctx or {})
    dim_evidences = []
    per_q_domain_counter = Counter()
    success, fail = 0, 0
    dim_academic_hits = 0

    all_queries_fired = []
    needed_hits = EARLY_STOP_ACADEMIC.get(dim, 3)
    dim_soft_cap = DIM_SOFT_EARLY_STOP.get(dim, 8)

    # 维度模板：来自 generated_questions.json 的 query_templates 对应维度
    dim_templates = (QUERY_TEMPLATES_ALL.get(dim, []) or [])[:10]
    templates_before = len(dim_templates)

    # ===== 新增：构造基础 must/should 子句 & 合并 search_hints =====
    base_clause, must_terms_used, should_terms_used = build_base_clause(dim, qcfg, qsets_meta)
    merged_hints = list(dict.fromkeys((qcfg.get("search_hints") or []) + (qsets_meta.get("doc_policy", {}) or {}).get("query_hints_merged", [])))[:10]

    for q in questions:
        if dim_academic_hits >= dim_soft_cap:
            print(f"🛑 维度累计学术命中已达软阈值 {dim_soft_cap}，对后续题缩减投放。")
            max_per_question = 2
        else:
            max_per_question = MAX_QUERIES_PER_DIM

        print(f"\n🧭 问题: {q}")
        # 1) LLM 生成首轮强 query
        llm_queries = llm_generate_queries(
            q, (ctx or {}).get("summary",""), dim,
            hints=merged_hints,
            entities=ents, numbers=nums, key_terms=key_terms
        )

        # 2) 维度模板占位展开（自动带入时间窗）
        time_hint = "2019..2025" if dim in ("strategy","objectives","feasibility") else "2021..2025"
        expanded_tpl = expand_templates_for_dim(dim, dim_templates, ents, key_terms, nums, time_hint)

        # 3) 回退模板
        fallbacks = _inject_fallbacks(dim, ents, [q] + DIM_HINTS.get(dim, []))

        # 4) 基于 hints 的单独查询（每个 hint 一条）
        hint_queries = []
        for h in merged_hints:
            if not h: continue
            hq = h
            if base_clause:
                hq = f"{base_clause} {h}".strip()
            hint_queries.append(clean_query(hq))

        # 合并顺序：原问题 -> LLM -> 模板 -> 回退 -> hint 单发
        merged_queries = uniq([q] + llm_queries + expanded_tpl + fallbacks + hint_queries)

        # 在每条 query 前拼接基础 must/should 子句
        if base_clause:
            merged_queries = [clean_query(f"{base_clause} {qq}") for qq in merged_queries]

        # 控制每题投放上限
        merged_queries = merged_queries[:max_per_question]

        # ---- 查询执行 ----
        per_question_academic_hits = 0
        empty_hits = 0
        for i, query in enumerate(merged_queries, start=1):
            print(f"🔍 ({i}/{len(merged_queries)}) 搜索: {query}")
            all_queries_fired.append(query)
            try:
                texts, urls = simple_search(
                    query, max_results=MAX_RESULTS_PER_QUERY,
                    dimension=dim, hints=merged_hints, source="LLM"
                )
                got = 0
                for t, u in zip(texts, urls):
                    dim_evidences.append({"query": query, "text": t, "url": u})
                    host = urlparse(u).hostname or ""
                    if host:
                        per_q_domain_counter[host] += 1
                        global_domain_counter[host] += 1
                        if any(ad in host for ad in ACADEMIC_SITES):
                            per_question_academic_hits += 1
                            dim_academic_hits += 1
                    got += 1
                if got == 0: empty_hits += 1
                success += 1 if got > 0 else 0
                fail += 1 if got == 0 else 0
            except Exception as e:
                print(f"❌ 搜索失败: {e}"); fail += 1; empty_hits += 1

            if per_question_academic_hits >= needed_hits:
                print("🛑 该题学术来源已足够，提前停止扩张。"); break
            time.sleep(SLEEP_BETWEEN_QUERIES)

        # 空击回退：若前面全空，尝试“hint/实体 词袋 + 时间窗 + site”的布尔兜底
        if per_question_academic_hits == 0 and empty_hits >= len(merged_queries):
            bag = uniq((merged_hints or []) + ents + key_terms)
            bag = [b for b in bag if len(b) >= 2][:6]
            if bag:
                bag_q = " ".join(f'"{b}"' for b in bag)
                bag_q = f'{bag_q} (site:iso.org OR site:ieee.org OR site:nist.gov OR site:oecd.org OR site:who.int) {time_hint}'
                bag_q = clean_query(bag_q)
                print(f"🧯 兜底搜索：{bag_q}")
                all_queries_fired.append(bag_q)
                try:
                    texts, urls = simple_search(
                        bag_q, max_results=MAX_RESULTS_PER_QUERY,
                        dimension=dim, hints=merged_hints, source="LLM"
                    )
                    for t, u in zip(texts, urls):
                        dim_evidences.append({"query": bag_q, "text": t, "url": u})
                        host = urlparse(u).hostname or ""
                        if host:
                            per_q_domain_counter[host] += 1
                            global_domain_counter[host] += 1
                            if any(ad in host for ad in ACADEMIC_SITES):
                                per_question_academic_hits += 1
                                dim_academic_hits += 1
                    success += 1 if texts else 0
                    fail += 1 if not texts else 0
                except Exception as e:
                    print(f"❌ 兜底失败: {e}")

    # 记录维度 query（诊断信息更丰富）
    queries_record = {
        "dimension": dim,
        "templates_loaded": templates_before,
        "templates_expanded": len(expanded_tpl) if 'expanded_tpl' in locals() else 0,
        "queries_total": len(all_queries_fired),
        "queries": all_queries_fired
    }
    (EVIDENCE_DIR / f"{dim}_queries.json").write_text(
        json.dumps(queries_record, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    if len(dim_evidences) < MIN_SAVE_EVIDENCE:
        print(f"⚠️ {dim} 维度 evidence 过少（{len(dim_evidences)}），跳过保存。")
        continue

    raw_ev = EVIDENCE_DIR / f"{dim}_evidence_raw.json"
    try:
        raw_ev.write_text(json.dumps(dim_evidences, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"⚠️ 写入原始 evidence 失败：{e}")

    elapsed = round(time.time() - t0, 1)
    success_rate = round(success / (success + fail + 1e-6), 2)
    print(f"📊 {dim} 完成：成功率 {success_rate}, evidence {len(dim_evidences)}, 耗时 {elapsed}s")

    stats[dim] = {
        "问题数": len(questions), "成功": success, "失败": fail,
        "成功率": success_rate, "Evidence条数": len(dim_evidences),
        "Top域名": per_q_domain_counter.most_common(6), "耗时(s)": elapsed,
        "维度累计学术命中": dim_academic_hits
    }
    debug_overview[dim] = {
        "queries_total": queries_record["queries_total"],
        "evidence_kept": len(dim_evidences),
        "elapsed_s": elapsed,
        "early_stop_threshold_per_question": EARLY_STOP_ACADEMIC.get(dim, 3),
        "early_stop_soft_dim": DIM_SOFT_EARLY_STOP.get(dim, 8)
    }

report = {
    "proposal_id": proposal_id, "provider": provider, "model": model_name,
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "stats": stats, "top_domains_global": global_domain_counter.most_common(12)
}
summary_path = EVIDENCE_DIR / "dimension_summary_index.json"
summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

(EVIDENCE_DIR / "dimension_debug_overview.json").write_text(
    json.dumps(debug_overview, ensure_ascii=False, indent=2), encoding="utf-8"
)

print("\n✅ 检索完成。报告保存至：", summary_path)
print("🌐 Top域名：", report["top_domains_global"])
print(f"📁 输出目录：{EVIDENCE_DIR}")
print(f"📈 生成的 combined 文件：{len(list(EVIDENCE_DIR.glob('*_combined.json')))} 个")
