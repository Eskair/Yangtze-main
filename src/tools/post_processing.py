# -*- coding: utf-8 -*-
"""
post_processing.py · Structured Candidate Post-Processing (v3.2 · no-web-hard-gates)

质量闸门：选中覆盖率、一致性（consistency_ratio）低于阈值时仅写入 warnings（告警），不否决批次；
解析成功率仅统计展示，不因「解析降级」单独卡闸门（见 derive_quality_gates）。

适配新版 llm_answering 输出：
  - 期望输入 schema: "llm_answering.v2" 或 "refined_items.v2.proposal_aware_with_general_insights"
  - 顶层结构：{"meta": {...}, "items": [ {dimension, q_index, question, candidates: [...]}, ... ]}

输出：
  - src/data/refined_answers/<pid>/postproc/metrics.json
  - src/data/refined_answers/<pid>/postproc/selected_by_question.json
  - src/data/refined_answers/<pid>/postproc/final_payload.json
  - src/data/refined_answers/<pid>/postproc/report.md（总篇幅默认不超过 MAX_AGGREGATED_REPORT_CHARS，见 report_limits.py）
  - src/data/refined_answers/<pid>/postproc/drops_debug.json
"""

import os
import re
import json
import math
import string
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
from functools import lru_cache
import re as _re
from run_context import get_context_proposal_id

from report_limits import MAX_AGGREGATED_REPORT_CHARS, clip_markdown_total_chars

# ============================ 路径与默认 ============================

ROOT = Path(__file__).resolve().parents[1]  # .../src
DATA_DIR = ROOT / "data"
REFINED_ROOT = DATA_DIR / "refined_answers"
CONF_DIR = DATA_DIR / "config" / "postproc"
QS_CONF_PATH = DATA_DIR / "config" / "question_sets" / "generated_questions.json"

# —— 与 answering 对齐的权威词表（多辖区/标准） ——
# ⚠️ 已移除 "hc"，避免把普通文本误计为 Health Canada；保留 "health canada"
AUTHORITY_TOKENS = [
    "iso 9001","iso 27001","iso 27017","iso 27018","iso 31000",
    "iec 62304","iec 62366","nist","soc 2","gdpr","hipaa","pipeda","phipa",
    "who","oecd","ieee","ietf","w3c",
    "doi","orcid","scopus","arxiv","github","gitlab",
    "uspto","epo","cnipa","wipo",
    "audit","compliance","standard","certification","governance","risk register"
]

COVERAGE_BANK = {
    "regulatory": ["iso","iec","nist","soc 2","gdpr","pipeda","phipa","hipaa","audit","compliance","standard"],
    "validation": ["benchmark","kpi","acceptance test","verification","validation"],
    "publication": ["scopus","doi","orcid","arxiv"],
    "patent": ["uspto","epo","cnipa","wipo"],
    "repo": ["github","gitlab","model card","data card"]
}

# === BEGIN PATCH · alias + tokenizer ===
# 压缩串 → 标准空格写法（用于 21CFRPart11 / ISO13485 / ICHQ10 等）
ALIAS_MAP = {
    # 法规/标准归一
    "21cfrpart11": "21 cfr part 11",
    "21cfr11": "21 cfr part 11",
    "iso13485": "iso 13485",
    "iso14971": "iso 14971",
    "iso27001": "iso 27001",
    "iso27017": "iso 27017",
    "iso27018": "iso 27018",
    "iec62304": "iec 62304",
    "iec62366": "iec 62366",
    "gamp5": "gamp 5",
    "ichq8": "ich q8",
    "ichq9": "ich q9",
    "ichq10": "ich q10",
    "pics": "pic/s",

    # 站点/域名归一
    "soc2": "soc 2",
}

# 对齐加权：权威 token 给更高权重
AUTHORITY_KEYWORDS = {
    "fda": 2.0, "ema": 2.0, "ich": 2.0, "q8": 1.4, "q9": 1.4, "q10": 1.6,
    "iso": 2.0, "13485": 2.0, "14971": 2.0, "27001": 1.6, "27017": 1.4, "27018": 1.4,
    "21": 1.1, "cfr": 2.0, "part": 1.1, "11": 1.1,
    "benchmark": 1.6, "validation": 1.8, "verification": 1.8,
    "pubmed": 2.0, "scopus": 2.0, "orcid": 1.5,
    "uspto": 2.0, "epo": 2.0, "cnipa": 2.0,
    "gdpr": 2.0, "hipaa": 2.0, "phipa": 2.0, "pipeda": 2.0,
    "gxp": 1.4, "gmp": 1.4, "glp": 1.4, "gcp": 1.4, "doi": 1.4,
}

# ==== 中文高权词（权重可按需微调）====
AUTHORITY_KEYWORDS_ZH = {
    "监管": 1.8, "合规": 1.8, "备案": 1.2, "认证": 1.6,
    "发表": 1.4, "论文": 1.6, "期刊": 1.4, "doi": 1.6, "scopus": 1.8,
    "专利": 1.8, "发明专利": 2.0, "uspto": 2.0, "epo": 2.0, "cnipa": 2.0,
    "隐私": 1.4, "安全": 1.4, "治理": 1.4, "风控": 1.4,
    "上线": 1.4, "量产": 1.4, "认证": 1.4, "iso": 2.0, "iec": 1.6,
    "标准": 1.6, "审计": 1.6, "可追溯": 1.4
}

STOPWORDS_ALIGN = {
    "the","and","of","to","for","in","on","by","with","a","an","is","are",
    "及","与","的","和","在","对","为","以及"
}

_RE_NON_ALNUM = _re.compile(r"[^a-z0-9]+", _re.IGNORECASE)

def _apply_aliases(s: str) -> str:
    """统一空格/点号，做数字↔字母断词，并用 ALIAS_MAP 展开压缩串。"""
    if not s:
        return ""
    base = str(s).lower()
    base = base.replace("\u00a0", " ").replace("\u3000", " ")

    # 先生成去标点“紧凑串”做 alias 命中
    compact = _RE_NON_ALNUM.sub("", base)
    if compact in ALIAS_MAP:
        base = ALIAS_MAP[compact]

    # 数字-字母边界插空格：21CFRPart11 -> 21 CFR Part 11
    base = _re.sub(r"([a-z])([0-9])", r"\1 \2", base)
    base = _re.sub(r"([0-9])([a-z])", r"\1 \2", base)

    # ICHQ10 → ich q10（兜底）
    base = _re.sub(r"\b(ich)\s*q\s*(\d+)\b", r"\1 q\2", base)
    # SOC2 / ISO 等压缩写法归一
    base = base.replace("soc2", "soc 2")
    return base

def _tokens_for_alignment(s: str):
    r"""切出 [a-z]+ / \d+ / 中文块 词元，并构造 bigrams。"""
    if not s:
        return set(), set()
    s = _apply_aliases(s)

    # 保留原有：把非字母数字替换为空格（英文/数字通道）
    s_en = _RE_NON_ALNUM.sub(" ", s)

    # 英文/数字 token
    toks_en = [t for t in _re.findall(r"[a-z]+|\d+", s_en) if t not in STOPWORDS_ALIGN]

    # —— 中文通道：直接抽取连续的中日韩统一表意文字（至少两个字避免噪音）——
    toks_zh = _re.findall(r"[\u4e00-\u9fff]{2,}", s)

    # 合并
    toks = toks_en + toks_zh

    unigrams = set(toks)
    bigrams = set()
    for i in range(len(toks)-1):
        bigrams.add(toks[i] + " " + toks[i+1])
    return unigrams, bigrams


def _weighted_overlap(q_uni, q_bi, c_uni, c_bi):
    """基于问题/提示 tokens 与候选 tokens 的带权重重合度，bigrams 轻度加成。"""
    if not q_uni:
        return 0.0

    def w(t):
        # 既支持英文权重，也支持中文高权词
        if t in AUTHORITY_KEYWORDS:
            return AUTHORITY_KEYWORDS[t]
        if t in AUTHORITY_KEYWORDS_ZH:
            return AUTHORITY_KEYWORDS_ZH[t]
        return 1.0

    den = sum(w(t) for t in q_uni)
    hit = sum(w(t) for t in (q_uni & c_uni))
    bonus = 0.0
    if q_bi and c_bi:
        bi_hit = len(q_bi & c_bi)
        bonus = min(0.30, 0.10 * bi_hit)
    return min(1.0, (hit / max(1e-9, den)) * (1.0 + bonus))
# === END PATCH ===


DEFAULT_CONF = {
    # ========= 题内标准化参数 =========
    "length_ref_chars": 280,
    "claims_ref": 3,
    "evidence_ref": 4,
    "jaccard_threshold": 0.30,

    # ========= 权重配置 =========
    # 说明：在“无 web 检索”模式下，align 作为主评判；
    # auth/cover 仅做轻微加分，不作为减分或硬过滤依据。
    # 👉 降低一致性权重，增加“对齐 + claims”的权重，让强维度更容易拉开
    "consistency_weight": 0.10,
    "fields_weight": {
        "length": 0.18,              # 原 0.20
        "claims": 0.23,              # 为 auth/cover 预留少量加分权重
        "evidence_count": 0.0,
        "evidence_authority": 0.04,  # 仅加分，不单独触发惩罚
        "evidence_coverage": 0.04,   # 仅加分，不单独触发惩罚
        "structure": 0.20,
        "alignment": 0.30,           # 原 0.25（更看重和问题/维度的贴合度）
        "calibrated_confidence": 0.05  # 原 0.10（置信度稍微降一点）
    },

    # ========= 维度权重 =========
    "dimension_weight": {
        "team": 1.00,
        "objectives": 1.00,
        "strategy": 1.00,
        "innovation": 1.10,
        "feasibility": 1.20
    },

    # ========= 扣分项 =========
    # 👉 所有 penalty 稍微变“软”一点，避免把分数全往 0.4 附近压扁
    "penalties": {
        "contradiction": 0.05,      # 原 0.08
        "overclaim": 0.03,          # 原 0.05
        "understructure": 0.03,     # 原 0.05
        "redline_residual": 0.06,   # 原 0.08
        "dimension_drift": 0.04     # 原 0.06
    },

    # ========= 过滤阈值（支持软窗口） =========
    # 注：权威/覆盖相关阈值只保留作兼容字段，不再触发硬过滤
    "filters": {
        # 证据相关：不再做硬门槛，仅统计
        "min_evidence_count": 0,

        # 基础结构要求
        "min_bullet_lines": 3,
        "min_median_bullet_len": 6,
        "min_structured_score": 0.05,
        "soft_window": True,

        # 对齐阈值：仅用于过滤严重跑题的垃圾回答
        "min_alignment_for_keep": 0.10,

        # 兼容字段（不再参与过滤）
        "min_auth_hits_for_keep": 0,
        "min_coverage_bins_for_keep": 0,
        "min_authority_ratio": 0.0,
        "min_coverage_ratio": 0.0,
        "dyn_align_relax_trigger": 0.0,
        "dyn_align_relax_delta": 0.0
    },

    # ========= 一致性修正参数 =========
    "consistency_correction": {
        "k_contradiction": 0.15,
        "beta": {
            "jaccard_sweetspot": [0.25, 0.65],
            "provider_calibration": {
                "deepseek": {"beta_delta": -0.01},
                "openai":   {"beta_delta":  0.00},
                "default":  {"beta_delta":  0.00}
            }
        }
    },

    # ========= 维度差异化放宽（仅过滤时用） =========
    "dimension_specific": {
        "objectives": {"min_bullet_lines": 3, "min_alignment_for_keep": 0.18},
        "strategy":   {"min_bullet_lines": 3, "min_alignment_for_keep": 0.24},
        "innovation": {"min_bullet_lines": 3, "min_alignment_for_keep": 0.18},
        "feasibility":{"min_bullet_lines": 3, "min_alignment_for_keep": 0.12}
    },

    # ========= 报告输出 =========
    "bar_symbols": 20,
    "adv_topk": 5,
    "report_width": 92,
    "unknown_warn_ratio": 0.10
}

PUNC = set(string.punctuation + "，。；：！？、（）【】《》…—-·")
DIM_ORDER = ["team", "objectives", "strategy", "innovation", "feasibility"]
DIM_LABELS_ZH = {
    "team": "团队与治理",
    "objectives": "项目目标",
    "strategy": "实施路径与战略",
    "innovation": "技术与产品创新",
    "feasibility": "资源与可行性",
    "unknown": "未归类维度",
}


def _gate_reason_zh(code: str) -> str:
    m = {
        "selected_coverage_low": "选中覆盖率偏低",
        "parse_success_low": "结构化解析成功率偏低",
        "consistency_low": "候选答案一致性偏低",
    }
    return m.get(code, code)


def _provider_label_zh(name: str) -> str:
    n = (name or "").strip().lower()
    return {"openai": "开放人工智能接口", "deepseek": "深度求索"}.get(n, name or "推理接口")

RE_DATE   = re.compile(r"\b(20\d{2}|19\d{2})([-/.]|年)\d{1,2}([-/\.日]|月)\d{1,2}\b|\b(Q[1-4]\s*-\s*20\d{2})\b", re.I)
RE_MONEY  = re.compile(r"\b(\$|USD|EUR|CNY|RMB|CAD)\s*\d{2,}(,\d{3})*(\.\d+)?\b|\b\d+(\.\d+)?\s*(million|billion|万|亿)\b", re.I)
RE_VALIDATION_ID = re.compile(r"\b([A-Z]{2,8}-\d{3,}|\d{4,}-[A-Z]{2,8})\b", re.I)
RE_DOI    = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.I)
RE_PATENT = re.compile(r"\b(US|EP|CN)\d{5,}\b|\bWO\d{7,}\b", re.I)
RE_ISO    = re.compile(r"\b(ISO|IEC)\s?\d{4,5}(-\d+)?\b", re.I)  # 兼容 IEC
RE_STDNUM = re.compile(r"\bEN\s?\d{3,5}\b|\bASTM\s?[A-Z]?\d{2,5}\b", re.I)
RE_ID_ANY = re.compile(r"(注册号|登记号|批准文号|备案号)", re.I)

# ====== 行清洗与软拼接 ======

DOMAIN_FIXES = [
    (re.compile(r"soc\s*[\.\-]?\s*2", re.I), "SOC 2"),
    (re.compile(r"iso\s*[\.\-]?\s*27001", re.I), "ISO 27001"),
    (re.compile(r"pubmed\s*[\.\-]?\s*ncbi\s*[\.\-]?\s*nlm\s*[\.\-]?\s*nih\s*[\.\-]?\s*gov", re.I), "PubMed"),
]
BULLET_NORM = re.compile(r"^\s*(?:[-*•·■▪︎▶️●]|(\d+)[\.\)]|[（(]?[一二三四五六七八九十][)）.、])\s*")

def _soft_join(text: str) -> str:
    if not text:
        return ""
    s = text
    s = re.sub(r"(\w)-\n(\w)", r"\1\2", s)
    s = re.sub(r"([A-Za-z0-9])\n([A-Za-z0-9])", r"\1 \2", s)
    s = re.sub(r"\s*\.\s*(gov|com|org|net|io)\b", r".\1", s, flags=re.I)
    for pat, rep in DOMAIN_FIXES:
        s = pat.sub(rep, s)
    return s

def _normalize_bullets(text: str) -> str:
    lines = [ln.rstrip() for ln in (text or "").splitlines()]
    out = []
    for ln in lines:
        base = BULLET_NORM.sub("", ln).strip()
        if not base:
            continue
        out.append(base)
    return "\n".join(out)

# ========= 两套清洗 =========

def sanitize_for_scoring(raw: str) -> str:
    s = raw or ""
    s = _soft_join(s)
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()

def sanitize_for_display(raw: str) -> str:
    s = raw or ""
    s = _soft_join(s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\b([A-Za-z])(?:\s+[A-Za-z]){1,3}\b",
               lambda m: m.group(0).replace(" ", ""), s)
    s = _normalize_bullets(s)
    return s.strip()


def _selected_answer_is_readable(sel: dict, *, min_chars: int = 12) -> bool:
    """
    选中答案是否有足够可读正文（与候选过滤 too_short 阈值对齐）。
    若可读，则不因 caveats 中的解析/降级关键字计入 degraded。
    """
    if not isinstance(sel, dict):
        return False
    t = sanitize_for_display(str(sel.get("answer") or "")).strip()
    return len(t) >= min_chars

def sanitize_answer(raw: str) -> str:
    return sanitize_for_display(raw)

# ====== 占位行检测 ======
PLACEHOLDER_LINE = re.compile(
    r"""^(
        [\-\–—=~\._]{2,}$                      |
        [\(\)\[\]\{\}]$                        |
        \d+\s*[\.\)]\s*\d?$                    |
        (?:原则|区间|示例|条款)(?:\s*/\s*(?:原则|区间))?$ |
        [A-Za-z]$                              |
        [\u3000\s]*$
    )""",
    re.X
)

def _placeholder_ratio(cleaned_text: str) -> float:
    if not cleaned_text:
        return 1.0
    lines = [ln.strip() for ln in cleaned_text.splitlines() if ln.strip()]
    if not lines:
        return 1.0
    bad = sum(1 for ln in lines if PLACEHOLDER_LINE.match(ln))
    return bad / max(1, len(lines))

# ============================ 工具函数（带缓存） ============================

def load_config():
    cfg_path = CONF_DIR / "config.json"
    if cfg_path.exists():
        try:
            user_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            return _merge_conf(DEFAULT_CONF, user_cfg)
        except Exception:
            pass
    return DEFAULT_CONF

def _merge_conf(base, user):
    out = dict(base)
    for k, v in (user or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            nv = dict(base[k]); nv.update(v); out[k] = nv
        else:
            out[k] = v
    return out

def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def detect_latest_pid() -> str:
    if not REFINED_ROOT.exists():
        return ""
    cands = []
    for d in REFINED_ROOT.iterdir():
        if d.is_dir() and (d / "all_refined_items.json").exists():
            cands.append((d.name, (d / "all_refined_items.json").stat().st_mtime))
    cands.sort(key=lambda x: x[1], reverse=True)
    return cands[0][0] if cands else ""

def read_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))

def write_json(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def norm01(x, lo, hi):
    """
    把 x 在线性映射到 [0,1] 之间：
      - hi <= lo 时直接返回 0
      - 小于区间下界截到 0，大于上界截到 1
    """
    if hi <= lo:
        return 0.0
    v = (x - lo) / (hi - lo)
    if v < 0:
        return 0.0
    if v > 1:
        return 1.0
    return float(v)

def safe_float(x, default=0.6):
    try:
        f = float(x)
        if math.isnan(f):
            return default
        return max(0.0, min(1.0, f))
    except Exception:
        return default

@lru_cache(maxsize=8192)
def _sanitize_cached(raw: str) -> str:
    return sanitize_for_display(raw)

@lru_cache(maxsize=8192)
def _word_tokens_cached(text: str):
    """
    词级 token：英文/数字用 [a-z0-9]+，中文取连续中日韩统一表意文字（长度≥2）。
    与语义对齐逻辑保持一致，不做词干化与停用词处理。
    """
    s = (sanitize_for_scoring(text) or "").lower()
    # 英文/数字 token
    en = re.findall(r"[a-z0-9]+", s)
    # 中文 token：连续汉字（≥2，降噪；如需更激进可改为 {1,}）
    zh = re.findall(r"[\u4e00-\u9fff]{2,}", s)
    return tuple(t for t in en + zh if t)

_tokenize_cached = _word_tokens_cached

def tokenize(text: str):
    return list(_word_tokens_cached(text))


def jaccard(a_tokens, b_tokens):
    if not a_tokens or not b_tokens:
        return 0.0
    A, B = set(a_tokens), set(b_tokens)
    inter = len(A & B)
    uni = len(A | B)
    return inter / uni if uni else 0.0

def looks_structured(ans: str) -> float:
    cleaned = sanitize_for_scoring(ans)
    lines = [ln.strip() for ln in (cleaned or "").splitlines() if ln.strip()]
    if not lines:
        return 0.0
    bulletish = re.compile(r"^(\d+[\.\)]\s+|[-*•·■▪︎▶️●]\s+|[（(]?[一二三四五六七八九十][)）.、]\s+|[A-Za-z]\))")
    bullets = sum(1 for ln in lines if bulletish.match(ln))
    ratio_b = bullets / max(1, len(lines))
    shortish = sum(1 for ln in lines if 6 <= len(ln) <= 64)
    ratio_s = shortish / max(1, len(lines))
    return max(0.0, min(1.0, 0.4 * ratio_b + 0.6 * ratio_s))

def overclaim_score(ans: str) -> float:
    terms = ["必须", "保证", "绝对", "完全", "零风险", "一定", "毫无", "不可能出错", "100%"]
    cnt = sum((ans or "").count(t) for t in terms)
    length = max(1, len(ans or ""))
    dens = cnt / length
    return max(0.0, min(1.0, dens * 200))

def contradiction_pair(a: str, b: str) -> float:
    neg = ["不", "无", "否", "不可", "禁止", "避免", "不得"]
    pos = ["可", "能", "允许", "建议", "推荐", "可行", "通过"]
    a_neg = sum((sanitize_for_scoring(a) or "").count(t) for t in neg)
    b_neg = sum((sanitize_for_scoring(b) or "").count(t) for t in neg)
    a_pos = sum((sanitize_for_scoring(a) or "").count(t) for t in pos)
    b_pos = sum((sanitize_for_scoring(b) or "").count(t) for t in pos)
    jac = jaccard(_tokenize_cached(sanitize_for_scoring(a)), _tokenize_cached(sanitize_for_scoring(b)))
    diff = abs((a_pos - a_neg) - (b_pos - b_neg))
    base = norm01(diff, 0, 20)
    penal = 1.0 - jac
    return max(0.0, min(1.0, base * penal))

def bar(value: float, n: int = 20) -> str:
    n = max(1, n)
    k = max(0, min(n, int(round(float(value) * n))))
    return "█" * k + "░" * (n - k)

def authority_ratio(hints: list) -> float:
    if not hints:
        return 0.0
    hits = 0
    for h in hints:
        s = (h or "").lower()
        if any(tok in s for tok in AUTHORITY_TOKENS):
            hits += 1
    return hits / max(1, len(hints))

def coverage_score(hints: list) -> float:
    if not hints:
        return 0.0
    cats = set()
    for h in hints:
        s = (h or "").lower()
        for cat, toks in COVERAGE_BANK.items():
            if any(tok in s for tok in toks):
                cats.add(cat)
    return len(cats) / len(COVERAGE_BANK)

def has_redline(text: str) -> bool:
    t = text or ""
    return any(p.search(t) for p in (RE_DATE, RE_MONEY, RE_VALIDATION_ID, RE_DOI, RE_PATENT, RE_ISO, RE_STDNUM, RE_ID_ANY))

# ============================ 去串味 + 权威导向 ============================

OTHER_DIMS = set(DIM_ORDER)

def _strip_cross_dim_tags(dim: str, tags: list) -> list:
    keep, seen = [], set()
    for t in tags or []:
        s = str(t or "").strip()
        if not s:
            continue
        low = s.lower()
        if low in OTHER_DIMS and low != dim.lower():
            continue
        if low in seen:
            continue
        seen.add(low)
        keep.append(s)
    return keep

def _authority_hints_from_qs(qs_cfg: dict, dim: str, limit: int = 8) -> list:
    try:
        block = qs_cfg.get(dim, {}) or {}
        hints = block.get("search_hints", []) or []
        hints = _strip_cross_dim_tags(dim, hints)
        def score(h):
            s = str(h).lower()
            return -sum(1 for t in AUTHORITY_TOKENS if t in s)
        hints = list({h: None for h in hints}.keys())
        hints.sort(key=score)
        return hints[:limit]
    except Exception:
        return []

# ===== 语义化对齐（替换原先的字面包含比对） =====
# === BEGIN PATCH · improved alignment ===
def _alignment_ratio(dim: str, auth_hints: list, answer: str, topic_tags: list, evidence_hints: list,
                     question: str = "", claims: list = None) -> float:
    """
    语义对齐评分（支持中英）：把候选语料（answer + topic_tags + evidence_hints*2 [+ question] [+ claims]）
    与每条 auth_hint 做 token 对齐。
    - 若某条 hint 切词后为空（常见于中文），跳过该条；
    - 若最终“有效切词的 hints 条数”为 0，则用题干+维度+claims 与候选语料做 token 重合（无固定常数保底）。
    """
    claims = claims or []
    # 候选语料：把 evidence_hints 重复一次，提高权重；question/claims 可控注入
    pool = [
        sanitize_for_scoring(answer or ""),
        " ".join([str(t) for t in (topic_tags or [])]),
        " ".join([str(h) for h in (evidence_hints or [])]),
        " ".join([str(h) for h in (evidence_hints or [])])
    ]
    if question:
        pool.append(str(question))
    if claims:
        pool.append(" ".join([str(c) for c in claims if str(c).strip()]))

    corpus = " \n ".join(pool)
    c_uni, c_bi = _tokens_for_alignment(corpus)

    # 无 search_hints：用题干 + 维度 与候选语料做 token 重合（claims/question 已在 corpus 中，不再重复注入）
    if not auth_hints:
        pseudo = " ".join([str(question or ""), str(dim or "")]).strip()
        q_uni, q_bi = _tokens_for_alignment(pseudo)
        if not q_uni and not q_bi:
            return min(0.35, 0.12 + 0.04 * min(12, len(c_uni))) if c_uni else 0.0
        base = _weighted_overlap(q_uni, q_bi, c_uni, c_bi)
        return max(0.0, min(1.0, 0.88 * base + 0.04))

    cover_hits = 0
    scores = []
    valid_hints = 0  # 关键：仅统计切得出 token 的 hint

    for h in auth_hints:
        q_uni, q_bi = _tokens_for_alignment(str(h))
        if not q_uni and not q_bi:
            continue
        valid_hints += 1
        s = _weighted_overlap(q_uni, q_bi, c_uni, c_bi)
        scores.append(s)
        if s >= 0.15:
            cover_hits += 1

    # 若所有 hints 切完都无有效 token，则当作“无 hint 场景”，给保底分
    if valid_hints == 0:
        return 0.5 if c_uni else 0.0

    if not scores:
        return 0.0

    coverage = cover_hits / max(1, valid_hints)
    mean_hit = sum(scores) / len(scores)
    return max(0.0, min(1.0, 0.5 * coverage + 0.5 * mean_hit))
# === END PATCH ===


def _dimension_drift_score(dim: str, answer: str, topic_tags: list, evidence_hints: list) -> float:
    tags = " ".join([str(t) for t in (topic_tags or [])]).lower()
    others = sorted(OTHER_DIMS - {dim})
    hit = sum(1 for od in others if od in tags)
    weak_corpus = (" ".join([
        sanitize_for_scoring(answer or ""),
        " ".join([str(h) for h in (evidence_hints or [])])
    ])).lower()
    weak = sum(1 for od in others if od in weak_corpus)
    score = 0.6 * (hit / max(1, len(others))) + 0.4 * (weak / max(1, len(others)))
    return max(0.0, min(1.0, score))

# =============== 证据短语 / 通识要点抽取（报告用） ===============

def _top_evidence_phrases(hints: list, topk: int = 3):
    if not hints:
        return []
    def _norm(h: str):
        s = re.sub(r"\s+", " ", (h or "")).strip()
        return s[:120]
    candidates = []
    for h in hints:
        s = _norm(str(h))
        if not s:
            continue
        score = 0
        low = s.lower()
        for tok in AUTHORITY_TOKENS:
            if tok in low:
                score += 1
        candidates.append((s, score))
    counter = Counter([c[0] for c in candidates])
    ranked = sorted(
        counter.items(),
        key=lambda x: (-max([sc for (txt, sc) in candidates if txt == x[0]]), -x[1], x[0])
    )
    return [r[0] for r in ranked[:topk]]

def _uniq_general_insights(gi_list, topk: int = 10, max_len: int = 220):
    """
    将各问答里的 general_insights 聚合成维度级“行业经验层”：
    - 去重
    - 统一空白
    - 控制长度
    """
    if not gi_list:
        return []
    seen = set()
    out = []
    for g in gi_list:
        s = re.sub(r"\s+", " ", str(g or "").strip())
        if not s:
            continue
        if s in seen:
            continue
        seen.add(s)
        if len(s) > max_len:
            s = s[:max_len].rstrip() + "…"
        out.append(s)
        if len(out) >= topk:
            break
    return out

# ============================ 打分逻辑 ============================

def _strong_alignment_bonus(ans: str, evids: list) -> float:
    if not ans and not evids:
        return 0.0
    text = (sanitize_for_scoring(ans) or "") + " " + " ".join([str(x) for x in (evids or [])])
    low = text.lower()
    strong = 0
    if ("validation" in low or "benchmark" in low or "acceptance" in low) and RE_VALIDATION_ID.search(low):
        strong += 1
    if ("pubmed" in low or "doi" in low) and re.search(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", low, re.I):
        strong += 1
    if ("uspto" in low or "epo" in low or "cnipa" in low) and re.search(r"\b(US|EP|CN)\d{5,}|WO\d{7,}\b", low, re.I):
        strong += 1
    if ("audit" in low or "compliance" in low or "iso" in low or "iec" in low) and re.search(r"\b(20\d{2}|19\d{2})\b", low, re.I):
        strong += 1
    return min(0.10, 0.03 * strong)

def score_candidate(ans_item: dict, cfg: dict, peer_tokens_list=None, dim: str = "", auth_hints: list = None, question: str = "") -> dict:
    ans = (ans_item.get("answer") or "").strip()
    claims = ans_item.get("claims") or []
    evids = ans_item.get("evidence_hints") or []
    tags  = ans_item.get("topic_tags") or []
    conf  = safe_float(ans_item.get("confidence"), 0.6)
    diag  = ans_item.get("diag") or {}

    s_len   = norm01(len(sanitize_for_scoring(ans)), 0, cfg["length_ref_chars"])
    s_clm   = norm01(len([c for c in claims if isinstance(c, str) and c.strip()]), 0, cfg["claims_ref"])
    s_evc   = norm01(len([e for e in evids if isinstance(e, str) and e.strip()]), 0, cfg["evidence_ref"])
    s_eva   = authority_ratio(evids)
    s_evg   = coverage_score(evids)
    s_str   = looks_structured(ans)

    # —— 对齐（计分阶段只注入 claims，不注入 question；可通过 A/B 关闭 claims 注入）——
    use_qc = not cfg.get("_ablate_no_question_claims", False)
    s_aln   = _alignment_ratio(dim, auth_hints or [], ans, tags, evids,
                               question=(question if use_qc else ""),
                               claims=(claims if use_qc else []))

    s_cal   = conf

    # 这里 auth_boost / cov_boost 仅用于 diagnostics，不再进入主权重
    auth_boost = min(1.0, float(diag.get("auth_hits", 0)) / 5.0) if isinstance(diag.get("auth_hits", 0), (int, float)) else 0.0
    cov_boost  = min(1.0, float(len(set(diag.get("coverage_bins") or []))) / 4.0)
    s_eva = max(s_eva, auth_boost)
    s_evg = max(s_evg, cov_boost)

    s_aln = min(1.0, s_aln + _strong_alignment_bonus(ans, evids))

    if peer_tokens_list:
        me = list(_tokenize_cached(sanitize_for_scoring(ans)))
        sims = [jaccard(me, tks) for tks in peer_tokens_list if tks is not None]
        s_con = sum(sims) / max(1, len(sims))
    else:
        s_con = 0.5

    pen = 0.0
    oc = overclaim_score(ans)
    if oc > 0.15:
        pen += cfg["penalties"]["overclaim"]
    if s_str < 0.20:
        pen += cfg["penalties"]["understructure"]
    if any(has_redline(str(c)) for c in claims):
        pen += cfg["penalties"]["redline_residual"]

    drift_raw = _dimension_drift_score(dim, ans, tags, evids)
    if drift_raw > 0:
        scale = 0.4 if drift_raw <= 0.2 else (0.7 if drift_raw <= 0.5 else 1.0)
        pen += min(cfg["penalties"]["dimension_drift"] * scale, drift_raw * cfg["penalties"]["dimension_drift"])
    if (diag.get("cross_dim") is True):
        pen += min(cfg["penalties"]["dimension_drift"], 0.02)

    fw = cfg["fields_weight"]
    fields_part = (
        fw["length"] * s_len +
        fw["claims"] * s_clm +
        fw["evidence_count"] * s_evc +
        fw["evidence_authority"] * s_eva +
        fw["evidence_coverage"] * s_evg +
        fw["structure"] * s_str +
        fw["alignment"] * s_aln +
        fw["calibrated_confidence"] * s_cal
    )

    total = (1.0 - cfg["consistency_weight"]) * fields_part + cfg["consistency_weight"] * s_con
    total = max(0.0, min(1.0, total - pen))

    alpha = max(0.0, min(1.0, fields_part - pen))

    if (dim == "innovation") and (ans_item.get("diag", {}).get("repro_signal") is True):
        total = min(1.0, total + 0.01)
        alpha = min(1.0, alpha + 0.01)

    return {
        "scores": {
            "length": s_len,
            "claims": s_clm,
            "evidence_count": s_evc,
            "evidence_authority": s_eva,
            "evidence_coverage": s_evg,
            "structure": s_str,
            "alignment": s_aln,
            "calibrated_confidence": s_cal,
            "consistency": s_con
        },
        "penalties": {
            "overclaim": oc,
            "dimension_drift": drift_raw,
            "applied": pen
        },
        "alpha": alpha,
        "total": total
    }

# —— 更强坏候选过滤：返回 (bool_bad, reason)
def _bad_candidate_with_reason(c: dict, cfg: dict, dim_name: str, auth_hints: list, q_text: str = ""):
    ans_raw = (c.get("answer") or "").strip()
    ans = sanitize_for_scoring(ans_raw)

    # 1. 明确错误 / 过短
    if c.get("error") is True:
        return True, "error"
    if len(ans) < 12:
        return True, "too_short"

    # 2. 占位/垃圾行比例：只有当非常高时才视为垃圾
    ph_ratio = _placeholder_ratio(sanitize_for_display(ans_raw))
    if ph_ratio > 0.60:
        return True, "placeholder_noise"

    # 3. 维度特定参数
    dim_conf = (cfg.get("dimension_specific") or {}).get(dim_name, {})
    min_struct  = cfg["filters"]["min_structured_score"]
    min_bullets = dim_conf.get("min_bullet_lines", cfg["filters"]["min_bullet_lines"])
    min_align   = float(dim_conf.get("min_alignment_for_keep", cfg["filters"]["min_alignment_for_keep"]))

    if dim_name == "unknown":
        # unknown 维度对齐阈值稍微低一点
        min_align = min(0.15, min_align)

    # 4. 结构评分：极差直接丢弃；略差在 soft_window 下放行
    s_struct = looks_structured(ans_raw)
    if s_struct < min_struct:
        if cfg["filters"].get("soft_window", True) and s_struct >= 0.5 * min_struct:
            pass
        else:
            return True, "poor_structure"

    # 5. 去重后的条目行（用于 bullet 数量 & 中位长度）
    _raw_lines = [ln.strip() for ln in sanitize_for_display(ans_raw).splitlines() if ln.strip()]
    seen = set()
    bullet_lines = []
    for ln in _raw_lines:
        if ln not in seen:
            seen.add(ln)
            bullet_lines.append(ln)

    # 6. 基本分点要求：不再依赖 evidence/authority 做动态放宽
    if len(bullet_lines) < min_bullets:
        if cfg["filters"].get("soft_window", True) and len(bullet_lines) >= max(1, min_bullets - 1):
            pass
        else:
            return True, "few_bullets"

    # 7. 行中位长度：过滤口号式 / 极短句堆砌
    if bullet_lines:
        lens = sorted(len(x) for x in bullet_lines)
        med = lens[len(lens)//2]
        if med < cfg["filters"]["min_median_bullet_len"]:
            if cfg["filters"].get("soft_window", True) and med >= max(2, cfg["filters"]["min_median_bullet_len"] - 2):
                pass
            else:
                return True, "too_short_lines"

    # 8. 语义对齐检查：只用来丢弃严重跑题的 candidate，不再看 evidence/authority
    use_qc = not cfg.get("_ablate_no_question_claims", False)
    aln = _alignment_ratio(
        dim_name,
        auth_hints or [],
        ans_raw,
        c.get("topic_tags") or [],
        c.get("evidence_hints") or [],
        question=(q_text if use_qc else ""),
        claims=(c.get("claims") or [] if use_qc else [])
    )

    dyn_min_align = float(min_align)
    if aln < dyn_min_align:
        # 若结构还不错，或者只是略低一点，在 soft_window 下放行
        if cfg["filters"].get("soft_window", True) and (aln >= dyn_min_align * 0.6 or s_struct >= 0.5):
            pass
        else:
            return True, "weak_alignment"

    return False, ""

def _fallback_pick(cands: list, dim: str, auth_hints: list):
    def _alignment_ratio_local(answer: str, evids: list):
        # 仅作为兜底时的快速对齐（共性 token 命中）
        corpus = (sanitize_for_scoring(answer or "")) + " " + " ".join([str(h) for h in (evids or [])])
        total = len(auth_hints) or 1
        hit = 0
        low = corpus.lower()
        for h in auth_hints or []:
            if (h or "").strip().lower() in low:
                hit += 1
        return hit / total

    def _fallback_score(c):
        ans = (c.get("answer") or "")
        evids = c.get("evidence_hints") or []
        s_str = looks_structured(ans)
        s_eva = authority_ratio(evids)
        s_evg = coverage_score(evids)
        s_evc = norm01(len([e for e in evids if str(e).strip()]), 0, 4)
        s_aln = _alignment_ratio_local(ans, evids)
        red_p = 0.0
        if len((c.get("facts_redlined") or [])) >= 3:
            red_p = 0.08
        # 兜底场景：结构 + 对齐 为主，evidence 指标只给很轻的权重
        base = 0.50 * s_str + 0.25 * s_aln + 0.15 * s_evc + 0.05 * s_eva + 0.05 * s_evg
        return max(0.0, base - red_p)

    ranked = sorted([(i, _fallback_score(c)) for i, c in enumerate(cands)], key=lambda x: -x[1])
    return ranked[0][0] if ranked else 0

def _provider_name(cand: dict) -> str:
    pv = (cand.get("provider") or "").strip().lower()
    if pv:
        return pv
    mdl = (cand.get("model") or "").lower()
    if "deepseek" in mdl:
        return "deepseek"
    if "gpt" in mdl or "openai" in mdl or "o" == mdl[:1]:
        return "openai"
    return "default"

def _beta_with_sweetspot_and_provider(beta_raw: float, avg_jac: float, avg_ctr: float, cfg: dict, provider: str) -> float:
    lo, hi = cfg["consistency_correction"]["beta"]["jaccard_sweetspot"]
    if avg_jac > hi:
        beta_adj = (1 - 0.05) * beta_raw
    elif avg_jac < lo:
        beta_adj = (1 - 0.05 * (lo - avg_jac) / max(1e-9, lo)) * beta_raw
    else:
        beta_adj = beta_raw
    cal = cfg["consistency_correction"]["beta"]["provider_calibration"].get(
        provider, cfg["consistency_correction"]["beta"]["provider_calibration"]["default"]
    )
    delta = float(cal.get("beta_delta", 0.0))
    beta_final = max(0.0, min(1.0, beta_adj + delta))
    return beta_final

def select_best_candidate(cands: list, cfg: dict, dim: str, auth_hints: list, q_text: str = "", last_provider: str = None):
    if not cands:
        return {"best": None, "all": [], "pairwise": {"avg_jaccard": 0.0, "avg_contradiction": 0.0}, "drop_stats": {}}

    drop_stats = Counter()
    cleaned = []
    for c in cands:
        bad, reason = _bad_candidate_with_reason(c, cfg, dim_name=dim, auth_hints=auth_hints, q_text=q_text)
        if bad:
            drop_stats[reason] += 1
        else:
            cleaned.append(c)

    if not cleaned:
        idx = _fallback_pick(cands, dim, auth_hints)
        cleaned = [cands[idx]]

    cands = cleaned

    tokens = [list(_tokenize_cached(sanitize_for_scoring((c.get("answer") or "")))) for c in cands]
    jac_sum = ctr_sum = 0.0
    pair_cnt = 0
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            pair_cnt += 1
            jv = jaccard(tokens[i], tokens[j])
            cv = contradiction_pair(cands[i].get("answer", ""), cands[j].get("answer", ""))
            jac_sum += jv
            ctr_sum += cv

    avg_jac = jac_sum / pair_cnt if pair_cnt else 0.0
    avg_ctr = ctr_sum / pair_cnt if pair_cnt else 0.0

    k1 = cfg["consistency_correction"]["k_contradiction"]
    beta_raw = max(0.0, min(1.0, (1 - k1 * avg_ctr) * (0.5 + 0.5 * avg_jac)))

    scored = []
    for idx, c in enumerate(cands):
        peer = [tokens[k] if k != idx else None for k in range(len(cands))]
        sc = score_candidate(c, cfg, peer_tokens_list=peer, dim=dim, auth_hints=auth_hints, question=q_text)
        pv = _provider_name(c)
        beta = _beta_with_sweetspot_and_provider(beta_raw, avg_jac, avg_ctr, cfg, provider=pv)
        final = max(0.0, min(1.0, sc.get("alpha", sc.get("total", 0.0)) * beta))
        scored.append((idx, final, sc, beta, pv))

    def _tie_key(t):
        final = t[1]
        sd = t[2].get("scores", {})
        return (
            -float(final),
            -float(sd.get("alignment", 0.0)),
            -float(sd.get("claims", 0.0)),
            -float(sd.get("structure", 0.0)),
            -float(sd.get("evidence_authority", 0.0)),
            -float(sd.get("evidence_coverage", 0.0)),
            -float(sd.get("length", 0.0)),
        )

    scored.sort(key=_tie_key)

    # === Provider 近分轮换策略（新增） ===
    def _pick_with_provider_balance(scored_list, last_pv, margin=0.02):
        """
        在与top候选分差小于 margin 的范围内，优先选择 provider != last_pv 的候选；
        若找不到，则保持原top。
        scored_list 元素结构: (idx, final_score, detail_dict, beta, provider)
        """
        if not scored_list:
            return None
        top = scored_list[0]
        if len(scored_list) == 1 or last_pv is None:
            return top
        top_score = top[1]
        # 只看前三个，避免质量抖动
        for cand in scored_list[:3]:
            _, sc_final, _, _, sc_pv = cand
            if sc_pv != last_pv and (top_score - sc_final) < margin:
                return cand
        return top

    picked_tuple = _pick_with_provider_balance(scored, last_provider)
    if picked_tuple is None:
        best_idx, final_score, best_detail, best_beta, best_provider = (
            0, 0.0, {"scores": {}, "penalties": {}, "alpha": 0.0, "total": 0.0}, 0.0, "default"
        )
    else:
        best_idx, final_score, best_detail, best_beta, best_provider = picked_tuple

    conflict_pen = cfg["penalties"]["contradiction"] * avg_ctr
    topic_reliability = max(0.0, min(1.0, best_detail.get("total", 0.0) - conflict_pen))

    out_all = []
    for idx, fscore, detail, b, pv in scored:
        item = dict(cands[idx])
        item["_score_detail"] = detail
        item["_score_total"]  = detail.get("total", 0.0)
        item["_score_alpha"]  = detail.get("alpha", 0.0)
        item["_score_final"]  = fscore
        item["_score_beta"]   = b
        item["_provider_used"] = pv
        out_all.append(item)

    return {
        "best": {
            "index": best_idx,
            "candidate": cands[best_idx] if cands else None,
            "score": {
                "alpha": best_detail.get("alpha", 0.0),
                "beta": best_beta,
                "final": final_score,
                "raw_total": best_detail.get("total", 0.0),
                "after_topic_conflict": topic_reliability,
                "avg_pairwise_jaccard": avg_jac,
                "avg_pairwise_contradiction": avg_ctr
            },
            "detail": best_detail,
            "provider": best_provider
        },
        "all": out_all,
        "pairwise": {"avg_jaccard": avg_jac, "avg_contradiction": avg_ctr},
        "drop_stats": dict(drop_stats)
    }

# ============================ 聚合与报告 ============================

def aggregate_dimensions(items: list, cfg: dict, qs_cfg: dict):
    per_question = []

    dim_bucket = defaultdict(list)
    dropped_reason_bucket = defaultdict(Counter)
    # —— 用于 provider 近分轮换 ——
    last_provider_per_dim = {d: None for d in DIM_ORDER + ["unknown"]}

    auth_map = {dim: _authority_hints_from_qs(qs_cfg, dim, limit=8) for dim in DIM_ORDER}
    auth_map.setdefault("unknown", [])

    provider_stats = Counter()
    provider_alpha = defaultdict(list)
    provider_final = defaultdict(list)

    for it in items:
        dim0 = (it.get("dimension") or "").strip().lower()
        dim = dim0 if dim0 in DIM_ORDER else "unknown"
        qidx = it.get("q_index")
        ques = (it.get("question") or "").strip()
        cands = it.get("candidates") or []

        picked = select_best_candidate(
            cands, cfg,
            dim=dim,
            auth_hints=auth_map.get(dim, []),
            q_text=ques,
            last_provider=last_provider_per_dim.get(dim)
        )
        best = picked["best"]

        # 记录本维度上一次选中的 provider，供下一题“近分轮换”微调
        if best and best.get("candidate"):
            last_provider_per_dim[dim] = (best.get("provider") or last_provider_per_dim.get(dim))

        if picked.get("drop_stats"):
            dropped_reason_bucket[dim].update(picked["drop_stats"])

        if best and best["candidate"]:
            selc = best["candidate"]
            best_item = {
                "dimension": dim,
                "q_index": qidx,
                "question": ques,
                "selected": {
                    "provider": selc.get("provider"),
                    "model": selc.get("model"),
                    "variant_id": selc.get("variant_id"),
                    "answer": selc.get("answer"),
                    "claims": selc.get("claims") or [],
                    "evidence_hints": selc.get("evidence_hints") or [],
                    "topic_tags": selc.get("topic_tags") or [],
                    "confidence": safe_float(selc.get("confidence"), 0.6),
                    "alignment_ratio": best["detail"]["scores"].get("alignment", 0.0),
                    "dimension_drift": best["detail"]["penalties"].get("dimension_drift", 0.0),
                    "facts_redlined": selc.get("facts_redlined", []) or [],
                    # ★ 新增：保留通识经验层，供后续 ai_expert_opinion 使用
                    "general_insights": selc.get("general_insights") or []
                },
                "score": best["score"],
                "score_detail": best["detail"],
                "pairwise": picked["pairwise"],
                "all_candidates": picked["all"],
                "auth_hints_used": auth_map.get(dim, []),
                "drop_stats": picked["drop_stats"],  # ← 新增
            }
            per_question.append(best_item)
            dim_bucket[dim].append(best_item)

            pv = (picked.get("best") or {}).get("provider") or "default"
            provider_stats[pv] += 1
            provider_alpha[pv].append(float(best["detail"].get("alpha", 0.0)))
            provider_final[pv].append(float(best["score"].get("final", 0.0)))
        else:
            per_question.append({
                "dimension": dim, "q_index": qidx, "question": ques,
                "selected": None,
                "score": {"alpha": 0.0, "beta": 0.0, "final": 0.0,
                          "raw_total": 0.0, "after_topic_conflict": 0.0,
                          "avg_pairwise_jaccard": 0.0, "avg_pairwise_contradiction": 0.0},
                "score_detail": {},
                "pairwise": {"avg_jaccard": 0.0, "avg_contradiction": 0.0},
                "all_candidates": [],
                "auth_hints_used": auth_map.get(dim, []),
                "drop_stats": picked["drop_stats"]  # ← 新增
            })

    per_dimension = {}
    dims_for_report = list(DIM_ORDER)
    if "unknown" in dim_bucket:
        dims_for_report.append("unknown")

    drop_reasons_global = Counter()
    for d, cnts in dropped_reason_bucket.items():
        drop_reasons_global.update(cnts)

    for dim in dims_for_report:
        qs = sorted([x for x in per_question if x["dimension"] == dim], key=lambda z: z["q_index"])
        if not qs:
            per_dimension[dim] = {
                "avg": 0.0, "n": 0,
                "avg_alignment": 0.0, "avg_drift": 0.0,
                "strengths": [], "risks": [], "snippets": [],
                "auth_hints": auth_map.get(dim, []),
                "redlined_samples": [],
                "dropped_reasons": {},
                "top_evidence_phrases": [],
                "general_insights": [],
                "explain": {
                    "top_contributors": [],
                    "top_penalties": []
                }
            }
            continue

        # ✅ 安全取每题 final 分数（缺就当 0）
        scores_final = []
        for q in qs:
            score_block = q.get("score") or {}
            val = score_block.get("final", score_block.get("after_topic_conflict", 0.0))
            try:
                scores_final.append(float(val))
            except (TypeError, ValueError):
                scores_final.append(0.0)

        # ✅ 新逻辑：均值 + 最大值 加权，拉开维度差异
        if scores_final:
            mean_sc = sum(scores_final) / len(scores_final)
            max_sc = max(scores_final)
            # 强维度通常会有几题明显高分；弱维度整体偏平
            avg = 0.6 * mean_sc + 0.4 * max_sc
        else:
            mean_sc = 0.0
            max_sc = 0.0
            avg = 0.0

        strengths, risks, snippets = [], [], []
        aln_vals, drf_vals = [], []
        redlined_samples = []

        evid_pool = []
        gi_pool = []  # ★ 新增：聚合 general_insights
        contrib_counter = Counter()
        penalty_counter = Counter()

        for q in qs:
            sel = q.get("selected")
            if sel:
                eva = authority_ratio(sel.get("evidence_hints") or [])
                evg = coverage_score(sel.get("evidence_hints") or [])
                aln = float(sel.get("alignment_ratio", 0.0))
                drf = float(sel.get("dimension_drift", 0.0))

                aln_vals.append(aln)
                drf_vals.append(drf)

                # 优势判断：不再强依赖权威/覆盖，只要 claims + 对齐即可
                if len(sel.get("claims") or []) >= 2 and aln >= 0.40:
                    strengths.append(
                        f"Q{q['q_index']}：要点充分，结构/对齐较好（auth={eva:.2f}, cover={evg:.2f}, align={aln:.2f}）"
                    )
                if drf > 0.0:
                    risks.append(f"Q{q['q_index']}：跨维度串味迹象（drift={drf:.2f}），建议人工复核维度边界")
                if (q.get("score_detail") or {}).get("penalties", {}).get("applied", 0.0) > 0.0:
                    risks.append("存在过度断言/结构不足/红线残留等扣分（请抽检）")

                ans_txt = sel.get("answer") or ""
                clean_snip = sanitize_for_display(ans_txt)
                snippets.append(f"Q{q['q_index']}：{(clean_snip[:160] + '…') if clean_snip else '（无）'}")

                for s in (sel.get("facts_redlined") or [])[:2]:
                    if s not in redlined_samples and len(redlined_samples) < 8:
                        redlined_samples.append(s)

                evid_pool.extend(sel.get("evidence_hints") or [])
                gi_pool.extend(sel.get("general_insights") or [])  # ★ 新增

                sd = q.get("score_detail", {}).get("scores", {})
                rank_pairs = sorted(sd.items(), key=lambda x: -float(x[1]))[:2]
                for k, _ in rank_pairs:
                    contrib_counter[k] += 1
                pen = q.get("score_detail", {}).get("penalties", {})
                if pen.get("applied", 0.0) > 0.0:
                    if pen.get("dimension_drift", 0.0) > 0:
                        penalty_counter["dimension_drift"] += 1
                    if pen.get("overclaim", 0.0) > 0.15:
                        penalty_counter["overclaim"] += 1
                    if sd.get("structure", 1.0) < 0.2:
                        penalty_counter["understructure"] += 1
            else:
                risks.append(f"Q{q['q_index']}：无有效答案")
                snippets.append(f"Q{q['q_index']}：（无）")

        top_evid = _top_evidence_phrases(evid_pool, topk=3)
        gi_agg = _uniq_general_insights(gi_pool, topk=10)

        per_dimension[dim] = {
            "avg": avg,
            "n": len(qs),
            "avg_alignment": (sum(aln_vals) / max(1, len(aln_vals))) if aln_vals else 0.0,
            "avg_drift": (sum(drf_vals) / max(1, len(drf_vals))) if drf_vals else 0.0,
            "strengths": strengths[:cfg["adv_topk"]],
            "risks": risks[:cfg["adv_topk"]],
            "snippets": snippets[:6],
            "auth_hints": auth_map.get(dim, []),
            "redlined_samples": redlined_samples,
            "dropped_reasons": dict(dropped_reason_bucket.get(dim, {})),
            "top_evidence_phrases": top_evid,
            "general_insights": gi_agg,
            "explain": {
                "top_contributors": [k for k, _ in contrib_counter.most_common(3)],
                "top_penalties": [k for k, _ in penalty_counter.most_common(3)]
            }
        }

    # overall
    dim_score = 0.0
    weight_sum = 0.0
    total_q_count = len(per_question)
    unknown_q_count = len([1 for q in per_question if q["dimension"] == "unknown"])
    for dim, info in per_dimension.items():
        if dim not in DIM_ORDER:
            continue
        w = (cfg.get("dimension_weight") or {}).get(dim, 1.0)
        dim_score += w * float(info.get("avg", 0.0))
        weight_sum += w
    overall_score = dim_score / max(1e-9, weight_sum if weight_sum > 0 else 1.0)

    all_conf, all_jac, all_ctr, cnt = [], 0.0, 0.0, 0
    selected_count = 0
    degraded_count = 0
    for q in per_question:
        if q.get("selected"):
            selected_count += 1
            all_conf.append(safe_float(q["selected"].get("confidence"), 0.6))
            cv = str(q["selected"].get("caveats", "") or "").lower()
            if any(k in cv for k in ("降级", "fallback", "non_json", "schema_validation_failed", "parse_failed")):
                if not _selected_answer_is_readable(q["selected"]):
                    degraded_count += 1
        all_jac += float(q.get("pairwise", {}).get("avg_jaccard", 0.0))
        all_ctr += float(q.get("pairwise", {}).get("avg_contradiction", 0.0))
        cnt += 1
    mean_conf = sum(all_conf) / max(1, len(all_conf))
    mean_jac = all_jac / max(1, cnt)
    mean_ctr = all_ctr / max(1, cnt)
    overall_confidence = max(0.0, min(1.0, (0.5 * mean_conf + 0.3 * mean_jac + 0.2 * (1 - mean_ctr))))

    provider_summary = {}
    for pv, n in provider_stats.items():
        provider_summary[pv] = {
            "selected_count": n,
            "avg_alpha": sum(provider_alpha[pv]) / max(1, len(provider_alpha[pv])),
            "avg_final": sum(provider_final[pv]) / max(1, len(provider_final[pv]))
        }

    drop_g = dict(drop_reasons_global)
    placeholder_cnt = drop_g.get("placeholder_noise", 0)
    few_bullets_cnt = drop_g.get("few_bullets", 0)

    dim_median_bullets = {}
    for dim in per_dimension:
        qs_d = [q for q in per_question if q["dimension"] == dim and q.get("selected")]
        nums = []
        for q in qs_d:
            cleaned = sanitize_for_display(q["selected"]["answer"] or "")
            nums.append(len([ln for ln in cleaned.splitlines() if ln.strip()]))
        if nums:
            nums.sort()
            dim_median_bullets[dim] = nums[len(nums)//2]
        else:
            dim_median_bullets[dim] = 0

    quality_gate = derive_quality_gates(
        total_q_count=total_q_count,
        selected_count=selected_count,
        degraded_count=degraded_count,
        consistency_ratio=mean_jac,
    )

    overall = {
        "overall_score": overall_score,
        "overall_confidence": overall_confidence,
        "mean_pairwise_jaccard": mean_jac,
        "mean_pairwise_contradiction": mean_ctr,
        "unknown_ratio": (unknown_q_count / max(1, total_q_count)),
        "drop_reasons_global": dict(drop_reasons_global),
        "provider_stats": provider_summary,
        "placeholder_ratio_global": placeholder_cnt / max(1, sum(drop_g.values())) if drop_g else 0.0,
        "few_bullets_ratio_global": few_bullets_cnt / max(1, sum(drop_g.values())) if drop_g else 0.0,
        "dim_median_bullets": dim_median_bullets,
        "quality_gates": quality_gate
    }

    return per_question, per_dimension, overall


def derive_quality_gates(total_q_count: int, selected_count: int, degraded_count: int, consistency_ratio: float) -> dict:
    coverage_ratio = selected_count / max(1, total_q_count)
    degraded_ratio = degraded_count / max(1, selected_count)
    parse_success_ratio = max(0.0, min(1.0, 1.0 - degraded_ratio))
    warnings = []
    if coverage_ratio < 0.85:
        warnings.append("selected_coverage_low")
    # 解析成功率仍写入 metrics，但不参与告警列表。
    if consistency_ratio < 0.08:
        warnings.append("consistency_low")
    return {
        "selected_count": selected_count,
        "total_questions": total_q_count,
        "selected_coverage_ratio": coverage_ratio,
        "degraded_selected_ratio": degraded_ratio,
        "parse_success_ratio": parse_success_ratio,
        "consistency_ratio": consistency_ratio,
        "pass": True,
        "fail_reasons": [],
        "reasons": [],
        "warnings": warnings,
    }

def build_report_md(pid: str, meta: dict, per_dim: dict, overall: dict, cfg: dict) -> str:
    lines = []
    lines.append(f"# 项目后处理报告 · {pid}")
    lines.append("")
    lines.append(f"- 生成时间：{now_str()}")
    if meta:
        m = {k: meta.get(k) for k in ("generated_at", "pid", "schema") if k in meta}
        if "args" in meta:
            m["args"] = meta["args"]
        lines.append(f"- 元信息：{json.dumps(m, ensure_ascii=False)}")
    lines.append("")
    lines.append("## 总览")
    sc = overall["overall_score"]
    cf = overall["overall_confidence"]
    lines.append(f"- 综合评分（0~1）：**{sc:.3f}**  {bar(sc, cfg['bar_symbols'])}")
    lines.append(f"- 综合信心度（0~1）：**{cf:.3f}**  {bar(cf, cfg['bar_symbols'])}")
    lines.append(f"- 全局一致性（平均题间重合度）：**{overall['mean_pairwise_jaccard']:.3f}**")
    lines.append(f"- 全局冲突度（平均）：**{overall['mean_pairwise_contradiction']:.3f}**")
    qg = overall.get("quality_gates") or {}
    if qg:
        hard = qg.get("fail_reasons") or qg.get("reasons") or []
        lines.append(
            f"- 质量闸门：**{'通过' if qg.get('pass') and not hard else '未通过'}**"
        )
        lines.append(f"- 选中覆盖率：{qg.get('selected_coverage_ratio', 0.0):.1%}")
        lines.append(f"- 一致性（题间重合度，consistency_ratio）：{float(qg.get('consistency_ratio', 0.0)):.3f}")
        lines.append(f"- 解析成功率（估计）：{qg.get('parse_success_ratio', 0.0):.1%}")
        if hard:
            lines.append(f"- 闸门失败原因：{', '.join(_gate_reason_zh(x) for x in hard)}")
        ws = qg.get("warnings") or []
        if ws:
            lines.append(f"- 质量告警（不否决批次）：{', '.join(_gate_reason_zh(w) for w in ws)}")

    if "drop_reasons_global" in overall and overall["drop_reasons_global"]:
        drg = overall["drop_reasons_global"]
        disp = ", ".join([f"{k}:{v}" for k, v in sorted(drg.items(), key=lambda x: (-x[1], x[0]))])
        lines.append(f"- 候选被丢弃原因（全局居前）：{disp}")
    lines.append(f"- 估计占位符噪声占比：{overall.get('placeholder_ratio_global', 0.0):.1%}")
    lines.append(f"- 估计分点过少占比：{overall.get('few_bullets_ratio_global', 0.0):.1%}")

    if "unknown" in per_dim and per_dim["unknown"]["n"] > 0:
        unk_n = per_dim["unknown"]["n"]
        unk_ratio = overall.get("unknown_ratio", 0.0)
        lines.append(f"- 警示：存在 **{unk_n}** 道题落在「未归类维度」（占比 {unk_ratio:.1%}），建议复核问题集与维度抽取。")
        if unk_ratio > cfg.get("unknown_warn_ratio", 0.10):
            lines.append(f"- **严重提示**：未归类维度占比超过 {int(cfg.get('unknown_warn_ratio', 0.10)*100)}% 的警戒阈值。")

    pvstats = overall.get("provider_stats", {})
    if pvstats:
        parts = []
        for k, v in pvstats.items():
            pk = _provider_label_zh(k)
            parts.append(f"{pk}：选中{v['selected_count']}，内部加权均值={v['avg_alpha']:.3f}，综合分均值={v['avg_final']:.3f}")
        lines.append(f"- 推理来源统计：{'； '.join(parts)}")

    lines.append("")
    lines.append("## 维度分解")

    dims_in_report = list(DIM_ORDER) + [d for d in per_dim.keys() if d not in DIM_ORDER]

    for dim in dims_in_report:
        if dim not in per_dim:
            continue
        info = per_dim[dim]
        avg = info["avg"]
        dim_zh = DIM_LABELS_ZH.get(dim, dim)
        lines.append(f"### {dim_zh}  · 评分 {avg:.3f}  {bar(avg, cfg['bar_symbols'])}")
        if info.get("auth_hints"):
            lines.append(f"- 参考方向（非事实）：{'; '.join(info['auth_hints'])}")
        lines.append(f"- 对齐均值/漂移均值：**{info.get('avg_alignment', 0.0):.2f} / {info.get('avg_drift', 0.0):.2f}**")
        if "explain" in info:
            ex = info["explain"] or {}
            if ex.get("top_contributors"):
                lines.append(f"- 主要得分贡献因子：{', '.join(ex['top_contributors'])}")
            if ex.get("top_penalties"):
                lines.append(f"- 主要扣分因子：{', '.join(ex['top_penalties'])}")
        if info.get("top_evidence_phrases"):
            lines.append("- **主要证据短语（权威命中优先）**：")
            for s in info["top_evidence_phrases"]:
                lines.append(f"  - {s}")
        gi_list = info.get("general_insights") or []
        if gi_list:
            lines.append("- **行业通识要点（通用建议，不代表本项目已达成）**：")
            for s in gi_list[:5]:
                lines.append(f"  - {s}")
        if info["strengths"]:
            lines.append("- **优势**：")
            for s in info["strengths"]:
                lines.append(f"  - {s}")
        if info["risks"]:
            lines.append("- **风险**：")
            for r in info["risks"]:
                lines.append(f"  - {r}")
        if info["snippets"]:
            lines.append("- **代表性片段**：")
            for sn in info["snippets"]:
                lines.append(f"  - {sn}")
        redlined_samples = info.get("redlined_samples") or []
        if redlined_samples:
            lines.append("- **已转为检索线索的原句（抽样）**：")
            for s in redlined_samples[:6]:
                lines.append(f"  - {s}")
        drop_r = info.get("dropped_reasons") or {}
        if drop_r:
            lines.append("- **被丢弃候选统计**（原因：次数）：")
            disp = ", ".join([f"{k}:{v}" for k, v in sorted(drop_r.items(), key=lambda x: (-x[1], x[0]))])
            lines.append(f"  - {disp}")
        lines.append(f"- 中位条目数（估计）：{overall.get('dim_median_bullets', {}).get(dim, 0)}")
        lines.append("")
    lines.append("> 注：本报告仅基于候选答案的结构化指标（长度/要点/证据提示/权威度/覆盖度/结构/一致性/置信度/维度对齐等），"
                 "以及 LLM 给出的“行业通识建议”（general_insights）。通识建议仅为行业基准参考，并不代表本项目已经实现或满足相关要求。"
                 "定稿前建议人工抽检与项目原文/事实证据对齐。")
    return "\n".join(lines)

# ============================ 主流程 ============================

def main():
    ap = argparse.ArgumentParser(description="Post-processing for structured candidates (no LLM calls).")
    ap.add_argument("--pid", type=str, default="", help="提案ID，不填则自动选择 refined_answers 下最新目录")
    ap.add_argument("--input", type=str, default="", help="可选：all_refined_items.json 的绝对/相对路径")

    # === A/B 开关 ===
    ap.add_argument("--ablate_no_question_claims", action="store_true",
                    help="关闭对齐语料中的 question/claims 注入（用于消融实验）")
    ap.add_argument("--ablate_no_dyn_relax", action="store_true",
                    help="关闭基于权威/覆盖触发的 min_alignment 放宽（已废弃，仅保留兼容）")

    args = ap.parse_args()

    cfg = load_config()

    # 把 A/B 开关塞入 cfg，便于下游函数访问
    cfg["_ablate_no_question_claims"] = bool(args.ablate_no_question_claims)
    cfg["_ablate_no_dyn_relax"] = bool(args.ablate_no_dyn_relax)

    # 读取问题集（用于权威/题材 hints 对齐）
    if QS_CONF_PATH.exists():
        try:
            qs_cfg = read_json(QS_CONF_PATH)
        except Exception:
            qs_cfg = {}
    else:
        qs_cfg = {}

    if args.input:
        refined_path = Path(args.input)
        if not refined_path.exists():
            raise FileNotFoundError(f"未找到输入文件：{refined_path}")
        pid = refined_path.parent.name
    else:
        pid = args.pid.strip() or get_context_proposal_id() or detect_latest_pid()
        if not pid:
            raise RuntimeError("未检测到 refined_answers 下的最新项目目录，也未提供 --pid / --input")
        refined_path = REFINED_ROOT / pid / "all_refined_items.json"
        if not refined_path.exists():
            raise FileNotFoundError(f"未找到文件：{refined_path}")

    data = read_json(refined_path)
    meta = (data.get("meta") or {})
    schema = meta.get("schema", "")

    # 新版支持 llm_answering.v2 以及 refined_items.v2.proposal_aware_with_general_insights
    allowed_schemas = {"", "llm_answering.v2", "refined_items.v2.proposal_aware_with_general_insights"}
    if schema and schema not in allowed_schemas:
        print(f"⚠️ 警告：输入 schema={schema}（预期 {allowed_schemas} 之一），将继续处理。")

    # items / questions 兼容读取（新版 llm_answering 使用 items）
    items = data.get("items") or data.get("questions") or []

    bad = []
    for i, it in enumerate(items):
        if "dimension" not in it or "q_index" not in it or "question" not in it or "candidates" not in it:
            bad.append(i)
    if bad:
        raise ValueError(f"输入 items 中存在缺失必要字段的条目：索引 {bad[:10]} ...，请检查 llm_answering 输出结构。")

    per_question, per_dimension, overall = aggregate_dimensions(items, cfg, qs_cfg)

    out_dir = REFINED_ROOT / pid / "postproc"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "meta": {
            "pid": pid,
            "generated_at": now_str(),
            "source_file": str(refined_path),
            "schema": schema or "",
            "args": meta.get("args", {})
        },
        "config_used": cfg,
        "overall": overall,
        "dimensions": per_dimension,
        "questions": per_question
    }
    write_json(out_dir / "metrics.json", metrics)

    # 逐题选中结果（便于人工抽检）
    write_json(out_dir / "selected_by_question.json", per_question)

    # final_payload（下游报告的统一入口）
    final_payload = {
        "meta": {"pid": pid, "generated_at": now_str()},
        "dimensions": {}
    }
    for dim in DIM_ORDER:
        qs = [q for q in per_question if q["dimension"] == dim and q.get("selected")]
        final_payload["dimensions"][dim] = {
            "score": round(float(per_dimension.get(dim, {}).get("avg", 0.0)) * 100, 1),
            "qas": [
                {
                    "q": q["question"],
                    "answer": q["selected"]["answer"],
                    "claims": q["selected"]["claims"],
                    "evidence_hints": q["selected"]["evidence_hints"],
                    "topic_tags": q["selected"].get("topic_tags", []),
                    "provider": q["selected"].get("provider"),
                    "model": q["selected"].get("model"),
                    "alignment": q["selected"].get("alignment_ratio", 0.0),
                    "dimension_drift": q["selected"].get("dimension_drift", 0.0),
                    "confidence": q["selected"]["confidence"],
                    "caveats": "",
                    # ★ 新增：逐问通识经验层
                    "general_insights": q["selected"].get("general_insights", [])
                } for q in qs
            ],
            "rationales": (per_dimension.get(dim, {}).get("strengths") or [])[:3],
            # ★ 新增：维度级聚合通识经验层，供 ai_expert_opinion 作为“行业经验层”使用
            "general_insights": per_dimension.get(dim, {}).get("general_insights", [])
        }
    write_json(out_dir / "final_payload.json", final_payload)

    report_md = build_report_md(pid, meta, per_dimension, overall, cfg)
    report_md, _report_truncated = clip_markdown_total_chars(
        report_md, MAX_AGGREGATED_REPORT_CHARS
    )
    (out_dir / "report.md").write_text(report_md, encoding="utf-8")

    # 丢弃原因调试（仅写逐题的精简信息）
    write_json(
        out_dir / "drops_debug.json",
        [
            {
                "dimension": q.get("dimension"),
                "q_index": q.get("q_index"),
                "question": q.get("question"),
                "drop_stats": q.get("drop_stats", {})
            }
            for q in metrics["questions"]
            if q.get("drop_stats")  # 只有存在统计才写入
        ]
    )

    print(f"✅ metrics.json              -> {out_dir/'metrics.json'}")
    print(f"✅ selected_by_question.json -> {out_dir/'selected_by_question.json'}")
    print(f"✅ final_payload.json        -> {out_dir/'final_payload.json'}")
    print(f"✅ report.md                 -> {out_dir/'report.md'}")

    total_q = len(metrics["questions"])
    dim_brief = ", ".join(f"{d}:{info['n']}" for d, info in metrics["dimensions"].items())
    ov = metrics["overall"]
    print(f"ℹ️  题目数：{total_q} | 维度题量：{dim_brief}")
    print(f"ℹ️  综合评分={ov['overall_score']:.3f}  信心度={ov['overall_confidence']:.3f}  "
          f"Jaccard={ov['mean_pairwise_jaccard']:.3f}  冲突度={ov['mean_pairwise_contradiction']:.3f}")
    if "unknown" in per_dimension and per_dimension["unknown"]["n"] > 0:
        unk_ratio = ov.get("unknown_ratio", 0.0)
        print(f"⚠️ 提示：存在 {per_dimension['unknown']['n']} 道题被归入 unknown 维度（占比 {unk_ratio:.1%}），请检查问题集/维度抽取。")

    drg = ov.get("drop_reasons_global", {})
    if drg:
        disp = ", ".join([f"{k}:{v}" for k, v in sorted(drg.items(), key=lambda x: (-x[1], x[0]))])
        print(f"ℹ️ 丢弃原因（全局Top）：{disp}")

    pvstats = ov.get("provider_stats", {})
    if pvstats:
        parts = []
        for k, v in pvstats.items():
            parts.append(f"{k}: 选中{v['selected_count']} | α均值={v['avg_alpha']:.3f} | final均值={v['avg_final']:.3f}")
        print(f"ℹ️ Provider 统计：{'； '.join(parts)}")

    print("🎯 后处理完成。")

if __name__ == "__main__":
    main()
