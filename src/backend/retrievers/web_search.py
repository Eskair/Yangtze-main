# -*- coding: utf-8 -*-
"""
模块：Web Search（v2025.12 ProClean · TrustLayer + MustHave + Diagnostics + ReRank v1）
保持签名：simple_search(query, max_results=8, dimension="general", hints=None, source="LLM")
输出不变：src/data/evidence/{proposal_id}/{dimension}_combined.json

新增要点（对相关性友好，且默认安全）：
- Query 智能拼接：若用户 query 已含 site:/时间窗/负向词，则不重复注入；team 维度更少误杀
- 轻量 ReRank（BM25-lite + 关键短语/实体加权 + 信息密度 + 来源置信度）
- 文本近重复抑制（n-gram Jaccard，默认阈值 0.92）
- 结果有序写入 combined（更相关的排前面，但仍保存全部）
- Cache 命中不再覆盖 combined（之前已修）
- 多线程 host 配额加锁（之前已修）
"""
import os, re, time, json, hashlib, warnings, math
import requests, trafilatura
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict
from pathlib import Path
import threading

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# ===== 外部依赖（可选）=====
try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None
try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None

# ===== LLM （仅用于小摘要，非必须）=====
try:
    from backend.utils.model_selector import get_llm_client
    _llm_info = get_llm_client()
    LLM_CLIENT = _llm_info["client"]
    LLM_MODEL  = _llm_info["model_name"]
    LLM_PROVIDER = _llm_info["provider"]
    print(f"💬 WebSearch 使用 {LLM_PROVIDER.upper()} 模型：{LLM_MODEL}")
except Exception:
    LLM_CLIENT, LLM_MODEL = None, None

# ===== 环境变量 =====
TAVILY_KEY = os.getenv("TAVILY_API_KEY")
GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CX  = os.getenv("GOOGLE_CX")
client_tavily = TavilyClient(api_key=TAVILY_KEY) if (TavilyClient and TAVILY_KEY) else None

# 可调控开关（全部向后兼容）
MUST_HAVE_STRICT = os.getenv("WEB_MUST_HAVE_STRICT", "1") != "0"
RERANK_ENABLE    = os.getenv("WEB_RERANK_ENABLE", "1") != "0"   # 新增：开启轻量重排
DEDUP_ENABLE     = os.getenv("WEB_DEDUP_ENABLE", "1") != "0"     # 新增：开启文本近重复抑制
DEDUP_JACCARD    = float(os.getenv("WEB_DEDUP_JACCARD", "0.92")) # 新增：n-gram Jaccard 阈值

# ===== 全局参数 =====
MIN_LEN_BASE = 140
MAX_LEN = 2600
MAX_WORKERS = 8
RETRY = 2
TIMEOUT = 12
HOST_QUOTA = 3
EARLY_ACADEMIC_MIN = {"strategy": 4, "objectives": 4, "feasibility": 4, "innovation": 3, "team": 3}

LANG_ALLOW = {"en", "zh", "zh-cn", "zh-tw", "und"}
BLOCK_HOSTS = (
    "facebook.", "twitter.", "x.com", "reddit.", "zhihu.", "bilibili.",
    "medium.com", "pinterest.", "wechat.", "weibo.", "quora.", "csdn.",
    "scribd.com", "moomoo.com", "sol-war.ru", "xmind.com", "islandenergy.je",
    "glassdoor.", "indeed.", "join.com", "job", "careers", "recruit", "press"  # （已移除 "news"）
)

ACADEMIC_DOMAINS = [
    "pubmed.ncbi.nlm.nih.gov","pmc.ncbi.nlm.nih.gov","nature.com","sciencedirect.com",
    "who.int","iso.org","ieee.org","oecd.org","nist.gov",
    "thelancet.com","bmj.com","cell.com","springer.com","biorxiv.org","medrxiv.org","arxiv.org","nejm.org",
    "w3.org"
]
INSTITUTIONAL_HINTS = (".edu", ".ac.", "university", "hospital")

MUST_HAVE_BY_DIM = {
    "innovation": ["architecture", "benchmark", "baseline", "novelty", "patent", "prototype"],
    "strategy":   ["compliance", "standard", "guidance", "approval", "go-to-market", "partnership"],
    "objectives": ["target metric", "milestone", "endpoint", "success criteria", "timeline", "kpi"],
    "feasibility":["reliability", "stability", "scale-up", "resource", "budget", "operations"],
    "team":       ["project lead", "core team", "experience", "organization", "department", "track record"]
}
MIN_LEN_BY_DIM = {"team": 80}

NEGATIVE_BY_DIM = {
    "strategy":   "-stock -finance -press -news -forum -blog -marketing",
    "feasibility":"-press -news -forum -blog -marketing -stock -finance",
    "objectives": "-news -press -blog -forum",
    "innovation": "-press -news -blog -forum",
    "team":       "-jobs -career -recruit -hiring -admissions"  # 放宽 team（不再 -press/-news）
}

_HTTP = requests.Session()
_HTTP.headers.update({"User-Agent": "Mozilla/5.0 (compatible; RAG6View/2025; +https://rag6view.local/agent)"})


# ===== 基础打分 =====
def _clamp(x, lo=0.3, hi=1.0):
    try: return max(lo, min(hi, float(x)))
    except Exception: return lo

def source_confidence(domain: str) -> float:
    if not domain: return 0.55
    d = domain.lower()
    high = ["pubmed","pmc","who","nature","sciencedirect","springer","arxiv","iso","ieee","nist","oecd"]
    medium = ["arxiv","biorxiv","medrxiv","researchsquare"]
    low = ["news","press","blog","medium.com"]
    if any(x in d for x in high): return 0.98
    if any(x in d for x in medium): return 0.80
    if any(x in d for x in low): return 0.50
    return 0.62

def info_density_score(text: str) -> float:
    n_year = len(re.findall(r"\b(19|20)\d{2}\b", text))
    n_unit = len(re.findall(r"\b(ms|s|gb|tb|ai|ml|benchmark|latency|throughput|auc|f1|iso|iec|soc|nist)\b", text, re.I))
    return _clamp((n_year + n_unit) / 24.0, 0.0, 1.0)


# ===== URL/域名工具 =====
UTM_KEYS = {"utm_source","utm_medium","utm_campaign","utm_term","utm_content","utm_id","gclid","fbclid","msclkid"}
def normalize_url(u: str) -> str:
    try:
        p = urlparse(u)
        scheme = p.scheme or "https"
        q = [(k,v) for k,v in parse_qsl(p.query, keep_blank_values=True) if k.lower() not in UTM_KEYS]
        new_q = urlencode(q, doseq=True)
        path = p.path or "/"
        if len(path) > 1 and path.endswith("/"): path = path[:-1]
        return urlunparse((scheme, p.netloc.lower(), path, "", new_q, ""))
    except Exception:
        return u

def is_whitelisted(host: str) -> bool:
    h = (host or "").lower()
    return any(h.endswith(d) for d in ACADEMIC_DOMAINS) or any(x in h for x in INSTITUTIONAL_HINTS)

def _is_homepage(url: str) -> bool:
    return bool(re.match(r"^https?://[^/]+/?$", url or ""))

def rough_lang(text: str) -> str:
    if not text: return "en"
    chinese = len([ch for ch in text[:1200] if "\u4e00" <= ch <= "\u9fff"])
    ratio = chinese / max(1, len(text[:1200]))
    return "zh" if ratio > 0.02 else "en"


# ===== 标题 & HEAD =====
def fetch_title(html: str) -> str:
    try:
        soup = BeautifulSoup(html, "html.parser")
        t = soup.title.get_text(" ", strip=True) if soup.title else ""
        return re.sub(r"\s+", " ", t)[:200]
    except Exception:
        return ""

def head_content_type(url: str) -> str:
    try:
        r = _HTTP.head(url, timeout=8, allow_redirects=True)
        return r.headers.get("Content-Type", "").lower()
    except Exception:
        return ""


# ===== 正文抽取 =====
_BAD_HINTS = ["cookie", "privacy", "subscribe", "登录", "订阅", "广告", "forbidden", "please enable javascript"]

def _extract_pdf_with_trafilatura(url: str) -> str:
    try:
        html2 = trafilatura.fetch_url(url)
        tx = trafilatura.extract(html2) if html2 else ""
        return (tx or "").strip()
    except Exception:
        return ""

def fetch_clean_text(url: str, dimension: str, title_hint: str = "", host: str = ""):
    # team 放宽高校/医院 PDF
    ctype = head_content_type(url)
    is_pdf = "pdf" in ctype or url.lower().endswith(".pdf")
    allow_pdf = False
    if is_pdf:
        if dimension in ("strategy","objectives","feasibility") and is_whitelisted(host):
            allow_pdf = True
        if dimension == "team" and (".edu" in host or "university" in host or "hospital" in host):
            allow_pdf = True
        if not allow_pdf:
            return None

    if is_pdf:
        text_pdf = _extract_pdf_with_trafilatura(url)
        if not text_pdf:
            return None
        text_pdf = re.sub(r"\s+", " ", text_pdf)
        return text_pdf[:MAX_LEN]

    html = None
    for r in range(RETRY + 1):
        try:
            resp = _HTTP.get(url, timeout=TIMEOUT)
            if resp.status_code == 200:
                html = resp.text
                break
        except Exception:
            time.sleep(0.35 * (2 ** r))
    if not html:
        return None

    text = ""
    try:
        soup = BeautifulSoup(html, "html.parser")
        main = soup.find(["article","main"]) or soup
        text = " ".join(p.get_text(" ", strip=True) for p in main.find_all(["p","li"]))
    except Exception:
        text = ""

    if len(text) < 180:
        try:
            html2 = trafilatura.fetch_url(url)
            text2 = trafilatura.extract(html2) if html2 else ""
            if text2 and len(text2) > len(text):
                text = text2
        except Exception:
            pass

    if not text:
        return None

    text = re.sub(r"\s+", " ", text).strip()
    low = text.lower()
    if any(b in low for b in _BAD_HINTS):
        return None

    min_len = MIN_LEN_BY_DIM.get(dimension, MIN_LEN_BASE)
    if dimension == "team" and any(k in (title_hint or "").lower() for k in ["faculty","lab","principal investigator","pi","department"]):
        min_len = min(min_len, 60)

    lang = rough_lang(text)
    if lang not in LANG_ALLOW: return None
    if len(text) < min_len:   return None
    return text[:MAX_LEN]


# ===== 基础搜索器 =====
def google_search(q, n=8):
    if not GOOGLE_KEY or not GOOGLE_CX: return []
    try:
        r = _HTTP.get("https://www.googleapis.com/customsearch/v1",
                      params={"key": GOOGLE_KEY, "cx": GOOGLE_CX, "q": q, "num": n}, timeout=TIMEOUT)
        if r.status_code == 200:
            items = r.json().get("items", []) or []
            return [i.get("link") for i in items if i.get("link")]
    except Exception as e:
        print(f"⚠️ Google 搜索失败: {e}")
    return []

def tavily_search(q, n=8):
    if not client_tavily: return []
    try:
        r = client_tavily.search(query=q, max_results=n)
        return [x.get("url") for x in (r.get("results") or []) if x.get("url")]
    except Exception as e:
        if "limit" in str(e).lower():
            print("⚠️ Tavily 限额已达，降级 DuckDuckGo")
            return duckduckgo_search_fn(q, n)
        print(f"⚠️ Tavily 搜索失败: {e}")
    return []

def duckduckgo_search_fn(q, n=8):
    if not DDGS: return []
    try:
        with DDGS() as d:
            return [r.get("href") for r in d.text(q, max_results=n) if r.get("href")]
    except Exception as e:
        print(f"⚠️ DuckDuckGo 搜索失败: {e}")
    return []


# ===== 学术回补关键词 =====
ACADEMIC_BACKFILL = {
    "strategy":   "guidance OR approval OR NDA OR BLA OR IND OR regulatory pathway",
    "objectives": '"primary endpoint" OR randomized OR Phase',
    "feasibility":'reliability OR stability OR operational risk OR scale-up OR budget',
    "innovation": '"lipid nanoparticle" OR siRNA OR mRNA OR inhaled OR aerosol',
    "team":       'faculty OR "Principal Investigator" OR lab OR ORCID'
}


# ===== 原子写 =====
def atomic_write(path: Path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


# ===== 轻量相关性工具 =====
_WORD_SPLIT = re.compile(r"[^\w\-\./]+")

def _tokenize(s: str):
    return [t.lower() for t in _WORD_SPLIT.split(s or "") if t]

def _bm25lite_score(text_tokens, query_tokens, k1=1.2, b=0.75):
    if not text_tokens or not query_tokens: return 0.0
    tf = Counter(text_tokens)
    L = len(text_tokens)
    avgL = 400.0  # 一个经验值：提取后正文常在 400~1500 tokens
    score = 0.0
    # 这里没有全语料 idf，只给查询词出现一个固定 idf_boost（出现=1，不出现=0）
    for q in set(query_tokens):
        f = tf.get(q, 0)
        if f == 0:
            continue
        idf = 1.6  # 经验常数，保证出现即有显著贡献
        denom = f + k1 * (1 - b + b * (L / avgL))
        score += idf * (f * (k1 + 1)) / (denom if denom > 0 else 1.0)
    # 归一
    return _clamp(score / 12.0, 0.0, 1.0)

def _phrase_boost(text: str, phrases: list):
    lo = text.lower()
    hits = 0
    for p in phrases or []:
        p = str(p or "").lower().strip()
        if not p: continue
        if p in lo:
            hits += 1
    # 每命中一个短语 +0.12，上限 1.0
    return _clamp(hits * 0.12, 0.0, 1.0)

def _relevance_score(item, query: str, hints, dimension: str):
    text = item.get("text","") or ""
    domain = item.get("domain","") or ""
    toks_text = _tokenize(text)
    toks_q    = _tokenize(query)

    # must-have、hints 合起来算短语 boost
    musts = MUST_HAVE_BY_DIM.get(dimension.lower(), [])
    phrases = (hints or [])[:6] + musts
    bm25p  = _bm25lite_score(toks_text, toks_q)
    pboost = _phrase_boost(text, phrases)
    dens   = info_density_score(text)
    src    = source_confidence(domain)
    wl     = 0.07 if is_whitelisted(domain) else 0.0

    # 线性融合：可通过环境变量微调
    w_src  = float(os.getenv("WEB_RR_W_SRC", "0.45"))
    w_den  = float(os.getenv("WEB_RR_W_DEN", "0.20"))
    w_bm25 = float(os.getenv("WEB_RR_W_BM25","0.25"))
    w_phra = float(os.getenv("WEB_RR_W_PHRA","0.10"))
    score  = (w_src*src + w_den*dens + w_bm25*bm25p + w_phra*pboost + wl)
    return _clamp(score, 0.0, 1.0)

def _shingles(text: str, n=7):
    toks = _tokenize(text)
    return set(tuple(toks[i:i+n]) for i in range(0, max(0, len(toks)-n+1)))

def _jaccard(a: set, b: set):
    if not a or not b: return 0.0
    inter = len(a & b); union = len(a | b)
    return (inter / union) if union else 0.0


# ===== 主函数（保持签名）=====
def simple_search(query, max_results=8, dimension="general", hints=None, source="LLM"):
    raw_q = (query or "").strip()
    query = re.sub(r"\s+", " ", raw_q)

    # ---- Query 智能拼接（避免重复注入）----
    qlower = query.lower()
    has_site   = " site:" in qlower or " site(" in qlower
    has_year   = re.search(r"\b(19|20)\d{2}\.\.(19|20)\d{2}\b", qlower) is not None
    has_neg    = any(tok in qlower for tok in [" -news", " -press", " -blog", " -forum", " -finance", " -jobs"])
    neg = NEGATIVE_BY_DIM.get(dimension.lower(), "")
    if has_neg:  # 用户已带负向，则不重复注入
        neg = ""
    suffix_map = {
        "team": '("project lead" OR "core team" OR "track record")',
        "objectives": "(target metric OR success criteria OR milestone plan)",
        "strategy": "(compliance pathway OR go-to-market OR commercialization plan)",
        "innovation": "(AI model OR benchmark OR technical novelty)",
        "feasibility": "(safety OR reliability OR scalability OR budget)"
    }
    suffix = suffix_map.get(dimension.lower(), "")

    site_hint = ""
    if (dimension.lower() in ("strategy","feasibility","objectives")) and not has_site:
        site_hint = " site:(iso.org OR ieee.org OR oecd.org OR who.int OR gov)"

    # 若 query 已含时间窗则尊重
    full_query = f"{query} {suffix} {neg} {site_hint}".strip()
    full_query = re.sub(r"\s+", " ", full_query)
    print(f"\n🔍 [{source}] 维度 {dimension} 搜索: {full_query}")

    proposal_id = os.getenv("CURRENT_PROPOSAL_ID", "default")
    results_dir = Path(f"src/data/evidence/{proposal_id}")
    results_dir.mkdir(parents=True, exist_ok=True)

    combined_path = results_dir / f"{dimension}_combined.json"
    cache_file = results_dir / f"{dimension}_cache.json"
    debug_path = results_dir / f"{dimension}_debug_stats.json"
    diag_path = results_dir / f"{dimension}_diag.json"

    diag_rows = []
    dbg = defaultdict(int)

    # === 缓存命中（不覆盖 combined；空缓存不落盘）===
    cache = json.loads(cache_file.read_text(encoding="utf-8")) if cache_file.exists() else {}
    cache_key = f"{dimension}::{hashlib.md5(full_query.encode('utf-8')).hexdigest()[:10]}"
    if cache_key in cache:
        print("🧠 命中缓存")
        cached_items = cache[cache_key] or []
        existing = []
        if combined_path.exists():
            try:
                existing = json.loads(combined_path.read_text(encoding="utf-8"))
            except Exception:
                existing = []
        seen = set(x.get("url") for x in existing if isinstance(x, dict))
        merged = list(existing)
        for item in cached_items:
            u = (item or {}).get("url")
            if u and u not in seen:
                merged.append(item); seen.add(u)
        if cached_items:
            # 排序：若启用 rerank，则对 merged 重排（只改变顺序，不过滤）
            if RERANK_ENABLE:
                merged = _sort_by_relevance(merged, full_query, hints, dimension)
            atomic_write(combined_path, merged)

        texts = [i.get("text","") for i in cached_items if isinstance(i, dict)]
        urls  = [i.get("url","")  for i in cached_items if isinstance(i, dict)]
        return texts, urls

    # === 三层检索 ===
    urls = []
    for idx, fn in enumerate((google_search, tavily_search, duckduckgo_search_fn), start=1):
        try:
            got = fn(full_query, max_results)
            dbg[f"api_{fn.__name__}_urls"] += len(got)
            urls.extend(got)
            if len(urls) >= max_results: break
        except Exception:
            pass
        time.sleep(0.25 * idx)
    dbg["urls_found_by_api"] = len(urls)

    # 归一化/去重/屏蔽
    norm_urls, seen = [], set()
    for u in urls:
        if not u: continue
        nu = normalize_url(u)
        host = (urlparse(nu).hostname or "").lower()
        if (not host) or any(b in host for b in BLOCK_HOSTS):
            dbg["filtered_block_host"] += 1; continue
        if nu not in seen:
            norm_urls.append(nu); seen.add(nu)
    dbg["urls_after_normalize"] = len(norm_urls)

    # 过少则二次尝试：中文强化
    if len(norm_urls) < max_results // 2:
        q2 = f"{query} 技术方案 AI 工程实施 合规标准 2019..2025"
        more = duckduckgo_search_fn(q2, max_results)
        for u in more:
            nu = normalize_url(u)
            host = (urlparse(nu).hostname or "").lower()
            if nu not in seen and host and not any(b in host for b in BLOCK_HOSTS):
                norm_urls.append(nu); seen.add(nu)
        dbg["urls_after_backoff_add"] = len(norm_urls)

    if not norm_urls:
        print("❌ 无搜索结果")
        atomic_write(combined_path, []); atomic_write(debug_path, dict(dbg))
        return [], []

    # === 抓取正文 ===
    extracted = []
    host_seen = defaultdict(int)
    _host_seen_lock = threading.Lock()

    def fetch_one(u):
        host = (urlparse(u).hostname or "").lower()
        whitelisted = is_whitelisted(host)
        if _is_homepage(u) and not whitelisted:
            return None, ("filtered_homepage", u)
        with _host_seen_lock:
            if host_seen[host] >= HOST_QUOTA and not whitelisted:
                return None, ("filtered_host_quota", u)

        title = ""
        try:
            r = _HTTP.get(u, timeout=TIMEOUT)
            if r.status_code != 200:
                return None, ("fetch_non200", u)
            title = fetch_title(r.text)
        except Exception:
            pass

        text = fetch_clean_text(u, dimension=dimension, title_hint=title, host=host)
        if not text:
            return None, ("filtered_by_rules", u)

        mhs_base = MUST_HAVE_BY_DIM.get(dimension.lower(), [])
        hints_low = (hints or [])[:3] if isinstance(hints, list) else []
        mhs = [tok for tok in (mhs_base + hints_low) if isinstance(tok, str) and tok.strip()]

        if MUST_HAVE_STRICT and mhs and not whitelisted:
            low = text.lower()
            if not any(tok.lower() in low for tok in mhs):
                # 宽松兜底：若文本信息密度很高（≥0.6），且标题/域名具备研究/学院线索，则放行
                if info_density_score(text) >= 0.60 and any(key in (title.lower()+" "+host) for key in ["lab","faculty","research","university","hospital"]):
                    pass
                else:
                    return None, ("filtered_must_have_miss", u)

        with _host_seen_lock:
            host_seen[host] += 1

        conf_base = source_confidence(host)
        conf = _clamp(conf_base + 0.25 * info_density_score(text) + (0.05 if whitelisted else 0.0))
        text = text.encode("utf-8","ignore").decode("utf-8","ignore")
        return {"url": u, "text": text, "domain": host, "confidence": conf, "len": len(text)}, None

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(fetch_one, u) for u in norm_urls[:max_results * 2]]
        for f in as_completed(futures):
            r, reason = f.result()
            if r:
                extracted.append(r); dbg["fetch_ok"] += 1
            else:
                if reason:
                    dbg[reason[0]] += 1
                    if len(diag_rows) < 400:
                        diag_rows.append({"url": reason[1], "reason": reason[0]})

    # 学术回补
    academic_hits = [x for x in extracted if any(ad in x["domain"] for ad in ACADEMIC_DOMAINS)]
    if len(academic_hits) < EARLY_ACADEMIC_MIN.get(dimension, 2):
        print("🧠 学术来源过少，触发 Academic 回补...")
        bf = ACADEMIC_BACKFILL.get(dimension, "")
        scholar_query = f'{query} ({bf}) site:(' + " OR ".join(ACADEMIC_DOMAINS[:10]) + ") 2019..2025"
        more = google_search(scholar_query, 10)
        for u in more:
            nu = normalize_url(u)
            host = (urlparse(nu).hostname or "").lower()
            if nu in {x["url"] for x in extracted}: continue
            with _host_seen_lock:
                over = host_seen[host] >= HOST_QUOTA and not is_whitelisted(host)
            if over:
                continue
            t = fetch_clean_text(nu, dimension=dimension, host=host)
            if t and len(t) >= MIN_LEN_BASE:
                extracted.append({"url": nu, "text": t, "domain": host, "confidence": 0.99, "len": len(t)})
                with _host_seen_lock:
                    host_seen[host] += 1
                if len([x for x in extracted if any(ad in x["domain"] for ad in ACADEMIC_DOMAINS)]) >= EARLY_ACADEMIC_MIN.get(dimension, 2):
                    break

    if not extracted:
        print("❌ 未抓取到正文。")
        atomic_write(combined_path, []); atomic_write(debug_path, dict(dbg))
        if diag_rows: atomic_write(diag_path, diag_rows)
        return [], []

    # === （新增）近重复抑制 + 相关性排序 ===
    extracted_proc = extracted
    if DEDUP_ENABLE:
        extracted_proc = []
        shingle_bank = []
        for item in extracted:
            s = _shingles(item.get("text",""), n=7)
            if not s:
                extracted_proc.append(item); shingle_bank.append(s); continue
            similar = False
            for prev in shingle_bank:
                if not prev: continue
                if _jaccard(s, prev) >= DEDUP_JACCARD:
                    similar = True; break
            if not similar:
                extracted_proc.append(item); shingle_bank.append(s)

    if RERANK_ENABLE:
        extracted_proc = _sort_by_relevance(extracted_proc, full_query, hints, dimension)

    # === 合并去重（按 URL）===
    existing = []
    if combined_path.exists():
        try: existing = json.loads(combined_path.read_text(encoding="utf-8"))
        except Exception: existing = []
    union = existing + extracted_proc
    seen_urls, merged = set(), []
    for item in union:
        u = item.get("url")
        if u and u not in seen_urls:
            merged.append(item); seen_urls.add(u)

    # 重排 merged 顺序（不丢数据，只调整排序）
    if RERANK_ENABLE:
        merged = _sort_by_relevance(merged, full_query, hints, dimension)

    # === 写 combined & cache ===
    atomic_write(combined_path, merged)
    cache[cache_key] = merged
    atomic_write(cache_file, cache)

    # === 统计 ===
    academic_hits = [x for x in merged if any(ad in x["domain"] for ad in ACADEMIC_DOMAINS)]
    avg_conf = round(sum(x["confidence"] for x in merged) / len(merged), 2) if merged else 0
    ratio = round((len(academic_hits) / len(merged)), 2) if merged else 0.0
    print(f"✅ 抓取完成：{len(merged)} 条 | 平均置信度 {avg_conf} | 学术比例 {ratio:.0%}")
    print(f"🔝 Top 域名: {Counter([x['domain'] for x in merged]).most_common(5)}")

    dbg["kept_total"] = len(merged); dbg["academic_hits"] = len(academic_hits)
    atomic_write(debug_path, dict(dbg))
    if diag_rows: atomic_write(diag_path, diag_rows)

    # === 维度摘要（保持不变）===
    try:
        if merged and LLM_CLIENT:
            joined = "\n\n".join(x["text"][:600] for x in merged[:3])
            refs = "\n".join([f"[{i+1}] {x['url']}" for i, x in enumerate(merged[:3])])
            prompt = (
                f"请基于以下网页内容，总结维度 {dimension} 的关键信息（120~160字，学术语气，避免夸大/臆测；涉及年份/批准必须引用 [x]）。\n\n"
                f"{joined}\n\n参考：\n{refs}"
            )
            res = LLM_CLIENT.chat_completions.create(  # 兼容某些 SDK 写法差异
                model=LLM_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.3,
            ) if hasattr(LLM_CLIENT, "chat_completions") else LLM_CLIENT.chat.completions.create(
                model=LLM_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.3,
            )
            summary = (res.choices[0].message.content or "").strip()
            atomic_write(results_dir / f"{dimension}_summary.json", {
                "dimension": dimension, "summary": summary,
                "urls": [x['url'] for x in merged], "avg_conf": avg_conf, "academic_ratio": ratio
            })
    except Exception as e:
        print(f"⚠️ 摘要生成失败: {e}")

    texts = [x["text"] for x in merged]
    urls_extracted = [x["url"] for x in merged]
    return texts, urls_extracted


# ===== 排序主函数（集中定义，便于切换算法）=====
def _sort_by_relevance(items, query, hints, dimension):
    scored = []
    for it in items:
        s = _relevance_score(it, query, hints, dimension)
        # 提升与维度 Must-Have 命中数（用于 tie-break）
        mh = 0
        low = (it.get("text","") or "").lower()
        for tok in MUST_HAVE_BY_DIM.get(dimension.lower(), []):
            if tok.lower() in low:
                mh += 1
        scored.append((s, mh, it))
    # 先按得分，再按 must-have 命中数，最后轻微倾向更长文本
    scored.sort(key=lambda x: (x[0], x[1], min(2600, x[2].get("len",0))), reverse=True)
    return [it for _,__,it in scored]
