# Yangtze

AI 辅助评审流水线（维度抽取 → 出题 → 结构化答题 → 后处理 → 专家评审与报告）。

开发时在仓库根目录运行工具模块，例如：

```bash
python -m src.tools.<模块名>
```

（或将 `src/tools` 加入 `PYTHONPATH`，按各脚本 docstring 说明执行。）

---

## 环境变量（`.env`）

复制 `.env.example` 为 `.env` 后按需修改。与 **题干长度 / 答题长度 / 报告篇幅** 相关的变量见下表及 `.env.example` 中的「文本长度与质量闸门」一节。

| 变量 | 默认（代码内） | 作用 |
|------|----------------|------|
| `MAX_QUESTION_CHARS` | `60` | 出题阶段：`question_zh` 目标字数上限；**语义完整优先**，程序侧 `clip_question_text` 仅在超长时按句读兜底截断。 |
| `MAX_PROMPT_QUESTION_CHARS` | `60` | 答题阶段：写入答题 prompt 的题干上限；应与出题侧保持一致或略大。 |
| `MAX_REVIEW_CHARS` / `MAX_ANSWER_CHARS` | `150` | 答题结构化输出：**每一行分点正文**（不含编号）及 **claims / evidence_hints / general_insights 每一条** 的字数上限；主约束由模型自控，程序按句读兜底截断。 |
| `MAX_AGGREGATED_REPORT_CHARS` | `2500` | 汇总类 Markdown 报告（一页摘要、综合报告等）正文总字符上限。 |

---

## 题干策略（与代码一致）

- 默认建议 **约 60 字** 一道题，**一整句、以「？」收束**，信息多时用紧凑改写而非指望事后截断。
- `generate_questions.py` 中 `clip_question_text()`：超长时优先在 `？` `。` `；` `，` 等处断开。
- `llm_answering.py` 的 `get_q_list()` 使用同一套裁剪逻辑与 `MAX_PROMPT_QUESTION_CHARS`。

---

## 答题策略（与代码一致）

- 每条分点 / 每条结构化短句默认 **约 150 字**，须在字数内 **语义完整**；超长时模型应删套话、合并表述。
- 程序裁剪：`answer` 分点与 `claims` 等列表项共用 **句读优先** 的 `_clip_answer_to_max_chars`。

---

## 后处理质量闸门（`post_processing.derive_quality_gates`）

- **`pass`**：当前实现为 **`True`**（不因覆盖率 / 一致性单独否决批次）。
- **`warnings`**：选中覆盖率低于 **0.85** → `selected_coverage_low`；平均题间重合度（`consistency_ratio`）低于 **0.08** → `consistency_low`。二者仅作 **告警**，写入 metrics 与报告。
- **`parse_success_ratio`**：仍写入 metrics；**不因解析降级单独卡闸门**。
- **`fail_reasons` / `reasons`**：预留硬否决（当前为空）；下游展示若有硬失败原因会与告警区分。

专家评审结论是否 `INSUFFICIENT_EVIDENCE` 仍以 metrics 中的 **`quality_gates.pass`** 等为准；详见 `ai_expert_opinion.py` 与生成报告脚本。

---

## 测试

```bash
python -m pytest tests/ -q
```
