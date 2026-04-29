# src/tools/run_pipeline.py
# -*- coding: utf-8 -*-
import argparse
import os
import subprocess
from pathlib import Path

from run_context import init_context, mark_stage

# 项目根目录（包含 src/）
BASE_DIR = Path(__file__).resolve().parents[2]
PREPARED_DIR = BASE_DIR / "src" / "data" / "prepared"


def detect_latest_prepared_pid() -> str:
    if not PREPARED_DIR.exists():
        raise FileNotFoundError(f"未找到 prepared 目录: {PREPARED_DIR}")
    cands = [d for d in PREPARED_DIR.iterdir() if d.is_dir()]
    if not cands:
        raise FileNotFoundError(f"prepared 目录下没有任何提案目录: {PREPARED_DIR}")
    cands.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return cands[0].name


def run_cmd(cmd: list, env: dict):
    """在项目根目录下执行一个子命令，并在失败时直接抛出异常。"""
    print("正在执行：", " ".join(cmd))
    r = subprocess.run(cmd, cwd=BASE_DIR, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def run_full_pipeline(proposal_id: str = "", proposal_file: str = ""):
    """
    Canonical pipeline (single source of truth):
      1) prepare_proposal_text.py
      2) extract_facts_by_chunk.py
      3) build_dimensions_from_facts.py
      4) generate_questions.py
      5) llm_answering.py
      6) post_processing.py
      7) ai_expert_opinion.py
      8) generate_final_report.py
    """
    env = os.environ.copy()
    # Avoid stale parent-shell keys overriding local .env in child scripts.
    env.pop("OPENAI_API_KEY", None)
    env.pop("DEEPSEEK_API_KEY", None)

    # 1) 准备文本（可选显式文件/ID）
    prepare_cmd = ["python", "src/tools/prepare_proposal_text.py"]
    if proposal_file:
        prepare_cmd.extend(["--file", proposal_file])
    if proposal_id:
        prepare_cmd.extend(["--proposal_id", proposal_id])
    run_cmd(prepare_cmd, env=env)

    # 准备阶段后确定最终 pid（若未显式传入则以最新 prepared 为准）
    pid = proposal_id.strip() if proposal_id else detect_latest_prepared_pid()
    env["CURRENT_PROPOSAL_ID"] = pid
    init_context(pid, source_path=proposal_file or "")
    mark_stage("prepare_proposal_text", {"proposal_id": pid})

    # 2) facts 抽取
    run_cmd(["python", "src/tools/extract_facts_by_chunk.py", "--proposal_id", pid], env=env)
    mark_stage("extract_facts_by_chunk")

    # 3) 由 facts 构建维度
    run_cmd(["python", "src/tools/build_dimensions_from_facts.py", "--proposal_id", pid], env=env)
    mark_stage("build_dimensions_from_facts")

    # 4) 生成问题
    run_cmd(["python", "src/tools/generate_questions.py", "--proposal_id", pid], env=env)
    mark_stage("generate_questions")

    # 5) LLM 回答
    run_cmd(["python", "src/tools/llm_answering.py", "--proposal-id", pid], env=env)
    mark_stage("llm_answering")

    # 6) post-processing
    run_cmd(["python", "src/tools/post_processing.py", "--pid", pid], env=env)
    mark_stage("post_processing")

    # 7) AI 专家意见
    run_cmd(["python", "src/tools/ai_expert_opinion.py", "--pid", pid], env=env)
    mark_stage("ai_expert_opinion")

    # 8) 最终报告
    run_cmd(["python", "src/tools/generate_final_report.py", "--pid", pid], env=env)
    mark_stage("generate_final_report")

    print(f"Pipeline finished for proposal_id={pid}")


def parse_args():
    ap = argparse.ArgumentParser(description="Run canonical Yangtze pipeline with shared context")
    ap.add_argument("--proposal_id", default="", help="显式提案 ID（可选）")
    ap.add_argument("--file", default="", help="显式提案文件路径（可选）")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_full_pipeline(proposal_id=args.proposal_id, proposal_file=args.file)
