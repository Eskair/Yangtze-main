from dotenv import load_dotenv
load_dotenv()

import argparse

from tools.run_pipeline import run_full_pipeline


def parse_args():
    p = argparse.ArgumentParser(description="Yangtze project pipeline entrypoint")
    p.add_argument(
        "--mode",
        choices=["all", "single"],
        default="all",
        help="all=运行完整流水线；single 为旧模式（已弃用）",
    )
    p.add_argument("--dimension", default="team", help="旧参数，仅用于兼容提示")
    p.add_argument("--question", default=None, help="旧参数，仅用于兼容提示")
    return p.parse_args()


def main():
    args = parse_args()
    if args.mode == "single":
        raise RuntimeError(
            "single 模式依赖已移除的 legacy chains。请使用 --mode all 运行完整流水线，"
            "或直接调用 src/tools 下的单阶段脚本。"
        )
    run_full_pipeline()

 
if __name__ == "__main__":
    main()
