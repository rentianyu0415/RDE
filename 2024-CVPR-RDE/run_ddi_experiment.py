import argparse
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch

from ddi.experiment import DDIExperiment
from ddi.qwen_client import QwenVLClient


DEFAULT_RUN = "run_logs/RSTPReid/20260714_224938_RDE_TAL+sr0.3_tau0.015_margin0.1_n0.0"


def parse_args():
    parser = argparse.ArgumentParser(description="RDE dual-grained disagreement interaction")
    parser.add_argument("--mode", choices=("all", "main", "ablation"), default="all")
    parser.add_argument("--config-file", default=DEFAULT_RUN + "/configs.yaml")
    parser.add_argument("--checkpoint", default=DEFAULT_RUN + "/best.pth")
    parser.add_argument("--root-dir", default="/root/datasets")
    parser.add_argument("--output-dir", default="ddi_outputs/rstpreid_200q_qwen36_flash")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--m", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument(
        "--base-url",
        default=os.getenv("DASHSCOPE_BASE_URL", ""),
        help="OpenAI-compatible base URL; may also be set with DASHSCOPE_BASE_URL",
    )
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--model", default="qwen3.6-flash-2026-04-16")
    parser.add_argument("--fallback-model", default="qwen3.6-flash")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--skip-baseline-check", action="store_true")
    args = parser.parse_args()
    if not args.base_url:
        parser.error("--base-url or DASHSCOPE_BASE_URL is required")
    if args.m > 2 * args.k:
        parser.error("--m cannot exceed the BGE/TSE Top-K union size")
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(str(output_dir / "ddi_experiment.log"), mode="a"),
        ],
    )
    client = QwenVLClient(
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        model=args.model,
        fallback_model=args.fallback_model,
        cache_path=str(output_dir / "qwen_cache.json"),
        timeout=args.timeout,
        max_retries=args.max_retries,
    )
    experiment = DDIExperiment(
        config_file=args.config_file,
        checkpoint_file=args.checkpoint,
        root_dir=args.root_dir,
        output_dir=str(output_dir),
        client=client,
        device=args.device,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        k=args.k,
        m=args.m,
        rounds=args.rounds,
    )
    if not args.skip_baseline_check:
        experiment.verify_full_baseline()
    preflight = not args.skip_preflight
    if args.mode == "main":
        experiment.run_main(do_preflight=preflight)
    elif args.mode == "ablation":
        experiment.run_main(rounds=1, do_preflight=preflight)
        experiment.run_ablation(do_preflight=False)
    else:
        experiment.run_main(do_preflight=preflight)
        experiment.run_ablation(do_preflight=False)


if __name__ == "__main__":
    main()
