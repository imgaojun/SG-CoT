import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


HOST_PROJECT_ROOT = Path("/mnt/disk/gaojun/research/progressive-ee")
CONTAINER_PROJECT_ROOT = Path("/workspace/project")


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def log(msg: str, log_file):
    line = f"[{time.strftime('%F %T')}] {msg}"
    print(line, flush=True)
    if log_file is not None:
        log_file.write(line + "\n")
        log_file.flush()


def wait_for_selection_summaries(runs, poll_seconds: int, log_file):
    while True:
        missing = [run["selection_summary"] for run in runs if not Path(run["selection_summary"]).exists()]
        if not missing:
            log("all selection summaries are ready", log_file)
            return
        log(f"waiting for selection summaries: {len(missing)} missing", log_file)
        for path in missing:
            log(f"missing: {path}", log_file)
        time.sleep(poll_seconds)


def normalize_project_path(path_value: str):
    """Map host absolute project paths to the Docker project mount when needed."""
    path = Path(path_value)
    if path.exists():
        return path.as_posix()
    if path.is_absolute():
        try:
            rel = path.relative_to(HOST_PROJECT_ROOT)
        except ValueError:
            return path.as_posix()
        candidate = CONTAINER_PROJECT_ROOT / rel
        if candidate.exists():
            return candidate.as_posix()
    return path.as_posix()


def build_tasks(manifest):
    tasks = []
    for run in manifest["runs"]:
        selection = load_json(Path(run["selection_summary"]))
        checkpoint_path = normalize_project_path(selection["best"]["checkpoint_path"])
        checkpoint_tag = selection["best"]["checkpoint_tag"]
        for eval_spec in run["evals"]:
            tasks.append(
                {
                    "tag": run["tag"],
                    "selection_summary": run["selection_summary"],
                    "checkpoint_tag": checkpoint_tag,
                    "checkpoint_path": checkpoint_path,
                    "split_name": eval_spec["name"],
                    "eval_jsonl": eval_spec["eval_jsonl"],
                    "output_dir": eval_spec["output_dir"],
                }
            )
    return tasks


def write_status(status_path: Path, payload):
    status_path.parent.mkdir(parents=True, exist_ok=True)
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_json", required=True)
    parser.add_argument("--gpu_ids", nargs="+", required=True)
    parser.add_argument("--poll_seconds", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--eval_script",
        default="src/stage2_quality_validation/eval_adapter_generation.py",
        help="Evaluation script to run for each formal split. The script must write predictions.jsonl and summary.json.",
    )
    parser.add_argument("--log_path", required=True)
    parser.add_argument("--status_json", required=True)
    parser.add_argument("--reuse_existing", action="store_true")
    args = parser.parse_args()

    manifest = load_json(Path(args.manifest_json))
    log_path = Path(args.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "a", encoding="utf-8")

    try:
        log(f"loaded manifest: {args.manifest_json}", log_file)
        log(f"gpu_ids={args.gpu_ids}", log_file)
        wait_for_selection_summaries(manifest["runs"], args.poll_seconds, log_file)

        tasks = build_tasks(manifest)
        pending = []
        completed = []
        skipped = []
        failed = []

        for task in tasks:
            summary_path = Path(task["output_dir"]) / "summary.json"
            if args.reuse_existing and summary_path.exists():
                task = dict(task)
                task["status"] = "skipped_existing"
                skipped.append(task)
            else:
                pending.append(task)

        log(f"prepared tasks: total={len(tasks)} pending={len(pending)} skipped={len(skipped)}", log_file)

        running = {}
        queue = list(pending)
        gpu_cycle = list(args.gpu_ids)

        while queue or running:
            while queue and len(running) < len(gpu_cycle):
                used_gpus = {item["gpu"] for item in running.values()}
                gpu = next(g for g in gpu_cycle if g not in used_gpus)
                task = queue.pop(0)
                output_dir = Path(task["output_dir"])
                output_dir.mkdir(parents=True, exist_ok=True)
                task_log = output_dir / "launcher.log"
                cmd = [
                    sys.executable,
                    args.eval_script,
                    "--base_model",
                    manifest["base_model"],
                    "--adapter_path",
                    task["checkpoint_path"],
                    "--eval_jsonl",
                    task["eval_jsonl"],
                    "--output_dir",
                    task["output_dir"],
                    "--batch_size",
                    str(args.batch_size),
                    "--max_new_tokens",
                    str(args.max_new_tokens),
                    "--temperature",
                    str(args.temperature),
                ]
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = gpu
                f = open(task_log, "a", encoding="utf-8")
                if not Path(task["checkpoint_path"]).exists():
                    raise FileNotFoundError(f"checkpoint path does not exist: {task['checkpoint_path']}")
                proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
                running[proc.pid] = {
                    "proc": proc,
                    "gpu": gpu,
                    "task": task,
                    "log_handle": f,
                    "task_log": task_log.as_posix(),
                }
                log(
                    f"launched pid={proc.pid} gpu={gpu} tag={task['tag']} split={task['split_name']} checkpoint={task['checkpoint_tag']}",
                    log_file,
                )

            time.sleep(10)

            finished_pids = []
            for pid, item in running.items():
                ret = item["proc"].poll()
                if ret is None:
                    continue
                item["log_handle"].close()
                record = dict(item["task"])
                record["gpu"] = item["gpu"]
                record["pid"] = pid
                record["task_log"] = item["task_log"]
                record["returncode"] = ret
                if ret == 0:
                    record["status"] = "completed"
                    completed.append(record)
                    log(
                        f"completed pid={pid} gpu={item['gpu']} tag={record['tag']} split={record['split_name']}",
                        log_file,
                    )
                else:
                    record["status"] = "failed"
                    failed.append(record)
                    log(
                        f"failed pid={pid} gpu={item['gpu']} tag={record['tag']} split={record['split_name']} returncode={ret}",
                        log_file,
                    )
                finished_pids.append(pid)

            for pid in finished_pids:
                del running[pid]

            status_payload = {
                "manifest_json": args.manifest_json,
                "base_model": manifest["base_model"],
                "gpu_ids": args.gpu_ids,
                "pending_count": len(queue),
                "running_count": len(running),
                "completed_count": len(completed),
                "skipped_count": len(skipped),
                "failed_count": len(failed),
                "completed": completed,
                "skipped": skipped,
                "failed": failed,
            }
            write_status(Path(args.status_json), status_payload)

        final_payload = {
            "manifest_json": args.manifest_json,
            "base_model": manifest["base_model"],
            "gpu_ids": args.gpu_ids,
            "pending_count": 0,
            "running_count": 0,
            "completed_count": len(completed),
            "skipped_count": len(skipped),
            "failed_count": len(failed),
            "completed": completed,
            "skipped": skipped,
            "failed": failed,
        }
        write_status(Path(args.status_json), final_payload)
        log(
            f"all evaluations finished: completed={len(completed)} skipped={len(skipped)} failed={len(failed)}",
            log_file,
        )
    finally:
        log_file.close()


if __name__ == "__main__":
    main()
