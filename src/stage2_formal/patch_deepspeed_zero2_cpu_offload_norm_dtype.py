import argparse
from pathlib import Path


OLD_TENSOR_LINE = (
    "                norm = torch.tensor(self.complete_grad_norm_calculation_for_cpu_offload(self.params_in_partition[i]),\n"
    "                                    device=self.device)\n"
)

NEW_TENSOR_LINE = (
    "                norm = torch.tensor(\n"
    "                    self.complete_grad_norm_calculation_for_cpu_offload(self.params_in_partition[i]),\n"
    "                    device=self.device,\n"
    "                    dtype=torch.float32,\n"
    "                )\n"
)

OLD_SENTINEL = "            total_norm = -1\n"
NEW_SENTINEL = "            total_norm = -1.0\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        default="/opt/venv/lib/python3.12/site-packages/deepspeed/runtime/zero/stage_1_and_2.py",
    )
    args = parser.parse_args()

    path = Path(args.target)
    text = path.read_text(encoding="utf-8")

    changed = False

    if OLD_TENSOR_LINE in text:
        text = text.replace(OLD_TENSOR_LINE, NEW_TENSOR_LINE)
        changed = True

    if OLD_SENTINEL in text:
        text = text.replace(OLD_SENTINEL, NEW_SENTINEL)
        changed = True

    path.write_text(text, encoding="utf-8")
    print(
        {
            "target": str(path),
            "changed": changed,
            "tensor_line_patched": NEW_TENSOR_LINE in text,
            "sentinel_patched": NEW_SENTINEL in text,
        }
    )


if __name__ == "__main__":
    main()
