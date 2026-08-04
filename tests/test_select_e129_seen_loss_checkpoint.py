import json
import tempfile
import unittest
from pathlib import Path

from scripts.select_e129_seen_loss_checkpoint import select_checkpoint


class SelectE129SeenLossCheckpointTest(unittest.TestCase):
    def make_run(self, root: Path, losses: list[tuple[int, float]]) -> Path:
        run_dir = root / "run"
        run_dir.mkdir()
        for step, _ in losses:
            (run_dir / f"checkpoint-{step}").mkdir()
        state = {
            "log_history": [
                {"step": step, "epoch": index + 1, "eval_loss": loss}
                for index, (step, loss) in enumerate(losses)
            ]
        }
        state_path = run_dir / f"checkpoint-{losses[-1][0]}" / "trainer_state.json"
        state_path.write_text(json.dumps(state), encoding="utf-8")
        return run_dir

    def test_selects_lowest_seen_loss(self):
        with tempfile.TemporaryDirectory() as temp:
            run_dir = self.make_run(Path(temp), [(73, 0.2), (146, 0.15), (219, 0.17)])
            result = select_checkpoint(run_dir, expected_candidates=3)
            self.assertEqual(result["selected_checkpoint"], "checkpoint-146")
            self.assertEqual(result["selection_split"], "dev_seen")

    def test_rejects_missing_retained_candidate(self):
        with tempfile.TemporaryDirectory() as temp:
            run_dir = self.make_run(Path(temp), [(73, 0.2), (146, 0.15), (219, 0.17)])
            (run_dir / "checkpoint-146").rmdir()
            with self.assertRaisesRegex(ValueError, "expected 3 retained candidates"):
                select_checkpoint(run_dir, expected_candidates=3)


if __name__ == "__main__":
    unittest.main()
