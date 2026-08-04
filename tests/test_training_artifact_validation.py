import json
import tempfile
import unittest
from pathlib import Path

from scripts.validate_training_artifact_20260712 import (
    ArtifactValidationError,
    validate_artifact,
)


class TrainingArtifactValidationTests(unittest.TestCase):
    @staticmethod
    def _write_model(directory: Path, *, missing_second_shard: bool = False) -> None:
        directory.mkdir()
        (directory / "config.json").write_text("{}\n", encoding="utf-8")
        index = {
            "weight_map": {
                "layer.one": "model-00001-of-00002.safetensors",
                "layer.two": "model-00002-of-00002.safetensors",
            }
        }
        (directory / "model.safetensors.index.json").write_text(
            json.dumps(index) + "\n", encoding="utf-8"
        )
        (directory / "model-00001-of-00002.safetensors").write_bytes(b"one")
        if not missing_second_shard:
            (directory / "model-00002-of-00002.safetensors").write_bytes(b"two")

    @staticmethod
    def _write_state(path: Path, *, loss=1.5, grad_norm=0.25) -> None:
        path.write_text(
            json.dumps(
                {
                    "global_step": 1,
                    "log_history": [
                        {"step": 1, "loss": loss, "grad_norm": grad_norm}
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )

    def test_complete_smoke_artifact_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model"
            self._write_model(model)
            state = model / "trainer_state.json"
            self._write_state(state)
            result = validate_artifact(
                model,
                trainer_state_path=state,
                min_global_step=1,
                require_finite_step_log=True,
            )
            self.assertTrue(result["passed"])
            self.assertEqual(result["weight_file_count"], 2)
            self.assertEqual(result["trainer_state"]["global_step"], 1)

    def test_missing_weight_shard_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model"
            self._write_model(model, missing_second_shard=True)
            with self.assertRaisesRegex(ArtifactValidationError, "missing model shard"):
                validate_artifact(model)

    def test_nonfinite_step_log_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model"
            self._write_model(model)
            state = model / "trainer_state.json"
            self._write_state(state, grad_norm=float("nan"))
            with self.assertRaisesRegex(ArtifactValidationError, "finite loss"):
                validate_artifact(
                    model,
                    trainer_state_path=state,
                    min_global_step=1,
                    require_finite_step_log=True,
                )


if __name__ == "__main__":
    unittest.main()
