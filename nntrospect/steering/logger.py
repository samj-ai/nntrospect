"""JSONL + pickle experiment logger for control vector runs."""

import json
import pickle
from datetime import datetime
from pathlib import Path


class ExperimentLogger:
    """Logs experiment results to a JSONL file with pickle sidecars for tensors.

    Each call to log_result() appends a JSON line and optionally saves
    lens_data, score_data, and tokens to separate .pkl files under
    per-type subdirectories.

    Usage::

        logger = ExperimentLogger(log_dir="/path/to/logs")
        logger.log_result(steering_word="dust", layers=[15,16,17], ...)

        # Load existing log
        logger = ExperimentLogger.from_latest(log_dir="/path/to/logs")
        df = logger.to_dataframe()
        lens = logger.load_ext_data(run_id=5, str_name="lens")
    """

    def __init__(self, log_dir="control_vector_experiments", log_file=None):
        self.log_dir = Path(log_dir).resolve()
        self.log_dir.mkdir(exist_ok=True)

        self.lens_dir = self.log_dir / "lens_data"
        self.lens_dir.mkdir(exist_ok=True)
        self.score_dir = self.log_dir / "score_data"
        self.score_dir.mkdir(exist_ok=True)
        self.tokens_dir = self.log_dir / "tokens_data"
        self.tokens_dir.mkdir(exist_ok=True)

        self._subdir = {
            "tokens": self.tokens_dir,
            "lens": self.lens_dir,
            "score": self.score_dir,
        }

        if log_file:
            self.log_file = self.log_dir / log_file
            self.run_counter = self._get_last_run_id() + 1
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_file = self.log_dir / f"experiments_{timestamp}.jsonl"
            self.run_counter = 0

    @classmethod
    def from_file(cls, log_file, log_dir="control_vector_experiments"):
        return cls(log_dir=log_dir, log_file=log_file)

    @classmethod
    def from_latest(cls, log_dir="control_vector_experiments"):
        log_dir = Path(log_dir)
        jsonl_files = sorted(log_dir.glob("experiments_*.jsonl"))
        if not jsonl_files:
            raise FileNotFoundError(f"No log files found in {log_dir}")
        return cls(log_dir=log_dir, log_file=jsonl_files[-1].name)

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def _get_last_run_id(self):
        results = self.read_all()
        return max((r["run_id"] for r in results), default=-1)

    def read_all(self):
        if not self.log_file.exists():
            return []
        with open(self.log_file, "r") as f:
            return [json.loads(line) for line in f]

    def query(self, **filters):
        """Filter results by any field value.

        Example::
            logger.query(steering_word="dust")
            logger.query(strength=1.0, layers=[15, 16, 17])
        """
        results = self.read_all()
        for key, value in filters.items():
            results = [r for r in results if r.get(key) == value]
        return results

    def get_run(self, run_id):
        runs = self.query(run_id=run_id)
        return runs[0] if runs else None

    def to_dataframe(self, remove_lens_file=False):
        import pandas as pd
        results = self.read_all()
        if remove_lens_file:
            results = [{k: v for k, v in r.items() if k != "lens_file"} for r in results]
        return pd.DataFrame(results)

    def load_ext_data(self, run_id=None, str_name=None, filename=None):
        """Load a pickle sidecar file.

        Either supply (run_id, str_name) to load by run ID, or supply
        filename directly. str_name must be one of: "lens", "tokens", "score".
        """
        if filename is not None:
            str_name = filename.split("_")[0]
            fn = filename
        elif run_id is not None and str_name is not None:
            fn = f"{str_name}_{run_id:04d}.pkl"
        else:
            raise ValueError("Provide either (run_id, str_name) or filename")

        fp = self._subdir[str_name] / fn
        if not fp.exists():
            raise FileNotFoundError(f"{str_name} data not found: {fp}")
        with open(fp, "rb") as f:
            return pickle.load(f)

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def log_result(
        self,
        steering_word,
        layers,
        strength,
        prompt,
        output,
        tokens=None,
        answer=None,
        lens_data=None,
        score_data=None,
        notes="",
    ):
        """Append one result to the log file and save any tensor sidecars."""
        result = {
            "run_id": self.run_counter,
            "timestamp": datetime.now().isoformat(),
            "steering_word": steering_word,
            "layers": layers,
            "strength": strength,
            "prompt": prompt,
            "output": output,
            "answer": answer,
            "notes": notes,
        }

        def _save(data, name):
            if data is not None:
                fn = f"{name}_{self.run_counter:04d}.pkl"
                with open(self._subdir[name] / fn, "wb") as f:
                    pickle.dump(data, f)
                result[f"{name}_file"] = fn

        _save(lens_data, "lens")
        _save(score_data, "score")
        _save(tokens, "tokens")

        with open(self.log_file, "a") as f:
            f.write(json.dumps(result) + "\n")

        self.run_counter += 1
        return result
