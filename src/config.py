from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    mode: str
    random_state: int
    sample_frac: float
    cv: int
    rf_use_randomized: bool
    rf_n_iter: int
    svm_use_search: bool
    svm_use_randomized: bool
    svm_n_iter: int


def get_experiment_config(mode: str = "MODO_RAPIDO", random_state: int = 42, sample_frac: float | None = None) -> ExperimentConfig:
    """Return runtime configuration for Projeto 8 experiments.

    `MODO_RAPIDO` is optimized for fast end-to-end validation.
    `MODO_COMPLETO` uses larger sampling and broader searches for article runs.
    """
    mode_key = mode.strip().upper()
    if mode_key not in {"MODO_RAPIDO", "MODO_COMPLETO"}:
        raise ValueError("mode deve ser 'MODO_RAPIDO' ou 'MODO_COMPLETO'.")

    if mode_key == "MODO_COMPLETO":
        default = ExperimentConfig(
            mode=mode_key,
            random_state=random_state,
            # ~65% keeps representativeness while reducing runtime to fit ~1h on Colab.
            sample_frac=0.65,
            cv=3,
            rf_use_randomized=True,
            rf_n_iter=24,
            svm_use_search=True,
            svm_use_randomized=True,
            svm_n_iter=8,
        )
    else:
        default = ExperimentConfig(
            mode=mode_key,
            random_state=random_state,
            # ~20% accelerates end-to-end validation in a few minutes.
            sample_frac=0.20,
            cv=2,
            rf_use_randomized=True,
            rf_n_iter=10,
            svm_use_search=False,
            svm_use_randomized=False,
            svm_n_iter=0,
        )

    chosen_frac = default.sample_frac if sample_frac is None else float(sample_frac)
    if not (0 < chosen_frac <= 1):
        raise ValueError("sample_frac deve estar no intervalo (0, 1].")
    return replace(default, sample_frac=chosen_frac)


def ensure_project_dirs(root: str | Path) -> dict[str, Path]:
    """Create and return the main project directories.

    Args:
        root: Project root path.

    Returns:
        Dict with Path values for keys: `raw_dir`, `reports_dir`, `figures_dir`.
    """
    root = Path(root)
    paths = {
        "raw_dir": root / "data" / "raw",
        "reports_dir": root / "reports",
        "figures_dir": root / "reports" / "figures",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths
