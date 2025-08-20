from __future__ import annotations
import numpy as np
import joblib
from pathlib import Path
from typing import Sequence

class LinTSBandit:
    """
    Contextual linear Thompson Sampling per arm.
    For each arm a: reward ~ x^T theta_a + noise, theta_a ~ N(mu, Sigma).
    """
    def __init__(self, arms: Sequence[str], d: int, alpha: float = 0.5, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.arms = list(arms)
        self.d = int(d)
        self.alpha = float(alpha)
        # A = (X^T X) + I, b = X^T y per arm
        self.A = {a: np.eye(self.d) for a in self.arms}
        self.b = {a: np.zeros(self.d) for a in self.arms}

    def _sample_theta(self, arm: str):
        A_inv = np.linalg.inv(self.A[arm])
        mu = A_inv @ self.b[arm]
        cov = (self.alpha ** 2) * A_inv
        return self.rng.multivariate_normal(mu, cov, check_valid="ignore")

    def choose(self, x: np.ndarray) -> str:
        scores = []
        for a in self.arms:
            theta = self._sample_theta(a)
            scores.append((a, float(x @ theta)))
        scores.sort(key=lambda t: t[1], reverse=True)
        return scores[0][0]

    def update(self, arm: str, reward: float, x: np.ndarray):
        Ax = np.outer(x, x)
        self.A[arm] += Ax
        self.b[arm] += reward * x

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"arms": self.arms, "A": self.A, "b": self.b, "d": self.d, "alpha": self.alpha}, path)

    @classmethod
    def load(cls, path: Path):
        obj = joblib.load(path)
        inst = cls(obj["arms"], obj["d"], alpha=obj["alpha"])
        inst.A, inst.b = obj["A"], obj["b"]
        return inst
