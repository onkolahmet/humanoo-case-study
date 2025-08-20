import numpy as np
from pathlib import Path
from src.models.bandit import LinTSBandit

def test_bandit_learns_and_persists(tmp_path: Path):
    arms = ["a", "b"]
    d = 4
    b = LinTSBandit(arms, d=d, alpha=0.3, seed=0)

    # Arm 'a' pays when x[1] is high; 'b' otherwise
    for t in range(400):
        x = np.array([1.0, float(t % 2), 0.2, 0.0])
        chosen = b.choose(x)
        reward = 1.0 if (chosen == "a" and x[1] > 0.5) or (chosen == "b" and x[1] <= 0.5) else 0.0
        b.update(chosen, reward, x)

    x_hi = np.array([1.0, 1.0, 0.2, 0.0])
    x_lo = np.array([1.0, 0.0, 0.2, 0.0])
    assert b.choose(x_hi) == "a"
    assert b.choose(x_lo) == "b"

    # save/load cycle
    p = tmp_path / "bandit.joblib"
    b.save(p)
    b2 = LinTSBandit.load(p)
    assert b2.arms == arms and b2.d == d and np.allclose(b2.A["a"], b.A["a"])
