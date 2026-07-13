"""Small pure-Python recursive least squares with exponential forgetting.

Dimensions are tiny (6 parameters), so plain lists are fine; numpy is
deliberately avoided to keep the integration dependency-free.
"""

from __future__ import annotations


class RLS:
    def __init__(
        self,
        n: int,
        lam: float = 0.998,
        p0: float = 100.0,
        theta0: list[float] | None = None,
    ) -> None:
        self.n = n
        self.lam = lam
        self.theta: list[float] = list(theta0) if theta0 else [0.0] * n
        self.p: list[list[float]] = [[p0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        self.samples = 0

    def predict(self, phi: list[float]) -> float:
        return sum(t * x for t, x in zip(self.theta, phi, strict=True))

    def update(self, phi: list[float], y: float) -> float:
        """One RLS step; returns the pre-update residual."""
        n = self.n
        p_phi = [sum(self.p[i][j] * phi[j] for j in range(n)) for i in range(n)]
        denom = self.lam + sum(phi[i] * p_phi[i] for i in range(n))
        gain = [p_phi[i] / denom for i in range(n)]
        residual = y - self.predict(phi)
        for i in range(n):
            self.theta[i] += gain[i] * residual
        # P = (P - K * (phi^T P)) / lam ; phi^T P == p_phi for symmetric P
        for i in range(n):
            for j in range(n):
                self.p[i][j] = (self.p[i][j] - gain[i] * p_phi[j]) / self.lam
        # Re-symmetrise to fight numerical drift.
        for i in range(n):
            for j in range(i + 1, n):
                avg = 0.5 * (self.p[i][j] + self.p[j][i])
                self.p[i][j] = avg
                self.p[j][i] = avg
        self.samples += 1
        return residual

    @property
    def trace(self) -> float:
        return sum(self.p[i][i] for i in range(self.n))

    def to_dict(self) -> dict:
        return {
            "theta": list(self.theta),
            "p": [list(row) for row in self.p],
            "samples": self.samples,
            "lam": self.lam,
        }

    @classmethod
    def from_dict(cls, data: dict, n: int) -> RLS:
        rls = cls(n, lam=data.get("lam", 0.998))
        theta = data.get("theta")
        p = data.get("p")
        if theta and len(theta) == n:
            rls.theta = list(theta)
        if p and len(p) == n and all(len(row) == n for row in p):
            rls.p = [list(row) for row in p]
        rls.samples = int(data.get("samples", 0))
        return rls
