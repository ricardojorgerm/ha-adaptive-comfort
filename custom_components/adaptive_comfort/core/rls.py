"""Small pure-Python recursive least squares with exponential forgetting.

Dimensions are tiny (6 parameters), so plain lists are fine; numpy is
deliberately avoided to keep the integration dependency-free.
"""

from __future__ import annotations

# Cap P-trace growth under weak excitation so one noisy sample cannot slam
# theta after long unexcited stretches (forgetting still divides by lam).
TRACE_MAX_FACTOR = 1e4
# Skip the 1/lam inflation when phi carries almost no information.
MIN_EXCITATION = 1e-6


class RLS:
    def __init__(
        self,
        n: int,
        lam: float = 0.998,
        p0: float = 100.0,
        theta0: list[float] | None = None,
        *,
        trace_max: float | None = None,
    ) -> None:
        self.n = n
        self.lam = lam
        self.theta: list[float] = list(theta0) if theta0 else [0.0] * n
        self.p: list[list[float]] = [[p0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        self.samples = 0
        self.trace_max = float(trace_max) if trace_max is not None else TRACE_MAX_FACTOR * p0 * n

    def predict(self, phi: list[float]) -> float:
        return sum(t * x for t, x in zip(self.theta, phi, strict=True))

    def update(self, phi: list[float], y: float) -> float:
        """One RLS step; returns the pre-update residual."""
        n = self.n
        p_phi = [sum(self.p[i][j] * phi[j] for j in range(n)) for i in range(n)]
        phi_p_phi = sum(phi[i] * p_phi[i] for i in range(n))
        denom = self.lam + phi_p_phi
        gain = [p_phi[i] / denom for i in range(n)]
        residual = y - self.predict(phi)
        for i in range(n):
            self.theta[i] += gain[i] * residual
        # P = (P - K * (phi^T P)) / lam ; skip forgetting inflate when weakly excited.
        forget = self.lam if phi_p_phi >= MIN_EXCITATION else 1.0
        for i in range(n):
            for j in range(n):
                self.p[i][j] = (self.p[i][j] - gain[i] * p_phi[j]) / forget
        # Re-symmetrise to fight numerical drift.
        for i in range(n):
            for j in range(i + 1, n):
                avg = 0.5 * (self.p[i][j] + self.p[j][i])
                self.p[i][j] = avg
                self.p[j][i] = avg
        tr = self.trace
        if tr > self.trace_max and tr > 0.0:
            scale = self.trace_max / tr
            for i in range(n):
                for j in range(n):
                    self.p[i][j] *= scale
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
