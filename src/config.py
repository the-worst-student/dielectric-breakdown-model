from dataclasses import dataclass

@dataclass
class SimulationConfig:
    nx: int = 128
    ny: int = 128

    voltage: float = 1.0

    tolerance: float = 1e-5
    max_iterations: int = 50_000
    max_steps: int = 5000
    random_seed: int = 42
    eta: float = 1.0
