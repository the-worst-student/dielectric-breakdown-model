from dataclasses import dataclass, field

@dataclass(frozen=True)
class WeakPathConfig:
    kind: str = "none"
    width: float = 0.05

    factor: float = 0.6

    start: tuple[float, float] = (0.5, 0.1)
    end: tuple[float, float] = (0.5, 1.0)

    branch_point: tuple[float, float] = (0.5, 0.5)
    left_end: tuple[float, float] = (0.25, 1.0)
    right_end: tuple[float, float] = (0.75, 1.0)

    left_factor: float = 0.6
    right_factor: float = 0.6

@dataclass
class SimulationConfig:
    nx: int = 128
    ny: int = 128
    voltage: float = 1.0
    lx: float = 1.0
    ly: float = 1.0
    mean_breakdown_field: float = 0.7
    tolerance: float = 1e-5
    max_iterations: int = 50_000
    max_growth_steps: int = 5000
    random_seed: int = 42
    eta: float = 1.0
    omega: float = 1.96
    residual_check_interval: int = 10
    correlation_length: int = 10
    weibull_shape: float = 5.0
    disorder_seed: int = 123
    needle_length: float = 0.1
    needle_width: float = 0.02
    weak_path: WeakPathConfig = field(default_factory=WeakPathConfig)