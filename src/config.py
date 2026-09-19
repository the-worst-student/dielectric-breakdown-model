from dataclasses import dataclass


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
