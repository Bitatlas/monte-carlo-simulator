from .base_model import BaseModel
from .monte_carlo import MonteCarloModel
from .brownian_motion import GeometricBrownianMotionModel
from .markov_chain import MarkovChainModel
from .path_integral import FeynmanPathIntegralModel

# Try to import GARCHModel, but don't fail if arch package is not available
try:
    from .garch import GARCHModel
    HAS_GARCH = True
except ImportError:
    HAS_GARCH = False
    # Create a placeholder for GARCHModel
    class GARCHModel:
        def __init__(self, *args, **kwargs):
            raise ImportError("The 'arch' package is required for GARCH models. Install it with 'pip install arch'.")

__all__ = [
    'BaseModel',
    'MonteCarloModel',
    'GeometricBrownianMotionModel',
    'GARCHModel',
    'MarkovChainModel',
    'FeynmanPathIntegralModel',
    'HAS_GARCH',
]
