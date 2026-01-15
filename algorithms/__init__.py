"""
BMCA Algorithm Implementations
Tuna Girisken, Ege University
"""

from .standard_bmca import (
    StandardBMCA,
    BMCADataSet,
    ClockQuality,
    ClockAccuracy,
    ClockClass
)
from .net_bmca import NetBMCA, CentralityMetrics

# Legacy aliases for backward compatibility
CA_BMCA = NetBMCA
WeightedBMCA = NetBMCA

__all__ = [
    'StandardBMCA',
    'NetBMCA',
    'CA_BMCA',  # Legacy alias
    'WeightedBMCA',  # Legacy alias
    'BMCADataSet',
    'ClockQuality',
    'ClockAccuracy',
    'ClockClass',
    'CentralityMetrics'
]

__version__ = '2.0.0'
