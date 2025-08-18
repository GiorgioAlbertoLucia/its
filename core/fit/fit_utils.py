#!/usr/bin/env python3
"""
Data structures for Bethe-Bloch fitting results
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from ROOT import TF1, TCanvas, TGraphErrors

@dataclass
class FitResult:
    function: TF1
    graph: TGraphErrors
    result: Optional['ROOT.FitResult']
    strategy: str
    chi2_ndf: float
    chi2_ndf_from_one: float
    params: List[float]
    errors: List[float]
    optimizer: str = "ROOT"
    success_score: float = 0.0
    chi2: float = 0.0
    ndf: int = 0

@dataclass
class CanvasSpace:
    canvas: TCanvas
    content: list

def calculate_success_score(fit_result: FitResult) -> float:
    """Calculate combined success score for ranking fits"""
    chi2_score = fit_result.chi2_ndf_from_one
    
    # Penalize very high chi2/ndf
    if fit_result.chi2_ndf > 10:
        chi2_score += (fit_result.chi2_ndf - 10) * 2
    
    # Penalize very low chi2/ndf (overfitting)
    if fit_result.chi2_ndf < 0.1:
        chi2_score += (0.1 - fit_result.chi2_ndf) * 5
    
    return chi2_score