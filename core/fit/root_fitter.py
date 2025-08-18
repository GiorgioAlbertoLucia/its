#!/usr/bin/env python3
"""
ROOT-based Bethe-Bloch fitter using traditional ROOT fitting methods
"""

from typing import Optional
from ROOT import TGraphErrors, TF1

from fit_utils import FitResult, calculate_success_score
from fit_parameter_estimation import ParameterEstimator

class RootBetheFitter:
    """ROOT-based Bethe-Bloch fitter using traditional ROOT fitting methods"""
    
    def __init__(self):
        pass
    
    def setup_function(self, func, graph, particle, func_type):
        """Set initial parameters and limits for a function"""
        x_data, y_data, y_errors = ParameterEstimator.extract_data(graph)
        params, limits = ParameterEstimator.estimate_initial_params_smart(
            x_data, y_data, func_type)
        
        config = {
            'params': params,
            'limits': limits
        }
        
        func.SetParameters(*config['params'])
        
        for i, (low, high) in enumerate(config['limits']):
            func.SetParLimits(i, low, high)

    @staticmethod
    def compute_chi2_ndf(func, graph):
        """Manually compute chi2 for a ROOT function and graph"""
        if graph.GetN() == 0:
            return float('inf'), 0
        
        chi2 = 0
        n_points = 0
        
        for ix in range(graph.GetN()):
            x = graph.GetPointX(ix)
            y_obs = graph.GetPointY(ix)
            y_err_obs = graph.GetErrorY(ix)
            
            if y_err_obs <= 0:
                continue
            
            try:
                y_pred = func.Eval(x)
                if not (abs(y_pred) < 1e10):  # Avoid inf/nan
                    continue
                
                residual = y_obs - y_pred
                chi2 += (residual * residual) / (y_err_obs * y_err_obs)
                n_points += 1
                
            except:
                continue
        
        ndf = n_points - func.GetNpar() if n_points > func.GetNpar() else 0
        
        return chi2, ndf
    
    def try_single_fit(self, graph, func, strategy):
        """Try a single fitting strategy"""
        try:
            result = graph.Fit(func, strategy + "SQ0")
            
            if result and result.IsValid() and result.Ndf() > 0:
                chi2, ndf = self.compute_chi2_ndf(func, graph)
                return result, chi2, ndf
            else:
                return None, float('inf'), 0
                
        except Exception:
            return None, float('inf'), 0
    
    def fit_with_strategies(self, graph, func, strategies=None):
        """Try multiple fitting strategies and return the best result"""
        if strategies is None:
            strategies = ["RMS+", "RME+", "RS+", "WL+"]
        
        best_chi2 = float('inf')
        best_chi2_ndf = float('inf')
        best_chi2_ndf_from_one = float('inf')
        best_result = None
        best_strategy = ""
        
        original_params = [func.GetParameter(i) for i in range(func.GetNpar())]
        
        print(f"    ROOT strategies: {strategies}")
        
        for strategy in strategies:
            # Reset parameters
            for i, param in enumerate(original_params):
                func.SetParameter(i, param)
            
            result, chi2, ndf = self.try_single_fit(graph, func, strategy)
            chi2_ndf = chi2 / ndf if ndf > 0 else float('inf')
            chi2_ndf_from_one = abs(chi2_ndf - 1.0)
            
            if result:
                print(f"      {strategy:4s}: χ²/ndf = {chi2_ndf:.3f}")
                if chi2_ndf_from_one < best_chi2_ndf_from_one:
                    best_chi2 = chi2
                    best_chi2_ndf = chi2_ndf
                    best_chi2_ndf_from_one = chi2_ndf_from_one
                    best_result = result
                    best_strategy = strategy
            else:
                print(f"      {strategy:4s}: Failed")
        
        if best_result:
            print(f"    → Best ROOT strategy: {best_strategy} (χ²/ndf = {best_chi2_ndf:.3f})")
        
        return best_result, best_strategy, best_chi2, best_chi2_ndf, best_chi2_ndf_from_one
    
    def fit_function(self, graph: TGraphErrors, func: TF1, particle: str, func_type: str) -> Optional[FitResult]:
        """Main fitting method using ROOT strategies"""
        
        print(f"    ROOT optimization:")
        self.setup_function(func, graph, particle, func_type)
        result, strategy, chi2, chi2_ndf, chi2_ndf_from_one = self.fit_with_strategies(graph, func)
        
        if result:
            fit_result = FitResult(
                function=func,
                graph=graph,
                result=result,
                strategy=strategy,
                chi2_ndf=chi2_ndf,
                chi2_ndf_from_one=chi2_ndf_from_one,
                params=[func.GetParameter(i) for i in range(func.GetNpar())],
                errors=[func.GetParError(i) for i in range(func.GetNpar())],
                optimizer="ROOT",
                chi2=chi2,
                ndf=result.Ndf()
            )
            fit_result.success_score = calculate_success_score(fit_result)
            
            return fit_result
        
        return None