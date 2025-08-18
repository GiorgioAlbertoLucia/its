#!/usr/bin/env python3
"""
Scipy-based Bethe-Bloch fitter with robust optimization strategies
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution, curve_fit, basinhopping
from typing import Optional, List
from ROOT import TGraphErrors, TF1

from torchic.physics import BetheBloch
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'core', 'fit')))
from fit_utils import FitResult, calculate_success_score
from fit_parameter_estimation import ParameterEstimator

class ScipyBetheFitter:
    """Scipy-based Bethe-Bloch fitter with robust optimization strategies"""
    
    def __init__(self):
        self.x_data = None
        self.y_data = None
        self.y_errors = None
          
    def bethe_bloch_wrapper(self, x, *params):
        """Wrapper for Bethe-Bloch function compatible with scipy"""
        try:
            x_arr = np.atleast_1d(x)
            result = np.zeros_like(x_arr, dtype=float)
            
            for i, xi in enumerate(x_arr):
                result[i] = BetheBloch(xi, *params)
            
            return result if len(result) > 1 else result[0]
        except:
            return np.full_like(x, 1e6)
    
    def simple_power_law(self, x, a, b, c):
        """Simple power law: a/x^b + c"""
        return a / np.power(x, b) + c
    
    def generate_parameter_grid(self, base_params, bounds, n_trials=20):
        """Generate grid of initial parameters for robustness"""
        param_sets = [base_params]
        
        # Random variations
        np.random.seed(42)
        for _ in range(n_trials):
            varied_params = []
            for i, (param, (low, high)) in enumerate(zip(base_params, bounds)):
                log_low = np.log10(max(low, 1e-6))
                log_high = np.log10(high)
                log_param = np.random.uniform(log_low, log_high)
                varied_params.append(10**log_param)
            param_sets.append(varied_params)
        
        # Systematic variations
        for factor in [0.5, 2.0, 0.1, 5.0]:
            scaled_params = [np.clip(p * factor, low, high) 
                           for p, (low, high) in zip(base_params, bounds)]
            param_sets.append(scaled_params)
        
        return param_sets
    
    def generate_bethe_bloch_parameter_grid(self, base_params, bounds, particle):
        """Generate specialized parameter grid for Bethe-Bloch ALEPH fits"""
        param_sets = [base_params]
        
        # Literature values for different particles (approximate)
        literature_params = {
            'Pi': [1.0, 10.0, 0.5, 2.0, 2.5],   # Pions
            'Ka': [1.2, 12.0, 0.4, 2.0, 2.8],   # Kaons  
            'Pr': [1.5, 15.0, 0.3, 2.0, 3.0],   # Protons
            'De': [2.0, 18.0, 0.2, 2.0, 3.2],   # Deuterons
            'He': [3.0, 25.0, 0.15, 2.0, 3.5]   # Helium
        }
        
        # Add literature-based starting points
        if particle in literature_params:
            lit_params = literature_params[particle]
            # Scale to bounds
            scaled_params = []
            for i, (param, (low, high)) in enumerate(zip(lit_params, bounds)):
                scaled_params.append(np.clip(param, low, high))
            param_sets.append(scaled_params)
        
        # Add systematic variations of physics-motivated values
        physics_variations = [
            # [kp1, kp2, kp3, kp4, kp5]
            [0.5, 8.0, 1.0, 2.0, 2.0],    # Low energy loss
            [2.0, 15.0, 0.5, 2.0, 2.5],   # Medium energy loss
            [5.0, 25.0, 0.2, 2.0, 3.0],   # High energy loss
            [1.0, 12.0, 2.0, 1.5, 2.2],   # Different kp4
            [1.5, 10.0, 0.8, 2.5, 1.8],   # Different kp4, kp5
        ]
        
        for params in physics_variations:
            # Ensure parameters are within bounds
            clipped_params = []
            for i, (param, (low, high)) in enumerate(zip(params, bounds)):
                clipped_params.append(np.clip(param, low, high))
            param_sets.append(clipped_params)
        
        # Add random variations (fewer than before, since we have better starting points)
        np.random.seed(42)
        for _ in range(10):
            varied_params = []
            for i, (base_param, (low, high)) in enumerate(zip(base_params, bounds)):
                # Use both log-uniform and uniform sampling
                if i in [0, 1]:  # kp1, kp2 - use log sampling
                    log_low = np.log10(max(low, 1e-6))
                    log_high = np.log10(high)
                    log_param = np.random.uniform(log_low, log_high)
                    varied_params.append(10**log_param)
                else:  # other parameters - uniform sampling
                    varied_params.append(np.random.uniform(low, high))
            param_sets.append(varied_params)
        
        return param_sets
    
    def chi2_objective(self, params, x_data, y_data, y_errors, func_type):
        """Chi-squared objective function for scipy optimization"""
        try:
            if func_type == 'bethe_bloch_aleph':
                y_pred = self.bethe_bloch_wrapper(x_data, *params)
            else:
                y_pred = self.simple_power_law(x_data, *params)
            
            chi2 = np.sum(((y_data - y_pred) / y_errors)**2)
            
            if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
                return 1e6
            
            return chi2
        except:
            return 1e6
    
    def _estimate_parameter_errors(self, params, x_data, y_data, y_errors, func_type):
        """Estimate parameter errors using finite differences"""
        try:
            eps = 1e-8
            errors = []
            
            for i in range(len(params)):
                params_plus = params.copy()
                params_minus = params.copy()
                
                delta = max(eps, abs(params[i]) * eps)
                params_plus[i] += delta
                params_minus[i] -= delta
                
                chi2_plus = self.chi2_objective(params_plus, x_data, y_data, y_errors, func_type)
                chi2_minus = self.chi2_objective(params_minus, x_data, y_data, y_errors, func_type)
                
                second_deriv = (chi2_plus - 2*self.chi2_objective(params, x_data, y_data, y_errors, func_type) + chi2_minus) / (delta**2)
                
                if second_deriv > 0:
                    error = np.sqrt(2.0 / second_deriv)
                else:
                    error = abs(params[i]) * 0.1
                
                errors.append(error)
            
            return np.array(errors)
        except:
            return np.ones(len(params)) * 0.1
    
    def fit_function(self, graph: TGraphErrors, func: TF1, particle: str, func_type: str) -> Optional[FitResult]:
        """Main fitting method that tries multiple scipy strategies"""
        
        x_data, y_data, y_errors = ParameterEstimator.extract_data(graph)
        base_params, bounds = ParameterEstimator.estimate_initial_params_smart(x_data, y_data, func_type)
        
        if func_type == 'bethe_bloch_aleph':
            param_sets = self.generate_bethe_bloch_parameter_grid(base_params, bounds, particle)
        else:
            param_sets = self.generate_parameter_grid(base_params, bounds, n_trials=15)
        
        best_result = None
        best_chi2_ndf_from_one = float('inf')
        best_method = ""
        
        methods = [
            ('curve_fit', self._try_curve_fit),
            ('differential_evolution', self._try_differential_evolution),
            ('basinhopping', self._try_basinhopping),
            ('minimize_grid', self._try_minimize_grid)
        ]
        
        print(f"    Scipy optimization:")
        
        for method_name, method_func in methods:
            try:
                if method_name == 'differential_evolution':
                    result = method_func(x_data, y_data, y_errors, func_type, bounds)
                elif method_name == 'basinhopping':
                    result = method_func(x_data, y_data, y_errors, func_type, base_params, bounds)
                else:
                    result = method_func(x_data, y_data, y_errors, func_type, param_sets, bounds)
                
                if result is None:
                    print(f"      {method_name:20s}: Failed")
                else:
                    ndf = len(x_data) - len(result['params'])
                    chi2_ndf = result['chi2'] / ndf if ndf > 0 else float('inf')
                    chi2_ndf_from_one = abs(chi2_ndf - 1.0)
                    if chi2_ndf_from_one < best_chi2_ndf_from_one:
                        best_chi2_ndf_from_one = chi2_ndf_from_one
                        best_result = result
                        best_method = method_name
                        ndf = len(x_data) - len(result['params'])
                        print(f"      {method_name:20s}: χ²/ndf = {chi2_ndf:.3f}")
                    else:
                        print(f"      {method_name:20s}: χ²/ndf = {chi2_ndf:.3f} (worse)")
            except Exception as e:
                print(f"      {method_name:20s}: Error")
        
        if best_result:
            print(f"    → Best scipy method: {best_method}")
            
            # Setup ROOT function with scipy results
            for i, param in enumerate(best_result['params']):
                func.SetParameter(i, param)
                func.SetParError(i, best_result['errors'][i])
            
            # Create FitResult
            ndf = len(x_data) - len(best_result['params'])
            chi2_ndf = best_result['chi2'] / ndf if ndf > 0 else float('inf')
            
            fit_result = FitResult(
                function=func,
                graph=graph,
                result=None,
                strategy=best_method,
                chi2_ndf=chi2_ndf,
                chi2_ndf_from_one=abs(chi2_ndf - 1.0),
                params=list(best_result['params']),
                errors=list(best_result['errors']),
                optimizer="scipy",
                chi2=best_result['chi2'],
                ndf=ndf
            )
            fit_result.success_score = calculate_success_score(fit_result)
            
            return fit_result
        
        return None
    
    def _try_curve_fit(self, x_data, y_data, y_errors, func_type, param_sets, bounds):
        """Try curve_fit with multiple initial parameters"""
        best_chi2 = float('inf')
        best_params = None
        best_errors = None
        
        func = self.bethe_bloch_wrapper if func_type == 'bethe_bloch_aleph' else self.simple_power_law
        
        for params in param_sets:
            try:
                popt, pcov = curve_fit(
                    func, x_data, y_data, 
                    p0=params, 
                    sigma=y_errors,
                    absolute_sigma=True,
                    maxfev=5000
                )
                
                chi2 = self.chi2_objective(popt, x_data, y_data, y_errors, func_type)
                if chi2 < best_chi2:
                    best_chi2 = chi2
                    best_params = popt
                    best_errors = np.sqrt(np.diag(pcov))
            except:
                continue
        
        if best_params is not None:
            return {'params': best_params, 'errors': best_errors, 'chi2': best_chi2}
        return None
    
    def _try_differential_evolution(self, x_data, y_data, y_errors, func_type, bounds):
        """Try differential evolution global optimizer"""
        try:
            result = differential_evolution(
                lambda params: self.chi2_objective(params, x_data, y_data, y_errors, func_type),
                bounds,
                seed=42,
                maxiter=1000,
                atol=1e-8
            )
            
            if result.success:
                errors = self._estimate_parameter_errors(result.x, x_data, y_data, y_errors, func_type)
                return {'params': result.x, 'errors': errors, 'chi2': result.fun}
        except:
            pass
        return None
    
    def _try_basinhopping(self, x_data, y_data, y_errors, func_type, base_params, bounds):
        """Try basin hopping for global optimization"""
        try:
            result = basinhopping(
                lambda params: self.chi2_objective(params, x_data, y_data, y_errors, func_type),
                base_params,
                niter=100,
                minimizer_kwargs={'bounds': bounds, 'method': 'L-BFGS-B'}
            )
            
            if result.lowest_optimization_result.success:
                params = result.x
                errors = self._estimate_parameter_errors(params, x_data, y_data, y_errors, func_type)
                return {'params': params, 'errors': errors, 'chi2': result.fun}
        except:
            pass
        return None
    
    def _try_minimize_grid(self, x_data, y_data, y_errors, func_type, param_sets, bounds):
        """Try scipy minimize with multiple starting points"""
        best_chi2 = float('inf')
        best_result = None
        
        for params in param_sets[:10]:
            try:
                result = minimize(
                    lambda p: self.chi2_objective(p, x_data, y_data, y_errors, func_type),
                    params,
                    bounds=bounds,
                    method='L-BFGS-B'
                )
                
                if result.success and result.fun < best_chi2:
                    best_chi2 = result.fun
                    errors = self._estimate_parameter_errors(result.x, x_data, y_data, y_errors, func_type)
                    best_result = {'params': result.x, 'errors': errors, 'chi2': result.fun}
            except:
                continue
        
        return best_result