#!/usr/bin/env python3
"""
Parameter estimation utilities for Bethe-Bloch fitting
"""

import numpy as np
from scipy.stats import linregress
from typing import Tuple, List, Dict
from ROOT import TGraphErrors

class ParameterEstimator:
    """Utilities for estimating initial parameters for Bethe-Bloch fits"""
    
    @staticmethod
    def extract_data(graph: TGraphErrors) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract data from ROOT TGraphErrors"""
        n_points = graph.GetN()
        x_data = np.array([graph.GetX()[i] for i in range(n_points)])
        y_data = np.array([graph.GetY()[i] for i in range(n_points)])
        y_errors = np.array([graph.GetEY()[i] for i in range(n_points)])
        
        # Handle zero errors
        y_errors = np.where(y_errors == 0, np.std(y_data) * 0.1, y_errors)
        
        return x_data, y_data, y_errors
    
    @staticmethod
    def analyze_bethe_bloch_behavior(x_data, y_data):
        """Analyze data to understand Bethe-Bloch behavior for better parameter estimation"""
        
        # Find minimum (most probable value region)
        min_idx = np.argmin(y_data)
        bg_min = x_data[min_idx]
        y_min = y_data[min_idx]
        
        # Low βγ region (1/βγ dominance)
        low_bg_mask = x_data < bg_min
        if np.sum(low_bg_mask) > 2:
            low_bg_x = x_data[low_bg_mask]
            low_bg_y = y_data[low_bg_mask]
        else:
            low_bg_x = x_data[:len(x_data)//3]
            low_bg_y = y_data[:len(y_data)//3]
        
        # High βγ region (relativistic rise)
        high_bg_mask = x_data > bg_min * 2
        if np.sum(high_bg_mask) > 2:
            high_bg_x = x_data[high_bg_mask]
            high_bg_y = y_data[high_bg_mask]
        else:
            high_bg_x = x_data[len(x_data)//2:]
            high_bg_y = y_data[len(y_data)//2:]
        
        # Calculate slopes for parameter estimation
        try:
            # Low βγ slope (should be negative for 1/βγ behavior)
            if len(low_bg_x) > 2:
                low_slope, _, _, _, _ = linregress(np.log(low_bg_x), np.log(low_bg_y))
            else:
                low_slope = -1.5
            
            # High βγ slope (relativistic rise)
            if len(high_bg_x) > 2:
                high_slope, _, _, _, _ = linregress(np.log(high_bg_x), np.log(high_bg_y))
            else:
                high_slope = 0.1
                
        except:
            low_slope = -1.5
            high_slope = 0.1
        
        return {
            'bg_min': bg_min,
            'y_min': y_min,
            'low_slope': low_slope,
            'high_slope': high_slope,
            'y_range': np.max(y_data) - np.min(y_data)
        }
    
    @staticmethod
    def estimate_initial_params_smart(x_data, y_data, func_type='bethe_bloch_aleph'):
        """Smart parameter estimation using data characteristics and physics"""
        
        if func_type == 'bethe_bloch_aleph':
            # Analyze Bethe-Bloch specific behavior
            analysis = ParameterEstimator.analyze_bethe_bloch_behavior(x_data, y_data)
            
            # ALEPH formula: (kp2 - β^kp4 - log(kp3 + (1/βγ)^kp5)) * kp1 / β^kp4
            
            # Parameter estimation based on physics and data analysis:
            
            # kp1: Overall scale factor (related to material properties)
            # Should scale with the magnitude of energy loss
            kp1_est = analysis['y_min'] * 2.0  # Start with twice the minimum
            
            # kp2: Controls the plateau height in the relativistic region
            # Should be related to the maximum dE/dx value
            kp2_est = 10.0  # Typical value for relativistic plateau
            
            # kp3: Controls the transition from 1/βγ to relativistic behavior
            # Typical values are 0.1 - 10
            kp3_est = 1.0
            
            # kp4: Power of β in denominator and β^kp4 term
            # Physics suggests values around 2 (from β² in classical formula)
            kp4_est = 2.0
            
            # kp5: Power in the logarithmic term (1/βγ)^kp5
            # Related to how steep the low βγ region is
            kp5_est = abs(analysis['low_slope']) if abs(analysis['low_slope']) < 5 else 2.0
            
            # Adjust kp1 based on minimum value and other parameters
            # At minimum: derivative = 0, so we can estimate kp1 better
            beta_min = analysis['bg_min'] / np.sqrt(1 + analysis['bg_min']**2)
            kp1_est = analysis['y_min'] * (beta_min**kp4_est) / max(0.1, 
                (kp2_est - beta_min**kp4_est - np.log(kp3_est + (1/analysis['bg_min'])**kp5_est)))
            
            params = [
                max(0.1, kp1_est),     # kp1: scale factor
                max(1.0, kp2_est),     # kp2: plateau level  
                max(0.01, kp3_est),    # kp3: transition parameter
                max(0.5, min(4.0, kp4_est)),  # kp4: β power (constrained)
                max(0.1, min(5.0, kp5_est))   # kp5: log term power (constrained)
            ]
            
            # Physics-based bounds for ALEPH parameters
            bounds = [
                (0.01, 100.0),    # kp1: positive scale
                (0.5, 50.0),      # kp2: reasonable plateau
                (0.001, 100.0),   # kp3: transition parameter
                (0.1, 8.0),       # kp4: β power (physics constraint)
                (0.05, 10.0)      # kp5: log power
            ]
            
        elif func_type == '3p_bethe_bloch':
            # Keep the existing logic for simple power law
            try:
                c_est = np.min(y_data) * 0.9
                y_shifted = y_data - c_est
                
                valid_mask = (y_shifted > 0) & (x_data > 0)
                if np.sum(valid_mask) > 3:
                    log_y = np.log(y_shifted[valid_mask])
                    log_x = np.log(x_data[valid_mask])
                    slope, intercept, _, _, _ = linregress(log_x, log_y)
                    
                    a_est = np.exp(intercept)
                    b_est = -slope
                else:
                    a_est = np.mean(y_shifted) * np.mean(x_data)
                    b_est = 1.5
                    
            except:
                a_est = 2.0
                b_est = 1.5
                c_est = np.min(y_data)
            
            params = [max(0.1, a_est), max(0.1, abs(b_est)), max(0.1, c_est)]
            bounds = [(0.1, 20.0), (0.3, 5.0), (0.1, 10.0)]
        
        else:
            params = [1.0]
            bounds = [(0.1, 10.0)]
        
        return params, bounds