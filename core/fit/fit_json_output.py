#!/usr/bin/env python3
"""
JSON output utilities for storing fit parameters
"""

import json
import datetime
from typing import Dict, Any
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'core', 'fit')))
from fit_utils import FitResult

def create_fit_parameters_dict(particle: str, best_result: FitResult, best_func_name: str) -> Dict[str, Any]:
    """Create dictionary with fit parameters for JSON serialization"""
    
    # Get parameter names from the function
    func = best_result.function
    param_names = []
    param_dict = {}
    
    if best_func_name == 'bethe_bloch_aleph':
        param_names = ['kp1', 'kp2', 'kp3', 'kp4', 'kp5']
    elif best_func_name == '3p_bethe_bloch':
        param_names = ['coeff', 'power', 'offset']
    else:
        # Generic parameter names
        param_names = [f'par_{i}' for i in range(len(best_result.params))]
    
    # Create parameter dictionary
    for i, (name, value, error) in enumerate(zip(param_names, best_result.params, best_result.errors)):
        param_dict[name] = {
            'value': float(value),
            'error': float(error),
            'index': i
        }
    
    fit_info = {
        'particle': particle,
        'function_type': best_func_name,
        'optimizer': best_result.optimizer,
        'strategy': best_result.strategy,
        'chi2': float(best_result.chi2),
        'ndf': int(best_result.ndf),
        'chi2_ndf': float(best_result.chi2_ndf),
        'chi2_ndf_from_one': float(best_result.chi2_ndf_from_one),
        'success_score': float(best_result.success_score),
        'parameters': param_dict,
        'fit_quality': {
            'is_good_fit': 'true' if best_result.chi2_ndf_from_one < 0.5 else 'false',
            'is_acceptable_fit': 'true' if best_result.chi2_ndf_from_one < 1.0 else 'false',
            'fit_status': 'excellent' if best_result.chi2_ndf_from_one < 0.2 
                         else 'good' if best_result.chi2_ndf_from_one < 0.5
                         else 'acceptable' if best_result.chi2_ndf_from_one < 1.0
                         else 'poor'
        }
    }
    
    return fit_info

def save_fit_parameters_json(all_results: Dict[str, Dict], output_filename: str):
    """Save all fit parameters to a JSON file"""
    
    # Create comprehensive output dictionary
    output_data = {
        'metadata': {
            'analysis_type': 'bethe_bloch_fitting',
            'fitting_method': 'hybrid_root_scipy',
            'timestamp': datetime.datetime.now().isoformat(),
            'description': 'Best fit parameters from hybrid Bethe-Bloch fitting analysis'
        },
        'summary': {
            'total_particles': len(all_results),
            'particles_analyzed': list(all_results.keys())
        },
        'fit_results': {}
    }
    
    # Add results for each particle
    for particle, result_data in all_results.items():
        best_result = result_data['best_result']
        best_func_name = result_data['best_func']
        
        fit_info = create_fit_parameters_dict(particle, best_result, best_func_name)
        output_data['fit_results'][particle] = fit_info
    
    # Add global statistics
    chi2_values = [result_data['best_result'].chi2_ndf 
                   for result_data in all_results.values()]
    
    output_data['global_statistics'] = {
        'mean_chi2_ndf': float(sum(chi2_values) / len(chi2_values)),
        'min_chi2_ndf': float(min(chi2_values)),
        'max_chi2_ndf': float(max(chi2_values)),
        'function_usage': {},
        'optimizer_usage': {}
    }
    
    # Count function and optimizer usage
    for result_data in all_results.values():
        func_name = result_data['best_func']
        optimizer = result_data['best_result'].optimizer
        
        if func_name not in output_data['global_statistics']['function_usage']:
            output_data['global_statistics']['function_usage'][func_name] = 0
        output_data['global_statistics']['function_usage'][func_name] += 1
        
        if optimizer not in output_data['global_statistics']['optimizer_usage']:
            output_data['global_statistics']['optimizer_usage'][optimizer] = 0
        output_data['global_statistics']['optimizer_usage'][optimizer] += 1
    
    # Save to JSON file
    try:
        with open(output_filename, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nFit parameters saved to JSON: {output_filename}")
        print("JSON file contains:")
        print(f"  - Metadata and analysis information")
        print(f"  - Best fit parameters for {len(all_results)} particles")
        print(f"  - Parameter values with uncertainties")
        print(f"  - Fit quality metrics")
        print(f"  - Global statistics")
        
        return True
        
    except Exception as e:
        print(f"Error saving JSON file: {e}")
        return False

def load_fit_parameters_json(filename: str) -> Dict[str, Any]:
    """Load fit parameters from JSON file"""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading JSON file {filename}: {e}")
        return {}

def print_json_summary(json_data: Dict[str, Any]):
    """Print summary of JSON data"""
    if not json_data:
        return
    
    metadata = json_data.get('metadata', {})
    summary = json_data.get('summary', {})
    global_stats = json_data.get('global_statistics', {})
    
    print(f"\nJSON File Summary:")
    print(f"  Analysis: {metadata.get('analysis_type', 'Unknown')}")
    print(f"  Timestamp: {metadata.get('timestamp', 'Unknown')}")
    print(f"  Particles: {summary.get('total_particles', 0)}")
    print(f"  Mean χ²/ndf: {global_stats.get('mean_chi2_ndf', 0.0):.4f}")
    
    func_usage = global_stats.get('function_usage', {})
    opt_usage = global_stats.get('optimizer_usage', {})
    
    if func_usage:
        print(f"  Function usage: {dict(func_usage)}")
    if opt_usage:
        print(f"  Optimizer usage: {dict(opt_usage)}")