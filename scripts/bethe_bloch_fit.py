#!/usr/bin/env python3
"""
Main script for unified Bethe-Bloch fitting with both ROOT and scipy optimization methods
Uses consistent FitResult class and modular fitter interfaces
"""

import sys
import warnings
warnings.filterwarnings('ignore')

from ROOT import TFile, TF1, gStyle, gROOT, TIter
from ROOT import kGreen, kRed, kBlue, kBlack
from typing import Dict, List, Tuple

from torchic.physics import BetheBloch

sys.path.append('..')
from core.fit.fit_utils import FitResult, CanvasSpace
from core.fit.scipy_fitter import ScipyBetheFitter
from core.fit.root_fitter import RootBetheFitter
from core.fit.fit_visualisation import (create_comparison_canvas, create_final_canvas,
                         print_comparison_results, print_final_summary)
from core.fit.fit_json_output import save_fit_parameters_json, print_json_summary

# Suppress ROOT info messages
gROOT.SetBatch(False)
gStyle.SetOptFit(1111)

def create_fit_functions(name, x_min=0.1, x_max=5.0):
    """Create all fitting functions"""
    functions = {}
    
    # Bethe-Bloch function
    functions['bethe_bloch_aleph'] = TF1(f"{name}_bethe", 
        BetheBloch,
        x_min, x_max, 5)
    
    # Original function
    functions['3p_bethe_bloch'] = TF1(f"{name}_orig", '[0]/x^[1] + [2]', x_min, x_max)
    functions['3p_bethe_bloch'].SetParNames("Coeff", "Power", "Offset")
    
    return functions

def test_single_function(graph, func, particle, func_name) -> List[FitResult]:
    """Test function with both ROOT and scipy methods"""
    print(f"  Testing {func_name}:")
    
    results = []
    
    # Try scipy first
    scipy_fitter = ScipyBetheFitter()
    scipy_result = scipy_fitter.fit_function(graph, func, particle, func_name)
    if scipy_result:
        results.append(scipy_result)
    
    # Try ROOT method
    root_fitter = RootBetheFitter()
    # Create a new function for ROOT (to avoid parameter conflicts)
    x_min, x_max = get_graph_range(graph)
    root_functions = create_fit_functions(f"{particle}_root", x_min, x_max)
    root_func = root_functions[func_name]
    
    root_result = root_fitter.fit_function(graph, root_func, particle, func_name)
    if root_result:
        results.append(root_result)
    
    return results

def compare_functions(graph, particle, x_min, x_max):
    """Compare different functional forms using both fitters"""
    print(f"\n  Comparing fit functions for {particle} (Hybrid Method):")
    print(f"  " + "="*65)
    
    functions = create_fit_functions(particle, x_min, x_max)
    all_results = {}
    
    for func_name, func in functions.items():
        results = test_single_function(graph, func, particle, func_name)
        if results:
            # Store all results for this function type
            all_results[func_name] = results
    
    return all_results

def find_best_function(results: Dict[str, List[FitResult]]) -> Tuple[str, FitResult]:
    """Find the best function from all results"""
    if not results:
        return None, None
    
    best_result = None
    best_func_name = ""
    best_score = float('inf')
    
    for func_name, func_results in results.items():
        for result in func_results:
            if result.success_score < best_score:
                best_score = result.success_score
                best_result = result
                best_func_name = func_name
    
    return best_func_name, best_result

def save_particle_results(outfile, particle, results: Dict[str, List[FitResult]], best_func_name, 
                          comparison_space: CanvasSpace, final_space: CanvasSpace):
    """Save results for a single particle"""
    if not outfile:
        return
    
    outfile.cd()
    particle_dir = outfile.mkdir(f"{particle}_fits")
    particle_dir.cd()
    
    comparison_space.canvas.Write()
    final_space.canvas.Write()
    
    # Save all functions from best function type
    if best_func_name in results:
        for i, result in enumerate(results[best_func_name]):
            result.function.SetName(f"{best_func_name}_{result.optimizer}_{i}")
            result.function.Write()

def get_graph_range(graph):
    """Get x-range from graph with some padding"""
    x_values = [graph.GetX()[i] for i in range(graph.GetN())]
    x_min = min(x_values)
    x_max = max(x_values)
    return x_min, x_max

def process_particle(infile, particle, outfile=None):
    """Process a single particle type with both fitters"""
    print(f"\nProcessing particle: {particle}")
    print("="*50)
    
    # Get particle directory and graph
    particle_dir = infile.Get(particle)
    
    canvas = particle_dir.Get('c_mean')
    _next = TIter(canvas.GetListOfPrimitives())
    graph = None
    while (obj := _next()):
        if obj.GetName() != 'g_mean':
            continue
        graph = obj
    
    if not graph:
        print(f"  No graph found for {particle}")
        return None
    
    # Get fitting range
    x_min, x_max = get_graph_range(graph)
    print(f"  βγ range: {x_min:.3f} to {x_max:.3f}")
    
    # Compare functions using both fitters
    results = compare_functions(graph, particle, x_min, x_max)
    
    if not results:
        print(f"  No successful fits for {particle}")
        return None
    
    # Find best function across all fitters
    best_func_name, best_result = find_best_function(results)
    print_comparison_results(particle, results, best_func_name, best_result)
    
    # Create canvases
    comparison_space = create_comparison_canvas(particle, results)
    final_space = create_final_canvas(graph, particle, best_func_name, best_result)
    
    # Save results
    save_particle_results(outfile, particle, results, best_func_name, 
                         comparison_space, final_space)
    
    return {
        'particle': particle,
        'results': results,
        'best_func': best_func_name,
        'best_result': best_result,
    }

def main():
    """Main function with hybrid approach using both fitters"""
    
    input_file = '../output/data/LHC24_pass1_skimmed_calibration_beta_gamma_mean.root'
    output_file = '../output/data/fit_LHC24_pass1_skimmed_calibration_beta_gamma_mean.root'
    json_output_file = '../output/data/fit_parameters_LHC24_pass1_skimmed_calibration_beta_gamma_mean.json'
    
    print(f"Hybrid Bethe-Bloch Fitting Analysis")
    print(f"Using both ROOT and scipy optimization methods")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"JSON Output: {json_output_file}")
    print(f"{'='*80}")
    
    infile = TFile(input_file, "READ")
    outfile = TFile(output_file, "RECREATE")
    
    particles = ['Pi', 'Ka', 'Pr', 'De', 'He']
    all_results = {}
    
    for particle in particles:
        result = process_particle(infile, particle, outfile)
        if result:
            all_results[particle] = result
    
    print_final_summary(all_results)
    
    # Save fit parameters to JSON
    if all_results:
        success = save_fit_parameters_json(all_results, json_output_file)
        if success:
            # Load and print summary of saved JSON
            from core.fit.fit_json_output import load_fit_parameters_json
            json_data = load_fit_parameters_json(json_output_file)
            print_json_summary(json_data)
    
    print(f"\nResults saved to {output_file}")
    print(f"Processed {len(all_results)} particles successfully")
    
    infile.Close()
    outfile.Close()
    print("Analysis complete!")

if __name__ == "__main__":
    main()