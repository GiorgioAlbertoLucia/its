#!/usr/bin/env python3
"""
Visualization utilities for Bethe-Bloch fitting results
"""

from typing import Dict, List, Tuple
from ROOT import TCanvas, TPaveText, TF1, TGraphErrors
from ROOT import kGreen, kRed, kBlue, kBlack

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'core', 'fit')))
from fit_utils import FitResult, CanvasSpace

def create_info_box(result: FitResult, particle, func_name, is_best=False):
    """Create info box for fit results"""
    info = TPaveText(0.55, 0.35, 0.85, 0.85, "NDC")
    info.SetFillColor(kGreen-2 if is_best else 0)
    info.SetBorderSize(1)
    
    info.AddText(f"Particle: {particle}")
    info.AddText(f"Function: {func_name}")
    info.AddText(f"Optimizer: {result.optimizer}")
    info.AddText(f"#chi^{{2}}/ndf = {result.chi2_ndf:.4f}")
    info.AddText(f"Strategy: {result.strategy}")
    info.AddText("")
    
    # Add parameters
    func = result.function
    for i in range(len(result.params)):
        if i < func.GetNpar():
            par_name = func.GetParName(i)
        else:
            par_name = f"Par{i}"
        par_val = result.params[i]
        par_err = result.errors[i]
        info.AddText(f"{par_name} = {par_val:.4f} #pm {par_err:.4f}")
    
    return info

def create_comparison_canvas(particle, results: Dict[str, List[FitResult]]) -> CanvasSpace:
    """Create canvas showing comparison of different fits"""
    # Flatten results for display
    all_results = []
    for func_name, func_results in results.items():
        for result in func_results:
            all_results.append((func_name, result))
    
    # Sort by success score
    all_results.sort(key=lambda x: x[1].success_score)
    
    n_results = min(4, len(all_results))  # Show max 4 results
    canvas = TCanvas(f'c_comparison_{particle}', f'Fit Comparison - {particle}', 1600, 800)
    canvas.Divide(2, 2)
    space = CanvasSpace(canvas, [])
    
    colors = [1, 4, 6, 8, 9, 2, 3, 7]
    
    for i, (func_name, fit_result) in enumerate(all_results[:n_results]):
        canvas.cd(i + 1)
        
        graph = fit_result.graph
        graph.SetTitle(f"{particle} - {func_name};#beta#gamma;#LT Cluster Size #GT")
        graph.SetMarkerStyle(20)
        graph.SetMarkerColor(1)
        graph.GetListOfFunctions().Clear()
        graph.Draw("AP")
        space.content.append(graph)
        
        func = fit_result.function
        func.SetLineColor(colors[i % len(colors)])
        func.SetLineWidth(2)
        func.Draw("SAME")
        space.content.append(func)
        
        info = TPaveText(0.6, 0.7, 0.9, 0.9, "NDC")
        info.SetFillColor(0)
        info.SetBorderSize(1)
        info.AddText(f"{func_name}")
        info.AddText(f"#chi^{{2}}/ndf = {fit_result.chi2_ndf:.3f}")
        info.AddText(f"{fit_result.optimizer}-{fit_result.strategy}")
        info.Draw()
        space.content.append(info)
    
    return space

def create_final_canvas(graph, particle, best_func_name, best_result: FitResult) -> CanvasSpace:
    """Create final canvas with best fit"""
    canvas = TCanvas(f'c_final_{particle}', f'Best Fit - {particle}', 800, 600)
    space = CanvasSpace(canvas, [])
    
    graph.Draw("AP")
    graph.SetTitle(f"{particle} - Best Fit ({best_func_name});#beta#gamma;#LT Cluster Size #GT")
    graph.SetMarkerStyle(20)
    graph.SetMarkerColor(1)
    space.content.append(graph)
    
    best_function = best_result.function
    best_function.SetLineColor(2)
    best_function.SetLineWidth(3)
    best_function.Draw("SAME")
    space.content.append(best_function)
    
    info = create_info_box(best_result, particle, best_func_name, is_best=True)
    info.Draw()
    space.content.append(info)
    
    return space

def print_comparison_results(particle, results: Dict[str, List[FitResult]], best_func_name, best_result):
    """Print comparison results with optimizer info"""
    print(f"\n  COMPARISON RESULTS for {particle}:")
    print(f"  " + "-"*75)
    
    # Flatten all results for sorting
    all_results = []
    for func_name, func_results in results.items():
        for result in func_results:
            all_results.append((func_name, result))
    
    # Sort by success score
    all_results.sort(key=lambda x: x[1].success_score)
    
    for func_name, fit_result in all_results:
        is_best = (fit_result is best_result)
        marker = "★" if is_best else " "
        opt_str = fit_result.optimizer
        strategy_str = fit_result.strategy
        print(f"  {marker} {func_name:15s}: χ²/ndf = {fit_result.chi2_ndf:6.3f} ({opt_str}-{strategy_str}) Score: {fit_result.success_score:.3f}")
    
    print(f"\n  → Best function for {particle}: {best_func_name} ({best_result.optimizer}-{best_result.strategy})")

def print_final_summary(all_results):
    """Print final summary to console"""
    print(f"\n" + "="*80)
    print("OVERALL SUMMARY - HYBRID FITTING")
    print("="*80)
    
    for particle, fit_results in all_results.items():
        best_func = fit_results['best_func']
        best_result = fit_results['best_result']
        best_chi2 = best_result.chi2_ndf
        optimizer = best_result.optimizer
        strategy = best_result.strategy
        score = best_result.success_score
        print(f"{particle}: {best_func:15s} (χ²/ndf={best_chi2:.4f}, {optimizer}-{strategy}, Score={score:.3f})")