'''
    Script devoted to the visualisation of the average cluster size per species
'''

import numpy as np
from scipy.special import erf

from ROOT import TFile, TCanvas, TF1, TLegend, TPaveText, TGraphErrors, TLatex
from ROOT import kOrange, kBlue, kGreen, kRed, kMagenta, kYellow, kBlack, kAzure,kViolet

import sys
sys.path.append('..')
from utils.utils import set_root_object


FIT_PARAMS = {
    'Z=1': {
        'mean': [0.9883, 1.894, 1.950],
        'mean_errors': [0.0077, 0.0256, 0.0054],
        'resolution': [0.187, -1.015, 1.834]
    },
    'Z=2': {
        'mean': [2.172, 1.872, 4.699],
        'mean_errors': [0.0202, 0.0168, 0.0179],
        'resolution': [0.1466, -0.0246, 0],
    }
}

def get_alice_watermark(x_min, y_min, x_max, y_max) -> TPaveText:

    watermark = TPaveText(x_min, y_min, x_max, y_max, 'NDC')
    watermark.SetBorderSize(0)
    watermark.SetFillColor(0)
    watermark.SetTextAlign(12)
    watermark.SetTextSize(0.04)
    watermark.AddText('#bf{ALICE Performance}')
    watermark.AddText('#bf{Run 3}')
    watermark.AddText('#bf{pp #sqrt{#it{s}} = 13.6 TeV}')
    #watermark.AddText('#bf{LHC24 pass1}')

    return watermark

def init_legend(x_min, y_min, x_max, y_max) -> TLegend:
    legend = TLegend(x_min, y_min, x_max, y_max)
    legend.SetBorderSize(0)
    legend.SetTextSize(0.04)
    legend.SetFillColor(0)
    legend.SetNColumns(1)
    return legend

def create_graph_from_function(func: TF1, func_err:TF1, x_values: list) -> TGraphErrors:
    
    y_values = [func.Eval(x) for x in x_values]
    
    def error_propagation(func: TF1, x: float) -> float:
        """Calculate the propagated error for a function at a given x value."""

        ## Numerical derivative approximation
        #delta = 1e-5
        #derivative = (func.Eval(x + delta) - func.Eval(x - delta)) / (2 * delta)
        #
        ## Propagate parameter errors
        #error_squared = 0
        #for i in range(func.GetNpar()):
        #    param_error = func.GetParError(i)
        #    error_squared += (derivative * param_error) ** 2
        #
        #return error_squared ** 0.5

        propagated_error_0 = func.GetParError(0) / (x**func.GetParameter(1))
        propagated_error_1 = func.GetParError(1) * func.GetParameter(0) * np.log(x) / (x**func.GetParameter(1))
        propagated_error_2 = func.GetParError(2)

        total_error = np.sqrt(propagated_error_0**2 + propagated_error_1**2 + propagated_error_2**2)
        return total_error

    #y_errors = [error_propagation(func, x) for x in x_values]
    y_resolutions = [func_err.Eval(x) for x in x_values]
    y_errors = [res * y for res, y in zip(y_resolutions, y_values)]

    graph = TGraphErrors(len(x_values))
    for i, (x, y, y_err) in enumerate(zip(x_values, y_values, y_errors)):
        graph.SetPoint(i, x, y)
        graph.SetPointError(i, 0, y_err)

    return graph

def draw_plot(outfile: TFile):

    canvas = TCanvas('cMeanClusterSize', '', 800, 600)
    canvas.SetLeftMargin(0.11)
    canvas.SetRightMargin(0.05)
    canvas.SetBottomMargin(0.15)
    
    hframe = canvas.DrawFrame(0, 1, 5.5, 9.5, ';#beta#gamma;#LT ITS cluster size#kern[0.2]{{#GT}}')
    betagamma_values = np.linspace(0.1, 5.5, 140)
    #betagamma_values = np.linspace(1, 5.5, 90)

    z1_curve = TF1('z1_curve', '[2] + [0]/(x^[1])', 0, 6)
    z1_resolution = TF1('z1_resolution', '[0] * ROOT::Math::erf((x - [1])/[2])', 0, 6)
    for param, (value, error, res) in enumerate(zip(FIT_PARAMS['Z=1']['mean'], FIT_PARAMS['Z=1']['mean_errors'], FIT_PARAMS['Z=1']['resolution'])):
        z1_curve.SetParameter(param, value)
        z1_curve.SetParError(param, error)
        z1_resolution.SetParameter(param, res)
    set_root_object(z1_curve, line_color=kBlue, line_style=1, line_width=2)
    graph1 = create_graph_from_function(z1_curve, z1_resolution, betagamma_values)
    set_root_object(graph1, marker_style=20, marker_size=0, marker_color=kBlue, fill_color=kBlue, fill_style=3004)

    z2_curve = TF1('z2_curve', '[2] + [0]/(x^[1])', 0, 6)
    z2_resolution = TF1('z2_resolution', '[0] + [1]*x + [2]*x*x', 0, 6)
    for param, (value, error, res) in enumerate(zip(FIT_PARAMS['Z=2']['mean'], FIT_PARAMS['Z=2']['mean_errors'], FIT_PARAMS['Z=2']['resolution'])):
        z2_curve.SetParameter(param, value)
        z2_curve.SetParError(param, error)
        z2_resolution.SetParameter(param, res)
    set_root_object(z2_curve, line_color=kRed, line_style=1, line_width=2)
    graph2 = create_graph_from_function(z2_curve, z2_resolution, betagamma_values)
    set_root_object(graph2, marker_style=21, marker_size=0, marker_color=kRed, fill_color=kRed, fill_style=3004)

    text = TLatex(0.625, 0.64, '1#sigma band contour')
    text.SetNDC()
    text.SetTextSize(0.04)
    text.SetTextFont(42)
    text.Draw()
    
    graph1.Draw('3 SAME')
    graph2.Draw('3 SAME')
    z1_curve.Draw('SAME')
    z2_curve.Draw('SAME')

    legend = init_legend(0.6, 0.7, 0.85, 0.85)
    legend.AddEntry(z1_curve, 'Z=1', 'lf')
    legend.AddEntry(z2_curve, 'Z=2 (#Omega#Omega)', 'lf')
    legend.Draw()

    outfile.cd()
    canvas.Write()
    
if __name__ == '__main__':
    
    output_path = 'output/omega_cluster_size.root'
    output_file = TFile(output_path, 'RECREATE')

    draw_plot(output_file)
