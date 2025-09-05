from ROOT import TPaveText, TLegend
from typing import List

def get_alice_watermark(x_min: float, y_min: float, x_max: float, y_max: float, additional_texts: List[str] = None) -> TPaveText:

    watermark = TPaveText(x_min, y_min, x_max, y_max, 'NDC')
    watermark.SetBorderSize(0)
    watermark.SetFillColor(0)
    watermark.SetTextAlign(12)
    watermark.SetTextSize(0.04)
    
    watermark.AddText('#bf{ALICE Performance}')
    watermark.AddText('#bf{Run 3}')
    watermark.AddText('#bf{pp #sqrt{#it{s}} = 13.6 TeV}')

    if additional_texts:
        for text in additional_texts:
            watermark.AddText(text)

    return watermark

def init_legend(xmin, ymin, xmax, ymax):
    legend = TLegend(xmin, ymin, xmax, ymax)
    legend.SetNColumns(2)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    return legend

