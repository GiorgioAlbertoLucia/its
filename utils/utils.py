import numpy as np
from ROOT import RooRealVar, RooDataHist, TH1F, TGraphErrors

def calibration_fit_slice(model, hist: TH1F, x: RooRealVar, signal_pars, pt_low_edge, pt_high_edge):
    '''
        Fit a slice of the TOF mass histogram. Return the frame and the fit results

        Parameters
        ----------
        model (RooAbsPdf): model to be fitted
        hist (TH1F): histogram to be fitted
        x (RooRealVar): variable to be fitted
        signal_pars (dict): dictionary with the signal parameters
        pt_low_edge (float): lower edge of the pT bin
        pt_high_edge (float): higher edge of the pT bin

        Returns
        -------
        frame (RooPlot): frame with the fit results
        fit_results (dict): dictionary with the fit results
            - mean (float): mean value
            - mean_err (float): mean error
            - sigma (float): sigma value
            - sigma_err (float): sigma error
            - resolution (float): resolution value
            - resolution_err (float): resolution error
    '''

    datahist = RooDataHist(f'dh_{pt_low_edge:.2f}_{pt_high_edge:.2f}', f'dh_{pt_low_edge:.2f}_{pt_high_edge:.2f}', [x], Import=hist)
    model.fitTo(datahist, PrintLevel=-1)

    frame = x.frame(Title=f'{pt_low_edge:.2f} < p_{{T}} < {pt_high_edge:.2f} GeV/#it{{c}}')
    datahist.plotOn(frame)
    model.plotOn(frame, LineColor=2)
    model.paramOn(frame)
    for icomp, component in enumerate(model.getComponents(), start=3):
        component.plotOn(frame, LineColor=icomp, LineStyle='--')

    mean_err = signal_pars['sigma'].getVal() / np.sqrt(hist.Integral())
    resolution = signal_pars['sigma'].getVal() / signal_pars['mean'].getVal()
    resolution_error = resolution * np.sqrt((mean_err / signal_pars['mean'].getVal())**2 + (signal_pars['sigma'].getError() / signal_pars['sigma'].getVal())**2)
    fit_results = {
        'mean': signal_pars['mean'].getVal(),
        'mean_err': mean_err,
        'sigma': signal_pars['sigma'].getVal(),
        'sigma_err': signal_pars['sigma'].getError(),
        'resolution': resolution,
        'resolution_err': resolution_error,
    }

    return frame, fit_results

def create_graph(df, x: str, y: str, ex, ey, name:str='', title:str='') -> TGraphErrors:
        '''
            Create a TGraphErrors from the input DataFrame

            Parameters
            ----------
            x (str): x-axis variable
            y (str): y-axis variable
            ex (str): x-axis error
            ey (str): y-axis error
        '''

        # eliminate None values on x, y
        #df = df.filter(df[x].is_not_null())
        #df = df.filter(df[y].is_not_null())

        if len(df) == 0:
            return TGraphErrors()
        graph = TGraphErrors(len(df[x]))
        for irow, row in df.iterrows():
            graph.SetPoint(irow, row[x], row[y])
            xerr = row[ex] if ex != 0 else 0.
            yerr = row[ey] if ey != 0 else 0.
            graph.SetPointError(irow, xerr, yerr)
        
        graph.SetName(name)
        graph.SetTitle(title)

        return graph
