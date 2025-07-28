import numpy as np
from sklearn.cluster import KMeans
from ROOT import RooRealVar, RooDataHist, TH1F, TGraphErrors

def initialize_means_and_sigmas(hist: TH1F, n_components: int):
    '''
        Initialize means and sigmas using KMeans clustering.
        They are ordered from the lowest mean value to the highest.
        hist: histogram to be fitted
        n_components: number of components to fit
    '''

    data_points = []
    for ibin in range(1, hist.GetNbinsX()+1):
        data_points.extend([hist.GetBinCenter(ibin)] * int(hist.GetBinContent(ibin)))

    data_points = np.array(data_points)

    if len(data_points) <= 0:
        print('No data points to fit')
        return

    kmeans = KMeans(n_clusters=n_components, init='k-means++', n_init='auto').fit(data_points.reshape(-1, 1))
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    covariances = []
    for icomp in range(n_components):
        comp_data = data_points[np.where(np.array(labels)==icomp)[0]]
        covariances.append(np.cov(comp_data.T))

    # Sort centers and get the sorted indices
    sorted_indices = np.argsort(centers.flatten())
    #sorted_indices = sorted_indices[::-1]

    # Reorder centers, covariances, and weights based on the sorted indices
    centers = centers[sorted_indices]
    covariances = [covariances[i] for i in sorted_indices]
    return centers, covariances

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

    datahist = RooDataHist(f'dh_tof_{pt_low_edge}_{pt_high_edge}', f'dh_{pt_low_edge}_{pt_high_edge}', [x], Import=hist)
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
        'mean_err': signal_pars['mean'].getError(),
        'sigma': signal_pars['sigma'].getVal(),
        'sigma_err': signal_pars['sigma'].getError(),
        'resolution': resolution,
        'resolution_err': resolution_error,
    }

    return frame, fit_results