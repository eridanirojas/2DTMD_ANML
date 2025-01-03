#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np
from lmfit import Parameters, Minimizer
from lmfit.models import ConstantModel, GaussianModel, VoigtModel, LorentzianModel
import sklearn.metrics as slms
import matplotlib.pyplot as plt


# In[18]:


def raman_gfitter(x, y, peaks):
    """
    Fits input raman spectra data via gaussian method.

    Inputs:
    - x: Array containing Raman shift values.
    - y: Array containing respective intensity values.
    - peaks: Dictionary containing peak indices, values, peak ratio, raman shift peak locations, width values per peak, width heights, 
             and raman shift locations for widths

    Returns:
    - fit_intensity: Array containing newly fitted intensity values.
    - r2: Float number that is the R^2 score of the fit.
    
    """
    raman_shift = np.array(x)
    intensity = np.array(y)
    rs_guesses = peaks["peak_raman_shifts"]
    int_guesses = peaks["peak_intensity"]
    fwhm = peaks['peak_widths']
    min = raman_shift.min()
    max = raman_shift.max()
    model = ConstantModel()
    params = model.make_params()
    params['c'].set(intensity[0], min=0, max=intensity[-1])
    
    for i, (rs, int, fwhm) in enumerate(zip(rs_guesses, int_guesses, fwhm)):
        for j, offset in enumerate([-1, 0, 1]):
            gaussian = GaussianModel(prefix=f'g_{i+1}_{j}_')
            p = gaussian.make_params()
            center = rs + (offset * (fwhm/2))
            p[f'g_{i+1}_{j}_center'].set(center, min=min, max=max)
            p[f'g_{i+1}_{j}_amplitude'].set(int * (1 if offset == 0 else 0.5))
            if offset == 0:
                p[f'g_{i+1}_{j}_fwhm'].set(fwhm)
            params.update(p)
            model += gaussian
    init = model.eval(params=params, x=raman_shift)
    plt.plot(raman_shift, init)
    plt.title("initial guess")
    plt.ylim(0, np.max(init)+(np.max(init)/4))
    plt.xlim(np.min(raman_shift), np.max(raman_shift))
    plt.show()
    plt.clf()
    result = model.fit(data=intensity, params=params, x=raman_shift)
    fit_intensity = result.best_fit
    plt.plot(raman_shift, intensity, label = "raw")
    plt.plot(raman_shift, fit_intensity, label = "fit")
    plt.title("gaussian fit")
    plt.ylim(0, np.max(intensity)+(np.max(intensity)/4))
    plt.xlim(np.min(raman_shift), np.max(raman_shift))
    plt.legend()
    plt.show()
    r2 = slms.r2_score(intensity, fit_intensity)
    print(f'R^2 score: {r2}')
    return fit_intensity, r2


# In[19]:


def raman_lfitter(x, y, peaks):
    """
    Fits input raman spectra data via lorentzian method.

    Inputs:
    - x: Array containing Raman shift values.
    - y: Array containing respective intensity values.
    - peaks: Dictionary containing peak indices, values, peak ratio, raman shift peak locations, width values per peak, width heights, 
             and raman shift locations for widths

    Returns:
    - fit_intensity: Array containing newly fitted intensity values.
    - r2: Float number that is the R^2 score of the fit.
    
    """
    raman_shift = np.array(x)
    intensity = np.array(y)
    rs_guesses = peaks["peak_raman_shifts"]
    int_guesses = peaks["peak_intensity"]
    fwhm = peaks["peak_widths"]
    min = raman_shift.min()
    max = raman_shift.max()
    model = ConstantModel()
    params = model.make_params()
    params['c'].set(intensity[0], min=0, max=intensity[-1])

    for i, (rs, int, fwhm) in enumerate(zip(rs_guesses, int_guesses, fwhm)):
        for j, offset in enumerate([-1, 0, 1]):
            lorentzian = LorentzianModel(prefix=f'l_{i+1}_{j}_')
            p = lorentzian.make_params()
            center = rs + (offset * (fwhm/2))
            p[f'l_{i+1}_{j}_center'].set(center, min=min, max=max)
            p[f'l_{i+1}_{j}_amplitude'].set(int * (1 if offset == 0 else 0.5))
            if offset == 0:
                p[f'l_{i+1}_{j}_fwhm'].set(fwhm)
            params.update(p)
            model += lorentzian
    init = model.eval(params=params, x=raman_shift)
    plt.plot(raman_shift, init)
    plt.title("initial guess")
    plt.ylim(0, np.max(init)+(np.max(init)/4))
    plt.xlim(np.min(raman_shift), np.max(raman_shift))
    plt.show()
    plt.clf()
    result = model.fit(data=intensity, params=params, x=raman_shift)
    fit_intensity = result.best_fit
    plt.plot(raman_shift, intensity, label = "raw")
    plt.plot(raman_shift, fit_intensity, label = "fit")
    plt.title("lorentzian fit")
    plt.ylim(0, np.max(intensity)+(np.max(intensity)/4))
    plt.xlim(np.min(raman_shift), np.max(raman_shift))
    plt.legend()
    plt.show()
    r2 = slms.r2_score(intensity, fit_intensity)
    print(f'R^2 score: {r2}')
    return fit_intensity, r2


# In[20]:


def raman_vfitter(x, y, peaks):
    """
    Fits input raman spectra data via voigt method.

    Inputs:
    - x: Array containing Raman shift values.
    - y: Array containing respective intensity values.
    - peaks: Dictionary containing peak indices, values, peak ratio, raman shift peak locations, width values per peak, width heights, 
             and raman shift locations for widths

    Returns:
    - fit_intensity: Array containing newly fitted intensity values.
    - r2: Float number that is the R^2 score of the fit.
    
    """
    raman_shift = np.array(x)
    intensity = np.array(y)
    rs_guesses = peaks["peak_raman_shifts"]
    int_guesses = peaks["peak_intensity"]
    fwhm = peaks["peak_widths"]
    min = raman_shift.min()
    max = raman_shift.max()
    model = ConstantModel()
    params = model.make_params()
    params['c'].set(intensity[0], min=0, max=intensity[-1])
    
    for i, (rs, int, fwhm) in enumerate(zip(rs_guesses, int_guesses, fwhm)):
        for j, offset in enumerate([-1, 0, 1]):
            voigt = VoigtModel(prefix=f'v_{i+1}_{j}_')
            p = voigt.make_params()
            center = rs + (offset * (fwhm/2))
            p[f'v_{i+1}_{j}_center'].set(center, min=min, max=max)
            p[f'v_{i+1}_{j}_amplitude'].set(int * (1 if offset == 0 else 0.5))
            if offset == 0:
                p[f'v_{i+1}_{j}_fwhm'].set(fwhm)
            params.update(p)
            model += voigt
    init = model.eval(params=params, x=raman_shift)
    plt.plot(raman_shift, init)
    plt.title("initial guess")
    plt.ylim(0, np.max(init)+(np.max(init)/4))
    plt.xlim(np.min(raman_shift), np.max(raman_shift))
    plt.show()
    plt.clf()
    result = model.fit(data=intensity, params=params, x=raman_shift)
    fit_intensity = result.best_fit
    plt.plot(raman_shift, intensity, label = "raw")
    plt.plot(raman_shift, fit_intensity, label = "fit")
    plt.title("voigt fit")
    plt.ylim(0, np.max(intensity)+(np.max(intensity)/4))
    plt.xlim(np.min(raman_shift), np.max(raman_shift))
    plt.legend()
    plt.show()
    r2 = slms.r2_score(intensity, fit_intensity)
    print(f'R^2 score: {r2}')
    return fit_intensity, r2
    
def deconvolute(x, y, peaks, selection):
    """
    Deconvolutes selected peaks.

    Inputs:
    - x: Array containing x data.
    - y: Array containing y data.
    - peaks: Dictionary containing peak information, from peaks module.
    - selection: Array selecting peaks to deconvolute.
    
    """
    selection = [i - 1 for i in selection]
    for i in selection:
        if i < 0 or i >= len(peaks["peak_raman_shifts"]):
            raise ValueError(f"Selection index {i + 1} is out of range.")
        if not peaks["peak_raman_shifts"][i]:
            raise ValueError(f"No valid peak at selection index {i + 1}.")
    x = np.array(x)
    y = np.array(y)
    cen_guesses = [peaks["peak_raman_shifts"][i] for i in selection]
    amp_guesses = [peaks["peak_intensity"][i] for i in selection]
    fwhm_guesses = [peaks['peak_widths'][i] for i in selection]
    x_min = x.min()
    x_max = x.max()
    mod = ConstantModel()
    params = mod.make_params()
    params['c'].set(value=y[0], min=0, max=y.max())
    for i, (guess_x, guess_y, fwhm) in enumerate(zip(cen_guesses, amp_guesses, fwhm_guesses)):
        gaussian = GaussianModel(prefix=f'g{i+1}_')
        p = gaussian.make_params()
        p[f'g{i+1}_center'].set(guess_x, min=x_min, max=x_max)
        p[f'g{i+1}_amplitude'].set(guess_y)
        p[f'g{i+1}_fwhm'].set(fwhm)
        params.update(p)
        mod += gaussian
    display(params)
    init = mod.eval(params, x=x)
    out = mod.fit(y, params, x=x)
    comps = out.eval_components(x=x)
    plt.plot(x, y)
    for i in selection:
        plt.plot(x, comps[f'g{i+1}_'], '--', label=f'Gaussian component {i+1}')
    plt.legend()
    plt.show()

