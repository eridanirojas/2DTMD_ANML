#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import peak_widths
from scipy.interpolate import interp1d
from scipy.integrate import simpson
import pandas as pd


# In[ ]:


def index_to_xdata(xdata, indices):
    "interpolate the values from scipy.signal.peak_widths to proper raman_shift data, not for your use"
    ind = np.arange(len(xdata))
    f = interp1d(ind,xdata)
    return f(indices)


# In[1]:


def ratiolabeler(labels, desired_ratio, peak_intensity_ratio = False, peak_integral_ratio = False):
    """
    Labels calculated ratio. 

    Inputs:
    - labels: Array containing user inputted strings. 
    - desired_ratio: Array containing user inputted integers ranging from 0-2.
    - peak_intensity_ratio: Calculated value from intensity_ratio().
    - peak_integral_ratio: Calculated value from integral_ratio().

    Returns:
    - ratio_label: Label of ratio found from labels and desired ratio.
    - df: pandas dataframe neatly displaying desired ratio values
    """
    if len(labels)!= len(desired_ratio):
        raise ValueError(f"labels has {len(labels)} indices while desired_ratio has {len(desired_ratio)} indices.")
    top = None
    bottom = None
    for i, j in enumerate(desired_ratio):
        if j == 2:
            top = labels[i]
        elif j == 1:
            bottom = labels[i]
        else:
            pass
    ratio_label = f"{top}/{bottom}"
    if peak_intensity_ratio:
        if peak_integral_ratio:
            df = pd.DataFrame({"Intensity": peak_intensity_ratio,"Integral": peak_integral_ratio}, index = [f"{ratio_label}"])
    else:
        df = None
    return ratio_label, df


# In[2]:


def intensity_ratio(desired_ratio, peaks):
    """
    Calculates desired peak intensity ratio. 

    Inputs:
    - desired_ratio: Array containing user inputted integers ranging from 0-2.
    - peaks: Dictionary containing peak indices, values, peak ratio, raman shift peak locations, width values per peak, width heights, 
             and raman shift locations for widths

    Returns:
    - peak_intensity_ratio: Intensity ratio between selected peaks.
    """
    peak_intensity = peaks['peak_intensity']
    large = None
    small = None
    for i, j in enumerate(desired_ratio):
        if j == 2:
            large = peak_intensity[i]
        elif j == 1:
            small = peak_intensity[i]
        else:
            pass
    peak_intensity_ratio = large/small
    return peak_intensity_ratio


# In[ ]:


def integral_ratio(desired_ratio, integrals):
    """
    Calculates desired peak integral ratio. 

    Inputs:
    - desired_ratio: Array containing user inputted integers ranging from 0-2.
    - integrals: Array containing peaks' corresponding integral values.

    Returns:
    - peak_integral_ratio: Ratio between selected peak integrals.
    """
    if len(desired_ratio) != len(integrals):
        raise ValueError(f"There are {len(integrals)} integrals and you put in {len(desired_ratio)} indices in desired_ratio")
    large = None
    small = None
    for i, j in enumerate(desired_ratio):
        if j == 2:
            large = integrals[i]
        elif j == 1:
            small = integrals[i]
        else:
            pass
    peak_integral_ratio = large/small
    return peak_integral_ratio


# In[ ]:


def peak_distances(peaks, labels):
    """
    Calculates distances between all possible peak pairs found via identify_peaks() function.

    Inputs:
    - peaks: Dictionary containing peak indices, values, peak ratio, raman shift peak locations, width values per peak, width heights, 
             and raman shift locations for widths
    - intensity: Array containing corresponding intensity values.
    - threshold: Minimum intensity required to be considered as a peak.
    - distance: Minimum distance between peaks (in terms of array indices).

    Returns:
    - distance_values: Array containing all possible peak-peak distances.
    - distance_labels: Array containing respective names for all peak-peak distances.
    - df: pandas dataframe neatly displaying all peak-peak distance information.
    """
    distances = []
    peak_raman_shifts = peaks['peak_raman_shifts']
    for i in range(len(peak_raman_shifts)):
        for j in range(i + 1, len(peak_raman_shifts)):
            distance = abs(peak_raman_shifts[i]-peak_raman_shifts[j])
            distances.append((labels[i], labels[j], distance))
    distance_values = [distance[2] for distance in distances]
    distance_labels = [f"{distance[0]}-{distance[1]}" for distance in distances]
    df = pd.DataFrame([distance_values], columns=distance_labels)
    df.index = ['Peak distances']
    return distance_values, distance_labels, df


# In[1]:


def peak_integrals(peaks, raman_shift, intensity):
    """
    Calculates area underneath located peaks via simpson method, left to right.

    Inputs:
    - peaks: Dictionary containing peak indices, values, peak ratio, raman shift peak locations, width values per peak, width heights, 
             and raman shift locations for widths
    - raman_shift: Array containing wavelength data. 
    - intensity: Array containing corresponding intensity values.

    Returns:
    - integrals: Array containing all peak integral values.
    """
    left_ips = peaks['left_ips']
    right_ips = peaks['right_ips']
    integrals = []
    additives = peaks['peak_widths']
    for left, right, additive in zip(left_ips, right_ips, additives):
        lower_bound = left - additive
        upper_bound = right + additive
        
        if lower_bound > upper_bound:
            raise ValueError(f"Invalid range: lower_bound={lower_bound}, upper_bound={upper_bound}")
        
        mask = (raman_shift >= lower_bound) & (raman_shift <= upper_bound)
        x_seg = raman_shift[mask]
        y_seg = intensity[mask]
    
        if x_seg.size == 0 or y_seg.size == 0:
            raise ValueError(
                f"No data points found for the range: left={left}, right={right}, "
                f"additive={additive}, computed range=({lower_bound}, {upper_bound})"
            )
    
        if x_seg.shape != y_seg.shape:
            raise ValueError(f"Shape mismatch: x_seg={x_seg.shape}, y_seg={y_seg.shape}")
            
        integral_simps = simpson(y=y_seg, x=x_seg)
        integrals.append(integral_simps)
    return integrals


# In[1]:


def identify_peaks(raman_shift, intensity, height, spacing):
    """
    Identifies peaks in the Raman spectrum using scipy's find_peaks function.
    Identifies peak widths in the Raman spectrum using scipy's peak_widths function. 

    Inputs:
    - raman_shift: Array containing Raman shift values.
    - intensity: Array containing corresponding intensity values.
    - threshold: Minimum intensity required to be considered as a peak.
    - distance: Minimum distance between peaks (in terms of array indices).

    Returns:
    - peaks: Dictionary containing peak indices, values, raman shift peak locations, width values per peak, width heights,
             and raman shift locations for widths. fixed?
    """
    peaks, _ = find_peaks(intensity, height=height, distance=spacing) 
    widths, width_heights, left_ips, right_ips = peak_widths(intensity, peaks)
    fwhm = []
    left_ips = index_to_xdata(raman_shift, left_ips) 
    right_ips = index_to_xdata(raman_shift, right_ips) 
    for i,j in zip(left_ips, right_ips):
        fixed_width = j - i
        fwhm.append(fixed_width)
    peak_intensity = intensity[peaks]
    peak_raman_shifts = raman_shift[peaks]
    plt.scatter(peak_raman_shifts, peak_intensity, color='blue', marker='o')
    plt.plot(raman_shift, intensity)
    plt.title("located peaks")
    plt.show()
    return {'peak_indices': peaks, 'peak_intensity': peak_intensity, 
            'peak_raman_shifts': peak_raman_shifts, 'peak_widths': fwhm, 'width_heights': width_heights, 'left_ips': left_ips,
            'right_ips': right_ips}

