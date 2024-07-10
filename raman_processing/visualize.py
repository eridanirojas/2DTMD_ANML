#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns
import processing as pro
import peaks as pks
import fitting as fit
import pandas as pd
import numpy as np


# In[2]:


def plot_spectrum(raman_shift, intensity, peaks=None, labels=None, mtl_name=None):
    """
    Plots the Raman spectrum, alongside peak information.

    Inputs:
    - raman_shift: Array containing Raman shift values.
    - intensity: Array containing Respective intensity values.
    - peaks: Dictionary containing peaks data, if it exists.
    - labels: User created list containing labels, left to right, of expected peaks, if it exists.
    - mtl_name: User created string labelling name of plotted material, if it exists.
    
    """
    plt.figure(figsize=(11, 6))
    plt.plot(raman_shift, intensity, color='red')
    if peaks:
        if len(peaks['peak_intensity']) != len(labels):
            raise ValueError(f"Length of labels list does not match amount of peaks present: {len(peaks)}")
        peak_values = peaks['peak_intensity']
        widths = peaks['peak_widths']
        width_heights = peaks['width_heights']
        left_ips = peaks['left_ips']
        right_ips = peaks['right_ips']
        peak_raman_shifts = peaks['peak_raman_shifts']
        additives = peaks['peak_widths']
        for i, (shift, value, width, label, left_ip, right_ip, additive) in enumerate(zip(peak_raman_shifts, peak_values, widths, labels,                                                                       left_ips, right_ips, additives)):
            plt.text(shift, value, label, ha='center', va='bottom', color='blue')
            plt.text(shift, value/3, f"W:{round(width, 3)}", ha='center', va='bottom', color='black')
            plt.plot([shift, shift], [0, value], color='black', linestyle='--', linewidth=1) 
            plt.text(shift, value/40, f"Sh:{round(shift, 3)}", ha='center', va='bottom', color='black')
            plt.fill_between(raman_shift,intensity, 0, where=(raman_shift>=left_ip-(additive*2))&(raman_shift<=right_ip+(additive*2)),color = 'blue', alpha = 0.25)
        plt.hlines(width_heights, left_ips, right_ips, color='g')
    plt.ylim(0, max(intensity)+(max(intensity)/4))
    plt.xlim(min(raman_shift), max(raman_shift))    
    plt.xlabel('Raman Shift (cm^-1)')
    plt.ylabel('Intensity (a.u.)')
    if mtl_name:
        plt.title(f"{mtl_name}")
    plt.show()


# In[1]:


def heat_map(df, yx, raman_shift, height, spacing, deg, desired_ratio, ratio_label, voigt=False, lorentzian=False, gaussian=False):
    """
    Generates a heatmap of a sample's Raman map scan, while also generating a heatmap displaying the R2 scores applied 
    to each fit across entire map scan.

    Inputs:
    - df: DataFrame containing all data from the input file.
    - yx: DataFrame containing map coordinates.
    - raman_shift: Array containing Raman shift values.
    - height: Dictionary containing all spectra and their respective coordinates on the map scan
    - spacing: Desired x coordinate to view spectra
    - deg: Desired y coordinate to view spectra
    - desired_ratio: Array containing user inputted integers ranging from 0-2.
    - ratio_label: Label of ratio found from labels and desired ratio.
    - voigt: Boolean; if true, voigt fitting will be used to generate heatmap.
    - lorentzian: Boolean; if true, lorentzian fitting will be used to generate heatmap.
    - gaussian: Boolean; if true, gaussian fitting will be used to generate heatmap.

    Returns:
    - avg_ratio: Average desired intensity ratio across entire heatmap.
    - fig1: Figure of heatmap.
    - fig2: Figure of R2 score heatmap.
    - spectra: Dictionary containing all spectra and their respective coordinates on the map scan
    """
    df = df.drop(0, axis = 1)
    df = df.T.reset_index(drop=True).T
    int_ratios = []
    r2_scores = []
    spectra = {}
    for i in list(df.columns):
        y = df[i].values
        y = pro.normalize(raman_shift, y)
        y = pro.remove_baseline(raman_shift, y, deg)
        raw_y = y
        peaks_y = pks.identify_peaks(raman_shift, y, height, spacing)
        if lorentzian:
            y, l_r2 = fit.raman_lfitter(raman_shift, y, peaks_y)
            r2_scores.append(l_r2)
        if gaussian:
            y, g_r2 = fit.raman_gfitter(raman_shift, y, peaks_y)
            r2_scores.append(g_r2)
        if voigt:
            y, v_r2 = fit.raman_vfitter(raman_shift, y, peaks_y)
            r2_scores.append(v_r2)
        peaks_y = pks.identify_peaks(raman_shift, y, height, spacing)
        int_ratio = pks.intensity_ratio(desired_ratio, peaks_y)
        int_ratios.append(int_ratio)
        spectra[(yx['x'][i], yx['y'][i])] = {'raw': raw_y, 'spectrum': y, 'r2_score': r2_scores[-1], 'int_ratio': int_ratios[-1]}
    int_ratios_df = pd.DataFrame(int_ratios)
    r2_scores_df = pd.DataFrame(r2_scores)
    yxr2 = yx.join(r2_scores_df)
    yxf = yx.join(int_ratios_df)
    yxf['y'] = yxf['y'].astype(float)
    yxf['x'] = yxf['x'].astype(float)
    yxr2['y'] = yxf['y'].astype(float)
    yxr2['x'] = yxf['x'].astype(float)
    yxf_pivot = yxf.pivot(index='y', columns='x', values=0)
    yxr2_pivot = yxr2.pivot(index='y', columns='x', values=0)
    avg_ratio = np.average(int_ratios)
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    sns.heatmap(yxf_pivot, cmap="inferno", ax=ax1)
    ax1.invert_yaxis()
    ax1.set_title(f"Intensity ratio of {ratio_label} over map scan")
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.heatmap(yxr2_pivot, cmap="inferno", ax=ax2)
    ax2.invert_yaxis()
    ax2.set_title("R2 scores over generated heatmap")
    return avg_ratio, fig1, fig2, spectra


# In[2]:


def lookup_spectrum(raman_shift, spectra, x, y):
    """
    Looks up desired spectrum from outputs of heat_map.

    Inputs:
    - raman_shift: Array containing Raman shift values.
    - spectra: Dictionary containing all spectra and their respective coordinates on the map scan
    - x: Desired x coordinate to view spectra
    - y: Desired y coordinate to view spectra
    """
    data = spectra.get((x, y), None)
    if data is not None:
        raw_spectrum = data['raw']
        spectrum = data['spectrum']
        r2 = data['r2_score']
        ratio_value = data['int_ratio']
        plt.plot(raman_shift, raw_spectrum, label = "raw")
        plt.plot(raman_shift, spectrum, label = "fitted")
        plt.title(f'Fitted Spectrum at ({x},{y}) R2:{round(r2, 4)} Ratio:{round(ratio_value, 4)}')
    else:
        raise ValueError(f'No Spectrum at Coordinates ({x},{y})')

