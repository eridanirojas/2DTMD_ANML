#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt


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
            plt.fill_between(raman_shift,intensity, 0, where=(raman_shift>=left_ip-(additive*3.75))&(raman_shift<=right_ip+(additive*3.75)),color = 'blue', alpha = 0.25)
        plt.hlines(width_heights, left_ips, right_ips, color='g')
    plt.ylim(0, max(intensity)+(max(intensity)/4))
    plt.xlim(min(raman_shift), max(raman_shift))    
    plt.xlabel('Raman Shift (cm^-1)')
    plt.ylabel('Intensity (a.u.)')
    if mtl_name:
        plt.title(f"{mtl_name}")
    plt.show()

