#!/usr/bin/env python
# coding: utf-8

# In[29]:


import numpy as np
import peakutils as pku
import pandas as pd
import sklearn.preprocessing as slp
import matplotlib.pyplot as plt


# In[30]:


def read_raman_data(file_path, col_num, csv=False, map=False):
    """
    Reads Raman data from a text file or an Excel file.

    Inputs:
    - file_path: Path to the file containing Raman data.
    - col_num: Column in which you'd like to read data from.
    - csv: If true, this function will return a cleaned up .csv of your Raman data.
    - map: If true, this function assumes the data is in map format.

    Returns:
    - raman_shift: Array containing Raman shift values.
    - intensity: Array containing respective intensity values.
    - df: DataFrame containing all data from the input file.
    - yx: DataFrame containing map coordinates (if applicable).
    """
    yx = None
    if file_path.endswith((".txt", ".csv")):
        if file_path.endswith(".txt"):
            print("Reading text file...")
            df = pd.read_csv(file_path, delimiter="\t", header=None)
            header_row = pd.DataFrame([df.columns.tolist()], columns=df.columns)
            df = pd.concat([header_row, df], ignore_index=True)
            df = df.iloc[1:]
            df = df.reset_index(drop=True)
        if file_path.endswith(".csv"):
            print("Reading csv file...")
            df = pd.read_csv(file_path, delimiter=",")
    elif file_path.endswith(".xlsx"):
        print("Reading Excel file...")
        df = pd.read_excel(file_path, index_col=None, na_values=['NA'])
        df = df.map(lambda x: np.nan if isinstance(x, str) else x)
        df = df.dropna(axis=1, how='all')
        df = df.dropna(axis=0, how='all')
        df = df.iloc[1:]
    else:
        raise ValueError("Unsupported file format. Supported formats: .txt, .xlsx")
    if map:
        df.rename(columns={df.columns[0]: 'y', df.columns[1]: 'x'}, inplace=True)
        header_row = pd.DataFrame([df.columns.tolist()], columns=df.columns)
        df = pd.concat([header_row, df], ignore_index=True)
        yx = df[['y', 'x']]
        yx = yx.dropna(axis=0, how='all')
        yx = yx.reset_index(drop=True)
        yx = yx.drop(index=0)
        yx = yx.reset_index(drop=True)
        df = df.drop(df.columns[[0, 1]], axis=1)
        df = df.T
        df = df.reset_index(drop=True)
        df = df.drop(df.columns[0], axis=1)
        df.columns = range(df.shape[1])
    display(df)
    raman_shift = df.iloc[:, 0].values
    intensity = df.iloc[:, col_num].values
    if csv == True:
        df.to_csv(f"{file_path}_cleaned.csv", index=False, header=False)
    else:
        pass
    
    return raman_shift, intensity, df, yx


# In[35]:


def remove_baseline(x, y, deg):
    """
    Removes baseline from input y data.

    Inputs:
    - x: Array containing wavelength data.
    - y: Array containing intensity data.

    Returns:
    - y0: Array containing y data with intensity removal.
    """
    base = pku.baseline(y, deg)
    y0 = y-base
    plt.plot(x,y)
    plt.title("raw")
    plt.ylim(0, max(y)+(max(y)/4))
    plt.xlim(min(x), max(x))
    plt.show()
    plt.clf()
    plt.plot(x,y0)
    plt.title("baseline subtraction")
    plt.ylim(0, max(y0)+(max(y0)/4))
    plt.xlim(min(x), max(x))
    plt.show()
    return y0


# In[36]:


def normalize(x,y):
    """
    Normalizes input y data.

    Inputs:
    - x: Array containing x values.
    - y: Array containing y values.

    Returns:
    - y_norm: Array containing normalized y values.
    
    """
    y = np.array(y)
    y_norm = slp.normalize([y])
    y_norm = y_norm[0]
    plt.plot(x,y)
    plt.title("raw")
    plt.ylim(0, max(y)+(max(y)/4))
    plt.xlim(min(x), max(x)) 
    plt.show()
    plt.clf()
    plt.plot(x,y_norm)
    plt.ylim(0, max(y_norm)+(max(y_norm)/4))
    plt.xlim(min(x), max(x))
    plt.title("normalized")
    plt.show()
    return y_norm

