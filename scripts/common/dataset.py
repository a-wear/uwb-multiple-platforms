import pandas as pd
import numpy as np
import os
from ast import literal_eval
import random

def read_process_df(root_dir, device=None, filename="processed_data.csv"):
        df = pd.read_csv(os.path.join(root_dir, filename), 
            converters={"cir": literal_eval})
        df.rename(columns={"los_nlos_label": "label"}, inplace=True)
        return df
    
def get_samples_at_distance(df, distance, min_nr_samples=None):
    df = df.reset_index(drop=True)
    idxs_bool = (np.abs(df["true_distance"] - distance)) < 0.05
    if min_nr_samples is None:
        sel_df = df.loc[idxs_bool].copy().reset_index(drop=True)
    else:
        nr_samples = np.sum(idxs_bool)
        if min_nr_samples >= nr_samples:
            # Return all current samples
            sel_df = df.loc[idxs_bool].copy().reset_index(drop=True)
        else:
            # Select a number of subsamples 
            idxs = list(np.where(idxs_bool)[0])
            idxs_random = random.sample(idxs, min_nr_samples)
            sel_df = df.loc[idxs_random].copy().reset_index(drop=True)

    return sel_df

def get_cirs(df, cir_len):
    cirs = np.array([np.array(a) for a in df["cir"]])
    if cirs.shape[1] > cir_len:
        cirs = cirs[:, :cir_len] # Reduce CIR length
    else:
        # TODO: pad with zeros
        pass
    return cirs

def get_samples_at_multiple_distances(df, distances, min_nr_samples=None):
    eps = 0.05
    all_idxs = []
    df = df.reset_index(drop=True)
    for d in distances:
        idxs = np.where((np.abs(df["true_distance"] - d)) < eps)[0]

        if min_nr_samples is None:
            all_idxs.extend(list(idxs))
        else:
            nr_samples = len(idxs)
            if min_nr_samples >= nr_samples:
                all_idxs.extend(list(idxs))
            else:
                idxs_random = random.sample(list(idxs), min_nr_samples)
                all_idxs.extend(list(idxs_random))

    sel_df = df.loc[all_idxs].copy().reset_index(drop=True)
    return sel_df

def scale_cirs_dataset(x, maxx=None, minx=None, desired_max_cir=1):
    # Scale all CIRs to be between 0 and 1
    if maxx is None and minx is None:
        minx = np.min(x)
        maxx = np.max(x)

    rangex = maxx - minx
    x = (x - minx) * desired_max_cir / rangex
    return x, minx, maxx