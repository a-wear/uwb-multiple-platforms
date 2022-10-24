import numpy as np
import pandas as pd
import os
from time import time
import random
import smogn
import matplotlib.pyplot as plt
import matplotlib.colors as clr
from common.dataset import read_process_df, get_samples_at_distance, \
    get_cirs, get_samples_at_multiple_distances, scale_cirs_dataset
from pprint import pprint
import argparse
import scipy.stats as sps
from scipy.interpolate import make_interp_spline, BSpline
from common.custom_colors import mycolor

parser = argparse.ArgumentParser()
# CPU only
parser.add_argument('--pad-before', type=int, default=30, 
    help='How many samples to keep from the CIR before the TOA')
parser.add_argument('--pad-after', type=int, default=10, 
    help='How many samples to keep from the CIR after the TOA')
parser.add_argument('--nr-sets', type=int, default=6, 
    help='How many sets to generate')
parser.add_argument('--apply-smogn', action="store_true",
                    help='Apply SMOGN algorithm to augment the training set.')
parser.add_argument('--root-dir-save',  type=str, 
                    default="../data/split_train_test_val/",
                    help='Directory where to save the sets')
args = parser.parse_args()


RECORDINGS_FIXED_LOCATIONS = {
    "TAU_TD416": ["NLOS1"],
    "TAU_TD424": ["NLOS1"],
    "TAU_TD418": ["LOS1", "LOS2", "NLOS1"],
    "TAU_Tietotalo_Floor2": ["LOS1", "NLOS1"],
    "TAU_Reaktori": ["NLOS1"],
    "TAU_Hertsi": ["LOS1", "LOS3", "NLOS1", "NLOS3"],
    "TAU_Bunker": ["LOS1"],
    "TAU_Sahkotalo": ["NLOS1"],
}


all_devices = ["dw3000", "3db_midas3", "tdsr"]
dir_save_figs = "./images/"
root_recording_dir = "../data/parallel_measurements/"

file_root_run = "{:.0f}".format(time())
print("File root for this run: ", file_root_run)


def main():
    start_t = time()

    print("CIR settings: pad before {}, pad after: {}".format(
        args.pad_before, args.pad_after))

    args.root_dir_save = os.path.join(args.root_dir_save, 
        "before{}_after{}".format(args.pad_before, args.pad_after))
    create_dir(args.root_dir_save)
        
    for i_set in range(args.nr_sets): 
        # We generate a number of sets in which we select at random a physical location
        # from each room to go to the test set, one for the validation set (if any left),
        # and the rest to the train set (if any left). Most importantly, set with index i
        # from all devices contain *the same* physical locations in the train, test, and 
        # validation set. So when we train a model on a certain set, if it doesn't work 
        # with a different device, it's because of the device characteristics, since the
        # locations were identical.

        print("*" * 100)
        print("CURRENT SET: ", i_set)
        print("*" * 100)

        data_sets = {}
        for d in all_devices:
            data_sets[d] = {}
            for s in ["train", "train_smogn", "test", "val"]:
                data_sets[d][s] = {}
                data_sets[d][s]["cirs"] = np.array([])
                data_sets[d][s]["toas"] = []
                data_sets[d][s]["corr_toas"] = []
                data_sets[d][s]["true_dist"] = []
                data_sets[d][s]["dist_err"] = []
                data_sets[d][s]["room"] = []
                data_sets[d][s]["recording"] = []
                data_sets[d][s]["los_nlos_label"] = []

        for room, list_recordings in RECORDINGS_FIXED_LOCATIONS.items():
            for recording in list_recordings:
                print("\tRoom {}, recording {}".format(room, recording))
                # First read all the true distances/physical locations/steps from a 
                # selected recording; they are the same for all devices

                initial_dev = all_devices[0]
                dev_dir = os.path.join(root_recording_dir, initial_dev, room, recording)
                dev_df = read_process_df(dev_dir, filename="unaligned_processed_data.csv")
                dev_df = dev_df.dropna(axis=0) # Drop nan fields

                all_true_distances = list(set(list(dev_df["true_distance"])))

                # Choose at random one physical location to go to the test set
                test_step = random.choice(all_true_distances)
                remaining_steps = all_true_distances.copy()
                remaining_steps.remove(test_step) # Remove step from remaining ones

                # Choose at random one physical location to go to the val set
                if len(remaining_steps) >= 1:    
                    val_step = random.choice(remaining_steps)
                    remaining_steps.remove(val_step)
                else:
                    val_step = None

                # If more steps left, add them to the train set
                if len(remaining_steps) >= 1:    
                    train_steps = remaining_steps
                else:
                    train_steps = []

                # Initialize the label of the recording, we'll need it later
                if recording.startswith("LOS"):
                    los_nlos_label = 0
                else:
                    los_nlos_label = 1

                # Build a set for each device
                for idx_dev, dev in enumerate(all_devices):
                    print("\t\t", dev)

                    if dev == "dw3000":
                        min_nr_samples = 50
                    else:
                        min_nr_samples = 100

                    if dev != initial_dev:
                        # Avoid reading again the dataset of the initial device; we've just read it
                        dev_dir = os.path.join(root_recording_dir, dev, room, recording)
                        dev_df = read_process_df(dev_dir, filename="unaligned_processed_data.csv")
                        dev_df = dev_df.dropna(axis=0) # Drop nan fields

                    # Get data for the chosen set and add it to the test set
                    df_test = get_samples_at_distance(dev_df, test_step, min_nr_samples)
                    add_data_to_set(data_sets[dev], which_set="test", df=df_test, room=room, recording=recording,
                        label=los_nlos_label, pad_before=args.pad_before, pad_after=args.pad_after)
                    print("\t\tAdded {} samples to test from {}, {}, step {}".format(
                        df_test.shape[0], room, recording, test_step))

                    if val_step is not None:
                        # Get data for validation set and add it
                        df_val = get_samples_at_distance(dev_df, val_step, min_nr_samples)
                        add_data_to_set(data_sets[dev], which_set="val", df=df_val, room=room, recording=recording,
                            label=los_nlos_label, pad_before=args.pad_before, pad_after=args.pad_after)
                        print("\t\tAdded {} samples to val from {}, {}, step {}".format(
                            df_val.shape[0], room, recording, val_step))

                    if len(train_steps) > 0:
                        # Add the rest of the steps to the train set:
                        df_train = get_samples_at_multiple_distances(dev_df, train_steps, min_nr_samples)
                        add_data_to_set(data_sets[dev], which_set="train", df=df_train, room=room, recording=recording,
                            label=los_nlos_label, pad_before=args.pad_before, pad_after=args.pad_after)
                        print("\t\tAdded {} samples to train from {}, {}, steps {}".format(
                            df_train.shape[0], room, recording, train_steps))

                    print("\t\t---")
                print("\t***************")

        print("=" * 80)
        print("Built set number {}, now post-processing".format(i_set))

        for dev in all_devices:
            print("Post-processing for device ", dev)
            sel_set = data_sets[dev]

            # Plot distribution of train, test, val to check everything is ok
            fig, ax = plt.subplots(1, 3, figsize=(15, 4))
            for i, which_set in enumerate(["train", "val", "test"]):
                x = np.array(sel_set[which_set]["dist_err"])
                if which_set == "train":
                    _, _, bins = plot_pdf_x(x, fig=fig, ax=ax[i], 
                        xlabel="Distance error [m]", color=mycolor["color1"])
                else:
                    _ = plot_pdf_x(x, bins=bins, fig=fig, ax=ax[i], 
                        xlabel="Distance error [m]", color=mycolor["color1"])
                ax[i].set_title("{}, {} samples".format(which_set, x.size))
                ax[i].set_xlim([-0.5, 3])
                ax[i].set_ylim([0, 0.5])
            fig.suptitle("{}, set {}".format(dev, i_set))
            fig.tight_layout()
            filename = "{}_test-{}_{}.pdf".format(
                dev, i_set, file_root_run)
            fig.savefig(os.path.join("./images/", filename), format="pdf")
            plt.show()
            print("\t\tSaved file ", filename)

            # Scale CIRs 
            scale_sets(sel_set) 
            
            # Shuffle data in all sets
            shuffle_all_sets(sel_set)

            # Save to file
            curr_run_dir = "{}_test_{}".format(dev, i_set)
            dir_save = os.path.join(args.root_dir_save, curr_run_dir)
            for s in ["train", "test", "val"]:
                fn = "{}.csv".format(s)
                save_set_to_file(sel_set[s], root_dir=dir_save, filename=fn)
                print("\t\tSaved ", s)

            if args.apply_smogn:
                print("\t\tApplying SMOGN...")
                # Apply SMOGN on train
                df_smogn = None
                for i_try in range(3): # Try 3 times to apply SMOGN
                    try:
                        df_smogn = apply_smogn(sel_set["train"])
                        print("\t\tSMOGN succeeded")
                        break # If succeeded, break
                    except Exception as e:
                        # Sometimes SMOGN fails and I'm not sure why, because if you run
                        # it again it usually works. If it fails, just don't save the SMOGN
                        # set for this run and continue
                        print("\t\tSMOGN failed on try {}: {}".format(i_try, e))
                        df_smogn = None

                if df_smogn is None:
                    continue # Continue with the next device

                # Plot target data before and after
                y = np.array(sel_set["train"]["dist_err"])
                y_smogn = np.array(df_smogn["dist_err"])

                fig, ax, bins = plot_pdf_x(x=y, label="Initial", color=mycolor["color1"],
                    nr_bins=60)
                fig, ax, _ = plot_pdf_x(x=y_smogn, bins=bins, ax=ax, fig=fig,
                    label="After SMOTR", color=mycolor["neutral1"])
                ax.legend()
                ax.set_xlabel("Distance error [m]")
                ax.grid("--")
                ax.set_xlim([-0.7, 3])
                ax.set_ylim([0, 0.5])
                fig.tight_layout()
                filename = "{}_test-{}_smoter_{}.pdf".format(
                    dev, i_set, file_root_run)
                fig.savefig(os.path.join("./images/", filename), format="pdf")
                plt.show()
                
                # Save SMOGN set
                fn = "train_smogn.csv"
                df_smogn.to_csv(os.path.join(dir_save, fn), index=False)
                print("\t\tSaved SMOGN set")

    print("Run time: ", time() - start_t)

def add_data_to_set(data_sets, which_set, df, room, recording, label, pad_before=None, pad_after=None):
    # Get all cirs as a matrix
    cirs = get_cirs(df, np.inf) # Get original CIR len

    nr_meas = cirs.shape[0]

    toas = np.array(df["toa"], dtype="int")
    dist_err = np.array(df["measured_distance"]) - np.array(df["true_distance"])
    ns_err = dist_err / 3e-1 
    corr_toas = toas.astype("float") - ns_err

    # Realign CIRs to TOA
    cirs = align_cir_matrix_pad(
        cirs, toas, pad_before=pad_before, pad_after=pad_after)

    if data_sets[which_set]["cirs"].size == 0:
        data_sets[which_set]["cirs"] = cirs
    else:
        data_sets[which_set]["cirs"] = np.vstack((data_sets[which_set]["cirs"], cirs))
    data_sets[which_set]["toas"].extend(list(toas))
    data_sets[which_set]["corr_toas"].extend(list(corr_toas))
    data_sets[which_set]["true_dist"].extend(list(df["true_distance"]))
    data_sets[which_set]["dist_err"].extend(list(dist_err))
    data_sets[which_set]["room"].extend([room for i in range(nr_meas)])
    data_sets[which_set]["recording"].extend([recording for i in range(nr_meas)])
    data_sets[which_set]["los_nlos_label"].extend([label for i in range(nr_meas)])

def align_cir_matrix_pad(cirs, toas, pad_before=50, pad_after=150):
    cir_len = cirs.shape[1]
#     print("cir shape:", cirs.shape)
    new_cirs = np.zeros((pad_before + pad_after,))
    for cir_line, toa in zip(cirs, toas):
        if toa >= pad_before:
            # Enough samples before TOA, so we just get the portion before the TOA
            cir_beginning = cir_line[toa - pad_before:toa]
        else:
            # Not enough samples before TOA, so we need to pad the portion. We get the 
            # mean and std dev of the noise before the TOA and we generate a number of
            # (pad_before - toa) samples distributed normally around that mean with 
            # that standard deviation
            pad_len = pad_before - toa
            pad_noise = np.zeros((pad_len,))
            cir_beginning = np.hstack((pad_noise, cir_line[:toa]))
        
        if toa + pad_after <= cir_len:
            # Enough samples after TOA
            cir_end = cir_line[toa:toa + pad_after]
        else:
            # Not enough samples after TOA, pad with noise with the same distribution
            # as before the TOA
            pad_len = pad_after - (cir_len - toa)
            pad_noise = pad_noise = np.zeros((pad_len,))
            cir_end = np.hstack((cir_line[toa:], pad_noise))
            
        new_cir_line = np.hstack((cir_beginning, cir_end))
#         print("new_cir_line shape: ", new_cir_line.shape)
        new_cirs = np.vstack((new_cirs, new_cir_line))
    new_cirs = new_cirs[1:, :]
    return new_cirs

def plot_pdf_x(x, ax=None, fig=None, **kwargs):
    if "label" in kwargs.keys():
        label = kwargs["label"]
    else:
        label = ""

    if "color" in kwargs.keys():
        color = kwargs["color"]
    else:
        color = np.random((3, 1))

    if "nr_bins" in kwargs.keys():
        nr_bins = kwargs["nr_bins"]
    else:
        nr_bins = 50

    if ax is None or fig is None:
        fig, ax = plt.subplots()
        
    if "bins" in kwargs.keys():
        bins = kwargs["bins"]
    else:
        bins = np.linspace(np.min(x), np.max(x), nr_bins)
    
    hist, _ = np.histogram(x, bins, density=False)
    hist = hist / x.size

    # Draw histogram bar
    b1 = ax.bar(bins[:-1], hist, width=bins[1] - bins[0], alpha=0.3, 
        label=label, color=color)

    # Draw outline of distribution
    x_int, y_int = compute_interp(bins, hist)
    ax.plot(x_int, y_int, color=color, linewidth=2)

    if "xlabel" in kwargs.keys():
        ax.set_xlabel(kwargs["xlabel"])

    ax.set_ylabel("PDF")
    
    return fig, ax, bins

def compute_interp(bins, data, nr_bins=200):
    xnew = np.linspace(bins.min(), bins.max(), nr_bins) 
    spl = make_interp_spline(bins[:-1], data, k=3)  # type: BSpline
    power_smooth = spl(xnew)
    return xnew, power_smooth

def scale_cirs(x, minx=None, maxx=None, target_max=1):
    if minx is None or maxx is None:
        minx = np.min(x)
        maxx = np.max(x)
    rangex = maxx - minx
    x = (x - minx) * target_max / rangex
    return x, minx, maxx

def create_dir(my_dir):
    if not os.path.exists(my_dir): # Create dir if it doesn't exist
        os.makedirs(my_dir)

def scale_sets(data_dict):
    # Scale train and remember min, max
    data_dict["train"]["cirs"], min_cir, max_cir = scale_cirs(data_dict["train"]["cirs"])
    print("\t\tCIR scaling min and max: ", min_cir, max_cir)

    # Scale test and val with the same params
    data_dict["test"]["cirs"], _, _ = scale_cirs(data_dict["test"]["cirs"],
                                              minx=min_cir, maxx=max_cir)
    data_dict["val"]["cirs"], _, _ = scale_cirs(data_dict["val"]["cirs"],
                                              minx=min_cir, maxx=max_cir)

def shuffle_all_sets(all_data_sets):
    for s in ["train", "test", "val"]:
        shuffle_set(all_data_sets[s])

def shuffle_set(data_dict):
    nr_meas = len(data_dict["toas"])
    perm = np.arange(nr_meas).astype("int")
    np.random.shuffle(perm)

    # Apply permutation to CIRs
    data_dict["cirs"] = data_dict["cirs"][perm, :]

    # Apply permutation to all other fields
    all_keys = list(data_dict.keys())
    all_keys.remove("cirs")
    for k in all_keys:
        data_dict[k] = np.array(data_dict[k])[perm]

def save_set_to_file(data_dict, root_dir, filename):
    create_dir(root_dir)
    x = data_dict["cirs"]

    # Each CIR sample is a separate column
    df = pd.DataFrame(x)

    # Save the rest of the fields with the column name
    all_keys = list(data_dict.keys())
    all_keys.remove("cirs")
    for k in all_keys:
        df[k] = np.array(data_dict[k])
    file_path = os.path.join(root_dir, filename)

    df.to_csv(file_path, index=False)

def apply_smogn(data_dict):
    x = data_dict["cirs"]
    y = np.array(data_dict["dist_err"])
    df = pd.DataFrame(x)
    df["dist_err"] = y

    df_smoter = smogn.smoter(
        data=df,
        y="dist_err",
        k=3,
        samp_method="extreme",
        rel_method="auto",
        rel_thres=0.4,
        rel_xtrm_type="both",
    )
    return df_smoter

if __name__ == '__main__':
    main()

