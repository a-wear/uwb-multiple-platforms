import numpy as np
from torch.utils import data
import pandas as pd

class CirErrorDataset(data.Dataset):
    def __init__(self, data_dir, split):
        self.split = split

        df = pd.read_csv(data_dir)
        self.df = df # Keep also df

        try:
            cols_to_drop = ["toas", "corr_toas", "true_dist", "dist_err",
                "room", "recording", "los_nlos_label"]
            self.x = np.array(df.drop(cols_to_drop, axis=1), dtype='float32')
        except KeyError:
            self.x = np.array(df.drop("dist_err", axis=1), dtype='float32')
        self.y = np.array(df["dist_err"], dtype='float32')

        self.weights = None

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        index = index % self.x.shape[0]
        feature = self.x[index, :]
        label = np.expand_dims(np.asarray(self.y[index]), axis=0)

        if self.weights is not None:
            # TODO Reweighting not implemented
            raise ValueError("Reweighting not yet implemented")
            # weight = np.asarray([self.weights[index]]).astype('float32')
        else:
            weight = np.asarray([np.float32(1.)])

        return feature, label, weight

    def _get_labels(self):
        return self.y

    def _get_df(self):
        return self.df
