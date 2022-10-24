# Code for Dataset: Ultra-Wideband Ranging Measurements Acquired With Three Different Platforms (Qorvo, TDSR, 3db Access)

This repository contains code examples written in Python to analyze the data
and reproduce the results from the paper:

L. Flueratoru, E. S. Lohan, and D. Niculescu, ``Challenges in
platform-independent UWB ranging and localization systems,'' in _The 16th ACM
Workshop on Wireless Network Testbeds, Experimental evaluation and
Characterization (WiNTECH) 2022_, ACM, 2022.


If you find the dataset useful, please consider citing our work:
```
@inproceedings{flueratoru2022challenges,
	title={Challenges in Platform-Independent {UWB} Ranging and Localization Systems},
	author={Flueratoru, Laura and Lohan, Elena Simona and Niculescu, Drago{\c{s}}},
	booktitle={The 16th ACM Workshop on Wireless Network Testbeds, Experimental evaluation and Characterization (WiNTECH) 2022},
	pages={},
	year={2022},
	organization={ACM}
}
```

The code accompanies the dataset published at:
[https://doi.org/10.5281/zenodo.6984698](https://doi.org/10.5281/zenodo.6984698)

## Prerequisites

1. Download the dataset and extract it to the root directory of this
repository. Rename the extracted directory to `data` such that the structure of
the repository becomes:
- `data`:
    - `parallel_measurements`
        - ...
    - `split_train_test_val`
    - `trained_models_error_prediction`
- `scripts` 2. Install the required Python packages listed in `env-uwb.yml`.
  
On a system where Anaconda is installed, run: 
``` 
conda create -n environment_name -f env-uwb.yml 
``` 
to create the environment from the YAML file.

To avoid deprecation errors in the future, the environment specifies the
package versions used at the date of writing this file. If any compatibility
issues arise, you can try to remove (some of) the versions and hope that
nothing will crash.


## Parsing the data in a recording

To read a recording as a Pandas DataFrame (for easier processing):
```
import pandas as pd
from ast import literal_eval

df = pd.read_csv(path_to_file, converters={"cir": literal_eval}) 
```

Please refer to the Jupyter notebook `read_dataset.ipynb` for more examples on how to read files and use the data. 

## Reproducing the results for the cross-platform performance of an error prediction model

### Split the train, test, validation sets

The script that splits the datasets is `split_sets_disjunct_locations.py` and
can be called with the default parameters: 
``` 
python split_sets_disjunct_locations.py 
``` 

The following parameters can be adjusted
for the script: number of samples before the TOA to keep, number of samples
after TOA to keep, number of datasets to generate, and whether to apply the
SMOGN algorithm to mitigate the sample imbalance. From brief preliminary tests,
the SMOGN augmentation did not seem to significantly improve the results. Also,
because it generates "artificial" CIRs by interpolating from the existing ones,
it is a rather opaque method, so we decided not to use it.

The method used to generate the splits is described in the paper, Section 4.

The pre-split datasets used in the paper can be found in the directory
`data/split_train_test_val/sets_paper/before30_after10`.

### Train the neural network

The script `train_all_error_prediction.py` trains models on all devices, for
all input datasets provided in a root directory. By default, the root directory
is set to the directory where the split sets used in the paper are found. The
models are saved in `data/trained_models_error_prediction/`. The trained models
used in the paper are found in a subdirectory of this path called
`models_paper`.

To run the script with the default arguments: 
``` 
python train_all_error_prediction.py 
```

For a list and description of all arguments run: 
``` 
python train_all_error_prediction.py --help 
```

The training script is loosely based on the training method for deep imbalanced
regression found at:
[https://github.com/YyzHarry/imbalanced-regression](https://github.com/YyzHarry/imbalanced-regression).

### Test the neural network
The script `test_all_error_prediction.py` cross-tests all the trained
model on all the devices. The model from the directory
`train_device1_test_k`, which was trained on the dataset
`device1_test_k/train.csv`, will be tested on all datasets
`device_i_test_k/test.csv`, where `device_i` can take the values
`3db_midas3`, `dw3000`, and `tdsr` (so it iterates through all
devices). Notice that the dataset index `k` is the same across the model and
the dataset used for training and testing.

To run the script on the pre-split datasets and models used in the paper:
```
python test_all_error_prediction.py
```

Run the same command with the argument `--help` to display the possible arguments
and their descriptions.

The script generates file with the name
`test_[device_name]_predicted_true_[timestamp].csv` inside the
directories where the models are saved. The files contain the predicted and the
true error values. Because the script inserts the timestamp in the filename,
the testing script can be run multiple times without overwriting the previous
test result files. This allows, for instance, testing a model on different test
datasets.

### Analyze the results

We can use the result files generated during the testing to analyze the
results. The trained models and the corresponding results can be found in the
directory `data/trained_models_error_prediction`. The notebook
`analyze_cross_testing_results.ipynb` reads the true and the predicted
errors from the test set `i` when applied on a model trained on the training
set `i` with each device. The notebook recreates Fig. 2 from the paper, which
shows the CDF of the original distance errors vs. the corrected errors using
models trained on the same or on different devices.
