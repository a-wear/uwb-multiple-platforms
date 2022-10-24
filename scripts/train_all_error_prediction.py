import time
import argparse
import os
import numpy as np
from common.TrainErrorPrediction import TrainErrorPrediction
import pandas as pd
os.environ["KMP_WARNINGS"] = "FALSE"

parser = argparse.ArgumentParser()

# training/optimization related
parser.add_argument('--root-dataset', type=str,
    default='../data/split_train_test_val/sets_paper/before30_after10/',
    help='Path where to find the pre-split training, test, and validation datasets.')
parser.add_argument('--root-checkpoints', type=str,
    default='../data/trained_models_error_prediction/',
    help='Root path for storing models, checkpoints and logs.')
parser.add_argument('--use-smogn', action="store_true",
    help='Use the SMOGN-augmented version of the train set. (By default, not used.)')
parser.add_argument('--device', type=str, default='all',
    choices=["all", "3db_midas3", "dw3000", "tdsr"],
    help="Which device to use for training (either all or one device only).")
parser.add_argument('--layers', type=int, default=256,
    help='Number of layers in the NN model.')
parser.add_argument('--optimizer', type=str, default='adam',
    choices=['adam', 'sgd'], help='Optimizer type')
parser.add_argument('--loss', type=str, default='l1',
    choices=['mse', 'l1', 'focal_l1', 'focal_mse', 'huber'],
    help='Training loss type')
parser.add_argument('--lr', type=float, default=1e-3,
    help='Initial learning rate')
parser.add_argument('--epoch', type=int, default=10,
    help='Number of epochs to train')
parser.add_argument('--momentum', type=float, default=0.9,
    help='Optimizer momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4,
    help='Optimizer weight decay')
parser.add_argument('--schedule', type=int, nargs='*', default=[60, 80],
    help='Learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--print_freq', type=int, default=10,
    help='Logging frequency')
parser.add_argument('--patience', type=int, default=10,
    help='Patience for early stopping')
parser.add_argument('--workers', type=int, default=32,
    help='Number of workers used in data loading')
parser.add_argument('--resume', type=str, default='',
    help='Checkpoint file path to resume training')

parser.set_defaults(augment=True)
args, unknown = parser.parse_known_args()

TEST_LOCATIONS = np.arange(0, 10, 1).astype("int")

if args.device == "all":
    all_devices = ["3db_midas3", "tdsr", "dw3000"]
else:
    all_devices = [args.device]

file_root_run = "{:.0f}".format(time.time())
print("File root for this run: ", file_root_run)
args.file_root_run = file_root_run

print("Root dir dataset: ", args.root_dataset)

def main():
    list_results = []
    for dev in all_devices:
        for test_loc in TEST_LOCATIONS:
            dataset_name = "{}_test_{}".format(
                dev, test_loc)
            print("Training on {}".format(dataset_name))
            trainer = TrainErrorPrediction(train_device=dev,
                dataset_name=dataset_name, args=args)
            trainer.load_datasets()
            trainer.init_model()
            trainer.train_all()
            results = trainer.test()

            # Add results to DF
            dict_results = {"train_dataset": dataset_name,
                "abs_err_before": results["abs_err_before"],
                "abs_err_after": results["abs_err_after"],
                "std_before": results["std_before"],
                "std_after": results["std_after"],
                "smogn": args.use_smogn}
            list_results.append(dict_results)

            print("=" * 80)

    df_results = pd.DataFrame.from_records(list_results)
    df_results["nr_epochs"] = args.epoch
    df_results["nr_layers"] = args.layers
    results_filepath = "../data/train_results_same_device_{}.csv".format(
        file_root_run)
    df_results.to_csv(results_filepath, index=False)
    print("Wrote results in ", results_filepath)
    print(df_results)

if __name__ == '__main__':
    main()

