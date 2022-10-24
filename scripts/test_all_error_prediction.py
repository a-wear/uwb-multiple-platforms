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
    default='../data/trained_models_error_prediction/models_paper/',
    help='Root path where to find the models.')
parser.add_argument('--use-smogn', action="store_true",
    help='Use the SMOGN-augmented version of the train set.')
parser.add_argument('--layers', type=int, default=256,
    help='Nr of layers in the NN model')
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
# checkpoints
parser.add_argument('--resume', type=str, default='',
    help='Checkpoint file path to resume training')

parser.set_defaults(augment=True)
args, unknown = parser.parse_known_args()
args.resume = True

TEST_LOCATIONS = np.arange(0, 10, 1).astype("int")

all_devices = ["3db_midas3", "tdsr", "dw3000"]

file_root_run = "{:.0f}".format(time.time())
print("File root for this run: ", file_root_run)
args.file_root_run = file_root_run

print("Root dir dataset: ", args.root_dataset)

def main():
    list_results = []

    for test_loc in TEST_LOCATIONS:
        for train_device in all_devices:
            # Will load the model trained on train_device
            resume_dir = "train_{}_test_{}".format(train_device, test_loc)
            if args.use_smogn:
                resume_dir += "_smogn"
            print("Resume dir: ", args.resume)
            args.resume = os.path.join(args.root_checkpoints, resume_dir,
                "ckpt.best.pth.tar")
            # Check if file exists, otherwise continue
            if not os.path.exists(args.resume):
                print("File {} not found! Continuing...".format(args.resume))
                continue

            for test_device in all_devices:
                # Load the datasets for the test device
                dataset_name = "{}_test_{}".format(
                    test_device, test_loc) # Use the same test set
                print("Training on {}".format(dataset_name))
                trainer = TrainErrorPrediction(train_device=test_device,
                    dataset_name=dataset_name, args=args)
                trainer.load_datasets()
                trainer.init_model()
                print("Train: {}, test: {}".format(train_device, test_device))
                results = trainer.test()
                y_true, y_pred = trainer.get_predictions()

                # Save predictions in training directory
                curr_df = pd.DataFrame.from_dict({"y_true": y_true, "y_pred": y_pred})
                curr_df.to_csv(os.path.join(args.root_checkpoints, resume_dir,
                    "test_{}_predicted_true_{}.csv".format(test_device, file_root_run)), index=False)

                # Add results to DF
                dict_results = {
                    "train_dataset": train_device,
                    "test_dataset": test_device,
                    "test_location": test_loc,
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
    results_dir = "../data/cross_test_results/"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    results_filepath = "test_results_{}.csv".format(file_root_run)
    df_results.to_csv(os.path.join(results_dir, results_filepath), index=False)
    print("Wrote results in ", results_dir, results_filepath)
    print(df_results)

if __name__ == '__main__':
    main()

