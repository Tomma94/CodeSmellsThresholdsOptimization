import argparse
import pandas as pd
import numpy as np
from origin_thresholds_model import train_origin_thresholds_model, test_origin_thresholds_model
from thresholds_model import train_thresholds_model, test_thresholds_model
from significance_checker import significance_checker
from models import *
import random
from args_validation import valid_int_positive, valid_int_positive_zero, valid_float_range
import data_creation

def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    # Python random module
    random.seed(seed_value)
    # NumPy
    np.random.seed(seed_value)
    # PyTorch
    torch.manual_seed(seed_value)
    torch.use_deterministic_algorithms(True)

def main():
    # Set up argparse
    parser = argparse.ArgumentParser(description="Code Smells thresholds optimization")
    parser.add_argument('-d','--results-dir', type=str, default='output', help='Main directory to save the results')
    parser.add_argument('-n', '--run-number', type=valid_int_positive, required=True,
                        help='Experiment run number, which will also be used as a directory name under the results directory')
    parser.add_argument('-b', '--batch-size', type=valid_int_positive, default=256, help='input batch size for training (default: 256)')
    parser.add_argument('-e', '--epochs', type=valid_int_positive, default=100, help='number of epochs to train (default: 100)')
    parser.add_argument('-lrc', '--lr-classifier', type=valid_float_range, default=0.001, help='learning rate for classifier model (default: 0.001)')
    parser.add_argument('-lrg', '--lr-generator', type=valid_float_range, default=0.0005, help='learning rate for generator model (default: 0.0005)')
    parser.add_argument('-lrd', '--lr-discriminator', type=valid_float_range, default=0.001, help='learning rate for discriminator model (default: 0.001)')
    parser.add_argument('-hc', '--hidden-size-classifier', type=valid_int_positive, default=256, help='The size of each hidden layer of the classifier. All hidden layers will have this number of neurons (default: 256).')
    parser.add_argument('-hg', '--hidden-size-generator', type=valid_int_positive, default=256, help='The size of each hidden layer of the generator. All hidden layers will have this number of neurons (default: 256).')
    parser.add_argument('-hd', '--hidden-size-discriminator', type=valid_int_positive, default=256, help='The size of each hidden layer of the discriminator. All hidden layers will have this number of neurons (default: 256).')
    parser.add_argument('-nlc','--num-layers-classifier', type=valid_int_positive, default=1, help='The number of hidden layers in the classifier network (default: 1).')
    parser.add_argument('-nlg','--num-layers-generator', type=valid_int_positive, default=2, help='The number of hidden layers in the generator network (default: 2).')
    parser.add_argument('-nld','--num-layers-discriminator', type=valid_int_positive, default=1, help='The number of hidden layers in the discriminator network (default: 1).')
    parser.add_argument('-es','--earlystop-patience', type=valid_int_positive_zero, default=20,
                        help='Number of epochs with no improvement to wait before stopping the training. Set to 0 to disable early stopping (default: 20).')
    parser.add_argument('-lwg','--loss-weight-generator', type=valid_int_positive_zero, default=1.0,
                        help='Weight of the classification loss in the generator loss calculation (default: 1).')
    parser.add_argument('-s', '--seed', type=valid_int_positive_zero, default=42, help='Random seed for reproducibility. (default: 42).')
    parser.add_argument('-f', '--all-data-file', type=str, help='Path to the all data file. If you already have train\\test files you do not nedd to pass it.')
    parser.add_argument('-f1', '--train-file', type=str, help='Path to the train data file. ')
    parser.add_argument('-f2', '--test-file', type=str, help='Path to the test data file.')


    args = parser.parse_args()

    # Set the seed
    set_seed(args.seed)

    # Printing the run arguments
    print('Run arguments:')
    for arg in vars(args):
        if 'file' in arg:
            continue
        print(f"{arg}: {getattr(args, arg)}")

    # calculate or load the data
    if args.all_data_file:
        try:
            df_train, df_test = data_creation.train_test_projects_version_split(args.all_data_file)
        except:
            raise ValueError("Error: There is a problem with your data.")

    elif args.train_file and args.test_file:
        # Read it to CSV
        try:
            df_train = pd.read_csv(args.train_file)
            df_test = pd.read_csv(args.test_file)
        except:
            raise ValueError("Error: There is a problem with your data.")

    else:
        try:
            df_train = pd.read_csv('df_train.csv')
            df_test = pd.read_csv('df_test.csv')
        except:
            raise ValueError("Error: No files of data provided. Please specify at least one file.")

    best_generator, best_classifier = train_thresholds_model(df_train= df_train,seed_arg= args.seed,
                                                             output_directory=args.results_dir, run_number=args.run_number,
                                                             num_of_epochs= args.epochs, batch_size=args.batch_size,
                                                             early_stop= args.earlystop_patience, hidden_dim_class= args.hidden_size_classifier,
                                                             hidden_dim_dis= args.hidden_size_discriminator, hidden_dim_gen=args.hidden_size_generator,
                                                             layers_num_dis= args.num_layers_discriminator, layers_num_class= args.num_layers_classifier,
                                                             layers_num_gen= args.num_layers_generator, learning_rate_gen= args.lr_generator,
                                                             learning_rate_class= args.lr_discriminator, learning_rate_dis=args.lr_classifier ,
                                                             loss_weight_generator= args.loss_weight_generator)

    results_model, means_metrics_model = test_thresholds_model(output_directory=args.results_dir, run_number=args.run_number,
                                                               df_test= df_test, best_generator=best_generator,
                                                               best_classifier=best_classifier)

    origin_thresholds_model = train_origin_thresholds_model(df_train= df_train, seed= args.seed, output_directory=args.results_dir,
                                                            run_number=args.run_number, batch_size= args.batch_size,
                                                            hidden_dim_class= args.hidden_size_classifier,
                                                            layers_num_class= args.num_layers_classifier,
                                                            learning_rate_class=args.lr_classifier, early_stop= args.earlystop_patience,
                                                            num_of_epochs=args.epochs)

    results_origin, means_metrics_origin = test_origin_thresholds_model(df_test= df_test, output_directory=args.results_dir,
                                                                        run_number=args.run_number, final_model=origin_thresholds_model)

    significance_results = significance_checker(output_directory=args.results_dir, run_number=args.run_number,
                                                results_model=results_model, results_origin=results_origin )

    print(significance_results)




if __name__ == "__main__":
    main()








