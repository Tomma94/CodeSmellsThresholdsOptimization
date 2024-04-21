# Code Smells Thresholds Optimization: Defect Prediction as a Case Study
This repository includes the training and evaluation code, along with the trained model and report, for the proposed model described in the paper 'Code Smells Thresholds Optimization: Defect Prediction as a Case Study'.

## Table of Contents
- [Motivation](#motivation)
- [Authors](#authors)
- [List of Code Smells](#list-of-code-smells)
- [Data](#data)
- [How To Run The Code](#how-to-run-the-code)
  - [Requirements](#requirements)
  - [Run Manually](#run-manually)
  - [Run the Entire Project](#run-the-entire-project)
  - [Run a Batch of Experiments](#run-a-batch-of-experiments)

    
## Motivation

Motivation
In software engineering, the impact of code smells on software quality and maintainability is profound, highlighting the necessity for refined detection methods. 
Traditional approaches to identifying code smells typically use static thresholds, which can lead to inaccuracies as they fail to consider the unique characteristics of different software components. 
Our research tackles this issue by developing a novel model that customizes code smell thresholds for each software component. 
This model is designed to enhance specific software practices by determining the most effective thresholds for each context. 
this research specifically demonstrates the code smell thresholds optimization approach as a case study to leverage the tailored thresholds to improve the performances of defect prediction models.


## Authors
Ben Gurion University of the Negev, Be'er Sheva, Israel.
- Tom Mashiach - tommas@post.bgu.ac.il
- Gilad Katz - giladkz@post.bgu.ac.il
- Meir Kalech - kalech@post.bgu.ac.il

## List of Code Smells
The 11 code smells considered in this work:
| Code Smell Name            | Description                                                                          | Calculation Formula                                        |
|----------------------------|--------------------------------------------------------------------------------------|------------------------------------------------------------|
| Lazy Class                 | A class does not have a single, well-defined responsibility.                         | LOC < T                                                    |
| Swiss Army Knife           | An abstract class has many responsibilities and functionality.                       | IA == 1 & IMC > T                                          |
| Refused Bequest            | A subclass does not use or override methods or properties that are inherited from its superclass. | OR > T                                        |
| Large Class                | A class has grown too large and contains too many methods or properties.             | LOC > T                                                    |
| Class Data Should Be Private | A class data is exposed publicly, rather than being kept private and only accessible through methods. | NOPF > T                                         |
| God Class                  | A class has too many responsibilities, and becomes too large and complex to understand and maintain. | LOC > T1 & TCC < T2                               |
| Multifaceted Abstraction   | A class has more than one responsibility assigned to it.                             | LCOM > T1 & NOF > T2 & NOM > T3                             |
| Unnecessary Abstraction    | A class that is actually not needed (and thus could have been avoided).              | NOPM == 0 & NOF < T                                        |
| Broken Modularization      | A class that is not cohesively encapsulating its responsibilities                    | NOPM == 0 & NOF > T                                        |
| Hub-Like Modularization    | A class has dependencies (both incoming and outgoing) with a large number of other classes. | FANIN > T1 & FANOUT > T2                            |
| Deep Hierarchy             | A class's inheritance hierarchy is excessively deep.                                 | DOI > T                                                    |

The relevance code metrics for these Code Smells are:
| Metric Name                           | Abbreviation | Description                                                                        |
|---------------------------------------|--------------|------------------------------------------------------------------------------------|
| Lines Of Code                         | LOC          | The number of lines of code in a class.                                            |
| Is Abstract                           | IA           | A boolean metric that indicates if a class is abstract (1) or not (0).             |
| Interface Method Declaration Count    | IMC          | The number of methods declared in an interface.                                    |
| Override Ratio                        | OR           | Overridden Methods / Overridable Superclass Methods.                               |
| Tight Class Cohesion                  | TCC          | The degree of relatedness of methods within a class based on their shared access to instance variables. |
| Lack of Cohesion in Methods           | LCOM         | The lack of cohesion in methods within a class.                                    |
| Number Of Fields                      | NOF          | Total number of fields in a class.                                                 |
| Number Of Public Fields               | NOPF         | Number of public fields in a class.                                                |
| Number Of Methods                     | NOM          | Total number of methods in a class.                                                |
| Number Of Public Methods              | NOPM         | Number of public methods in a class.                                               |
| FAN-IN                                | FANIN        | The number of other classes or components that use this class.                     |
| FAN-OUT                               | FANOUT       | The number of distinct classes that a given class uses.                            |
| Depth Of Inheritance                  | DIT          | The length of the inheritance path from a given class to its highest ancestor class. |

## Data
The dataset utilized in this research is available at [insert link or location where the data can be accessed]. 
If you wish to use your own dataset, please ensure that the names of the code metrics correspond with those used in our code. 
This alignment is crucial for the successful application of the model and analysis scripts provided in this repository.

In addition, a comprehensive table listing all the projects used in this research is provided. 
You can find this table in the file `projects.md`. 

## How To Run The Code

### Requirements

All the software requirements necessary to run this project are listed in a `requirements.txt` file, which can be found at [link to requirements.txt]. This file includes all the Python libraries and versions that are needed.

Additionally, to use the `run.sh` script provided in this repository, ensure that you have `bash` and `jq` installed on your system. 
`bash` is required to execute the script, and `jq` is used for parsing JSON data that is involved in the script processes.

### Run Manually

If you prefer to manually run each part of the project, follow these steps:

1. **Prepare the Data:**
   - Ensure you have the raw data downloaded from [link to raw data].
   - Execute the `train_test_projects_version_split` function from the `data_creation.py` file to preprocess the data and split it into training and testing sets. This script also provides the option to save these sets as CSV files.

2. **Train the Model:**
   - Use `thresholds_model.py` to train the model that generates optimal thresholds. Execute the `train_thresholds_model` function with the appropriate arguments to build and train your model. The output models and thresholds will be saved in the specified `{output_directory}/{run_number}` directory.

3. **Test the Model:**
   - After training, use the appropriate testing function (ensure to specify the function name) in the `thresholds_model.py` file to evaluate the model's performance on the test set.

4. **Baseline Model Training:**
   - Train the baseline model using the `train_origin_thresholds_model` function in `origin_thresholds_model.py`. Make sure the input arguments align with the baseline model's parameters for a fair comparison.

5. **Baseline Model Testing:**
   - Evaluate the baseline model using `test_origin_thresholds_model` to calculate and record the F1 score, PRC-AUC, and ROC-AUC metrics.

6. **Compare Models:**
   - To compare the results from the custom threshold model and the baseline model, run the `significance_checker.py` script. This will perform t-tests for each metric and plot the mean values for each metric across all projects for both models.

By following these steps, you can manually execute each component of the project, allowing for greater control and understanding of each phase.

### Run the Entire Project

To facilitate easy and efficient execution of the entire project, a main script is provided that runs all components sequentially. 
Below, you will find detailed explanations of the flags and options you can use with this script to customize the execution according to your needs.

#### Usage

```bash
python main_script.py [options]
```
- `-d, --results-dir RESULTS_DIR`: Specify the main directory to save the results. This will contain all output files.
- `-n, --run-number RUN_NUMBER`: Set the experiment run number. This also serves as a directory name under the results directory.
- `-b, --batch-size BATCH_SIZE`: Define the input batch size for training (default: 256).
- `-e, --epochs EPOCHS`: Set the number of epochs to train (default: 100).
- `-lrc, --lr-classifier LR_CLASSIFIER`: Learning rate for the classifier model (default: 0.001).
- `-lrg, --lr-generator LR_GENERATOR`: Learning rate for the generator model (default: 0.0005).
- `-lrd, --lr-discriminator LR_DISCRIMINATOR`: Learning rate for the discriminator model (default: 0.001).
- `-hc, --hidden-size-classifier HIDDEN_SIZE_CLASSIFIER`: Size of each hidden layer of the classifier (default: 256).
- `-hg, --hidden-size-generator HIDDEN_SIZE_GENERATOR`: Size of each hidden layer of the generator (default: 256).
- `-hd, --hidden-size-discriminator HIDDEN_SIZE_DISCRIMINATOR`: Size of each hidden layer of the discriminator (default: 256).
- `-nlc, --num-layers-classifier NUM_LAYERS_CLASSIFIER`: Number of hidden layers in the classifier network (default: 1).
- `-nlg, --num-layers-generator NUM_LAYERS_GENERATOR`: Number of hidden layers in the generator network (default: 2).
- `-nld, --num-layers-discriminator NUM_LAYERS_DISCRIMINATOR`: Number of hidden layers in the discriminator network (default: 1).
- `-es, --earlystop-patience EARLYSTOP_PATIENCE`: Number of epochs with no improvement to wait before stopping the training. Set to 0 to disable early stopping (default: 20).
- `-lwg, --loss-weight-generator LOSS_WEIGHT_GENERATOR`: Weight of the classification loss in the generator loss calculation (default: 1).
- `-s, --seed SEED`: Random seed for reproducibility (default: 42).
- `-f, --all-data-file ALL_DATA_FILE`: Path to the all data file. If you already have train/test files you do not need to pass it. 
- `-f1, --train-file TRAIN_FILE`: Path to the train data file. If this does not pass it will check for the `df_train.csv` file.
- `-f2, --test-file TEST_FILE`: Path to the test data file. If this does not pass it will check for the `df_test.csv` file.

Using this script will save all results and the trained model in the directory specified by `RESULTS_DIR\RUN_NUMBER`.

### Run a Batch of Experiments

For conducting multiple experiments in a batch, you can utilize the `run.sh` script in conjunction with a JSON configuration file. 
This script is designed to execute each experiment as an `sbatch` job, which is suitable for systems that use SLURM for job scheduling.
It is important to specify the required environment settings in the `run.sh` file.

The JSON file should structure its primary keys as run numbers, with their corresponding values being dictionaries. 
These dictionaries contain the script flags (except the `-n` flag, which is implicitly derived from the run number) as keys, and their specified values as the dictionary values. 

An example configuration for such a JSON file is available in `runs_dict.json`.

To execute a batch of experiments using this setup, run the following command:

```bash
./run.sh runs_dict.json
```
This command will create a separate folder for each experiment run, organizing the outputs neatly into designated directories within an output directory based on their run numbers. 
Each run is submitted as an individual sbatch job, allowing for efficient management and execution of multiple experiments.




