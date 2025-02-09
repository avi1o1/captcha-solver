# Codebase

## Project Structure
```
captcha-solver/
├── setup.sh                        # Script to create and activate virtual environment
├── requirements.txt                # File containing all the python dependencies
├── Task0/
│   ├── easyCreation.py             # Script to create easy dataset
│   ├── hardCreation.py             # Script to create hard dataset
│   ├── bonusCreation.py            # Script to create bonus dataset
├── Task1/
│   ├── Plots/                      # Folder containing all the plots generated during the training and hyperparameter tuning
│   ├── file.py                     # Python script to train (and test) the model
│   ├── results.csv                 # Results of the final test set
├── Task2/
│   ├── Plots/                      # Folder containing all the plots generated during the training and hyperparameter tuning
│   ├── Models/                     # Folder containing the trained models (for each subtask)
│   ├── Results/                    # Folder containing the results of the final evaluations of the models
│   ├── createTrain.py              # Script to create the training dataset
│   ├── createTest.py               # Script to create the test dataset
│   ├── file.py                     # Python script to train (and test) the model
│   ├── testModel.py                # Python script to test the model
```

## Installation and Set-up

1. Clone the repository: 
```bash
git clone https://github.com/avi1o1/captcha-solver
cd captcha-solver
```

2. Run the following script to create and activate virtual environment (named PreGawk), along with all the needed python dependencies:
```bash
chmod +x setup.sh
./setup.sh
```

## Dataset Generation

1. Run the following files to create the easy, hard and bonus datasets.
    - [Easy Dataset Creation](./Task0/easyCreation.py)
    - [Hard Dataset Creation](./Task0/hardCreation.py)
    - [Bonus Dataset Creation](./Task0/bonusCreation.py)

2. One may choose to tune parameters like font size, font style, number of labels and label frequency in either file. Note that these variables are declared as macros at the top of the file for the ease of access.

## Task 1: Image Classification

1. Ensure that the Data Generation step (Task 0) has been completed with appropriate label count and frequencies (For eg: 50 labels with 200 samples each).

2. Run [file.py](./Task1/file.py) to create dataset, train and evaluate the model.

3. One may choose to tune parameters like  NUM_SAMPLES, LEARNING_RATE, NUM_EPOCHS, etc. in the file. Note that these variables are declared as macros at the top of the file for the ease of access.

4. The results of the final test set will be stored in [results.csv](./Task1/results.csv).

5. The plots generated during the training and hyperparameter tuning can be found in the [Plots](./Task1/Plots) folder.

## Task 2: Image Segmentation

1. Run the following files to create the training and test datasets. You may adjust the parameters like IMAGE_COUNT.
    - [Training Dataset Creation](./Task2/createTrain.py)
    - [Test Dataset Creation](./Task2/createTest.py)

2. Run [file.py](./Task2/file.py) to train and test the model.

3. One may choose to tune parameters like NUM_SAMPLES, LEARNING_RATE, NUM_EPOCHS, BATCH_SIZE, etc. in the file. Note that these variables are declared as macros at the top of the file for the ease of access.

4. The plots generated during the training and hyperparameter tuning can be found in the [Plots](./Task2/Plots) folder.

5. The trained models can be found in the [Models](./Task2/Models) folder.

6. The results of the final evaluation will be stored in [results.csv](./Task2/results.csv).

7. Further, there is a script [testModel.py](./Task2/testModel.py) to test the model on a seperate random dataset.

8. The results of the final evaluations of the models can be found in the [Results](./Task2/Results) folder.
