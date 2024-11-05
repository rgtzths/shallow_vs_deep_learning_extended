# shallow_vs_deep_learning_extended
This repository is the code basis for the paper titled ""

## Install

(Using python 10)

Create a venv: `python -m venv env`

Use the environment `source env/bin/activate`

Install the requirements `pip install -r requirements.txt`

## Dataset

### Generate the dataset

To generate the dataset in the default folder run on the git root folder `python dataset/preprocessing.py`

If you want any of the other two data genereation options running `python dataset/preprocessing.py -h` will show the possible options.

### Dataset characteristics

| Total examples | Nº Features | <div>Examples per class <br>(URLLC\eMBB\mMTC)</div> |
|----------------|-------------|----------------------------------------------------|
| | **Training set** | |
| 373391         | 60          | 45\27\28                               |
| | **Test set** | |
| 93348          | 60          | 45\27\28                               |


## Model optimization
The shallow model's optimized hyperparameters are presented in the tables below.
For more information regarding the model's hyperparamerters visit the Scikit-learn website https://scikit-learn.org/ 

#### Logistic Regression
| Hyperparameter     | Values                        |
|--------------------|-------------------------------|
| C                  | 0.001, 0.01, 0.1, 1, 10       |
| Penalty            | l1, l2, Elasticnet            |
| Solver             | Sag, Saga, Lbfgs              |


#### Support Vector Machine
| Hyperparameter     | Values                        |
|--------------------|-------------------------------|
| C                  | 0,001, 0.01, 0.1, 1, 10       |
| Kernel             | Linear, RBF                   |

#### K-Nearest Neighbors
| Hyperparameter     | Values                        |
|--------------------|-------------------------------|
| Nº Neighbors       | 3, 5, 7                       |
| Weights            | Uniform, Distance             |

#### Decision Tree
| Hyperparameter     | Values                        |
|--------------------|-------------------------------|
| Criterion          | Gini, Entropy                 |
| Max depth          | 3, 5, 7, 9, 15, 20, 25, 30    |
| Max features       | auto, sqrt, log2              |

#### Gaussian Naive Bayes
| Hyperparameter     | Values                        |
|--------------------|-------------------------------|
| Variable smoothing | logspace(0, -9)               |

#### Random forest
| Hyperparameter     | Values                        |
|--------------------|-------------------------------|
| Nº estimators      | 2, 3, 5, 10, 20, 50, 100, 200 |
| Max depth          | 3, 5, 7, 9, 15, 20, 25, 30    |
| Max features       | auto, sqrt, log2              |

#### AdaBoost
| Hyperparameter     | Values                        |
|--------------------|-------------------------------|
| Nº estimators      | 2, 3, 5, 10, 20, 50, 100, 200 |

#### Gradient boosting
| Hyperparameter     | Values                        |
|--------------------|-------------------------------|
| Nº estimators      | 2, 3, 5, 10, 20, 50, 100, 200 |
| Max depth          | 3, 5, 7, 9, 15, 20, 25, 30    |
| Max features       | auto, sqrt, log2              |

## Run experiments

If the preprocessed dataset is in the default folder:

In the root folder run `python dl_models/train.py` or `python shallow_models/train.py`

If the dataset is on a different folder run `python dl_models/train.py -h` to see the arguments on how to define the dataset folder.

The seed used can also be changed as shown in the possible arguments.

The shallow models are automatically go through the grid search when running the experiments.

## Results