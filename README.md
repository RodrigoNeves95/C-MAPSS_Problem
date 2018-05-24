# C_MAPSS Turbofan problem

Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) is a turbofan simulation model. C-MAPSS was used to generate a simulated run-to-failure dataset from a turbofan engine and it is published in NASA’s prognostics center of excellence repository, that is [publicly available](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository). A big bottleneck in data driven approaches to tackle problem in predictive maintenance scope is the lack of availability run-to-failure datasets. This simulated dataset allows researchers to build, test and benchmark different approaches to this problem.

The C-MAPSS dataset is composed of four sub-datasets with different operation and fault conditions. Each sub-dataset is further divided in training and test sets. The data consists of multiple multivariate time series measurements. Within each dataset, each engine is considered from a fleet of engines of the same type and where each time series is from a single engine. Each engine starts with different levels of initial wear and manufacturing variation which is unknown. These wear and variations are considered as a normal behavior for each engine. The engine is operating normally at the start of each time series, and develops a fault at some point. In the training set, the fault grows in magnitude until engine failure. In the test set, the time series ends some time prior to the failure. The objective is to predict the number of remaining operational cycles before failure in the test set, i.e., the number of operational cycles left in the engine. This is called the Remaining Useful Life (RUL). There are 21 sensors measurements and three operational settings. Each row in the data is a snapshot taken during a single operation time cycle, for a certain engine.

# Algorithms

The following list of architectures were used to solve this problem with the purpose of comparing them:

  * RNN Architectures [RNN + GRU + LSTM cell]
  * Dilated Recurrent Architectures [paper](https://arxiv.org/abs/1710.02224) [code](https://github.com/code-terminator/DilatedRNN)
  
  * Quasi-RNN [paper](https://arxiv.org/abs/1611.01576) [code](https://github.com/salesforce/pytorch-qrnn)
  * TCN [paper](https://arxiv.org/abs/1803.01271) [code](https://github.com/locuslab/TCN)
  
For a detailed explanation about the models check the links.
  
# Requirements

This codebase was developed using Python 3 and PyTorch 0.3.1. Only GPU option is supported for now.

Install the requiremnts `pip install -r requirements.txt`.

To run the scripts you should install the `turbofan_pkg`

```
git clone https://github.com/RodrigoNeves95/C-MAPSS_Problem
cd C-MAPSS_Problem
pip install -e turbofan_pkg
```

# Preprocessing

The data cames in different .txt files. To read and create train and test set dataframes run the following command:

`python process.py --data_path DATA_FOLDER`

`DATA_FOLDER` should contain and txt files. This script will read and create two dataframes as well as creating the labels. The RUL is clipped to the maximum of 130 cycles. HDBScan is used to cluster 6 operational settings that we know that exist and have a big influence in the problem. It will be created two dataframes `df_train_cluster_piecewise.pkl` and `df_test_cluster_piecewise.pkl` under `DATA_FOLDER` file.

# Methodology

All algorithms will be trained using CV, where it used the [skopt](https://scikit-optimize.github.io/) package to hyperparameter optimization. Sensors are normalizard using Min-Max normalization, taking in account the operational setting.
The data is transformed on the fly to feed the models.

# Usage

This script will use the `model_to_run` to train and make predictions, where the skopt package will optimize the hyperparameters. Then the best model will be used on the test set. The results are saved on `models_storage_folder/model_to_run/Run_Best_Model/`

```
python FinalScript.py --data_train_path PATH_TO_DATAFRAME_TRAIN \
                      --data_test_path PATH_TO_DATAFRAME_TEST \
                      --SCRIPTS_FOLDER models_storage_folder \
                      --model model_to_run \
                      --file file_name
```
