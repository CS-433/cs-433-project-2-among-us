# Project 2 Machine Learning
Project submission for the second project of the machine learning course.

Team Among Us

17 December 2020


# Usage
1. Install the necessary libraries
2. Extract the two pickle zip folders under Models/RF
	(see below, in [Contents](#Contents))
3. Run the file `run.py`
4. Follow the on-screen prompts:
	* Select the model between `MK` for Markovian, `LSTM`, `RF` or `TCN`.
		Note that the code will loop until you input a valid model.
	* Select whether to perform a hyperparameter optimization,
		inputting `1` if you wish to perform one, or `0` if you do not.
	* Select the number of days you want to take into consideration.
		Note that if you are not performing a hyperparamter optimization, this value
		must be 2, 10, 50, 100 or 150.
	* Note that the last two options will not appear if you have chosen a
	  Markovian model
5. Wait until the code finished to execute.
	 If you are performing a hyperparamter optimization, this can take a while.
6. Visualize the results in the plots that appear or by seeing the performance
	 indcators in the console.


# Contents
```
project
│   README.md
│   LSTM_run.py    
│   MK_run.py  
│   preprocess.py  
│   RF_run.py  
│   run.py  
│   TCN_run.py  
│
└───Archives
└───Data
└───Figures
└───Helpers
|   │   p_indicators.py
|   │   performance_comparison.py
|   |   relabel.py
|   |   state_estimation.py
|
└───Models
│   └───LSTM
│   └───TCN
│   └───RF
│       │   RF_pickles_1.zip  <----- THIS FILE MUST BE EXTRACTED BEFORE RUNNING!
│       │   RF_pickles_2.zip  <----- THIS FILE MUST BE EXTRACTED BEFORE RUNNING!
│
└───Readings
```


# Dependencies
The project should run with any version of Python 3, although Python 3.7.7 was
used for this project. The following libraries must be installed for the project
to run properly.

* Keras 3.1.1
* Keras-tuner 1.0.2
* Keras-tcn 3.1.1
* Numpy 1.17.0
* Pandas 1.0.5
* Pyarrow 1.0.1
* Python-louvain 0.14
* Scikit-learn 0.23.1
* Seaborn 0.10.1
* Sklearn 0.21.1
* Tensorflow 2.3.1


# Collaborators
Team Among Us

Joshua Cayetano-Emond

Lorenzo Germini

Benoit Pahud
