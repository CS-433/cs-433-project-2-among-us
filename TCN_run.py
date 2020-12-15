# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from tensorflow.keras import Input, Model
from keras.models import load_model
from tensorflow.keras.layers import Dense
from keras.utils import np_utils
from kerastuner import Hyperband, HyperModel
from keras.optimizers import Adam
from tcn import TCN, tcn_full_summary


class MyHyperModel(HyperModel):
    
    """MYHYPERMODEL
     """
    def __init__(self, X_timesteps, X_features, Y_shape):
        """ __INIT__ Creates the hypermodel
        
        Parameters
        ----------
        X_timesteps : int
            Number of timesteps in the X dataset given.
        X_features : int
            Number of features in the X dataset given.
        Y_shape : int
            dimensions of the Y vector. Corresponds to number of classes
        Returns
        -------
        None.
        """
        self.X_timesteps = X_timesteps
        self.X_features = X_features
        self.Y_shape = Y_shape
     #define a  TCN model-building function. It takes an hp argument from which you can sample hyperparameters
     
     # perform the tuning of teh parameters on the model and then return the output of the tuned model
    def build(self, hp):
         """Build the custom HyperModel
         
         
         Parameters
         ----------
         hp : Hyperparameter list
            A list containing the hyperparameters to try.
            
            
         Returns
         -------
         model :  Hypermodel
         The custom hypermodel.
         
         """
         # build the model with different hyperparameters choice
         i = Input(shape=(self.X_timesteps, 1))
         
         tcn_layer = TCN(kernel_size=3, nb_stacks=1, dilations=(1, 2, 4, 8, 16,32), padding='causal',dropout_rate=hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1))
         
         #the tcn layers are here
         o = tcn_layer(i)
         
         #build the dense layer at the end
         o = Dense(self.Y_shape, activation='softmax')(o)
         model = Model(inputs=[i], outputs=[o])
         
         #compile the model
         model.compile('adam','categorical_crossentropy', metrics=['acc'])
         
         #show receptive field and ckeck if you have full history coverage
         print(tcn_layer.receptive_field)
         if tcn_layer.receptive_field>=self.X_timesteps:
              print('full history coverage assured')
         
         
         #detailed summary with residual blocks expanded
         tcn_full_summary(model, expand_residual_blocks=False)
         
         return model  
     
        
     
def prepare_data(df, memory, valid_ratio=0.8, form='timestep'):
    """ PREPARE_DATA Puts the data in a format ready for keras LSTM
        Prepares the data found in the database df into a format ready for
        keras LSTM using the last 'memory' days as a moving window.
    
    Parameters
    ----------
    df : Pandas dataframe (N x (D+1))
        Dataframe to prepare data with N datapoints each having D+1 features
        The D + 1 features represent the last D day states and the day's state
    memory : int
        Number of days to take for the moving window
    valid_ratio : float
        The fraction of data to take for the training set
        Default is 0.8
    form : string, either 'timestep' or 'feature'
        The format to give the X array. Either the last 'memory' days are
        taken as a feature or as a timestep.
        Default is'timestep'
    Returns
    -------
    X_train : Numpy array (N x (memory * valid_ratio) x 1) or
                          (N x 1 x (memory * valid_ratio))
        An array representing the features/timesteps for the training data
    Y_train : Numpy array (N x 1)
        A numpy array representing the true labels for the training data
    X_test : Numpy array (N x (memory * (1 - valid_ratio)) x 1) or
                          (N x 1 x (memory * (1 - valid_ratio)))
        An array representing the features/timesteps for the testing data
    Y_test : Numpy array (N x 1)
        A numpy array representing the true labels for the testing data
        """
    
    df_numpy = df.to_numpy() # put the dataframe in numpy
    
         # find quantities for splitting
    N = len(df_numpy)
    n_train = int(N * valid_ratio)
    n_test = N - n_train
    n_classes = len(np.unique(df_numpy))
    
         # get X and Y data
    dataX = df_numpy[:,1:]
    dataY = df_numpy[:,0]
        
        # reshape X to be [samples, time steps, features]
    if form == 'feature':
        X_train = np.reshape(dataX[:n_train,:memory], (n_train, 1, memory))
        X_test = np.reshape(dataX[n_train:n_test,:memory], (n_test, 1, memory))
    else: # default to timestep, even if form was wrongly defined
        X_train = np.reshape(dataX[:n_train,:memory], (n_train, memory, 1))
        X_test = np.reshape(dataX[n_train:,:memory], (n_test, memory, 1))
        
        # normalize
        X_train = X_train / float(n_classes)
        X_test = X_test / float(n_classes)
    
        # one hot encode the output variable
        Y_encoded = np_utils.to_categorical(dataY)
        Y_train = Y_encoded[:n_train]
        Y_test = Y_encoded[n_train:]
    
        return X_train, Y_train, X_test, Y_test
                
         
def tcn_predict(hyperparam_opt, history_window):
         """ TCN_PREDICT Performs an TCN prediction on the given data
        Performs an TCN prediction with or without hyperparameter optimization
        Returns the predictions and the test data that was used
    
        Parameters
        ----------
        hyperparam_opt : boolean
            Whether or not to perform hyperparameter optimization.
        history_window : int
            The number of days to take in memory for the prediction.
        Returns
        -------
        y_pred : numpy array
            The predicted labels of the test data.
        Y_test_val : numpy array
            The true labels of the test data..
        """
        
        # define input data path
         in_file_path = './Data/preprocessed.csv'
        # define the model path
         filepath="./Models/TCN/model-tcn-{}mem.hdf5".format(history_window)
        
        # load the data
         df = pd.read_csv(in_file_path,index_col=0)
        # shape the data properly
         X_train, Y_train, X_test, Y_test = prepare_data(df, history_window)
        
         if hyperparam_opt:
             #perform hyperparam optimization
             epoch_num = 3
            
            #define the tcn model
             hypermodel = MyHyperModel(X_train.shape[1], X_train.shape[2], Y_train.shape[1])
            
             #define the optimizer
             tuner = Hyperband(hypermodel, objective='val_acc', max_epochs=epoch_num, factor=3, directory='./Models/TCN/', project_name='tuning_{}mem'.format(history_window), overwrite=True)
             # perform the hyperparameter optimization
             
             tuner.search_space_summary()
             
             tuner.search(X_train,Y_train,epochs=epoch_num,validation_data=(X_test, Y_test))
             #tuner.search(X_train,Y_train,epochs=30,validation_data=(X_test, Y_test),callbacks=[tf.keras.callbacks.EarlyStopping(patience=1)])
             
             tuner.results_summary()
             
             #get best hyperparameters
             best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]
             
             # build the best model
             best_model = hypermodel.build(best_hps)
             
             #fit data to model
             best_model.fit(X_train, Y_train, epochs=epoch_num)
             
             # print the best hyperparameters. done here after the fitting
             # so it remains on the console and doesn't get lost
             # print(f"""
             # The hyperparameter search is complete.\n
             # : {best_hps.get('units')}\n
             # Learn rate: {best_hps.get('learning_rate')}\n
             # Activation function: {best_hps.get('activation')}\n
             # Number of layers: {best_hps.get('num_layers')}\n
             # """)
             
             # save the model
             best_model.save(filepath)
             
             
         else:
             #don't perform hyperparameter tuning, just directly load best model
             best_model = load_model(filepath)
             
        #predict
         prediction = best_model.predict(X_test, verbose=0)
        
        # un-encode the y data
         y_pred = np.argmax(prediction,axis=1)
         Y_test_val = np.argmax(Y_test,axis=1)
    
         return y_pred, Y_test_val
  
if __name__ == "__main__":
    tcn_predict(True, 10)

           
          
               
               
               
