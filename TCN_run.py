# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from tensorflow.keras import Input, Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from keras.utils import np_utils
from kerastuner import Hyperband, HyperModel
from tcn import TCN, tcn_full_summary
from LSTM_run import prepare_data


class MyHyperModel(HyperModel):
    """ MYHYPERMODEL Custom Hypermodel with vector shapes
        A custom hypermodel that includes the shapes of the vectors in its
        definition to create the layers with the right sizes.
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
        
    def build(self, hp):
         """ Build the custom HyperModel
             Define a  TCN model-building function.
             It takes an hp argument from which you can sample hyperparameters
         
         
         Parameters
         ----------
         hp : Hyperparameter list
            A list containing the hyperparameters to try.
            
            
         Returns
         -------
         model :  Hypermodel
         The custom hypermodel.
         
         """
         
         
         # select the parameters to tune
         # common values
         dropout = hp.Float('dropout_rate', default=0.0,
                            min_value=0.0, max_value=0.7, step=0.1)
         if self.X_timesteps < 50:
             # choice for models for 2 or 10 days in memory.
             filters = hp.Int('nb_filters', default=64,
                              min_value=8, max_value=64, step=8)
             kern_sz = hp.Int('kernel_size', default=2,
                                min_value=2, max_value=7, step=1)
         else:
             # choice for models with 50, 100 or 150 days in memory.
             filters = hp.Int('nb_filters', default=64,
                              min_value=64, max_value=64)
             kern_sz = hp.Int('kernel_size', min_value=2, max_value=3, step=1)
         
         # setup the input layer
         i = Input(shape=(self.X_timesteps, self.X_features))
         
         
         # build the model with different hyperparameters choice
         tcn_layer = TCN(nb_filters=filters,
                         kernel_size=kern_sz,
                         nb_stacks=1,
                         dilations=(1, 2, 4, 8, 16,32,64),
                         padding='causal',
                         dropout_rate=dropout)
         
         #the tcn layers are here
         o = tcn_layer(i)
         
         #build the dense layer at the end
         o = Dense(self.Y_shape, activation='softmax')(o)
         model = Model(inputs=[i], outputs=[o])
         
         #compile the model
         model.compile('adam','categorical_crossentropy', metrics=['acc'])
         
         #show receptive field and ckeck if you have full history coverage
         print(tcn_layer.receptive_field)
         if tcn_layer.receptive_field >= self.X_timesteps:
              print('full history coverage assured')
         
         #detailed summary with the 5 residual blocks
         tcn_full_summary(model, expand_residual_blocks=False)
         
         return model  
     
     
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
        epoch_num = 25
        
        #define the tcn model
        hypermodel = MyHyperModel(X_train.shape[1],
                                  X_train.shape[2],
                                  Y_train.shape[1])
        
        #define the optimizer
        tuner = Hyperband(hypermodel,
                          objective='val_acc',
                          max_epochs=epoch_num,
                          factor=3, directory='./Models/TCN/',
                          project_name='tuning_{}mem'.format(history_window),
                          overwrite=True)
        
        # perform the hyperparameter optimization
        tuner.search_space_summary()
         
        # perform the tuning of the parameters on the model
        # and then return the output of the tuned model
        tuner.search(X_train,
                     Y_train,
                     epochs=epoch_num,
                     validation_data=(X_test, Y_test))
        # tuner.search(X_train,
        #              Y_train,
        #              epochs=30,
        #              validation_data=(X_test, Y_test),
        #              callbacks=[tf.keras.callbacks.EarlyStopping(patience=1)])
         
        tuner.results_summary()
         
        #get best hyperparameters
        best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]
         
        # build the best model
        best_model = hypermodel.build(best_hps)
         
        #fit data to model
        best_model.fit(X_train, Y_train, epochs=epoch_num)
         
        # print the best hyperparameters. done here after the fitting
        # so it remains on the console and doesn't get lost
        print(f"""
        The hyperparameter search is complete.\n
        Number of filters: {best_hps.get('nb_filters')}\n
        Kernel size: {best_hps.get('kernel_size')}\n
        Dropout rate: {best_hps.get('dropout_rate')}\n
        """)
         
        # save the model
        best_model.save(filepath)
         
         
    else:
        #don't perform hyperparameter tuning, just directly load best model
        best_model = load_model(filepath, custom_objects={'TCN': TCN})
         
    # predict
    prediction = best_model.predict(X_test, verbose=0)
    
    # un-encode the y data
    y_pred = np.argmax(prediction,axis=1)
    Y_test_val = np.argmax(Y_test,axis=1)

    return y_pred, Y_test_val
  
    
if __name__ == "__main__":
    y_pred, Y_test_val = tcn_predict(True, 150)