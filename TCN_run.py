# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Dropout
from keras.utils import np_utils
from kerastuner.tuners import RandomSearch
from tcn import TCN, tcn_full_summary
import Helpers.p_indicators as p_ind
import LSTM.prepare_data as p_d

if __name__ == "__main__":
    # if code ran standalone, perform LSTM
    
    # filenames
    #in_file_path = './Data/'
    in_file_path = './Data/preprocessed.csv'
    out_file_path = './Data/cleaned_data.csv'
    # load the data
    df = pd.read_csv(in_file_path,index_col=0,parse_dates=True)    
    X_train, Y_train, X_test, Y_test=p_d(df, memory, valid_ratio=0.8)

    
#define a  TCN model-building function. It takes an hp argument from which you can sample hyperparameters
def tcn_predict(hyperparam_opt, history_window):
    if hyperparam_opt==True:
    # perform the tuning of teh parameters on the model and then return the output of the tuned model
        def build_model(hp,history_window):
            i = Input(shape=(history_window, 1))
            o = TCN(nb_filters = hp.Int('nb_filters', min_value=32, max_value=512, step=32), kernel_size=hp.Int('kernel', min_value=2, max_value=10, step=1), nb_stacks=1, dilations=[1,2,4,8,16,32], padding='causal')(i
            o = Dense(Y_train.shape[1], activation='softmax')(o)
            model = Model(inputs=[i], outputs=[o])
            #model.compile(optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])), 'categorical_crossentropy', metrics=['accuracy'])
            model.compile('adam','categorical_crossentropy', metrics=['accuracy'])
            tcn_full_summary(model, expand_residual_blocks=True)
            return model
       
        #instiatate a tuner on a given model with the objective of optimizing val_accuracy usig random search/hyperband more efficient 
        #def tune_model(build_model,):
        #Hyperband version
        #tuner = Hyperband(build_model,objective='val_accuracy', max_epochs=30, hyperband_iterations=2, executions_per_trial=3, directory='C:/Users/loren/OneDrive/Documenti/my_dir/ggg')

        #tuner.search_space_summary()

        #tuner.search(X_train,Y_train,epochs=30,validation_data=(X_test, Y_test),callbacks=[tf.keras.callbacks.EarlyStopping(patience=1)])
        #best_model = tuner.get_best_models(1)[0]
        #best_hyperparameters = tuner.get_best_hyperparameters(1)[0]
        #tuner.results_summary()
        #still have to adjust the parameter settings and some spaces :)
        #tuner = RandomSearch(build_model,objective='val_accuracy',max_trials=5,executions_per_trial=3,directory='..')
        #get a search space summary
        #tuner.search_space_summary()
        #start the tuning process with random search
        tuner.search(X_train,Y_train,epochs=5,validation_data=(X_test, Y_test))
        
        tuner.results_summary()
        #best_model = tuner.get_best_models(num_models=2)
        
        #run model
        best_model.fit(X_train, Y_train, epochs=30)
        
        #predict
        prediction = best_model.predict(X_test, verbose=0)
        
        #unencode the one hot encoded labels
        y_pred = np.argmax(prediction, axis=1)
        y_true = np.argmax(Y_test,axis=1)
        #return predicted labels
        return y_pred, y_true
        
    elif hyperparam_opt==False:
       #directly return the otuput of the tuned model
       def build_already_tuned_model(history_window):
           i = Input(shape=(history_window, 1))
           o = TCN(nb_filters = 6, kernel_size=9, nb_stacks=2, dilations=[2 ** i for i in range(10)], padding='causal', dropout_rate=0.5, activation='softmax', kernel_initializer='he_normal' )(i)
           o = Dense(Y_train.shape[1], activation='softmax')(o)
           model = Model(inputs=[i], outputs=[o])
           model.compile('adam','categorical_crossentropy', metrics=['accuracy'])
           tcn_full_summary(model, expand_residual_blocks=True)
           return model
           
       #run model
       model.fit(X_train, Y_train, epochs=5)
        
       #predict
       prediction = model.predict(X_test, verbose=0)
        
       #unencode the one hot encoded labels
       y_pred = np.argmax(prediction, axis=1)
       y_true = np.argmax(Y_test,axis=1)
       #return predicted labels
       return y_pred, y_true           
       
       
           
          
               
               
               
