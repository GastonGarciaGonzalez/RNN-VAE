# -*- coding: utf-8 -*-
"""
Created on Mon Aug 2022

@author: gastong@fing.edu.uy
"""

import sys
import pandas as pd
import json
from rnn_vae import RNNVAE
from utils import set_index, preprocessing
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import optuna



def objective(trial):

    data_train_path = sys.argv[1]
    data_val_path = sys.argv[2]
    settings_path = sys.argv[3]
    
    # Data
    print('Reading the data_train...')
    data_train  = pd.read_csv(data_train_path)
    data_val  = pd.read_csv(data_val_path)
    # Parameters
    settings = json.load(open(settings_path, 'r'))

    # Preprocess
    print('Preprocessing the data_train...')
    sc = StandardScaler()
    df_X_train = set_index(data_train)
    df_X_train = preprocessing(df_X_train, settings['scale'], sc, settings['model_name'],
                        settings['wo_outliers'], settings['max_std'], 'fit')
    df_X_val = set_index(data_val)
    df_X_val = preprocessing(df_X_val, settings['scale'], sc, settings['model_name'],
                        settings['wo_outliers'], settings['max_std'], 'transform')


    # Model initialization
    model = RNNVAE(
        trial.suggest_categorical("length_seq", [32, 128, 512]),
        settings['M'],
        settings['layer_type'],
        [trial.suggest_categorical("units", [8, 16, 32])
         for i in range(trial.suggest_categorical("layers", [2, 4]))],
        trial.suggest_categorical("batch_size", [32, 64]),
        trial.suggest_categorical("J", [4, 2, 1]),
        settings['epochs'],
        trial.suggest_categorical("learning_rate", [1e-3, 1e-4]),
        settings['lr_decay'],
        settings['decay_rate'],
        settings['decay_step'],
        settings['model_name'],
        summary = settings['summary']
        )   

    # Train
    model.fit(df_X_train, settings['val_percent'], settings['seed'])

    # Evaluate
    loss, reconstruction, kl = model.evaluate(True, df_X_val)
    
    return loss

if __name__ == '__main__':

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))