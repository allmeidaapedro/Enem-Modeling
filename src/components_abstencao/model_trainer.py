'''
This script aims to train and save the selected final model from the modelling notebook.
'''

'''
Importing the libraries
'''

# File handling.
import os
from dataclasses import dataclass

# Debugging and verbose.
import sys
from src.exception import CustomException
from src.logger import logging

# Modelling.
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, classification_report

# Utils.
from src.artifacts_utils import save_object


@dataclass
class ModelTrainerConfig:
    '''
    Configuration class for model training.

    This data class holds configuration parameters related to model training. It includes attributes such as
    `model_file_path` that specifies the default path to save the trained model file.

    Attributes:
        model_file_path (str): The default file path for saving the trained model. By default, it is set to the
                              'artifacts' directory with the filename 'model_abstencao.pkl'.

    Example:
        config = ModelTrainerConfig()
        print(config.model_file_path)  # Output: 'artifacts/model_abstencao.pkl'

    Note:
        This class uses the @dataclass decorator to automatically generate special methods like __init__ and __repr__
        based on the defined attributes.
    '''

    model_file_path = os.path.join('artifacts', 'model_abstencao.pkl')


class ModelTrainer:
    '''
    This class is responsible for training and saving the best LightGBM model from modelling notebook.

    Attributes:
        model_trainer_config (ModelTrainerConfig): An instance of `ModelTrainerConfig` for configuration settings.

    Methods:
        apply_model_trainer(train_prepared, test_prepared):
            Trains the best LightGBM model using the provided prepared training, testing and validation data,
            and returns ROC AUC and classification report on the test set.

    '''

    def __init__(self) -> None:
        '''
        Initializes a new instance of the `ModelTrainer` class.

        Attributes:
            model_trainer_config (ModelTrainerConfig): An instance of `ModelTrainerConfig` for configuration settings.
        '''
        self.model_trainer_config = ModelTrainerConfig()
    
    
    def apply_model_trainer(self, train_prepared, test_prepared, val_prepared):
        '''
        Trains the best LightGBM model using the provided prepared training, testing and validation data, 
        the best hyperparameters found during the modelling notebook using validation set and bayesian 
        optimization and returns the ROC AUC score, and classification report on the test set.

        Args:
            train_prepared (pd.DataFrame): The prepared training data.
            test_prepared (pd.DataFrame): The prepared testing data.
            val_prepared (pd.DataFrame): The prepared validation data.

        Returns:
            float: The ROC AUC score and classification report of the best model on the test set.

        Raises:
            CustomException: If an error occurs during the training and evaluation process.
        '''

        try:
            logging.info('Split train, test and validation prepared sets.')
            
            X_train_prepared = train_prepared.drop(columns=['abstencao'])
            X_test_prepared = test_prepared.drop(columns=['abstencao'])
            X_val_prepared = val_prepared.drop(columns=['abstencao']) 

            y_train = train_prepared['abstencao'].copy()
            y_test = test_prepared['abstencao'].copy()
            y_val = val_prepared['abstencao'].copy()       
       
            X_train_prepared_full = pd.concat([X_train_prepared, X_val_prepared])
            y_train_full = pd.concat([y_train, y_val])

            logging.info('Train the best LightGBM model.')

            best_params = {
                            'objective': 'binary',
                            'metric': 'roc_auc',
                            'n_estimators': 1000,
                            'verbosity': -1,
                            'bagging_freq': 1,
                            'class_weight': 'balanced',
                            'learning_rate': 0.028095483425366566, 
                            'num_leaves': 167, 
                            'subsample': 0.802292150133624, 
                            'colsample_bytree': 0.9025697978357053, 
                            'min_data_in_leaf': 20
                          }
            
            best_model = LGBMClassifier(**best_params)

            best_model.fit(X_train_prepared_full, y_train_full)

            logging.info('Save the best model.')

            save_object(
                file_path=self.model_trainer_config.model_file_path,
                object=best_model
            )

            logging.info('Best model ROC AUC, and classification report on test set.')

            y_pred = best_model.predict(X_test_prepared)
            probas = best_model.predict_proba(X_test_prepared)[:, 1]

            roc_auc = roc_auc_score(y_test, probas)
            class_report = classification_report(y_test, y_pred)

            return roc_auc, class_report

        except Exception as e:
            raise CustomException(e, sys)