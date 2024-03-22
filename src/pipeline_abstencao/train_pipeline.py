'''
This script aims to execute the entire training pipeline using the components. Specifically, data ingestion, transformation and model training.
'''

# Debugging and verbose.
import sys
from src.logger import logging
from src.exception import CustomException

# Components for data ingestion, transformation and model training.

# Data ingestion.
from src.components_abstencao.data_ingestion import DataIngestion

# Data transformation.
from src.components_abstencao.data_transformation import DataTransformation

# Model trainer.
from src.components_abstencao.model_trainer import ModelTrainer


class TrainPipeline:
    '''TrainPipeline class for training a machine learning pipeline.

    This class handles the training pipeline, including data ingestion,
    data transformation, and model training.

    Attributes:
        None

    Methods:
        __init__: Constructor method to initialize the class.
        train: Method to execute the training pipeline.

    '''

    def __init__(self) -> None:
        pass

    def train(self):
        '''Execute the training pipeline.

        This method performs the following steps:
        1. Data ingestion to obtain training, testing and validation datasets.
        2. Data transformation for preprocessing.
        3. Model training using the best hyperparameters.

        It also prints the ROC AUC and classification report.

        Raises:
            CustomException: If an error occurs during the training pipeline.

        '''

        try:
            logging.info('Start train pipeline - Absence Model.')

            logging.info('Start data ingestion.')

            data_ingestion = DataIngestion()
            train_path, test_path, val_path = data_ingestion.apply_data_ingestion()

            logging.info('Finish data ingestion. Train, test and validation entire sets obtained (artifacts paths).')

            logging.info('Start data transformation.')
            
            data_transformation = DataTransformation()
            train_prepared, test_prepared, val_prepared, _ = data_transformation.apply_data_transformation(train_path, test_path, val_path)

            logging.info('Finish data transformation. Train, test and validation entire prepared sets obtained (artifacts).')

            logging.info('Start model trainer.')
            
            model_trainer = ModelTrainer()

            roc_auc, class_report = model_trainer.apply_model_trainer(train_prepared, test_prepared, val_prepared)

            print(f'ROC AUC: {roc_auc}')
            print(class_report)

            logging.info('Finish model trainer. Final best model obtained (artifacts).')

        except Exception as e:
            raise CustomException(e, sys)
        

if __name__ == '__main__':

    # Apply full train pipeline using the components above.
    train_pipeline = TrainPipeline()
    train_pipeline.train()