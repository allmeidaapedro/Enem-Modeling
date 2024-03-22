'''
This script aims to apply data cleaning and preprocessing to the data.
'''

# Debugging and verbose.
import sys
from src.exception import CustomException
from src.logger import logging

# File handling.
import os

# Data manipulation.
import numpy as np
import pandas as pd
from dataclasses import dataclass

# Preprocessing.
from sklearn.pipeline import Pipeline
from feature_engine.selection import ProbeFeatureSelection
from src.modelling_utils import ClassificationFeatureEngineer, OneHotFeatureEncoder, TargetFeatureEncoder, OrdinalFeatureEncoder, ColumnDropper
from lightgbm import LGBMClassifier

# Utils.
from src.artifacts_utils import save_object


@dataclass
class DataTransformationConfig:
    '''
    Configuration class for data transformation.

    This data class holds configuration parameters related to data transformation. It includes attributes such as
    `preprocessor_file_path` that specifies the default path to save the preprocessor object file.

    Attributes:
        preprocessor_file_path (str): The default file path for saving the preprocessor object. By default, it is set
                                     to the 'artifacts' directory with the filename 'preprocessor_abstencao.pkl'.

    Example:
        config = DataTransformationConfig()
        print(config.preprocessor_file_path)  # Output: 'artifacts/preprocessor_abstencao.pkl'

    Note:
        This class uses the @dataclass decorator to automatically generate special methods like __init__ and __repr__
        based on the defined attributes.
    '''

    preprocessor_file_path = os.path.join('artifacts', 'preprocessor_abstencao.pkl')


class DataTransformation:
    '''
    Data transformation class for preprocessing and transformation of train, test and validation sets.

    This class handles the preprocessing and cleaning of datasets, including
    categorical encoding and feature scaling.

    :ivar data_transformation_config: Configuration instance for data transformation.
    :type data_transformation_config: DataTransformationConfig
    '''
    def __init__(self) -> None:
        '''
        Initialize the DataTransformation instance with a DataTransformationConfig.
        '''
        self.data_transformation_config = DataTransformationConfig()


    def get_preprocessor(self):
        '''
        Get a preprocessor for data transformation.

        This method sets up pipelines for ordinal encoding, one-hot encoding,
        and scaling of features.

        :return: Preprocessor object for data transformation.
        :rtype: ColumnTransformer
        :raises CustomException: If an exception occurs during the preprocessing setup.
        '''

        try:
            # Construct the preprocessor
            ordinal_encoding_orders = {
                                        'faixa_etaria': ['Adolescente (< 18)', 
                                                        'Jovem adulto (18-24)', 
                                                        'Adulto (25-45)', 
                                                        'Meia idade a idoso (46+)'
                                                        ],
                                        
                                        'status_conclusao_ensino_medio': ['Não concluído', 
                                                                        'Cursando', 
                                                                        'Último ano',
                                                                        'Concluído'
                                                                        ],
                                        
                                        'escolaridade_pai': ['Nunca estudou', 
                                                            'Não sei',
                                                            'Ensino fundamental incompleto', 
                                                            'Ensino fundamental completo', 
                                                            'Ensino médio completo', 
                                                            'Ensino superior completo',
                                                            'Pós-graduação'
                                                            ],
                                        
                                        'escolaridade_mae': ['Nunca estudou', 
                                                            'Não sei',
                                                            'Ensino fundamental incompleto', 
                                                            'Ensino fundamental completo', 
                                                            'Ensino médio completo', 
                                                            'Ensino superior completo',
                                                            'Pós-graduação'
                                                            ],
                                        
                                        'numero_pessoas_em_casa': ['1 a 3', 
                                                                '4 a 5', 
                                                                '6 a 10', 
                                                                '11 a 20'],
                                        
                                        'renda_familiar_mensal': ['Nenhuma Renda', 
                                                                'Renda baixa', 
                                                                'Renda média baixa', 
                                                                'Renda média alta', 
                                                                'Renda alta'],
                                        
                                        'possui_celular_em_casa': ['Não',
                                                                'Um',
                                                                'Dois ou mais'
                                                                ],  
                                        
                                        'possui_computador_em_casa': ['Não',
                                                                    'Um',
                                                                    'Dois ou mais'
                                                                    ]  
                                        }

            target_encoding_features = [
                                        'escola',
                                        'regiao',
                                        'estado_civil',
                                        'possui_celular_em_casa',
                                        'possui_computador_em_casa',
                                        ]

            one_hot_encoding_features = [
                                        'sexo',
                                        'lingua',
                                        'acesso_internet_em_casa'
                                        ]
            
            to_drop_features = [
                    'municipio_prova', 
                    'presenca_cn', 
                    'presenca_ch', 
                    'presenca_lc', 
                    'presenca_mt', 
                    'nota_cn', 
                    'nota_ch', 
                    'nota_lc', 
                    'nota_mt', 
                    'nota_comp1', 
                    'nota_comp2', 
                    'nota_comp3', 
                    'nota_comp4', 
                    'nota_comp5', 
                    'nota_redacao', 
                    'treineiro', 
                    'uf_prova', 
                    ]

            preprocessor = Pipeline(
                                    steps=[
                                        ('feature_engineer', ClassificationFeatureEngineer()),
                                        ('one_hot_encoder', OneHotFeatureEncoder(to_encode=one_hot_encoding_features)),
                                        ('ordinal_encoder', OrdinalFeatureEncoder(to_encode=ordinal_encoding_orders)),
                                        ('target_encoder', TargetFeatureEncoder(to_encode=target_encoding_features)),
                                        ('column_dropper', ColumnDropper(to_drop=to_drop_features)),
                                        ('probe_feature_selector', ProbeFeatureSelection(estimator=LGBMClassifier(),
                                                                                         variables=None,
                                                                                         scoring='neg_root_mean_squared_error',
                                                                                         n_probes=1,
                                                                                         distribution='normal',
                                                                                         cv=3,
                                                                                         random_state=42,
                                                                                         confirm_variables=False
                                                                                         ))
                                        ]
                                    )
            
            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
    
    
    def apply_data_transformation(self, train_path, test_path, val_path):
        '''
        Apply data transformation process.

        Reads, preprocesses, and transforms training, testing and validation datasets.

        :param train_path: Path to the training dataset CSV file.
        :param test_path: Path to the test dataset CSV file.
        :param val_path: Path to the validation dataset CSV file.
        :return: Prepared training, testing and validation datasets and the preprocessor file path.
        :rtype: tuple
        :raises CustomException: If an exception occurs during the data transformation process.
        '''
        
        try:

            logging.info('Read train, test and validation sets.')

            # Obtain train, test and validation entire sets from artifacts.
            train = pd.read_csv(train_path)
            test = pd.read_csv(test_path)
            val = pd.read_csv(val_path)

            logging.info('Obtain preprocessor object.')

            preprocessor = self.get_preprocessor()

            # Get train, test and validation predictor and target sets.
            X_train = train.drop(columns=['abstencao'])
            y_train = train['abstencao'].copy()

            X_test = test.drop(columns=['abstencao'])
            y_test = test['abstencao'].copy()

            X_val = val.drop(columns=['abstencao'])
            y_val = val['abstencao'].copy()

            logging.info('Preprocess train, test and validation sets.')

            X_train_prepared = preprocessor.fit_transform(X_train, y_train)
            X_test_prepared = preprocessor.transform(X_test)
            X_val_prepared = preprocessor.transform(X_val)

            # Get final train, test and validation entire prepared sets.

            train_prepared = pd.concat([X_train_prepared, y_train.reset_index(drop=True)], axis=1)
            test_prepared = pd.concat([X_test_prepared, y_test.reset_index(drop=True)], axis=1)
            val_prepared = pd.concat([X_val_prepared, y_val.reset_index(drop=True)], axis=1)

            logging.info('Entire train, test and validation sets prepared.')

            logging.info('Save preprocessing object.')

            save_object(
                file_path=self.data_transformation_config.preprocessor_file_path,
                object=preprocessor
            )
        
            return train_prepared, test_prepared, val_prepared, self.data_transformation_config.preprocessor_file_path
        
        except Exception as e:
            raise CustomException(e, sys)
        