'''
This script aims to provide functions that will turn the modelling process easier
'''

'''
Importing libraries
'''

# Data manipulation and visualization.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Modelling.
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder, TargetEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve, auc, brier_score_loss
import time
import math

# Debugging.
from src.exception import CustomException
import sys

# Warnings.
from warnings import filterwarnings
filterwarnings('ignore')

# Definições de cores -> todas estão numa escala de mais escura para mais clara.
VERMELHO_FORTE = '#461220'
CINZA1, CINZA2, CINZA3 = '#231F20', '#414040', '#555655'
CINZA4, CINZA5, CINZA6 = '#646369', '#76787B', '#828282'
CINZA7, CINZA8, CINZA9 = '#929497', '#A6A6A5', '#BFBEBE'
AZUL1, AZUL2, AZUL3, AZUL4 = '#174A7E', '#4A81BF', '#94B2D7', '#94AFC5'
VERMELHO1, VERMELHO2, VERMELHO3, VERMELHO4, VERMELHO5 = '#DB0527', '#E23652', '#ED8293', '#F4B4BE', '#FBE6E9'
VERDE1, VERDE2 = '#0C8040', '#9ABB59'
LARANJA1 = '#F79747'
AMARELO1, AMARELO2, AMARELO3, AMARELO4, AMARELO5 = '#FFC700', '#FFCC19', '#FFEB51', '#FFE37F', '#FFEEB2'
BRANCO = '#FFFFFF'


class ColumnDropper(BaseEstimator, TransformerMixin):
    '''
    A transformer class to drop specified columns from a DataFrame.

    Attributes:
        to_drop (list): A list of column names to be dropped.

    Methods:
        fit(X, y=None): Fit the transformer to the data. This method does nothing and is only provided to comply with the Scikit-learn API.
        transform(X): Transform the input DataFrame by dropping specified columns.
    '''

    def __init__(self, to_drop):
        '''
        Initialize the ColumnDropper transformer.

        Args:
            to_drop (list): A list of column names to be dropped.
        '''
        self.to_drop = to_drop

    def fit(self, X, y=None):
        '''
        Fit the transformer to the data.

        This method does nothing and is only provided to comply with the Scikit-learn API.

        Args:
            X (pandas.DataFrame): Input features.
            y (array-like, default=None): Target labels. Ignored.

        Returns:
            self: Returns an instance of self.
        '''
        return self

    def transform(self, X):
        '''
        Transform the input DataFrame by dropping specified columns.

        Args:
            X (pandas.DataFrame): Input features.

        Returns:
            pandas.DataFrame: Transformed DataFrame after dropping specified columns.
        '''
        # Certify that only present columns will be dropped.
        self.to_drop = [col for col in self.to_drop if col in X.columns]
        
        # Drop the specified columns.
        return X.drop(columns=self.to_drop)
    

class RegressionFeatureEngineer(BaseEstimator, TransformerMixin):
    '''
    A transformer class for performing feature engineering on performance-related data.

    Methods:
        fit(X, y=None): Fit the transformer to the data. This method does nothing and is only provided to comply with the Scikit-learn API.
        transform(X): Transform the input DataFrame by engineering performance-related features.
    '''

    def fit(self, X, y=None):
        '''
        Fit the transformer to the data.

        This method does nothing and is only provided to comply with the Scikit-learn API.

        Args:
            X (pandas.DataFrame): Input features.
            y (array-like, default=None): Target labels. Ignored.

        Returns:
            self: Returns an instance of self.
        '''
        return self

    def transform(self, X):
        '''
        Transform the input DataFrame by engineering performance-related features.

        Args:
            X (pandas.DataFrame): Input features.

        Returns:
            pandas.DataFrame: Transformed DataFrame after engineering performance-related features.
        '''
        X_copy = X.copy()

        # Create region variable.
        region_mapping = {
            'AC': 'Norte',
            'AL': 'Nordeste',
            'AP': 'Norte',
            'AM': 'Norte',
            'BA': 'Nordeste',
            'CE': 'Nordeste',
            'DF': 'Centro-Oeste',
            'ES': 'Sudeste',
            'GO': 'Centro-Oeste',
            'MA': 'Nordeste',
            'MT': 'Centro-Oeste',
            'MS': 'Centro-Oeste',
            'MG': 'Sudeste',
            'PA': 'Norte',
            'PB': 'Nordeste',
            'PR': 'Sul',
            'PE': 'Nordeste',
            'PI': 'Nordeste',
            'RJ': 'Sudeste',
            'RN': 'Nordeste',
            'RS': 'Sul',
            'RO': 'Norte',
            'RR': 'Norte',
            'SC': 'Sul',
            'SP': 'Sudeste',
            'SE': 'Nordeste',
            'TO': 'Norte'
        }
        X_copy['regiao'] = X_copy['uf_prova'].replace(region_mapping)

        # Create technology access variable.
        X_copy['acesso_internet_ordinal'] = X_copy['acesso_internet_em_casa'].replace({'Sim': 1, 'Não': 0}).astype('int8')
        X_copy['possui_celular_ordinal'] = X_copy['possui_celular_em_casa'].replace({'Dois ou mais': 2, 'Um': 1, 'Não': 0}).astype('int8')
        X_copy['possui_computador_ordinal'] = X_copy['possui_computador_em_casa'].replace({'Dois ou mais': 2, 'Um': 1, 'Não': 0}).astype('int8')
        X_copy['acesso_tecnologico'] = X_copy['acesso_internet_ordinal'] + \
                                       X_copy['possui_celular_ordinal'] + \
                                       X_copy['possui_computador_ordinal']
        X_copy = X_copy.drop(columns=['acesso_internet_ordinal', 'possui_celular_ordinal', 'possui_computador_ordinal'])

        # Create income per person at home variable.
        income_mapping = {
            'Até R$ 1.212,00': 1212,
            'Nenhuma Renda': 0,
            'R$ 1.818,01 - R$ 3.030,00': (1818 + 3030) / 2,
            'R$ 1.212,01 - R$ 1.818,00': (1212 + 1818) / 2,
            'R$ 3.030,01 - R$ 4.848,00': (3030 + 4848) / 2,
            'R$ 4.848,01 - R$ 7.272,00': (4848 + 7272) / 2,
            'R$ 7.272,01 - R$ 10.908,00': (7272 + 10908) / 2,
            'Acima de R$ 24.240,00': 24240,
            'R$ 18.180,01 - R$ 24.240,00': (18180 + 24240) / 2,
            'R$ 10.908,01 - R$ 18.180,00': (10908 + 18180) / 2
        }

        X_copy['renda_numerica'] = X_copy['renda_familiar_mensal'].map(income_mapping).astype('int32')
        
        # Garantee numero_pessoas_em_casa is int.
        X_copy['numero_pessoas_em_casa'] = X_copy['numero_pessoas_em_casa'].astype('int32')
        
        X_copy['renda_por_pessoa'] = (X_copy['renda_numerica'] / X_copy['numero_pessoas_em_casa']).astype('float32')

        # Create income per technology access variable.
        X_copy['renda_por_acesso_tecnologico'] = (X_copy['renda_numerica'] / X_copy['acesso_tecnologico']).astype('float32')
        X_copy['renda_por_acesso_tecnologico'] = X_copy['renda_por_acesso_tecnologico'].replace({np.inf: 0,
                                                                                                 np.nan: 0})
        X_copy = X_copy.drop(columns=['renda_numerica'])

        # Create technology access per numer of people at home variable.
        X_copy['acesso_tecnologico_por_pessoa'] = (X_copy['acesso_tecnologico'] / X_copy['numero_pessoas_em_casa']).astype('float32')

        # Discretize number of people at home.
        X_copy['numero_pessoas_em_casa'] = pd.cut(X_copy['numero_pessoas_em_casa'],
                                                  bins=[1, 3, 5, 10, 20],
                                                  labels=['1 a 3', '4 a 5', '6 a 10', '11 a 20'],
                                                  include_lowest=True)

        # Combine similar target distributed low proportion income categories.
        income_mapping = {
            'Até R$ 1.212,00': 'Renda baixa',
            'R$ 1.212,01 - R$ 1.818,00': 'Renda baixa',
            'R$ 1.818,01 - R$ 3.030,00': 'Renda média baixa',
            'R$ 3.030,01 - R$ 4.848,00': 'Renda média baixa',
            'R$ 4.848,01 - R$ 7.272,00': 'Renda média alta',
            'R$ 7.272,01 - R$ 10.908,00': 'Renda média alta',
            'R$ 10.908,01 - R$ 18.180,00': 'Renda alta',
            'R$ 18.180,01 - R$ 24.240,00': 'Renda alta',
            'Acima de R$ 24.240,00': 'Renda alta'
        }
        X_copy['renda_familiar_mensal'] = X_copy['renda_familiar_mensal'].replace(income_mapping)

        # Combine similar target distributed low proportion age categories.
        age_mapping = {
            'Adulto de meia idade (36-45)': 'Adulto a meia idade (36-55)',
            'Meia idade (46-55)': 'Adulto a meia idade (36-55)',
            'Pré aposentadoria (56-65)': 'Pré aposentadoria a idoso (> 56)',
            'Idoso (> 66)': 'Pré aposentadoria a idoso (> 56)'
        }
        X_copy['faixa_etaria'] = X_copy['faixa_etaria'].replace(age_mapping)

        return X_copy
    

class ClassificationFeatureEngineer(BaseEstimator, TransformerMixin):
    '''
    A transformer class for performing feature engineering on absence-related data.

    Methods:
        fit(X, y=None): Fit the transformer to the data. This method does nothing and is only provided to comply with the Scikit-learn API.
        transform(X): Transform the input DataFrame by engineering absence-related features.
    '''

    def fit(self, X, y=None):
        '''
        Fit the transformer to the data.

        This method does nothing and is only provided to comply with the Scikit-learn API.

        Args:
            X (pandas.DataFrame): Input features.
            y (array-like, default=None): Target labels. Ignored.

        Returns:
            self: Returns an instance of self.
        '''
        return self

    def transform(self, X):
        '''
        Transform the input DataFrame by engineering absence-related features.

        Args:
            X (pandas.DataFrame): Input features.

        Returns:
            pandas.DataFrame: Transformed DataFrame after engineering absence-related features.
        '''
        X_copy = X.copy()

        # Create region variable.
        region_mapping = {
            'AC': 'Norte',
            'AL': 'Nordeste',
            'AP': 'Norte',
            'AM': 'Norte',
            'BA': 'Nordeste',
            'CE': 'Nordeste',
            'DF': 'Centro-Oeste',
            'ES': 'Sudeste',
            'GO': 'Centro-Oeste',
            'MA': 'Nordeste',
            'MT': 'Centro-Oeste',
            'MS': 'Centro-Oeste',
            'MG': 'Sudeste',
            'PA': 'Norte',
            'PB': 'Nordeste',
            'PR': 'Sul',
            'PE': 'Nordeste',
            'PI': 'Nordeste',
            'RJ': 'Sudeste',
            'RN': 'Nordeste',
            'RS': 'Sul',
            'RO': 'Norte',
            'RR': 'Norte',
            'SC': 'Sul',
            'SP': 'Sudeste',
            'SE': 'Nordeste',
            'TO': 'Norte'
        }
        X_copy['regiao'] = X_copy['uf_prova'].replace(region_mapping)

        # Create technology access variable.
        X_copy['acesso_internet_ordinal'] = X_copy['acesso_internet_em_casa'].replace({'Sim': 1, 'Não': 0}).astype('int8')
        X_copy['possui_celular_ordinal'] = X_copy['possui_celular_em_casa'].replace({'Dois ou mais': 2, 'Um': 1, 'Não': 0}).astype('int8')
        X_copy['possui_computador_ordinal'] = X_copy['possui_computador_em_casa'].replace({'Dois ou mais': 2, 'Um': 1, 'Não': 0}).astype('int8')
        X_copy['acesso_tecnologico'] = X_copy['acesso_internet_ordinal'] + \
                                       X_copy['possui_celular_ordinal'] + \
                                       X_copy['possui_computador_ordinal']
        X_copy = X_copy.drop(columns=['acesso_internet_ordinal', 'possui_celular_ordinal', 'possui_computador_ordinal'])

        # Create income per person at home variable.
        income_mapping = {
            'Até R$ 1.212,00': 1212,
            'Nenhuma Renda': 0,
            'R$ 1.818,01 - R$ 3.030,00': (1818 + 3030) / 2,
            'R$ 1.212,01 - R$ 1.818,00': (1212 + 1818) / 2,
            'R$ 3.030,01 - R$ 4.848,00': (3030 + 4848) / 2,
            'R$ 4.848,01 - R$ 7.272,00': (4848 + 7272) / 2,
            'R$ 7.272,01 - R$ 10.908,00': (7272 + 10908) / 2,
            'Acima de R$ 24.240,00': 24240,
            'R$ 18.180,01 - R$ 24.240,00': (18180 + 24240) / 2,
            'R$ 10.908,01 - R$ 18.180,00': (10908 + 18180) / 2
        }

        X_copy['renda_numerica'] = X_copy['renda_familiar_mensal'].map(income_mapping).astype('int32')

        # Garantee numero_pessoas_em_casa is int.
        X_copy['numero_pessoas_em_casa'] = X_copy['numero_pessoas_em_casa'].astype('int32')

        X_copy['renda_por_pessoa'] = (X_copy['renda_numerica'] / X_copy['numero_pessoas_em_casa']).astype('float32')

        # Create income per technology access variable.
        X_copy['renda_por_acesso_tecnologico'] = (X_copy['renda_numerica'] / X_copy['acesso_tecnologico']).astype('float32')
        X_copy['renda_por_acesso_tecnologico'] = X_copy['renda_por_acesso_tecnologico'].replace({np.inf: 0,
                                                                                                 np.nan: 0})
        X_copy = X_copy.drop(columns=['renda_numerica'])

        # Create technology access per numer of people at home variable.
        X_copy['acesso_tecnologico_por_pessoa'] = (X_copy['acesso_tecnologico'] / X_copy['numero_pessoas_em_casa']).astype('float32')

        # Discretize number of people at home.
        X_copy['numero_pessoas_em_casa'] = pd.cut(X_copy['numero_pessoas_em_casa'],
                                                  bins=[1, 3, 5, 10, 20],
                                                  labels=['1 a 3', '4 a 5', '6 a 10', '11 a 20'],
                                                  include_lowest=True)

        # Combine similar target distributed low proportion income categories.
        income_mapping = {
            'Até R$ 1.212,00': 'Renda baixa',
            'R$ 1.212,01 - R$ 1.818,00': 'Renda baixa',
            'R$ 1.818,01 - R$ 3.030,00': 'Renda média baixa',
            'R$ 3.030,01 - R$ 4.848,00': 'Renda média baixa',
            'R$ 4.848,01 - R$ 7.272,00': 'Renda média alta',
            'R$ 7.272,01 - R$ 10.908,00': 'Renda média alta',
            'R$ 10.908,01 - R$ 18.180,00': 'Renda alta',
            'R$ 18.180,01 - R$ 24.240,00': 'Renda alta',
            'Acima de R$ 24.240,00': 'Renda alta'
        }
        X_copy['renda_familiar_mensal'] = X_copy['renda_familiar_mensal'].replace(income_mapping)

        # Combine similar target distributed low proportion age categories.
        age_mapping = {
            'Adulto jovem (25-35)': 'Adulto (25-45)', 
            'Adulto de meia idade (36-45)': 'Adulto (25-45)',
            'Meia idade (46-55)': 'Meia idade a idoso (46+)',
            'Pré aposentadoria (56-65)': 'Meia idade a idoso (46+)',
            'Idoso (> 66)': 'Meia idade a idoso (46+)',
        }
        
        X_copy['faixa_etaria'] = X_copy['faixa_etaria'].replace(age_mapping)

        return X_copy


class OneHotFeatureEncoder(BaseEstimator, TransformerMixin):
    '''
    A transformer class for one-hot encoding specified categorical variables.

    Attributes:
        to_encode (list): A list of column names to be one-hot encoded.

    Methods:
        fit(X, y=None): Fit the transformer to the data.
        transform(X): Transform the input DataFrame by one-hot encoding specified columns.
    '''

    def __init__(self, to_encode):
        '''
        Initialize the OneHotFeatureEncoder transformer.

        Args:
            to_encode (list): A list of column names to be one-hot encoded.
        '''
        self.to_encode = to_encode
        self.encoder = OneHotEncoder(drop='first',
                                     sparse_output=False,
                                     dtype=np.int8,
                                     handle_unknown='ignore',
                                     feature_name_combiner='concat')

    def fit(self, X, y=None):
        '''
        Fit the transformer to the data.

        Args:
            X (pandas.DataFrame): Input features.
            y (array-like, default=None): Target labels. Ignored.

        Returns:
            self: Returns an instance of self.
        '''
        self.encoder.fit(X[self.to_encode])
        return self

    def transform(self, X):
        '''
        Transform the input DataFrame by one-hot encoding specified columns.

        Args:
            X (pandas.DataFrame): Input features.

        Returns:
            pandas.DataFrame: Transformed DataFrame after one-hot encoding specified columns.
        '''
        # One-hot encode the columns.
        X_one_hot = self.encoder.transform(X[self.to_encode])

        # Create a dataframe for the one-hot encoded data.
        one_hot_df = pd.DataFrame(X_one_hot,
                                  columns=self.encoder.get_feature_names_out(self.to_encode))

        # Reset for mapping and concatenate constructing a final dataframe of features.
        X_reset = X.reset_index(drop=True)

        return pd.concat([X_reset.drop(columns=self.to_encode), one_hot_df], axis=1)
    

class StandardFeatureScaler(BaseEstimator, TransformerMixin):
    '''
    A transformer class for standard scaling specified numerical features and retaining feature names.

    Attributes:
        to_scale (list): A list of column names to be scaled.

    Methods:
        fit(X, y=None): Fit the transformer to the data.
        transform(X): Transform the input DataFrame by standard scaling specified columns and retaining feature names.
    '''
    def __init__(self, to_scale):
        '''
        Initialize the StandardFeatureScaler transformer.

        Args:
            to_scale (list): A list of column names to be scaled.
        '''
        self.to_scale = to_scale
        self.scaler = StandardScaler()
        
    def fit(self, X, y=None):
        '''
        Fit the transformer to the data.

        Args:
            X (pandas.DataFrame): Input features.
            y (array-like, default=None): Target labels. Ignored.

        Returns:
            self: Returns an instance of self.
        '''
        self.scaler.fit(X[self.to_scale])
        return self

    def transform(self, X):
        '''
        Transform the input DataFrame by standard scaling specified columns and retaining feature names.

        Args:
            X (pandas.DataFrame): Input features.

        Returns:
            pandas.DataFrame: Transformed DataFrame after standard scaling specified columns and retaining feature names.
        '''
        # Scale the columns.
        X_scaled = self.scaler.transform(X[self.to_scale])
        
        # Create a dataframe for the scaled data.
        scaled_df = pd.DataFrame(X_scaled,
                                 columns=self.scaler.get_feature_names_out(self.to_scale))
        
        # Reset for mapping and concatenated constructing a final dataframe of features.
        X_reset = X.reset_index(drop=True)
        
        return pd.concat([X_reset.drop(columns=self.to_scale), scaled_df], axis=1)
    
    
    
class OrdinalFeatureEncoder(BaseEstimator, TransformerMixin):
    '''
    A transformer class for ordinal encoding specified categorical features and retaining feature names.

    Attributes:
        to_encode (dict): A dictionary where keys are column names and values are lists representing the desired category orders.

    Methods:
        fit(X, y=None): Fit the transformer to the data.
        transform(X): Transform the input DataFrame by ordinal encoding specified columns and retaining feature names.
    '''
    def __init__(self, to_encode):
        '''
        Initialize the OrdinalFeatureEncoder transformer.

        Args:
            to_encode (dict): A dictionary where keys are column names and values are lists representing the desired category orders.
        '''
        self.to_encode = to_encode
        self.encoder = OrdinalEncoder(dtype=np.int8, 
                                      categories=[to_encode[col] for col in to_encode])

    def fit(self, X, y=None):
        '''
        Fit the transformer to the data.

        Args:
            X (pandas.DataFrame): Input features.
            y (array-like, default=None): Target labels. Ignored.

        Returns:
            self: Returns an instance of self.
        '''
        self.encoder.fit(X[list(self.to_encode.keys())])
        return self

    def transform(self, X):
        '''
        Transform the input DataFrame by ordinal encoding specified columns and retaining feature names.

        Args:
            X (pandas.DataFrame): Input features.

        Returns:
            pandas.DataFrame: Transformed DataFrame after ordinal encoding specified columns and retaining feature names.
        '''
        # Ordinal encode the columns.
        X_ordinal = self.encoder.transform(X[list(self.to_encode.keys())])
        
        # Create a dataframe for the ordinal encoded data.
        ordinal_encoded_df = pd.DataFrame(X_ordinal,
                                          columns=self.encoder.get_feature_names_out(list(self.to_encode.keys())))
        
        # Reset for mapping and concatenated constructing a final dataframe of features.
        X_reset = X.reset_index(drop=True)
        
        return pd.concat([X_reset.drop(columns=list(self.to_encode.keys())), ordinal_encoded_df], axis=1)
    

class TargetFeatureEncoder(BaseEstimator, TransformerMixin):
    '''
    A transformer class for target encoding specified categorical variables.

    Attributes:
        to_encode (list): A list of column names to be target encoded.

    Methods:
        fit(X, y=None): Fit the transformer to the data.
        transform(X): Transform the input DataFrame by target encoding specified columns.
    '''

    def __init__(self, to_encode):
        '''
        Initialize the TargetFeatureEncoder transformer.

        Args:
            to_encode (list): A list of column names to be target encoded.
        '''
        self.to_encode = to_encode
        self.encoder = TargetEncoder()

    def fit(self, X, y):
        '''
        Fit the transformer to the data.

        Args:
            X (pandas.DataFrame): Input features.
            y (array-like): Target labels.

        Returns:
            self: Returns an instance of self.
        '''
        self.encoder.fit(X[self.to_encode], y)
        return self

    def transform(self, X):
        '''
        Transform the input DataFrame by target encoding specified columns.

        Args:
            X (pandas.DataFrame): Input features.

        Returns:
            pandas.DataFrame: Transformed DataFrame after target encoding specified columns.
        '''
        # Target encode the columns.
        X_target = self.encoder.transform(X[self.to_encode])

        # Create a dataframe for the target encoded data.
        target_df = pd.DataFrame(X_target,
                                 columns=self.encoder.get_feature_names_out(self.to_encode))

        # Reset for mapping and concatenate constructing a final dataframe of features.
        X_reset = X.reset_index(drop=True)

        return pd.concat([X_reset.drop(columns=self.to_encode), target_df], axis=1)


# Regression modelling.

def regression_kfold_cv(models, X_train, y_train, n_folds=5):
    '''
    Evaluate multiple machine learning models using k-fold cross-validation.

    This function evaluates a dictionary of machine learning models by training each model on the provided training data
    and evaluating their performance using k-fold cross-validation. The evaluation metric used is RMSE score.

    Args:
        models (dict): A dictionary where the keys are model names and the values are instantiated machine learning model objects.
        X_train (array-like): The training feature data.
        y_train (array-like): The corresponding target labels for the training data.

    Returns:
        pd.DataFrame: A DataFrame containing the evaluation results for each model, including their average validation scores
                  and training scores.

    Raises:
        CustomException: If an error occurs while evaluating the models.

    '''


    try:
        # Dictionaries with validation and training scores of each model for plotting further.
        models_val_scores = dict()
        models_train_scores = dict()

        for model in models:
            # Get the model object from the key with his name.
            model_instance = models[model]

            # Measure training time.
            start_time = time.time()
            
            # Fit the model to the training data.
            model_instance.fit(X_train, y_train)

            end_time = time.time()
            training_time = end_time - start_time

            # Make predictions on training data and evaluate them.
            y_train_pred = model_instance.predict(X_train)
            train_score = np.sqrt(mean_squared_error(y_train, y_train_pred))

            # Evaluate the model using k-fold cross validation, obtaining a robust measurement of its performance on unseen data.
            val_scores = cross_val_score(model_instance, X_train, y_train, scoring='neg_mean_squared_error', cv=n_folds)
            avg_val_score = np.sqrt(-1 * val_scores.mean())
            val_score_std = np.sqrt((-1 * val_scores)).std()

            # Add the model scores to the validation and training scores dictionaries.
            models_val_scores[model] = avg_val_score
            models_train_scores[model] = train_score

            # Print the results.
            print(f'{model} results: ')
            print('-'*50)
            print(f'Training score: {train_score}')
            print(f'Average validation score: {avg_val_score}')
            print(f'Standard deviation: {val_score_std}')
            print(f'Training time: {round(training_time, 5)} seconds')
            print()

        # Convert scores to a dataframe
        val_df = pd.DataFrame(list(models_val_scores.items()), columns=['model', 'avg_val_score'])
        train_df = pd.DataFrame(list(models_train_scores.items()), columns=['model', 'train_score'])
        eval_df = val_df.merge(train_df, on='model')

        # Sort the dataframe by the best RMSE.
        eval_df  = eval_df.sort_values(['avg_val_score'], ascending=True).reset_index(drop=True)

        return eval_df
    
    except Exception as e:
        raise(CustomException(e, sys))
    

def plot_regression_kfold_cv(eval_df, figsize=(20, 7), bar_width=0.35, title_size=15,
                             title_pad=30, label_size=11, labelpad=20, legend_x=0.08, legend_y=1.08):
    '''
    Plot regression performance using k-fold cross-validation.

    Parameters:
        eval_df (DataFrame): DataFrame containing evaluation metrics for different models.
        figsize (tuple, optional): Figure size (width, height). Defaults to (20, 7).
        bar_width (float, optional): Width of bars in the plot. Defaults to 0.35.
        title_size (int, optional): Font size of the plot title. Defaults to 15.
        title_pad (int, optional): Padding of the plot title. Defaults to 30.
        label_size (int, optional): Font size of axis labels. Defaults to 11.
        labelpad (int, optional): Padding of axis labels. Defaults to 20.
        legend_x (float, optional): x-coordinate of legend position. Defaults to 0.08.
        legend_y (float, optional): y-coordinate of legend position. Defaults to 1.08.

    Raises:
        CustomException: Raised if an unexpected error occurs.

    Returns:
        None
    '''
    try:
        # Plot each model and their train and validation (average) scores.
        fig, ax = plt.subplots(figsize=figsize)

        x = np.arange(len(eval_df['model']))
        y = np.arange(len(eval_df['train_score']))

        val_bars = ax.bar(x - bar_width/2, eval_df['avg_val_score'], bar_width, label='Val score', color=AZUL1)
        train_bars = ax.bar(x + bar_width/2, eval_df['train_score'], bar_width, label='Train score', color=CINZA7)

        ax.set_xlabel('Model', color=CINZA1, labelpad=labelpad, fontsize=label_size, loc='left')
        ax.set_ylabel('RMSE', color=CINZA1, labelpad=labelpad, fontsize=label_size, loc='top')
        ax.set_title("Models' performances", fontweight='bold', fontsize=title_size, pad=title_pad, color=CINZA1, loc='left')
        ax.set_xticks(x, eval_df['model'], rotation=0, color=CINZA1, fontsize=10.8)
        ax.tick_params(axis='y', color=CINZA1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(CINZA9)
        ax.spines['bottom'].set_color(CINZA9)

        # Add scores on top of each bar
        for bar in val_bars + train_bars:
            height = bar.get_height()
            plt.annotate('{}'.format(round(height, 2)),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha='center', va='bottom', color=CINZA1)

        # Define handles and labels for the legend with adjusted sizes
        handles = [plt.Rectangle((0,0), 0.1, 0.1, fc=AZUL1, edgecolor = 'none'),
                plt.Rectangle((0,0), 0.1, 0.1, fc=CINZA7, edgecolor = 'none')]
        labels = ['Val score', 'Train score']
            
        ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(legend_x, legend_y), frameon=False, ncol=2, fontsize=10)
    
    except Exception as e:
        raise CustomException(e, sys)


def compare_actual_predicted(y_true, y_pred):
    '''
    Compares actual and predicted values and calculates the residuals.

    Args:
    y_true : The true target values.
    y_pred : The predicted target values.

    Returns:
    pandas.DataFrame: A dataframe containing the actual, predicted, and residual values.

    Raises:
    CustomException: An error occurred during the comparison process.
    '''
    try:
        actual_pred_df = pd.DataFrame({'Actual': np.round(y_true, 2),
                                    'Predicted': np.round(y_pred, 2), 
                                    'Residual': np.round(np.abs(y_pred - y_true), 2)})
        return actual_pred_df
    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_regressor(y_true, y_pred, y_train, model_name):
    '''
    Evaluates a regression model based on various metrics and plots.

    Args:
    y_true : The true target values.
    y_pred : The predicted target values.
    y_train : The actual target values from the training set.
    model_name (str): The name of the regression model.

    Returns:
    pandas.DataFrame: A dataframe containing the evaluation metrics.

    Raises:
    CustomException: An error occurred during the evaluation process.
    '''
    try:
        mae = round(mean_absolute_error(y_true, y_pred), 4)
        mse = round(mean_squared_error(y_true, y_pred), 4)
        rmse = round(np.sqrt(mse), 4)
        r2 = round(r2_score(y_true, y_pred), 4)
        mape = round(np.mean(np.abs((y_true - y_pred) / y_true)) * 100, 4)
        
        # Metrics
        print(f'Mean Absolute Error (MAE): {mae}')
        print(f'Mean Absolute Percentage Error (MAPE): {mape}')
        print(f'Mean Squared Error (MSE): {mse}')
        print(f'Root Mean Squared Error (RMSE): {rmse}')
        print(f'R-Squared (R2): {r2}')
        
        # Obtaining a dataframe of the metrics.
        df_results = pd.DataFrame({'Model': model_name, 'MAE': mae, 'MAPE': mape, 'RMSE': rmse, 'R2': r2}, index=['Results'])

        # Residual Plots
        
        # Analysing the results
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.set_title('Valores verdadeiros vs valores preditos', fontweight='bold', fontsize=12, color=CINZA1, pad=20, loc='left')
        ax.plot([y_train.min(),y_train.max()],[y_train.min(),y_train.max()], color=CINZA1, linestyle='--')
        ax.scatter(y_true, y_pred, color=AZUL1)
        ax.set_xlabel('Verdadeiro', fontsize=10.8, color=CINZA1, labelpad=20, loc='left')
        ax.set_ylabel('Predito', fontsize=10.8, color=CINZA1, labelpad=20, loc='top')
        ax.tick_params(axis='both', color=CINZA1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(CINZA9)
        ax.spines['bottom'].set_color(CINZA9)
        
        # Distribution of the residuals
        fig, ax = plt.subplots(figsize=(7, 3))
        sns.distplot((y_true - y_pred), ax=ax, color=AZUL1)
        ax.set_title('Distribuição dos resíduos', fontweight='bold', fontsize=12, color=CINZA1, pad=20, loc='left')
        ax.set_xlabel('Resíduos', fontsize=10.8, color=CINZA1, labelpad=20, loc='left')
        ax.set_ylabel('Frequência', fontsize=10.8, color=CINZA1, labelpad=20, loc='top')
        ax.tick_params(axis='both', color=CINZA1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(CINZA9)
        ax.spines['bottom'].set_color(CINZA9)
        
        return df_results

    except Exception as e:
        raise CustomException(e, sys)


def linear_interpretation_df(variables, coefficients, exp=False):
    '''
    Construct a DataFrame for the interpretation of coefficients from a linear regression.

    Parameters:
    - variables (list): List of variable names.
    - coefficients (list): List of corresponding coefficients.

    Returns:
    pandas.DataFrame: DataFrame with variables as the index, and columns for coefficient and direction of the relationship.

    Example:
    ```python
    variables = ['Variable1', 'Variable2', 'Variable3']
    coefficients = [0.5, -0.8, 0.2]
    result_df = linear_interpretation_df(variables, coefficients)
    ```

    The resulting DataFrame will have variables as the index, with columns for the coefficient and the direction of the relationship (positive, negative, or irrelevant).

    Raises:
    - CustomException: If an exception occurs during the DataFrame creation.

    Note:
    - The direction of the relationship is categorized as 'Positive' if the coefficient is greater than 0, 'Negative' if less than 0, and 'Irrelevant' if 0.

    '''
    try:
        coef_df = pd.DataFrame({'Variável': variables, 
                                'Coeficiente': coefficients})
        coef_df['Correlação'] = coef_df['Coeficiente'].apply(lambda x: 'Positiva' if x > 0 else 'Negativa' if x < 0 else 'Irrelevante')
        
        # For a logistic regression.
        if exp:
            coef_df['Exponencial'] = np.exp(coef_df['Coeficiente'])

        coef_df = coef_df.reindex(coef_df['Coeficiente'].abs().sort_values(ascending=False).index)
        coef_df.set_index('Variável', inplace=True)
        return coef_df
    except Exception as e:
        raise CustomException(e, sys)


# Classification modelling.

def classification_kfold_cv(models, X_train, y_train, n_folds=5):
    '''
    Evaluate multiple machine learning models using k-fold cross-validation.

    This function evaluates a dictionary of machine learning models by training each model on the provided training data
    and evaluating their performance using k-fold cross-validation. The evaluation metric used is ROC-AUC score.

    Args:
        models (dict): A dictionary where the keys are model names and the values are instantiated machine learning model objects.
        X_train (array-like): The training feature data.
        y_train (array-like): The corresponding target labels for the training data.

    Returns:
        pd.DataFrame: A DataFrame containing the evaluation results for each model, including their average validation scores
                  and training scores.

    Raises:
        CustomException: If an error occurs while evaluating the models.

    '''


    try:
        # Stratified KFold in order to maintain the target proportion on each validation fold - dealing with imbalanced target.
        stratified_kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        # Dictionaries with validation and training scores of each model for plotting further.
        models_val_scores = dict()
        models_train_scores = dict()

        for model in models:
            # Get the model object from the key with his name.
            model_instance = models[model]

            # Measure training time.
            start_time = time.time()
            
            # Fit the model to the training data.
            model_instance.fit(X_train, y_train)

            end_time = time.time()
            training_time = end_time - start_time

            # Make predictions on training data and evaluate them.
            y_train_pred = model_instance.predict(X_train)
            train_score = roc_auc_score(y_train, y_train_pred)

            # Evaluate the model using k-fold cross validation, obtaining a robust measurement of its performance on unseen data.
            val_scores = cross_val_score(model_instance, X_train, y_train, scoring='roc_auc', cv=stratified_kfold)
            avg_val_score = val_scores.mean()
            val_score_std = val_scores.std()

            # Add the model scores to the validation and training scores dictionaries.
            models_val_scores[model] = avg_val_score
            models_train_scores[model] = train_score

            # Print the results.
            print(f'{model} results: ')
            print('-'*50)
            print(f'Training score: {train_score}')
            print(f'Average validation score: {avg_val_score}')
            print(f'Standard deviation: {val_score_std}')
            print(f'Training time: {round(training_time, 5)} seconds')
            print()

        # Convert scores to a dataframe
        val_df = pd.DataFrame(list(models_val_scores.items()), columns=['model', 'avg_val_score'])
        train_df = pd.DataFrame(list(models_train_scores.items()), columns=['model', 'train_score'])
        eval_df = val_df.merge(train_df, on='model')

        # Sort the dataframe by the best ROC-AUC score.
        eval_df  = eval_df.sort_values(['avg_val_score'], ascending=False).reset_index(drop=True)
        
        return eval_df
    
    except Exception as e:
        raise(CustomException(e, sys))
    

def plot_classification_kfold_cv(eval_df, figsize=(20, 7), bar_width=0.35, title_size=15,
                             title_pad=30, label_size=11, labelpad=20, legend_x=0.08, legend_y=1.08):
    '''
    Plot classification performance using k-fold cross-validation.

    Parameters:
        eval_df (DataFrame): DataFrame containing evaluation metrics for different models.
        figsize (tuple, optional): Figure size (width, height). Defaults to (20, 7).
        bar_width (float, optional): Width of bars in the plot. Defaults to 0.35.
        title_size (int, optional): Font size of the plot title. Defaults to 15.
        title_pad (int, optional): Padding of the plot title. Defaults to 30.
        label_size (int, optional): Font size of axis labels. Defaults to 11.
        labelpad (int, optional): Padding of axis labels. Defaults to 20.
        legend_x (float, optional): x-coordinate of legend position. Defaults to 0.08.
        legend_y (float, optional): y-coordinate of legend position. Defaults to 1.08.

    Raises:
        CustomException: Raised if an unexpected error occurs.

    Returns:
        None
    '''
    try:
        # Plot each model and their train and validation (average) scores.
        fig, ax = plt.subplots(figsize=figsize)

        x = np.arange(len(eval_df['model']))
        y = np.arange(len(eval_df['train_score']))

        val_bars = ax.bar(x - bar_width/2, eval_df['avg_val_score'], bar_width, label='Val score', color=AZUL1)
        train_bars = ax.bar(x + bar_width/2, eval_df['train_score'], bar_width, label='Train score', color=CINZA7)

        ax.set_xlabel('Model', color=CINZA1, labelpad=labelpad, fontsize=label_size, loc='left')
        ax.set_ylabel('ROC-AUC', color=CINZA1, labelpad=labelpad, fontsize=label_size, loc='top')
        ax.set_title("Models' performances", fontweight='bold', fontsize=title_size, pad=title_pad, color=CINZA1, loc='left')
        ax.set_xticks(x, eval_df['model'], rotation=0, color=CINZA1, fontsize=10.8)
        ax.tick_params(axis='y', color=CINZA1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(CINZA9)
        ax.spines['bottom'].set_color(CINZA9)

        # Add scores on top of each bar
        for bar in val_bars + train_bars:
            height = bar.get_height()
            plt.annotate('{}'.format(round(height, 2)),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha='center', va='bottom', color=CINZA1)

        # Define handles and labels for the legend with adjusted sizes
        handles = [plt.Rectangle((0,0), 0.1, 0.1, fc=AZUL1, edgecolor = 'none'),
                plt.Rectangle((0,0), 0.1, 0.1, fc=CINZA7, edgecolor = 'none')]
        labels = ['Val score', 'Train score']
            
        ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(legend_x, legend_y), frameon=False, ncol=2, fontsize=10)
    
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_classifier(y_true, y_pred, probas):
    '''
    Evaluate the performance of a binary classifier and visualize the results.

    This function calculates and displays various evaluation metrics for a binary classifier,
    including the classification report, confusion matrix, ROC curve and AUC, PR curve and AUC,
    brier score, gini and ks.

    Args:
    - y_true (pd.series): True binary labels.
    - y_pred (pd.series): Predicted binary labels.
    - probas (pd.series): Predicted probabilities of positive class.

    Returns:
    - model_metrics (pd.DataFrame): A dataframe containing the classification metrics for the passed set.

    Raises:
    - CustomException: If an error occurs during evaluation.
    '''

    try:
        # Print classification report and calculate its metrics to include in the final metrics df.
        print(classification_report(y_true, y_pred))
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        # Calculate and print brier score, gini and ks.
        brier_score = brier_score_loss(y_true, probas)
        print(f'Brier Score: {round(brier_score, 2)}')
        
        fpr, tpr, thresholds = roc_curve(y_true, probas)
        roc_auc = roc_auc_score(y_true, probas)
        gini = 2 * roc_auc - 1
        print(f'Gini: {round(gini, 2)}')
        
        scores = pd.DataFrame()
        scores['actual'] = y_test.reset_index(drop=True)
        scores['absent_probability'] = probas
        sorted_scores = scores.sort_values(by=['absent_probability'], ascending=False)
        sorted_scores['cum_present'] = (1 - sorted_scores['actual']).cumsum() / (1 - sorted_scores['actual']).sum()
        sorted_scores['cum_absent'] = sorted_scores['actual'].cumsum() / sorted_scores['actual'].sum()
        sorted_scores['ks'] = np.abs(sorted_scores['cum_absent'] - sorted_scores['cum_present'])
        ks = sorted_scores['ks'].max()
        
        print(f'KS: {round(ks, 2)}')
        
        # Confusion matrix.
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot = True, fmt = 'd')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Values')
        plt.ylabel('Real Values')
        plt.show()
        
        # Plot ROC Curve and ROC-AUC.
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.2f}', color=AZUL1)
        ax.plot([0, 1], [0, 1], linestyle='--', color=CINZA4)  # Random guessing line.
        ax.set_xlabel('False Positive Rate', fontsize=10.8, color=CINZA1, labelpad=20, loc='left')
        ax.set_ylabel('True Positive Rate', fontsize=10.8, color=CINZA1, labelpad=20, loc='top')
        ax.set_xticks(ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'], color=CINZA1)
        ax.set_yticks(ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'], color=CINZA1)
        ax.set_title('Receiver operating characteristic (ROC) curve', fontweight='bold', fontsize=12, color=CINZA1, pad=20, loc='left')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(CINZA9)
        ax.spines['bottom'].set_color(CINZA9)
        ax.legend()
    
        # PR AUC Curve and score.

        # Calculate model precision-recall curve.
        p, r, _ = precision_recall_curve(y_true, probas)
        pr_auc = auc(r, p)
        
        # Plot the model precision-recall curve.
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(r, p, marker='.', label=f'PR AUC = {pr_auc:.2f}', color=AZUL1)
        ax.set_xlabel('Recall', fontsize=10.8, color=CINZA1, labelpad=20, loc='left')
        ax.set_ylabel('Precision', fontsize=10.8, color=CINZA1, labelpad=20, loc='top')
        ax.set_xticks(ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'], color=CINZA1)
        ax.set_yticks(ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'], color=CINZA1)
        ax.set_title('Precision-recall (PR) curve', fontweight='bold', fontsize=12, color=CINZA1, pad=20, loc='left')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(CINZA9)
        ax.spines['bottom'].set_color(CINZA9)
        ax.legend()

        # Construct a DataFrame with metrics for passed sets.
        model_metrics = pd.DataFrame({
                                    'Metric': ['Accuracy',
                                               'Precision',
                                               'Recall',
                                               'F1-Score',
                                               'ROC-AUC',
                                               'KS',
                                               'Gini',
                                               'PR-AUC',
                                               'Brier'],
                                    'Value': [accuracy, 
                                              precision, 
                                              recall,
                                              f1,
                                              roc_auc,
                                              ks,
                                              gini, 
                                              pr_auc,
                                              brier_score,
                                              ],
                                    })
        
        return model_metrics

    except Exception as e:
        raise CustomException(e, sys)


def plot_probability_distributions(y_true, probas):
    '''
    Plots the kernel density estimate (KDE) of predicted probabilities for absent and present candidates.

    Parameters:
    - y_true (array-like): The true class labels (0 for present, 1 for absent).
    - probas (array-like): Predicted probabilities for the positive class (absent candidates).

    Raises:
    - CustomException: Raised if an unexpected error occurs during plotting.

    Example:
    ```python
    plot_probability_distributions(y_true, probas)
    ```

    Dependencies:
    - pandas
    - seaborn
    - matplotlib

    Note:
    - The function assumes the existence of color constants VERMELHO_FORTE, CINZA7, CINZA1, CINZA9.

    The function creates a KDE plot illustrating the distribution of predicted probabilities for absent and present candidates.
    It provides visual insights into the model's ability to distinguish between the two classes.

    '''
    try:
        probas_df = pd.DataFrame({'Probabilidade de Abstenção': probas,
                                'Abstenção': y_true})

        fig, ax = plt.subplots(figsize=(10, 4))
        sns.kdeplot(data=probas_df, x='Probabilidade de Abstenção', hue='Abstenção', fill=True, ax=ax, palette=[CINZA7, VERMELHO_FORTE])
        ax.set_title('Distribuição das probabilidades preditas - ausentes e presentes', fontweight='bold', fontsize=12, color=CINZA1, pad=20, loc='left')
        ax.set_xlabel('Probabilidades Preditas', fontsize=10.8, color=CINZA1, labelpad=20, loc='left')
        ax.set_ylabel('Densidade', fontsize=10.8, color=CINZA1, labelpad=20, loc='top')
        ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                    ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'],
                    color=CINZA1)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6],
                    ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0', '1.2', '1.4', '1.6'],
                    color=CINZA1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(CINZA9)
        ax.spines['bottom'].set_color(CINZA9)
        
        handles = [plt.Rectangle((0,0), 0.1, 0.1, fc=VERMELHO_FORTE, edgecolor = 'none'),
                plt.Rectangle((0,0), 0.1, 0.1, fc=CINZA7, edgecolor = 'none')]
        labels = ['Ausentes', 'Presentes']
            
        ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.17, 1.05), frameon=False, ncol=2, fontsize=10)

        
    except Exception as e:
        raise CustomException(e, sys)


def probability_scores_ordering(y_true, probas):
    '''
    Order and visualize the probability scores in deciles based on predicted probabilities and true labels.

    Parameters:
    - y_true (pd.Series): Actual target values for the set. 1 is absent and 0 is present.
    - probas (pd.Series): Predicted probabilities of being absent for the passed set.

    Returns:
    - None: Plots the probability scores ordering.

    Raises:
    - CustomException: An exception is raised if an error occurs during the execution.
    
    Example:
    ```python
    probability_scores_ordering(y_test, probas)
    ```
    '''
    try:
        # Add some noise to the predicted probabilities and round them to avoid duplicate problems in bin limits.
        noise = np.random.uniform(0, 0.0001, size=probas.shape)
        probas += noise
        #probas = round(probas, 10)
        
        # Create a DataFrame with the predicted probabilities of being absent and actual values.
        probas_actual_df = pd.DataFrame({'probabilities': probas, 'actual': y_true.reset_index(drop=True)})
        
        # Sort the probas_actual_df by probabilities.
        probas_actual_df = probas_actual_df.sort_values(by='probabilities', ascending=True)
        
        # Calculate the deciles.
        probas_actual_df['deciles'] = pd.qcut(probas_actual_df['probabilities'], q=10, labels=False, duplicates='drop')
        
        # Calculate the absent rate per decile.
        decile_df = probas_actual_df.groupby(['deciles'])['actual'].mean().reset_index().rename(columns={'actual': 'absent_rate'})
        
        # Plot probability scores ordering.
        # Plot bar graph of deciles vs event rate.
        fig, ax = plt.subplots(figsize=(12, 3))
        
        bars = ax.bar(decile_df['deciles'], decile_df['absent_rate'], color=VERMELHO_FORTE)
        
        ax.set_title('Ordenação dos scores de probabilidade - Taxa de abstenção por decil', loc='left', fontweight='bold', fontsize=14)
        ax.set_xticks(range(10), ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'], color=CINZA1)
        ax.tick_params(axis=u'both', which=u'both',length=0)
        ax.set_xlabel('Decil', labelpad=25, loc='center', color=CINZA1)
        ax.yaxis.set_visible(False)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color(CINZA9)
        ax.grid(False)
        
        # Annotate absent rate inside each bar with increased font size
        for bar, absent_rate in zip(bars, decile_df['absent_rate']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height - 0.02, f'{absent_rate*100:.1f}%', ha='center', va='top', color='white', fontsize=10.4)
            
    except Exception as e:
        raise CustomException(e, sys)