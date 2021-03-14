# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
from typing import Tuple
from sklearn.preprocessing import  OneHotEncoder, StandardScaler, OrdinalEncoder
import numpy as np
@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    
def separate_target(df: pd.DataFrame,
                    target: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separates the target variable from the features.
    :param df: the dataframe that contains the full data
    :param target: the name of the target variable
    :return: a tuple of the features dataframe and the target series
    """
    X = df.copy(deep=True)
    y = X.pop(target)

    return X, y

def process_data(data, n_largest=1000):
    """
    Takes raw data, drops unnecessary columns and one hot encodes brewery name, and splits data into X and y sets.
    note that I have set the onehotencoder to take the n_largest breweries and ignore the others.
    Parameters
    ---------
    data: Pandas dataframe of original data
    n_largest: the number of categories to extract for the brewery names.

    Output
    ------
    X: numpy array of independent variables for modelling
    y: encoded target class
    y_encoder: encoder used to encode target class - used in converting index back to a class.
    ohe: One hot encoder of the brewery names
    scaler: Standard scaler for the numeric fields.

    """
    drop_columns = ['brewery_id','review_overall','review_time','review_profilename','beer_name','beer_beerid', 'beer_abv']
    data.drop(columns = drop_columns, inplace=True)
    print('columns dropped')
    ohe = OneHotEncoder(sparse=False, handle_unknown = 'ignore')
    categories= data.brewery_name.value_counts().nlargest(n_largest).index.tolist()
    ohe.fit(data.loc[data['brewery_name'].isin(categories),["brewery_name"]])
    brewery_names = ohe.transform(data[["brewery_name"]])
    print('brewery names encoded')
    data.drop(columns = ['brewery_name'], inplace = True)
    scaler = StandardScaler()
    X,y = separate_target(data,'beer_style')
    y_encoder = OrdinalEncoder()
    y = y_encoder.fit_transform(np.array(y).reshape(-1,1))
    X = scaler.fit_transform(X)
    X = np.concatenate([X,brewery_names],axis = 1)
    print('scaled data')
    return X,y,y_encoder,ohe,scaler


def inverse_transformer(y,encoder):
    """
    Transforms value from the index to the class name.
    Parameters
    ----------
    y: Target class index
    encoder: encoder used for inverse transormation
    """
    return encoder.inverse_transform(y)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
