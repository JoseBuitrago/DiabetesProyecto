"""
PyCharm
Creado sábado, 13 de febrero de 2021

@author: José Alejandro Buitrago Cardenas / Leydi Esperanza Perez Leal
"""
"""
Usage:
python preprocess.py <name of the input file> <name of the output file> <name of the log file>

Usage examples:
python preprocess.py diabetes_data_upload.csv preprocessed_data.csv preprocessing.log
"""

import datetime
import logging
import os
import sys
import csv
import pandas

def replace_categorical_column_with_dummies(the_dataframe, column_name):
    if column_name in the_dataframe.columns:
        the_dummies = pandas.get_dummies(the_dataframe[column_name])
        the_dummies.columns = [(column_name + '_{}').format(elem) for i, elem in enumerate(list(the_dummies.columns.values), 1)]
        the_dataframe = pandas.concat([the_dummies, the_dataframe], axis=1)
        the_dataframe = the_dataframe.drop(column_name, axis=1)
        return the_dataframe
    else:
        return the_dataframe

def convert_ordinal_categorical_values_to_numeric_with_dict(ordinal_categorical_column, the_dictionary):
    current_values = list(set(ordinal_categorical_column))
    current_values.sort()
    new_values = range(0, len(current_values))
    ordinal_categorical_column = ordinal_categorical_column.replace(the_dictionary)
    return ordinal_categorical_column

def main(raw_data_file_name, preprocessed_file_name, log_file_name):

    logging.info(str(datetime.datetime.now()) + ': Started.')

    raw_data_frame = pandas.read_csv(raw_data_file_name)

    logging.info(str(datetime.datetime.now()) + ': The size of the raw data matrix is (' + str(raw_data_frame.shape[0]) + ', ' + str(raw_data_frame.shape[1]) + ').')

    logging.info(str(datetime.datetime.now()) + ': Non-ordinal categorical variables are now going to be replaced with dummy variables.')

    # The lines below replace categorical variables with dummy features.
    # This is done only for non-ordinal categorical variables.
    raw_data_frame = replace_categorical_column_with_dummies(raw_data_frame, 'Gender')
    raw_data_frame = replace_categorical_column_with_dummies(raw_data_frame, 'Polyuria')
    raw_data_frame = replace_categorical_column_with_dummies(raw_data_frame, 'Polydipsia')
    raw_data_frame = replace_categorical_column_with_dummies(raw_data_frame, 'sudden weight loss')
    raw_data_frame = replace_categorical_column_with_dummies(raw_data_frame, 'weakness')
    raw_data_frame = replace_categorical_column_with_dummies(raw_data_frame, 'Polyphagia')
    raw_data_frame = replace_categorical_column_with_dummies(raw_data_frame, 'Genital thrush')
    raw_data_frame = replace_categorical_column_with_dummies(raw_data_frame, 'visual blurring')
    raw_data_frame = replace_categorical_column_with_dummies(raw_data_frame, 'Itching')
    raw_data_frame = replace_categorical_column_with_dummies(raw_data_frame, 'Irritability')
    raw_data_frame = replace_categorical_column_with_dummies(raw_data_frame, 'delayed healing')
    raw_data_frame = replace_categorical_column_with_dummies(raw_data_frame, 'partial paresis')
    raw_data_frame = replace_categorical_column_with_dummies(raw_data_frame, 'muscle stiffness')
    raw_data_frame = replace_categorical_column_with_dummies(raw_data_frame, 'Alopecia')
    raw_data_frame = replace_categorical_column_with_dummies(raw_data_frame, 'Obesity')

    logging.info(str(datetime.datetime.now()) + ': Finished adding the dummy variables.')

    logging.info(str(datetime.datetime.now()) + ': Categorical class variable is being replaced with a numerical variable.')

    class_dictionary = {'Positive' : 1, 'Negative' : 0}

    raw_data_frame['class'] = convert_ordinal_categorical_values_to_numeric_with_dict(raw_data_frame['class'], class_dictionary)

    logging.info(str(datetime.datetime.now()) + ': Categorical class variable was replaced with a numerical variable.')

    raw_data_frame.to_csv(preprocessed_file_name, index=False)

    logging.info(str(datetime.datetime.now()) + ': Ended.')

raw_data_file_name = sys.argv[1]
preprocessed_file_name = sys.argv[2]
log_file_name = sys.argv[3]

logging.basicConfig(filename=log_file_name, level=logging.DEBUG)

main(raw_data_file_name, preprocessed_file_name, log_file_name)
