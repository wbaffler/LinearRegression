from os import name
from numpy.lib.arraysetops import unique
from numpy.matrixlib.defmatrix import matrix
import pandas as pd
import numpy as np

dataset = pd.read_csv('/Users/wbaffler/Desktop/Программирование/LinearRegression/gdp_csv.csv')

countries = dataset.iloc[:, 0:2].values
codes = countries[:, 1]
unique_codes = np.unique(codes)
indexes = np.unique(codes, return_index=True)[1]
unique_countries = []
for index in indexes:
    unique_countries.append(countries[index, 0])


def get_data(code):   
    matrix = []
    num_data = dataset.iloc[:, 2:].values
    for i in range(len(codes)):
        if codes[i] == code:
            matrix.append(num_data[i])
    np_matrix = np.array(matrix)
    return np_matrix.astype(np.single)