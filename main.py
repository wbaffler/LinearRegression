import sys

import numpy as np
from numpy.matrixlib.defmatrix import matrix
from file_parser import unique_countries, unique_codes, get_data
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

bln = 1000000000
def main():
    if len (sys.argv) == 1:
        print('''
        Задайте код страны и год окончания регрессирования в аргументы:
        Существующие страны:
        ''')
        for code, country in zip(unique_codes, unique_countries):
            print(code, country)
        sys.exit (1)
    else:
        if len (sys.argv) < 3:
            print ("Ошибка. Слишком мало параметров.")
            sys.exit (1)

        if len (sys.argv) > 3:
            print ("Ошибка. Слишком много параметров.")
            sys.exit (1)

    int_year = int(sys.argv[2])
    matrix = get_data(sys.argv[1])
    last_year = int(matrix[len(matrix)-1, 0])
    if int_year > last_year:
        paint_graph(matrix, int_year)
    else:
        print("Input year error")
        sys.exit(1)

def paint_graph(matrix, year):
    X = matrix[:, 0].reshape(-1,1)
    Y = matrix[:, 1].reshape(-1,1)
    
    regressor = LinearRegression()
    regressor.fit(X, Y)
    m = regressor.coef_
    c = regressor.intercept_
    
    x_pred_temp = []
    for i in range(int(X[len(matrix)-1])+1, year):
        x_pred_temp.append(i)
    x_pred = np.array(x_pred_temp).reshape(-1,1)
    y_pred = regressor.predict(x_pred)

    y = m*X[:]+c
    x = X
    
   
    plt.plot(x, y/bln, c='r')
    plt.plot(X, Y/bln, '.')
    plt.plot(x_pred, y_pred/bln, '.')
    plt.title('GPD of country')
    plt.xlabel('Years')
    plt.ylabel('GPD (bln)')
    plt.show()

if __name__ == "__main__":
    main()