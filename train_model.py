import warnings
#warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from scipy import stats
from estimate_price import  compute_price, parse_coeficients
import matplotlib.pyplot as plt



def BIAS(train_data, coefficients):
    return(compute_price(coefficients, train_data['km']) - train_data['price'])

def MSE(train_data):
    return((train_data['bias']**2).sum())

def descent_gradient(train_data, coefficients, gamma_0 = 1, gamma_min = 1e-5, reduction_factor = .1):
    previous_mse = MSE(train_data)
    while gamma_0 > gamma_min:
        if MSE(train_data) > previous_mse: #Overshot the min -> recover values and Reduce gamma and 
            gamma_0*= reduction_factor
            coefficients = previous_coefficients
            train_data['bias'] = BIAS(train_data, coefficients)
        previous_mse = MSE(train_data)
        previous_coefficients = coefficients
        coefficients[0] = previous_coefficients[0] - gamma_0*2*train_data['bias'].sum()/len(train_data)
        coefficients[1] = previous_coefficients[1] - gamma_0*2*(train_data['bias']*train_data['km']).sum()/len(train_data)
        train_data['bias'] = BIAS(train_data, coefficients)
        
    return coefficients

def main():
    #Read data0
    train_data = pd.read_csv("./data.csv")
    
    #Normalize data
    normalized_data = train_data/train_data.max() 
    
    #Calculate coefficients
    tmp_theta = np.array([0,0])
    tmp_theta = [0, 0]
    normalized_data['bias'] = BIAS(normalized_data, tmp_theta)
    coefficients = descent_gradient(normalized_data, tmp_theta)
    #test_coefficients = np.polyfit(normalized_data['km'], normalized_data['price'], 1)
    
    # Denormalize coefficients
    coefficients[1] /= train_data.max()['km']
    coefficients = [i * train_data.max()['price'] for i in coefficients]
    #test_coefficients = [test_coefficients[0]*train_data.max()['price'], test_coefficients[1]*train_data.max()['price']/train_data.max()['km']] #Reverse Normalization
    
    #Calcular precision
    train_data['bias'] = BIAS(train_data, coefficients)
    P = (1 - train_data['bias'].var()/train_data['price'].var())*100
    print(f"Varianza explicada: {P:.2f}%")

    #Plot fit
    plt.scatter('km', 'price', data = train_data, label='data')
    plt.axline((0, coefficients[0]), slope = coefficients[1], label = 'gradient_descent')
    #test_coefficients[0] /= train_data.max()['km']
    #test_coefficients *= train_data.max()['price']
    #plt.axline((0, test_coefficients[1]), slope = test_coefficients[0], color = 'r', label='numpy')
    plt.legend()
    plt.show()
    
    #save result
    np.array(coefficients).tofile('.coefficients.txt', sep = '\t', format = '%s')
    #np.array(test_coefficients).tofile('.test_cf.txt', sep = '\t', format = '%s')

if __name__ == '__main__':
    main()