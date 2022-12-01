
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import sklearn.metrics
import statsmodels.api as sm
from statsmodels.formula.api import ols
sns.set()
from pandas_profiling import ProfileReport
from IPython.core.display import display
from pandas_profiling.report.presentation.flavours.widget.notebook import (get_notebook_iframe,)
import csv
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import itertools

def check_nan(df: pd.DataFrame) -> None:
    '''
    This function allows to check the nan in a dafa fram in pandas
    '''     
    nan_cols  = df.isna().mean()*100
    display(f'N nan cols: {len(nan_cols[nan_cols>0])}')
    display(nan_cols[nan_cols>0])
    plt.figure(figsize=(10,6))
    sns.heatmap(df.isna(),
               yticklabels=False,
               cmap='viridis',
               cbar = False)
    plt.show();


def summary_regression_model(x,y):
    '''
    This functions creates an accurate report for linear regression models

     x_const = sm.add_constant(x) # add a constant to the model
    
    modelo = sm.OLS(y, x_const).fit() # fit the model
    
    pred = modelo.predict(x_const) # make predictions
    '''
    
    x_const = sm.add_constant(x) # add a constant to the model
    
    modelo = sm.OLS(y, x_const).fit() # fit the model
    
    pred = modelo.predict(x_const) # make predictions
    
    print(modelo.summary())   

def print_corr(df):
    
    '''
    this functions plosts a correlation head map with pearson method
    '''
   
    correlation = df.corr(method='pearson')

    mask = np.zeros_like(correlation, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    f, ax = plt.subplots(figsize=(10, 12))

    cmap = sns.diverging_palette(180, 20, as_cmap=True)
    sns.heatmap(correlation, mask=mask, cmap=cmap, vmax=1, vmin =-1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)

    return plt.show()


def writelstcsv(guest_list, filename):
    """Write the list to csv file."""

    with open(filename, "w", encoding='utf-8') as outfile:
        for entries in guest_list:
            outfile.write(entries)
            outfile.write("\n")


def report_pd(df, to_html:bool ):

    '''
    This function implements the pandas data frame report
    If to_html is True, it will return an html report. If False, it will show the report 
    on your jupyter notebook
    '''
    x= ProfileReport(df)
    y= x.to_file("your_report.html")
    if to_html == True:
        return y
    else:
        return x
   
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    PLEASE COPY PASTE THE BELOW METHOD:
    cnf_matrix = confusion_matrix(y_test, neigh5_pre, labels=[0,1, 2])
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=['0=0','50=1', '100=2'],normalize= False,  title='Confusion matrix')
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')