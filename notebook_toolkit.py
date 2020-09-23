#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 16:28:20 2020

@author: sandylee
"""



def drop_column_keyword_search(dataframe = None, keywords = None):
    
    """
    blah blah blah
    """
    
    regex = ""
    for word in range(0,len(keywords)):
        regex += keywords[word]+"|"
    regex = regex[:-1]
    updated_df = dataframe[dataframe.columns.drop(list
                                                 (dataframe.filter
                                                 (regex = regex, axis = 1)))]
    return updated_df
        
    
def column_null_percentage(df = None):

    import pandas as pd
    
    """
    blah blah blah
    """
    
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/
               df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], 
                             axis=1, keys=['Total', 'Percent'])
    return missing_data[missing_data.Percent > 0.1]


def correlation_matrix(df = None):
    
    """
    blah,blah,blah
    """
    
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np

    sns.set(style="white")
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=np.bool))
    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    heatmap = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
              square=True, linewidths=.5, cbar_kws={"shrink": .5})
    return(heatmap)

def multicolinear_drop(df = None):
    
    """
    blah,blah,blah
    """
    
    import numpy as np
    
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), 
                              k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.75)]
    return(to_drop)

def norm_feat(series):
    
    """
    blah,blah,blah
    """
    
    return (series - series.mean())/series.std()
    
        
 