import math
import itertools
import numpy as np
import pandas as pd
import scipy
from scipy import stats
from sklearn import metrics 

def analyse(cn,jc,aa,ra,pa):
    features=[cn,jc,aa,ra,pa]

    df = pd.DataFrame(features).T
    df.columns=['CN','JC','AA','RA','PA']
    # display(df.cov())

    arranged_values =[[],[],[],[],[]]
    for k,v in cn.items():
        arranged_values[0]+=[cn[k]]
        arranged_values[1]+=[jc[k]]
        arranged_values[2]+=[aa[k]]
        arranged_values[3]+=[ra[k]]
        arranged_values[4]+=[pa[k]]

    pearson=[[],[],[],[],[]]

    for i in range(5):
        for j in range(5):
            pearson[i]+=[scipy.stats.pearsonr(arranged_values[i], arranged_values[j])[0]]
    # scipy.stats.pearsonr(arranged_values[2], arranged_values[0])

    pearson = pd.DataFrame(pearson)

    pearson.columns=['CN','JC','AA','RA','PA']
    # pearson.rows=['CN','JC','AA','RA','PA']
    print(pearson)


    fv=[{},{},{},{},{}]
    fn=0
    for f in features:
        f=dict(sorted(f.items(), key=lambda kv: kv[1], reverse=True))
        
        i=1
        for k in f:
            fv[fn][k]=i
            i+=1
        fv[fn]=dict(sorted(fv[fn].items(), key=lambda kv: kv[0]))
        
        fn+=1

    pearsonr=[[],[],[],[],[]]

    for i in range(5):
        for j in range(5):
            pearsonr[i]+=[scipy.stats.pearsonr(list(fv[i].values()),list(fv[j].values()))[0]]
    # scipy.stats.pearsonr(arranged_values[2], arranged_values[0])
    idx=pd.Series(['CN','JC','AA','RA','PA'])

    pearsonr = pd.DataFrame(pearsonr,index=idx)

    pearsonr.columns=['CN','JC','AA','RA','PA']
    # pearson.rows=['CN','JC','AA','RA','PA']
    pearsonr

    from pandas.plotting import scatter_matrix
    import matplotlib.pyplot as plt
    # %matplotlib inline

    # import matplotlib
    # matplotlib.style.use('ggplot')

    # plt.figure(figsize=(20,20))
    # fig_size = plt.rcParams["figure.figsize"]
    # fig_size[0]=15
    # fig_size[1]=15

    scatter_matrix(df,figsize=(12,12),diagonal='hist')#kde = Kernel Density Estimation
    plt.savefig('twitter_scatter_matrix.png')
    plt.show()

    pd.DataFrame.from_dict(cn, orient='index')
    df1 = pd.DataFrame([cn,cn]).T

    plt.matshow(df.corr(), cmap='viridis_r')
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0]=8
    fig_size[1]=8
    plt.xticks(range(len(df.columns)), df.columns)
    plt.yticks(range(len(df.columns)), df.columns)
    plt.colorbar()
    plt.title('Heatmap')
    plt.show()
