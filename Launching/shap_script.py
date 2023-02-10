import pandas as pd
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import shap
from BorutaShap import BorutaShap
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import seaborn as sns

import os
from scipy import stats
from datetime import datetime, timedelta
import copy
sns.set()


'''
OBJECTIVE:
The purpose of this script is to read the input table, 
pass it through a machine learning model 
and output a table for each metric analyzed as output from the script.



Example: 
If you analize CTR, CPC, VTR,
the output of this code are 3 tables called:
ctr_shap_results
cpc_shap_results
vtr_shap_results



REDEFINE FOR LAUNCHING:
IN THIS CODE IT'S NECESSARY TO REDEFINE HOW 
THE INPUT TABLE ENTERS TO THE SCRIPT AND HOW THE
SHAP_RESULTS TABLES IS OUTPUT.
'''




"""
Function prototype:
1.load_data
2.filter_impressions
3.filter_JSON_symbol
4.filter_confidence
5.calculate_weights
6.process_metrics
7.create_tags_df
8.instance_model
9.calculate_shap_scores
10.bar_plot_scores
"""

#1.------
def load_data(data_path: str):
    print('Loading data....')
    try:
        df = pd.read_csv(data_path)
        df = df.rename(columns={'account_name':'brand_name', 'filename':'asset_id'})
        df.tag_name = df.tag_name.astype(str)
        df.tag_name = df.tag_name.str.lower()
        df.file_type = df.file_type.str.lower()
        print('Data was uploaded.')
        return df
    except:
        name_function = 'load_data'
        print(f"An error occurred in: {name_function}")



#2.------
def filter_impressions(df: pd.DataFrame, threshold: float=0.1):
    try:
        df = df.loc[df.impressions > df.impressions.quantile(threshold)]
        print(f'Bottom {threshold*100}% of impressions was filtered.')
        return df
    except:
        name_function = 'filter_impressions'
        print(f"An error occurred in: {name_function}")


#3.------
def filter_JSON_symbol(df: pd.DataFrame):
    try:
        df['tag_name'] = df['tag_name'].replace('_', ':')
        df = df.loc[~df.tag_name.str.contains(':')]
        df = df.reset_index(drop=True)
        print('JSON symbol was filtered.')
        return df
    except:
        name_function = 'filter_JSON_symbol'
        print(f"An error occurred in: {name_function}")


#4.-----
def filter_confidence(df, conf=0.9):
    try:
        df['confidence'] = df['confidence'].apply(lambda x: x/100 if x > 1 else x)
        df = df.loc[df.confidence>conf]
        df = df.reset_index(drop=True)
        print(f'Confidence less than {conf} was filtered.')
        return df
    except:
        name_function = 'filter_confidence'
        print(f"An error occurred in: {name_function}")


#5.------
def calculate_weights(df: pd.DataFrame):
    try:
        df['weight'] = 1
        for i in range(len(df)):
            if df.carrousel_position[i]>0:
                df.weight[i] = 1/(1+df.carrousel_position[i])**2
        print('Weights were calculated.')
        return df
    except:
        name_function = 'calculate_weights'
        print(f"An error occurred in: {name_function}")


#SI EL AD NO ES CARROUSEL ------> carrousel_position = -1
#SI EL AD ES UN CARROUSEL ------> 0,1,2,3,4

#6.-----
def process_metrics(df, metric, index_col=['asset_id'], shift_to_zero=True, transform=True, transform_coef=1e-3):
    try:    
        metrics = df[index_col+[metric]].groupby(index_col).first() 
        metrics[metric+"_raw"] = (df[index_col + [metric]].groupby(index_col).first()).replace([np.inf, -np.inf], 0)
         
        if transform:
          metrics[[metric]] = np.log(metrics[[metric]] + transform_coef)
         
        if shift_to_zero:
          m = metrics[metric].mean()
          s = metrics[metric].std()
          metrics[metric] = ((metrics[metric] - m) / s).fillna(0)

        print('Metrics were normalized.') 
        return metrics
    except:
        name_function = 'process_metrics'
        print(f"An error occurred in: {name_function}")


#7......
def create_tags_df(df, index_col='asset_id', tag_id_col='tag_id', tag_name_col='tag_name', tag_type_col='tag_type'):
    try:    
        tags = (df.pivot_table(index=index_col, columns=[tag_type_col, tag_name_col],
                                values=tag_id_col, aggfunc='count').fillna(0) > 0).\
                                astype(int).\
                                replace([np.inf, -np.inf], 0)                          
        print('tags was created.')

        tags_shap = tags.copy()
        tags_shap.columns = tags_shap.columns.droplevel()
        tags_shap = tags_shap.loc[:,~tags_shap.columns.duplicated()]
        print('tags_shap was created.')

        return tags, tags_shap
    except:
        name_function = 'create_tags_df'
        print(f"An error occurred in: {name_function}")


#8......
def instance_model():
    try:
        model = LGBMRegressor(boosting_type='gbdt',
                                  max_depth=3,
                                  n_estimators=100,
                                  learning_rate=0.4,
                                  objective='mse',
                                  n_jobs=-1, random_state=0)
        return model
    except:
        name_function = 'instance_model'
        print(f"An error occurred in: {name_function}")


#9.....
def calculate_shap_scores(tags: pd.DataFrame, tags_shap: pd.DataFrame):
    try:
        tags_freq = tags.sum(0).reset_index().rename(columns={'tag_type':'tag_group', 'tag_name':'tag', 0:'usage_frequency'})
        tags_freq["usage_frequency"] = tags_freq.usage_frequency / len(tags)
        tags_freq = tags_freq.drop_duplicates(subset=['tag'])

        tags_score_per_asset = pd.DataFrame(shap_values.values, columns=shap_values.feature_names)
        tags_score_per_asset['asset_id'] = tags_shap.reset_index().asset_id.copy()

        list_of_mean_tags = []
        for col in tags_shap.columns:
            tag_mean = tags_shap[col].loc[tags_shap[col]==1].reset_index().drop(columns={col}).merge(tags_score_per_asset[[col, 'asset_id']], on='asset_id', how='inner').mean()[0]
            list_of_mean_tags.append(tag_mean)

        data = {'tag': tags_shap.columns, 'tag_mean': list_of_mean_tags}

        df_sign = pd.DataFrame(data)

        shap_results = pd.DataFrame({'score':tags_score_per_asset.drop(columns={'asset_id'}).abs().mean()})
        shap_results = shap_results.reset_index().rename(columns={'index':'tag'})

        for i,value in enumerate(df_sign.tag_mean):
            if value<0:
                shap_results['score'][i] = shap_results['score'][i]*(-1)


        shap_results = pd.merge(shap_results, tags_freq, how='inner', on='tag')
        shap_results['period'] = 'all'
        shap_results['metric'] = metric

        return shap_results
    except:
        name_function = 'calculate_shap_scores'
        print(f"An error occurred in: {name_function}")

#10.......
#def bar_plot_scores(shap_values):
#    try:
#        import copy
#        shap_values_modified = copy.deepcopy(shap_values)
#        shap_values_modified.values[0] = shap_results.score.values
#        plt.figure()
#        shap.plots.bar(shap_values_modified[0], neg_color='#CC0000', pos_color='#5687E1', 
#                        max_display=10, show_sum_of_features=False, edge_color='black', show=False)
#        #plt.xlim(-1,1)
#        plt.xlabel(f'Impact score on metric: {metric.upper()}')
#        plt.savefig(f'Img/bar_plot_for_{metric}.png')
#        plt.tight_layout()
#        plt.close()
#    except:
#        name_function = 'bar_plot_scores'
#        print(f"An error occurred in: {name_function}")

#----------------------------------MAIN CODE-------------------------------------------:
df = load_data(data_path='data/data_testing.csv') #1
df = filter_impressions(df, threshold=0.1) #2
df = filter_JSON_symbol(df) #3
df = filter_confidence(df, conf=0.9) #4
df = calculate_weights(df) #5



print('----------------------------')
print('Dataframe was processed correctly!')
print('----------------------------')
print('Starting to work with metrics and SHAP....')

metric_list = ['ctr', 'cpc', 'cpm', 'vtr', 'ad_recall_score', 'bai']
for metric in metric_list:
    print(f'Working with this metric: {metric.upper()}.....')
    df_metric = df.copy()
    if metric not in df_metric.columns:
        continue
    if metric=='vtr':
        df_metric = df_metric.loc[df_metric.file_type=='video']

    #Normalizing metrics and creating dataframes: tags, tags_shap, tags_freq
    metrics = process_metrics(df=df_metric, metric=metric) #6
    tags, tags_shap = create_tags_df(df=df_metric) #7

    #Creating weights df
    weights = df_metric[['asset_id', 'weight']].groupby('asset_id').first().iloc[:,0]
    
    #Instancing model and training it
    model = instance_model() #8
    model.fit(X=tags_shap, y=metrics[metric], sample_weight=weights)

    #Applying shap to model
    explainer = shap.Explainer(model, tags_shap)
    shap_values = explainer(tags_shap)

    #Saving beeswarm chart in the folder: Img
    #try:
    #    colors = [(1, 0, 0), (86/255, 135/255, 225/255)]
    #    n_bin = 100
    #    cmap_parameter = LinearSegmentedColormap.from_list(name='name_list', colors=colors, N=n_bin)
    #    shap.summary_plot(shap_values, tags_shap, plot_type="dot", max_display=10, plot_size=(8,6), show=False, cmap=cmap_parameter, color_bar=False)
    #    plt.xlim(-1,1)
    #    plt.xlabel(f'Impact value on metric: {metric.upper()}')
    #    plt.savefig(f'Img/beeswarm_for_{metric}.png')
    #    plt.close()
    #except:
    #    print(f'It was not possible save the beeswarm chart for metric: {metric}')

    #Calculating shap_scores
    shap_results = calculate_shap_scores(tags=tags, tags_shap=tags_shap) #9
    globals()[metric + '_shap_results'] = calculate_shap_scores(tags=tags, tags_shap=tags_shap)
    

    globals()[metric + '_shap_results'].to_csv(f'data/{metric}_shap_results.csv', index=False)
    
    #Plotting bar chart
    #bar_plot_scores(shap_values=shap_values) #10

    print(f'The work with this metric: {metric.upper()} has finished.')
    print('----------------------------')

print('Script finished.')