#----------------------------------------------------
# SETTING MODULES AND LIBRARIES

# data manipulation
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from operator import add
import os

# data visualization
import seaborn as sns;  
import matplotlib.pyplot as plt
sns.set()
from scipy import stats
colors = ["#F472B6", "#94A3B8", "#94A3B8", "#10B981", "#F59E0B", "#EF4444", "#94A3B8"] # color palette
sns.set_palette(sns.color_palette(colors, 7))

# machine learning
from sklearn.linear_model import LassoCV

# video and imagen visualization
from plotly import express as px
#----------------------------------------------------

#----------------------------------------------------
# Prototype of functions
'''
Prototype of functions:

- calculate_weights
- plot_metrics
- process_metrics
- preprocess_lasso
- train_lasso

'''
#----------------------------------------------------



def calculate_weights(df, index_col='input_id'):
  '''    
  Create list weights frequency of each asset

  Parameters
  ----------    
  df: DataFrame
  index_col: index column name 
    
  Return
  -------    
  weight: df with weights
  '''  

  weight = df.groupby(index_col)[index_col].count()
  weight = 1 / weight
  weight = len(weight) * weight / weight.sum()
  weight.replace([np.inf, -np.inf], 0, inplace=True)
  return weight



def plot_metrics(df: pd.DataFrame, index_col: list, metric: str, transform_type: str='log',
                transform_coef: list=[0.00001, 0.0001, 0.001, 0.01, 0.1, 1], lmbda: list=[0.1, 0.3, 0.5, 0.8, 1], 
                norm_metric: bool=True, print_metric=True, figsize=(16,8)):

  '''    
  Preprocesses columns with metrics and returns them normalized and transformed
  Parameters
  ----------    
  df: DataFrame
  index_col: a list of index column names.
  metric: target metric.
  transform_type: apply metrics transformation.
  transform_coef: transformation coefficient.
  lmbda: boxcox parameter. It means how normalize will be the distribution.
  norm_metrics: apply metrics normalization.
  print_metric: Print metric histogram.
  figsize: figure size.
  Return
  -------    
  45 histogram plots of the metrics.
  ''' 

  for tc in range(len(transform_coef)):
      metrics = df[index_col+[metric]].groupby(index_col).first() 
      metrics[metric+"_raw"] = (df[index_col + [metric]].groupby(index_col).first()).replace([np.inf, -np.inf], 0)

      if transform_type=='log':
          metrics[[metric]] = np.log(metrics[[metric]] + transform_coef[tc])

          if norm_metric:
            m = metrics[metric].mean()
            s = metrics[metric].std()
            metrics[metric] = ((metrics[metric] - m) / s).fillna(0)

          if print_metric:
            print(f'Left Plot: {metric} with log normalization')
            print(f'Right Plot: {metric} without normalization') 
            print('tranform_coef = {}'.format(transform_coef[tc]))

            plt.figure(figsize=figsize)
            plt.subplot(1,2,1).set_title(f'{metric}')
            plt.hist(x = metrics[metric], bins = 50)
            plt.subplot(1,2,2).set_title(f'{metric}_raw')
            plt.hist(x = metrics[metric+'_raw'], bins = 50)
            plt.show()
      elif transform_type=='boxcox':
          for l in range(len(lmbda)):
            metrics[metric] = stats.boxcox(metrics[metric] + transform_coef[tc], lmbda=lmbda[l])

            if norm_metric:
              m = metrics[metric].mean()
              s = metrics[metric].std()
              metrics[metric] = ((metrics[metric] - m) / s).fillna(0)

            if print_metric:
              print(f'Left Plot: {metric} with boxcox normalization')
              print(f'Right Plot: {metric} without normalization') 
              print('lmbda = {} tranform_coef = {}'.format(lmbda[l],transform_coef[tc]))

              plt.figure(figsize=figsize)
              plt.subplot(1,2,1).set_title(f'{metric}')
              plt.hist(x = metrics[metric], bins = 50)
              plt.subplot(1,2,2).set_title(f'{metric}_raw')
              plt.hist(x = metrics[metric+'_raw'], bins = 50)
              plt.show()
      else:
        print("We don't have that transformation.")


def process_metrics(df: pd.DataFrame, index_col: list, metric: str, transform_type: str='log', 
                    transform_coef: float=1e-2, lmbda=0.3, norm_metric: bool=True,
                    print_metric=True, figsize=(16,8)):
  '''    
  Preprocesses columns with metrics and returns them normalized and transformed

  Parameters
  ----------    
  df: DataFrame
  index_col: a list of index column names.
  metric: target metric.
  transform_type: apply metrics transformation.
  transform_coef: transformation coefficient.
  lmbda: boxcox parameter. It means how normalize will be the distribution.
  norm_metrics: apply metrics normalization.
  print_metric: Print metric histogram.
  figsize: figure size.

  Return
  -------    
  metrics: df with metrics processed
  '''  

  metrics = df[index_col+[metric]].groupby(index_col).first() 
  metrics[metric+"_raw"] = (df[index_col + [metric]].groupby(index_col).first()).replace([np.inf, -np.inf], 0)

  if transform_type=='log':
    metrics[[metric]] = np.log(metrics[[metric]] + transform_coef)
  elif transform_type=='boxcox':
    metrics[metric] = stats.boxcox(metrics[metric] + transform_coef, lmbda=lmbda)
  else:
    print("don't have this transform")

  if norm_metric:
    m = metrics[metric].mean()
    s = metrics[metric].std()
    metrics[metric] = ((metrics[metric] - m) / s).fillna(0)

  if print_metric:
    metrics.hist(column=metrics.columns, bins=50, figsize=figsize, layout=(1,2))

  return metrics



def preprocess_lasso(df: pd.DataFrame, index_col: list, date_col: str, objective_col: str, metric: str, 
                    tag_type_col: str='tag_type', tag_name_col: str='tag_name', tag_id_col: str='tag_id'):
  
  '''    
  Preprocess df with tags and return dfs that you need for train lasso

  Parameters
  ----------    
  df: DataFrame.
  index_col: index column name.
  date_col: date column name.
  objective_col: campaign objective column name.
  metric: target metric.
  tags_type_col: tags type column name.
  tags_name_col: tags name column name.
  tag_id_col: tags id column name.

  Return
  -------    
  ad_date: df with dates processed
  objectives: df with campaign objectives
  tags: df with tags processed of each asset
  weight: df with weights of each asset
  tags_freq: df with usage frequency of each tag
  metrics: df with metrics processed
  '''  

# Genero df con columna index y fechas
  ad_date = df.groupby(index_col)[date_col].first()

# Genero dataframe con index y objective
  objectives = df.groupby(index_col)[objective_col].first()

# Genero tabla dinamica
  tags = (df.pivot_table(index=index_col, columns=[tag_type_col, tag_name_col],
                          values=tag_id_col, aggfunc='count').fillna(0) > 0).\
                          astype(int).\
                          replace([np.inf, -np.inf], 0)                          

  tags_freq = tags.sum(0).reset_index().rename(columns={'tag_type':'tag_group',
                                                        'tag_name':'tag',
                                                         0:'usage_frequency'})
  
  print('¡Preprocess Lasso finished succesfully! Returning ad_date, objectives, tags and tags_freq.')
  return ad_date, objectives, tags, tags_freq



def train_lasso(df: pd.DataFrame, ad_date: pd.DataFrame, objectives: pd.DataFrame, tags: pd.DataFrame, 
                tags_freq: pd.DataFrame, metrics: pd.DataFrame, weight: pd.DataFrame,
                use_weight: bool=False, use_objective: bool=False, use_time: bool=False, 
                delta_days: int=180, nfolds: int=10, iterations: int=10000):
  '''    
  Train lasso and return results with the best tags, freq and respective scores

  Parameters
  ----------    
  ad_date: df with dates processed
  objectives: df with campaign objectives
  tags: df with tags processed of each asset
  weight: df with weights of each asset
  tags_freq: df with usage frequency of each tag
  metrics: df with metrics processed
  metric: target metric
  use_objective: split assets for objective when train models (Boolean)
  metric_objectives: dict with metrics and respective objectives
  use_time: apply delta time in assets for train model (Boolean)
  delta_days: delta time for select assets (recent model)
  n_folds: number of fold in cross validation
  iterations: number of max iterations to converge models


  Return
  -------    
  results: df with results for each tags (with tag_group, freq, scores, metric)
  '''  
  
  period = ['all']
  metric_objectives={metrics.columns[0]: 'None', metrics.columns[1]: 'None'}


  if use_time: 
    limit_date = ad_date.max() - timedelta(days=delta_days)
    period+=["recent"]

  results = []
  models = {}
  for m in metrics.columns:
      for t in period:
          print(f'training model: {m} for deltatime: {t} and objective: {metric_objectives[m]}')
          if t == "all":
              max_date, min_date = ad_date.max(), ad_date.min()
          else:
              max_date, min_date = ad_date.max(), limit_date

          if use_objective and metric_objectives[m]:
              X = tags[
                  (objectives.isin(metric_objectives[m])) & \
                  (ad_date >= min_date) & (ad_date <= max_date)
              ].copy()
              X.columns = X.columns.droplevel()
              y = metrics[
                  (objectives.isin(metric_objectives[m])) & \
                  (ad_date >= min_date) & (ad_date <= max_date)
              ][m]
              if use_weight:
                W = weight[
                    (objectives.isin(metric_objectives[m])) & \
                    (ad_date >= min_date) & (ad_date <= max_date)]
              else: 
                W = None
          else :
              X = tags[(ad_date >= min_date) & (ad_date <= max_date)].copy()
              X.columns = X.columns.droplevel()      
              y = metrics[(ad_date >= min_date) & (ad_date <= max_date)][m]
              if use_weight:
                W = weight[(ad_date >= min_date) & (ad_date <= max_date)]
              else: 
                W = None

          model = LassoCV(
              cv=nfolds, random_state=0, 
              n_jobs=-1, max_iter=iterations
              ).fit(X, y, sample_weight=W)              
          models[(m,t)] = model
          print('------> OK')

          for var, coef in zip(tags.columns, model.coef_):
              if coef == 0: continue
              results.append({
                  'metric' : m,
                  'period' : t,
                  'tag_group' : var[0],
                  'tag' : var[1],
                  'score' : coef
              })
  
  results = pd.DataFrame(results)

  if len(results)!=0:
    results = results.sort_values(["metric", "period", "tag_group", "tag"])  
    results = results.merge(tags_freq, on=["tag_group", "tag"])
    results["usage_frequency"] = results.usage_frequency / len(tags)

  import matplotlib.pyplot as plt

  fig = plt.figure(figsize=(20,25))
  i = 1
  for k, m in models.items():
      plt.subplot(7, 2, i)
      i+=1
      plt.scatter(m.alphas_, m.mse_path_.mean(axis=-1))
      plt.title(f"{k[0]} on {k[1]} data")
      plt.xlabel(" ")
      plt.ylabel("MSE")

  if use_time:
    print(limit_date)

  return results


print('¡Lasso model library imported succesfully!')


if __name__=='__main__':
  pass