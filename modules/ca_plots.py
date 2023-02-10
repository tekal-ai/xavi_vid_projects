#--------------------------------IMPORT USEFUL LIBRARIES--------------------------------------------------------
# data manipulation
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from operator import add
import os

# data visualization
import seaborn as sns;  import matplotlib.pyplot as plt
sns.set()
from scipy import stats
colors = ["#F472B6", "#94A3B8", "#94A3B8", "#10B981", "#F59E0B", "#EF4444", "#94A3B8"] # color palette
sns.set_palette(sns.color_palette(colors, 7))

# machine learning
from sklearn.linear_model import LassoCV

# video and imagen visualization
#import confidence as cf
from plotly import express as px
from PIL import Image
import urllib.request
from IPython.display import Video, Image

# code visualization
from IPython.core.display import display, HTML # Multiples outputs en una misma línea
display(HTML("<style>.container { width:100% !important; }</style>"))
print('All libraries were successfully imported! -> numpy, pandas, os, datetime from datetime, seaborn, matplotlib.')
print('LassoCV model was saccessfully imported!')


#--------------------------------INSTALL R PACKAGES IN PYTHON------------------------

def gg_scatterplot_install():
  # set R environment for rpy2
    import importlib.util
    import os
    import sys
    
    
    print('Ignore the WinError 2 above')
    
    package = 'rpy2'
    spec = importlib.util.find_spec(package)
    if spec is None:
        print('Installing rpy2...')
        import subprocess
        def install(package):
            package = 'rpy2==3.5.5'
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-Iv", package])
        install(package)
        print('rpy2 installed')
    else:
        print('rpy2 already installed')
    
    
    from rpy2.situation import (r_home_from_subprocess,
                                r_home_from_registry,
                                get_r_home, 
                                assert_python_version)
    R_HOME = get_r_home()
    if not os.environ.get("R_HOME"):
        os.environ['R_HOME'] = R_HOME
    
    packnames = ('ggplot2', 'ggrepel', 'lazyeval','showtext','sysfonts')
    
    import rpy2.robjects.packages as rpackages
    names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
    if len(names_to_install) > 0:
        print('Installing libraries')
        # import R's utility package
        utils = rpackages.importr('utils')
        # select a mirror for R packages
        utils.chooseCRANmirror(ind=1) # select the first mirror in the list
        # R vector of strings
        from rpy2.robjects.vectors import StrVector
        utils.install_packages(StrVector(names_to_install))
        print('Libraries installed')
    else:
        print('Libraries already installed')
    

  
#--------------------------------NEW SCATTER PLOT FOR CMI (LASSO REGRESSION)-------------------

def gg_scatterplot_2(df,metric,period='all',y_max=None,y_min=None,tags_plot=[],\
    y_title='Estimated\nregression\ncoefficient',\
        x_title='Usage frequency',n_tags=15,save=False,filename='scatterplot.png',\
            width=1200,height=800,plot=True, label_size=7, freq=0.5,y_ext=0.02,\
                symbol_size=3,\
                    metrics_positive=['ctr','vvr','likes','shares','comments','vtr'],\
                        metrics_negative=['cpc','cpm','cpmr','cpv'],
                        colors=[],legend_labels=[],shapes=[20,17,8,18,12,13]):

            '''
            Print or save scatterplot with tags and score from regression model
            ----------    
            df: dataframe results, output from lasso regression
            metric: metric to plot, ctr, vvr, likes... (str)
            period: recent or all, period to plot (str)
            y_max, y_min: upper and lower limits of y axis (int or float)
            tags_plot: list of tags to keep in the plot (str)
            y_title, x_title: titles for the axis (str)
            width: figure width in saved file (int)
            height: figure height in saved file (int)
            n_tags: max overlaps for tags labels (int)
            save: to save or not the graph in working directory (bool)
            filename: name plus extension for saved plot (str)
            plot: to display or not the plot in a new cell (bool)
            label_size: size of text labels for each tag (int)
            freq: freq value for threshold in x axis (float)
            y_ext: units to extend the y axis above and below limits (float > 0)
            symbol_size: size of the points/shapes for tags (int)
            metrics_positive: list with metrics that we want to increase (str)
            metrics_negative: list with metrics that we want to decrease (str)
            colors=list of Memorable colors for each quadrant (str)
            legend_labels: list with labels for legend (str)
            shapes: list of shapes codes for tag types (int). 
            If you want to use other symbols search 'R shapes' in Google.
            ----------
            '''
            
            from rpy2.robjects.packages import importr
            from rpy2.robjects import pandas2ri
            pandas2ri.activate()
            import rpy2.robjects.lib.ggplot2 as gg
            gr=importr('ggrepel')
            sf=importr('sysfonts')
            tx=importr('showtext')



            df['action']=''

            for i,x in enumerate(df.usage_frequency.values):
            
                    if df.metric[i] in metrics_positive:
                        
                        if ((x<=freq)&(df.score[i]>0)):
                            df.loc[i,'action']='D'
                        elif((x>freq)&(df.score[i]<=0)):
                            df.loc[i,'action']='B'
                        elif ((x>freq)&(df.score[i]>0)):
                            df.loc[i,'action']='A'
                        elif ((x<=freq)&(df.score[i]<=0)):
                            df.loc[i,'action']="C"
            
                    elif df.metric[i] in metrics_negative:
                        
                        if ((x<=freq)&(df.score[i]>=0)):
                            df.loc[i,'action']="D"
                        elif ((x<=freq)&(df.score[i]<0)):
                            df.loc[i,'action']='C'
                        elif ((x>freq)&(df.score[i]<0)):
                            df.loc[i,'action']='B'
                        elif ((x>freq)&(df.score[i]>=0)):
                            df.loc[i,'action']='A'

            df.tag_group=df.tag_group.str.lower()

            if len(tags_plot)>0:
                df=df.loc[df.tag.isin(tags_plot)]

            sf.font_add_google("Montserrat", "Montserrat")
            tx.showtext_auto()
            myFont1 = "Montserrat"

            pink="#F472B6"
            grey="#94A3B8"
            green="#10B981"
            orange="#F59E0B"
            red="#EF4444"

            if y_max==None: 
                y_max=df[df.metric==metric].score.abs().max()+y_ext
            if y_min==None:
                y_min=-df[df.metric==metric].score.abs().max()-y_ext

            if save:
                grdevices = importr('grDevices')
                grdevices.png(file=filename, width=width, height=height)

            df=df.loc[(df.metric==metric)&(df.period==period)]

            if metric in metrics_positive:
                if len(df.loc[df.action=='A'])>=1:
                    colors.append(green)
                if len(df.loc[df.action=='B'])>=1:
                    colors.append(red)
                if len(df.loc[df.action=='C'])>=1:
                    colors.append(red)
                if len(df.loc[df.action=='D'])>=1:
                    colors.append(green)
            elif metric in metrics_negative:
                if len(df.loc[df.action=='A'])>=1:
                    colors.append(red)
                if len(df.loc[df.action=='B'])>=1:
                    colors.append(green)
                if len(df.loc[df.action=='C'])>=1:
                    colors.append(green)
                if len(df.loc[df.action=='D'])>=1:
                    colors.append(red)

            if len(legend_labels)==0:
                legend_labels=list(df.tag_group.unique())

            shapes=shapes[:len(legend_labels)]

            sctt=gg.ggplot(df)+\
                gg.aes_string(x='usage_frequency',y='score',colour='action',shape='tag_group')+\
                gg.geom_point(size=symbol_size)+\
                gg.scale_colour_manual(values=colors,guide='none')+\
                gg.scale_shape_manual(values=shapes,\
                labels=legend_labels)+\
                gg.labs(x=x_title,y=y_title,shape='Tag type')+\
                gg.geom_hline(yintercept=0,colour='black',size=1)+\
                gg.geom_vline(xintercept=freq,colour='black',size=1)+\
                gg.ylim(y_min,y_max)+\
                gg.xlim(-0.0005,1.02)+\
                gg.theme(**{'text':gg.element_text(family=myFont1),\
                        'panel.background':gg.element_blank(),\
                        'panel.grid.major':gg.element_line(colour='lightgrey'),\
                        'axis.text':gg.element_text(size=9),\
                        'axis.title.y':gg.element_text(angle=0,vjust=0.5,size=12),\
                        'axis.title.x':gg.element_text(hjust=0.5,size=12),\
                        'axis.ticks':gg.element_blank(),\
                        'legend.key':gg.element_rect(fill='white'),\
                        'legend.text':gg.element_text(family=myFont1,size=10),\
                        'legend.title':gg.element_text(family=myFont1,size=10,face='bold')})+\
                gr.geom_label_repel(gg.aes(label=df.tag[df.metric==metric]),\
                        **{'segment.color':'grey','max.overlaps':15,\
                        'show.legend':False,'size':label_size})

            if save:
                plot=False
                sctt.plot()
                grdevices.dev_off()

            if plot:
                import warnings
                warnings.filterwarnings("ignore")
                from rpy2.ipython.ggplot import image_png
                display(image_png(sctt))

#--------------------------------OLD SCATTER PLOT FOR CMI (LASSO REGRESSION)-------------------

def plot_lasso_scatter(results, metric, period="all"):
    fig = px.scatter(
        results[(results.period==period) & (results.metric==metric)],
        y="score",
        x="usage_frequency",
        color="tag_group",
        text="tag",
        title=f"Score vs usage frequency - {metric.upper()} - {period.upper()}"
        )
    fig = fig.update_layout(height=1024, width=1024)
    fig = fig.update_traces(textposition='top center', textfont_size=14)
    return fig


#--------------------------------SCATTER PLOT FOR LASSO IN R------------------------

def gg_scatterplot(df,metric,y_max=None,y_min=None,\
    y_title='Estimated regression coefficient',\
        x_title='Usage frequency',n_tags=15,save=False,filename='scatterplot.png',\
            width=900,height=600,plot=True, label_size=4):

            '''
            Return or save scatterplot with tags and score from regression model
            ----------    
            df: dataframe
            metric: metric to plot, ctr, vvr, likes...
            y_max, y_min: upper and lower limits of y axis
            y_title, x_title: titles for the axis
            width: figure width in saved file
            height: figure height in saved file
            n_tags: max overlaps for tags labels
            save: to save or not the graph in working directory
            filename: name plus extension for saved plot
            plot: to display or not the plot in a new cell
            ----------
            '''
            from IPython.display import display
            import os
            from rpy2.robjects.packages import importr
            from rpy2.robjects import pandas2ri
            pandas2ri.activate()
            import rpy2.robjects.lib.ggplot2 as gg
            gr=importr('ggrepel')

            if y_max==None: 
                y_max=df[df.metric==metric].score.abs().max()
            if y_min==None:
                y_min=-df[df.metric==metric].score.abs().max()

            if save:
                grdevices = importr('grDevices')
                grdevices.png(file=filename, width=width, height=height)

            sctt=gg.ggplot(df[df.metric==metric])+\
                gg.aes_string(x='usage_frequency',y='score',colour='tag_group')+\
                    gg.geom_point()+gg.labs(x=x_title,y=y_title,colour="Tag type\n")+\
                        gg.geom_hline(yintercept=10,linetype='dashed',colour='dimgrey')+\
                            gg.ylim(y_min,y_max)+\
                                gg.xlim(0,1)+\
                                gg.theme(**{'panel.background':gg.element_blank(),\
                                    'panel.grid':gg.element_line(colour='lightgrey'),\
                                        'axis.line':gg.element_line(colour='black'),\
                                            'axis.title':gg.element_text(size=20)})+\
                                                gr.geom_label_repel(gg.aes(label=df.tag[df.metric==metric]),\
                                                    **{'segment.color':'grey','max.overlaps':15,\
                                                        'show.legend':False, 'size' : label_size})

            if save:
                plot=True
                sctt.plot()
                grdevices.dev_off()

            if plot:
                import warnings
                warnings.filterwarnings("ignore")
                from rpy2.ipython.ggplot import image_png
                display(image_png(sctt))