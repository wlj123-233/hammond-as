import pandas as pd
import numpy as np


train = pd.read_csv('C:\\Users\\49210\Desktop\\21.csv')

#训练样本特征因子化
from pyecharts.charts import Pie, Bar, Map, WordCloud,Line,Grid,Scatter,Radar,Page  #可视化
from pyecharts import options as opts
from pyecharts.globals import SymbolType
from pyecharts.globals import ThemeType
from pyecharts.faker import Faker
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import string
import seaborn as sns

df = train
df['Dates'] = pd.to_datetime(df['Dates'])
df['year'] = pd.to_datetime(df['Dates']).dt.year
df['month'] = pd.to_datetime(df['Dates']).dt.month
df['day'] = pd.to_datetime(df['Dates']).dt.day

df['week']=pd.to_datetime(df['Dates']).dt.dayofweek+1
df.head()
df_cat = df['Category'].value_counts()
district = df['PdDistrict'].value_counts()
Category10 = df_cat.index.tolist()[:10]
Category10_num = df_cat.tolist()[:10]
df_cat10 = df.loc[df['Category'].isin(Category10)]
gp_cat_h = df_cat10.groupby(['Category','month']).size()
gp_cat_h.unstack().T


c = (
        Pie()
            .add(
            "",
            [list(z) for z in zip(district.index.tolist(),
                                  district.values.tolist())],
            radius=["30%", "75%"],
            center=["50%", "50%"],
            rosetype="radius",
            label_opts=opts.LabelOpts(is_show=False),
        )
            .set_global_opts(title_opts=opts.TitleOpts(title="区域占比"),
                             legend_opts=opts.LegendOpts(
                                 orient="vertical", pos_top="20%", pos_left="0.5%"
                             ))
            .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {c}"))
            .render("pie_base.html")
    )
