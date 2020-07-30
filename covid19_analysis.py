#!/usr/bin/env python3

import pandas as pd
import numpy as np

import os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pylab as plot

WINDOW = 100	# sliding window size for calculating N, A, T, and D
PLOT_FONT_SIZE = 26
LEGEND_FONT_SIZE = 20
params = {'legend.fontsize': LEGEND_FONT_SIZE,
          'legend.handlelength': 2}
plot.rcParams.update(params)

IMG_FOLDER = "/opendata/data/COVID19-Analysis/images/"
if not os.path.exists(IMG_FOLDER):
  os.makedirs(IMG_FOLDER)

PLOT_Countries = {
    "Asia"    : {
        "fname" : "asia",
        "countries" : ['Taiwan','Japan','Thailand','Vietnam','Singapore',
                      'Philippines','Hong Kong, China','Macau, China', 
                      'China', 'Korea, South']
    },
    "Europe"  : {
        "fname" : "europe",
        "countries" : ['Italy','Iceland','France','United Kingdom','Germany',
                      'Spain','Portugal','Russia','Sweden','Czechia']
    },
    "America & Oceania" : {
        "fname" : "america",
        "countries" : ['US','Mexico','Brazil','Argentina','Chile',
                      'Venezuela','New Zealand','Australia']
    },
    "Others"  : {
        "fname" : "other",
        "countries" : ['India','Iran','Israel','Turkey',
                       'Egypt','Nigeria','Kenya']
    },
    "Case 1"  : {
        "fname" : "case1",
        "countries" : ['Taiwan','Iceland','Germany','New Zealand',"Malta",
                       'Mauritius','San Marino','Malaysia']
    },   
}

# confirmed cases
df1 = pd.read_csv('https://data.humdata.org/hxlproxy/api/data-preview.csv?url=https%3A%2F%2Fraw.githubusercontent.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series%2Ftime_series_covid19_confirmed_global.csv&filename=time_series_covid19_confirmed_global.csv', keep_default_na=False)  
# deadth cases
df2 = pd.read_csv('https://data.humdata.org/hxlproxy/api/data-preview.csv?url=https%3A%2F%2Fraw.githubusercontent.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series%2Ftime_series_covid19_deaths_global.csv&filename=time_series_covid19_deaths_global.csv', keep_default_na=False)
# recovered cases
df3 = pd.read_csv('https://data.humdata.org/hxlproxy/api/data-preview.csv?url=https%3A%2F%2Fraw.githubusercontent.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series%2Ftime_series_covid19_recovered_global.csv&filename=time_series_covid19_recovered_global.csv', keep_default_na=False)


for df in [df1, df2, df3]:
  for index, row in df.iterrows():
    if row['Country/Region'] == "Taiwan*":
     df.at[index,'Country/Region'] = "Taiwan"
    elif row['Country/Region'] == "China":
      if row['Province/State'] in ["Hong Kong", "Macau"]:
        row['Country/Region'] = row['Province/State'] + ", " + row['Country/Region']
        df.at[index,'Country/Region'] = row['Country/Region']
    elif row['Country/Region'] in ["Denmark","France","Netherlands","United Kingdom"]:
      if row['Province/State']:
        row['Country/Region'] = row['Country/Region'] + " (" + row['Province/State'] + ")"
        df.at[index,'Country/Region'] = row['Country/Region']

  df.drop(df[df['Country/Region'] == "Diamond Princess"].index, inplace=True)

  for h in df.columns:
    if h not in ["Country/Region", "Province/State", "Lat", "Long"]:
      df[h] = df[h].astype(float)

df_groups = df1.groupby(df1["Country/Region"])
df1 = df_groups.sum().reset_index()
df_groups = df2.groupby(df2['Country/Region'])
df2 = df_groups.sum().reset_index()
df_groups = df3.groupby(df3['Country/Region'])
df3 = df_groups.sum().reset_index()

for col in ['Province/State','Lat','Long']:
  if col in df1.columns:
    df1 = df1.drop(columns=col)
    df2 = df2.drop(columns=col)
    df3 = df3.drop(columns=col)


# N: total number of unrecovered cases, which is equal to the # of confirmed cases - # of death - # of recovered cases
N = df1.copy()

for index, row in df1.iterrows():
  country = df1.at[index,"Country/Region"]
  index2 = df2["Country/Region"].loc[lambda x: x==country].index.tolist()
  index3 = df3["Country/Region"].loc[lambda x: x==country].index.tolist()
  if len(index2)>0:
    index2 = index2[0]
  else:
    index2 = -1
  if len(index3)>0:
    index3 = index3[0]
  else:
    index3 = -1
  
  for h in N.columns:
    if h != "Country/Region":
      n1 = df1.at[index,h]
      if index2>=0:
        n2 = df2.at[index2,h]
      else:
        n2 = 0
      if index3>=0:
        n3 = df3.at[index3,h]
      else:
        n3 = 0
      N.at[index,h] = n1 - n2 - n3


# A: the arrival rate of the confirmed cases
A = df1.copy()
if WINDOW > len(A.columns):
  WINDOW = len(A.columns)
for i in range(1,WINDOW):
  A[A.columns[i]] = A[A.columns[i]] / i

for i in range(WINDOW,len(A.columns)):
  A[A.columns[i]] = (df1[A.columns[i]] - df1[A.columns[i-WINDOW+1]]) / WINDOW



# T: the average time for each case in the COVID-19 status
T = df1.copy()

for i in range(2,len(T.columns)):
  for index, row in T.iterrows():
    if A.iat[index,i]==0:
      if T.iat[index,i-1]>0:
        T.iat[index,i] = T.iat[index,i-1] + 1
      else:
        T.iat[index,i] = 0
    else:
      T.iat[index,i] = N.iat[index,i] / A.iat[index,i]


# D: the deadth rate among all confirmed cases
D = df1.copy()

for index, row in df1.iterrows():
  country = df1.at[index,"Country/Region"]
  index2 = df2["Country/Region"].loc[lambda x: x==country].index.tolist()
  if len(index2)>0:
    index2 = index2[0]
  else:
    index2 = -1

  for h in D.columns:
    if h != "Country/Region":
      n1 = int(df1.at[index,h])
      if n1==0:
        n1 = 1.0
      if index2>=0:
        n2 = int(df2.at[index2,h])
      else:
        n2 = 0.0
      D.at[index,h] = n2 * 1.0 / n1
      #print(n2,n1,D.at[index,h])

T2 = T.copy()
T2 = T2.set_index('Country/Region').T
A2 = A.copy()
A2 = A2.set_index('Country/Region').T
D2 = D.copy()
D2 = D2.set_index('Country/Region').T

T2.index = pd.to_datetime(T2.index)
A2.index = pd.to_datetime(A2.index)
D2.index = pd.to_datetime(D2.index)

for country in PLOT_Countries:
  plot = T2.plot(ylim=(0,160),figsize=(20,10),logy=False,fontsize=26,y=PLOT_Countries[country]["countries"])
  plt.xlabel('Date', fontsize=PLOT_FONT_SIZE)
  plt.ylabel('Average COVID-19 Case Time (days)', fontsize=PLOT_FONT_SIZE)
  #plt.show()
  fig = plot.get_figure()
  fig.savefig(IMG_FOLDER + "/latest_" + PLOT_Countries[country]["fname"] + "_T.png", bbox_inches='tight')
  plt.close(fig)

  plot = A2.plot(figsize=(20,10),logy=True,fontsize=26,y=PLOT_Countries[country]["countries"])
  plt.xlabel('Date', fontsize=PLOT_FONT_SIZE)
  plt.ylabel('New COVID-19 Case Rate', fontsize=PLOT_FONT_SIZE)
  #plt.show()
  fig = plot.get_figure()
  fig.savefig(IMG_FOLDER + "/latest_" + PLOT_Countries[country]["fname"] + "_A.png", bbox_inches='tight')
  plt.close(fig)



########

to_show = []
for index, row in T.iterrows():
  if T.at[index,T.columns[len(T.columns)-1]]>=120:
    to_show.append(T.at[index,T.columns[0]])

plot = T2.plot(ylim=(0,400),figsize=(20,10),logy=False,fontsize=26,y=to_show)
plt.xlabel('Date', fontsize=PLOT_FONT_SIZE)
plt.ylabel('Average COVID-19 Case Time (days)', fontsize=PLOT_FONT_SIZE)
#plt.show()
fig = plot.get_figure()
fig.savefig(IMG_FOLDER + "/latest_worst_T.png", bbox_inches='tight')
plt.close(fig)

plot = A2.plot(ylim=(0,5000),figsize=(20,10),logy=True,fontsize=26,y=to_show)
plt.xlabel('Date', fontsize=PLOT_FONT_SIZE)
plt.ylabel('New COVID-19 Case Rate', fontsize=PLOT_FONT_SIZE)
#plt.show()
fig = plot.get_figure()
fig.savefig(IMG_FOLDER + "/latest_worst_A.png", bbox_inches='tight')
plt.close(fig)


to_show = []
for index, row in T.iterrows():
  if T.at[index,T.columns[len(T.columns)-1]]<20 and df1.at[index,T.columns[len(T.columns)-1]] > 400 and A.at[index,T.columns[len(T.columns)-1]]<50:
    to_show.append(T.at[index,T.columns[0]])

plot = T2.plot(ylim=(0,120),figsize=(20,10),logy=False,fontsize=26,y=to_show)
plt.xlabel('Date', fontsize=PLOT_FONT_SIZE)
plt.ylabel('Average COVID-19 Case Time (days)', fontsize=PLOT_FONT_SIZE)
#plt.show()
fig = plot.get_figure()
fig.savefig(IMG_FOLDER + "/latest_best_T.png", bbox_inches='tight')
plt.close(fig)

plot = A2.plot(ylim=(0,100),figsize=(20,10),logy=False,fontsize=26,y=to_show)
plt.xlabel('Date', fontsize=PLOT_FONT_SIZE)
plt.ylabel('New COVID-19 Case Rate', fontsize=PLOT_FONT_SIZE)
#plt.show()
fig = plot.get_figure()
fig.savefig(IMG_FOLDER + "/latest_best_A.png", bbox_inches='tight')
plt.close(fig)


to_show = []
for index, row in T.iterrows():
  if df1.at[index,T.columns[len(T.columns)-1]] < 20:
    to_show.append(T.at[index,T.columns[0]])

plot = T2.plot(ylim=(0,120),figsize=(20,10),logy=False,fontsize=26,y=to_show)
plt.xlabel('Date', fontsize=PLOT_FONT_SIZE)
plt.ylabel('Average COVID-19 Case Time (days)', fontsize=PLOT_FONT_SIZE)
#plt.show()
fig = plot.get_figure()
fig.savefig(IMG_FOLDER + "/latest_rare_T.png", bbox_inches='tight')
plt.close(fig)

plot = A2.plot(ylim=(0,100),figsize=(20,10),logy=False,fontsize=26,y=to_show)
plt.xlabel('Date', fontsize=PLOT_FONT_SIZE)
plt.ylabel('New COVID-19 Case Rate', fontsize=PLOT_FONT_SIZE)
#plt.show()
fig = plot.get_figure()
fig.savefig(IMG_FOLDER + "/latest_rare_A.png", bbox_inches='tight')
plt.close(fig)


to_show = []
for index, row in T.iterrows():
  if A.at[index,T.columns[len(T.columns)-1]] < 1 and df1.at[index,T.columns[len(T.columns)-1]] > 400:
    to_show.append(T.at[index,T.columns[0]])

plot = T2.plot(ylim=(0,120),figsize=(20,10),logy=False,fontsize=26,y=to_show)
plt.xlabel('Date', fontsize=PLOT_FONT_SIZE)
plt.ylabel('Average COVID-19 Case Time (days)', fontsize=PLOT_FONT_SIZE)
#plt.show()
fig = plot.get_figure()
fig.savefig(IMG_FOLDER + "/latest_zero_T.png", bbox_inches='tight')
plt.close(fig)

plot = A2.plot(figsize=(20,10),logy=False,fontsize=26,y=to_show)
plt.xlabel('Date', fontsize=PLOT_FONT_SIZE)
plt.ylabel('New COVID-19 Case Rate', fontsize=PLOT_FONT_SIZE)
#plt.show()
fig = plot.get_figure()
fig.savefig(IMG_FOLDER + "/latest_zero_A.png", bbox_inches='tight')
plt.close(fig)


to_show = []
for index, row in D.iterrows():
  if D.at[index,D.columns[len(D.columns)-1]]>0.1:
    to_show.append(D.at[index,D.columns[0]])
    #print(D.at[index,D.columns[len(D.columns)-1]], df1.at[index,D.columns[len(D.columns)-1]], df2.at[index,D.columns[len(D.columns)-1]], D.at[index,D.columns[0]])


plot = D2.plot(ylim=(0,0.4),figsize=(20,10),logy=False,fontsize=26,y=to_show)
plt.xlabel('Date', fontsize=PLOT_FONT_SIZE)
plt.ylabel('Death ratio (%)', fontsize=PLOT_FONT_SIZE)
#plt.show()
fig = plot.get_figure()
fig.savefig(IMG_FOLDER + "/latest_deathly_D.png", bbox_inches='tight')
plt.close(fig)

plot = A2.plot(ylim=(0.1,5000),figsize=(20,10),logy=True,fontsize=26,y=to_show)
plt.xlabel('Date', fontsize=PLOT_FONT_SIZE)
plt.ylabel('New COVID-19 Case Rate', fontsize=PLOT_FONT_SIZE)
#plt.show()
fig = plot.get_figure()
fig.savefig(IMG_FOLDER + "/latest_deathly_A.png", bbox_inches='tight')
plt.close(fig)


output_df = pd.DataFrame({"Country":T[T.columns[0]],"N":N[T.columns[len(T.columns)-1]],"A":A[T.columns[len(T.columns)-1]],"T":T[T.columns[len(T.columns)-1]],"D":D[T.columns[len(T.columns)-1]]})
output_df_t = output_df.set_index('Country').T

plot = output_df.plot.scatter(x='T', y='A', xlim=(0.1,200),ylim=(0.0001,10000),figsize=(15,15),fontsize=26,logy=True, logx=True, c='DarkBlue')
plt.xlabel('Average Case Time (days)', fontsize=PLOT_FONT_SIZE)
plt.ylabel('Average Case Rate (cases/day)', fontsize=PLOT_FONT_SIZE)
plt.show()
fig = plot.get_figure()
fig.savefig(IMG_FOLDER + "/latest_all_A-T.png", bbox_inches='tight')
plt.close(fig)

plot = output_df.plot.scatter(x='T', y='D', xlim=(0.1,200),ylim=(0.0001,1),figsize=(15,15),fontsize=26,logy=True, logx=True, c='DarkBlue')
plt.xlabel('Average Case Time (days)', fontsize=PLOT_FONT_SIZE)
plt.ylabel('Death ratio (%)', fontsize=PLOT_FONT_SIZE)
plt.show()
fig = plot.get_figure()
fig.savefig(IMG_FOLDER + "/latest_all_D-T.png", bbox_inches='tight')
plt.close(fig)

plot = output_df.plot.scatter(x='N', y='A', xlim=(1,10000000),ylim=(0.0001,100000),figsize=(15,15),fontsize=26,logy=True, logx=True, c='DarkBlue')
plt.xlabel('Average Active Case Number (cases)', fontsize=PLOT_FONT_SIZE)
plt.ylabel('Average Case Rate (cases/day)', fontsize=PLOT_FONT_SIZE)
plt.show()
fig = plot.get_figure()
fig.savefig(IMG_FOLDER + "/latest_all_A-N.png", bbox_inches='tight')
plt.close(fig)

plot = output_df.plot.scatter(x='N', y='D', xlim=(1,10000000),ylim=(0.0001,1),figsize=(15,15),fontsize=26,logy=True, logx=True, c='DarkBlue')
plt.xlabel('Average Active Case Number (cases)', fontsize=PLOT_FONT_SIZE)
plt.ylabel('Death ratio (%)', fontsize=PLOT_FONT_SIZE)
plt.show()
fig = plot.get_figure()
fig.savefig(IMG_FOLDER + "/latest_all_D-N.png", bbox_inches='tight')
plt.close(fig)
