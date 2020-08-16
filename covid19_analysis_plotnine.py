#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import os

import warnings
warnings.filterwarnings("ignore")

from plotnine import *
from plotnine.data import *


import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M',
                    handlers=[logging.FileHandler('covid19_ana.log', 'w', 'utf-8'), ])

# logging.debug('Hello debug!')
# logging.info('Hello info!')
# logging.warning('Hello warning!')
# logging.error('Hello error!')
# logging.critical('Hello critical!')


WINDOW = 1000    # sliding window size for calculating N, A, T, and D
PLOT_FONT_SIZE = 26
LEGEND_FONT_SIZE = 20
params = {'legend.fontsize': LEGEND_FONT_SIZE,
          'legend.handlelength': 2}

IMG_FOLDER = "./images_plotnine/"
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
    "Case 2"  : {
        "fname" : "tw_flight",
        "countries" : ['Canada', 'US', 'Macau, China', 'Hong Kong, China', 
                       'China', 'Japan', 'Philippines', 'Vietnam', 'Thailand', 
                       'United Arab Emirates', 'Korea, South', 'Malaysia', 
                       'Singapore', 'Cambodia', 'Indonesia', 'Netherlands', 
                       'France', 'United Kingdom']  
    }, 
}


# confirmed cases
df1 = pd.read_csv('https://data.humdata.org/hxlproxy/api/data-preview.csv?url=https%3A%2F%2Fraw.githubusercontent.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series%2Ftime_series_covid19_confirmed_global.csv&filename=time_series_covid19_confirmed_global.csv', keep_default_na=False)  
# deadth cases
df2 = pd.read_csv('https://data.humdata.org/hxlproxy/api/data-preview.csv?url=https%3A%2F%2Fraw.githubusercontent.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series%2Ftime_series_covid19_deaths_global.csv&filename=time_series_covid19_deaths_global.csv', keep_default_na=False)
# recovered cases
df3 = pd.read_csv('https://data.humdata.org/hxlproxy/api/data-preview.csv?url=https%3A%2F%2Fraw.githubusercontent.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series%2Ftime_series_covid19_recovered_global.csv&filename=time_series_covid19_recovered_global.csv', keep_default_na=False)

logging.info( "Finish query data from data.humdata.org.\nWith confirmed_global: {} rows, deaths_global: {} rows, recovered_global: {} rows.\nThe latest date: {}.".format(len(df1), len(df2), len(df3), df1.columns[-1]) )


def clean_df( df ):
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
    
    df_groups = df.groupby(df["Country/Region"])
    df = df_groups.sum().reset_index()
    
    for col in ['Province/State','Lat','Long']:
        if col in df.columns:
            df = df.drop(columns=col)
            
    return df
        
df1 = clean_df(df1)
df2 = clean_df(df2)
df3 = clean_df(df3)


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



# T: the average recovery time for each case in the COVID-19 status
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

T3 = T2.reset_index()
A3 = A2.reset_index()
D3 = D2.reset_index()

logging.info( "Finish analysis raw datas.\nWith T2: {} colums, A2: {} colums, D2: {} colums.\nThe latest date: {}.".format(len(T2.columns), len(A2.columns), len(D2.columns), A2.index[-1]) )


def draw_T( DF, X="date", Y="count", COLOR="Country/Region", Ylim=(0,161), Xlab='Date', Ylab="", Breaks=[], Minor_breaks=[], FILENAME = "", TITLE = "" ):
    pic = (
        ggplot(DF, aes(x=X, y=Y, ymax=Y, ymin=0, color=COLOR))
        + geom_line(size=1)
#         + geom_ribbon(alpha=0.1)
#         + geom_area(colour = 'black', size =1, alpha = .7)
        + coord_cartesian(ylim=Ylim)
        + xlab(Xlab)
        + ylab(Ylab)
        + scale_y_continuous(breaks=Breaks, minor_breaks=Minor_breaks)
        + scale_x_datetime(date_breaks='2 week', date_minor_breaks="1 week", date_labels="%b-%d", expand=(0,0))
        + ggtitle(TITLE)
        + theme_light()
    )
    pic.save( filename=FILENAME, format="png", path=IMG_FOLDER, width=20, height=10, dpi=100, limitsize=False )
    

def draw_A( DF, X="date", Y="count", COLOR="Country/Region", Xlab='Date', Ylab="", FILENAME = "", TITLE = "" ):
    pic = (
        ggplot(DF, aes(x=X, y=Y, ymax=Y, ymin=0, color=COLOR))
        + geom_line(size=1) 
        + xlab(Xlab)
        + ylab(Ylab)
        + scale_y_continuous(trans = 'log10')
        + scale_x_datetime(date_breaks='2 week', date_minor_breaks="1 week", date_labels="%b-%d", expand=(0,0))
        + ggtitle(TITLE)
        + theme_light()
    )
    pic.save( filename=FILENAME, format="png", path=IMG_FOLDER, width=20, height=10, dpi=100, limitsize=False )


def draw_D( DF, X="date", Y="count", COLOR="Country/Region", Xlab='Date', Ylab="", FILENAME = "", TITLE = "" ):
    pic = (
        ggplot(DF, aes(x=X, y=Y, ymax=Y, ymin=0, color=COLOR))
        + geom_line(size=1) 
        + xlab(Xlab)
        + ylab(Ylab)
        + scale_x_datetime(date_breaks='2 week', date_minor_breaks="1 week", date_labels="%b-%d", expand=(0,0))
        + ggtitle(TITLE)
        + theme_light()
    )
    pic.save( filename=FILENAME, format="png", path=IMG_FOLDER, width=20, height=10, dpi=100, limitsize=False )


def X_Y(DF, X, Y, Xlab, Ylab, FILENAME, Xlim=[], Ylim=(0,2), TITLE="" ):
    pic = (
        ggplot(DF, aes(x=X, y=Y))
        + geom_point(fill='DarkBlue', size=1)
        + geom_smooth(method='lm', fill="green")
        + xlab(Xlab)
        + ylab(Ylab)
        + scale_y_log10()
        + scale_x_log10(expand=(0,0))
        + xlim(Xlim[0], Xlim[1])
        + coord_cartesian(ylim=Ylim)
        + ggtitle(TITLE)
        + theme_light()
        )

    pic.save( filename=FILENAME, format="png", path=IMG_FOLDER, width=10, height=10, dpi=100, limitsize=False )

def Y_Y(DF, X, Y, Xlab, Ylab, FILENAME, Xlim=(0,2), Ylim=(0,2), TITLE="" ):
    pic = (
        ggplot(DF, aes(x=X, y=Y))
        + geom_point(fill='DarkBlue', size=1)
        + geom_smooth(method='lm', fill="green")
        + xlab(Xlab)
        + ylab(Ylab)
        + scale_y_log10()
        + scale_x_log10(expand=(0,0))
        + coord_cartesian(xlim=xlim)
        + coord_cartesian(ylim=Ylim)
        + ggtitle(TITLE)
        + theme_light()
        )

    pic.save( filename=FILENAME, format="png", path=IMG_FOLDER, width=10, height=10, dpi=100, limitsize=False )

def draw_to_show( to_show, Ymax=200, TYPE="best" ):
    T_best = pd.DataFrame()
    for C in to_show:
        T_C = pd.DataFrame()
        T_C["count"] = T3[C].tolist()
        T_C["Country/Region"] = C
        T_C["date"] = T3["index"]
        T_best = T_best.append( T_C, ignore_index=True )

    draw_T( T_best, Y="count", Ylim=(0,Ymax), Ylab="Average Recovery Time (days)", Breaks=range(0, Ymax,30), Minor_breaks=range(0,Ymax,10), FILENAME = "latest_" + TYPE + "_T.png", TITLE = 'Latest ' + TYPE.capitalize() + ' Average Recovery Time' )

    A_worst = pd.DataFrame()
    for C in to_show:
        A_C = pd.DataFrame()
        A_C["count"] = A3[C].tolist()
        A_C["Country/Region"] = C
        A_C["date"] = A3["index"]
        A_worst = A_worst.append( A_C, ignore_index=True )

    draw_A( A_worst, Y="count", Ylab="Onset Rate (#/days)", FILENAME = "latest_" + TYPE + "_A.png", TITLE = 'Latest ' + TYPE.capitalize() + ' Onset Rate' )
    logging.info("Finish drawing A,T plot of latest_{}".format(TYPE))


for country in PLOT_Countries:
    
    ##### For T #####
    T_country = pd.DataFrame()
    for C in PLOT_Countries[country]["countries"]:
        T_C = pd.DataFrame()
        T_C["count"] = T3[C].tolist()
        T_C["Country/Region"] = C
        T_C["date"] = T3["index"]
        T_country = T_country.append( T_C, ignore_index=True )
        
    draw_T( T_country, Y="count", Ylim=(0,161), Ylab="Average Recovery Time (days)", Breaks=range(0, 161, 20), Minor_breaks=range(0,161,5), FILENAME = "latest_" + PLOT_Countries[country]["fname"] + "_T.png", TITLE = 'Average Recovery Time in ' + country )
    
    ##### For A #####
    A_country = pd.DataFrame()
    for C in PLOT_Countries[country]["countries"]:
        A_C = pd.DataFrame()
        A_C["count"] = A3[C].tolist()
        A_C["Country/Region"] = C
        A_C["date"] = A3["index"]
        A_country = A_country.append( A_C, ignore_index=True )
        
    draw_A( A_country, Y="count", Ylab="Onset Rate (#/days)", FILENAME = "latest_" + PLOT_Countries[country]["fname"] + "_A.png", TITLE = 'Onset Rate in ' + country )
    
    
    logging.info("Finish drawing A,T plot of {}".format( PLOT_Countries[country]["fname"] ))

    
##### Best ####
to_show = []
for index, row in T.iterrows():
    if T.at[index,T.columns[len(T.columns)-1]]<20 and df1.at[index,T.columns[len(T.columns)-1]] > 400 and A.at[index,T.columns[len(T.columns)-1]]<50:
        to_show.append(T.at[index,T.columns[0]])
        
draw_to_show( to_show, Ymax=210, TYPE="best" )

##### Worst ####
to_show = []
for index, row in T.iterrows():
    if T.at[index,T.columns[len(T.columns)-1]]>=120:
        to_show.append(T.at[index,T.columns[0]])

draw_to_show( to_show, Ymax=220, TYPE="worst" )

##### Rare #####
to_show = []
for index, row in T.iterrows():
    if df1.at[index,T.columns[len(T.columns)-1]] < 20:
        to_show.append(T.at[index,T.columns[0]])
        
draw_to_show( to_show, Ymax=200, TYPE="rare" )

##### Zero #####
to_show = []
for index, row in T.iterrows():
    if A.at[index,T.columns[len(T.columns)-1]] < 3 and df1.at[index,T.columns[len(T.columns)-1]] > 400:
        to_show.append(T.at[index,T.columns[0]])
        
draw_to_show( to_show, Ymax=200, TYPE="zero" )

##### Deathly #####
to_show = []
for index, row in D.iterrows():
    if D.at[index,D.columns[len(D.columns)-1]]>0.1:
        to_show.append(D.at[index,D.columns[0]])

A_deathly = pd.DataFrame()
TYPE = "deathly"
for C in to_show:
    A_C = pd.DataFrame()
    A_C["count"] = A3[C].tolist()
    A_C["Country/Region"] = C
    A_C["date"] = A3["index"]
    A_deathly = A_deathly.append( A_C, ignore_index=True )


draw_A( A_deathly, Y="count", Ylab="Onset Rate (#/days)", FILENAME = "latest_" + TYPE + "_A.png", TITLE = 'Latest ' + TYPE.capitalize() + ' Onset Rate' )

D_deathly = pd.DataFrame()
for C in to_show:
    D_C = pd.DataFrame()
    D_C["count"] = D3[C].tolist()
    D_C["Country/Region"] = C
    D_C["date"] = D3["index"]
    D_deathly = D_deathly.append( D_C, ignore_index=True )

draw_D( D_deathly, Y="count", Ylab="Death ratio (%)", FILENAME = "latest_deathly_D.png", TITLE = 'Latest ' + TYPE.capitalize() + ' Death ratio' )
logging.info("Finish drawing A,D plot of latest_{}".format("Deathly"))


output_df = pd.DataFrame({"Country":T[T.columns[0]],"N":N[T.columns[len(T.columns)-1]],"A":A[T.columns[len(T.columns)-1]],"T":T[T.columns[len(T.columns)-1]],"D":D[T.columns[len(T.columns)-1]]})

X_Y( output_df, X="T", Y="A", Xlab="Average Recovery Time (days)", Ylab="Onset Rate (#/days)", FILENAME="latest_all_A-T.png", Xlim=[0.1,200], Ylim=(-2,5), TITLE="latest_all_A-T" )
X_Y( output_df, X="T", Y="D", Xlab="Average Recovery Time (days)", Ylab="Death ratio (%)", FILENAME="latest_all_D-T.png", Xlim=[0.1,200], Ylim=(-3,0), TITLE="latest_all_D-T" )
Y_Y( output_df, X="N", Y="A", Xlab="Active Case Number (cases)", Ylab="Onset Rate (#/days)", FILENAME="latest_all_A-N.png", Xlim=(-1,6), Ylim=(-2,5), TITLE="latest_all_A-N" )
Y_Y( output_df, X="N", Y="D", Xlab="Active Case Number (cases)", Ylab="Death ratio (%)", FILENAME="latest_all_D-N.png", Xlim=(-1,7), Ylim=(-4,1), TITLE="latest_all_D-N" )
logging.info("Finish drawing last_all_ plot between T-A, T-D, N-A, N-D")



