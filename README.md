# Warsaw Air Quality Estimator: Project Overview
*This is the world we live in and this is the air we breathe*  

so it matters! (at least for me :) )  
During the last couple of years we observe alarming level of air polution in Poland. Since I live in Warsaw and this is capital of Poland, I will try to investigate and understand air quality I breathe using historical weather data. Based on that I will build air quality estimator which can predict air polution level based on weather forecast/real time data.

The aim of this project is to utilize machine learning models to predict air quality in Warsaw based on historical weather data. This project consist following steps:
* Build unique dataset based on NASA POWER LARC and Air Quality Historical Data Platform and utilized Pandas for data structuring
* EDA: Analysis and visualisation of data using matplotlib and seaborn. Investigate air polution patterns and impact of the covid 19 lockdown on air quality in Warsaw
* Deployed linear regression, ridge, random forest and boosted methods to identify best model. 
* Optimized best performing model using GridsearchCV to reach the best score. 
* Validate model performance on real time data using IMGW API and web scrapping (BeautifulSoup)


## Technologies Used:

*	Python version 3.8.3
*	Pandas
*	Numpy
*	Matplotlib
*	Seaborn 
*	Location
*	Scikit-Learn
*	XGBoost
*	JSON
*	Requests
*	BeautifulSoup


## Data Preparation
For my analysis I collected data from available resources. Details below:

Date range: 01/01/2015 - 05/31/2021
### Weather data
File: 'nasa_warsaw_weather_clean.csv'  
Source: https://power.larc.nasa.gov/data-access-viewer/![image.png](attachment:image.png)  
  
  Data explenation:
  
- T2M_MAX - Maximum Temperature at 2 Meters (C) 
- T2M_MIN - Minimum Temperature at 2 Meters (C) 
- T2M - Temperature at 2 Meters (C) 
- WS10M_RANGE - Wind Speed Range at 10 Meters (m/s) 
- WS10M - Wind Speed at 10 Meters (m/s) 
- T2M_RANGE - Temperature Range at 2 Meters (C) 
- WS50M - Wind Speed at 50 Meters (m/s) 
- PRECTOT - Precipitation (mm day-1) 
- WS50M_RANGE - Wind Speed Range at 50 Meters (m/s) 
- QV2M - Specific Humidity at 2 Meters (g/kg) 
- PS - Surface Pressure (kPa) 
- RH2M - Relative Humidity at 2 Meters (%)  
  
### Air polution data
File: 'air_qual_marszalkowska.csv'  
Source: https://aqicn.org/data-platform/register/![image.png](attachment:image.png)  
- PM2,5
- PM 10
- O3
- NO2
- SO2
- CO

I merged both files, cleaned data, corrected types and set datetime index.

## EDA
I looked at the general info and distributions of the data. Here are some highlights:
### Temperature
<p align="center">
  <img src="https://github.com/azebryk/Warsaw_air_quality_estimator/blob/master/images/Temperatures.jpg" width=600>
</p>

### PM2.5 level
I created heatmap for each year presenting PM2.5 level. Here is a plot for 2018 (plots for other years can be found in notebook).
<p align="center">
  <img src="https://github.com/azebryk/Warsaw_air_quality_estimator/blob/master/images/PM25_2018.jpg" width=600>
</p>
As it is shown PM2.5 level is significantly higher during winter. Let's see how does PM2.5 level looks on the real plot in terms of temperature and wind.
<p align="center">
  <img src="https://github.com/azebryk/Warsaw_air_quality_estimator/blob/master/images/pm25_temp_wind.JPG" width=600>
</p>

### Covid 19 impact on air quality in Warsaw
During the first weeks of pandemia the world stopped for a second. Schools, factories and offices were closed to prevet spread of the virus.  
Let's see if it changes PM2.5 level in the Warsaw city ceter by comparing it to previous years.
<p align="center">
  <img src="https://github.com/azebryk/Warsaw_air_quality_estimator/blob/master/images/PM25_covid.jpg" width=600>
</p>

#### Comment/Observation:

- Note: 2021 curve is highlighted
- PM2.5 lvl is significantly lower during firts week and in general rather lower than in previous years
- missing days can be observed (2019, 2017)
- 2015 was omitted due to bad quality of data
  
To confirm our observation let's check average temperature and average PM2.5 level in April for each year.

<p align="center">
  <img src="https://github.com/azebryk/Warsaw_air_quality_estimator/blob/master/images/PM25_covid.jpg" width=600>
</p>

### Correlations
Quality of air is obtain by measuring several factors such as PM2.5, PM10, NO2.  
For this analysis I will focus on PM2.5, thus I will remove columns with other air polutions.
<p align="center">
  <img src="https://github.com/azebryk/Warsaw_air_quality_estimator/blob/master/images/Correlation_heatmap.jpg" width=600>
</p>

#### Comment/Observation:

- Winds at 10m and 50m are highly correlated. For further analysis I will use only wind speed at 50m.
- Max, min and avg temperatures plus Specific Humidity at 2 Meters are highly correlated (above 0.9). I will keep only average temperature


## Model Building 
#### Base models
I started from compering models with default hyperparameters. Prior to this, all features were standarized using StandardScaler.
Models, which were compared:
- KNeighborsRegressor,
- LinearRegression,
- Ridge, 
- DecisionTreeRegressor, 
- RandomForestRegressor, 
- GradientBoostingRegressor
- XGBRegressor

#### Tuning hyperparameters using GrudSearchCV

Based on results, I selected the best performing model and I optimized its hyperparameters using GridSearchCV.
Parmeters grid:
<p align="center">
  <img src="https://github.com/azebryk/Warsaw_air_quality_estimator/blob/master/images/param_grid.jpg" width=600>
</p>
Best parameters:

#### Adding new feature
Traffic jams are one of the factor that contributes to bad air quality.
For further improvement of my model performance I decided to introduce new feature - "Weekday", because as I live in Warsaw I see how traffic jams vary for different days. 

## Model Performance Comparison
Tabel below present comparison of model performance comparison.
<p align="center">
  <img src="https://github.com/azebryk/Warsaw_air_quality_estimator/blob/master/images/model_results.jpg" width=600>
</p>
The best baseline model with default parameters was:

* **GradientBoostingRegressor** : MSE = 621.50.
which with further tuning using GridSearchCV ends up with:

* **GradientBoostingRegressor with best parameters** : MSE = 616.62.
and finally adding additional feature "weekday"

* **GradientBoostingRegressor** : MSE = 613.42.

### Feature importance
For better understending of my model I plotted feature importance:
<p align="center">
  <img src="https://github.com/azebryk/Warsaw_air_quality_estimator/blob/master/images/feature importance.jpg" width=600>
</p>
Temperature is the most important feature used to predict PM2.5 level.

## Validation
To check how my model performs I used real-time data.
- Weather data: IMGW API 'https://danepubliczne.imgw.pl/api/data/synop'  
- Air polution data: web scraping using URL: 'https://aqicn.org/city/poland/mazowieckie/warszawa/marszalkowska/pl/'  

Using IMGW API I get current weather in Warsaw. I used my model to predict PM2.5 level, which was later compared with actual PM2.5 level scraped from AQICN using request and BeautifulSoup libraries. 
**Validation run (14th July 2021): MSE = 55.94**


## Code and Resources Used 
**Python Version:** 3.8.3  
**Packages:** pandas, numpy, sklearn, matplotlib, seaborn, requests, BeautifulSoup
**Plote Feature Importance, Source:** https://www.analyseup.com/learn-python-for-data-science/python-random-forest-feature-importance-plot.html


