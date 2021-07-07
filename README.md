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
![alt text](https://github.com/azebryk/Warsaw_air_quality_estimator/blob/master/images/Temperatures.jpg "Salary by Position")
### PM2.5 level
I created heatmap for each year presenting PM2.5 level. Here is a plot for 2018 (plots for other years can be found in notebook)

![alt text](https://github.com/azebryk/Warsaw_air_quality_estimator/blob/master/images/PM25_2018.jpg "Job Opportunities by State")
### Covid 19 impact on air quality in Warsaw

During the first weeks of pandemia the world stopped for a second. Schools, factories and offices were closed to prevet spread of the virus.  
Let's see if it changes PM2.5 level in the Warsaw city ceter by comparing it to previous years.

![alt text](https://github.com/PlayingNumbers/ds_salary_proj/blob/master/correlation_visual.png "Correlations")

## Model Building 

First, I transformed the categorical variables into dummy variables. I also split the data into train and tests sets with a test size of 20%.   

I tried three different models and evaluated them using Mean Absolute Error. I chose MAE because it is relatively easy to interpret and outliers aren’t particularly bad in for this type of model.   

I tried three different models:
*	**Multiple Linear Regression** – Baseline for the model
*	**Lasso Regression** – Because of the sparse data from the many categorical variables, I thought a normalized regression like lasso would be effective.
*	**Random Forest** – Again, with the sparsity associated with the data, I thought that this would be a good fit. 

## Model performance
The Random Forest model far outperformed the other approaches on the test and validation sets. 
*	**Random Forest** : MAE = 11.22
*	**Linear Regression**: MAE = 18.86
*	**Ridge Regression**: MAE = 19.67

## Productionization 
In this step, I built a flask API endpoint that was hosted on a local webserver by following along with the TDS tutorial in the reference section above. The API endpoint takes in a request with a list of values from a job listing and returns an estimated salary. 

## Code and Resources Used 
**Python Version:** 3.8.3  
**Packages:** pandas, numpy, sklearn, matplotlib, seaborn, requests, BeautifulSoup
**Plote Feature Importance, Source:** https://www.analyseup.com/learn-python-for-data-science/python-random-forest-feature-importance-plot.html


