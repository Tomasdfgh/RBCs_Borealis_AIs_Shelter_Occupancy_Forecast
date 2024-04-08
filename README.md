# Optimising Homeless Shelter Operations in Toronto: A Machine Learning Approach
Let's Solve It (2024)

_Project by Nida Copty, Tom Nguyen, and India Tory_

## Problem Statement
### Background
In Toronto, over 9000 people are homeless each night due to various factors like family violence, mental illness, and job loss. The city provides services such as shelters, respite sites, drop-in programs, and warming centres to assist them. Recently, there has been unprecedented pressure on the shelter system due to the increased cost of living and lack of affordable housing. Despite having one of the largest shelter systems in the country, Toronto shelters had to turn away almost 200 people each day in December. Homelessness support services in Toronto encounter difficulties in forecasting the demand for shelter beds, resources, and staff. The current reactive approach relies on outreach efforts and community support when shortages arise. This strain on the shelter system highlights the need for improved demand forecasting and resource allocation to improve operation efficiency.

### Importance
By enhancing forecasting and resource allocation for homeless support services, we can alleviate the strain on the shelter system and better meet the needs of those experiencing homelessness. This initiative aims to not only improve operational efficiency but also uphold the dignity and rights of all individuals in Toronto by ensuring access to stable housing, healthcare, and support services. Ultimately, by tackling homelessness, we strive to create a more compassionate and inclusive city where everyone has the opportunity to thrive.

### What We Will Predict
Our machine learning model aims to predict the demand for shelter beds and resources within Toronto's homeless support services. By analysing various factors such as weather conditions, social events, demographics, and historical occupancy rates, the model will forecast the expected need for temporary accommodation and support services. This prediction will enable better resource allocation and planning, allowing shelters to proactively address fluctuations in demand and optimise their operations.

## Data
### Data Sets
We currently plan on using the following datasets:
* Daily Shelter Occupancy: https://open.toronto.ca/dataset/daily-shelter-overnight-service-occupancy-capacity/
* Weather: https://climate.weather.gc.ca/historical_data/search_historic_data_e.html
* Available low-cost housing: https://open.canada.ca/data/en/dataset/324befd1-893b-42e6-bece-6d30af3dd9f1
* (un)employment rate: https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1410029501

Additional datasets we could consider include:
* Mental health and mental health service availability: https://data.torontopolice.on.ca/datasets/79c8e950bfe54ce39334ba108e1b325f_0/explore
* Economic Conditions: https://www.statcan.gc.ca/en/start
* GDP: https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=3610043402
* Inflation: https://www.genengnews.com/

### [Data Visualization]([url](https://github.com/Tomasdfgh/RBCs_Borealis_AIs_Shelter_Occupancy_Forecast/blob/main/Data_Visualization.ipynb))
The Data_Visualization.ipynb notebook imports and prepares our datasets. It explains the key data trends through clear visualizations, aiding in quick understanding and analysis.
<img width="1425" alt="Screenshot 2024-04-07 at 10 13 09 PM" src="https://github.com/Tomasdfgh/RBCs_Borealis_AIs_Shelter_Occupancy_Forecast/assets/105636722/355d6aa5-d05d-4664-a6e3-8dfee57ef237">

## Machine Learning Approach
In our project, we have decided to consider a comprehensive machine learning approach by experimenting with a selection of different models. This strategy involves designing and testing multiple models, including Long Short-Term Memory (LSTM) networks, Random Forest Regression, Facebook Prophet, and Seasonal Autoregressive Integrated Moving Average (SARIMA) models. Our objective is to compare their performances rigorously to identify strengths and weaknesses unique to each. By leveraging the comparative analysis, we aim to combine these models strategically, capitalizing on their individual advantages to enhance overall predictive accuracy. This approach allows us to tailor our solution to deliver the best possible performance in optimizing homeless shelter operations.

### LSTM
LSTM (Long Short-Term Memory) networks offer a powerful solution for time series forecasting tasks, particularly in predicting shelter occupancy rates. LSTMs excel in retaining relevant information over extended sequences through their memory cell state, allowing them to recognize seasonalities, trends, and other critical patterns. Furthermore, these networks demonstrate flexibility in handling varying sequence lengths, accommodating the diverse historical data available for each shelter. Robustness to noisy data and the capability to learn hierarchical representations further enhance their utility in forecasting tasks. In essence, LSTM networks present a sophisticated yet adaptable approach to shelter occupancy prediction, leveraging their strengths in temporal modeling to provide accurate and insightful forecasts. The currently implementation is only the output feature, we are looking into implementing a multivariate LSTM.

#### Data Preprocessing

To preprocess the data, we join the 4 years of data together and break them up into individual datasets for different shelters. Furthermore, the training dataset is created through combining the occupancy rates for every shelters together into a singular data point. The model shall be trained on that dataframe. From a trained model, future data can be inferred for individual shelters using their respective occupancy rates.
<p align="center">
  <img src="https://github.com/Tomasdfgh/RBCs_Borealis_AIs_Shelter_Occupancy_Forecast/assets/86145397/1bb30d82-ed31-4084-a914-5fd124fcfe7e)" width="450" alt="chessBoard">
  <br>
  <em>The combined occupancy rates of all shelters from 2021 to March 25th 2024</em>
</p>

#### Model Result
This same dataset is then used to train the model, resulting in this inference on that same dataset:

<p align="center">
  <img src="https://github.com/Tomasdfgh/RBCs_Borealis_AIs_Shelter_Occupancy_Forecast/assets/86145397/732b93c8-f316-4880-a215-c4fb7b9979ab" width="450" alt="chessBoard">
  <br>
  <em>Model prediction on dataset</em>
</p>

Then, the same model is used for inference on 9 random shelters in the city. The first half of the data for that shelter is passed into the model for it to help predict the other half:

<p align="center">
  <img src="https://github.com/Tomasdfgh/RBCs_Borealis_AIs_Shelter_Occupancy_Forecast/assets/86145397/06c4a6a7-b32f-4f3a-8768-2a6d91900c79" width="900" alt="chessBoard">
  <br>
  <em>Model Inference on random shelters</em>
</p>


### RFR

### SARIMA

### Prophet


## UI
![image](https://github.com/Tomasdfgh/RBCs_Borealis_AIs_Shelter_Occupancy_Forecast/assets/86145397/6646b2c4-cc94-46e7-95d2-e737af68d1ed)
