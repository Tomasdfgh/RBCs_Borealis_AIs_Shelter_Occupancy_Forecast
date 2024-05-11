<img width="1437" alt="Screenshot 2024-04-24 at 6 30 50 PM" src="https://github.com/Tomasdfgh/RBCs_Borealis_AIs_Shelter_Occupancy_Forecast/assets/105636722/2f893bee-4825-437b-9fda-5dacf9281ac7">

# LSTM

## Background
LSTM is the main model that we developed for this project. We went through multiple iterations of the LSTM as we developed from a basic model to a more advance one. You may view our progress in the different jypiter notebook files in this folder:
* [1] Univariate LSTM
* [2] Combined Shelter Multivariate
* [3] Location Grouping Multivariate

The last script is the testing script for all the different variations of the model. We are testing loss of the models when inferring the last 60 days of occupancy rates on 5 predetermined shelters where the models are trained on different sets of features. The result reported in script [4] will showcase which sets of features are best for what model. This readme file will go over the different implementation of LSTMs, and their training result. Starting of with how we decided which features are most valuable through feature importance.

## Feature Importance

To identify the most correlated features for our LSTM model, we conducted a comprehensive analysis of all our features utilizing a random forest regression model. This involved assessing the Gini importance of each feature, allowing us to gain valuable insights into their predictive power and influence on our model's performance. By leveraging this approach, we were able to pinpoint the key factors driving the predictive capabilities of our LSTM model, enabling us to make informed decisions regarding feature selection and refinement. This meticulous process not only enhances the accuracy and reliability of our model but also provides a deeper understanding of the underlying data dynamics, empowering us to optimize our predictive outcomes effectively. It is apparent that the 4 most important features are VALUE (housing index value), snow on ground, max and min temperature. As a result, these 4 features will be the chosen features to train our LSTM on.

<p align="center">
  <img src="https://github.com/Tomasdfgh/RBCs_Borealis_AIs_Shelter_Occupancy_Forecast/assets/86145397/d99dd17c-45db-4d15-95e6-1db8b0c3e539" width="550" alt="chessBoard">
  <br>
  <em>Figure 1: Feature Importance of different features</em>
</p>

## Univariate LSTM

The very first iteration is the univariate LSTM. Based on its name, the model is trained on only the ouput feature: the occupancy rates. Due to the fact that the univariate lstm model only takes in one feature, we cannot encode the data of each individual shelter. As a result, we must come up with a way to unify all shelters together into a singular dataset. This can be achieved by finding the city wide average occupancy rates of each shelter and training the model on that data. It is important to realize that in order to train our models on individual shelters' data, we need to encode them by adding additional features for the encoder. That is not possible for a model trained on only the output feature. This model will serve as a baseline for other implementations of LSTMs. This iteration of LSTM is expected to perform the worst.

### Training Result

<p align="center">
  <img src="https://github.com/Tomasdfgh/RBCs_Borealis_AIs_Shelter_Occupancy_Forecast/assets/86145397/d4f3759f-1c18-41a5-8755-9ba9dee618ca" width="950" alt="chessBoard">
  <br>
  <em>Figure 2: Training and Validation Loss for the univariate LSTM</em>
</p>

### City Wide Inference
One method of visually determining the accuracy of the model is to infer the results of the last 60 days of data on the Toronto city wide dataset, which is the same dataset that this model has been trained on. The results are below, and it is pretty obvious visually that the results are terrible. 

<p align="center">
  <img src="https://github.com/Tomasdfgh/RBCs_Borealis_AIs_Shelter_Occupancy_Forecast/assets/86145397/e95fa5e5-4770-40a1-be14-676ec95242ff" width="550" alt="chessBoard">
  <br>
  <em>Figure 3: Univariate LSTM model inference on city wide dataset</em>
</p>

### Random Shelters Inference
This section is the more practical application of the model since the goal of this project is to build a model to predict data for individual shelters. The results can be seen below.

<p align="center">
  <img src="https://github.com/Tomasdfgh/RBCs_Borealis_AIs_Shelter_Occupancy_Forecast/assets/86145397/8b834d34-afb5-4c4f-9ac4-3309e4770fe0" width="950" alt="chessBoard">
  <br>
  <em>Figure 4: Univariate LSTM model Inference on random shelters</em>
</p>

## Combined Shelter Multivariate LSTM

This is the second implementation of the LSTM. This implementation is exactly the same as the univariate lstm; however, now the model can be trained on multiple features instead of just the output feature. The model is still being trained on the Toronto city wide occupancy rates due to the fact that we want to see the results of the model before we begin to encode the data and train our model on individual shelter data (our next implementations, which we will discuss different strategies in how we encode the shelters). The challenge with training a model on multiple features is that to get the model to output the future occupancy rates, we will need the future data of the other features as well, which obviously does not exist. To explain this point further, for the univariate LSTM, we ask the model to predict the future data for the next day; therefore, to get the result of the next 60 days, we will have to ask the model to infer it 60 times. That method will not work for any multivariate implementations of the LSTM. To overcome this, we have asked the model to take in the data and infer 60 days into the future instead of just the next day. Using this method, it is now no longer a requirement to know the data of other features for the days that we are inferring. The features used for this implementation is OCCUPANCY_DATE, Mean Temp (Â°C), Person in Crisis, VALUE (housing index value) , OCCUPIED_PERCENTAGE (output feature).

### Training Result

<p align="center">
  <img src="https://github.com/Tomasdfgh/RBCs_Borealis_AIs_Shelter_Occupancy_Forecast/assets/86145397/346deeb2-9280-4587-b0fd-3f31fe5e02bd" width="950" alt="chessBoard">
  <br>
  <em>Figure 5: Training and Validation Loss for City wide Multivariate LSTM</em>
</p>

### City Wide Inference
Due to the fact that this model is being trained on the combined dataset, we can ask it to infer the last 60 days in the combined dataset again like we did with the Univariate LSTM. Visually, the result is much better than the univariate LSTM. 

<p align="center">
  <img src="https://github.com/Tomasdfgh/RBCs_Borealis_AIs_Shelter_Occupancy_Forecast/assets/86145397/34bd2917-e948-4436-8777-01fa656e485a" width="550" alt="chessBoard">
  <br>
  <em>Figure 6: City wide Multivariate LSTM model inference on city wide dataset</em>
</p>

### Random Shelters Inference
Similarly to the univariate lstm, we will get the model to infer the last 60 days of data on random shelters.

<p align="center">
  <img src="https://github.com/Tomasdfgh/RBCs_Borealis_AIs_Shelter_Occupancy_Forecast/assets/86145397/e69bd3e6-f4ff-4922-b4ae-1aed3e30d406" width="950" alt="chessBoard">
  <br>
  <em>Figure 7: City wide Multivariate LSTM model Inference on random shelters</em>
</p>

## Geo-spatial Multivariate LSTM: Distortions

While the two LSTMs implementations mentioned above has acceptable performance, it does not perform well when inferring occupancy rates for individual shelters. That is expected due to the fact both models are trained on a city wide average dataset instead of the individual shelter's data. The rest of the implementations will now train the model on individual shelter's data. The difficulty here is that to train the model on individual shelter's data, we will have to encode the data. With 62 viable shelters, the data frame will now contain 62 extra features mostly containing 0's. This will undoubtedly cause overfitting. As a result, the first objective is to come up with a strategy to reduce the extra features in the dataframe. To do that, we can group the shelters through a specific grouping schematic. For this implementation, we will group the shelter based on their location through distortions analysis. We will run k-means with 2 to 9 centroids and find their corresponding distortions. We can then pick the right centroids to use using the "elbow rule" where the number of centroids chosen is at the elbow of the distortions curve.

<p align="center">
  <img src="https://github.com/Tomasdfgh/RBCs_Borealis_AIs_Shelter_Occupancy_Forecast/assets/86145397/41af818a-0137-4a71-aa10-1a71e0b93ba9" width="550" alt="chessBoard">
  <br>
  <em>Figure 7: Map of the Shelters in Toronto in a 2d Grid</em>
</p>

The first step to do is to map the shelters on a 2d Grid. Using this grid, we can run k-means algorithm on it. As a reference, here are the same shelters in a Toronto map. From visual inspection, it is obvious that the plot above is simply the shelters from the Toronto map translated into a plot. 

<p align="center">
  <img src="https://github.com/Tomasdfgh/RBCs_Borealis_AIs_Shelter_Occupancy_Forecast/assets/86145397/0f00a9cb-e8c0-401c-bf96-1378ea10f45a" width="550" alt="chessBoard">
  <br>
  <em>Figure 8: Map of individual shelters in Toronto</em>
</p>

### Distortion Analysis

Running k-means 8 times with 2 to 9 centroids on the shelters location grid will yield us the distortion curve below. From this curve, the "elbow" of the curve belongs to 4 centroids. As a result, we will pick that as our grouping schematic to encode our data.

<p align="center">
  <img src="https://github.com/Tomasdfgh/RBCs_Borealis_AIs_Shelter_Occupancy_Forecast/assets/86145397/4cf2a932-675c-4ea8-a1c4-1d85a326dee9" width="550" alt="chessBoard">
  <br>
  <em>Figure 9: Distortion curve with 2 to 9 centroids</em>
</p>

### Grouping Strategy

With 4 centroids selected, the shelters will be divided into 4 different regions. Similar to 4 different neighborhoods, every shelter in any neighbourhood will share the same encoding. As a result, the dataframe will now contain 4 extra dimensions for the encoder instead of 62. The 4 neighbourhoods are displayed below.

<p align="center">
  <img src="https://github.com/Tomasdfgh/RBCs_Borealis_AIs_Shelter_Occupancy_Forecast/assets/86145397/70299e69-dde8-4f20-a51b-884787f8561c" width="550" alt="chessBoard">
  <br>
  <em>Figure 10: Distortion grouping neighbourhoods</em>
</p>

### Training Result

<p align="center">
  <img src="https://github.com/Tomasdfgh/RBCs_Borealis_AIs_Shelter_Occupancy_Forecast/assets/86145397/e5cee899-2bc8-427a-ab0f-0476a8d38644" width="950" alt="chessBoard">
  <br>
  <em>Figure 11: Training and Validation Loss for Distortion Grouping LSTM Model</em>
</p>


### Random Shelter Inference

When asked to infer random shelters, this model performs significantly better.
<p align="center">
  <img src="https://github.com/Tomasdfgh/RBCs_Borealis_AIs_Shelter_Occupancy_Forecast/assets/86145397/8bda6923-d123-41ed-acc5-7dc322f919e8" width="950" alt="chessBoard">
  <br>
  <em>Figure 12: Distortion Grouping LSTM model Inference on random shelters</em>
</p>

## Geo-spatial Multivariate LSTM: Correlation

The last LSTM implementation looks at correlation instead of distortions. This implementation is the same as the distortion one where k-means is run 8 times with 2 to 9 centroids; however, we will now judge which centroids result in the shelters sharing the highest correlation.

### Correlation Analysis

Running k-means 8 times with 2 to 9 centroids on the shelters location grid will yield us the correlation graph below. We have also included the distortions on top of that to show the two together. The centroids with the highest correlation is 8 centroids; therefore, we will pick 8 as the number of neighbourhoods to divide ours shelters into.

<p align="center">
  <img src="https://github.com/Tomasdfgh/RBCs_Borealis_AIs_Shelter_Occupancy_Forecast/assets/86145397/f3671e2c-3829-487a-b08a-148f6733fac5" width="750" alt="chessBoard">
  <br>
  <em>Figure 13: Correlation bar graph with 2 to 9 centroids</em>
</p>

### Grouping Strategy

With 8 centroids selected, the shelters will be divided into 8 different regions. Similar to the implementation above, each shelter in any neighbourhood will share the same encoding. As a result, the dataframe will now contain 8 extra dimensions for the encoder instead of 62. The 8 neighbourhoods are displayed below.

<p align="center">
  <img src="https://github.com/Tomasdfgh/RBCs_Borealis_AIs_Shelter_Occupancy_Forecast/assets/86145397/6637b294-3967-4063-93f4-fbd1063b011a" width="550" alt="chessBoard">
  <br>
  <em>Figure 14: Correlation grouping neighbourhoods</em>
</p>

### Training Result

<p align="center">
  <img src="https://github.com/Tomasdfgh/RBCs_Borealis_AIs_Shelter_Occupancy_Forecast/assets/86145397/e7d80195-7067-4532-be6f-1cce72527007" width="950" alt="chessBoard">
  <br>
  <em>Figure 15: Training and Validation Loss for Correlation Grouping LSTM Model</em>
</p>

### Random Shelter Inference

Similar to the distortion grouping lstm, correlation grouping model performs incredibly well when inferencing individual shelters.

<p align="center">
  <img src="https://github.com/Tomasdfgh/RBCs_Borealis_AIs_Shelter_Occupancy_Forecast/assets/86145397/2bc50a79-d864-4a29-b924-e69c0d7a5754" width="950" alt="chessBoard">
  <br>
  <em>Figure 16: Correlation Grouping LSTM model Inference on random shelters</em>
</p>

## Best Feature for Each Implementation

From the 4 features that we have selected by running our feature importance analysis, we can now begin to test the models and determine which set of features performs the best for which model implementation. The testing and result can be found in the Model Implementation Testing script of this folder.

### Best Set of Features For City Wide Data Multivariate LSTM
The best features for the city wide data multivariate lstm are 'OCCUPANCY_DATE', 'VALUE', 'Min Temp (Â°C)', 'OCCUPIED_PERCENTAGE'. 

<p align="center">
  <img src="https://github.com/Tomasdfgh/RBCs_Borealis_AIs_Shelter_Occupancy_Forecast/assets/86145397/70a29a34-f0de-48f5-9fde-d2cc3a485019" width="550" alt="chessBoard">
  <br>
  <em>Figure 17: Losses for all models using best feature set for city wide data multivariate lstm</em>
</p>

### Best Set of Features For Geolocation Grouping LSTM Using Distortions Analysis
The best features for the Geolocation Grouping LSTM using distortions analysis are 'OCCUPANCY_DATE', 'VALUE', 'Max Temp (Â°C)', 'Min Temp (Â°C)', 'OCCUPIED_PERCENTAGE'.

<p align="center">
  <img src="https://github.com/Tomasdfgh/RBCs_Borealis_AIs_Shelter_Occupancy_Forecast/assets/86145397/0a057978-a1c3-4883-ae36-b951b1fc2b04" width="550" alt="chessBoard">
  <br>
  <em>Figure 18: Losses for all models using best feature set for Geolocation Grouping LSTM Using Distortions Analysis</em>
</p>

### Best Set of Features For Geolocation Grouping LSTM Using Correlation Analysis
The best features for the Geolocation Grouping LSTM Using Correlation Analysis are 'OCCUPANCY_DATE', 'VALUE', 'Snow on Grnd (cm)', 'Max Temp (Â°C)', 'Min Temp (Â°C)', 'OCCUPIED_PERCENTAGE'.


<p align="center">
  <img src="https://github.com/Tomasdfgh/RBCs_Borealis_AIs_Shelter_Occupancy_Forecast/assets/86145397/d01e6cd7-641f-43f2-9e45-be552b5b46be" width="550" alt="chessBoard">
  <br>
  <em>Figure 19: Losses for all models using best feature set for Geolocation Grouping LSTM Using Correlation Analysis</em>
</p>
