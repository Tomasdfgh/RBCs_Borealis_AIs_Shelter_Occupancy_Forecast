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
