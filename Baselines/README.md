# Baseline Models

In this section, we outline the baseline models we used to evaluate the performance of our proposed model. These baselines were carefully selected to ensure a robust evaluation of our proposed methodology. We will describe the process behind each baseline model, detail their implementation, and outline the results. We will also show a comparative analysis of the performance of all the baseline models. This comprehensive benchmarking provides a clear context for assessing the strengths and potential improvements of our proposed solution.

## Autoregressive Model
Autoregressive models predict future behavior based on past behavior by assuming that past data points are useful for predicting future values. These models are particularly useful in time series forecasting, where the next value in the series is predicted as a linear function of the previous values. AR models are often favored for their simplicity and effectiveness in cases where data shows significant continuity and stability over time.

### Results
* **RMSE:**
* **MAE:**

## Random Forest Regression Model
Random Forest Regression utilizes an ensemble learning method for regression tasks. It operates by constructing a multitude of decision trees at training time and outputting the mean prediction of the individual trees. RFR is highly effective for complex regression tasks because it can model nonlinear relationships and interactions between features without requiring extensive data preprocessing or linear assumptions.

### Results
* **RMSE:** 7.04
* **MAE:** 2.59

## Prophet Model
The Facebook Prophet model is a robust, open-source tool designed for forecasting time series data that exhibits patterns on different time scales such as yearly, weekly, and daily, including those with missing data and outliers. It is particularly user-friendly, requiring minimal tuning to produce high-quality forecasts. Prophet is adaptable to various seasonalities automatically and includes components to handle holidays and special events, making it very useful for applications like forecasting economic time series or web traffic. The results listed below are for a 5-day horizon forecast.

### Results
* **RMSE:** 13.66
* **MAE:** 13.44
