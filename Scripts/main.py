import dataload as dl
import training as tr
import model as md
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from copy import deepcopy as dc

from torch.utils.data import DataLoader

if __name__ == "__main__":

	#Occupancy Rate (Output Data):
	data_23 = r"C:\Users\tomng\Desktop\RBC's Borealis AI Lets Solve It\Datasets\daily-shelter-overnight-service-occupancy-capacity-2023.csv"
	data_22 = r"C:\Users\tomng\Desktop\RBC's Borealis AI Lets Solve It\Datasets\daily-shelter-overnight-service-occupancy-capacity-2022.csv"
	data_21 = r"C:\Users\tomng\Desktop\RBC's Borealis AI Lets Solve It\Datasets\daily-shelter-overnight-service-occupancy-capacity-2021.csv"
	data_24 = r"C:\Users\tomng\Desktop\RBC's Borealis AI Lets Solve It\Datasets\Daily shelter overnight occupancy.csv"
	links = [data_24, data_23, data_22, data_21]

	#Weather Data
	data_w_23 = r"C:\Users\tomng\Desktop\RBC's Borealis AI Lets Solve It\Datasets\en_climate_daily_ON_6158355_2023_P1D.csv"
	data_w_24 = r"C:\Users\tomng\Desktop\RBC's Borealis AI Lets Solve It\Datasets\en_climate_daily_ON_6158355_2024_P1D.csv"
	data_w_22 = r"C:\Users\tomng\Desktop\RBC's Borealis AI Lets Solve It\Datasets\en_climate_daily_ON_6158355_2022_P1D.csv"
	data_w_21 = r"C:\Users\tomng\Desktop\RBC's Borealis AI Lets Solve It\Datasets\en_climate_daily_ON_6158355_2021_P1D.csv"
	links_weather = [data_w_24, data_w_23, data_w_22, data_w_21]

	#Housing
	data_housing = r"C:\Users\tomng\Desktop\RBC's Borealis AI Lets Solve It\Datasets\Housing.csv"

	#Crisis helpline
	data_crisis = r"C:\Users\tomng\Desktop\RBC's Borealis AI Lets Solve It\Datasets\Persons_in_Crisis_Calls_for_Service_Attended_Open_Data.csv"
	
	#Load Data takes in all the datasets and create a general dataframe to be adapted again for training different types of Model
	#--Features include:
	# Date
	# Program_ID 				(Not Trainable Feature)
	# CAPACITY_TYPE 			(Not Trainable Feature)
	# CAPACITY_ACTUAL_BED 		(Part of output feature, raw numbers instead of percentage)
	# OCCUPIED_BEDS				(Part of output feature, raw numbers instead of percentage)
	# CAPACITY_ACTUAL_ROOM		(Part of output feature, raw numbers instead of percentage)
	# OCCUPIED_ROOMS			(Part of output feature, raw numbers instead of percentage)

	#Weather Features
	# Max Temp
	# Min Temp
	# Mean Temp
	# Heat Deg Days
	# Total Precip
	# Snow on Grind

	#Housing Data
	# Value

	#Person in crisis data
	# Overdose
	# Person in Crisis
	# Suicide-related

	#Final Output
	#OCCUPANCY_RATE_BEDS		(Output percentage)
	#OCCUPANCY_RATE_ROOMS		(Output percentage)
	dataframe, iso_data = dl.loadData(links, links_weather, data_housing, data_crisis)

	#Function output explanation:
	#	--DataFrame-- is the general combined data of all datasets, unaltered.
	#	--iso_data-- is the dataframe but broken up into a hashmap where the key is the shelter id and the value is the data for that specific shelter


	df = dl.prep_Data(dataframe)

	df = df[['OCCUPANCY_DATE', 'OCCUPIED_PERCENTAGE']]

	n_steps = 7
	batch_size = 16
	lstm_df = dl.time_series_for_lstm(df, n_steps)

	scaler = MinMaxScaler(feature_range = (-1, 1))
	lstm_df = scaler.fit_transform(lstm_df)
	X = dc(np.flip(lstm_df[:, 1:], axis = 1))
	y = lstm_df[:, 0]

	split_index = int(len(X) * 0.95)

	X_train = X[:split_index]
	X_test = X[split_index:]

	y_train = y[:split_index]
	y_test = y[split_index:]

	X_train = X_train.reshape((-1, n_steps, 1))
	X_test = X_test.reshape((-1, n_steps, 1))

	y_train = y_train.reshape((-1, 1))
	y_test = y_test.reshape((-1, 1))

	X_train = torch.tensor(X_train).float()
	y_train = torch.tensor(y_train).float()
	X_test = torch.tensor(X_test).float()
	y_test = torch.tensor(y_test).float()

	train_dataset = dl.TimeSeriesDataset(X_train, y_train)
	test_dataset = dl.TimeSeriesDataset(X_test, y_test)

	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


	model = md.LSTM(1,4,1)

	learning_rate = 0.001
	num_epochs = 10
	loss_function = nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

	for epoch in range(num_epochs):
	    tr.train_one_epoch(model,epoch, train_loader, loss_function, optimizer)
	    tr.validate_one_epoch(model,epoch, test_loader, loss_function)


	with torch.no_grad():
	    predicted = model(X_train).numpy()

	plt.plot(y_train, label='Actual Close')
	plt.plot(predicted, label='Predicted Close')
	plt.xlabel('Day')
	plt.ylabel('Close')
	plt.legend()
	plt.show()
	