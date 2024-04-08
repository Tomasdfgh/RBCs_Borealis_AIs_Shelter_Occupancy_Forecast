import dataload as dl
import training as tr
import model as md
import plotting as pl

import torch
import torch.nn as nn

import numpy as np
if __name__ == "__main__":

	#Do these first, in this order!
	#1)Finish the inferring future data feature.																Done
	#2)Use this new feature and incorporate it into the plot_random_shelters function.							Done
	#  Do it so that so that you can divide the dataset into before and after to infer data.
	#3)Add in a testing set error. May need to rework a lot of the structure of the pipeline. Think about it.	
	#  But this needs to be done for better analysis
	#4)Start working on adding in additional features to the model

	#Work to be done:
	# shelters with less than 7 data points will not work. Figure out a solution.
	# Add plots for Training Errors
	# Test the 100 output options instead of 1 to create a distribution and test which one is more likely

	#Long Term Work:
	# Start Building Software

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
	dataframe, iso_data = dl.loadData(links, links_weather, data_housing, data_crisis)
	#Function output explanation:
	#	--DataFrame-- is the general combined data of all datasets, unaltered.
	#	--iso_data-- is the dataframe but broken up into a hashmap where the key is the shelter id and the value is the data for that specific shelter

	#Initialize the scaler
	scaler = dl.get_scaler()

	print(dataframe)

	#
	df = dl.prep_Data(dataframe)
	df = df[['OCCUPANCY_DATE', 'OCCUPIED_PERCENTAGE']]

	#Hyper parameters
	n_steps = 7
	batch_size = 16
	learning_rate = 0.001
	num_epochs = 70
	train_test_split = 0.75
	loss_function = nn.MSELoss()

	#Model's Hyper Parameters
	input_size = 1 
	hidden_size = 4 
	num_stacked_layers = 1

	#LSTM model
	#Initialize Model and optimizers
	model = md.LSTM(input_size, hidden_size, num_stacked_layers)
	optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


	#Training Model
	train_loader, test_loader, X_train, y_train, X_test, y_test, date_frame = dl.time_series_for_lstm(df, n_steps, scaler, batch_size, train_test_split)
	
	model, training_loss, valid_loss, avg_valid_loss = tr.begin_training(model,num_epochs, train_loader, test_loader, loss_function, optimizer)

	df_use = iso_data[11812]
	if df_use['OCCUPANCY_RATE_ROOMS'].isna().all():
		df_use = df_use.rename(columns = {'OCCUPANCY_RATE_BEDS': 'OCCUPIED_PERCENTAGE'})
	else:
		df_use = df_use.rename(columns = {'OCCUPANCY_RATE_ROOMS': 'OCCUPIED_PERCENTAGE'})

	df_example = torch.tensor(scaler.fit_transform(np.array(df_use['OCCUPIED_PERCENTAGE']).reshape(-1, 1)).reshape((-1, df_use.shape[0], 1))).float()
	traced_model = torch.jit.trace(model, df_example)

	traced_model.save('time-series-lstm.pt')

	# df_final = dl.infer_future_dates(df_use, model, 60, scaler)
	# print(df_final)
	#------Post Training Analysis------#

	#Temporary Measure: Removing any shelters with less than 7 data points
	for i in iso_data.copy():
		if iso_data[i].shape[0] <= n_steps - 1:
			del iso_data[i]

	#Flags to indicate plotting
	plot_general = False
	plot_random = False
	plot_errors = False

	#Inferring Data for All Shelters
	if plot_general:
		pl.plot_general(X_train, y_train, n_steps, scaler, model, date_frame)

	#Plot Random Shelters
	if plot_random:
		num_sq = 3
		per_ = 0.5
		pl.plot_random_shelters(iso_data, model, num_sq, scaler, per_)

	if plot_errors:
		pl.plot_errors(training_loss, valid_loss, avg_valid_loss)