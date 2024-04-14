import dataload as dl
import training as tr
import model as md
import plotting as pl
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

if __name__ == "__main__":

	#Flags to see which model types to train and show
	uni_lstm = False
	multi_lstm = True

	#---------------------------------------------------PREP DATA---------------------------------------------------#

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

	#------------------------------------------------Univariate LSTM-------------------------------------------------#

	if uni_lstm:

		dc = dl.get_dc()
		df = dc(dataframe)

		#Initialize the scaler
		scaler = dl.get_scaler()

		df = dl.prep_Data(df)
		df = df[['OCCUPANCY_DATE', 'OCCUPIED_PERCENTAGE']]

		#Hyper parameters
		n_steps = 7
		batch_size = 16
		learning_rate = 1e-3
		num_epochs = 10
		train_test_split = 0.75
		loss_function = nn.MSELoss()


		#Model's Hyper Parameters
		input_size = 1 
		hidden_size = 120
		num_stacked_layers = 1
		output_size = 1

		#Univariate LSTM model
		#Initialize Model and optimizers
		model = md.LSTM(input_size, hidden_size, num_stacked_layers, output_size)
		optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

		#Training Model

		train_loader, test_loader, X_train, y_train, X_test, y_test, date_frame = dl.time_series_for_lstm(df, n_steps, scaler, batch_size, train_test_split)

		model, training_loss, valid_loss, avg_valid_loss = tr.begin_training(model,num_epochs, train_loader, test_loader, loss_function, optimizer)

		#------Post Training Analysis------#

		#Temporary Measure: Removing any shelters with less than 7 data points
		for i in iso_data.copy():
			if iso_data[i].shape[0] <= n_steps - 1:
				del iso_data[i]

		#Flags to indicate plotting
		plot_general = True
		plot_random = True
		plot_errors = True

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

	#-----------------------------------------------Multivariate LSTM------------------------------------------------#

	if multi_lstm:

		dc = dl.get_dc()
		df = dc(dataframe)

		#Initialize the scaler
		scaler = dl.get_scaler()

		df = dl.prep_Data(dataframe)
		df = df[['OCCUPANCY_DATE','Mean Temp (Â°C)' ,'OCCUPIED_PERCENTAGE']]

		#Hyper Parameters
		n_future = 60
		n_past = 90
		train_test_split = 0.8
		batch_size = 16
		learning_rate = 1e-3
		num_epochs = 100
		loss_function = nn.MSELoss()

		#Model's Hyperparameters
		input_size = 2
		hidden_size = 120
		num_stacked_layers = 1
		output_size = n_future

		#Multivariate LSTM model
		#Initialize Model and optimizers
		model = md.LSTM(input_size, hidden_size, num_stacked_layers, output_size)
		optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

		#Getting Dataloader
		train_loader, test_loader = dl.time_series_multivariate(df, scaler, n_past, n_future, train_test_split, batch_size)

		#Training Model
		model, training_loss, valid_loss, avg_valid_loss = tr.begin_training(model, num_epochs, train_loader, test_loader, loss_function, optimizer)

		#------Post Training Analysis------#

		#Flags to indicate plotting
		plot_general = True
		plot_random = True
		plot_errors = True

		if plot_general:
			test_check = True
			pl.plot_general_2(model, df, n_future, scaler, test_check)

		if plot_random:
			test_check = True
			num_sq = 3
			pl.plot_random_shelters_2(model, iso_data, n_future, num_sq, scaler, test_check)

		if plot_errors:
			pl.plot_errors(training_loss, valid_loss, avg_valid_loss)