import dataload as dl
import training as tr
import model as md
import plotting as pl
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

#Immediate work:
# Clean up loadData function so it is a lot cleaner																				DONE
# Build data clean up function so that features can be swapped in and out with ease 											DONE
# A lot of the code base has been changed. Test CityCompassUI.py script again and make sure has no problems
#	- Test all buttons make sure they all work with no errors. If errors, figure out why.
#	- Integrate new model. Looks much cleaner.
#	- Eliminate any functions that is in dataload.py that has been replicated to CityCompassUI.py. 
#	  dataload.py functions has been built for generality; therefore, should work for specific cases of CityCompassUI.py

#Long Term work:

#This is the testing Idea. All long term work should build towards this:

#Have a starting base case example with the Univariate time series lstm. Calculate the MAE loss for all the individual shelters after inference

#For Multivariate, break into different categories:

#	-Model trained on combined dataset of all shelters
#	-Model trained on Different types of grouping
#		- Location Based Grouping
#		- Correlation Based Grouping

#For each type of grouping categories. Have a bar group showing the average MAE for test of each shelter after inference of individual shelters
#Ofcourse for each of those tests, also have the baseline as a comparison

#Continue with that approach but now with different features. Show how each subset of feature selection impacts the testing result of each type of grouping

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
	dataframe, iso_data = dl.loadData(links.copy(), links_weather.copy(), data_housing, data_crisis)
	#Function output explanation:
	#	--DataFrame-- is the general combined data of all datasets, unaltered.
	#	--iso_data-- is the dataframe but broken up into a hashmap where the key is the shelter id and the value is the data for that specific shelter

	#---------------------------------------Combined Shelters Univariate LSTM----------------------------------------#

	if uni_lstm:

		#Deepcopy the dataframe for any changes
		dc = dl.get_dc()
		df = dc(dataframe)

		#Initialize the scaler
		scaler = dl.get_scaler()

		#Combined All Shelters together into same days
		df = dl.merge_Shelters_Data(df)

		#Select features (Univariate)
		used_features = ['OCCUPANCY_DATE', 'OCCUPIED_PERCENTAGE']
		df = df[used_features]

		#Hyper parameters
		n_steps = 7
		n_future = 1
		batch_size = 16
		learning_rate = 1e-3
		num_epochs = 5
		train_test_split = 0.75
		loss_function = nn.MSELoss()

		#Model's Hyper Parameters
		input_size = 1 
		hidden_size = 120
		num_stacked_layers = 1
		output_size = 1

		#Initialize Model and optimizers
		model = md.LSTM(input_size, hidden_size, num_stacked_layers, output_size)
		optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

		#Getting Dataloader
		train_loader, test_loader = dl.time_series_converter(df, scaler, n_steps, n_future, train_test_split, batch_size)

		#Begin Training
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
			#Test_check flag is to move the data back inorder to predict data that already exists so you can view accuracy
			test_check = True
			pl.plot_general(model, df, n_future, scaler, test_check, future_days = 60)

		#Plot Random Shelters
		if plot_random:
			test_check = True
			num_sq = 3
			pl.plot_random_shelters(model, iso_data, n_future, num_sq, scaler, used_features, test_check, future_days = 60)

		if plot_errors:
			pl.plot_errors(training_loss, valid_loss, avg_valid_loss)

	#--------------------------------------Combined Shelters Multivariate LSTM---------------------------------------#

	if multi_lstm:

		#Deepcopy the dataframe for any changes
		dc = dl.get_dc()
		df = dc(dataframe)

		#Initialize the scaler
		scaler = dl.get_scaler()

		#Combined All Shelters together into same days
		df = dl.merge_Shelters_Data(dataframe)

		for i in df:
			print(i)

		#Select Features
		#used_features = ['OCCUPANCY_DATE','Mean Temp (Â°C)', 'Person in Crisis', 'VALUE' ,'OCCUPIED_PERCENTAGE']
		used_features = ['OCCUPANCY_DATE', 'Max Temp (Â°C)' , 'Min Temp (Â°C)', 'Mean Temp (Â°C)', 'Heat Deg Days (Â°C)', 'Cool Deg Days (Â°C)', 'Total Precip (mm)', 'Snow on Grnd (cm)', 'VALUE', 'Overdose', 'Person in Crisis', 'Suicide-related', 'OCCUPIED_PERCENTAGE']
		df = df[used_features]

		#Pass the dataframe into a fill function that fills up all missing data
		df = dl.feature_check(df)

		#Hyper Parameters
		n_future = 60
		n_past = 90
		train_test_split = 0.8
		batch_size = 16
		learning_rate = 1e-3
		num_epochs = 50
		loss_function = nn.MSELoss()

		#Model's Hyperparameters
		input_size = len(used_features) - 1
		hidden_size = 120
		num_stacked_layers = 1
		output_size = n_future

		#Initialize Model and optimizers
		model = md.LSTM(input_size, hidden_size, num_stacked_layers, output_size)
		optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

		#Getting Dataloader
		train_loader, test_loader = dl.time_series_converter(df, scaler, n_past, n_future, train_test_split, batch_size)

		#Training Model
		model, training_loss, valid_loss, avg_valid_loss = tr.begin_training(model, num_epochs, train_loader, test_loader, loss_function, optimizer)

		#------Post Training Analysis------#

		#Temporary Measure: Removing any shelters with less than 7 data points
		for i in iso_data.copy():
			if iso_data[i].shape[0] <= n_past - 1:
				del iso_data[i]

		#Flags to indicate plotting
		plot_general = True
		plot_random = True
		plot_errors = False

		if plot_general:
			#Test_check flag is to move the data back inorder to predict data that already exists so you can view accuracy
			test_check = True
			pl.plot_general(model, df, n_future, scaler, test_check)

		if plot_random:
			test_check = True
			num_sq = 3
			pl.plot_random_shelters(model, iso_data, n_future, num_sq, scaler, used_features, test_check)

		if plot_errors:
			pl.plot_errors(training_loss, valid_loss, avg_valid_loss)