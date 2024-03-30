import dataload as dl
import training as tr
import model as md

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
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
	dataframe, iso_data = dl.loadData(links, links_weather, data_housing, data_crisis)
	#Function output explanation:
	#	--DataFrame-- is the general combined data of all datasets, unaltered.
	#	--iso_data-- is the dataframe but broken up into a hashmap where the key is the shelter id and the value is the data for that specific shelter


	#Initialize the scaler
	scaler = MinMaxScaler(feature_range = (-1, 1))
	df = dl.prep_Data(dataframe)
	df = df[['OCCUPANCY_DATE', 'OCCUPIED_PERCENTAGE']]

	#Hyper parameters
	n_steps = 7
	batch_size = 16
	learning_rate = 0.001
	num_epochs = 10
	loss_function = nn.MSELoss()

	train_loader, test_loader, X_train, y_train = dl.time_series_for_lstm(df, n_steps, scaler, batch_size)
	model = md.LSTM(1,4,1) 
	optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
	model = tr.begin_training(model,num_epochs, train_loader, test_loader, loss_function, optimizer)

	#print(iso_data[17671])
	plot_x, plot_y = dl.time_series_to_model_inputtable(iso_data[16691][['OCCUPANCY_DATE', 'OCCUPANCY_RATE_ROOMS']], n_steps)

	with torch.no_grad():
	    predicted = model(plot_x).numpy()

	# plt.plot(plot_y)
	# plt.plot(predicted)
	# plt.show()

	with torch.no_grad():
	    predicted = model(X_train).numpy()

	plt.plot(y_train, label='Actual')
	plt.plot(predicted, label='Predicted')
	plt.xlabel('Day')
	plt.ylabel('Close')
	plt.legend()
	plt.show()