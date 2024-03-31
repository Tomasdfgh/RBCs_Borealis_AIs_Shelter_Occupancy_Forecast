import matplotlib.pyplot as plt
import random
import dataload as dl
import torch
import pandas as pd

def plot_random_shelters(iso_data, model, num_sq, n_steps, scaler):
	#Inferring Data for random shelters
	random_keys = random.sample(list(iso_data), num_sq ** 2)
	fig, axs = plt.subplots(num_sq, num_sq, figsize=(11, 8))

	for i in range(num_sq):
		for j in range(num_sq):

			shelter_index = random_keys[i * num_sq + j]  # Calculate the index from the 1D array
			if iso_data[shelter_index]['OCCUPANCY_RATE_ROOMS'].isna().all():
				iso_data[shelter_index] = iso_data[shelter_index].rename(columns = {'OCCUPANCY_RATE_BEDS': 'OCCUPIED_PERCENTAGE'})
				plot_x, plot_y = dl.time_series_to_model_inputtable(iso_data[shelter_index][['OCCUPANCY_DATE', 'OCCUPIED_PERCENTAGE']], n_steps, scaler)
			else:
				iso_data[shelter_index] = iso_data[shelter_index].rename(columns = {'OCCUPANCY_RATE_ROOMS': 'OCCUPIED_PERCENTAGE'})
				plot_x, plot_y = dl.time_series_to_model_inputtable(iso_data[shelter_index][['OCCUPANCY_DATE', 'OCCUPIED_PERCENTAGE']], n_steps, scaler)

			with torch.no_grad():
				predicted_ = model(plot_x).numpy()

			#Getting Y-axis values for actual and predicted
			predicted_ = dl.transform_back(predicted_, n_steps, scaler)
			plot_y = dl.transform_back(plot_y, n_steps, scaler)

			#Getting X-axis values
			date_frame = iso_data[shelter_index]['OCCUPANCY_DATE']
			min_date = iso_data[shelter_index]['OCCUPANCY_DATE'].min()
			new_date = min_date + pd.Timedelta(days=n_steps)
			date_frame = date_frame[date_frame >= new_date]

			axs[i, j].plot(date_frame, plot_y)
			axs[i, j].plot(date_frame, predicted_)

			#Labeling
			axs[i, j].set_title(f'Shelter {shelter_index}')
			axs[i, j].set_xlabel('Date')
			axs[i, j].set_ylabel('Occupied Percentage (%)')
		
	plt.tight_layout()
	plt.show()


def plot_general(X_train, y_train, n_steps, scaler, model, date_frame):
	#Inferring Data for All Shelters
	with torch.no_grad():
		predicted = model(X_train).numpy()

	#Getting the Y-axis values for actual and predicted
	predicted = dl.transform_back(predicted, n_steps, scaler)
	y_train = dl.transform_back(y_train, n_steps, scaler)

	#Getting the X-axis values
	max_date = date_frame.min() + pd.Timedelta(days=len(y_train))
	date_frame = date_frame[date_frame < max_date]

	plt.plot(date_frame, y_train, label='Actual')
	plt.plot(date_frame, predicted, label='Predicted')
	plt.title('All Shelters Occupancy Rates')
	plt.xlabel('Date')
	plt.ylabel('Occupied Percentage (%)')
	plt.legend()
	plt.show()