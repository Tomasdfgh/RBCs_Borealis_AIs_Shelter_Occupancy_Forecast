import matplotlib.pyplot as plt
import random
import dataload as dl
import torch
import pandas as pd

#Inferring Data for random shelters
def plot_random_shelters(iso_data, model, num_sq, scaler, per_):

	model.train(False)
	random_keys = random.sample(list(iso_data), num_sq ** 2)
	fig, axs = plt.subplots(num_sq, num_sq, figsize=(11, 8))
	for i in range(num_sq):
		for j in range(num_sq):

			shelter_index = random_keys[i * num_sq + j]  # Calculate the index from the 1D array

			#Preprocess the iso by copying it into local variables to avoid changing the iso_data itself
			df_use = iso_data[shelter_index]
			max_date = df_use['OCCUPANCY_DATE'].max()

			if df_use['OCCUPANCY_RATE_ROOMS'].isna().all():
				df_use = df_use.rename(columns = {'OCCUPANCY_RATE_BEDS': 'OCCUPIED_PERCENTAGE'})
			else:
				df_use = df_use.rename(columns = {'OCCUPANCY_RATE_ROOMS': 'OCCUPIED_PERCENTAGE'})

			#Isolate the data to non and inferring section
			df_temp = df_use.head(int(len(iso_data[shelter_index]) * per_))
			cut_date = df_temp['OCCUPANCY_DATE'].max()
			df_before = df_use[df_use['OCCUPANCY_DATE'] < cut_date]

			#Find the difference in date to get how many days to infer
			date_diff = (max_date - cut_date).days + 1

			try:
				df = dl.infer_future_dates(df_before, model, date_diff, scaler)

				axs[i, j].plot(df_use['OCCUPANCY_DATE'], df_use['OCCUPIED_PERCENTAGE'])
				axs[i, j].plot(df['OCCUPANCY_DATE'], df['OCCUPIED_PERCENTAGE'])

				#Labeling
				axs[i, j].set_title(f'Shelter {shelter_index}')
				axs[i, j].set_xlabel('Date')
				axs[i, j].set_ylabel('Occupied Percentage (%)')

			except Exception as e:
				print("Error: " + str(e))

	plt.tight_layout()
	plt.show()

#Inferring Data for All Shelters
def plot_general(X_train, y_train, n_steps, scaler, model, date_frame):
	model.train(False)
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


def plot_errors(training_loss, valid_loss, avg_valid_loss):
	epochs = [i for i in range(1, len(training_loss) + 1)]
	fig, axs = plt.subplots(1, 3, figsize=(12, 3))
	axs[0].plot(epochs, training_loss)
	axs[1].plot(epochs, valid_loss)
	axs[2].plot(epochs, avg_valid_loss)

	axs[0].set_title('Training Loss')
	axs[0].set_xlabel('Epoch')
	axs[0].set_ylabel('Error')

	axs[1].set_title('Validation Loss')
	axs[1].set_xlabel('Epoch')
	axs[1].set_ylabel('Error')

	axs[2].set_title('Average Validation Loss Per Batch')
	axs[2].set_xlabel('Epoch')
	axs[2].set_ylabel('Error')
	plt.tight_layout()
	plt.show()


#This function is just to test which shelter is failing current implementation. No practical use
def plot_random_shelters_test(iso_data, model, n_steps, scaler):
	model.train(False)
	for shelter_index in iso_data:
		try:
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

			plt.plot(date_frame, plot_y)
			plt.plot(date_frame, predicted_)

		except Exception as e:
			print(shelter_index)

		plt.cla()