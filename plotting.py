import matplotlib.pyplot as plt
import random
import dataload as dl
import torch
import pandas as pd

#Inferring Data for random shelters
def plot_random_shelters(model, iso_data, n_future, num_sq, scaler, used_features, test_check = False, future_days = None):

	dc = dl.get_dc()

	random_keys = random.sample(list(iso_data), num_sq ** 2)
	fig, axs = plt.subplots(num_sq, num_sq, figsize=(11, 8))
	for i in range(num_sq):
		for j in range(num_sq):

			shelter_index = random_keys[i * num_sq + j]  # Calculate the index from the 1D array

			#Preprocess the iso by copying it into local variables to avoid changing the iso_data itself
			df_use = dc(iso_data[shelter_index])
			max_date = df_use['OCCUPANCY_DATE'].max()

			if df_use['OCCUPANCY_RATE_ROOMS'].isna().all():
				df_use = df_use.rename(columns = {'OCCUPANCY_RATE_BEDS': 'OCCUPIED_PERCENTAGE'})
			else:
				df_use = df_use.rename(columns = {'OCCUPANCY_RATE_ROOMS': 'OCCUPIED_PERCENTAGE'})

			df_infer = df_use[used_features]

			#If test check, move the data back by n_future days inorder to view model's performance
			if test_check:

				if future_days is None:
					use_date = max(df_use['OCCUPANCY_DATE']) - pd.Timedelta(days = n_future)
					df_infer = df_infer[df_infer['OCCUPANCY_DATE'] <= use_date]

				elif future_days is not None:
					use_date = max(df_use['OCCUPANCY_DATE']) - pd.Timedelta(days = future_days)
					df_infer = df_infer[df_infer['OCCUPANCY_DATE'] <= use_date]

			try:
				df = dl.infer_date_(model, df_infer, scaler, n_future, future_days)

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

def plot_general(model, df, n_future, scaler, test_check = False, future_days = None):

	#Deep copying the dataframe
	dc = dl.get_dc()
	copy_df = dc(df)

	#If test check, move the data back by n_future days inorder to view model's performance
	if test_check:

		if future_days is None:
			use_date = max(copy_df['OCCUPANCY_DATE']) - pd.Timedelta(days = n_future)
			copy_df = copy_df[copy_df['OCCUPANCY_DATE'] <= use_date]

		elif future_days is not None:
			use_date = max(copy_df['OCCUPANCY_DATE']) - pd.Timedelta(days = future_days)
			copy_df = copy_df[copy_df['OCCUPANCY_DATE'] <= use_date]

	#Getting the inferred data
	data_frame = dl.infer_date_(model, copy_df, scaler,n_future, future_days)

	plt.plot(df['OCCUPANCY_DATE'], df['OCCUPIED_PERCENTAGE'], label='Actual')
	plt.plot(data_frame['OCCUPANCY_DATE'], data_frame['OCCUPIED_PERCENTAGE'], label='Predicted')
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