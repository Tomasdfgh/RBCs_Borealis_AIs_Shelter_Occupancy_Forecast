import matplotlib.pyplot as plt
import random
import dataload as dl
import torch
import pandas as pd
import folium

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

			df_infer = dl.feature_check(df_infer)

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

def plot_random_shelters_one_hot(model, iso_data, n_future, num_sq, scaler, used_features, shel_group, test_check = False, future_days = None):

	dc = dl.get_dc()

	random_keys = random.sample(list(shel_group), num_sq ** 2)
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

			df_infer = dl.feature_check(df_infer)

			#If test check, move the data back by n_future days inorder to view model's performance
			if test_check:

				if future_days is None:
					use_date = max(df_use['OCCUPANCY_DATE']) - pd.Timedelta(days = n_future)
					df_infer = df_infer[df_infer['OCCUPANCY_DATE'] <= use_date]

				elif future_days is not None:
					use_date = max(df_use['OCCUPANCY_DATE']) - pd.Timedelta(days = future_days)
					df_infer = df_infer[df_infer['OCCUPANCY_DATE'] <= use_date]

			try:

				df = dl.infer_date_(model, df_infer, scaler, n_future, future_days, [shel_group[shelter_index], max(shel_group.values()) + 1])

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
	copy_df = dl.feature_check(copy_df)

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

def plot_coord(x, y):

	# Creating the plot
	plt.figure(figsize=(8, 6))
	plt.plot(y, x, 'o', color='blue')
	plt.title('Toronto Shelters Location')
	plt.xlabel('Longitude')
	plt.ylabel('Latitude')
	plt.show()


def plot_distortions(x, y):

	# Creating the plot
	plt.figure(figsize=(8, 6))
	plt.plot(y, x, 'o', color='blue')
	plt.title('Distortions')
	plt.xlabel('Number of Centroids (k)')
	plt.ylabel('Distortions')
	plt.show()

def plot_shelters_n_centroids(cx, cy, sx, sy):

	# Creating the plot
	plt.figure(figsize=(8, 6))
	plt.plot(cy, cx, 'o', color='blue')
	plt.plot(sy, sx, '.', color='red')
	plt.title('Distortions')
	plt.xlabel('Longitude')
	plt.ylabel('Latitude')
	plt.show()


def plot_coord_actual(location_x, location_y, filename="Toronto_Shelter_Location.html"):
    # Creating the map centered at the mean of coordinates
    map_center = [sum(location_x)/len(location_x), sum(location_y)/len(location_y)]
    map_osm = folium.Map(location=map_center, zoom_start=12)

    # Adding markers for each location
    for x, y in zip(location_x, location_y):
        folium.Marker(location=[x, y]).add_to(map_osm)

    map_osm.save(filename)