import pandas as pd
import numpy as np
from copy import deepcopy as dc
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader

class TimeSeriesDataset(Dataset):
	def __init__(self, X, y):
		self.X = X
		self.y = y

	def __len__(self):
		return len(self.X)

	def __getitem__(self, i):
		return self.X[i], self.y[i]

def load_csv_to_pandas(file_path):
	try:
		# Load CSV file into a pandas data_23Frame
		df = pd.read_csv(file_path, header=0, low_memory=False, encoding='unicode_escape')
		print("Number of rows in the data_23Frame:", file_path, len(df))
		return df
	except FileNotFoundError:
		print(f"File '{file_path}' not found.")
		return None
	except Exception as e:
		print("An error occurred:", str(e))
		return None

#This function converts All dataset from csv files into one dataframe
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
def loadData(output_data, weather_data, housing, crisis):

	#-------Output Data-------#
	#Loading up the links to the output dataset
	for i in range(len(output_data)):
		output_data[i] = load_csv_to_pandas(output_data[i])

	#Creating the list of columns to drop for datasets above 2020
	cols_above_20 = [i for i in range(0, output_data[0].shape[1])]
	cols_above_20.remove(1)
	cols_above_20.remove(12)
	cols_above_20.remove(19)
	cols_above_20.remove(20)
	cols_above_20.remove(22)
	cols_above_20.remove(25)
	cols_above_20.remove(27)
	cols_above_20.remove(30)
	cols_above_20.remove(31)

	#Dropping irrelevant columns for datasets above 2020
	for i in range(len(output_data)):
		output_data[i] = output_data[i].drop(columns = output_data[i].columns[cols_above_20])
		output_data[i]['OCCUPANCY_DATE'] = output_data[i]['OCCUPANCY_DATE'].astype(str)
		output_data[i]['OCCUPANCY_DATE'] =  pd.to_datetime(output_data[i]['OCCUPANCY_DATE'])

	#Joining the Output data together
	big_data = output_data[0]
	for i in range(1,len(output_data)):
		big_data = pd.concat([big_data, output_data[i]], ignore_index = True)

	#Determine the max and min date in the dataset to create a date vector to fill out empty values
	max_date = big_data['OCCUPANCY_DATE'].max()
	min_date = big_data['OCCUPANCY_DATE'].min()
	date_range = pd.date_range(start=min_date, end=max_date, freq = 'D')
	date_df = pd.DataFrame({'OCCUPANCY_DATE': date_range})
	#1, 11, 12 for 2020 and below

	#-------Weather Data-------#

	#loading up the links to the weather dataset
	for i in range(len(weather_data)):
		weather_data[i] = load_csv_to_pandas(weather_data[i])

	#Creating the list of columns to drop for weather data
	weather_cols = [i for i in range(0, weather_data[0].shape[1])]
	weather_cols.remove(4)
	weather_cols.remove(9)
	weather_cols.remove(11)
	weather_cols.remove(13)
	weather_cols.remove(15)
	weather_cols.remove(17)
	weather_cols.remove(23)
	weather_cols.remove(25)

	#Dropping irrelevant columns for weather datasets
	for i in range(len(weather_data)):
		weather_data[i] = weather_data[i].drop(columns = weather_data[i].columns[weather_cols])
		weather_data[i]['Date/Time'] = weather_data[i]['Date/Time'].astype(str)
		weather_data[i]['Date/Time'] = pd.to_datetime(weather_data[i]['Date/Time'])

	#Joining the Weather data together
	big_weather = weather_data[0]
	for i in range(1, len(weather_data)):
		big_weather = pd.concat([big_weather, weather_data[i]], ignore_index = True)

	#Cut down all data with dates that is bigger than the biggest date and smaller than the smallest date with an output
	big_weather = big_weather[big_weather['Date/Time'] <= max_date]
	big_weather = big_weather[big_weather['Date/Time'] >= min_date]

	#Fill out datasets' entries w no data w 0
	big_weather = big_weather.fillna(0)

	#Changing non output dataset's date column to 'OCCUPANCY_DATE'
	big_weather = big_weather.rename(columns = {'Date/Time': 'OCCUPANCY_DATE'})

	#-------Housing Data-------#

	#loading up housing data
	housing = load_csv_to_pandas(housing)

	#Creating the list of columns to drop for housing
	housing_cols = [i for i in range(0, housing.shape[1])]
	housing_cols.remove(0)
	housing_cols.remove(10)

	#Dropping irrelevant columns for housing dataset
	housing = housing[housing['GEO'] == 'Toronto, Ontario']
	housing = housing[housing['New housing price indexes'] == 'Total (house and land)']
	housing = housing.drop(columns = housing.columns[housing_cols])
	housing = housing.rename(columns = {housing.columns[0]: 'OCCUPANCY_DATE'})
	housing["OCCUPANCY_DATE"] = pd.to_datetime(housing["OCCUPANCY_DATE"])
	housing = housing[housing["OCCUPANCY_DATE"] >= min_date]
	housing = housing[housing["OCCUPANCY_DATE"] <= max_date].reset_index(drop=True)
	housing = pd.merge(housing, date_df, on = 'OCCUPANCY_DATE', how = 'outer')
	housing = housing.sort_values(by='OCCUPANCY_DATE').reset_index(drop=True)
	housing = housing.ffill()

	#-------Crisis Data-------#
	
	#Loading the crisis dataset
	crisis = load_csv_to_pandas(crisis)

	#Creating the list of columns to drop for crisis dataset
	crisis_col = [i for i in range(0, crisis.shape[1])]
	crisis_col.remove(2)
	crisis_col.remove(7)

	crisis = crisis.drop(columns = crisis.columns[crisis_col])
	crisis = crisis.rename(columns = {'EVENT_DATE': 'OCCUPANCY_DATE'})
	crisis = crisis.groupby(['OCCUPANCY_DATE', 'EVENT_TYPE']).size().unstack(fill_value=0)
	crisis.reset_index(inplace=True)
	crisis = crisis.rename_axis(None, axis=1)
	crisis['OCCUPANCY_DATE'] = pd.to_datetime(crisis['OCCUPANCY_DATE']).dt.date
	crisis['OCCUPANCY_DATE'] = pd.to_datetime(crisis['OCCUPANCY_DATE'])
	crisis = crisis[crisis["OCCUPANCY_DATE"] >= min_date]
	crisis = crisis[crisis["OCCUPANCY_DATE"] <= max_date]
	crisis = pd.merge(date_df, crisis, on='OCCUPANCY_DATE', how='left')

	#-------Final Data Prep-------#

	#Merge the datasets together through date
	big_data = pd.merge(big_data, big_weather, on = 'OCCUPANCY_DATE', how = 'inner')
	big_data = pd.merge(big_data, housing, on = 'OCCUPANCY_DATE', how = 'inner')
	big_data = pd.merge(big_data, crisis, on = 'OCCUPANCY_DATE', how = 'inner')

	big_data = big_data.sort_values(by='OCCUPANCY_DATE')

	#Placing the bed and room occupancy column last
	room_occupancy = big_data.pop('OCCUPANCY_RATE_ROOMS')
	bed_occupancy = big_data.pop('OCCUPANCY_RATE_BEDS')
	big_data['OCCUPANCY_RATE_BEDS'] = bed_occupancy
	big_data['OCCUPANCY_RATE_ROOMS'] = room_occupancy



	grouped_data = big_data.groupby('PROGRAM_ID')
	shelter_data_frames = {}
	for shelter_id, shelter_group in grouped_data:
		shelter_data_frames[shelter_id] = shelter_group
		shelter_data_frames[shelter_id]['OCCUPANCY_DATE'] = pd.to_datetime(shelter_data_frames[shelter_id]['OCCUPANCY_DATE'])

	big_data.reset_index(inplace=True)
	big_data = big_data.drop(columns = ['index'])

	return big_data, shelter_data_frames

#This function takes the dataframe and combine all the shelter occupancy together. The idea is that
#individual shelter's data needs to be added together into one big cohesive timeline to be trained on
#From there, the model will then infer data on individual shelters.
def prep_Data(df):

	df = df.drop(columns = ['PROGRAM_ID', 'CAPACITY_TYPE', 'OCCUPANCY_RATE_BEDS', 'OCCUPANCY_RATE_ROOMS'])
	grouped_capacity = df.groupby('OCCUPANCY_DATE')[['CAPACITY_ACTUAL_BED', 'CAPACITY_ACTUAL_ROOM']].sum()
	grouped_occupied = df.groupby('OCCUPANCY_DATE')[['OCCUPIED_BEDS', 'OCCUPIED_ROOMS']].sum()

	df = df.merge(grouped_capacity, on='OCCUPANCY_DATE', suffixes=('', '_TOTAL_CAPACITY'))
	df = df.merge(grouped_occupied, on='OCCUPANCY_DATE', suffixes=('', '_TOTAL_OCCUPIED'))
	df = df.drop(columns = ['CAPACITY_ACTUAL_BED', 'CAPACITY_ACTUAL_ROOM', 'OCCUPIED_BEDS', 'OCCUPIED_ROOMS'])
	df = df.drop_duplicates()

	df['TOTAL_OCCUPIED'] = df['OCCUPIED_BEDS_TOTAL_OCCUPIED'] + df['OCCUPIED_ROOMS_TOTAL_OCCUPIED']
	df['TOTAL_CAPACITY'] = df['CAPACITY_ACTUAL_BED_TOTAL_CAPACITY'] + df['CAPACITY_ACTUAL_ROOM_TOTAL_CAPACITY']
	df['OCCUPIED_PERCENTAGE'] = 100 * df['TOTAL_OCCUPIED']/df['TOTAL_CAPACITY']
	df = df.drop(columns = ['CAPACITY_ACTUAL_BED_TOTAL_CAPACITY', 'CAPACITY_ACTUAL_ROOM_TOTAL_CAPACITY', 'OCCUPIED_BEDS_TOTAL_OCCUPIED', 'OCCUPIED_ROOMS_TOTAL_OCCUPIED', 'TOTAL_CAPACITY', 'TOTAL_OCCUPIED'])
	return df

#This function converts takes in a df of just the 
def time_series_for_lstm(df, n_steps, scaler, batch_size, train_test_split):

	#Converting the df into a time series for lstm
	df = dc(df)
	df.set_index('OCCUPANCY_DATE', inplace=True)
	for i in range(1, n_steps+1):
		df[f'OCCUPIED_PERCENTAGE(t-{i})'] = df['OCCUPIED_PERCENTAGE'].shift(i)
	df.dropna(inplace=True)

	lstm_data = df.to_numpy()

	#Converting the data into dataloader
	lstm_data = scaler.fit_transform(lstm_data)
	X = dc(np.flip(lstm_data[:, 1:], axis = 1))
	y = lstm_data[:, 0]

	split_index = int(len(X) * train_test_split)

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

	train_dataset = TimeSeriesDataset(X_train, y_train)
	test_dataset = TimeSeriesDataset(X_test, y_test)

	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

	return train_loader, test_loader, X_train, y_train

#This function is for the UI. Pass a dataframe with date and output data into it, and it will return X where data can be inferred by the model and
#y to be plotted directly as a comparison
def time_series_to_model_inputtable(df, n_steps, scaler):


	df = dc(df)
	df.set_index('OCCUPANCY_DATE', inplace=True)
	for i in range(1, n_steps+1):
		df[f'OCCUPIED_PERCENTAGE(t-{i})'] = df['OCCUPIED_PERCENTAGE'].shift(i)
	df.dropna(inplace=True)


	lstm_data = df.to_numpy()
	lstm_data = scaler.fit_transform(lstm_data)

	#Converting the data into dataloader
	X = dc(np.flip(lstm_data[:, 1:], axis = 1)).reshape((-1, n_steps, 1))
	y = lstm_data[:, 0].reshape((-1, 1))

	X = torch.tensor(X).float()
	y = torch.tensor(y).float()

	return X, y

def transform_back(data, n_steps, scaler):

	dummies = np.zeros((data.shape[0], n_steps + 1))
	dummies[:, 0] = data.flatten()
	dummies = scaler.inverse_transform(dummies)
	data = dc(dummies[:, 0])

	return data