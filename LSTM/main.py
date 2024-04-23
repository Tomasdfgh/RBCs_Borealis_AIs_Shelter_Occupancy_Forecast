#Local Imports
import dataload as dl
import training as tr
import model as md
import plotting as pl
import k_means_project as km

#Library Imports
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
	multi_lstm = False
	location_grouping = False

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

	print(dataframe)

	df = dl.prep_Data(dataframe)

	print(df)

	filtered_df = dataframe[dataframe['PROGRAM_ID'] == 16191]

	print(filtered_df)
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
		num_epochs = 1
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

		#Select Features
		#used_features = ['OCCUPANCY_DATE','Mean Temp (Â°C)', 'Person in Crisis', 'VALUE' ,'OCCUPIED_PERCENTAGE']
		used_features = ['OCCUPANCY_DATE', 'Max Temp (Â°C)', 'OCCUPIED_PERCENTAGE']
		df = df[used_features]

		#Pass the dataframe into a fill function that fills up all missing data
		df = dl.feature_check(df)

		#Hyper Parameters
		n_future = 60
		n_past = 90
		train_test_split = 0.8
		batch_size = 16
		learning_rate = 1e-3
		num_epochs = 1
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

		#Temporary Measure: Removing incompatible shelters
		for i in iso_data.copy():
			if iso_data[i].shape[0] <= n_past - 1:
				del iso_data[i]

		#Flags to indicate plotting
		plot_general = True
		plot_random = False
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


	#-------------------------------------------Location Grouping Feature--------------------------------------------#


	if location_grouping:

		#Flags
		run_grouping = False
		k_means = True

		if run_grouping:

			#Locate Shelters Location

			location_x = []
			location_y = []

			hash_ = {}

			for i in iso_data:
				
				if len(iso_data[i]) >= 150:

					coordinates = dl.get_coordinates(iso_data[i]['LOCATION_ADDRESS'].iloc[0])
					#print(coordinates)
					try:
						if coordinates[0] > 43.4 and coordinates[0] < 44 and coordinates[1] < -79.4 and coordinates[1] > -80.2:

							location_x.append(coordinates[0])
							location_y.append(coordinates[1])
							hash_[i] = [coordinates[0], coordinates[1]]
					except:
						pass

			print(hash_)
			pl.plot_coord(location_x, location_y)

		#Running K-means Analysis on locations

		if k_means:

			#This dictionary is obtained from locating shelters Location from run_grouping
			hash_ = {11794: [43.6814522, -79.4181972], 11798: [43.6518512, -79.4036296], 11799: [43.6479162, -79.4114678], 11815: [43.742472, -79.4965543], 11831: [43.641817, -79.401972], 11871: [43.6818715, -79.4186979], 11891: [43.65866645, -79.40075323510317], 11895: [43.658445145098035, -79.40873056862745], 11911: [43.7733887, -79.4150282], 11971: [43.6919186, -79.4398677], 12011: [43.6658599, -79.44591857692308], 12053: [43.665538, -79.4631271], 12231: [43.6179513, -79.4973594], 12251: [43.65792664736842, -79.4071056368421], 12252: [43.68310682710351, -79.76882185782915], 12254: [43.7156276, -79.4674122], 12274: [43.87560463274981, -79.40554412497633], 12291: [43.7364663, -79.5802022], 12292: [43.7364663, -79.5802022], 12471: [43.6518512, -79.4036296], 12711: [43.641817, -79.401972], 13451: [43.667900849999995, -79.4055445043087], 13932: [43.87560463274981, -79.40554412497633], 14051: [43.7721425, -79.5382173], 14251: [43.6508952, -79.40135909072545], 14571: [43.7392967, -79.566092], 14572: [43.7392967, -79.566092], 14631: [43.650085250000004, -79.40251746774611], 14651: [43.67530011065648, -79.40151870869647], 14671: [43.67530011065648, -79.40151870869647], 15111: [43.7842498, -79.4168959], 15112: [43.7842498, -79.4168959], 15171: [43.6683775, -79.4830903], 15711: [43.63901885, -79.4465142944708], 15811: [43.75630675, -79.52718552084744], 15871: [43.75630675, -79.52718552084744], 16111: [43.7163873, -79.5928746], 16131: [43.6325184, -79.4200876], 16151: [43.6325184, -79.4200876], 16191: [43.718908049999996, -79.51456906328633], 16192: [43.718908049999996, -79.51456906328633], 16193: [43.718908049999996, -79.51456906328633], 16194: [43.7842498, -79.4168959], 16271: [43.718908049999996, -79.51456906328633], 16311: [43.718908049999996, -79.51456906328633], 16371: [43.6614308, -79.4288255], 16671: [43.718908049999996, -79.51456906328633], 16691: [43.718908049999996, -79.51456906328633], 16891: [43.6921591, -79.57658485154437], 16892: [43.6921591, -79.57658485154437], 16911: [43.7842498, -79.4168959], 17011: [43.718908049999996, -79.51456906328633], 17012: [43.718908049999996, -79.51456906328633], 17191: [43.7842498, -79.4168959], 17211: [43.7576821, -79.5286579], 17212: [43.7576821, -79.5286579], 17691: [43.869914157032476, -79.40087182664998], 17771: [43.7842498, -79.4168959], 17772: [43.7842498, -79.4168959], 17791: [43.7842498, -79.4168959], 17811: [43.7842498, -79.4168959]}

			data = []
			for i in hash_:
				data.append(hash_[i])

			x_axis = []
			y_axis = []

			for i in range(2,8):

				cens_coord,cens_index = km.k_means(data,i)
				x_axis.append(i)
				dist_s = 0
				distortions = 0
				for z in range(0,i):
					dist_s = 0
					q = 0
					while q < len(data):
						if cens_index[q] == z:
							for s in range(len(data[0])):
								dist_s += (data[q][s] - cens_coord[z][s])**2
							distortions += dist_s
						q += 1
				distortions = distortions/len(data)
				y_axis.append(distortions)

			plot_distortions = False
			if plot_distortions:
				pl.plot_distortions(y_axis, x_axis)

			#Getting the final Locations and Centroids Graph
			cens_coord,cens_index = km.k_means(data,4)

			cens_x = []
			cens_y = []
			for i in cens_coord:
				cens_x.append(i[0])
				cens_y.append(i[1])

			shel_x = []
			shel_y = []
			for i in hash_:
				shel_x.append(hash_[i][0])
				shel_y.append(hash_[i][1])

			plot_sheltes_w_centroids = False
			if plot_sheltes_w_centroids:
				pl.plot_shelters_n_centroids(cens_x, cens_y, shel_x, shel_y)

			#which means the grouping for each shelter is:
			shel_group = {}
			for i,n in enumerate(hash_):
				shel_group[n] = cens_index[i]

			#Creating and Training Model

			#Deepcopy the dataframe for any changes
			dc = dl.get_dc()
			df = dc(dataframe)

			#Initialize the scaler
			scaler = dl.get_scaler()

			#Combined All Shelters together into same days
			df = dl.merge_Shelters_Data(df)

			used_features = ['OCCUPANCY_DATE', 'Max Temp (Â°C)', 'OCCUPIED_PERCENTAGE']

			#Hyper Parameters
			n_future = 60
			n_past = 90
			train_test_split = 0.8
			batch_size = 16
			learning_rate = 1e-3
			num_epochs = 20
			loss_function = nn.MSELoss()
			scaler = dl.get_scaler()

			#Model's Hyperparameters
			input_size = len(used_features) + max(shel_group.values())
			hidden_size = 120
			num_stacked_layers = 1
			output_size = n_future

			#Initialize Model and optimizers
			model = md.LSTM(input_size, hidden_size, num_stacked_layers, output_size)
			optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

			#Getting Dataloader
			train_loader, test_loader = dl.time_series_one_hot_converter(iso_data, shel_group.copy(), scaler, n_past, n_future, train_test_split, batch_size, used_features)

			#Training Model
			model, training_loss, valid_loss, avg_valid_loss = tr.begin_training(model, num_epochs, train_loader, test_loader, loss_function, optimizer)


			#Flags to indicate plotting
			plot_random = True
			plot_errors = True
			plot_general = False

			if plot_random:
				test_check = True
				num_sq = 3
				pl.plot_random_shelters_one_hot(model, iso_data, n_future, num_sq, scaler, used_features,shel_group.copy(), test_check)

			if plot_errors:
				pl.plot_errors(training_loss, valid_loss, avg_valid_loss)