import dataload as dl

import matplotlib.pyplot as plt

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
	dataframe, iso_data = dl.loadData(links, links_weather, data_housing, data_crisis)

	#Function output explanation:
	#	--DataFrame-- is the general combined data of all datasets, unaltered.
	#	--iso_data-- is the dataframe but broken up into a hashmap where the key is the shelter id and the value is the data for that specific shelter


	df = dl.prep_Data(dataframe)

	plt.plot(df['OCCUPANCY_DATE'], df['OCCUPIED_PERCENTAGE'])
	plt.show()