import pandas as pd
import numpy as np
from copy import deepcopy as dc
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from geopy.geocoders import Nominatim


#Turn this on to see the numpy array in full instead of partially
print_all = True
if print_all:
    import sys
    np.set_printoptions(threshold=sys.maxsize)

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

def get_scaler():
    scaler = MinMaxScaler(feature_range = (-1, 1))
    return scaler

def get_dc():
    return dc

def prep_Data(df):

    df = df.drop(columns = ['CAPACITY_TYPE', 'OCCUPANCY_RATE_BEDS', 'OCCUPANCY_RATE_ROOMS'])
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

def load_csv_to_pandas(file_path):
    try:
        # Load CSV file into a pandas data_23Frame
        df = pd.read_csv(file_path, header=0, low_memory=False, encoding='unicode_escape')
        print("Number of rows in the dataFrame:", file_path, len(df))
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
    # Program_ID                 (Not Trainable Feature)
    # CAPACITY_TYPE             (Not Trainable Feature)
    # CAPACITY_ACTUAL_BED         (Part of output feature, raw numbers instead of percentage)
    # OCCUPIED_BEDS                (Part of output feature, raw numbers instead of percentage)
    # CAPACITY_ACTUAL_ROOM        (Part of output feature, raw numbers instead of percentage)
    # OCCUPIED_ROOMS            (Part of output feature, raw numbers instead of percentage)

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
    #OCCUPANCY_RATE_BEDS        (Output percentage)
    #OCCUPANCY_RATE_ROOMS        (Output percentage)
def loadData(output_data, weather_data, housing, crisis):

    #-------Output Data-------#
    #Loading up the links to the output dataset
    for i in range(len(output_data)):
        output_data[i] = load_csv_to_pandas(output_data[i])

    #Dropping irrelevant columns for output datasets
    for i in range(len(output_data)):
        output_data[i] = output_data[i].drop(columns = ['_id', 'ORGANIZATION_ID', 'SHELTER_ID', 'LOCATION_ID', 'LOCATION_CITY', 'LOCATION_PROVINCE', 'PROGRAM_NAME', 'SECTOR', 'PROGRAM_MODEL','OVERNIGHT_SERVICE_TYPE', 'PROGRAM_AREA', 'SERVICE_USER_COUNT', 'CAPACITY_FUNDING_BED', 'UNOCCUPIED_BEDS', 'UNAVAILABLE_BEDS', 'CAPACITY_FUNDING_ROOM', 'UNOCCUPIED_ROOMS', 'UNAVAILABLE_ROOMS'])
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

    #-------Weather Data-------#

    #loading up the links to the weather dataset
    for i in range(len(weather_data)):
        weather_data[i] = load_csv_to_pandas(weather_data[i])

    #Dropping irrelevant columns for weather datasets
    for i in range(len(weather_data)):
        weather_data[i] = weather_data[i].drop(columns = ['ï»¿"Longitude (x)"', 'Latitude (y)', 'Station Name', 'Climate ID', 'Year', 'Month', 'Day', 'Data Quality', 'Max Temp Flag', 'Min Temp Flag', 'Mean Temp Flag', 'Heat Deg Days Flag', 'Cool Deg Days Flag', 'Total Rain (mm)', 'Total Rain Flag', 'Total Snow (cm)', 'Total Snow Flag', 'Total Precip Flag',
        'Snow on Grnd Flag', 'Dir of Max Gust (10s deg)', 'Dir of Max Gust Flag', 'Spd of Max Gust (km/h)', 'Spd of Max Gust Flag'])
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

    #Dropping irrelevant columns for housing dataset
    housing = housing[housing['GEO'] == 'Toronto, Ontario']
    housing = housing[housing['New housing price indexes'] == 'Total (house and land)']
    housing = housing.drop(columns = ['GEO', 'DGUID', 'New housing price indexes', 'UOM', 'UOM_ID', 'SCALAR_FACTOR', 'SCALAR_ID', 'VECTOR', 'COORDINATE', 'STATUS', 'SYMBOL', 'TERMINATED', 'DECIMALS'])
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

    #Analyize Data
    crisis = crisis.drop(columns = ['ï»¿OBJECTID', 'EVENT_ID', 'EVENT_YEAR', 'EVENT_MONTH', 'EVENT_DOW', 'EVENT_HOUR', 'DIVISION', 'OCCURRENCE_CREATED', 'APPREHENSION_MADE', 'MCIT_ATTEND', 'HOOD_158', 'NEIGHBOURHOOD_158', 'HOOD_140', 'NEIGHBOURHOOD_140'])
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
def merge_Shelters_Data(df):

    df = df.drop(columns = ['ORGANIZATION_NAME', 'SHELTER_GROUP', 'LOCATION_NAME', 'LOCATION_ADDRESS', 'LOCATION_POSTAL_CODE', 'PROGRAM_ID', 'CAPACITY_TYPE', 'OCCUPANCY_RATE_BEDS', 'OCCUPANCY_RATE_ROOMS'])
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

#Function to check if there are any missing dates in the individual data Return True if missing, and False if not
def check_consistent_dates(df):
    # Convert date column to DateTimeIndex and sort the DataFrame
    df['OCCUPANCY_DATE'] = pd.to_datetime(df['OCCUPANCY_DATE'])
    df = df.sort_values('OCCUPANCY_DATE').reset_index(drop=True)
    
    # Calculate the expected range of dates
    min_date = df['OCCUPANCY_DATE'].min()
    max_date = df['OCCUPANCY_DATE'].max()
    expected_dates = pd.date_range(start=min_date, end=max_date)
    
    # Check for any missing dates
    missing_dates = expected_dates[~expected_dates.isin(df['OCCUPANCY_DATE'])]
    
    if len(missing_dates) == 0:
        return False
    else:
        return True

# def time_series_converter(df, scaler, n_past, n_future, train_test_split, batch_size):


#     df = dc(df)
#     df.set_index('OCCUPANCY_DATE', inplace=True)
#     df = df.astype(float)
#     scaler = scaler.fit(df)

#     df_scaled = scaler.transform(df)

#     num_feat = len([i for i in df])

#     train_x = []
#     train_y = []

#     for i in range(n_past, len(df_scaled) - n_future + 1):
#         train_x.append(df_scaled[i - n_past:i, 0:df_scaled.shape[1]])
#         train_y.append(df_scaled[i: i + n_future, -1])

#     train_x, train_y = np.array(train_x), np.array(train_y)

#     split_index = int(len(train_x) * train_test_split)

#     X_train = train_x[:split_index]
#     X_test = train_x[split_index:]

#     Y_train = train_y[:split_index]
#     Y_test = train_y[split_index:]

#     X_train_ = X_train.reshape((-1, n_past, num_feat))
#     X_test_ = X_test.reshape((-1, n_past, num_feat))

#     X_train = torch.tensor(X_train).float()
#     Y_train = torch.tensor(Y_train).float()
#     X_test = torch.tensor(X_test).float()
#     Y_test = torch.tensor(Y_test).float()

#     train_Dataset = TimeSeriesDataset(X_train, Y_train)
#     test_Dataset = TimeSeriesDataset(X_test, Y_test)

#     train_loader = DataLoader(train_Dataset, batch_size = batch_size, shuffle = True)
#     test_loader = DataLoader(test_Dataset, batch_size = batch_size, shuffle = False)

#     return train_loader, test_loader

def time_series_converter(iso_data, scaler, n_past, n_future, train_test_split, batch_size, used_features, shel_group = None):

    train_x = []
    train_y = []

    if shel_group is not None:

        one_hot_len = max(shel_group.values()) + 1

        num_feat = len(used_features) - 1 + one_hot_len

        dfs = []

        #Iterate through all useable Shelters
        for i in shel_group:

            #Getting the df from Iso Data
            df = iso_data[int(i)]

            #Unifying the df to have the same output column name
            if df['OCCUPANCY_RATE_ROOMS'].isna().all():
                df = df.rename(columns = {'OCCUPANCY_RATE_BEDS': 'OCCUPIED_PERCENTAGE'})
            else:
                df = df.rename(columns = {'OCCUPANCY_RATE_ROOMS': 'OCCUPIED_PERCENTAGE'})

            df = df[used_features]

            df = feature_check(df)

            for z in range(one_hot_len):
                if shel_group[i] == z:
                    df['Feature_' + str(z)] = 1
                else:
                    df['Feature_' + str(z)] = 0

            dfs.append(df)

            #Concatenating all Dfs together
            concatenated_df = pd.concat(dfs, ignore_index=True)

            #Isolate the feature columns
            iso_col = concatenated_df[['Feature_' + str(i) for i in range(one_hot_len)]]

            #Scaled the dfs
            scaler = scaler.fit(concatenated_df[used_features[1:]])
            np_df = scaler.fit_transform(concatenated_df[used_features[1:]])
            df_scaled = pd.DataFrame(np_df, columns=used_features[1:])

            #Combined the final df together
            np_df = pd.concat([iso_col, df_scaled], axis=1).values

    else:
        df = dc(df)
        df.set_index('OCCUPANCY_DATE', inplace=True)
        df = df.astype(float)
        scaler = scaler.fit(df)

        np_df = scaler.transform(df)

        num_feat = len([i for i in df])

    #Converting it into a time series
    for i in range(n_past, len(np_df) - n_future + 1):
        train_x.append(np_df[i - n_past: i, 0:np_df.shape[1]])
        train_y.append(np_df[i: i + n_future, - 1])

    train_x, train_y = np.array(train_x), np.array(train_y)

    split_index = int(len(train_x) * train_test_split)

    X_train = train_x[:split_index]
    X_test = train_x[split_index:]

    Y_train = train_y[:split_index]
    Y_test = train_y[split_index:]

    X_train_ = X_train.reshape((-1, n_past, num_feat))
    X_test_ = X_test.reshape((-1, n_past, num_feat))

    X_train = torch.tensor(X_train).float()
    Y_train = torch.tensor(Y_train).float()
    X_test = torch.tensor(X_test).float()
    Y_test = torch.tensor(Y_test).float()

    train_Dataset = TimeSeriesDataset(X_train, Y_train)
    test_Dataset = TimeSeriesDataset(X_test, Y_test)

    train_loader = DataLoader(train_Dataset, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(test_Dataset, batch_size = batch_size, shuffle = False)

    return train_loader, test_loader

def infer_date_(model, df, scaler, n_future, future_days = None, one_hot_feature = None):

    #one_hot_feature guide
    #0: Index of 1's
    #1: Length of Vector

    #Set model training to False
    model.train(False)
    dc = get_dc()
    copy_df = dc(df)

    #Case 1: if model predicts multiple days at once
    if n_future > 1:

        #Build Dateframe for future days
        max_date = df['OCCUPANCY_DATE'].max()
        date_range = pd.date_range(start=max_date, end=max_date + pd.Timedelta(days=n_future), freq = 'D')
        df_new = pd.DataFrame({'OCCUPANCY_DATE': date_range})

        #Scaling input Dataframe
        copy_df.set_index('OCCUPANCY_DATE', inplace = True)
        df_scaled = scaler.fit_transform(copy_df)

        if one_hot_feature is not None:
            for i in range(one_hot_feature[1]):

                if i != one_hot_feature[0]:
                    arr = np.zeros((df_scaled.shape[0], 1))
                    df_scaled = np.hstack((arr, df_scaled))
                else:
                    arr = np.ones((df_scaled.shape[0], 1))
                    df_scaled = np.hstack((arr, df_scaled))

        #Convert data to tensor and passing it into the model to get predicted data and converting it into a panda dataframe before returning it
        y = model(torch.tensor(df_scaled).unsqueeze(0).float()).detach().numpy().reshape(-1,1)
        y_ = np.repeat(y, copy_df.shape[1], axis = -1)
        y_actual = scaler.inverse_transform(y_)[:,-1]
        y_inserted = np.insert(y_actual, 0, df['OCCUPIED_PERCENTAGE'].iloc[-1])
        df_new['OCCUPIED_PERCENTAGE'] = pd.DataFrame(y_inserted, columns = ['OCCUPIED_PERCENTAGE'])


    #Case 2: if model predicts one day at a time; therefore need loop to predict all future_days days.
    if n_future == 1 and future_days is not None:

        data = torch.tensor(scaler.fit_transform(np.array(copy_df['OCCUPIED_PERCENTAGE']).reshape(-1, 1)).reshape((-1, copy_df.shape[0], 1))).float()
        for i in range(future_days):
            y = model(data).unsqueeze(0)
            data = torch.cat((data, y), dim = 1)
        data = scaler.inverse_transform(data.squeeze().detach().numpy().reshape(-1, 1)).flatten()

        #Adding a data column to the new data
        max_date = copy_df['OCCUPANCY_DATE'].max()
        date_range = pd.date_range(start=max_date , end=max_date + pd.Timedelta(days=future_days), freq = 'D')
        df_new = pd.DataFrame({'OCCUPANCY_DATE': date_range})

        #Getting the newly generated portion of data
        new_data = data[-future_days:]
        new_data = np.insert(new_data, 0, copy_df['OCCUPIED_PERCENTAGE'].iloc[-1])
        new_data_df = pd.DataFrame(new_data, columns = ['OCCUPIED_PERCENTAGE'])

        #Combined
        df_new['OCCUPIED_PERCENTAGE'] = new_data_df

    return df_new

def feature_check(df):
    df_ = dc(df)
    for i in df_.columns:
        if df_[i].isna().any():
            avg = df_[i].mean()
            df_.loc[:, i] = df_[i].fillna(avg)
    return df_

def get_coordinates(address):
    geolocator = Nominatim(user_agent="my_geocoder", timeout=10000)
    location = geolocator.geocode(str(address) + ", Ontario, Canada", country_codes='CA')
    if location:
        return location.latitude, location.longitude
    else:
        return None