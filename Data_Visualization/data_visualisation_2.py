# -*- coding: utf-8 -*-
"""Data Visualisation 2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1GspAcj5cm-daby8TmN11dPgv5PZqyw9u

# Imports
"""

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters
import seaborn as sns
import numpy as np

"""# Loading the Data

## 1. Shelter Occupancy Data
"""

file_paths = ['/content/drive/MyDrive/Colab Notebooks/Borealis Project/daily-shelter-overnight-service-occupancy-capacity-2021.csv', '/content/drive/MyDrive/Colab Notebooks/Borealis Project/daily-shelter-overnight-service-occupancy-capacity-2022.csv', '/content/drive/MyDrive/Colab Notebooks/Borealis Project/daily-shelter-overnight-service-occupancy-capacity-2023.csv']

dataframes = [pd.read_csv(file) for file in file_paths]

for file in file_paths:
    try:
        df = pd.read_csv(file)
        if df.empty:
            print(f"The file at {file} is empty.")
        else:
            print(f"Loaded {len(df)} rows from {file}")
            dataframes.append(df)
    except FileNotFoundError:
        print(f"No file found at {file}")
    except pd.errors.EmptyDataError:
        print(f"File at {file} is empty or corrupted.")

if dataframes:
    combined_df = pd.concat(dataframes, ignore_index=True)
    print(f"The combined dataframe has {len(combined_df)} rows.")
else:
    print("No data has been loaded. Please check the file paths and contents.")

# COLUMN SELECTION -------------------------------------------------------------

combined_df = combined_df.dropna(subset=['CAPACITY_TYPE'])

# Separate data based on Capacity Type --> Bed Based and Room Based
bed_capacity_df = combined_df[combined_df['CAPACITY_TYPE'] == 'Bed Based Capacity'].copy()
room_capacity_df = combined_df[combined_df['CAPACITY_TYPE'] == 'Room Based Capacity'].copy()

# Removing unnecessary columns
columns_to_drop = ['_id', 'PROGRAM_ID', 'PROGRAM_MODEL', 'SERVICE_USER_COUNT', 'OVERNIGHT_SERVICE_TYPE', 'PROGRAM_AREA', 'ORGANIZATION_NAME', 'SHELTER_GROUP', 'LOCATION_NAME', 'LOCATION_ADDRESS', 'LOCATION_PROVINCE', 'PROGRAM_NAME']
bed_columns_to_drop = ['CAPACITY_ACTUAL_ROOM', 'CAPACITY_FUNDING_ROOM', 'OCCUPIED_ROOMS', 'UNOCCUPIED_ROOMS', 'UNAVAILABLE_ROOMS', 'OCCUPANCY_RATE_ROOMS']
room_columns_to_drop = ['CAPACITY_ACTUAL_BED', 'CAPACITY_FUNDING_BED', 'OCCUPIED_BEDS', 'UNOCCUPIED_BEDS', 'UNAVAILABLE_BEDS', 'OCCUPANCY_RATE_BEDS']

bed_capacity_df = bed_capacity_df.drop(columns_to_drop, axis=1)
bed_capacity_df = bed_capacity_df.drop(bed_columns_to_drop, axis=1)

room_capacity_df = room_capacity_df.drop(columns_to_drop, axis=1)
room_capacity_df = room_capacity_df.drop(room_columns_to_drop, axis=1)

# DATA PREPROCESSING -----------------------------------------------------------

# Convert OCCUPANCY_DATE to datetime
bed_capacity_df['OCCUPANCY_DATE'] = pd.to_datetime(bed_capacity_df['OCCUPANCY_DATE'],format='mixed')
room_capacity_df['OCCUPANCY_DATE'] = pd.to_datetime(room_capacity_df['OCCUPANCY_DATE'],format='mixed')

# Extract year, month, and day as separate features
bed_capacity_df['YEAR'] = bed_capacity_df['OCCUPANCY_DATE'].dt.year
bed_capacity_df['MONTH'] = bed_capacity_df['OCCUPANCY_DATE'].dt.month
bed_capacity_df['DAY'] = bed_capacity_df['OCCUPANCY_DATE'].dt.day

room_capacity_df['YEAR'] = room_capacity_df['OCCUPANCY_DATE'].dt.year
room_capacity_df['MONTH'] = room_capacity_df['OCCUPANCY_DATE'].dt.month
room_capacity_df['DAY'] = room_capacity_df['OCCUPANCY_DATE'].dt.day

print("Bed Capacity DF")
print(bed_capacity_df.head())

print("Room Capacity DF")
(room_capacity_df.head())

# Calculate daily average occupancy rate for beds
daily_avg_bed_occupancy = bed_capacity_df.groupby('OCCUPANCY_DATE')['OCCUPANCY_RATE_BEDS'].mean().reset_index()
# print(daily_avg_bed_occupancy.head())

# Calculate daily average occupancy rate for rooms
daily_avg_room_occupancy = room_capacity_df.groupby('OCCUPANCY_DATE')['OCCUPANCY_RATE_ROOMS'].mean().reset_index()
# print(daily_avg_room_occupancy.head())

combined_total_df = combined_df
combined_total_df['OCCUPANCY_RATE'] = np.where(combined_df['OCCUPANCY_RATE_BEDS'].isna(),
                                       combined_df['OCCUPANCY_RATE_ROOMS'],
                                       combined_df['OCCUPANCY_RATE_BEDS'])

"""# Finding Relationships Between Shelter Occupancy Rates

## For All Shelters
"""

# combined_total_df = combined_df
# combined_total_df['OCCUPANCY_RATE'] = np.where(combined_df['OCCUPANCY_RATE_BEDS'].isna(),
#                                        combined_df['OCCUPANCY_RATE_ROOMS'],
#                                        combined_df['OCCUPANCY_RATE_BEDS'])

# combined_total_df.head()


location_combined_df = combined_total_df.pivot_table(index='OCCUPANCY_DATE', columns='PROGRAM_ID', values='OCCUPANCY_RATE', aggfunc='mean')

correlation_matrix_room = location_combined_df.corr()

# correlation_matrix.head()
# print(correlation_matrix)

plt.figure(figsize=(75, 75))
sns.heatmap(correlation_matrix_room, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap for ALL Shelters')
plt.show()

"""## For Bed Based Capacity"""

bed_capacity_df.head()

# structured_bed_occupancy = bed_capacity_df.pivot_table(index='OCCUPANCY_DATE', columns='PROGRAM_ID', values='OCCUPANCY_RATE_BEDS', aggfunc='mean')

# correlation_matrix = structured_bed_occupancy.corr()

# # correlation_matrix.head()
# print(correlation_matrix)

plt.figure(figsize=(50, 50))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap for Bed Shelters')
plt.show()

"""## Room Based"""

structured_room_occupancy = room_capacity_df.pivot_table(index='OCCUPANCY_DATE', columns='LOCATION_ID', values='OCCUPANCY_RATE_ROOMS', aggfunc='mean')

correlation_matrix_room = structured_room_occupancy.corr()

# correlation_matrix.head()
# print(correlation_matrix)

plt.figure(figsize=(55, 55))
sns.heatmap(correlation_matrix_room, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap for Room Shelters')
plt.show()

"""# Finding the Most Impactful Attributes
For example, is postal code or program type more influential?

### Postal Code
"""

postcode_combined_df = combined_total_df.pivot_table(index='OCCUPANCY_DATE', columns='LOCATION_POSTAL_CODE', values='OCCUPANCY_RATE', aggfunc='mean')
correlation_matrix_postcode = postcode_combined_df.corr()

# correlation_matrix.head()
# print(correlation_matrix)

plt.figure(figsize=(75, 75))
sns.heatmap(correlation_matrix_postcode, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap by Postal Code')
plt.show()

"""## Sector

### Mens Sector
"""

mens_only_df = combined_total_df[combined_total_df['SECTOR'] == 'Men']
mens_only_df.head()

men_combined_df = mens_only_df.pivot_table(index='OCCUPANCY_DATE', columns='LOCATION_ID', values='OCCUPANCY_RATE', aggfunc='mean')
correlation_matrix_men = men_combined_df.corr()

# correlation_matrix.head()
# print(correlation_matrix)

plt.figure(figsize=(30, 30))
sns.heatmap(correlation_matrix_men, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap for Male Sector Only')
plt.show()

"""### Womens Sector"""

women_only_df = combined_total_df[combined_total_df['SECTOR'] == 'Women']
women_only_df.head()

women_combined_df = women_only_df.pivot_table(index='OCCUPANCY_DATE', columns='LOCATION_ID', values='OCCUPANCY_RATE', aggfunc='mean')
correlation_matrix_women = women_combined_df.corr()

# correlation_matrix.head()
# print(correlation_matrix)

plt.figure(figsize=(30, 30))
sns.heatmap(correlation_matrix_women, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap for Women Sector Only')
plt.show()

"""### Family Sector

"""

famillies_only_df = combined_total_df[combined_total_df['SECTOR'] == 'Families']
famillies_only_df.head()

famillies_combined_df = famillies_only_df.pivot_table(index='OCCUPANCY_DATE', columns='LOCATION_ID', values='OCCUPANCY_RATE', aggfunc='mean')
correlation_matrix_famillies = famillies_combined_df.corr()

# correlation_matrix.head()
# print(correlation_matrix)

plt.figure(figsize=(30, 30))
sns.heatmap(correlation_matrix_famillies, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap for Families Sector Only')
plt.show()

"""### Mixed Adult Sector"""

mixed_only_df = combined_total_df[combined_total_df['SECTOR'] == 'Mixed Adult']
mixed_only_df.head()

mixed_combined_df = mixed_only_df.pivot_table(index='OCCUPANCY_DATE', columns='LOCATION_ID', values='OCCUPANCY_RATE', aggfunc='mean')
correlation_matrix_mixed = mixed_combined_df.corr()

# correlation_matrix.head()
# print(correlation_matrix)

plt.figure(figsize=(30, 30))
sns.heatmap(correlation_matrix_mixed, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap for Mixed Adult Sector Only')
plt.show()

"""### Youth Sector"""

youth_only_df = combined_total_df[combined_total_df['SECTOR'] == 'Youth']
# youth_only_df.head()

youth_combined_df = youth_only_df.pivot_table(index='OCCUPANCY_DATE', columns='LOCATION_ID', values='OCCUPANCY_RATE', aggfunc='mean')
correlation_matrix_youth = youth_combined_df.corr()

# correlation_matrix.head()
# print(correlation_matrix)

plt.figure(figsize=(30, 30))
sns.heatmap(correlation_matrix_youth, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap for Youth Sector Only')
plt.show()

"""### Conclusion

In conclusion, it seems that the correlation does change based on the sector. Here, we ranked the amount of correlation within each sector as follows:
1. Mixed Adult
2. Women
3. Families
4. Youth
5. Men

## City
"""

# combined_total_df
unique_cities = combined_total_df
unique_cities['LOCATION_CITY'] = unique_cities['LOCATION_CITY'].str.strip()
unique_cities = unique_cities['LOCATION_CITY'].unique()

# unique_cities will contain a numpy array of unique entries in the 'LOCATION_CITY' column
print(unique_cities)

"""### North York"""

north_york_df = combined_total_df[combined_total_df['LOCATION_CITY'] == 'North York']

north_york_df.head()

north_york_corr_df = north_york_df.pivot_table(index='OCCUPANCY_DATE', columns='LOCATION_ID', values='OCCUPANCY_RATE', aggfunc='mean')
north_york_corr_matrix = north_york_corr_df.corr()

# correlation_matrix.head()
# print(correlation_matrix)

plt.figure(figsize=(10, 10))
sns.heatmap(north_york_corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap for North York Locations')
plt.show()

"""
### Etobicoke"""

etobicoke_df = combined_total_df[combined_total_df['LOCATION_CITY'] == 'Etobicoke']

etobicoke_corr_df = etobicoke_df.pivot_table(index='OCCUPANCY_DATE', columns='LOCATION_ID', values='OCCUPANCY_RATE', aggfunc='mean')
etobicoke_corr_matrix = etobicoke_corr_df.corr()

plt.figure(figsize=(5, 5))
sns.heatmap(etobicoke_corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap for Etobicoke Locations')
plt.show()