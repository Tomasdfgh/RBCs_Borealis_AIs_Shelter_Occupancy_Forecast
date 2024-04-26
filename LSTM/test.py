import pandas as pd

def load_csv_to_pandas(file_path):
    try:
        # Load CSV file into a pandas DataFrame
        df = pd.read_csv(file_path, header=0, low_memory=False, encoding='unicode_escape')
        print("Number of rows in the DataFrame:", len(df))
        return df
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        return None
    except Exception as e:
        print("An error occurred:", str(e))
        return None

def remove_time_from_date_column(filename, output_filename):
    # Read the CSV file into a DataFrame
    df = load_csv_to_pandas(filename)

    # Remove "T00:00:00" from the 'OCCUPANCY_DATE' column
    df['OCCUPANCY_DATE'] = df['OCCUPANCY_DATE'].str.replace('T00:00:00', '')

    # Save the modified DataFrame back to a new CSV file
    df.to_csv(output_filename, index=False)

# Replace 'filename' with the path to your input CSV file
filename = r"C:\Users\tomng\Desktop\RBC's Borealis AI Lets Solve It\Datasets\daily-shelter-overnight-service-occupancy-capacity-2023.csv"

# Replace 'output_filename' with the desired name of the output CSV file
output_filename = r"C:\Users\tomng\Desktop\RBC's Borealis AI Lets Solve It\Datasets\daily-shelter-overnight-service-occupancy-capacity-2023_new.csv"

# Call the function to remove the time part from the date column and save the modified DataFrame
remove_time_from_date_column(filename, output_filename)

print("Modification complete. Output saved to", output_filename)
