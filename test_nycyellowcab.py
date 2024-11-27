import sys
import numpy as np
import timeit
sys.path.append('./build')
import pandas as pd
import haversine_library

# Longitutde and Latitude Values
NYC_BBOX = [-74.15, 40.5774, -73.7004, 40.9176]

# Empty list to store the dataframes
df_list = []

# for loop to iterate ovver the 12 months
for month in range(1, 13):
    # Construct the file path for each month's data
    file_path = f"/data/csc59866_f24/tlcdata/yellow_tripdata_2009-{month:02d}.parquet"

    # Read the parquet file
    taxi = pd.read_parquet(file_path)

    # Filter the rows where the pickup and drop-off locations are within the bounding box
    taxi_filtered = taxi[
        (taxi['Start_Lon'] >= NYC_BBOX[0]) & (taxi['Start_Lon'] <= NYC_BBOX[2]) &
        (taxi['Start_Lat'] >= NYC_BBOX[1]) & (taxi['Start_Lat'] <= NYC_BBOX[3]) &
        (taxi['End_Lon'] >= NYC_BBOX[0]) & (taxi['End_Lon'] <= NYC_BBOX[2]) &
        (taxi['End_Lat'] >= NYC_BBOX[1]) & (taxi['End_Lat'] <= NYC_BBOX[3])
    ]

    # Append the filtered dataframe for the current month to the list
    df_list.append(taxi_filtered)


combined_taxi_data = pd.concat(df_list, ignore_index=True)


x1 = combined_taxi_data['Start_Lon'].to_numpy()
y1 = combined_taxi_data['Start_Lat'].to_numpy()
x2 = combined_taxi_data['End_Lon'].to_numpy()
y2 = combined_taxi_data['End_Lat'].to_numpy()


size = len(x1)


dist = np.zeros(size)


def haversine_python(x1, y1, x2, y2):
    R = 6371.0
    lat1 = np.radians(y1)
    lon1 = np.radians(x1)
    lat2 = np.radians(y2)
    lon2 = np.radians(x2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

# Measure run time for regular Python:
start_time_python = timeit.default_timer()
distances_python = haversine_python(x1, y1, x2, y2)
end_time_python = timeit.default_timer()
python_runtime = end_time_python - start_time_python

print(f"Python Haversine runtime: {python_runtime:.4f} seconds")

# Measure CUDA runtime:
start_time_cuda = timeit.default_timer()
haversine_library.haversine_distance(size, x1, y1, x2, y2, dist)
end_time_cuda = timeit.default_timer()
cuda_runtime = end_time_cuda - start_time_cuda

print(f"CUDA Haversine runtime (end-to-end): {cuda_runtime:.4f} seconds")

# Compare both the Python and CUDA runtimes
print(f"Python runtime: {python_runtime:.4f} seconds")
print(f"CUDA runtime: {cuda_runtime:.4f} seconds")
if python_runtime < cuda_runtime:
    print("Python implementation was superior in respect to time!")
else:
    print("CUDA implementation was superior in respect to time!")
