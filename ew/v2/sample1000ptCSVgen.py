import csv
import math

# Survey parameters
num_stations = 200
station_interval = 5  # meters
base_easting = 500000
base_northing = 2000000
lateral_offsets = [-5, -2.5, 0, 2.5, 5]
point_codes = ['A', 'B', 'C', 'D', 'E']

def simulate_elevation(sta_idx, offset):
    # Simulate a valley profile: descent, bottom, ascent
    total_len = (num_stations - 1) * station_interval
    x = sta_idx * station_interval
    if x < total_len/2:
        base_elev = 110 - (x/total_len)*30  # down 15m to valley center
    else:
        base_elev = 95 + ((x-total_len/2)/total_len)*30  # up 15m
    # Add cross-slope (road camber) and roughness
    elev = base_elev + (-0.03 * offset) + (math.sin(sta_idx/12.0)*0.15) + (offset * 0.02 * math.cos(sta_idx/7.0))
    return round(elev,2)

with open('DGPS_Road_Survey_1000_Points.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['PointID','Easting','Northing','Elevation','Code'])
    for sta in range(num_stations):
        northing = base_northing + sta * station_interval
        for i, offset in enumerate(lateral_offsets):
            easting = base_easting + offset
            code = 'CL' if i==2 else ''
            pid = f'P{sta:03}{point_codes[i]}'
            elevation = simulate_elevation(sta, offset)
            writer.writerow([pid, round(easting,2), round(northing,2), elevation, code])
print("CSV file created as: DGPS_Road_Survey_1000_Points.csv")
