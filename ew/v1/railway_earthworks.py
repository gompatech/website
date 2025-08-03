import json
import pandas as pd
import numpy as np
from scipy.interpolate import griddata, interp1d
import plotly.graph_objects as go

def parse_slope(ratio_str):
    """Converts 'H:V' string to a numerical slope factor."""
    h, v = map(float, ratio_str.split(':'))
    return h / v

def get_cross_section_points(center_x, center_y, direction_vector, width, num_points=100):
    """Generates points for a cross-section line perpendicular to the track direction."""
    perp_vector = np.array([-direction_vector[1], direction_vector[0]])
    half_width = width / 2
    start_point = np.array([center_x, center_y]) - half_width * perp_vector
    end_point = np.array([center_x, center_y]) + half_width * perp_vector
    x_coords = np.linspace(start_point[0], end_point[0], num_points)
    y_coords = np.linspace(start_point[1], end_point[1], num_points)
    distances = np.linspace(-half_width, half_width, num_points)
    return x_coords, y_coords, distances

def get_material_type_and_cost(chainage, config):
    """Finds material type and cut cost for a given chainage."""
    for material in config['material_definitions']:
        if material['start_chainage'] <= chainage < material['end_chainage']:
            return material['type'], material['unit_cost_cut_per_m3']
    return "Unknown", 0

def is_in_special_section(chainage, config):
    """Checks if a chainage is within a special section (bridge/tunnel)."""
    for section in config['special_sections']:
        if section['start_chainage'] <= chainage < section['end_chainage']:
            return True
    return False

def calculate_earthworks():
    """Main function to perform earthworks calculation and visualization."""
    # 1. --- Data Ingestion and Setup ---
    print("Step 1: Loading data and configuration...")
    with open('config.json', 'r') as f:
        config = json.load(f)

    egm_df = pd.read_csv('egm.csv')
    alignment_df = pd.read_csv('alignment.csv')

    # Create ground interpolator function
    ground_points = egm_df[['Easting', 'Northing']].values
    ground_elevations = egm_df['Elevation'].values
    egm_interpolator = lambda points: griddata(ground_points, ground_elevations, points, method='linear')
    
    # Create alignment interpolator functions
    align_interpolator_x = interp1d(alignment_df['Chainage'], alignment_df['Easting'], kind='linear')
    align_interpolator_y = interp1d(alignment_df['Chainage'], alignment_df['Northing'], kind='linear')
    align_interpolator_z = interp1d(alignment_df['Chainage'], alignment_df['Design_Elevation'], kind='linear')
    
    # Parse config parameters
    station_interval = config['station_interval_m']
    formation_width = config['formation_width_m']
    cut_slope = parse_slope(config['side_slope_cut_ratio_hv'])
    fill_slope = parse_slope(config['side_slope_fill_ratio_hv'])

    # 2. --- Earthworks Calculation ---
    print(f"Step 2: Calculating volumes at {station_interval}m intervals...")
    results = []
    total_cut_volume = 0
    total_fill_volume = 0
    cost_by_material = {}
    
    chainages = np.arange(alignment_df['Chainage'].min(), alignment_df['Chainage'].max(), station_interval)
    
    last_cut_area, last_fill_area = 0, 0
    all_corridor_points = []

    for i, chainage in enumerate(chainages):
        if is_in_special_section(chainage, config):
            print(f"  - Chainage {chainage}m: In special section (bridge/tunnel). Skipping.")
            cut_area, fill_area = 0, 0
        else:
            # Get center point and direction
            cx = align_interpolator_x(chainage)
            cy = align_interpolator_y(chainage)
            cz = align_interpolator_z(chainage)
            
            # Get direction vector for perpendicular cross-section
            next_chainage = chainage + 1
            nx = align_interpolator_x(next_chainage)
            ny = align_interpolator_y(next_chainage)
            direction = np.array([nx - cx, ny - cy])
            direction = direction / np.linalg.norm(direction)
            
            # Define ground and design cross-sections
            cs_width = 100 # A sufficiently wide cross-section to find catch points
            cs_x, cs_y, cs_dist = get_cross_section_points(cx, cy, direction, cs_width)
            cs_ground_elev = egm_interpolator(np.vstack([cs_x, cs_y]).T)
            
            # Remove NaN values from interpolation at edges
            valid_indices = ~np.isnan(cs_ground_elev)
            cs_dist = cs_dist[valid_indices]
            cs_ground_elev = cs_ground_elev[valid_indices]

            # Design template points
            half_form = formation_width / 2
            design_xs = [-half_form, half_form]
            design_ys = [cz, cz]
            
            # Find catch points (intersections of side slopes with ground)
            # Left side
            left_ground_interp = interp1d(cs_dist[cs_dist <= -half_form], cs_ground_elev[cs_dist <= -half_form], fill_value="extrapolate")
            if cz > left_ground_interp(-half_form): # Fill
                slope = -1 / fill_slope
                intercept = cz - slope * (-half_form)
            else: # Cut
                slope = 1 / cut_slope
                intercept = cz - slope * (-half_form)
            
            # Simple numerical search for intersection
            left_catch_x = -half_form
            for d in cs_dist[cs_dist < -half_form][::-1]:
                if abs((slope * d + intercept) - left_ground_interp(d)) < 0.1: # Tolerance
                    left_catch_x = d
                    break

            # Right side
            right_ground_interp = interp1d(cs_dist[cs_dist >= half_form], cs_ground_elev[cs_dist >= half_form], fill_value="extrapolate")
            if cz > right_ground_interp(half_form): # Fill
                slope = 1 / fill_slope
                intercept = cz - slope * half_form
            else: # Cut
                slope = -1 / cut_slope
                intercept = cz - slope * half_form

            right_catch_x = half_form
            for d in cs_dist[cs_dist > half_form]:
                 if abs((slope * d + intercept) - right_ground_interp(d)) < 0.1: # Tolerance
                    right_catch_x = d
                    break
            
            # Create the final polygon for area calculation
            design_poly_xs = [left_catch_x, -half_form, half_form, right_catch_x]
            design_poly_ys = [left_ground_interp(left_catch_x), cz, cz, right_ground_interp(right_catch_x)]
            
            # Filter ground points within the construction width
            mask = (cs_dist >= left_catch_x) & (cs_dist <= right_catch_x)
            ground_poly_xs = np.concatenate(([left_catch_x], cs_dist[mask], [right_catch_x]))
            ground_poly_ys = np.concatenate(([left_ground_interp(left_catch_x)], cs_ground_elev[mask], [right_ground_interp(right_catch_x)]))
            
            # Calculate cut and fill areas using Shoelace formula
            # Combine polygons and find area of difference
            full_poly_x = np.concatenate((design_poly_xs, ground_poly_xs[::-1]))
            full_poly_y = np.concatenate((design_poly_ys, ground_poly_ys[::-1]))
            
            area = 0.5 * np.abs(np.dot(full_poly_x, np.roll(full_poly_y, 1)) - np.dot(full_poly_y, np.roll(full_poly_x, 1)))

            # Differentiate cut and fill
            if cz > np.mean(cs_ground_elev[(cs_dist > -half_form) & (cs_dist < half_form)]):
                fill_area = area
                cut_area = 0
            else:
                cut_area = area
                fill_area = 0

            # Store points for 3D visualization
            perp_vec = np.array([-direction[1], direction[0]])
            corridor_center = np.array([cx, cy])
            for dx, elev in zip(np.concatenate((design_poly_xs, ground_poly_xs)), np.concatenate((design_poly_ys, ground_poly_ys))):
                point = corridor_center + dx * perp_vec
                all_corridor_points.append([point[0], point[1], elev])


        if i > 0:
            length = chainages[i] - chainages[i-1]
            cut_volume = (cut_area + last_cut_area) / 2 * length
            fill_volume = (fill_area + last_fill_area) / 2 * length
            
            total_cut_volume += cut_volume
            total_fill_volume += fill_volume
            
            material_type, cut_cost = get_material_type_and_cost(chainage, config)
            fill_cost = config['unit_cost_fill_per_m3']
            
            if material_type not in cost_by_material:
                cost_by_material[material_type] = {'cut_volume': 0, 'cost': 0}
            
            cost_by_material[material_type]['cut_volume'] += cut_volume
            cost_by_material[material_type]['cost'] += cut_volume * cut_cost
            
        last_cut_area, last_fill_area = cut_area, fill_area

    total_fill_cost = total_fill_volume * config['unit_cost_fill_per_m3']

    # 3. --- Output Summary ---
    print("\n--- Earthworks Calculation Summary ---")
    print(f"Total Cut Volume: {total_cut_volume:,.2f} m³")
    print(f"Total Fill Volume: {total_fill_volume:,.2f} m³")
    print("\n--- Preliminary Cost Estimation (INR) ---")
    total_cost = total_fill_cost
    for material, data in cost_by_material.items():
        print(f"  - Cut Cost ({material}): {data['cost']:,.2f} (for {data['cut_volume']:,.2f} m³)")
        total_cost += data['cost']
    print(f"  - Fill Cost (All): {total_fill_cost:,.2f} (for {total_fill_volume:,.2f} m³)")
    print("-----------------------------------------")
    print(f"Total Estimated Earthworks Cost: ₹ {total_cost:,.2f}")
    print("-----------------------------------------")


    # 4. --- Visualization ---
    print("\nStep 3: Generating 3D visualization...")
    fig = go.Figure()

    # EGM Surface
    fig.add_trace(go.Mesh3d(
        x=egm_df['Easting'], y=egm_df['Northing'], z=egm_df['Elevation'],
        opacity=0.6,
        color='lightgreen',
        name='Existing Ground'
    ))

    # Railway Corridor
    if all_corridor_points:
        corridor_points = np.array(all_corridor_points)
        fig.add_trace(go.Mesh3d(
            x=corridor_points[:,0], y=corridor_points[:,1], z=corridor_points[:,2],
            opacity=0.8,
            color='lightblue',
            name='Railway Corridor'
        ))

    # Alignment Centerline
    fig.add_trace(go.Scatter3d(
        x=align_interpolator_x(chainages),
        y=align_interpolator_y(chainages),
        z=align_interpolator_z(chainages),
        mode='lines',
        line=dict(color='red', width=5),
        name='Track Centerline'
    ))

    fig.update_layout(
        title='GompaTech Railway Earthworks 3D Visualization',
        scene=dict(
            xaxis_title='Easting (m)',
            yaxis_title='Northing (m)',
            zaxis_title='Elevation (m)',
            aspectratio=dict(x=1, y=1, z=0.3)
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    output_filename = 'earthworks_visualization.html'
    fig.write_html(output_filename)
    print(f"\nSuccess! Visualization saved to '{output_filename}'")


if __name__ == '__main__':
    calculate_earthworks()