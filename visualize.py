import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from shapely.wkt import loads as load_wkt
from shapely.geometry import MultiLineString, LineString
from shapely.errors import WKTReadingError
from shapely.geometry import box

receiving_terminals = ["St. Fergus", "Easington", "Dornum", "Emden", "Zeebrugge", "Dunkerque"]
processing_plants = ["Nyhamna", "Kollsnes", "Kårstø", "Vestprosess"]
pipelines = [
    "Europipe I",
    "Europipe II",
    "Franpipe",
    "Gjøa Gas Pipe",
    "Haltenpipe",
    "Kvitebjørn Pipeline",
    "Langeled North",
    "Langeled South",
    "Norne Gas Transport",
    "Norpipe",
    "OGT",
    "Ormen Lange",
    "Polarled",
    "Statpipe",
    "Tampen Link",
    "Vesterled",
    "Zeepipe",
    "Zeepipe II A",
    "Zeepipe II B"
]


# St. Fergus, Easington, and Dornum missing
# Nyhamna, Kollsnes, Kårstø, and Vestprosess missing

df = pd.read_excel("Data/Europe-Gas-Tracker-2024-05_REB.xlsx", sheet_name="Gas plants - data")

min_longitude = -3
max_longitude = 11
min_latitude = 50
max_latitude = 70

# Filter the data on the longitude and latitude
# df = df[(df["Longitude"] >= min_longitude) & (df["Longitude"] <= max_longitude) & (df["Latitude"] >= min_latitude) & (df["Latitude"] <= max_latitude)]

# Create a regex pattern that matches any of the receiving terminals
pattern = '|'.join(receiving_terminals+processing_plants)

df = df[df["Plant name"].str.contains(pattern)]
# print(df)
#
# Creating the map
fig = go.Figure(go.Scattermapbox(
    lat=df["Latitude"],
    lon=df["Longitude"],
    mode='markers',
    marker=go.scattermapbox.Marker(
        size=14
    ),
    text=df["Plant name"],
))

# Updating the layout of the map
fig.update_layout(
    mapbox=dict(
        style="carto-positron",  # Options: "open-street-map", "stamen-terrain", "carto-positron", "stamen-watercolor"
        center=dict(lat=55.0, lon=5.0),  # Centered over Northwest Europe
        zoom=4,
    ),
    margin={"r":0,"t":0,"l":0,"b":0}  # Removing extra margins around the map
)

# # Display the map
# fig.show()

pipeline_df = pd.read_excel("Data/Europe-Gas-Tracker-2024-05_REB.xlsx", sheet_name="Gas pipelines - data")

# pattern = '|'.join(pipelines)

# pipeline_df = pipeline_df[pipeline_df["PipelineName"].str.contains(pattern)]
# print(pipeline_df.shape)

# Function to clean and parse WKT strings
def clean_wkt_string(wkt_string):
    # Remove any potential invalid characters or extra text
    wkt_string = wkt_string.strip()
    # Optionally, you can add more cleaning logic here if needed
    return wkt_string


# Function to extract coordinates from WKT
def extract_coordinates_from_wkt(wkt_string):
    try:
        shape = load_wkt(clean_wkt_string(wkt_string))
        latitudes = []
        longitudes = []

        # Check if the shape is a MultiLineString or a LineString
        if isinstance(shape, MultiLineString):
            for linestring in shape.geoms:
                coords = list(linestring.coords)
                lons, lats = zip(*coords)
                longitudes.append(lons)
                latitudes.append(lats)
        elif isinstance(shape, LineString):
            coords = list(shape.coords)
            lons, lats = zip(*coords)
            longitudes.append(lons)
            latitudes.append(lats)

        return latitudes, longitudes

    except WKTReadingError as e:
        print(f"Error reading WKT: {e}")
        return [], []  # Return empty lists if there's an error

def filter_pipeline_by_bounds(shape, min_lon, max_lon, min_lat, max_lat):
    bounding_box = box(min_lon, min_lat, max_lon, max_lat)
    return shape.within(bounding_box)

filtered_pipeline_rows = []

for index, row in pipeline_df.iterrows():
    try:
        shape = load_wkt(clean_wkt_string(row['WKTFormat']))
        if isinstance(shape, (LineString, MultiLineString)):
            if filter_pipeline_by_bounds(shape, min_longitude, max_longitude, min_latitude, max_latitude):
                filtered_pipeline_rows.append(row)
    except WKTReadingError as e:
        print(f"Error reading WKT: {e}")

filtered_pipeline_df = pd.DataFrame(filtered_pipeline_rows)


# Adding pipeline traces to the map
terminals = {}
for index, row in filtered_pipeline_df.iterrows():
    latitudes, longitudes = extract_coordinates_from_wkt(row['WKTFormat'])
    start_location, end_location = row["StartLocation"], row["EndLocation"]
    terminals[start_location] = {"lon": longitudes[0][0], "lat": latitudes[0][0]}
    terminals[end_location] = {"lon": longitudes[0][-1], "lat": latitudes[0][-1]}
    for lats, lons in zip(latitudes, longitudes):
        fig.add_trace(go.Scattermapbox(
            lon=lons,
            lat=lats,
            mode='lines',
            line=dict(width=2, color='blue'),
            text=row["PipelineName"]
        ))

# Adding terminal markers to the map
for terminal, location in terminals.items():
    fig.add_trace(go.Scattermapbox(
        lon=[location["lon"]],
        lat=[location["lat"]],
        mode='markers',
        marker=go.scattermapbox.Marker(size=14, color='red'),
        text=terminal
    ))

fig.show()