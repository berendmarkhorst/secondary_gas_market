import geopandas as gpd
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# Load the data
def load_data(inputfile):
    df_pipes = pd.read_excel(inputfile, sheet_name="pipes")

    # Drop all the rows with missing geometry
    df_pipes = df_pipes.dropna(subset=['geometry'])

    gdf_pipes = gpd.GeoDataFrame(df_pipes, geometry=gpd.GeoSeries.from_wkt(df_pipes['geometry']))

    df_facilities = pd.read_excel(inputfile, sheet_name="facilities")

    return gdf_pipes, df_facilities


def interactive_plot(gdf_pipes, gdf_facilities, extra_info=False, labels_ruud=False):

    # Extract coordinates and hover text for the pipelines
    pipe_latitudes = []
    pipe_longitudes = []
    hover_texts = []

    arrow_latitudes = []
    arrow_longitudes = []
    arrow_texts = []
    arrow_angles = []

    if extra_info:
        gdf_facilities["Hover"] = gdf_facilities["Name"] # Was eerst Node! # + " (" + gdf_facilities["Capacity"].astype(str) + " capacity)"
    elif labels_ruud:
        gdf_facilities["Hover"] = gdf_facilities['Name'].astype(str) + "<br>" + gdf_facilities['Ruud'].astype(str)
    else:
        gdf_facilities["Hover"] = gdf_facilities["Name"]

    for idx, line in enumerate(gdf_pipes.geometry):
        pipe_name = gdf_pipes.iloc[idx]['Name'] # Was eerst mapLabel
        if extra_info:
            pipe_name += f" ({gdf_pipes.iloc[idx]['Flow']} flow)"
        x, y = line.xy
        pipe_longitudes.extend(x)
        pipe_latitudes.extend(y)

        # Compute arrow positions (for simplicity, using the midpoint here)
        mid_idx = len(x) // 2 - 1
        arrow_latitudes.append(y[mid_idx])
        arrow_longitudes.append(x[mid_idx])
        arrow_texts.append(gdf_pipes.iloc[idx]['Name'])

        # Calculate the arrow angle based on the direction between two points
        dx = x[mid_idx + 1] - x[mid_idx]
        dy = y[mid_idx + 1] - y[mid_idx]
        arrow_angle = np.arctan2(dy, dx) * 180 / np.pi
        arrow_angles.append(arrow_angle)

        hover_texts.extend([pipe_name] * len(x))
        pipe_longitudes.append(None)  # None to break the line between different LineStrings
        pipe_latitudes.append(None)
        hover_texts.append(None)

    # Create a Plotly figure
    fig = go.Figure()

    # Add the pipelines (lines) to the plot with hover text
    fig.add_trace(go.Scattermapbox(
        lat=pipe_latitudes,
        lon=pipe_longitudes,
        mode='lines',
        line=dict(width=2, color='red'),
        text=hover_texts,  # Add hover text
        hoverinfo='text',
        name='Pipelines'
    ))

    # Add the facilities (points) to the plot
    fig.add_trace(go.Scattermapbox(
        lat=gdf_facilities["Lat"],
        lon=gdf_facilities["Lon"],
        mode='markers',
        marker=go.scattermapbox.Marker(size=10, color='blue'),
        text=gdf_facilities['Hover'],  # Assuming you have a column 'facility_name'
        hoverinfo='text',
        name='Facilities'
    ))

    # Add arrows for flow direction
    fig.add_trace(go.Scattermapbox(
        lat=arrow_latitudes,
        lon=arrow_longitudes,
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=100,
            symbol='arrow-bar-up',
            angle=arrow_angles,  # This sets the direction of the arrows
            color='black',
            allowoverlap=False
        ),
        text=arrow_texts,
        hoverinfo='text',
        name='Flow Direction'
    ))


    # Set up the layout for the map
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox_zoom=5,
        mapbox_center={"lat": 60, "lon": 15.2551},  # Center on North West Europe
        margin={"r":0,"t":0,"l":0,"b":0}
    )

    # Add title
    fig.update_layout(title_text='Pipeline and Facilities Visualization')

    # fig.write_html("network_visualization.html")

    fig.show()

    return fig

if __name__ == "__main__":
    gdf_pipes, df_facilities = load_data("Data/result network Norway_REB.xlsx")

    _ = interactive_plot(gdf_pipes, df_facilities, extra_info=False, labels_ruud=True)
