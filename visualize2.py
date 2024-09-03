import pandas as pd
import plotly.graph_objects as go

pipe_df = pd.read_excel("Data/72_Gassrorledninger_tabell_250220.xlsx", sheet_name="Fig-data", skiprows=22, index_col = 1)
node_df = pd.read_excel("Data/72_Gassrorledninger_tabell_250220.xlsx", sheet_name="Nodes")

# Drop rows with no elements in the Longitude column
node_df = node_df.dropna(subset=['Longitude'])
print(node_df)

# Create a scatter mapbox for nodes
fig = go.Figure()

# Add nodes to the figure
fig.add_trace(go.Scattermapbox(
    lat=node_df['Latitude'],
    lon=node_df['Longitude'],
    mode='markers+text',
    marker=dict(size=10, color='blue'),
    text=node_df['Node'],
    textposition="top right"
))

# Add lines for edges
for _, row in pipe_df.iterrows():
    from_node = node_df[node_df['Node'] == row['From']]
    to_node = node_df[node_df['Node'] == row['To']]
    if from_node.empty or to_node.empty:
        continue
    fig.add_trace(go.Scattermapbox(
        lat=[from_node['Latitude'].values[0], to_node['Latitude'].values[0]],
        lon=[from_node['Longitude'].values[0], to_node['Longitude'].values[0]],
        mode='lines',
        line=dict(width=2, color='red')
    ))

# Update layout for mapbox
fig.update_layout(
    mapbox=dict(
        style="open-street-map",
        center=dict(lat=node_df['Latitude'].mean(), lon=node_df['Longitude'].mean()),
        zoom=3
    ),
    margin=dict(l=0, r=0, t=0, b=0)
)

# Show the figure
fig.show()

