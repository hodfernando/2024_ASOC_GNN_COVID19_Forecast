import os
import glob
import igraph as ig
import pandas as pd
import geopandas as gpd
import numpy as np
import plotly.express as px

# Obtém o diretório atual do script (pasta 'codes')
pasta_atual = os.path.dirname(os.path.realpath(__file__))

# Retorna o diretório pai (pasta do projeto)
pasta_projeto = os.path.dirname(pasta_atual)

# Define o caminho para a pasta 'raw_data' dentro do projeto
pasta_raw_data = os.path.join(pasta_projeto, 'raw_data', 'China')

# Caminho para a pasta "dataverse_files"
networks_dir = os.path.join(pasta_raw_data, "networks")

# Lista apenas arquivos .GraphML no diretório de redes
networks_files = [
    next((file for file in glob.glob(os.path.join(networks_dir, "baidu_in_20200101.GraphML")) if os.path.isfile(file)),
         None)]

# reading the network from file
g = ig.Graph.Read_GraphML(networks_files[0])

g.vs['label'] = g.vs['City_EN']

min_x = np.min(g.vs["xcoord"])
max_x = np.max(g.vs["xcoord"])
min_y = np.min(g.vs["ycoord"])
max_y = np.max(g.vs["ycoord"])

dim_x = max_x - min_x
dim_y = max_y - min_y
scale = 20.0
width = dim_x * scale
height = dim_y * scale
print(width, height)

metrics = ['degree']

for metric in metrics:
    # Metrics
    df = pd.read_csv(os.path.join(pasta_raw_data, 'baidu_in_20200101', 'metrics', metric + '.csv'), delimiter=';',
                     header=None)
    df.columns = ['id', 'city_code', 'metric']
    print(df['metric'])

    g.vs[metric] = df['metric']

    g.vs["size"] = 12
    g.vs["color"] = "orange"
    g.es["color"] = "gray"

    layout = []
    for i in range(g.vcount()):

        # coordinates
        layout.append((g.vs[i]["xcoord"], -g.vs[i]["ycoord"]))

        g.vs[i]["vertex_shape"] = "circle"
        g.vs[i]["label"] = ""

    g.es['weight_plt'] = 1
    g.es["edge_width"] = 1

    visual_style = {
        "vertex_size": g.vs["size"],
        "vertex_shape": g.vs["vertex_shape"],
        "vertex_label_size": 20,
        "vertex_label_dist": 1,
        "vertex_label_color": "white",
        "edge_width": g.es['edge_width'],
        "layout": layout,
        "bbox": (width, height),
        "margin": 30,
        "edge_arrow_size": 0.2
    }

    ig.plot(g, pasta_atual + '/mobility_map_china.pdf', **visual_style)



    # Load the shapefile of China municipalities
    gdf = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    gdf = gdf[gdf['name'] == 'China']

    # Create a DataFrame for the cities
    cities_df = pd.DataFrame({
        'lon': [g.vs[i]["xcoord"] for i in range(g.vcount())],
        'lat': [g.vs[i]["ycoord"] for i in range(g.vcount())],
        'label': [g.vs[i]["label"] for i in range(g.vcount())]
    })

    # Create a choropleth map using Plotly Express
    fig = px.choropleth_mapbox(gdf,
                               geojson=gdf.geometry,
                               locations=gdf.index,
                               mapbox_style="carto-positron",
                               center={"lat": gdf.geometry.centroid.y.mean(), "lon": gdf.geometry.centroid.x.mean()},
                               zoom=2.0,
                               title='Choropleth Map of China')

    # Add city points to the map
    fig.add_scattermapbox(lon=cities_df['lon'], lat=cities_df['lat'], text=cities_df['label'],
                          mode='markers', marker=dict(size=5, color='orange'), name='Cities')

    # Save the choropleth map
    fig.write_image(pasta_atual + '/china_choropleth_map.pdf')