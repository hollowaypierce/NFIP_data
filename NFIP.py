# Open FEMA - National Flood Insurance Policy Claims - Python Script
# This script is developed so that we can quickly pull information on claims for a given area

# Import packagesimport requests
import requests
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
import time
import matplotlib.pyplot as plt
import folium
from folium import Choropleth
from pygris import block_groups
import branca.colormap as cm


# --- Step 2: Download NFIP Claims for NC from OpenFEMA ---
print("Downloading NFIP claims for North Carolina...")
# Correct URL for OpenFEMA v2 NFIP claims
url = "https://www.fema.gov/api/open/v2/FimaNfipClaims"
state_filter = "state eq 'NC'"
page_size = 10000
all_data = []
skip = 0

# Make request
while True:
    params = {
        "$filter": state_filter,
        "$top": page_size,
        "$skip": skip,
        "$format": "json"
    }

    response = requests.get(url, params=params, verify=False)
    if response.status_code != 200:
        raise RuntimeError(f"Request failed: {response.status_code}\n{response.text}")

    data = response.json().get("FimaNfipClaims", [])
    if not data:
        print("No more data to fetch.")
        break

    all_data.extend(data)
    print(f"Fetched records {skip + 1} to {skip + len(data)}")

    if len(data) < page_size:
        break

    skip += page_size
    time.sleep(1.5)  # be kind to the API

# Convert to DataFrame
df = pd.DataFrame(all_data)
print(f"\n Total records downloaded: {len(df)}")

# Display head to verify
print(df.head())

# create percent paid on damage column
df['PercentPaid'] = (
    (df['netBuildingPaymentAmount'].fillna(0) + df['netContentsPaymentAmount'].fillna(0)) /
    (df['buildingDamageAmount'].fillna(0) + df['contentsDamageAmount'].fillna(0))
) * 100
df['PercentPaid'].replace([np.inf, -np.inf], np.nan, inplace=True)
df['PercentPaid'] = df['PercentPaid'].round(0).astype('Int64')


# --- Step 3: Filter Data to intrests ---

# --- Step 3a: Filter to specific event ---
helene_df = df[
    (df['floodEvent'] == 'Hurricane Helene') |
    (df['eventDesignationNumber'] == 'NC0124')
].copy()

# --- Step 3b: Filter to intrested values ---
# Make sure originalConstructionDate is datetime (or at least year)
helene_df['originalConstructionDate'] = pd.to_datetime(helene_df['originalConstructionDate'], errors='coerce')
helene_df['constructionYear'] = helene_df['originalConstructionDate'].dt.year

# 1. Total claims per block group
claim_counts = helene_df.groupby('censusBlockGroupFips').size().reset_index(name='claim_count')

# 2. Count of each flood zone per block group
# Map original flood zones to combined groups
zone_map = {
    'AE': 'AE',
    'A04': 'AE',
    'A08': 'AE',
    'B': 'X',
    'C': 'X',
    'X': 'X'
}

# Create a new column with combined flood zones
helene_df['floodZoneCombined'] = helene_df['floodZoneCurrent'].map(zone_map).fillna(helene_df['floodZoneCurrent'])

# Group by block group and combined flood zone
floodzone_counts = helene_df.groupby(['censusBlockGroupFips', 'floodZoneCombined']).size().reset_index(name='count')

# Calculate total counts per block group
blockgroup_totals = floodzone_counts.groupby('censusBlockGroupFips')['count'].transform('sum')

# Calculate percent of each combined flood zone within block group
floodzone_counts['percent'] = (floodzone_counts['count'] / blockgroup_totals) * 100
floodzone_counts['percent'] = floodzone_counts['percent'].round(1)

# make into wide format for ease
floodzone_wide = floodzone_counts.pivot_table(
    index='censusBlockGroupFips',
    columns='floodZoneCombined',
    values='percent',
    fill_value=0  # optional: fill missing combinations with 0
).reset_index()

# 3. Average year of construction
avg_year = helene_df.groupby('censusBlockGroupFips')['constructionYear'].mean().reset_index(name='avgConstructionYear')

# 4. ave percent paid on each claim per block gorup
# Make sure PercentPaid is numeric (if not already)
helene_df['PercentPaid'] = pd.to_numeric(helene_df['PercentPaid'], errors='coerce')
avg_percent_paid = helene_df.groupby('censusBlockGroupFips')['PercentPaid'].mean().reset_index()
avg_percent_paid['PercentPaid'] = avg_percent_paid['PercentPaid'].round(1)


# 5. total paid in the blockgroup
helene_df['PaidAmount'] = helene_df['netBuildingPaymentAmount'].fillna(0) + helene_df['netContentsPaymentAmount'].fillna(0)
# Calculate total paid amount per block group
blockgroup_totalPaid = (
    helene_df.groupby('censusBlockGroupFips')['PaidAmount']
    .sum()
    .round(0)
    .reset_index(name='totalPaid')
)

# Combine all together
summary_df = claim_counts.merge(floodzone_wide, on='censusBlockGroupFips', how='left')
summary_df = summary_df.merge(avg_year, on='censusBlockGroupFips', how='left')
summary_df = summary_df.merge(avg_percent_paid, on='censusBlockGroupFips', how='left')
summary_df = summary_df.merge(blockgroup_totalPaid, on='censusBlockGroupFips', how='left')

# format for dollars later
summary_df['totalPaidFormatted'] = summary_df['totalPaid'].apply(lambda x: "${:,.0f}".format(x) if pd.notnull(x) else "")

# rename for joining later
summary_df = summary_df.rename(columns={'censusBlockGroupFips': 'GEOID'})

# round off years
summary_df['avgConstructionYear'] = summary_df['avgConstructionYear'].round(0).astype('Int64').astype(str)


# --- Step 5: Map selected claims ---

# --- Step 5a: pull census block group boundaries for WNC ---
# List of counties you want
target_counties = [
    "Avery", "Buncombe", "Burke", "Cherokee", "Clay", "Graham", "Haywood",
    "Henderson", "Jackson", "Macon", "McDowell", "Mitchell", "Polk",
    "Rutherford", "Swain", "Transylvania", "Yancey"
]

# Download block groups for each county using pygris
bg_gdfs = []

for county in target_counties:
    print(f"Downloading block groups for {county} County...")
    gdf = block_groups(state="NC", county=county, year = 2021, cache = True)
    bg_gdfs.append(gdf)

# Combine into a single GeoDataFrame
gdf_block_groups = pd.concat(bg_gdfs, ignore_index=True).to_crs(epsg=4326)


# join summary_Df and blockgroups geodataframe
gdf_joined = gdf_block_groups.merge(summary_df, on='GEOID', how='left')

# Plot points
# Ensure GeoDataFrame is in WGS84
gdf_joined = gdf_joined.to_crs(epsg=4326)

# Center the map on filtered claims
map_center = [
    gdf_joined.geometry.centroid.y.mean(),
    gdf_joined.geometry.centroid.x.mean()
]

# Create folium map
m = folium.Map(location=map_center, 
               zoom_start=9,
               tiles="CartoDB positron")

# Build color scale for non-zero values only
valid_values = gdf_joined['claim_count'].dropna()
valid_values = valid_values[valid_values > 0]
min_val, max_val = valid_values.min(), valid_values.max()

colormap = cm.linear.PuRd_09.scale(min_val, max_val)
colormap.caption = "Claim Count"
colormap.add_to(m)

# Style function that skips fill for 0 or None
def style_function(feature):
    value = feature['properties']['claim_count']
    if value in [None, 0]:
        return {
            'fillColor': 'transparent',
            'color': 'gray',
            'weight': 0.3,
            'fillOpacity': 0.1,
        }
    return {
        'fillColor': colormap(value),
        'color': 'black',
        'weight': 0.65,
        'fillOpacity': 0.7,
    }

# List all flood zone fields you want to include
flood_zone_fields = ["A", "AE", "AO", "X"]

# Construct tooltip
tooltip = folium.GeoJsonTooltip(
    fields=["GEOID", "claim_count", "avgConstructionYear", "totalPaidFormatted", "PercentPaid"] + flood_zone_fields,
    aliases=[
        "Block Group GEOID:", 
        "Number of Claims:", 
        "Avg Construction Year:", 
        "Total Dollars Paid",
        "Avg Percent Paid on Claim:"
    ] + [f"Flood Zone {zone} (%):" for zone in flood_zone_fields],
    sticky=True,
    localize=True,
    labels=True,
    style=(
        "background-color: white; "
        "color: #333; "
        "font-family: Arial; "
        "font-size: 12px; "
        "padding: 4px;"
    )
)
# Add interactive GeoJson layer
folium.GeoJson(
    gdf_joined,
    name="Claims by Block Group",
    style_function=style_function,
    tooltip=tooltip
).add_to(m)

# Add layer control
folium.LayerControl().add_to(m)

# Display map
m