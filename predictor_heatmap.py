import plotly.graph_objects as go
import pandas as pd
import numpy as np

"""
This is simply a mockup to demonstrate how might the vulnerability scores be visualized in a heatmap.
The y-axis intends to represent either the provinces or marginalized communities in Canada, while the x-axis
represents different climate indicators. The values in the heatmap represent the vulnerability scores for each
combination of province and climate indicator.
"""

# Define provinces and climate indicators
provinces = [
    'British Columbia',
    'Alberta',
    'Saskatchewan',
    'Manitoba',
    'Ontario',
    'Quebec',
    'New Brunswick',
    'Nova Scotia',
    'Prince Edward Island',
    'Newfoundland and Labrador',
    'Yukon',
    'Northwest Territories',
    'Nunavut'
]

indicators = [
    'Temperature Extremes',
    'Precipitation Change',
    'Sea Level Rise Risk',
    'Permafrost Thaw',
    'Wildfire Risk',
    'Flood Risk',
    'Drought Risk',
    'Storm Frequency',
    'Coastal Erosion'
]

# Generate sample vulnerability scores (0-1 scale)
# In real implementation, these would be actual vulnerability assessments
np.random.seed(42)  # For reproducibility
data = np.random.uniform(0, 1, size=(len(provinces), len(indicators)))

# Adjust some values to reflect known vulnerabilities
# Coastal provinces - higher sea level rise risk
coastal_provinces = ['British Columbia', 'Nova Scotia', 'New Brunswick', 
                    'Prince Edward Island', 'Newfoundland and Labrador']
for province in coastal_provinces:
    idx = provinces.index(province)
    data[idx, indicators.index('Sea Level Rise Risk')] = np.random.uniform(0.7, 1.0)

# Northern territories - higher permafrost thaw risk
northern_territories = ['Yukon', 'Northwest Territories', 'Nunavut']
for territory in northern_territories:
    idx = provinces.index(territory)
    data[idx, indicators.index('Permafrost Thaw')] = np.random.uniform(0.8, 1.0)

# Prairie provinces - higher drought risk
prairie_provinces = ['Alberta', 'Saskatchewan', 'Manitoba']
for province in prairie_provinces:
    idx = provinces.index(province)
    data[idx, indicators.index('Drought Risk')] = np.random.uniform(0.7, 0.9)

# Create heatmap
fig = go.Figure(data=go.Heatmap(
    z=data,
    x=indicators,
    y=provinces,
    colorscale='RdYlBu_r',  # Similar to the color scheme in your example
    zmin=0,
    zmax=1,
))

# Update layout
fig.update_layout(
    title='Climate Vulnerability Indicators by Canadian Province',
    xaxis_title='Vulnerability Indicators',
    yaxis_title='Provinces',
    xaxis={'side': 'bottom'},
    yaxis={'autorange': 'reversed'},  # To match the example orientation
    width=1000,
    height=800,
    font=dict(size=10),
)

# Rotate x-axis labels for better readability
fig.update_xaxes(tickangle=45)

# Show the figure
fig.show()