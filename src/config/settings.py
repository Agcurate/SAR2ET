# Since the best-performing SAR-based ET estimation model incorporates both ERA5 and DEM 
# variables along with SAR data, we need to pass a combination of all these variables to 
# the model during inference. This is done by concatenating the tensors along the channel
# dimension. The number of channels in the input tensor is 6 (number of channels in SAR) +
# 7 (number of channels in ERA5) + 5 (number of channels in DEM) = 18. 
N_CHANNELS = 6 + 7 + 5  # = 18

# The number of output channels is 1 (ET)
N_CLASSES = 1

# Band names for each data source
BANDNAMES_S1 = [
    "VV-DESC", 
    "VH-DESC", 
    "CR-DESC", 
    "VV-ASC", 
    "VH-ASC", 
    "CR-ASC"
    ]
BANDNAMES_ERA5 = [
    "temperature_2m",
    "dewpoint_temperature_2m",
    "total_precipitation_sum",
    "u_component_of_wind_10m",
    "v_component_of_wind_10m",
    "surface_net_solar_radiation_sum",
    "surface_pressure",
]
BANDNAMES_DEM = [
    "height",
    "slope",
    "aspect_cosine",
    "aspect_sine",
    "hillshade",
]

# Minimum and maximum values for the target ET data
LOW_ET = 0.0
HIGH_ET = 10.0
