# Description: Script to predict ET using best-performing SAR-based ET estimation model
#
# This script consists of multiple steps:
# 1. Initialize and load the pretrained model
# 2. Load input data (S1, ERA5 and DEM tensors)
# 3. Perform inference
# 4. Save and plot results
#
# The script can be run from the command line as follows:
# python predict_et.py \
#   --pretrained_model=<path-to-pretrained-model-weights> \
#   --s1_tensor=<path-to-S1-tensor> \
#   --era5_tensor=<path-to-ERA5-tensor> \
#   --dem_tensor=<path-to-DEM-tensor> \
#   --device=<device-to-use-for-inference>
#
import matplotlib.pyplot as plt
import argparse
import torch

from modules import *
from config import *


def main(args=None):
    # -------------------------------------------
    # Initialize and load the pretrained model
    # -------------------------------------------
    # Initialize the network
    et_estimator = UNet(d_in=N_CHANNELS, d_out=N_CLASSES)
    # Load pretrained model weights
    params = torch.load(args.pretrained_model)
    # Load weights into the initialized network
    et_estimator.load_state_dict(params["model_state_dict"])
    # Set device to use for inference
    et_estimator = et_estimator.to(args.device)
    # Set model to evaluation mode
    et_estimator.eval()

    # ------------------------------
    # Load input data
    # ------------------------------
    # Load tensors
    s1 = torch.load(args.s1_tensor)  # s1.shape = (1, 6, 128, 128) 
    era5 = torch.load(args.era5_tensor)  # era5.shape = (1, 7, 128, 128)
    dem = torch.load(args.dem_tensor)  # dem.shape = (1, 5, 128, 128)
    # Concatenate tensors along the channel dimension
    x = torch.cat([s1, era5, dem], dim=1).to(args.device)  # x.shape = (1, 18, 128, 128)

    # ------------------------------
    # Inference
    # ------------------------------
    with torch.no_grad():  # Disable gradient calculation
        # Pass through model
        et = et_estimator(x)  # et.shape = (1, 1, 128, 128)
        # Apply ReLU activation function
        et_pred = torch.relu(et)  # et_pred.shape = (1, 1, 128, 128)

    # ------------------------------
    # Save and plot results
    # ------------------------------
    # save the tensor
    torch.save(et_pred, "et_pred.pt")

    plt.imshow(et_pred.detach().cpu().numpy().squeeze(), cmap="coolwarm", vmin=0, vmax=8)
    plt.savefig("et_pred.png", dpi=300, bbox_inches="tight")

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Predict ET using best-performing SAR-based ET estimation model')
    parser.add_argument('--pretrained_model', type=str, help='path to pretrained model weights')
    parser.add_argument('--s1_tensor', type=str, help='path to S1 tensor')
    parser.add_argument('--era5_tensor', type=str, help='path to ERA5 tensor')
    parser.add_argument('--dem_tensor', type=str, help='path to DEM tensor')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'], type=str, help='device to use for inference')
    args = parser.parse_args()

    # Call main function
    main(args)
