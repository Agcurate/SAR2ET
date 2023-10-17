python predict_et.py \
    --pretrained_model ../models/et-estimator.pt \
    --s1_tensor ../data/tensors/s1/s1_0006.pt \
    --era5_tensor ../data/tensors/era5/era5_0006.pt \
    --dem_tensor ../data/tensors/dem/dem_0006.pt \
    --device cuda