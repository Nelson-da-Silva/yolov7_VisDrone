# Object Detection from Drones using a Fisheye Lens Camera

![plot](image/Overview-Image.png)

GitHub repository associated with the dissertation of Nelson da Silva's final year Master's project completed at the Imperial College London for the fulfilment of the requirements for the degree of Electronic and Information Engineering.

An associated Google Colab Notebook page which has some examples on how to use the scripts and also shows all of the code used to acquire the results is available here: https://colab.research.google.com/drive/1DAbd8m9Yv9iSG_Uhu1d6sAfCj6Q8aLW1?usp=sharing

## Data Augmentation
The fisheye projection model chosen for this project can be applied to existing datasets using the scripts found in distortion_scripts/. The function to apply the considered radial distortion model is included in apply-fisheye.py for user interest but is not used.

## Datasets
To access the datasets used in this project, simply run the scripts found in the get_datasets/ directory.

## Models
All models are trained using standard YOLOv7 (as opposed to YOLOv7-tiny) and an input resolution of 1280.

### Single distortion level models
Trained on f=150: 
[`fishdrone1280_150.pt`](https://github.com/Nelson-da-Silva/yolov7_VisDrone/releases/download/Models/fishdrone1280_150.pt)

Trained on f=300:
[`fishdrone1280_300.pt`](https://github.com/Nelson-da-Silva/yolov7_VisDrone/releases/download/Models/fishdrone1280_300.pt)

Trained on f=600:
[`fishdrone1280_600.pt`](https://github.com/Nelson-da-Silva/yolov7_VisDrone/releases/download/Models/fishdrone1280_600.pt)

Trained on f=830:
[`fishdrone1280_830.pt`](https://github.com/Nelson-da-Silva/yolov7_VisDrone/releases/download/Models/fishdrone1280_830.pt)

### Large range of distortion levels model
Trained on f=150, f=300, f=600 and f=830: [`fishdrone1280_all.pt`](https://github.com/Nelson-da-Silva/yolov7_VisDrone/releases/download/Models/fishdrone1280_all.pt)

### Upscaled + Small range of distortion levels model
Trained using the upscaled distortion model at f=300, f=600 and f=830: [`upscaled_fishdrone1280_all.pt`](https://github.com/Nelson-da-Silva/yolov7_VisDrone/releases/download/Models/upscaled_fishdrone1280_all.pt)

## Regional evaluation methods
The scripts found in eval_scripts/ provide two methods of regional evaluation: heatmaps and radial splitting. More details on these are provided in the report. Note that to use these scripts, the prediction files output from testing must include the confidences associated with each prediction.