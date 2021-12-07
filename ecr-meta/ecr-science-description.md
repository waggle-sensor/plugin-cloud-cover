## Science
Clouds have been widely studied in a variety of fields. The shape and distribution of clouds are not only important to modeling climate and weather, but also to understand interactions between aerosol and cloud for weather research, and to develop environment forecasting models including radiation and cloud properties. Additionally, detecting and understanding cloud cover over the sky have been studied to estimate and forecast solar irradiance and performance of renewable solar photovoltaic energy generation. For this reason, examining solar irradiance in photovoltaic power grids has been investigated in many ways. Even though the purpose of each study is diverse, it is common that they have approached to analyze the magnitude of cloud coverage. In this context, answering how much cloud covers the sky is a striking problem along with other factors such as wind direction, speed, temperature, and other meteorological factors [1].

Many of the cloud images used in the studies have been collected from satellites. Those image data are then used to analyze cloud types and solar irradiance. The images captured from satellites are advantageous to study large areas with regard to the resolution of the image to understand the overall environment of the area. However, to analyze the local cloud environment, which is a much smaller area and changing rapidly, using the images from the satellite are unrealistic, costly, and slow. Instead, ground-based images have been used predominantly for analyzing local cloud status which can affect the estimation and prediction of solar irradiance. Because local weather conditions and solar irradiance are affected significantly by the cloud coverage of local sky, the ground-based images are more suitable to represent local sky conditions.

As a method to estimate cloud cover, we used a machine learning model called U-net. To train each of the machine learning models with various sky conditions, we created a cloud dataset that can reinforce dark sky conditions and overcast on Singapore whole Sky imaging segmentation (SWIMSEG) dataset using a Waggle node deployed in Argonne National Laboratory.

## AI@Edge:
The application is up and running in the daytime. The definition of daytime needs to come from users. At this moment, the application is running from 7 am - 5 pm. The application first collects data from the sky facing camera based on periodic sampling. The image is then passed through the U-Net [2] model to segment cloud pixels. The actual calculation that the U-Net is calculating is the possibility of how much each pixel should be classified as cloud or sky. With the probability calculated, the application segments the cloud based on the threshold that the user determines.

## Using the code
Output: cloud cover ratio (0-1)<br />
Input: single image (1/30 second required)<br />
Image resolution: 300x300<br />
Inference time:<br />
Model loading time:<br />

## Arguments
   '-debug': Debug flag<br />
   '-stream': ID or name of a stream, e.g. top-camera<br />
   '-interval': Inference interval in seconds (default = 0, as soon as the plugin can get image)<br />
   '-sampling-interval': Sampling interval between inferencing (default = -1, no sampling)<br />
   '-threshold': Cloud pixel determination threshold (0-1) (default = 0.9)<br />


### Reference
[1] Seongha Park, Yongho Kim, Nicola J. Ferrier, Scott M. Collis, Rajesh Sankaran and Pete H. Beckman “Prediction of Solar Irradiance and Photovoltaic Solar Energy Product Based on Cloud Coverage Estimation Using Machine Learning Methods”, 2020, Atmosphere, Volume 12, Issue 3, pages 395.<br />
[2] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image segmentation." In International Conference on Medical image computing and computer-assisted intervention, Springer, Cham, pp. 234-241, 2015.
