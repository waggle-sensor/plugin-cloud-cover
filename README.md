## Science

Clouds have been widely studied in a variety of fields. The shape and distribution of clouds are not only important in modeling climate and weather, but also toward understanding the interactions between aerosols and clouds in weather research, and also to develop environment forecasting models including solar radiation and cloud properties. Additionally, detecting and understanding cloud cover has been fundamental to estimating and forecasting solar irradiance and photovoltaic energy generation. Across a diverse set of studies answering the percentage of cloud coverage, along with wind direction, speed, temperature, and other meteorological factors [1] is very valuable.

Many of the cloud images used in the studies are satellite images. Satellites based images can be insufficient in their resolution when studying large areas. Additionally in the case of local cloud environment focused on smaller regions with rapid changes, using the images from satellites is unrealistic, expensive, and latency prone. Instead, ground-based images have been used predominantly for analyzing local cloud conditions, and toward estimation and prediction of solar irradiance.

As a method to estimate cloud cover, we used a machine learning model called U-net. To train each of the machine learning models with various sky conditions, we created a cloud dataset that can reinforce dark sky conditions and overcast on Singapore whole Sky imaging segmentation (SWIMSEG) dataset using a Waggle node deployed in Argonne National Laboratory.

## AI@Edge:

This application is intended for daylight hours. The daylight hours for a deployment is defined by the users, for example 7AM - 5PM. The application first collects data from the sky facing camera and then passes through the U-Net [2] model to segment cloud pixels. The output of U-Net is the cloud/clear-sky probability for each pixel. With the probability calculated, the application segments the clouds based on the threshold set by the user.

