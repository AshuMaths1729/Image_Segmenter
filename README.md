# Skin Lesion Segmentation

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)

Skin Lesion Segmentation using both Deep Learning and normal Image Processing approaches.

Used the PH2 Dataset for the task.
___

## The DL Approach
Used FCNet to train the model on the images and their corresponding masks.

The data required was generated by a custom script: [Copy_file.py](https://github.com/AshuMaths1729/Image_Segmenter/blob/master/Copy_file.py)
![alt_text](https://github.com/AshuMaths1729/Image_Segmenter/blob/master/DL_result.png "DL Approach Results")


## The Image Processing Approach
Used Median filtering, Image thresolding and Canny edge detection to generate contours.
![alt_text](https://github.com/AshuMaths1729/Image_Segmenter/blob/master/ML_result.png "IP Approach Results")


___

References:
* https://towardsdatascience.com/a-keras-pipeline-for-image-segmentation-part-1-6515a421157d

* https://www.tensorflow.org/tutorials/images/segmentation
