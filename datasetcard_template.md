---
# Dataset Card
---

Dataset Card for Sign Language Letters Detection
This dataset contains images of hand signs for the letters of the alphabet in American Sign Language (ASL) and Filipino Sign Language (FSL). The goal is to train a computer vision model that can detect and recognize the sign language letters from the images.

Dataset Details
Dataset Description
This dataset is a combination of three datasets from Roboflow Universe:

- [American Sign Language Letters](https://public.roboflow.com/object-detection/american-sign-language-letters/1): This dataset contains 1728 images of 26 letters in various backgrounds and lighting conditions.
- [ASL and FSL Combo](https://universe.roboflow.com/hand-signs-9v6jr/asl-and-fsl-combo): This dataset contains 2128 images of 26 letters in American Sign Language and Filipino Sign Language, with different hand shapes and skin tones.
- [Sign Language](https://universe.roboflow.com/tfod-p4luj/sign_language-acf74): This dataset contains 5121 images of 26 letters in American Sign Language, with different hand shapes and skin tones.

The images are in JPG format and the annotations are in JSON format.

Curated by: Roboflow users
License: Public Domain
Repository: [Roboflow Universe](https://universe.roboflow.com)
Uses
Direct Use
This dataset can be used to train or evaluate a computer vision model that can detect and recognize sign language letters from images. Such a model can be useful for applications such as sign language translation, education, and accessibility.

Dataset Structure
The dataset consists of 6879 images and a JSON file with the annotations. The images are divided into train, test, and valid splits. The JSON file contains the following fields:

- info: a dictionary with information about the dataset, such as version, description, and URL.
- licenses: a list of dictionaries with information about the licenses of the images, such as name, URL, and id.
- images: a list of dictionaries with information about the images, such as file_name, height, width, id, and license.
- annotations: a list of dictionaries with information about the annotations, such as image_id, category_id, bbox, area, id, and segmentation.
- categories: a list of dictionaries with information about the categories, such as name, id, and supercategory.

Dataset Creation
Source Data
Data Collection and Processing
The data was collected and processed by the Roboflow users who created and shared the datasets. The data sources are:


- [American Sign Language Letters Dataset](https://public.roboflow.com/object-detection/american-sign-language-letters/1): The images were collected from various sources, such as YouTube videos, Google Images, and personal photos. The images were resized, cropped, and augmented using Roboflow. The annotations were created using Roboflow Annotate.
- [ASL and FSL Combo Object Detection Dataset](https://universe.roboflow.com/hand-signs-9v6jr/asl-and-fsl-combo): The images were collected from various sources, such as YouTube videos, Google Images, and personal photos. The images were resized, cropped, and augmented using Roboflow. The annotations were created using Roboflow Annotate.
- [Sign Language Object Detection Dataset](https://universe.roboflow.com/tfod-p4luj/sign_language-acf74): The images were collected from various sources, such as YouTube videos, Google Images, and personal photos. The images were resized, cropped, and augmented using Roboflow. The annotations were created using Roboflow Annotate.

Features and the target
The features are the images of hand signs for the letters of the alphabet in ASL and FSL. The target is the bounding box and the label of the sign language letter in each image.

Annotation process
The annotations were created using Roboflow Annotate, a web-based tool that allows users to draw bounding boxes and assign labels to the objects in the images.

Who are the annotators?
The annotators are the Roboflow users who created and shared the datasets.

Bias, Risks, and Limitations
Some possible biases, risks, and limitations of this dataset are:

- The dataset may not be representative of the diversity of sign language users, such as age, gender, ethnicity, and region.
- The dataset may not cover all the variations and nuances of sign language, such as hand shape, orientation, movement, and facial expression.
- The dataset may contain errors or inconsistencies in the annotations, such as missing, overlapping, or incorrect bounding boxes or labels.
- The dataset may not be suitable for some applications or domains, such as real-time or low-resource settings, due to the size, quality, or format of the images or annotations.


Source: 
(1) American Sign Language Letters Object Detection Dataset - Roboflow. https://public.roboflow.com/object-detection/american-sign-language-letters.
(2) ASL And FSL combo Object Detection Dataset by Hand signs. https://universe.roboflow.com/hand-signs-9v6jr/asl-and-fsl-combo.
(3) Sign_language Computer Vision Project - universe.roboflow.com. https://universe.roboflow.com/tfod-p4luj/sign_language-acf74.
