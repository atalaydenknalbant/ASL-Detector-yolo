# Sign Language Translator
![](https://github.com/iremgulcin/tahir-atalay-rumeysa-yusuf-mehmet/blob/main/readme_images/asl_demo.gif)
# Overview

<img src="https://github.com/iremgulcin/tahir-atalay-rumeysa-yusuf-mehmet/blob/main/readme_images/asl.png" width="400" />

The aim of this project was to develop an application that can translate American Sign Language (ASL) into written language, using object detection techniques and multiple data sources.​
# Defining The Problem ​And The Solution
* We want to improve communication and inclusion between people who use ASL and those who do not.​
* Our project uses computer vision and machine learning to detect and translate ASL signs into words.​
* Our project is an example of how technology can help overcome linguistic and cultural differences.
# SDGS
<p float="left">
  <img src="https://github.com/iremgulcin/tahir-atalay-rumeysa-yusuf-mehmet/blob/main/readme_images/sdg4.png" width="200" />
  <img src="https://github.com/iremgulcin/tahir-atalay-rumeysa-yusuf-mehmet/blob/main/readme_images/sdg10.png" width="200" /> 
</p>

1. SDG 4.5 mission is: "Achieve gender equality and inclusive education for all by 2030."​
* Our app helps to achieve SDG 4.5 by enabling deaf and hard-of-hearing students to access quality education and communicate with their peers and teachers.
2. SDG 10.3 mission is: "Promote equal opportunity and fight discrimination through laws and policies."​
- Our app helps to achieve SDG 10.3 by empowering deaf and hard-of-hearing people to overcome communication barriers and access equal opportunities in society.
# Our Framework
![](https://github.com/iremgulcin/tahir-atalay-rumeysa-yusuf-mehmet/blob/main/readme_images/framework.png)
# DATASET
![](https://github.com/iremgulcin/tahir-atalay-rumeysa-yusuf-mehmet/blob/main/readme_images/Dataset.png)
* Available on Roboflow Universe:​ 6879 total image. 6455 train – 276 val – 148 test.​
* 26 Class: A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z ​
* 3 Different Datasets​
* Images dimension: ​
(384, 384), (416,416) (640,480)
# Model
Our model is the Yolov9-small model, which was pretrained on the COCO dataset for object detection.
![](https://github.com/iremgulcin/tahir-atalay-rumeysa-yusuf-mehmet/blob/main/readme_images/model_parameters.png)
# Results
![](https://github.com/iremgulcin/tahir-atalay-rumeysa-yusuf-mehmet/blob/main/readme_images/confusion_matrix.png)![](https://github.com/iremgulcin/tahir-atalay-rumeysa-yusuf-mehmet/blob/main/readme_images/train_val_results.png)
We used various metrics and methods to evaluate our model’s performance and accuracy. On the left We used a confusion matrix, which is a table that shows how well our model predicted the correct classes for each image. On the right We also used graphs to visualize the training and validation loss and accuracy, as well as the precision and recall metrics, over epochs. These are some of the results we achieved. We will look closer to metrics in the next slide.
# Metrics
We used three metrics: box precision, recall, mAP50. These metrics are commonly used for object detection tasks, such as detecting and translating ASL signs. We calculated these metrics for each class, corresponding to each letter of the alphabet, and also for the overall model.
We achieved high values for all these metrics, as shown by the table on the left. We are very satisfied with our model’s results, and we believe that our model can translate ASL signs into words effectively and reliably.
![](https://github.com/iremgulcin/tahir-atalay-rumeysa-yusuf-mehmet/blob/main/readme_images/metrics.png)
# Test Results
## Labels
<p float="left">
  <img src="https://github.com/iremgulcin/tahir-atalay-rumeysa-yusuf-mehmet/blob/main/readme_images/labels1.jpg"  width="500"/>
  <img src="https://github.com/iremgulcin/tahir-atalay-rumeysa-yusuf-mehmet/blob/main/readme_images/labels2.jpg"  width="500" /> 
</p>

## Predicts
<p float="left">
  <img src="https://github.com/iremgulcin/tahir-atalay-rumeysa-yusuf-mehmet/blob/main/readme_images/predicts1.jpg" width="500" />
  <img src="https://github.com/iremgulcin/tahir-atalay-rumeysa-yusuf-mehmet/blob/main/readme_images/predicts2.jpg" width="500" /> 
</p>

<h3><b>Here are the results of labels we predicted by using our model. As you can see our model didn't miss any letter at given dataset.</b></h3>

# Demo
![](https://github.com/iremgulcin/tahir-atalay-rumeysa-yusuf-mehmet/blob/main/readme_images/demo_7sec.gif)

# Summary
## Our project has the following aspects:
![](https://github.com/iremgulcin/tahir-atalay-rumeysa-yusuf-mehmet/blob/main/readme_images/summary.png)

# Citation 
- Jocher, G., Chaurasia, A., & Qiu, J. (2023). Ultralytics YOLO [Computer software]. https://github.com/ultralytics/ultralytics
