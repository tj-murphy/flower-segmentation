Flower Segmentation using Convolutional Neural Networks
--------------------------------------------------------

This repository contains the code and documentation for the Flower Segmentation project, which focuses on accurately identifying and isolating flowers in complex images using deep learning techniques.
The project was conducted from January 2024 to May 2024.

Project Overview:
------------------
The project developed and compared two Convolutional Neural Network (CNN) architectures for the task of flower segmentation:
1. A fine-tuned U-Net model using transfer learning.
2. A custom-designed CNN model.

The custom model outperformed the U-Net in terms of overall accuracy and mean Intersection over Union (IoU).

Repository Contents:
---------------------
- "report.pdf" : Comprehensive scientific report detailing the literature review, methodology, results, and potential applications.
- "segmentationExist.m" : MATLAB code for the segmentation using the pre-trained U-Net model.
- "segmentationOwn.m" : MATLAB code for the custom-designed CNN model.
- "segmentationexist.mat" : Pre-trained U-Net CNN weights.

Results:
--------------------
Custom CNN model:
- Overall accuracy: 91.1%
- Mean IoU: 0.87

Pre-trained U-Net model:
- Overall accuracy: 88.1%
- Mean IoU: 0.84

The custom CNN model demonstrated superior performance, especially in accurately outlining flower boundaries.

Potential Applications:
------------------------
- Automated plant species identification
- Precision agriculture
- Ecological research

Grades:
-------
Coursework grade (including code, results, report): 76.5%

Presentation grade: 70%

Note:
-----------------------
The custom CNN weights file was too lage to upload. For running the custom model, you might need to train it from scratch or use a smaller pre-trained model.

Acknowledgements:
-----------------
This project was completed as part of the COMP4106 module at the University of Nottingham by Timothy Murphy. Special thanks to the professors for their support and guidance.

License:
---------
This project is licensed under the MIT License - see LICENSE file for details.
