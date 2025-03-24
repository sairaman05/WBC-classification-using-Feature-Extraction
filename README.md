# WBC-classification-using-Feature-Extraction

Methodology:
* This approach involves a combination of Deep Learning (VGG16) and Machine Learning (LightGBM) Techniques for White Blood Cell Classification. 
* First convert the image into grayscale and apply Gaussion Blur, perform OTSU's thresholding and morphological operations to reduce noise in the image. 
* Segment the WBC cells using watershed algorithm and extract them. 
* After segmentation, it employs a dual-feature strategy: Deep Learning based features to capture high level features and spatial arrangements, while Handcrafted features like area, perimeter, solidity, Aspect Ratio and Extent are calculated manually in order to increase the robustness in the model. 
* To counter class imbalance, SMOTE oversamples minority classes during the training process.
* A LightGBM Gradient-Boosting Classifier is then trained on this balanced dataset along with deep learning features and handcrafted extracted features. 
* Finally, the model is evaluated using accuracy score and classification report.


Dataset Link:
https://www.kaggle.com/datasets/masoudnickparvar/white-blood-cells-dataset
