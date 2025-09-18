<h1>Project Goal</h1>
This is a binary image classification problem where you will identify metastatic cancer in small image patches taken from larger digital pathology scans.

<h1>Basic Data information</h1>

* size = 7.76Gb
* 32x32px region of a patch contains at least one pixel of tumor tissue
* **Dataset:** There are 220,025 images with labels and without missing values or duplicates. This is a cleaned well-label dataset. The number matches between the .csv file and the numbers of images in the train/test folder."""

<h1>Exploratory Data Analysis (EDA)</h1>

* **Class distribution:** There is a class imbalance with 130908 of non-cancerous(0) vs 89117 of cancerous(1).
* **Sample image Visualization:** Display of sample images for both non-cancerous and cancerous classes helps visualization of the data."""

<h1>Data Cleaning & Structure Preparation</h1>

* The data cleaning procedure involved checking for duplicates (none found).
* The main cleaning step is handling the class imbalance by reducing the number of samples in each class to 50000 before training to create a balanced dataset.
* Image data generators were used with a preprocessing function to extract the center 32x32 region of the images."""

<h1>Model Architecture - Baseline CNN</h1>

* **Model Structure:** A sequential Convolutional Neural Network (CNN) with three convolutional blocks (32, 64, 128 filters), ReLU activation, and max pooling. Followed by flatten, dense (128 units), ReLU, dropout (0.2), and a final dense layer with sigmoid activation.
* **Optimizer:** Adam with learning rate 1e-3
* **Loss Function:** binary_crossentropy
* **Metrics:** accuracy, AUC"""

<h1>Model Architecture - BatchNorm CNN</h1>

* **Model Structure:** Similar to the baseline but with Batch Normalization layers added after each convolutional and dense layer. Uses more filters in convolutional layers (64, 128, 256) and a dense layer with 256 units. Includes dropout (0.5).
* **Optimizer:** Adam with learning rate 1e-3
* **Loss Function:** binary_crossentropy
* **Metrics:** accuracy, AUC
* **Callbacks:** EarlyStopping, ReduceLROnPlateau, ModelCheckpoint"""

<h1>Training</h1>

* **Balanced Dataset Size:** 100,000 images (50,000 per class).
* **Validation Split:** 20% of the balanced dataset used for validation.
* **Batch Size:** 64 (Training), 128 (Submission).
* **Epochs:** Trained for up to 10 epochs with Early Stopping.
* **Preprocessing:** Center 32x32 pixel region extracted from the original images."""

<h1>Results and Analysis</h1>

| Model        | Train Accuracy | Train AUC | Validation Accuracy | Validation AUC |
|--------------|----------------|-----------|---------------------|----------------|
| Baseline     | 0.8826         | 0.9535    | 0.8490              | 0.9271         |
| BatchNorm    | 0.9469         | 0.9880    | 0.8739              | 0.9537         |

* **Confusion Matrix:** Visualizations show the performance of both models on the validation set.
* **ROC Curve:** Visualizations show the ROC curves and AUC scores for both models on the validation set.
* **Summary:** The Batch Normalization model achieved higher training and validation accuracy and AUC compared to the baseline model."""

<h1>Conclusion</h1>

Batch normalization helps stabilize the learning process and can lead to faster convergence. The Batch Normalization model performed better on the validation set, demonstrating the effectiveness of batch normalization and increased filter size for this task. Future work could involve more extensive hyperparameter tuning and exploring other model architectures."""
