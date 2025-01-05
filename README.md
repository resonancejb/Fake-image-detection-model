"""# Fake Face Detection with DenseNet121

## Overview
This project aims to detect fake faces using a fine-tuned DenseNet121 model. The model is trained for binary classification (real vs. fake faces) using deep learning techniques.

## Features
- Utilizes DenseNet121 for feature extraction.
- Fine-tuned model with additional layers for improved performance.
- Achieves **98.7% accuracy** on the dataset.
- Implements data augmentation, dropout, and batch normalization for better generalization.
- **Streamlit-based frontend** for user-friendly interaction.
- Users can **upload an image** or **provide an image URL** to check if the face is real or fake.

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/fake-face-detection.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the Jupyter Notebook:
   ```bash
   jupyter notebook notebooks/fake_face_detection.ipynb
   ```
2. Train the model or use the pre-trained version in `models/`.
3. Evaluate performance on test images.
4. Run the **Streamlit App**:
   ```bash
   streamlit run scripts/app.py
   ```
5. Upload an image or enter an image URL in the web interface to check if it is real or fake.

## Dataset
The dataset consists of real and fake face images. It is recommended to preprocess and balance the dataset before training.

### Sample Dataset Image
![Dataset Sample](images/dataset_sample.jpg)

## Model Details
- **Backbone:** DenseNet121 (pre-trained on ImageNet)
- **Additional Layers:**
  - Global Average Pooling
  - Dropout (Regularization)
  - Dense Layers with ReLU Activation
- **Loss Function:** Binary Crossentropy
- **Optimizer:** Adam

## Results
The model achieves **98.7% accuracy** on the test dataset.

## Example Results
Here are some example results of real and fake face detection:

### Fake Face Example:
![Fake Face](images/fake_face_example.jpg)

### Real Face Example:
![Real Face](images/real_face_example.jpg)

### Model Performance Metrics:
1. **Accuracy Curve**
   ![Accuracy Curve](images/accuracy_curve.jpg)
2. **Loss Curve**
   ![Loss Curve](images/loss_curve.jpg)
3. **Confusion Matrix**
   ![Confusion Matrix](images/confusion_matrix.jpg)

(To upload images, place them inside an `images/` folder and use the correct path in the Markdown syntax above.)

## Future Improvements
- Use ensemble learning for better generalization.
- Improve dataset quality and diversity.
- Implement adversarial training to detect more sophisticated fake images.
- Enhance the **Streamlit UI** for a better user experience.

## License
This project is open-source under the MIT License.
"""
