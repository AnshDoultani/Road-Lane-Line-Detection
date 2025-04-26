
# Road Lane Line Detector

This project uses **Deep Learning** and a **U-Net architecture** to detect road lane lines from images.  
It involves **data preparation**, **image segmentation**, and **model training** using **TensorFlow** and **Keras**.

---

## Project Structure

- **Road_Lane_Line.ipynb** – The Jupyter Notebook containing all the code: data loading, preprocessing, model building, training, and evaluation.

---

## How It Works

1. **Import Libraries**  
   Load necessary libraries like TensorFlow, Keras, NumPy, OpenCV, and others.

2. **Load Training Data**  
   Use Keras' `ImageDataGenerator` to load and prepare images from the specified directory.

3. **Preprocess the Data**  
   - Separate ground frames (input images) and lane masks (labels).
   - Convert lane masks into binary images (lane = 1, background = 0).
   - Shuffle and split the dataset into training and validation sets.

4. **Build the Model**  
   - Implement a U-Net model designed for segmentation tasks.
   - Use multiple convolution, pooling, and upsampling layers.

5. **Compile and Train**  
   - Compile the model using **Adam optimizer** and **Binary Focal Loss**.
   - Train for 6 epochs and save checkpoints.

6. **Evaluate Performance**  
   - Plot training/validation loss and accuracy to visualize model learning.

---

## Model Summary

- **Model:** U-Net
- **Input Size:** (256, 256, 3)
- **Output:** Binary segmentation mask (road lanes)

---

## Requirements

Make sure you have the following installed:

```bash
python>=3.7
tensorflow>=2.8
keras
opencv-python
numpy
matplotlib
scikit-learn
pandas
```

Install the required libraries with:

```bash
pip install tensorflow keras opencv-python numpy matplotlib scikit-learn pandas
```

---

## Sample Outputs

During training, you will visualize:
- The input road images,
- Their corresponding lane masks,
- Training and validation loss curves,
- Training and validation accuracy curves.

---

## Notes

- The dataset is expected to have images categorized into two folders:
  - **Frames** (normal road images)
  - **Masks** (binary images showing lane markings)

- You may need to adjust the `trained_path` in the notebook according to your file structure.
