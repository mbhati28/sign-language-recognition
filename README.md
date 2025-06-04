# Sign Language Recognition System 🤟

## Overview

This project presents a Sign Language Recognition System that utilizes deep learning models to identify American Sign Language (ASL) hand gestures from static grayscale images. The objective is to bridge the communication gap for the Deaf and Hard of Hearing (DHH) community by providing a robust and accessible recognition solution.

## Problem Statement

Despite being an effective form of communication, sign language is often not understood by the general public. This project aims to reduce that barrier by developing a model that can recognize and classify ASL signs, allowing for real-time translation and improved communication.

## Dataset

**Sign Language MNIST**  
- **Size:** 34,627 grayscale images  
- **Split:** 27,455 training / 7,172 testing  
- **Image Dimensions:** 28x28 pixels  
- **Classes:** 24 ASL letters (A–Z excluding J and Z)  
- **Format:** CSV (784 pixels + label per row)

## Preprocessing Steps

- **Label Mapping**: Mapped numerical labels to ASL letters (excluding J, Z)
- **One-Hot Encoding**: Transformed labels into binary class vectors
- **Reshaping**: Converted images to 28×28×1 format
- **Normalization**: Scaled pixel values from [0–255] to [0–1]
- **Data Augmentation**: Applied rotation and horizontal flipping
- **Shuffling**: Ensured random distribution for unbiased learning

## Models Used

### 1. **Custom CNN**
- Two convolutional layers (64 and 128 filters)
- ReLU activations
- Dropout layers to reduce overfitting
- Softmax output for multi-class classification

### 2. **CNN with Data Augmentation**
- Same architecture as above, trained on augmented dataset
- Improved generalization and test accuracy

### 3. **MobileNetV2 (Transfer Learning)**
- Lightweight, efficient for mobile use
- Faced accuracy limitations due to input size constraint (28x28x1 vs required 32x32x3)

### 4. **EfficientNet B0 (Transfer Learning)**
- Used compound scaling
- Superior performance with lower parameters, but similarly constrained by image size

## Hyperparameter Tuning

- Performed **manual grid search** to optimize:
  - Dense Units: 64, 128
  - Optimizers: Adam, SGD
  - Learning Rates: 0.001, 0.01
  - Momentum: 0, 0.8, 0.9
  - Filter Sizes: (32, 64), (64, 128)

- **Best Configuration:**
  - Dense Units: 128
  - Optimizer: SGD (lr=0.01, momentum=0.9)
  - Filters: 64 → 128
  - Early Stopping with patience=3

## Results

| Model                     | Test Accuracy | Notes                            |
|--------------------------|---------------|----------------------------------|
| CNN (no augmentation)    | 91.79%        | Slight overfitting               |
| CNN (with augmentation)  | **91.93%**    | Best performance, no overfitting |
| MobileNetV2              | Lower         | Image size mismatch              |
| EfficientNet B0          | Lower         | Image size mismatch              |

## Conclusion

The **Custom CNN with Data Augmentation** performed best with a test accuracy of **91.93%**, providing robust and generalizable results. Transfer learning approaches like MobileNetV2 and EfficientNet B0 showed limited success due to image size constraints but remain promising for future improvements with larger and more complex datasets.

## References

1. [Sign Language Recognition - Towards Data Science](https://towardsdatascience.com/sign-language-recognition-with-advanced-computer-vision-7b74f20f3442)
2. [MobileNetV2 - Inverted Residuals and Linear Bottlenecks](https://towardsdatascience.com/mobilenetv2-inverted-residuals-and-linear-bottlenecks-8a4362f4ffd5)
3. [EfficientNet Overview - Scaler](https://www.scaler.com/topics/deep-learning/efficientNet/)
4. [EfficientNet Paper](https://arxiv.org/abs/1801.04381)
5. [Deep Learning CNN - UWO Course](https://owl.uwo.ca/access/lessonbuilder/item/187740185/group/2c0af2b2-25c0-4dce-ab30-793b2da68062/Course%20Content/ECE%209063_9603_6_Deep%20learning_CNN.pdf)

## License

This project was developed as part of the academic coursework for **ECE9063: Data Analytics Foundations** at the University of Western Ontario.  
All code and documentation are for educational use only.
