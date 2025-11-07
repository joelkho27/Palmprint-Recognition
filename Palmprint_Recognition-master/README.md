# Palmprint Recognition System# Palmprint_Recognition

This project is mainly to complete the palmprint feature extraction and classification tasks. The data set contains 99 people's palm print pictures, in which 3 palm print pictures of each person are distributed in the training set, and the other 3 palm print pictures are distributed in the test set. In this project, I tried the traditional method use SIFT to extract features and KNN for classification which get accuracy of 97.31%, and also tried the convolutional neural network method such as ResNet which get accuracy of 83.16%. In addition, I also tried to use the Gaussian filter, Gabor filter,etc. to process the palmprint image and extract the texture from the palmprint image, but these methods have not improved the accuracy of palmprint recognition.

A biometric authentication system using SIFT (Scale-Invariant Feature Transform) and CNN (Convolutional Neural Networks) for palmprint recognition.

## 参考博客：

## 🔑 Features[【Pytorch】使用ResNet-50迁移学习进行图像分类训练](https://blog.csdn.net/heiheiya/article/details/103028543)



- **SIFT-based Authentication**: Multi-layer validation system with feature matching[【pytorch】数据增强](https://wizardforcel.gitbooks.io/learn-dl-with-pytorch-liaoxingyu/4.7.1.html)

- **CNN Implementation**: Deep learning approach using ResNet-18 with contrastive learning

- **6-Layer Validation**: SIFT + Texture + Geometric + Template + SSIM + Edge detection[opencv python SIFT（尺度不变特征变换）](https://segmentfault.com/a/1190000015709719)

- **Dataset Support**: Compatible with Tongji and PolyU palmprint databases

[OpenCV-Python教程:41.特征匹配](https://www.jianshu.com/p/ed57ee1056ab)

## 📁 Project Structure

[opencv python 特征匹配](https://segmentfault.com/a/1190000015735549)

```

├── SIFT_DIP.py                 # Main SIFT implementation[opencv中 cv2.KeyPoint和cv2.DMatch的理解](https://blog.csdn.net/qq_29023939/article/details/81130987)

├── resnet18_DIP.py             # CNN ResNet-18 model

├── texture_extraction_DIP.py   # Texture feature extraction[K近邻算法](https://www.cnblogs.com/ybjourney/p/4702562.html)

├── tongji_sift_auth.py         # SIFT authentication script
├── tongji_cnn_trainer.py       # CNN training script
├── tongji_dataset_prep.py      # Dataset preparation utilities
├── palm_auth_ultimate_fixed.py # Complete authentication pipeline
├── CNN_Partner/                # CNN partner implementation
├── Palmprint/                  # Palmprint dataset
└── Tongji_Palmprint/           # Tongji dataset
```

## 🚀 Installation

### Prerequisites
- Python 3.8+
- OpenCV
- NumPy
- PyTorch (for CNN)
- scikit-image

### Setup
```bash
# Install dependencies
pip install opencv-python numpy torch torchvision scikit-image pillow matplotlib
```

## 💻 Usage

### SIFT Authentication
```python
python tongji_sift_auth.py
```

### CNN Training
```python
python tongji_cnn_trainer.py
```

### Dataset Preparation
```python
python tongji_dataset_prep.py
```

## 📊 Performance

### SIFT System
- **Accuracy**: 87.23%
- **GAR** (Genuine Accept Rate): 50.67%
- **FAR** (False Accept Rate): 0.45%
- **Processing Time**: ~0.5-1.0 seconds

### CNN System (ResNet-18)
- **Accuracy**: 91.68%
- **GAR**: 79.42%
- **FAR**: 1.21%
- **Processing Time**: ~0.12 seconds (GPU)

## 🎯 Key Components

### SIFT Implementation
- Feature extraction with 1200 max features
- Lowe's ratio test (threshold: 0.75)
- Multi-layer validation for robustness

### CNN Implementation
- ResNet-18 architecture
- 256-D embeddings
- NT-Xent contrastive loss
- Cosine similarity matching (threshold: 0.8)

## 📚 Dataset

Tested on:
- **Tongji Palmprint Database**: 50 individuals, 595 test comparisons
- **PolyU Palmprint Database**: Compatible

## 🤝 Contributing

This project was developed as part of a Digital Image Processing course comparing traditional and deep learning approaches to palmprint recognition.

## 📄 License

This project is for educational purposes.
