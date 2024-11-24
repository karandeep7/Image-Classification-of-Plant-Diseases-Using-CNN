# Crop Disease Classification with Deep Learning
## 🌟 Project Overview
This project focuses on developing and evaluating deep learning models to classify crop diseases using leaf images. Three datasets were utilized:

- Potato Disease Leaf Dataset
- Plant Village Dataset
- Crop Disease Classification Dataset
  
We trained and tested multiple CNN architectures, including ResNet-18, VGG-16, and MobileNet V2, using both standard and transfer learning approaches. The models were evaluated based on metrics like accuracy, precision, recall, F1-score, and confusion matrices. A t-SNE visualization was also performed to understand feature embeddings.

## Requirements
To run the code, the following libraries and tools are required:

- Python 3.x
- PyTorch: For building, training, and testing deep learning models.
- torchvision: For accessing pre-trained models and image transformations.
- pandas: For handling tabular data.
- scikit-learn: For splitting datasets and computing metrics.
- matplotlib and seaborn: For visualizations, including confusion matrices.
- tqdm: For displaying progress bars.
- PIL: For image processing.
  
Ensure all required libraries are installed

## Instructions
### Dataset
The datasets used in this project can be downloaded from Kaggle:

- Potato Disease Leaf Dataset: https://www.kaggle.com/datasets/rizwan123456789/potato-disease-leaf-datasetpld
- Crop Diseases Classification Dataset: https://www.kaggle.com/datasets/mexwell/crop-diseases-classification
- Plant Village Dataset: https://www.kaggle.com/datasets/adilmubashirchaudhry/plant-village-dataset

The datasets are automatically imported into the code using Kaggle's API. So you don't need to download them manually

### Training the Models
To train the models:

- Open the train file in Google Colab.
- Run all cells in the notebook.
- The script automatically downloads the dataset, trains the models using specified architectures, and saves the trained models to the current working directory.
  
Ensure adequate GPU resources are enabled in the Colab runtime for faster training.

### Testing the Models
To test the trained models:

- Open the test file in Google Colab.
- Upload the trained model files to the same directory as the test file.
- If using Google Colab, upload the models to the current working directory in Colab.
- Run all cells in the notebook.
  
The script evaluates the models on the test dataset and outputs metrics such as accuracy, precision, recall, F1-score, confusion matrices and TSNE visualization.

## Project Files
- train.ipynb: Contains the code for training the models.
- test.ipynb: Contains the code for evaluating the models using pre-trained weights.
- Pre-trained models: Ensure these are available in the same directory as the test.ipynb file before testing.
