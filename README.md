# Crop Disease Classification with Deep Learning
## 🌟 Project Overview
This project focuses on developing and evaluating deep learning models to classify crop diseases using leaf images. Three datasets were utilized:

- Potato Disease Leaf Dataset
- Plant Village Dataset
- Crop Disease Classification Dataset
  
We trained and tested multiple CNN architectures, including ResNet-18, VGG-16, and MobileNet V2, using both standard and transfer learning approaches. The models were evaluated based on metrics like accuracy, precision, recall, F1-score, and confusion matrices. A t-SNE visualization was also performed to understand feature embeddings.

## Requirements
To run the code, the following libraries and tools are required:

- torch==2.0.0
- torchvision==0.15.0
- pandas==1.5.3
- Pillow==9.4.0
- scikit-learn==1.2.0
- tqdm==4.65.0
- matplotlib==3.7.0
- seaborn==0.12.2
- numpy==1.24.0
- gdown==4.7.0
- scikit-image==0.20.3
  
Ensure all required libraries are installed. If you are using **Google Colab (recommended)** to run the code, all the requirements come preinstalled.

## Instructions
Please use **Google Collab** to run .ipynb files in the repository for seamless workflow.
### Dataset
The datasets used in this project can be downloaded from Kaggle:

- Potato Disease Leaf Dataset: https://www.kaggle.com/datasets/rizwan123456789/potato-disease-leaf-datasetpld
- Crop Diseases Classification Dataset: https://www.kaggle.com/datasets/mexwell/crop-diseases-classification
- Plant Village Dataset: https://www.kaggle.com/datasets/adilmubashirchaudhry/plant-village-dataset

The datasets are automatically imported into the code using Kaggle's API. So you don't need to download them manually

**The sample 100 images datasets and trained models are available on Google Drive:**
https://drive.google.com/drive/folders/1zsRzAamIAjCsfowdBV4kkZPplyG3i8vd

### Training the Models
To train the models:

- Open the train file in Google Colab.
- Run all cells in the notebook.
- The script automatically downloads the dataset, trains the models using specified architectures, and saves the trained models to the current working directory.
  
Ensure adequate GPU resources are enabled in the Colab runtime for faster training.

### Testing the Models
**On the sample test dataset (100 images per dataset):**
- Open the test_on_100_images.ipynb file in Google Colab.
- Run all cells in the notebook.
- The script automatically downloads the datasets and trained models from Google Drive so you don't have to manually upload anything.
  
The script evaluates the models on the test dataset and outputs metrics such as accuracy, precision, recall, F1-score, confusion matrices and TSNE visualization.

**On the complete test dataset**

- Open the test.ipynb file in Google Colab.
- Run all cells in the notebook.
- The script automatically downloads the datasets from kaggle and trained models from Google Drive.
  
The script evaluates the models on the test dataset and outputs metrics such as accuracy, precision, recall, F1-score, confusion matrices and TSNE visualization.

## Project Files
- train.ipynb: Contains the code for training the models.
- test.ipynb: Contains the code for evaluating the models using pre-trained weights.
- test_on_100_images.ipynb: Contains the code for evaluating the models on the sample dataset using pre-trained weights.
