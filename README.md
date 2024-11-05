# Image Generation from Text Using COCO 2017 Dataset
## This project generates images based on input text prompts using deep learning models. It leverages the COCO 2017 dataset for training and evaluation. The model is built with an encoder-decoder attention mechanism, utilizing a Vision Transformer (ViT) as the backbone.

# Table of Contents
Project Overview
Features
Requirements
Dataset
Setup Instructions
Training and Evaluation
Usage
Results
Contributors
License


# Project Overview
This project explores image generation from textual descriptions using transformer-based models. By training on the COCO 2017 dataset, the model learns to generate images that align with descriptive text inputs.

# Features
Generates photo-realistic images from text descriptions.
Utilizes Vision Transformer (ViT) as the core model.
Supports training and testing with the COCO 2017 dataset.
Sample generated images and captions are available in captions_sample.csv for reference.



# Requirements
Ensure you have the following libraries installed:

Python 3.11 \n
TensorFlow
PyTorch
Transformers
Pandas
Numpy
Matplotlib
Pillow
Tqdm
Requests
Scikit-Learn
Gradio

To install the dependencies, run:

pip install -r dev.txt

# Dataset
The project uses the COCO 2017 dataset, which includes:

Training images (train2017), validation images (val2017), and test images (test2017)
Annotations (annotations_trainval2017.zip)

# Downloading COCO 2017
To download the dataset directly in Google Colab:

!mkdir -p /content/coco2017
!wget -P /content/coco2017 http://images.cocodataset.org/zips/train2017.zip
!wget -P /content/coco2017 http://images.cocodataset.org/zips/val2017.zip
!wget -P /content/coco2017 http://images.cocodataset.org/zips/test2017.zip
!wget -P /content/coco2017 http://images.cocodataset.org/annotations/annotations_trainval2017.zip

#Unzipping
!unzip /content/coco2017/train2017.zip -d /content/coco2017/
!unzip /content/coco2017/val2017.zip -d /content/coco2017/
!unzip /content/coco2017/test2017.zip -d /content/coco2017/
!unzip /content/coco2017/annotations_trainval2017.zip -d /content/coco2017/

#Setup Instructions
1)Clone the Repository: git clone https://github.com/Devaunsh7/text-to-image.git

2)Set Up Environment:

If using Google Colab, upload the dataset or use the download commands above.
Set BASE_PATH in your code to point to the dataset location: BASE_PATH = '/content/coco2017'

3)Preparing Captions: Load captions and preprocess the dataset:
import json, pandas as pd

with open(f'{BASE_PATH}/annotations/captions_train2017.json', 'r') as f:
    data = json.load(f)['annotations']

img_cap_pairs = [[f"{BASE_PATH}/train2017/{'%012d.jpg' % sample['image_id']}", sample['caption']] for sample in data]
captions = pd.DataFrame(img_cap_pairs, columns=['image', 'caption'])
captions.to_csv('captions_sample.csv', index=False)

# Training and Evaluation
1)Training: Run the training script or notebook to start training. You may configure epoch, batch_size, and learning_rate in the code as needed.

2)Evaluation: Use the captions_sample.csv file for testing the model on a sample dataset.



# Usage
To generate an image from a text prompt:

Ensure the model weights are loaded (either from training or pretrained weights).
Input a descriptive prompt and observe the generated output image.

prompt = "A dog playing with a ball in a grassy field"
generated_image = model.generate_image(prompt)
plt.imshow(generated_image)


# Results
After training, sample images generated from text prompts are saved in the results folder.

# Contributors
Devaunsh V Vastrad - Devaunsh7







