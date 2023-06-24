# Advancing Wound Filling Extraction on 3D Faces: A Auto-Segmentation and Wound Face Regeneration Approach

## Prequisites
* python 3.7.4
* pytorch 1.4.0
* numpy 1.19.0
* plyfile 0.7.1

## Introduction
Facial wound segmentation plays a crucial role in preoperative planning and optimizing patient outcomes in various medical applications. In this paper, we propose an efficient approach for automating 3D facial wound segmentation using a two-stream graph convolutional network. Our method leverages the Cir3D-FaIR dataset and addresses the challenge of data imbalance through extensive experimentation with different loss functions. To achieve accurate segmentation, we conducted thorough experiments and selected a high-performing model from the trained models. The chosen model exhibits superior performance in dealing with the complexities of 3D facial wounds. Furthermore, we compared the results of our proposed method with a previously studied approach, both aiming to extract 3D facial wound fills. Our method achieved a remarkable accuracy of 0.9999986\% on the test suite, surpassing the performance of the previous method. From this result, we use 3D printing technology to illustrate the shape of the wound filling. The outcomes of this study have significant implications for physicians involved in preoperative planning and intervention design. By automating facial wound segmentation and improving the accuracy of wound filling extraction, our approach can assist in the careful assessment and optimization of interventions, leading to enhanced patient outcomes.


## Usage
To train the model, please put the trainning data and testing data into data/train and data/test, respectively. Then, you can start to train a model by following command.

```shell
python train.py
```
To extract the 3D facial wound fill, please use the notebook
```shell
get_filling_extraction.ipynb
```