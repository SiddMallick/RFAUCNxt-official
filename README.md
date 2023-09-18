# Response Fusion Attention U-ConvNext 🚀
This repository includes the official project of RFAUCNxt, presented in our paper: Response Fusion Attention U-ConvNext for accurate Segmentation of Optic Disc and Optic Cup (https://doi.org/10.1016/j.neucom.2023.126798) 


## Usage 💥

Train RFAUCNxt using a medical image dataset containing binary labels. For multi-class segmentation, the training script "train.py" needs to be modified.

Arguments to be parsed from the terminal:

1. dir : Directory of the dataset. Inside the dataset directory, the data should be organized in the following manner:
   a. '/train' : should contain training images.
   b. '/train_masks' : should contain corresponding binary masks of the training images.
   c. '/test' : should contain images for testing every epoch.
   d. '/test_masks' : should contain corresponding binary masks of the test images.
2. --result-dir : For storing the results per batch and finally saving the .csv file containing all the metric values per epoch. Defaults to "/results"
3. --epochs, -e : Number of training epochs. Defaults to 50.
4. --lr : Learning rate. Defaults to 1e-4
5. --batch_size, -B : batch size per training epoch. This is same for test dataloader as well. Defaults ot 16.
6. --model_size, -m :  Choice of model size. Choose from : (a) tiny (b) small (c) base (d) large. Defaults to tiny.
7. --loss_fn : Choice of loss function. Choose from : (a) dice (b) jaccard (c) bce_jaccard (d) bce_dice (e) jdbc. Here 'bce' refers to Binary Cross Entropy loss. This argument defaults to 'jdbc' (Our proposed loss function).
8. --vertheta, -v : Vertheta value of JDBC loss. Defaults to 0.25
9. --alpha : Alpha value for joint loss functions like bce_dice and bce_jaccard. Defaults to 0.5.
10. --num_workers, -w : Number of CPU workers for dataloaders. Defaults to 2.
11. --pin_mem : Boolean value for pinning to memory during training. Defaults to True.
12. --optimizer : Choice of optimizer function. Choices are : (a) adam and (b) adamw. Defaults to adamw.
    
## Citations 🌞

Please cite our paper in your project/ research paper if you have used our model in your work: 

'''bibtex
@article{MALLICK2023126798,
title = {Response Fusion Attention U-ConvNext for accurate segmentation of optic disc and optic cup},
journal = {Neurocomputing},
pages = {126798},
year = {2023},
issn = {0925-2312},
doi = {https://doi.org/10.1016/j.neucom.2023.126798},
url = {https://www.sciencedirect.com/science/article/pii/S0925231223009219},
author = {Siddhartha Mallick and Jayanta Paul and Jaya Sil},
keywords = {ConvNeXt, Fundus image analysis, Glaucoma diagnosis, Loss function, Semantic segmentation, U-Net}
}
'''
