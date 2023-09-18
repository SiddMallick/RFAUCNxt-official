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
6. --model_size, -m : Choice of loss function. Defaults to 'jdbc' (Our proposed loss function).