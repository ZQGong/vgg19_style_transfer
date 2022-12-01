# vgg19_style_transf
% Readme                                   
% Image Style Transfer Using Convolutional Neural Networks        

"VGG" stands for the Oxford Visual Geometry Group at the University of Oxford, which is part of the Robotics Research Group, founded in 1985, and whose research ranges from machine learning to mobile robotics.

About the Project - 
--------------------
This project implements the algorithm found in [(Gatys 2016)](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf).
This model is able to extract style features from style images and transfer them to content images.

Libraries required - 
--------------------
The model is mainly based on the pytorch framework, and the torch and torchvision libraries need to be installed. The image data is imported using the PIL library.

Input files - 
--------------------
The input files are "style image" for which style features are to be extracted, and "content image" for which style transfer is desired. They should all be saved in the folder /data/images in the same directory as the python file.

Running the code - 
--------------------
Run the file "style_transfer.py".
