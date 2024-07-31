import torch as th
import torch.nn as nn
#Mask R CNNs 
# Original Paper: https://arxiv.org/pdf/1703.06870

# The components that Mask R CNNs consist of 

# https://www.geeksforgeeks.org/faster-r-cnn-ml/
# Region proposal to identify locations where objects might be
# CNN extracts important features of the environment
# Classiciation predicts the class in the bounding box
# Regression to fine-tune bounding box coordinates

# Need a CNN backbone to extract features from the environment
# can use pretrained RESNET or VGG

# Region proposal network 
# Generates anchors for anchor boxes where anchor boxes are placed
# Uses a small CNN to process the feature map obtained from the feature extraction process

# Fast R-CNN detector
# Take the regions suggested by RPN and apply Roi pooling (for Faster R-Cnn), and ROI-align
# for mask r-cnn

class RPN(nn.Module):

    def __init__(
        self,
        input_channels, # number of channels received from feature maps
        anchor_scale, # Determines the scale of the anchor box in comparison to the base anchor size
        anchor_ratios, # Determines the ratio of the anchor box width to the anchor width length
    ) -> None:
        
        self.num_anchors = len(anchor_ratios) * len(anchor_scale)
        self.cnn = nn.Conv2d(
            in_channels = input_channels,
            out_channels = 512,
            kernel_size= 3
        )
        
        # Determine if the channel (of the anchor box) contains an object or not
        # Instead of just using self.num_anchors, use self.num_anchors * 2
        # to determine the score of the object if it is a object and if it is background
        # Then use softmax to determine the actual probability that it is an object
        self.objectness = nn.Conv2d(
            in_channels=512,
            out_channels=self.num_anchors * 2,
            kernel_size=1
        )

        self.anchors = nn.Conv2d(
            in_channels = 512,
            out_channels = self.num_anchors * 4,
            kernel_size= 1
        )

    
    def forward(
        self,
        input
    ):
        out = nn.LeakyReLU(self.cnn(input))

        classification = self.objectness(out)

        batch_size, channels, width, height = classification.shape

        scores = nn.Softmax(classification.view(batch_size, 2, -1), dim = 1)

        fg_probs = scores[:, 1] 
        # Take the second element of the second dimension of the tensor to be the foreground probabilitiy
        
        anchor_adjustments = self.anchors(out)

        return fg_probs, anchor_adjustments