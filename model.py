import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align
from FlashAttention import MultiHeadedFlashAttention
from utils import apply_deltas, iou
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


# Huge help from for figuring out loss functions and model heads: 
# https://github.com/Okery/PyTorch-Simple-MaskRCNN/blob/master/pytorch_mask_rcnn/

def flash_attention_layer(num_channels, num_heads):
    return MultiHeadedFlashAttention(num_channels, num_heads)
class RESNET(nn.Module):
    def __init__(
        self,
        input_channels : int,
        channel_layers : list[int],
        use_attention : bool = True
    ) -> None:
        cnn_layers = []

        cnn_layers.append(
            nn.Sequential(
                nn.Conv2d(
                    input_channels,
                    channel_layers[0],
                    padding = 1
                ),
                nn.BatchNorm2d(channel_layers[0]),
                flash_attention_layer(channel_layers[0], 4),
                nn.LeakyReLU(),
                nn.Conv2d(
                    channel_layers[0],
                    channel_layers[1],
                    padding = 1
                ),
                nn.BatchNorm2d(channel_layers[1]),
                flash_attention_layer(channel_layers[1], 4),
                nn.LeakyReLU(),
            )
        )

        for i in range(1, len(channel_layers) - 1):
            self.cnn_layer = nn.Sequential(
                nn.Conv2d(channel_layers[i], channel_layers[i + 1], padding = 1),
                nn.BatchNorm2d(channel_layers[i + 1]),
                nn.LeakyReLU(),
            )
            cnn_layers.append(self.cnn_layer)
        
        self.cnn = nn.Sequential(*cnn_layers)
    
    def forward(
        self,
        input : th.Tensor,
    ):
        return self.cnn(input)
        
class RPN(nn.Module):

    def __init__(
        self,
        input_channels, # number of channels received from feature maps
        anchor_scale, # Determines the scale of the anchor box in comparison to the base anchor size
        anchor_ratios, # Determines the ratio of the anchor box width to the anchor width length
    ) -> None:
        super(RPN, self).__init__()

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

        self.leakyReLU = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim = 1)

    def forward(
        self,
        input
    ):
        out = self.leakyReLU(self.cnn(input))

        classification = self.objectness(out)

        batch_size, channels, width, height = classification.shape

        scores = self.softmax(classification.view(batch_size, 2, -1))

        fg_probs = scores[:, 1] 
        # Take the second element of the second dimension of the tensor to be the foreground probabilitiy
        
        anchor_adjustments = self.anchors(out)

        return fg_probs, anchor_adjustments
    
    def find_loss(
        self,
        obj_pred,
        box_deltas,
        target_boxes,
        anchors,
        threshold,
    ) -> th.Tensor:
        
        changed_anchors = apply_deltas(anchors, box_deltas) # Change the anchors based on the box_deltas that was predicted for them

        iou_scores = iou(target_boxes, changed_anchors) # Generate the intersection over union scores for each anchor with each target box

        # match the anchors to their most likely target boxes based on their iou_scores
        # Give a 1 if intersection over union score was high enough over the region, and a 0 if intersection over union score was not high enough 
        # match_idx tells the indices that the anchors were matched to based on their scores
        bg_fg_score, matched_idx = match(iou_scores, threshold)

        # Since a lot of the outputs generated will be backgrounds instead of foregrounds, we want to ensure that there are sufficient correct cases for the 
        # model to deal with
        corr_idx, incorr_idx = sample(bg_fg_score)

        idx = th.cat((corr_idx, incorr_idx))

        # This loss determines how correctly the model predicted if there was an object in the bounding box
        # Remember, we set a threshold in the match function that determines whether or not there is an object in the bounding box (based on sufficient IOU)
        # with the original box, so the target is actual the labels

        # We use binary cross entropy loss which is good for determining if predictions between 0 and 1s are close to target results (i.e. prediction of 0.8 is good for target label of 1)
        objectness_loss = F.binary_cross_entropy_with_logits(obj_pred[idx], bg_fg_score[idx])

        # Remember that we also want to determine a loss function for optimizing where the bounding box should be placed. This loss is very important for
        # helping to change the location of the bounding boxes
        
        # We need to decompose the location of the bounding boxes down to their coordinates and compute a loss on them
        target_coords = decompose(target_boxes[matched_idx[corr_idx]]) # select the target box that each changed anchor ended up selecting
        predicted_coords = decompose(changed_anchors[corr_idx])
        regression_loss = F.l1_loss(predicted_coords, target_coords)

        return objectness_loss + regression_loss


class model(nn.Module):
    def __init__(
        self,
        input_channels : int,
        num_object_types : int,
        anchor_scales : list[int],
        anchor_ratios : list[int]
    ) -> None:
        feature_extractor_layers = [128, 128, 256, 512]
        self.feature_extractor = RESNET(input_channels, feature_extractor_layers)

        self.region_proposal_net = RPN(feature_extractor_layers[-1], anchor_scales, anchor_ratios)


    
    def forward(
        self,
        input : th.Tensor
    ) -> th.Tensor:
        
        return

    def find_loss(
        self,
        actual : th.Tensor,
        reconstruction : th.Tensor,
    ) -> th.Tensor:
        
        return