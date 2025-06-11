import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from functools import partial

# Constants
NUM_CLASSES = 2  # Background and wall
FC_DIM = 2048    # Feature channels for ResNet-50/101 (use 512 for ResNet-18)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Model definitions
class ResnetDilated(nn.Module):
    """Dilated ResNet class, created by dilating original ResNet architecture"""
    def __init__(self, orig_resnet, dilate_scale=8):
        super(ResnetDilated, self).__init__()

        if dilate_scale == 8:
            orig_resnet.layer3.apply(partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(partial(self._nostride_dilate, dilate=4))

        # take pretrained ResNet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        
        self.maxpool = orig_resnet.maxpool
        
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            elif m.kernel_size == (3, 3):  # other convolutions
                m.dilation = (dilate, dilate)
                m.padding = (dilate, dilate)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x

class PPM(nn.Module):
    """Pyramid Pooling Module (PPM) class"""
    def __init__(self, num_class, fc_dim, pool_scales=(1, 2, 3, 6)):
        super(PPM, self).__init__()

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
            
        self.ppm = nn.ModuleList(self.ppm)
        
        self.conv_last = nn.Sequential(
            nn.Conv2d(fc_dim + len(pool_scales)*512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_class, kernel_size=1)
        )

    def forward(self, x, seg_size=None):
        input_size = x.size()
        ppm_out = [x]
        
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.interpolate(
                pool_scale(x), 
                (input_size[2], input_size[3]),
                mode='bilinear', 
                align_corners=False
            ))
        
        ppm_out = torch.cat(ppm_out, 1)
        x = self.conv_last(ppm_out)

        if seg_size:  # is True during inference
            x = nn.functional.interpolate(x, size=seg_size, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
        else:        
            x = nn.functional.log_softmax(x, dim=1)
            
        return x

class SegmentationModule(nn.Module):
    """Segmentation Module class"""
    def __init__(self, net_encoder, net_decoder):
        super(SegmentationModule, self).__init__()
        self.encoder = net_encoder
        self.decoder = net_decoder
        
    def forward(self, input_dict, seg_size=None):
        if isinstance(input_dict, dict):
            x = input_dict['img_data'].to(DEVICE)
        else:
            x = input_dict.to(DEVICE)
        return self.decoder(self.encoder(x), seg_size=seg_size)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    """Bottleneck sequence of layers used for creating resnet architecture"""
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, 4*out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(4*out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """Forward pass of the bottleneck sequence"""
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class ResNet(nn.Module):
    """Class of Resnet architecture"""
    def __init__(self, block, layers, num_classes=1000):
        self.in_channels = 128
        super(ResNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        """Method for creating sequential layer with bottleneck sequence layer"""
        downsample = None
        if stride != 1 or self.in_channels != out_channels*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels*block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels*block.expansion)
            )

        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass of the network"""       
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

def resnet101(**kwargs):
    """Function for instantiating resnet101 with specific number of layers"""
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model

# Load pretrained models
def load_wall_segmentation_model(encoder_path, decoder_path):
    """Loads the wall segmentation model with pretrained weights"""
    # Build encoder
    print("Building encoder (ResNet101-dilated)")
    net_encoder = ResnetDilated(resnet101(), dilate_scale=8)
    
    # Build decoder
    print("Building decoder (PPM)")
    net_decoder = PPM(num_class=NUM_CLASSES, fc_dim=FC_DIM)
    
    # Load weights
    print("Loading weights")
    net_encoder.load_state_dict(
        torch.load(encoder_path, map_location=lambda storage, loc: storage), 
        strict=False
    )
    
    net_decoder.load_state_dict(
        torch.load(decoder_path, map_location=lambda storage, loc: storage), 
        strict=False
    )
    
    # Create segmentation module
    segmentation_module = SegmentationModule(net_encoder, net_decoder)
    segmentation_module.to(DEVICE)
    segmentation_module.eval()
    
    return segmentation_module

# Function to segment walls in an image
def segment_walls(segmentation_module, img):
    """Segments walls in the input image"""
    # Transform for input image
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    # Convert to PIL image if needed
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    
    # Get original image dimensions for output size
    img_original = np.array(img)
    seg_size = img_original.shape[:2]
    
    # Transform image
    img_data = transform(img)
    img_data = img_data.unsqueeze(0)  # Add batch dimension
    
    # Forward pass
    with torch.no_grad():
        scores = segmentation_module(img_data, seg_size=seg_size)
    
    # Get predicted class (0: background, 1: wall)
    _, pred = torch.max(scores, dim=1)
    pred = pred.cpu()[0].numpy()
    
    # Create binary mask (255 for wall pixels)
    wall_mask = (pred == 1).astype(np.uint8) * 255
    
    return wall_mask
