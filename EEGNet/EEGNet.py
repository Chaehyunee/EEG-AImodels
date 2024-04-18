"""
2024.04.18 Written by @Chahyunee (Chaehyun Lee)

EEG-AImodels - A collection of Convolutional Neural Network models for EEG
Signal Processing and Classification, using PyTorch

Requirements:
    (1) Python == 3.9 (verified with 3.9.11)
    (2) pytorch >= 2.0.0 (verifed)


To run the EEG/MEG ERP classification sample script, you will also need

    (1) scikit-learn >= 1.2.2
    (2) matplotlib >= 3.7.1
    
To use:
    
    (1) Place this file in the PYTHONPATH variable in your IDE
    (2) Import the model as
        
        from EEGModels import EEGNet
        
        model = EEGNet(C = ..., T = ..., ..., num_classes = ...)
        
    (3) Then train the model

        model = model.to(device)        
        model.train()
        outputs, _ = model(inputs) ( _ : output before the last layer)
        
    (4) Evaluate the model
        model.eval()


    While the original paper used Dropout, we found that SpatialDropout2D 
    sometimes produced slightly better results for classification of ERP 
    signals. However, SpatialDropout2D significantly reduced performance 
    on the Oscillatory dataset (SMR, BCI-IV Dataset 2A). We recommend using
    the default Dropout in most cases.
    
    Assumes the input signal is sampled at 128Hz. If you want to use this model
    for any other sampling rate you will need to modify the lengths of temporal
    kernels and average pooling size in blocks 1 and 2 as needed (double the 
    kernel lengths for double the sampling rate, etc). Note that we haven't 
    tested the model performance with this rule so this may not work well. 
    
    The model with default parameters gives the EEGNet-8,2 model as discussed
    in the paper. This model should do pretty well in general, although it is
	advised to do some model searching to get optimal performance on your
	particular dataset.
    

        We set F2 = F1 * D (number of input filters = number of output filters) for
    the SeparableConv2D layer. We haven't extensively tested other values of this
    parameter (say, F2 < F1 * D for compressed learning, and F2 > F1 * D for
    overcomplete). We believe the main parameters to focus on are F1 and D. 

"""

import torch

class EEGNet(torch.nn.Module):
    """
    
    Parameters
    
    C : number of channels in your EEG dataset
    T : number of time points in one trial (e.g. sec x sampling rate)
    dropout : the rate of dropout  Default: 0.5            
    kernelLength : size of the temporal kernel (recommand: half of T (timepoints))    
    
    F1, F2  : number of temporal filters (F1) and number of pointwise
            filters (F2) to learn. Default: F1 = 8, F2 = F1 * D. 
    D       : number of spatial filters to learn within each temporal
            convolution. Default: D = 2        
            
    num_classes = number of target classses
    
    
    """
    def __init__(self, C=64, T=128, dropout=0.5, kernelLength=64, F1=8, D=2, F2=16, num_classes=2):
        super().__init__()
        self.channelConv = torch.nn.Conv2d(1, F1, (1, C), padding=(0, int(C/2)))
        self.bn1 = torch.nn.BatchNorm2d(F1)
        self.depthwiseConv1 = torch.nn.Conv2d(F1, F1, (1, kernelLength), padding=(0, int(kernelLength/2)))
        self.pointwiseConv1 = torch.nn.Conv2d(F1, D * F1, 1)
        
        # Set the maximum value about model weights
        for param in self.pointwiseConv1.parameters():
            if param.dim() > 1:  # except to 1-dim tensor (bias)
                param.data = torch.clamp(param.data, min=-1.0, max=1.0)
        
        
        self.bn2 = torch.nn.BatchNorm2d(D*F1)
        self.elu1 = torch.nn.ELU()
        self.pooling1 = torch.nn.AvgPool2d((1,4))
        # self.pooling1 = torch.nn.MaxPool2d((1,4)) # If you want to try, you can replace AvgPool with MaxPool!
        self.dropout1 = torch.nn.Dropout(dropout)

        self.separableConv = torch.nn.Conv2d( D * F1, D * F1, kernel_size=(1,int(kernelLength/2)), padding=(0,int(kernelLength/4)), bias=False)
        self.pointwiseConv2 = torch.nn.Conv2d(D * F1, F2, 1, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(F2)
        self.elu2 = torch.nn.ELU()
        self.pooling2 = torch.nn.AvgPool2d((1,8))
        # self.pooling2 = torch.nn.MaxPool2d((1,8)) # If you want to try, you can replace AvgPool with MaxPool!
        self.dropout2 = torch.nn.Dropout(dropout)

        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(F2 * T, num_classes) 
        # self.max_norm = nn.utils.weight_norm(self.linear1, dim=None) # TODO
        self.classifier = torch.nn.Sigmoid() if num_classes == 2 else torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.channelConv(x)
        x = self.bn1(x)
        x = self.depthwiseConv1(x)
        x = self.pointwiseConv1(x)
        x = self.bn2(x)
        x = self.elu1(x)
        x = self.pooling1(x)
        x = self.dropout1(x)
        
        x = self.separableConv(x)
        x = self.pointwiseConv2(x)

        x = self.bn3(x)
        x = self.elu2(x)
        x = self.pooling2(x)
        x = self.dropout2(x)
        out = self.flatten(x)

        x = self.linear1(out)
        x = self.classifier(x)

        return x, out
