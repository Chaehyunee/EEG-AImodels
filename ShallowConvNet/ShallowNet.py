"""
2024.04.18 Written by @Chahyunee (Chaehyun Lee)

EEG-AImodels - A collection of Convolutional Neural Network models for EEG
Signal Processing and Classification, using PyTorch

Requirements:
    (1) Python == 3.9 (verified with 3.9.11)
    (2) pytorch >= 2.0.0 (verifed)


To run the EEG classification sample script, you will also need

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
        
        
        
Note:

    Class & def. for ShallowConvNet layer
    - Class : LinearWithConstraint
    - Class : Conv2dWithConstraint
    - Def. : initialize_weight

    class ShallowConvNet


"""

import torch
import torch.nn as nn


class LinearWithConstraint(nn.Linear):
    def __init__(self, *config, max_norm=1, **kwconfig):
        self.max_norm = max_norm
        super(LinearWithConstraint, self).__init__(*config, **kwconfig)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(LinearWithConstraint, self).forward(x)


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *config, max_norm=1, **kwconfig):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*config, **kwconfig)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dWithConstraint, self).forward(x)
    
    
def initialize_weight(model, method):
    method = dict(normal=['normal_', dict(mean=0, std=0.01)],
                xavier_uni=['xavier_uniform_', dict()],
                xavier_normal=['xavier_normal_', dict()],
                he_uni=['kaiming_uniform_', dict()],
                he_normal=['kaiming_normal_', dict()]).get(method)
    if method is None:
        return None

    for module in model.modules():
        # LSTM
        if module.__class__.__name__ in ['LSTM']:
            for param in module._all_weights[0]:
                if param.startswith('weight'):
                    getattr(nn.init, method[0])(getattr(module, param), **method[1])
                elif param.startswith('bias'):
                    nn.init.constant_(getattr(module, param), 0)
        else:
            if hasattr(module, "weight"):
                # Not BN
                if not ("BatchNorm" in module.__class__.__name__):
                    getattr(nn.init, method[0])(module.weight, **method[1])
                # BN
                else:
                    nn.init.constant_(module.weight, 1)
                if hasattr(module, "bias"):
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)




class ShallowConvNet(nn.Module):
    
    """
    
    Parameters
    
    < requisiment>
    n_classes : number of target classses
    input_shape : s, t (e.g. [32, 1048])
        s: number of channels
        t: number of timepoints
    
    
    < option >
    
    F1, F2 : filter size of the first and second layer for temporal information
                Default: F1 = 5, F2 = 10
    T1 : number of time points in one trial (e.g. sec x sampling rate)
                Default: T1 = 25
    
    P1_T : pooling layer-temporal
    P1_S : pooling layer-spatial
    
    dropout : the rate of dropout. Default: 0.5            
    pool_mode : mode of the pooling. (mean, max) Default: 'mean'
    
    F1, F2  : number of temporal filters (F1) and number of pointwise
            filters (F2) to learn. Default: F1 = 8, F2 = F1 * D. 
    D       : number of spatial filters to learn within each temporal
            convolution. Default: D = 2           
    
    """
    
    
    def __init__(
            self,
            n_classes,
            input_shape,
            F1=5,
            T1=25,
            F2=10,
            P1_T=75, # pooling layer-temporal
            P1_S=15, # pooling layer-spatial
            drop_out=0.5,
            pool_mode='mean',
            weight_init_method=None,
            last_dim= 3072 #F2*spatial size*temporal size 
    ):
        super(ShallowConvNet, self).__init__()
        s, t = input_shape
        
        pooling_layer = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[pool_mode]
        
        # chaehyunee edited ver.
        self.constConv2d1 = Conv2dWithConstraint(1, F1, (1, T1), max_norm=2)
        self.constConv2d2 = Conv2dWithConstraint(F1, F2, (s, 1), bias=False, max_norm=2)
        self.bn1 = nn.BatchNorm2d(F2)
        self.pool1 = pooling_layer((1, P1_T), (1, P1_S))
        self.dropout1 = nn.Dropout(drop_out)
        self.flatten1 = nn.Flatten()
        self.linear1 = nn.Linear(1990, last_dim) # TODO 1990 자리에 맞는 param 넣기
        self.linear2 = nn.Linear(last_dim, n_classes)
        self.linearConst1 = LinearWithConstraint(last_dim, n_classes, max_norm=1)
        
        initialize_weight(self, weight_init_method)

    def forward(self, x):
        
        x = self.constConv2d1(x)
        x = self.constConv2d2(x)
        x = self.bn1(x)
        ActSquare().forward(x)
        x = self.pool1(x)
        ActLog().forward(x)
        x = self.dropout1(x)
        x = self.flatten1(x)
        x = self.linear1(x)
        out = self.linear2(x)
        # out = self.linearConst1(x) # TODO constraint 적용한 version으로 수정 필요
        
        
        return out
    

class ActSquare(nn.Module):
    def __init__(self):
        super(ActSquare, self).__init__()
        pass

    def forward(self, x):
        return torch.square(x)


class ActLog(nn.Module):
    def __init__(self, eps=1e-06):
        super(ActLog, self).__init__()
        self.eps = eps

    def forward(self, x):
        return torch.log(torch.clamp(x, min=self.eps))