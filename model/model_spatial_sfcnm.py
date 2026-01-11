# encoding = utf-8

# @Author     ：Lecheng Wang
# @Time       : 2026/1/10 01:12 (Revised)
# @Function   : 
# @Description: model structure ......

import math
import torch
import torch.nn      as nn



class Spatial_Attention(nn.Module):
    def __init__ (self, kernel_size=3):
        super(Spatial_Attention, self).__init__()
        padding        = kernel_size//2
        self.conv      = nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.sigmoid   = nn.Sigmoid()

    def forward (self, x):
        B, C, H, W     = x.size()
        Max_pool_out,_ = torch.max(x,  dim=1, keepdim=True)
        Mean_pool_out  = torch.mean(x, dim=1, keepdim=True)
        pool_out       = torch.cat([Max_pool_out, Mean_pool_out], dim=1)
        out            = self.conv(pool_out)
        out            = self.sigmoid(out)
        return out

# atten_kernel_size must be odd number,otherwise the output_weight size won't be same.
class SegModel(nn.Module):
    def __init__(self, bands         = 6,  num_classes  = 3,
                       width         = 16, conv_kernel  = 3,
                       dilation_rate = 1,  atten_kernel = 9):
        super(SegModel, self).__init__()
        padding     = dilation_rate * (conv_kernel-1)//2

        self.stage1 = nn.Sequential(
              nn.Conv2d(bands, width, kernel_size=conv_kernel, stride=1, padding=padding, dilation=dilation_rate, bias=False),
              nn.BatchNorm2d(width),
              nn.ReLU()
        )
#        self.sa_module1 = Spatial_Attention(kernel_size=atten_kernel)

        self.stage2 = nn.Sequential(
              nn.Conv2d(width, width*2, kernel_size=conv_kernel, stride=1, padding=padding, dilation=dilation_rate, bias=False),
              nn.BatchNorm2d(width*2),
              nn.ReLU()
        )
#        self.sa_module2 = Spatial_Attention(kernel_size=atten_kernel)

        self.stage3 = nn.Sequential(
              nn.Conv2d(width*2, width*4, kernel_size=conv_kernel, stride=1, padding=padding, dilation=dilation_rate, bias=False),
              nn.BatchNorm2d(width*4),
              nn.ReLU()
        )
#        self.sa_module3 = Spatial_Attention(kernel_size=atten_kernel)

        self.stage4 = nn.Sequential(
              nn.Conv2d(width*4, width*4, kernel_size=conv_kernel, stride=1, padding=padding, dilation=dilation_rate, bias=False),
              nn.BatchNorm2d(width*4),
              nn.ReLU()
        )
        self.sa_module4 = Spatial_Attention(kernel_size=atten_kernel) 

        self.final = nn.Conv2d(width*4, num_classes, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, inputs):
        feat1 = self.stage1(inputs)
#        w1    = self.sa_module1(feat1)
#        feat1 = w1 * feat1

        feat2 = self.stage2(feat1)
#        w2    = self.sa_module2(feat2)
#        feat2 = w2 * feat2

        feat3 = self.stage3(feat2)
#        w3    = self.sa_module3(feat3)
#        feat3 = w3 * feat3

        feat4 = self.stage4(feat3)
        w4    = self.sa_module4(feat4)
        feat4 = w4 * feat4

        seg_head = self.final(feat4)
        return seg_head, feat4, w4
    
    # Called to get spatial attention map
    def get_atten_map(self, x):
        with torch.no_grad():
            seg, final_feat, map = self.forward(x)
        return map


# Test Model Structure and Outputsize
if __name__ == "__main__":
    from torchinfo  import summary
    from thop       import profile
    import matplotlib.pyplot as plt
    device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model           = SegModel(bands         = 10, num_classes  = 3,
                               width         = 16, conv_kernel  = 3,
                               dilation_rate = 1,  atten_kernel = 9).to(device)
    x               = torch.randn(1, 10, 256, 256).to(device)
    output          = model(x)
    flops, params   = profile(model, inputs=(x, ), verbose=False)
    print('GFLOPs: ', (flops/1e9)/x.shape[0], 'Params(M): ', params/1e6)
    print("Input shape:",         list(x.shape))
    print("SegOutput shape:",     list(output[0].shape))
    print("FeatOutput shape:",    list(output[1].shape))
    print("SpatialWeight shape:", list(output[2].shape))
#    print("Output w3 shape:", list(output[3].shape))
#    print("Output w4 shape:", list(output[4].shape))
    summary(model, (10, 256, 256), batch_dim=0)
