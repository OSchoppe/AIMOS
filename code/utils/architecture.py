import torch

#%% Building blocks
class ConvBnRelu2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, output_padding=1, dilation=1, stride=1, groups=1, is_bn=True, is_relu=True, is_decoder=False):
        super(ConvBnRelu2d, self).__init__()
        if is_decoder:
            self.transpConv = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, output_padding=output_padding, stride=stride, dilation=dilation, groups=groups, bias=False)
            self.conv = None
        else:
            self.transpConv = None
            self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels, eps=1e-4)
        self.relu = torch.nn.ReLU(inplace=True)
        if is_bn is False: self.bn = None
        if is_relu is False: self.relu = None

    def forward(self, x):
        if self.conv is None:
            x = self.transpConv(x)
        elif self.transpConv is None:
            x = self.conv(x)
            
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class StackEncoder(torch.nn.Module):
    def __init__(self, x_channels, y_channels, kernel_size=3, stride=1):
        super(StackEncoder, self).__init__()
        padding = (kernel_size - 1) // 2
        self.encode = torch.nn.Sequential(
            ConvBnRelu2d(x_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=stride, groups=1),
            ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=stride, groups=1),
        )

    def forward(self, x):
        y = self.encode(x)
        y_small = torch.nn.functional.max_pool2d(y, kernel_size=2, stride=2)
        return y, y_small


class StackDecoder(torch.nn.Module):
    def __init__(self, x_big_channels, x_channels, y_channels, kernel_size=3, stride=1):
        super(StackDecoder, self).__init__()
        padding = (kernel_size - 1) // 2

        self.decode = torch.nn.Sequential(
            ConvBnRelu2d(x_big_channels + x_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=stride, groups=1),
            ConvBnRelu2d(y_channels,                  y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=stride, groups=1),
            ConvBnRelu2d(y_channels,                  y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=stride, groups=1),
        )

    def forward(self, x_big, x):
        N, C, H, W = x_big.size()
        y = torch.nn.functional.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        y = torch.cat([y, x_big], 1)
        y = self.decode(y)
        return y


class StackDecoderTranspose(torch.nn.Module):
    def __init__(self, x_big_channels, x_channels, y_channels, kernel_size=3, stride=1, stride_transpose=2, padding=1, padding_transpose=1, output_padding=0):
        super(StackDecoderTranspose, self).__init__()
        padding = (kernel_size - 1) // 2

        self.decode = torch.nn.Sequential(
            ConvBnRelu2d(x_big_channels + x_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=stride, groups=1),
            ConvBnRelu2d(y_channels,                  y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=stride, groups=1),
            ConvBnRelu2d(y_channels,                  y_channels, kernel_size=kernel_size, padding=padding_transpose, output_padding=output_padding, dilation=1, stride=stride_transpose, groups=1, is_decoder=True),
        )

    def forward(self, x_big, x):
        y = torch.cat([x, x_big], 1)
        y = self.decode(y)
        return y

#%% Networks

class UNet1024(torch.nn.Module):
    def __init__(self, num_classes):
        super(UNet1024, self).__init__()
        self.down1 = StackEncoder(  1,   32, kernel_size=3)
        self.down2 = StackEncoder( 32,   64, kernel_size=3)
        self.down3 = StackEncoder( 64,  128, kernel_size=3)
        self.down4 = StackEncoder(128,  256, kernel_size=3)
        self.down5 = StackEncoder(256,  512, kernel_size=3)
        self.down6 = StackEncoder(512, 1024, kernel_size=3)
        self.center = ConvBnRelu2d(1024, 1024, kernel_size=3, padding=1, stride=1)
        self.up6 = StackDecoder(1024, 1024, 512, kernel_size=3) # in_skip, in_blow, out
        self.up5 = StackDecoder( 512,  512, 256, kernel_size=3)
        self.up4 = StackDecoder( 256,  256, 128, kernel_size=3)
        self.up3 = StackDecoder( 128,  128,  64, kernel_size=3)
        self.up2 = StackDecoder(  64,   64,  32, kernel_size=3)
        self.up1 = StackDecoder(  32,   32,  32, kernel_size=3) 
        self.classify = torch.nn.Conv2d(32, num_classes, kernel_size=1, padding=0, stride=1, bias=True)

    def forward(self, x):
        out = x
        skip1, out = self.down1(out) #256
        skip2, out = self.down2(out) #128
        skip3, out = self.down3(out) #64
        skip4, out = self.down4(out) #32
        skip5, out = self.down5(out) #16
        skip6, out = self.down6(out) #8
        out = self.center(out)
        out = self.up6(skip6, out)
        out = self.up5(skip5, out)
        out = self.up4(skip4, out)
        out = self.up3(skip3, out)
        out = self.up2(skip2, out)
        out = self.up1(skip1, out)
        out = self.classify(out)
        return out


class UNet768(torch.nn.Module):
    def __init__(self, num_classes):
        super(UNet768, self).__init__()
        self.down1 = StackEncoder(  1,   32, kernel_size=3)
        self.down2 = StackEncoder( 32,   64, kernel_size=3)
        self.down3 = StackEncoder( 64,  128, kernel_size=3)
        self.down4 = StackEncoder(128,  256, kernel_size=3)
        self.down5 = StackEncoder(256,  512, kernel_size=3)
        self.down6 = StackEncoder(512,  768, kernel_size=3)
        self.center = ConvBnRelu2d(768, 768, kernel_size=3, padding=1, stride=1)
        self.up6 = StackDecoder( 768,  768, 512, kernel_size=3) # in_skip, in_blow, out
        self.up5 = StackDecoder( 512,  512, 256, kernel_size=3)
        self.up4 = StackDecoder( 256,  256, 128, kernel_size=3)
        self.up3 = StackDecoder( 128,  128,  64, kernel_size=3)
        self.up2 = StackDecoder(  64,   64,  32, kernel_size=3)
        self.up1 = StackDecoder(  32,   32,  32, kernel_size=3) 
        self.classify = torch.nn.Conv2d(32, num_classes, kernel_size=1, padding=0, stride=1, bias=True)

    def forward(self, x):
        out = x
        skip1, out = self.down1(out) #256
        skip2, out = self.down2(out) #128
        skip3, out = self.down3(out) #64
        skip4, out = self.down4(out) #32
        skip5, out = self.down5(out) #16
        skip6, out = self.down6(out) #8
        out = self.center(out)
        out = self.up6(skip6, out)
        out = self.up5(skip5, out)
        out = self.up4(skip4, out)
        out = self.up3(skip3, out)
        out = self.up2(skip2, out)
        out = self.up1(skip1, out)
        out = self.classify(out)
        return out



class UNet512(torch.nn.Module):
    def __init__(self, num_classes):
        super(UNet512, self).__init__()
        self.down1 = StackEncoder(  1,   32, kernel_size=3)
        self.down2 = StackEncoder( 32,   64, kernel_size=3)
        self.down3 = StackEncoder( 64,  128, kernel_size=3)
        self.down4 = StackEncoder(128,  256, kernel_size=3)
        self.down5 = StackEncoder(256,  512, kernel_size=3)
        #self.down6 = StackEncoder(512,  768, kernel_size=3)
        self.center = ConvBnRelu2d(512, 512, kernel_size=3, padding=1, stride=1)
        #self.up6 = StackDecoder( 768,  768, 512, kernel_size=3) # in_skip, in_blow, out
        self.up5 = StackDecoder( 512,  512, 256, kernel_size=3)
        self.up4 = StackDecoder( 256,  256, 128, kernel_size=3)
        self.up3 = StackDecoder( 128,  128,  64, kernel_size=3)
        self.up2 = StackDecoder(  64,   64,  32, kernel_size=3)
        self.up1 = StackDecoder(  32,   32,  32, kernel_size=3) 
        self.classify = torch.nn.Conv2d(32, num_classes, kernel_size=1, padding=0, stride=1, bias=True)

    def forward(self, x):
        out = x
        skip1, out = self.down1(out) #256
        skip2, out = self.down2(out) #128
        skip3, out = self.down3(out) #64
        skip4, out = self.down4(out) #32
        skip5, out = self.down5(out) #16
        #skip6, out = self.down6(out) #8
        out = self.center(out)
        #out = self.up6(skip6, out)
        out = self.up5(skip5, out)
        out = self.up4(skip4, out)
        out = self.up3(skip3, out)
        out = self.up2(skip2, out)
        out = self.up1(skip1, out)
        out = self.classify(out)
        return out


class UNet256(torch.nn.Module):
    def __init__(self, num_classes):
        super(UNet256, self).__init__()
        self.down1 = StackEncoder(  1,   32, kernel_size=3)
        self.down2 = StackEncoder( 32,   64, kernel_size=3)
        self.down3 = StackEncoder( 64,  128, kernel_size=3)
        self.down4 = StackEncoder(128,  256, kernel_size=3)
        #self.down5 = StackEncoder(256,  512, kernel_size=3)
        #self.down6 = StackEncoder(512,  768, kernel_size=3)
        self.center = ConvBnRelu2d(256, 256, kernel_size=3, padding=1, stride=1)
        #self.up6 = StackDecoder( 768,  768, 512, kernel_size=3) # in_skip, in_blow, out
        #self.up5 = StackDecoder( 512,  512, 256, kernel_size=3)
        self.up4 = StackDecoder( 256,  256, 128, kernel_size=3)
        self.up3 = StackDecoder( 128,  128,  64, kernel_size=3)
        self.up2 = StackDecoder(  64,   64,  32, kernel_size=3)
        self.up1 = StackDecoder(  32,   32,  32, kernel_size=3) 
        self.classify = torch.nn.Conv2d(32, num_classes, kernel_size=1, padding=0, stride=1, bias=True)

    def forward(self, x):
        out = x
        skip1, out = self.down1(out) #256
        skip2, out = self.down2(out) #128
        skip3, out = self.down3(out) #64
        skip4, out = self.down4(out) #32
        #skip5, out = self.down5(out) #16
        #skip6, out = self.down6(out) #8
        out = self.center(out)
        #out = self.up6(skip6, out)
        #out = self.up5(skip5, out)
        out = self.up4(skip4, out)
        out = self.up3(skip3, out)
        out = self.up2(skip2, out)
        out = self.up1(skip1, out)
        out = self.classify(out)
        return out


class UNet128(torch.nn.Module):
    def __init__(self, num_classes):
        super(UNet128, self).__init__()
        self.down1 = StackEncoder(  1,   32, kernel_size=3)
        self.down2 = StackEncoder( 32,   64, kernel_size=3)
        self.down3 = StackEncoder( 64,  128, kernel_size=3)
        # self.down4 = StackEncoder(128,  256, kernel_size=3)
        #self.down5 = StackEncoder(256,  512, kernel_size=3)
        #self.down6 = StackEncoder(512,  768, kernel_size=3)
        self.center = ConvBnRelu2d(128, 128, kernel_size=3, padding=1, stride=1)
        #self.up6 = StackDecoder( 768,  768, 512, kernel_size=3) # in_skip, in_blow, out
        #self.up5 = StackDecoder( 512,  512, 256, kernel_size=3)
        # self.up4 = StackDecoder( 256,  256, 128, kernel_size=3)
        self.up3 = StackDecoder( 128,  128,  64, kernel_size=3)
        self.up2 = StackDecoder(  64,   64,  32, kernel_size=3)
        self.up1 = StackDecoder(  32,   32,  32, kernel_size=3) 
        self.classify = torch.nn.Conv2d(32, num_classes, kernel_size=1, padding=0, stride=1, bias=True)

    def forward(self, x):
        out = x
        skip1, out = self.down1(out) #256
        skip2, out = self.down2(out) #128
        skip3, out = self.down3(out) #64
        # skip4, out = self.down4(out) #32
        #skip5, out = self.down5(out) #16
        #skip6, out = self.down6(out) #8
        out = self.center(out)
        #out = self.up6(skip6, out)
        #out = self.up5(skip5, out)
        # out = self.up4(skip4, out)
        out = self.up3(skip3, out)
        out = self.up2(skip2, out)
        out = self.up1(skip1, out)
        out = self.classify(out)
        return out


class UNet64(torch.nn.Module):
    def __init__(self, num_classes):
        super(UNet64, self).__init__()
        self.down1 = StackEncoder(  1,   32, kernel_size=3)
        self.down2 = StackEncoder( 32,   64, kernel_size=3)
        # self.down3 = StackEncoder( 64,  128, kernel_size=3)
        # self.down4 = StackEncoder(128,  256, kernel_size=3)
        #self.down5 = StackEncoder(256,  512, kernel_size=3)
        #self.down6 = StackEncoder(512,  768, kernel_size=3)
        self.center = ConvBnRelu2d(64, 64, kernel_size=3, padding=1, stride=1)
        #self.up6 = StackDecoder( 768,  768, 512, kernel_size=3) # in_skip, in_blow, out
        #self.up5 = StackDecoder( 512,  512, 256, kernel_size=3)
        # self.up4 = StackDecoder( 256,  256, 128, kernel_size=3)
        # self.up3 = StackDecoder( 128,  128,  64, kernel_size=3)
        self.up2 = StackDecoder(  64,   64,  32, kernel_size=3)
        self.up1 = StackDecoder(  32,   32,  32, kernel_size=3) 
        self.classify = torch.nn.Conv2d(32, num_classes, kernel_size=1, padding=0, stride=1, bias=True)

    def forward(self, x):
        out = x
        skip1, out = self.down1(out) #256
        skip2, out = self.down2(out) #128
        # skip3, out = self.down3(out) #64
        # skip4, out = self.down4(out) #32
        #skip5, out = self.down5(out) #16
        #skip6, out = self.down6(out) #8
        out = self.center(out)
        #out = self.up6(skip6, out)
        #out = self.up5(skip5, out)
        # out = self.up4(skip4, out)
        # out = self.up3(skip3, out)
        out = self.up2(skip2, out)
        out = self.up1(skip1, out)
        out = self.classify(out)
        return out


