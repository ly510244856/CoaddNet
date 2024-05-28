import torch
import torch.nn as nn
import torch.nn.functional as F

# 激活函数：Leaky ReLU
def lrelu(x):
    return F.leaky_relu(x, 0.2)

# 上采样并拼接
class UpsampleAndConcat(nn.Module):
    def __init__(self, output_channels, in_channels, exp=False):
        super(UpsampleAndConcat, self).__init__()
        self.exp = exp
        self.deconv = nn.ConvTranspose2d(in_channels, output_channels, kernel_size=2, stride=2)
        self.output_channels = output_channels

    def forward(self, x1, x2, exp_time=None):
        deconv = self.deconv(x1)
        if not self.exp:
            deconv_output = torch.cat([deconv, x2], dim=1)
        else:
            cons = torch.full_like(deconv, exp_time)[:, :1, :, :]  # Assuming exp_time is a scalar
            deconv_output = torch.cat([deconv, x2, cons], dim=1)
        return deconv_output

class AstroUnet(nn.Module):
    def __init__(self, exp=False):
        super(AstroUnet, self).__init__()
        self.exp = exp
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        
        # 定义卷积层
        self.conv1_1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
        self.conv5_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        
        # 上采样并拼接层，正确地配置以支持exp_time
        self.up6 = UpsampleAndConcat(256, 512, exp=self.exp)
        self.conv6_1 = nn.Conv2d(512 + 1 if self.exp else 512,
                                 256, kernel_size=3, padding=1)  # 注意：这里加1是因为额外的exp_time通道
        self.conv6_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
        self.up7 = UpsampleAndConcat(128, 256)
        self.conv7_1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv7_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        self.up8 = UpsampleAndConcat(64, 128)
        self.conv8_1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv8_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        self.up9 = UpsampleAndConcat(32, 64)
        self.conv9_1 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv9_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        
        # 最后一层卷积
        self.conv10 = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x, exp_time=None):
        # 测试总体参数量时使用
        # if exp_time == None:
        #     exp_time = torch.tensor(0.5)  # 默认的exp_time值

        c1 = self.leaky_relu(self.conv1_1(x))
        c1 = self.leaky_relu(self.conv1_2(c1))
        p1 = F.max_pool2d(c1, kernel_size=2, stride=2)

        c2 = self.leaky_relu(self.conv2_1(p1))
        c2 = self.leaky_relu(self.conv2_2(c2))
        p2 = F.max_pool2d(c2, kernel_size=2, stride=2)

        c3 = self.leaky_relu(self.conv3_1(p2))
        c3 = self.leaky_relu(self.conv3_2(c3))
        p3 = F.max_pool2d(c3, kernel_size=2, stride=2)

        c4 = self.leaky_relu(self.conv4_1(p3))
        c4 = self.leaky_relu(self.conv4_2(c4))
        p4 = F.max_pool2d(c4, kernel_size=2, stride=2)

        c5 = self.leaky_relu(self.conv5_1(p4))
        c5 = self.leaky_relu(self.conv5_2(c5))

        # 从conv5开始的上采样层，注意传递exp_time
        up6 = self.up6(c5, c4, exp_time) # (1,512,16,16) + (1,256,32,32)
        up6 = F.leaky_relu(self.conv6_1(up6))
        up6 = F.leaky_relu(self.conv6_2(up6))

        up7 = self.up7(up6, c3)
        up7 = self.leaky_relu(self.conv7_1(up7))
        up7 = self.leaky_relu(self.conv7_2(up7))

        up8 = self.up8(up7, c2)
        up8 = self.leaky_relu(self.conv8_1(up8))
        up8 = self.leaky_relu(self.conv8_2(up8))

        up9 = self.up9(up8, c1)
        up9 = self.leaky_relu(self.conv9_1(up9))
        up9 = self.leaky_relu(self.conv9_2(up9))

        c10 = self.conv10(up9)
        return c10
    

if __name__ == '__main__':
    inp_shape=(1,256,256)
    
    net = AstroUnet(exp=False)

    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)

    params = float(params[:-3])
    macs = float(macs[:-4])

    print(macs, params)
    # print(net)
    
# if __name__ == '__main__':
#     # 设定输入张量的大小，模拟一张单通道的256x256的图像
#     inp = torch.rand(1, 1, 256, 256)  # Batch size, Channels, Height, Width
    
#     # 定义exp_time的值，这里假设它是一个标量，对应全张量的一个值
#     exp_time = torch.tensor(0.5)
    
#     # 实例化网络，exp=True以测试exp_time的传递
#     net = AstroUnet(exp=True)
    
#     # 调用网络的forward方法，传入图像数据和exp_time
#     output = net(inp, exp_time)
    
#     print(f"Output shape: {output.shape}")
#     # 打印输出张量的形状以确认输出的尺寸