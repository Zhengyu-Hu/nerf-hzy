import torch
import torch.nn as nn
import torch.nn.functional as F
class NeRF(nn.Module):
    def __init__(self,D=8,W=256,input_ch_pts=3,input_ch_views=3,output_ch=4,skips=[4],use_viewdirs=False):
        '''
        D是总的处理点的层数 W是线性层的连接数
        input_ch是输入位置的维数 input_ch_views是输入的方向维数-旋转
        '''
        super().__init__()
        self.D = D
        self.W = W
        self.input_ch_pts = input_ch_pts
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        # points linears 处理点即空间位置的层
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch_pts,W)] + [nn.Linear(W,W) if i not in skips 
                                       else nn.Linear(W+input_ch_pts,W)
                                       for i in range(D-1)] # 0至倒数第二层
        )
        # 处理方向的层
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views+W,W//2)]) # 最后一层才输入方向向量

        if use_viewdirs:
            self.feature_linear = nn.Linear(W,W)
            # 输出density
            self.alpha_linear = nn.Linear(W,1)
            self.rgb_linear = nn.Linear(W//2,3)
        else:
            self.output_linear = nn.Linear(W//2,output_ch)

    
    def forward(self,x):
        '''
        0~7处理pts
        8 输出density和拼接view
        '''
        input_pts,input_views = torch.split(x,[self.input_ch_pts,self.input_ch_views],dim=-1)
        
        # 0~7
        h = input_pts
        for i,l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            # 跳跃连接那一层的输出和网络最初的输入拼起来
            if i in self.skips: 
                h = torch.cat([input_pts,h],dim=-1)
        # 8
        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature,input_views],-1)

            for i,l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)
            
            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb,alpha],dim=-1)
        else:
            outputs = self.output_linear(h)

        # 注意,这里的密度没有过ReLu,颜色也没有过sigmoid
        return outputs
    
if __name__ == '__main__':
    model = NeRF(use_viewdirs=True)