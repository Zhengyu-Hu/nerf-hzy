import torch
import torch.nn as nn
import torch.nn.functional as F
from positional_encoding import get_embedder
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NeRF(nn.Module):
    def __init__(self,D=8,W=256,input_ch=3,input_ch_views=3,output_ch=4,skips=[4],use_viewdirs=False):
        '''
        D是总的处理点的层数 W是线性层的连接数
        input_ch是输入位置的维数 input_ch_views是输入的方向维数-旋转
        '''
        super().__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        # points linears 处理点即空间位置的层
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch,W)] + [nn.Linear(W,W) if i not in skips 
                                       else nn.Linear(W+input_ch,W)
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
        input_pts,input_views = torch.split(x,[self.input_ch,self.input_ch_views],dim=-1)
        
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
            # 原文设计

            # density
            alpha = self.alpha_linear(h)
            # 用来和方向拼接的特征向量
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

def run_network():
    pass

   
def create_nerf(args):
    
    # 对空间点进行位置编码
    embed_fn,input_ch = get_embedder(args.multires,args.i_embed)

    # 对方向进行位置编码
    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn,input_ch_views = get_embedder(args.multires_views,args.i_embed)

    # N_importance指示的是精细网络
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]

    # 创建粗糙网络
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch,skips=skips,
                 input_ch_views=input_ch_views,use_viewdirs=args.use_viewdirs).to(device)
    grad_vars = list(model.parameters())

    # 创建精细网络
    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch,skips=skips,
                          input_ch_views=input_ch_views,use_viewdirs=args.use_viewdirs).to(device)
        grad_vars += list(model_fine.parameters())
    
    network_query_fn = lambda inputs,viewdirs,network_fn : run_network(inputs,viewdirs,network_fn,
                                                                       embed_fn=embed_fn,embeddirs_fn=embeddirs_fn,
                                                                       netchunk = args.netchunk)
    
    optimizer = torch.optim.Adam(params=grad_vars,lr=args.lrate,betas=(0.9,0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    # 加载模型
    if args.ft_path is not None and args.ft_path!= 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir,expname,f) 
                 for f in sorted(os.listdir(basedir,expname))
                 if 'tar' in f]
    print('Found ckpts',f"==>> ckpts: {ckpts}")

    if len(ckpts)>0 and not args.no_reload:
        ckpt_path = ckpts[-1] # 从最长的轮数开始训练，因为sorted是升序排列的
        print('Reload from',f"==>> ckpt_path: {ckpt_path}")
        
        ckpt = torch.load(ckpt_path)
        start = ckpt['global_step']
        print(f"==>> start: {start}")
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None: 
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])
    
    render_kwargs_train = {
        'network_query_fn':network_query_fn,
        'perturb':args.perturb,
        'N_importance':args.N_importance,
        'network_fine':model_fine,
        'N_samples':args.N_samples,
        'network_fn':model,
        'use_viewdirs':args.use_viewdirs,
        'white_bkgd':args.white_bkgd,
        'raw_noise_std':args.raw_noise_std
    }

    render_kwargs_test = {k:render_kwargs_train[k] for k in render_kwargs_train.keys()}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0

    return render_kwargs_train,render_kwargs_test,start,grad_vars,optimizer

if __name__ == '__main__':
    model = NeRF(use_viewdirs=True)