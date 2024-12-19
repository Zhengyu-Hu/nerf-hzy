import torch
import torch.nn.functional as F
def accumulated_transmittance(alpha):
    T = torch.cumprod(1-alpha,dim=1)
    T = torch.cat([torch.ones([T.shape[0],1,T.shape[-1]]),T[:,:-1,:]],dim=1)
    return T

def rendering(model,rays_o,rays_d,tn,tf,N_samples=100):
    '''
    rays_o/d [num_rays,3]
    '''
    t = torch.linspace(tn,tf,N_samples) # [N_samples]
    delta = t[1:]-t[:-1]
    delta = torch.cat([delta,torch.tensor([1e10])])

    x = rays_o.unsqueeze(1) + \
        t.unsqueeze(0).unsqueeze(-1)*rays_d.unsqueeze(1) # [num_rays,N_samples,3]
    density,color = model.intersect(x.reshape(-1,3)) # 输入是3维的1个点
    # density: [num_rays*N_samples,1] 
    # color: [num_rays*N_samples,3]
    density = density.reshape([x.shape[0],x.shape[1],1])
    color = color.reshape(x.shape)

    alpha = 1 - torch.exp(-density*delta.unsqueeze(0).unsqueeze(-1))
    T = accumulated_transmittance(alpha)
    C = T*alpha*color
    return C.sum(1)

class Sphere():
    
    def __init__(self, p, r, c):
        self.p = p
        self.r = r
        self.c = c
        
    def intersect(self, x):
        """
        :param x: points [batch_size, 3]
        """
        
        # (x- xc)^2 + (y-yc)^2 + (z-zc)^2 <= r^2 
        
        cond = (x[:, 0] - self.p[0])**2 + (x[:, 1] - self.p[1])**2 + (x[:, 2] - self.p[2])**2 <= self.r**2
                
        num_rays = x.shape[0]
        colors = torch.zeros((num_rays, 3))
        density = torch.zeros((num_rays, 1))
        
        colors[cond] = self.c
        density[cond] = 10
        
        return  density,colors
    
def raw2outputs(raw,z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    '''
    将MPL的输出转换成需要的密度和颜色
    raw [num_rays, num_samples, 4]
    z_vals [num_rays, num_samples]
    rays_d [num_rays, 3]
    '''
    # 密度层的输出用这个
    raw2alpha = lambda x, dists, act_fun = F.relu : 1 - torch.exp(-act_fun(x)*dists) 
    # 后一项减前一项算出距离
    dists = z_vals[...,1:] - z_vals[...,:-1] # [num_rays, num_samples-1]
    dists = torch.cat([dists,torch.tensor(1e10).expand([dists.shape[0],1])],dim=-1) # [num_rays, num_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw[...,:3]) # [num_rays, num_samples, 3]

    # 论文里说给密度加噪声可以提高表现
    density = raw[...,3] # [num_rays, num_samples]
    noise = 0
    if raw_noise_std > 0:
        noise = torch.randn(density.shape) * raw_noise_std
        if pytest:
            torch.manual_seed(42)
            noise = torch.randn(density.shape) * raw_noise_std
    alpha = raw2alpha(density+noise,dists)
    tmp = torch.cat( [torch.ones((alpha.shape[0],1)), 1.-alpha + 1e-10], dim=1)
    weights = alpha*torch.cumprod(tmp, dim=1)[:,:-1] # [num_rays, num_samples]
    
    tmp = weights.unsqueeze(-1)*rgb # [num_rays, num_samples, 3]
    rgb_map = torch.sum(tmp, dim=1)
    
    # 计算平均的深度，相当于越亮的地方实体越多，越能代表深度
    # axis=2 相当于就是代表了不同的距离
    depth_map = torch.sum(z_vals*weights, dim=1)
    # 视差图，深度图的倒数 ???
    disp_map = 1./torch.max( 1e-10*torch.ones_like(depth_map), depth_map / torch.sum(weights, -1) )
    # 透明度图，对不同射线相同距离的透明度求和
    acc_map = torch.sum(weights, 1)
    
    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map.unsqueeze(-1))
    

    return rgb_map, disp_map, acc_map, weights, depth_map


if __name__ == '__main__':
    '''
    from load_data import get_data
    from rays import get_rays
    import matplotlib.pyplot as plt
    imgs,poses,K = get_data(splits=['test'])
    H,W = imgs.shape[1:3]
    idx = 10
    img = imgs[idx]
    pose = poses[idx]
    rays_o,rays_d = get_rays(H,W,K[0,0],pose)
    '''
    num_rays = 1000
    num_samples = 100

    raw = torch.randn([num_rays,num_samples,4])
    z_vals = torch.randn([num_rays,num_samples])
    rays_d = torch.randn([num_rays,3])

    raw2outputs(raw,z_vals,rays_d,white_bkgd=True)


