import torch

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
if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    H = 400
    W = 400
    f = 1200
    rays_o = np.zeros((H*W, 3))
    rays_d = np.zeros((H*W, 3))

    u = np.arange(W)
    v = np.arange(H)
    u, v = np.meshgrid(u, v)

    dirs = np.stack((u - W / 2,
                    -(v - H / 2),
                    - np.ones_like(u) * f), axis=-1)
    rays_d = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)
    rays_d = rays_d.reshape(-1, 3)
    model = Sphere(torch.tensor([0, 0, -1]), 0.1, torch.tensor([1, 0.2, 0.2]))
    px_colors = rendering(model, torch.from_numpy(rays_o), torch.from_numpy(rays_d), 0.8, 1.2)
    img = px_colors.reshape(H, W, 3).numpy()
    plt.figure()
    plt.imshow(img)
    plt.show()