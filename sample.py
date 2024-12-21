import torch
import numpy as np
def sample_pdf(bins, weights, N_importance, det=False, pytest=False):
    '''
    bins 粗采样的点
    weights 粗采样的点的权重
    '''
    weights = weights + 1e-5
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat( [torch.zeros([cdf.shape[0],1]), cdf], dim=-1 ) # [batch, num_samples]
    


    if pytest: 
            torch.manual_seed(42)
    # 是否采用随机采样
    if det:
        u = torch.linspace(0., 1., steps=N_importance)
        u = u.expand(list(cdf.shape[:-1]) + [N_importance]) # 输入的有可能是批数据，所以写排除最后一维
    else:
        
        u = torch.rand(list(cdf.shape[:-1]) + [N_importance])
    
    # 求cdf的逆
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True) - 1 # 找到使u>=cdf的最后一个索引
    

    # inds-1 就是cdf中最大的小于等于u的数
    # 确保索引不越界
    below = torch.max(torch.zeros_like(inds), inds)
    above = torch.min(torch.ones_like(inds) * (cdf.shape[-1]-1), inds+1)

    # 如果给每个粗采样点标号，那么inds_g就表示细采样点被哪两个粗采样点夹住
    inds_g = torch.stack([below, above], dim=-1) # [batch, N_importance, 2]
    

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]] # [batch, N_importance, num_samples]
    # 选夹住细点的两个粗点的cdf值
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = cdf_g[...,1] - cdf_g[...,0] # 上下界的差值，cdf的差，相当于是落在此区间的概率
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom) # 插值过小不行
    t = (u-cdf_g[...,0]) / denom #采样点到底在夹住他的两个点之间的哪个位置
    samples = bins_g[...,0] + t*(bins_g[...,1]-bins_g[...,0])

    return samples

if __name__ == '__main__':
    num_rays = 1000
    num_samples = 100
    weights = torch.randn([num_rays, num_samples-1])
    N_importance = 50
    sample_pdf(None, weights, N_importance)