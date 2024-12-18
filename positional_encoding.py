import torch

class Embedder():
    def __init__(self,**kwargs):
        self.kwargs = kwargs
        self.create_embedding()
    
    def create_embedding(self):
        embed_fns = [] #嵌入函数
        d = self.kwargs['input_dims']
        outdim = 0 #计数输出维度

        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x )
            outdim += d

        max_freq = self.kwargs['max_freq_log2'] # 就是论文中的L
        num_freqs = self.kwargs['num_freqs'] # 0~maxfreq中的分频步长

        # 生成所有的频率,相当于嵌入函数中的常量，与输入无关
        if self.kwargs['log_sampling']:
            freqs = 2.**torch.linspace(0,max_freq,num_freqs)
        else:
            freqs = torch.linspace(0,max_freq,num_freqs)

        # 生成所有的嵌入函数
        # ??? 为什么没有pi 场景很多归一化到[-1.5,1.5]，这里相当于做了一个1/pi的缩放，以免超出边界，造成串扰？
        # 小心闭包的延迟绑定
        for freq in freqs:
            for fn in self.kwargs['periodic_funs']:
                embed_fns.append(lambda x,fn=fn,freq=freq:fn(x*freq))
                outdim +=d #输出维度是freqs x fns x d
        
        self.embed_fns = embed_fns
        self.outdim = outdim
    
    def embed(self,input):
        return torch.cat([fn(input) for fn in self.embed_fns],dim=-1)

def get_embedder(multires,i=0):
    '''
    multires:相当于论文里的L
    返回嵌入后的向量和他的维度
    i = -1 不进行位置编码
    '''

    if i == -1:
        return torch.nn.Identity(),3
    
    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires - 1,
        'num_freqs': multires, #相当于以1为间隔
        'log_sampling': True,
        'periodic_funs': [torch.sin,torch.cos]
    }
    
    embedder = Embedder(**embed_kwargs)
    embed_fn = lambda x,embedder=embedder : embedder.embed(x)
    return embed_fn, embedder.outdim # 返回的是配置好的embed函数，和输出维度。并不是嵌入后的数据结果

