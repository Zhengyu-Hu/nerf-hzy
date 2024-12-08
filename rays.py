import numpy as np
import pylab as plt
def get_rays(H,W,f,c2w):

    # 生成所有的坐标点(u,v)
    u,v = np.meshgrid(np.arange(W,dtype=np.float32),
                      np.arange(H,dtype=np.float32))
    u,v = np.meshgrid(np.arange(W,dtype=np.float32),
                      np.arange(H,dtype=np.float32))
    u = u.reshape(-1)
    v = v.reshape(-1)

    # 像素坐标变成相机坐标
    # 默认原点(0,0)，故坐标即方向
    # (160000,)(160000,)(160000,)进行堆叠，添加第1维度，故会变成(160000,3)
    directions = np.stack([u-W/2,H/2-v,-f*np.ones_like(u)],axis=1)
    # 选取c2w的前3行和前3列，这是表示旋转的矩阵
    # c2w: 3x3; directions: Nx3
    # 3x3 @ Nx3x1
    # 相当于是N次3x3 @ (3,) 
    directions = (c2w[:3,:3] @ directions[...,None]).squeeze(-1)
    #把d变成方向向量
    # shape = [WxH,3]
    directions = directions / np.linalg.norm(directions,axis=1,keepdims = True)
    
    origins = np.tile(c2w[:3,-1],(directions.shape[0],1))
    
    return origins,directions

def plot_rays(origins,directions,t=1.,step=5000):
    plt.figure('rays')
    ax = plt.axes(projection = '3d')

    # 起点大家都是一样的
    np.random.shuffle(directions)
    pts1 = origins[::step]
    pts2 = origins[::step] + t * directions[::step]
    for p1,p2 in zip(pts1,pts2):
        plt.plot([p1[0],p2[0]],[p1[1],p2[1]],[p1[2],p2[2]],c='C0')

    plt.show()


def plot_image_and_rays(image, origins, directions, t=5.0, step=5000):
    """
    绘制图像和射线的联合显示。
    """
    
    # 在函数外用plt.figure()
    # 子图 1：原图
    ax1 = fig.add_subplot(121)
    ax1.imshow(image)
    ax1.set_title("Original Image")
    ax1.axis('off')

    # 子图 2：射线图
    ax2 = fig.add_subplot(122, projection='3d')
    np.random.shuffle(directions)
    pts1 = origins[::step]
    pts2 = origins[::step] + t * directions[::step]
    for p1, p2 in zip(pts1, pts2):
        ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], c='C0')

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title("Rays in 3D Space")
    ax2.set_xlim([-3,3])
    ax2.set_ylim([-3,3])
    ax2.set_zlim([0,3])

    


if __name__ == '__main__':
    from load_data import get_data
    imgs,poses,K = get_data(splits=['test'])
    H,W = imgs.shape[:2]
    # 挖掘机的爪子应该是沿着y轴负方向的
    fig = plt.figure()
    for i in range(0,200):
        plt.clf()
        o,dirs = get_rays(H,W,K[0,0],poses[i])
        plot_image_and_rays(imgs[i],o,dirs)
        plt.pause(0.01)