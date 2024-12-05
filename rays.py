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
    directions = np.stack([u-W/2,H/2-v,-f*np.ones_like(u)],axis=1)
    directions = (c2w[:3,:3] @ directions[...,None]).squeeze(-1)
    #把d变成方向向量
    # shape = [WxH,3]
    directions = directions / np.linalg.norm(directions,axis=1,keepdims = True)
    
    origins = np.tile(c2w[:3,-1],(directions.shape[0],1))
    
    return origins,directions

def plot_rays(origins,directions,t=1.):
    plt.figure('rays')
    ax = plt.axes(projection = '3d')
    pts1 = origins
    pts2 = origins + t*directions
    for p1,p2 in zip(pts1,pts2):
        plt.plot([p1[0],p2[0]],[p1[1],p2[1]],[p1[2],p2[2]],c='C0')

    plt.show()

import numpy as np
import matplotlib.pyplot as plt

def get_rays(H, W, f, c2w):
    """
    计算每个像素的光线起点和方向。
    """
    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    u, v = u.ravel(), v.ravel()

    # 像素坐标 -> 相机坐标
    directions = np.stack([u - W / 2, H / 2 - v, -f * np.ones_like(u)], axis=1)
    directions = (c2w[:3, :3] @ directions[..., None]).squeeze(-1)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    origins = np.tile(c2w[:3, -1], (directions.shape[0], 1))
    return origins, directions

def plot_image_and_rays(image, origins, directions, t=5.0, step=5000):
    """
    绘制图像和射线的联合显示。
    """
    fig = plt.figure()

    # 子图 1：原图
    ax1 = fig.add_subplot(121)
    ax1.imshow(image)
    ax1.set_title("Original Image")
    ax1.axis('off')

    # 子图 2：射线图
    ax2 = fig.add_subplot(122, projection='3d')
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
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    from load_data import get_data
    imgs,poses,K = get_data()
    H,W = imgs.shape[:2]

    # 挖掘机的爪子应该是沿着y轴负方向的
    for i in range(0,100,10):
        o,dirs = get_rays(H,W,K[0,0],poses[i])
        plot_image_and_rays(imgs[i],o,dirs)
pass