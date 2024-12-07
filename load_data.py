import os,json
import imageio.v2 as imageio
import numpy as np
import matplotlib.pylab as plt

def get_data(splits = ['train','val','test']):
    basedir = os.path.join('data','lego')
    metas = {}
    for s in splits:
        with open(os.path.join(basedir,f'transforms_{s}.json'),'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits :
        meta = metas[s]
        imgs = []
        poses = []
        # 取出frame下的所有key
        for frame in meta['frames'][::]:
            fname = os.path.join(basedir,frame['file_path']+'.png')
            imgs.append(imageio.imread(fname)) # 4-d
            poses.append(np.array(frame['transform_matrix']))
        
        # rgba图像归一化，a代表不透明度 并且通通变成ndarry
        imgs = (np.array(imgs)/255).astype(np.float32)
        poses = np.array(poses).astype(np.float32)

        # 累计计数?
        counts.append(counts[-1]+imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)

    # 不同数据集里面的图像个数不一样，所以all_imgs列表是不均匀排布的，不能直接np.array
    imgs = np.concatenate(all_imgs,0)
    poses = np.concatenate(all_poses,0)
    '''
    plt.figure('lego')
    plt.imshow(imgs[10])
    plt.title('lego_image')
    plt.show()
    '''
    # 各个数据集角度都差不多，算一个
    camera_angle = meta['camera_angle_x']
    H,W = imgs[0].shape[0:2]
    f = 0.5*W / np.tan(0.5*camera_angle)
    intrinsic_matrix = np.array([[f,0,0.5*W],
                                [0,f,0.5*H],
                                [0,0,1]])
    return imgs,poses,intrinsic_matrix