import glob
import os

from utils.dataset_processing import grasp, image
import torch
import numpy as np
import random

class CornellDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper for the Cornell dataset.
    """

    def __init__(self, file_path,output_size=400,resize_size=224,**kwargs):
        """
        :param file_path: Cornell Dataset directory.
        :param kwargs: kwargs for GraspDatasetBase
        """
        super(CornellDataset, self).__init__(**kwargs)

        self.grasp_files = glob.glob(os.path.join(file_path, '*', 'pcd*cpos.txt'))
        self.grasp_files.sort()
        self.length = len(self.grasp_files)
        self.len = 600*self.length
        
        if self.length == 0:
            raise FileNotFoundError('No dataset files found. Check path: {}'.format(file_path))

        self.depth_files = [f.replace('cpos.txt', 'd.tiff') for f in self.grasp_files]
        self.rgb_files = [f.replace('d.tiff', 'r.png') for f in self.depth_files]
        
        self.output_size = output_size
        self.resize_size = resize_size
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        
        index = idx % self.length

        rotations = [0, np.pi / 2, 2 * np.pi / 2, 3 * np.pi / 2]
        rot = random.choice(rotations)
        
        zoom = np.random.uniform(0.5, 1.0)

        img = self.get_rgd(index,rot,zoom)
        gtbbs = self.get_gtbb(index,rot,zoom)
        gtbb = gtbbs[0]       

        bb = np.array([gtbb.center[0],gtbb.center[1],np.sin(2*gtbb.angle),np.cos(2*gtbb.angle),gtbb.length,gtbb.width])
        
        sample = {'img': torch.from_numpy(img), 'bb': torch.from_numpy(bb)}
        
        return sample
    
    def _get_crop_attrs(self, idx):
        gtbbs = grasp.GraspRectangles.load_from_cornell_file(self.grasp_files[idx])
        center = gtbbs.center
        left = max(0, min(center[1] - self.output_size // 2, 640 - self.output_size))
        top = max(0, min(center[0] - self.output_size // 2, 480 - self.output_size))
        return center, left, top

    def get_gtbb(self, idx, rot=0, zoom=1.0):
        gtbbs = grasp.GraspRectangles.load_from_cornell_file(self.grasp_files[idx])
        center, left, top = self._get_crop_attrs(idx)
        gtbbs.rotate(rot, center)
        gtbbs.offset((-top, -left))
        gtbbs.zoom(zoom, (self.output_size // 2, self.output_size // 2))
        gtbbs.resize(self.output_size,self.resize_size)
        return gtbbs

    def get_depth(self, idx, rot=0, zoom=1.0):
        depth_img = image.DepthImage.from_tiff(self.depth_files[idx])
        center, left, top = self._get_crop_attrs(idx)
        depth_img.rotate(rot, center)
        depth_img.crop((top, left), (min(480, top + self.output_size), min(640, left + self.output_size)))
        depth_img.normalise()
        depth_img.zoom(zoom)
        depth_img.resize((self.resize_size, self.resize_size))
        return depth_img.img

    def get_rgb(self, idx, rot=0, zoom=1.0, normalise=True):
        rgb_img = image.Image.from_file(self.rgb_files[idx])
        center, left, top = self._get_crop_attrs(idx)
        rgb_img.rotate(rot, center)
        rgb_img.crop((top, left), (min(480, top + self.output_size), min(640, left + self.output_size)))
        rgb_img.zoom(zoom)
        rgb_img.resize((self.resize_size, self.resize_size))
        if normalise:
            rgb_img.normalise()
            rgb_img.img = rgb_img.img.transpose((2, 0, 1))
        return rgb_img.img
    
    def get_rgd(self, idx, rot=0, zoom=1.0):
        
        depth_img = self.get_depth(idx,rot,zoom)
        rgb_img = self.get_rgb(idx,rot,zoom)
        rgb_img[2,:,:] = depth_img        
        return rgb_img
        