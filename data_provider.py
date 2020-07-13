import torch
import rawpy
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as transforms
import os
from PIL import Image

class Data_Provider(Dataset):
    def __init__(self, base_path, txt_file, patch_size=512, train=True):

        super(Data_Provider, self).__init__()
        self.base_path = base_path
        self.patch_size = patch_size
        with open(txt_file, 'r') as txt:
            self.filename = txt.readlines()
        self.train = train

    def __getitem__(self, index):

        files = self.filename[index].split()
        files, gt_file = files[:-1], files[-1]
        raws = []
        for file in files:
            raws.append(torch.from_numpy(get_RAW_from_file(os.path.join(self.base_path, file))))
        raws = torch.stack(raws, dim=0)
        height, width = raws[0].size()
        max_crop_height = height - self.patch_size
        max_crop_width = width - self.patch_size
        crop_h, crop_w = np.random.randint(0, max_crop_height, 1)[0], np.random.randint(0, max_crop_width, 1)[0]
        train_data = raws[:, crop_h:crop_h+self.patch_size, crop_w:crop_w+self.patch_size]
       # if not self.train:
        #    return train_data
        in_img = 1
        if not self.train:
            in_img = get_sRGB_from_file(os.path.join(self.base_path, files[:-1][0]))
            in_img = in_img[crop_h:crop_h + self.patch_size, crop_w:crop_w + self.patch_size, :]
            in_img = Image.fromarray(in_img)
            in_img = transforms.ToTensor()(in_img)
        gt = get_sRGB_from_file(os.path.join(self.base_path, gt_file))
        gt = gt[crop_h:crop_h+self.patch_size, crop_w:crop_w+self.patch_size, :]
        gt = Image.fromarray(gt)
        # gt.show()
        gt = transforms.ToTensor()(gt)
        return train_data, gt, in_img

    def __len__(self):
        return len(self.filename)

    @staticmethod
    def get_rand_position(self):
        pass

def get_sRGB_from_file(filename):

    srgb = None
    try:
        raw = rawpy.imread(filename)
        srgb = raw.postprocess()
    except Exception:
        pass
    finally:
        return srgb

def get_RAW_from_file(filename):
    RAW = None
    try:
        raw = rawpy.imread(filename)

        bl = raw.black_level_per_channel
        RAW = raw.raw_image_visible.astype(np.float32)
        # color_desc = raw.color_desc.reshape(-1)  # color的顺序
        # color_pattern = str(raw.color_pattern, encoding='gbk')
        # for index, (x, y) in zip(range(4), [(0, 0), (0, 1), (1, 0), (1, 1)]):
        #     index_cur = color_pattern[index]
        #     raw_data[x::2, y::2] -= bl[index_cur]
        # RAW = np.clip(0, 65535)
    except Exception:
        pass
    finally:
        return RAW

if __name__ == '__main__':
    dp = Data_Provider('./', 'test.txt')
    dp = iter(dp)
    a,b = next(dp)
    print(a.size(), b.size())
