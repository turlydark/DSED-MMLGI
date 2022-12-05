
import os
import cv2 as cv
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torch.utils.data.sampler import Sampler
from pathlib import Path
import cv2
import torch
from dataset.transform import xception_default_data_transforms

class read_mask_and_label_data(Dataset):

    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.root = root
        self.train = train
        self.labels = []
        # print(root)

        fake_path = os.path.join(root,'0')
        fake_path = Path(fake_path)
        fake_list = list(fake_path.glob('*.png'))
        images_fake_str = [str(x) for x in fake_list]
        labels_fake_str = [0 for x in fake_list]

        real_path = os.path.join(root,'1')
        real_path = Path(real_path)
        real_list = list(real_path.glob('*.png'))
        images_real_str = [str(x) for x in real_list]
        labels_real_str = [1 for x in real_list]

        self.labels = labels_real_str + labels_fake_str
        images_list_str = images_real_str + images_fake_str
        self.images = images_list_str
        self.transform = transform

        print("number of images is :", len(images_list_str))
        print("number of labels is :", len(self.labels))




    def __getitem__(self, item):
        # self.images 中只包括 0和1的人脸图片
        image_path = self.images[item]
        #image = cv2.imread(image_path) # 读到的是BGR数据，该方法不能读取带中文路径的图像数据，下行则可读取中文路径。
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), 1)  # 1：彩色；2：灰度
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转化为RGB，也可以用img = img[:, :, (2, 1, 0)]
        # 这时的image是H,W,C的顺序，因此下面需要转化为：C, H, W
        # image = torch.from_numpy(image).permute(2, 0, 1)
        #image = torch.from_numpy(image).permute(2, 0, 1) / 255  # 归一化[0, 1]才能与PIL读取的数据一致

        image = Image.fromarray(image)
        image = self.transform(image)
        # print(image_path)
        # exit(0)
        # 路径示例 “C:\Users\98755\Desktop\Deepfake-Manipulation-Region-Localization\FF++_manipulation_data\c23\deepfake\train\1\113_26.jpg”
        label = 1
        label_file = image_path.split('\\')[-2]
        if label_file == '1':
            label = 1
        elif label_file == '0':
            label = 0
        else:
            print('There is exist wrong label name !')
            exit(0)

        # 编辑mask image
        mask_image = None
        if label == 0:
            mask_image_name = image_path.split('\\')[-1]
            mask_image_path = os.path.join(self.root, 'mask')
            mask_image_path = os.path.join(mask_image_path, mask_image_name)

            mask_image = cv2.imdecode(np.fromfile(mask_image_path, dtype=np.uint8), 1)  # 1：彩色；2：灰度
            mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)  # 转化为RGB，也可以用img = img[:, :, (2, 1, 0)]
            mask_image = Image.fromarray(mask_image)
            mask_image = self.transform(mask_image)
            # print(mask_image.size())
            # torch.Size([3, 256, 256])
            # exit(0)
        elif label == 1:
            mask_image = np.zeros((3, 256, 256))
            mask_image = torch.tensor(mask_image)


        return image, label, mask_image

    def __len__(self):
        return len(self.images)

if __name__ == "__main__":
    train_path = r"C:\Users\98755\Desktop\Deepfake-Manipulation-Region-Localization\FF++_dataset_for_localization_with_openface\c23\df\train"
    val_path = r"C:\Users\98755\Desktop\Deepfake-Manipulation-Region-Localization\FF++_dataset_for_localization_with_openface\c23\df\val"
    batch_size = 16
    train_dataloader = DataLoader(read_mask_and_label_data(root=train_path, transform=xception_default_data_transforms['train']), batch_size=batch_size,
                                  shuffle=True, num_workers=8)
    train_dataset_size = len(train_dataloader.dataset)
    print(train_dataset_size)

    for image, label, mask_image in train_dataloader:
        print(label)
        print(image.size())
        print(mask_image.size())
        # print(mask_image)
        exit(0)