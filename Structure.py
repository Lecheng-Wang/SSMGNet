# encoding = utf-8

# @Author  ：Lecheng Wang
# @Time    : ${2025/7/15} ${17:23}
# @function: 模型预测时加载模型权重文件，预测结果生成


import torch
import torch.nn            as nn
import torch.nn.functional as F
import numpy               as np

from PIL                          import Image
from model.model_spatial_sfcnm    import SegModel


class Model:
    def __init__(self, model_path, bands, num_class):
        self.model_path = model_path
        self.bands      = bands
        self.num_class  = num_class
        self.cuda       = torch.cuda.is_available()
        self.device     = torch.device('cuda' if self.cuda else 'cpu')
        self.generate()

    def generate(self):
        self.model = SegModel(bands=self.bands, num_classes=self.num_class)

        try:
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device, weights_only=True))
            self.model = self.model.eval()
        except Exception as e:
            print(f"Error loading model weight file: {e}")
            raise e

        if self.cuda:
            #self.model = nn.DataParallel(self.model)
            #self.model = self.model.cuda()
            self.model = self.model.to(self.device)



    """Predict single image patch(256×256)"""
    def predict_small_patch(self, image):
        assert image.ndim == 3, f"input image dimension show be [C, H, W], but get {image.shape} instead."
        c, h, w      = image.shape
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)

        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            pr, feat, map = self.model(image_tensor)            # [1, num_classes, H, W]
            pr            = F.softmax(pr, dim=1)                # [1, num_classes, H, W]
            pr            = pr.argmax(dim=1)                    # [1, H, W]
            pr            = pr.squeeze(0).cpu().numpy()         # [H, W]
        
        map = map.squeeze(0).squeeze(0)
        return pr,map

    """Predict large image(divided into patchs to predict and confuse)"""
    def predict_large_image(self, image, tile_size=256, overlap=200):
        assert image.ndim == 3, f"input image dimension show be [C, H, W], but get {image.shape} instead."
        c, h, w         = image.shape
        stride          = tile_size - overlap
        outputs         = np.zeros((h, w), dtype=np.float64)
        weights         = np.zeros((h, w), dtype=np.float32)
        patches, coords = [], []
        for i in range(0, h, stride):
            for j in range(0, w, stride):
                x_end   = min(i     + tile_size, h)
                y_end   = min(j     + tile_size, w)
                x_start = max(x_end - tile_size, 0)
                y_start = max(y_end - tile_size, 0)

                patches.append(image[:, x_start:x_end, y_start:y_end])
                coords.append((x_start, x_end, y_start, y_end))

        for patch, (x_start, x_end, y_start, y_end) in zip(patches, coords):
            prediction,_ = self.predict_small_patch(patch)
            outputs[x_start:x_end, y_start:y_end] += prediction
            weights[x_start:x_end, y_start:y_end] += 1

        outputs = outputs / (weights + 1e-6)
        return np.round(outputs).astype(np.uint8)


    def get_small_predict_png(self, image):
        if image is None:
            raise ValueError("​​Image failed to load. Please check the file path or format.​")
        
        mask,_    = self.predict_small_patch(image)
        palette = {
            0: [0,   0,   0],       # background
            1: [218, 218, 218],     # class 1(Clean Glacier)
            2: [0,   153, 0],       # class 2(Debris Covered Glacier)
        }
        h, w = mask.shape
        rgb  = np.zeros((h, w, 3), dtype=np.uint8)
        for cls_id, color in palette.items():
            rgb[mask == cls_id] = color
        return Image.fromarray(rgb)


    def get_large_predict_png(self, image):
        if image is None:
            raise ValueError("​​Image failed to load. Please check the file path or format.​")
        
        mask    = self.predict_large_image(image)
        palette = {
            0: [0,   0,   0],       # background
            1: [218, 218, 218],     # class 1(Clean Glacier)
            2: [0,  153, 0],       # class 2(Debris Covered Glacier)
        }
        h, w = mask.shape
        rgb  = np.zeros((h, w, 3), dtype=np.uint8)
        for cls_id, color in palette.items():
            rgb[mask == cls_id] = color
        return Image.fromarray(rgb)



    def grad_cam(self, image, target_class):
        """
        Args:
            image: np.array [C, H, W]
            target_class: int
        Returns:
            cam: [H, W], normalized 0~1
        """
        self.model.eval()
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(self.device)

        # 前向
        seg, feat4, _ = self.model(image)
        
        # 只保留 grad
        feat4.retain_grad()
        
        # 选择目标类别
        target = seg[0, target_class]
        
        # 反向传播
        target.sum().backward()

        # feat4 的梯度和 feature 都在 device 上
        gradients   = feat4.grad[0]       # [C, H, W]
        activations = feat4[0]            # [C, H, W]

        alpha       = gradients.mean(dim=(1, 2))
        cam         = (alpha[:, None, None] * activations).sum(dim=0)
        cam         = torch.relu(cam)
        cam        -= cam.min()
        cam        /= (cam.max() + 1e-8)
        
        # 转 numpy
        return cam.detach().cpu().numpy()

