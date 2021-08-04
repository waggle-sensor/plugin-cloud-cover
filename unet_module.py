import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from unet import UNet


class Unet_Main:
    def __init__(self):
        self.net = UNet(n_channels=3, n_classes=1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.to(device=self.device)
        self.net.load_state_dict(torch.load('wagglecloud_unet_300.pth', map_location=self.device))
        self.net.eval()

    def preprocess(self, img_nd, scale, img_size=(300, 300), n_classes=2):
        img_nd = cv2.resize(img_nd, img_size)
        if len(img_nd.shape) == 2:
            # mask target image
            img_nd = np.expand_dims(img_nd, axis=2)
        else:
            # grayscale input image
            # scale between 0 and 1
            img_nd = img_nd / 255
        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))

        return img_trans.astype(float)


    def run(self, full_img,
                  scale_factor=1,
                  out_threshold=0.7):

        img = torch.from_numpy(self.preprocess(full_img, scale_factor))

        img = img.unsqueeze(0)
        img = img.to(device=self.device, dtype=torch.float32)


        with torch.no_grad():
            output = self.net(img)

            if self.net.n_classes > 1:
                probs = F.softmax(output, dim=1)
            else:
                probs = torch.sigmoid(output)

            probs = probs.squeeze(0)
            scores = probs.detach().cpu().numpy().reshape(-1)

            ##### mask
            for i in range(len(scores)):
                if scores[i] > out_threshold:
                    scores[i] = True
                else:
                    scores[i] = False
            #### end
            cloud = 0
            for i in scores:
                if i == 1:
                    cloud += 1
            ratio = cloud/len(scores)

        return ratio
