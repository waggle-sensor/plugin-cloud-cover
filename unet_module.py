import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from unet import UNet

import math

def getInputPoint(x, y, srcwidth, srcheight):
    psph = []
    pfish = []

    FOV =float(math.pi/180 * 180)
    FOV2 = float(math.pi/180 * 180)
    width = srcwidth
    height = srcheight

    ## Polar angles
    theta = math.pi * (x / width - 0.5)   ## -pi/2 to pi/2
    phi = math.pi * (y / height - 0.5)    ## -pi/2 to pi/2

    ## Vector in 3D space
    psph.append(math.cos(phi) * math.sin(theta))   ## x
    psph.append(math.cos(phi) * math.cos(theta))   ## y
    psph.append(math.sin(phi) * math.cos(theta))   ## z

    ## Calculate fisheye angle and radius
    theta = math.atan2(psph[2],psph[0])
    phi = math.atan2(math.sqrt(psph[0]*psph[0]+psph[2]*psph[2]),psph[1])

    r = width * phi / FOV
    r2 = height * phi / FOV2

    ## Pixel in fisheye space
    pfish.append(0.5 * width + r * math.cos(theta))
    pfish.append(0.5 * height + r2 * math.sin(theta))

    return pfish

def dewarping(originalimage):
    h, w, c = originalimage.shape
    print(h,w)

    outimage = originalimage.copy()
    print(len(outimage), len(outimage[0]))
    for i in range(len(outimage)):
        for j in range(len(outimage[0])):
            inP = getInputPoint(i,j,h,w);
            inP2 = [int(inP[0]), int(inP[1])]

            if inP2[0] >= w or inP2[1] >= h:
                continue

            if inP2[0] < 0 or inP2[1] < 0:
                continue

            outimage[i][j][0] = originalimage[inP2[0]][inP2[1]][0]
            outimage[i][j][1] = originalimage[inP2[0]][inP2[1]][1]
            outimage[i][j][2] = originalimage[inP2[0]][inP2[1]][2]

    margin = 150
    outimage2 = outimage[margin:-margin, margin:-margin]

    return outimage2

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
                  out_threshold,
                  scale_factor=1):

        de_img = dewarping(full_img)
        img = torch.from_numpy(self.preprocess(de_img, scale_factor))

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
                    scores[i] = 255
                else:
                    scores[i] = 0
            #### end
            cloud = 0
            for i in scores:
                if i == 255:
                    cloud += 1
            ratio = cloud/len(scores)

            score2 = np.reshape(scores, (300,300))
            score3 = cv2.cvtColor(score2, cv2.COLOR_GRAY2BGR).astype(np.uint8)
            de = cv2.resize(de_img, (300,300))
            # full = cv2.resize(full_img, (300,300))
            hi = np.concatenate((de, score3), axis=1)
            # cv2.imshow('image', hi)
            # cv2.imshow('score2', score3)
            # cv2.imshow('full', full)

            # k = cv2.waitKey(0) & 0xFF
            # if k == ord('q'):
            #     return ratio, hi


        return ratio, hi
