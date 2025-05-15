import cv2, torch, torchvision, math, shapely
import numpy as np
import torch.nn.functional as F
from shapely.geometry import Polygon, Point
from skimage import measure
import matplotlib.pyplot as plt



def circular_sector(center_x, center_y, radius, start_angle, end_angle, num_segments=36):
    angles = [start_angle + (float(i) / num_segments) * (end_angle - start_angle) for i in range(num_segments + 1)]

    points = []
    points.append((center_x, center_y))
    for angle in angles:
      rad = math.radians(angle)
      x = center_x + radius * math.cos(rad)
      y = center_y + radius * math.sin(rad)
      points.append((x, y))
    points.append((center_x, center_y))
    return Polygon(points)


def additional_information(copy_pred, pred):
    new_data = copy_pred.copy() * 255
    mask = np.asarray(np.where(new_data > 0)).T
    mask = [Point(x, y) for x, y in mask]

    w, h = new_data.shape
    r = int((w/2) * (3/4))
    circle = Point(int(w/2), int(h/2)).buffer(r, resolution=10)
    circlepolygon = Polygon(circle)
    true_count = sum(circlepolygon.contains(point) for point in mask)
    ratio = true_count / circlepolygon.area

    conf = torch.maximum(pred[0][0], pred[0][1]).detach().cpu().numpy()

    if conf.mean() > 0 and conf.mean() < 5 and true_count > 500:

        points = circular_sector(int(w/2), int(h/2), r, 360, 0, num_segments=8)
        coords = np.array(points.exterior.coords[1:-1])

        c = 0
        for i in range(len(coords)-1):

            triangle = np.array([[int(w/2), int(h/2)], coords[i], coords[i+1], [int(w/2), int(h/2)]])
            tripolygon = Polygon(triangle)
            true_count = sum(tripolygon.contains(point) for point in mask)

            if true_count > c:
                fpoints = []
                c = true_count

                tempimage = np.zeros((256, 256))
                if i < 2:
                    for k in range(128, 256):
                        for l in range(128):
                            if shapely.within(Point(l, k), tripolygon):
                                tempimage[k][l] = True

                elif i < 4:
                    for k in range(128):
                        for l in range(128):
                            if shapely.within(Point(l, k), tripolygon):
                                tempimage[k][l] = True

                elif i < 6:
                    for k in range(128):
                        for l in range(128, 256):
                            if shapely.within(Point(l, k), tripolygon):
                                tempimage[k][l] = True

                else:
                    for k in range(128, 256):
                        for l in range(128, 256):
                            if shapely.within(Point(l, k), tripolygon):
                                tempimage[k][l] = True


                fcontours = measure.find_contours(tempimage)
                for fcontour in fcontours:
                    fcoords = measure.approximate_polygon(fcontour, tolerance=2.5)
                    if len(fcoords) < 25:
                        pass
                    else:
                        fpoint = shapely.centroid(Polygon(fcoords))
                        fpoints.append(fpoint)

    return ratio, fpoints




class InferMain:
    def __init__(self):
        self.net = torch.load('new_cloudcover_unet.pth')
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.net.to(device=self.device).eval()
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((256, 256)),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

    def run(self, full_image):
        image = torch.from_numpy(np.asarray(full_image, dtype=np.float16).transpose(2, 0, 1))
        image_transformed = self.transform(image).type(torch.uint8)
        image = torchvision.io.decode_image(torchvision.io.encode_jpeg(image_transformed))


        with torch.no_grad():
            output = self.net(image.unsqueeze(0).to(self.device).float())

            copy_pred = output.argmax(axis=1).float()
            copy_pred = copy_pred.squeeze(0).detach().cpu().numpy()

            ratio, fpoint = additional_information(copy_pred, output)

            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(full_image)
            ax[0].axis('off')
            # ax[0].set_title('Input image')
            # ax[0].axis((0, 256, 0, 256))
            ax[0].axis((0, 256, 256, 0))
            ax[1].imshow(copy_pred*255)
            ax[1].axis('off')
            # ax[1].set_title('predicted')
            # ax[1].axis((0, 256, 0, 256))
            ax[1].axis((0, 256, 256, 0))

            fig.tight_layout()
            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()

        return ratio, fpoint, data
