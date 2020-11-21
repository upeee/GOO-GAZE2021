from torchvision import transforms
import numpy as np
import random
from PIL import Image, ImageOps
from scipy import signal
import cv2
import torch

# data transform for image
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}


# generate a gaussion on points in a map with im_shap
def get_paste_kernel(im_shape, points, kernel, shape=(224 // 4, 224 // 4)):
    # square kernel
    k_size = kernel.shape[0] // 2
    x, y = points
    image_height, image_width = im_shape[:2]
    x, y = int(round(image_width * x)), int(round(y * image_height))
    x1, y1 = x - k_size, y - k_size
    x2, y2 = x + k_size, y + k_size
    h, w = shape
    if x2 >= w:
        w = x2 + 1
    if y2 >= h:
        h = y2 + 1
    heatmap = np.zeros((h, w))
    left, top, k_left, k_top = x1, y1, 0, 0
    if x1 < 0:
        left = 0
        k_left = -x1
    if y1 < 0:
        top = 0
        k_top = -y1

    heatmap[top:y2+1, left:x2+1] = kernel[k_top:, k_left:]
    return heatmap[0:shape[0], 0:shape[0]]



def gkern(kernlen=51, std=9):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d

kernel_map = gkern(21, 3)

def generate_data_field(eye_point):
    """eye_point is (x, y) and between 0 and 1"""
    height, width = 224, 224
    x_grid = np.array(range(width)).reshape([1, width]).repeat(height, axis=0)
    y_grid = np.array(range(height)).reshape([height, 1]).repeat(width, axis=1)
    grid = np.stack((x_grid, y_grid)).astype(np.float32)

    x, y = eye_point
    x, y = x * width, y * height

    grid -= np.array([x, y]).reshape([2, 1, 1]).astype(np.float32)
    norm = np.sqrt(np.sum(grid ** 2, axis=0)).reshape([1, height, width])
    # avoid zero norm
    norm = np.maximum(norm, 0.1)
    grid /= norm
    return grid

def preprocess_image(image_path, eye):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # crop face
    x_c, y_c = eye
    x_0 = x_c - 0.15
    y_0 = y_c - 0.15
    x_1 = x_c + 0.15
    y_1 = y_c + 0.15
    if x_0 < 0:
        x_0 = 0
    if y_0 < 0:
        y_0 = 0
    if x_1 > 1:
        x_1 = 1
    if y_1 > 1:
        y_1 = 1

    h, w = image.shape[:2]
    face_image = image[int(y_0 * h):int(y_1 * h), int(x_0 * w):int(x_1 * w), :]
    # process face_image for face net
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    face_image = Image.fromarray(face_image)
    face_image = data_transforms['test'](face_image)
    # process image for saliency net
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = data_transforms['test'](image)

    # generate gaze field
    gaze_field = generate_data_field(eye_point=eye)
    sample = {'image' : image,
              'face_image': face_image,
              'eye_position': torch.FloatTensor(eye),
              'gaze_field': torch.from_numpy(gaze_field)}

    return sample


