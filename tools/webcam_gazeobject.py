import argparse

import cv2
import torch

from mmdet.apis import inference_detector, init_detector, show_result

import numpy as np
import mmcv
from PIL import Image, ImageOps
from torch.nn import DataParallel
from torch.autograd import Variable

import sys
sys.path.append("..")
import time

from gazefollowing.utils import data_transforms
from gazefollowing.utils import get_paste_kernel, kernel_map
from gazefollowing.models.gazenet import GazeNet

def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection webcam demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file for object detection')
    parser.add_argument('--checkpoint_gaze', help='checkpoint file for gaze following')
    parser.add_argument('--device', type=int, default=0, help='CUDA device id')
    parser.add_argument(
        '--camera-id', type=int, default=0, help='camera device id')
    parser.add_argument(
        '--score-thr', type=float, default=0.5, help='bbox score threshold')
    parser.add_argument(
        '--out_res', nargs='+', type=float, default=[640,480], help='output resolution [width, height]')
    args = parser.parse_args()
    return args


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

def preprocess_image(image, eye):
    # image = cv2.imread(image_path, cv2.IMREAD_COLOR)

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

def test(net, image, eye):
    net.eval()
    heatmaps = []

    data = preprocess_image(image, eye)

    image, face_image, gaze_field, eye_position = data['image'], data['face_image'], data['gaze_field'], data['eye_position']
    image, face_image, gaze_field, eye_position = map(lambda x: Variable(x.unsqueeze(0).cuda(), volatile=True), [image, face_image, gaze_field, eye_position])

    _, predict_heatmap = net([image, face_image, gaze_field, eye_position])

    final_output = predict_heatmap.cpu().data.numpy()

    heatmap = final_output.reshape([224 // 4, 224 // 4])

    h_index, w_index = np.unravel_index(heatmap.argmax(), heatmap.shape)
    f_point = np.array([w_index / 56., h_index / 56.])


    return heatmap, f_point[0], f_point[1]

def draw_result(im, eye, heatmap, gaze_point, results_path='tmp.png'):
    x1, y1 = eye
    x2, y2 = gaze_point
    # im = cv2.imread(image_path)
    image_height, image_width = im.shape[:2]
    x1, y1 = image_width * x1, y1 * image_height
    x2, y2 = image_width * x2, y2 * image_height
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    cv2.circle(im, (x1, y1), 5, [255, 255, 255], -1)
    cv2.circle(im, (x2, y2), 5, [255, 255, 255], -1)
    cv2.line(im, (x1, y1), (x2, y2), [255, 0, 0], 3)

    # heatmap visualization
    heatmap = ((heatmap - heatmap.min()) / (heatmap.max() - heatmap.min()) * 255).astype(np.uint8)
    heatmap = np.stack([heatmap, heatmap, heatmap], axis=2)
    heatmap = cv2.resize(heatmap, (image_width, image_height))

    heatmap = (0.8 * heatmap.astype(np.float32) + 0.2 * im.astype(np.float32)).astype(np.uint8)
    img = np.concatenate((im, heatmap), axis=1)
    # cv2.imwrite(results_path, img)


    return img, heatmap, im

def init_gaze_model(model_path):
    net = GazeNet()
    net.cuda()

    # Load Model
    net = GazeNet()
    net.cuda()

    resume_path = './gazefollowing/saved_models/gazenet_goo/model_epoch25.pth.tar'
    net, optimizer, start_epoch = resume_checkpoint(net, None, resume_path)

    return net

def main():

    # print(h,w)
    args = parse_args()

    ###obj detection model
    model = init_detector(
        args.config, args.checkpoint, device=torch.device('cuda', args.device))

    ###gaze following model
    net = init_gaze_model(args.checkpoint_gaze)


    camera = cv2.VideoCapture(args.camera_id)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH,1920);
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT,1080);

    print('Press "Esc", "q" or "Q" to exit.')
    counter = 0
    start_time = time.time()
    while True:
        ret_val, img = camera.read()
        # img = cv2.resize(img, (853,h))
        # img = cv2.resize(img, (1080//2,1920//2))
        # img = np.copy(img[:640,:480,:])
        # img = np.copy(img[:640,:,:])
        # print(img.shape)
        result = inference_detector(model, img)
        # print(w,h)
        # print('head', result[-1])
        h,w,_ = img.shape
        if result[-1].any(): #if head exists
            if result[-1][0][4]>args.score_thr:
                x = np.mean([result[-1][0][0],result[-1][0][2]])/w
                y = np.mean([result[-1][0][1],result[-1][0][3]])/h
                heatmap, p_x, p_y = test(net, img, (x, y))

                pure_heatmap = heatmap.copy()

                _, heatmap,img = draw_result(img, (x, y), heatmap, (p_x, p_y))#, save_path)

                # retain only the most confident head
                result[-1] = np.array([result[-1][0].tolist()])


        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord('q') or ch == ord('Q'):
            break

        bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)

        labels = [
            np.full(bbox.shape[0], i, dtype=np.uint32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)

        # print(bboxes[0])
        bboxes[:,0] *= (args.out_res[0]/w)
        bboxes[:,2] *= (args.out_res[0]/w)
        bboxes[:,1] *= (args.out_res[1]/h)
        bboxes[:,3] *= (args.out_res[1]/h)
        mmcv.imshow_det_bboxes(
            cv2.resize(np.copy(img), (args.out_res[0], args.out_res[1])),
            bboxes,
            labels,
            # class_names=class_names,
            # score_thr=bboxes_ranked[1,-1] - 0.01,
            score_thr=args.score_thr,
            show=True,
            wait_time=1,
            win_name='selection',)
            # out_file=out_file)

        counter+=1
        timer = time.time() - start_time
        print("--- %f seconds, %f fps ---" % (timer, counter/timer))

if __name__ == '__main__':
    main()
