#!/usr/bin/python
# -*- encoding: utf-8 -*-

from pytorch_bisenet.logger import setup_logger
from pytorch_bisenet.model import BiSeNet

import torch

import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2

def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 85],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)
    required_facial_parts = [1,2,3,4,5,10,12,13,17]
    # for pi in range(1, num_of_class + 1):
    for pi in required_facial_parts:

        # FOR SHOWING EACH FACIAL PARTS
        # vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
        # vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
        # vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

        index = np.where(vis_parsing_anno == pi)
        # print(index)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi] 


        # FOR SHOWING EACH FACIAL PARTS
        # vis_parsing_anno_color[index[0], index[1], :] = [0,0,0]
        # vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
        # vis_im = cv2.cvtColor(vis_parsing_anno_color, cv2.COLOR_RGB2BGR)
        # cv2.imshow("test "+str(pi),vis_im)
        # cv2.waitKey(0) 
        # cv2.destroyAllWindows() 

        # print(vis_parsing_anno_color)


    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    cv2.imwrite(save_path[:-4]+'_anno' +'.png', vis_parsing_anno_color)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    # Save result or not
    if save_im:
        # cv2.imwrite(save_path[:-4]+'_segmented' +'.png', vis_parsing_anno)
        cv2.imwrite(save_path[:-4]+'_segmented' +'.png', vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    return vis_im

def evaluate(respth='./res/test_res', dspth='./data', cp='model_final_diss.pth'):

    if not os.path.exists(respth):
        os.makedirs(respth)

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    
    # net.cuda()
    net.cpu()

    save_pth = osp.join('model_pth', cp)
    net.load_state_dict(torch.load(save_pth, map_location=torch.device('cpu')))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        for image_path in os.listdir(dspth):
            img = Image.open(osp.join(dspth, image_path))
            image = img.resize((512, 512), Image.BILINEAR)
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.cpu()

            # img = img.cuda()
            out = net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            # print(parsing)
            # print(np.unique(parsing))

            vis_parsing_maps(image, parsing, stride=1, save_im=True, save_path=osp.join(respth, image_path))







if __name__ == "__main__":
    evaluate(respth = './image_folder/anno_mask_img/', dspth = './image_folder/test-img/', cp='79999_iter.pth')


