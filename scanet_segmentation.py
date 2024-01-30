import torch

import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
import onnxruntime

#def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
def visualization(im, parsing_anno, stride=1, save_im=False, save_path = None, num_of_class = 20):
    # Colors for all 20 parts
    part_colors = [[0, 0, 0, 150], [255, 0, 0, 150], [255, 170, 0, 150],  
                   [0, 0, 255, 150], [255, 255, 0, 150], [0, 255, 255, 150],  
                   [255, 0, 255, 150], [255, 255, 170, 150], 
                   [255, 170, 255, 150], [170, 255, 255, 150],
                   [170, 170, 255, 150], [170, 255, 170, 150], [255, 170, 170, 150], 
                   [0, 85, 255, 150], [0, 102, 153, 150], 
                   [85, 255, 0, 150], [85, 0, 255, 150], [85, 255, 255, 150], 
                   [255, 0, 85, 150], [153, 255, 0, 150], [255, 190, 190, 150], [204, 255, 0, 150]]  

    # part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
    #                [255, 0, 85], [255, 0, 170],
    #                [0, 255, 0], [85, 255, 0], [170, 255, 0],
    #                [0, 255, 85], [0, 255, 170],
    #                [0, 0, 255], [85, 0, 255], [170, 0, 255],
    #                [0, 85, 255], [0, 170, 255],
    #                [255, 255, 0], [255, 255, 85], [255, 255, 85],
    #                [255, 0, 255], [255, 85, 255], [255, 170, 255],
    #                [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 4)) + 255 #473,473,3 흰색 바탕


    num_of_class = np.max(vis_parsing_anno)
    required_facial_parts = [1,2,3,4,5,10,12,13,17]

    # for pi in range(0, num_of_class):
    for pi in required_facial_parts:
        index = np.where(vis_parsing_anno == pi) #각 클래스에 해당하는 부분 추가.
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    cv2.imwrite('testing.png', vis_parsing_anno_color)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_BGR2BGRA), 0.6, vis_parsing_anno_color, 0.4, 0,dtype=cv2.CV_32F) #원본이미지에 합성
    vis_parsing_anno_num = vis_parsing_anno.max(axis=0)
    #cv2.imshow('vis_im',vis_im)
    #cv2.waitkey(0)
    # Save result or not
    if save_im:
        cv2.imwrite(save_path[:-4] +'.png', vis_parsing_anno_num)
        cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    # print(vis_im.shape)  # (256, 256, 4)
    vis_im = cv2.cvtColor(vis_im, cv2.COLOR_BGRA2RGB)
    vis_im = np.uint8(vis_im)
    # print(vis_im.shape)
    return vis_im



def evaluate(respth='./model_pth', dspth='./data', cp='/best_256.onnx'):
    image = cv2.imread("./image_folder/test-img/5.jpg")
    # print(image.shape)  # (256, 256, 3)
    image = cv2.resize(image, (256,256), interpolation = cv2.INTER_AREA)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
    ])

    image_trans = transform(image)

    image_trans = image_trans.unsqueeze(0)
    # print(image_trans.shape)  # torch.Size([1, 3, 256, 256])

    ort_session = onnxruntime.InferenceSession("./model_pth/best_256.onnx", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    interp = torch.nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)

    ort_inputs = {ort_session.get_inputs()[0].name: np.array(image_trans)}
    # print(ort_inputs['input'].shape)  # (1, 3, 256, 256)

    results = torch.from_numpy(ort_session.run(None, ort_inputs)[0])
    # results = results.squeeze(0)
    # print(results.shape)

    parsing = interp(results).data.cpu().numpy()
    parsing = parsing.transpose(0, 2, 3, 1)  # NCHW NHWC
    parsing_preds = np.asarray(np.argmax(parsing, axis=3))
    parsing_preds = parsing_preds.squeeze(0)
    parsing_preds = np.uint8(parsing_preds)

    color_img = visualization(image, parsing_preds)
    cv2.imshow(f"Final Result ", color_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == "__main__":
    evaluate(respth = './image_folder/testing_scanet/', dspth = './image_folder/test-img/', cp='best_256.onnx')