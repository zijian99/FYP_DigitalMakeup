'''
Author : ZJ

Code for cropping certain parts of image based on color mapping reference

'''

import cv2
import numpy as np



def crop_by_color(original_image, segmentation_mask, target_color):
    # Find pixels with the target color in the segmentation mask
    non_target_pixels = np.any(segmentation_mask != target_color, axis=-1)
    # print(non_target_pixels)
    original_image[non_target_pixels] = [255,255,255]

    return original_image



def crop(img_name, img_path, anno_path, save_path, target_color):
    
    # Load the original image and segmentation mask
    original_image = cv2.imread(img_path + img_name + '.jpg')
    original_image = cv2.resize(original_image, (512,512), interpolation = cv2.INTER_AREA)

    segmentation_mask = cv2.imread(anno_path + img_name + "_anno.png")

    # # Crop the original image based on the segmentation mask color
    cropped_images = crop_by_color(original_image, segmentation_mask, target_color)
    cv2.imwrite(save_path, cropped_images)

    return cropped_images



if __name__ == "__main__":
    crop_img = crop(img_path = './image_folder/test-img/', anno_path = './image_folder/anno_mask_img/', save_path = './image_folder/crop_facial_parts/', target_color = [0,85,255])