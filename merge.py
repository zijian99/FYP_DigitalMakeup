'''
Author : ZJ

Code for merging a image part back to the original image

'''
import cv2
import numpy as np
from reinhard_color_transfer import color_transfer_method


def crop_by_color(original_image, segmentation_mask, target_color):
    # Find pixels with the target color in the segmentation mask
    non_target_pixels = np.any(segmentation_mask != target_color, axis=-1)
    original_image[non_target_pixels]= [255,255,255]

    return original_image



def merge(img, img_parts, img_anno_path, result_save_path, target_color):

    segmentation_mask = cv2.imread(img_anno_path)
    mask_image = crop_by_color(img_parts, segmentation_mask, target_color)

    background_color = [255,255,255]
    non_target_pixels = np.any(mask_image != background_color, axis = -1)
    img[non_target_pixels] = mask_image[non_target_pixels]

    # cv2.imshow(f"Result Image ", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    img = cv2.resize(img, (512,512), interpolation = cv2.INTER_AREA)
    cv2.imwrite(result_save_path, img)

    return img

if __name__ == "__main__":

    source_path = "./image_folder/crop_facial_parts/source.png"
    dest_path = "./image_folder/crop_facial_parts/dest.png"
    source = cv2.imread(source_path)
    destination = cv2.imread(dest_path)
    img_parts = color_transfer_method(source, destination)

    img = cv2.imread("./image_folder/test-img/5.jpg")
    img = cv2.resize(img, (512,512), interpolation = cv2.INTER_AREA)
    merge(img = img, img_parts = img_parts, target_color = [255,255,85])