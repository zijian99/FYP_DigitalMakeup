# Procedure :

# (Step 1) Use test.py evaluate function to retrieve the facial segmentation mapping for both source and destination
# (Step 2) Use crop.py crop function to crop out the necessary parts that require color transfer, can refer to facial-parts-index notes for different index parts
# (Step 3) For each facial parts, run colotr transfer from source img to dest img, and remember to make sure the surroundings is in white color after color transfer
# (Step 4) Use mergeback.py to copy the transferred facial parts back to the original img and save the result

# TODO: 1)合理化function应该带的param(DONE) 
#       2)改 filepath(DONE)
#       3)让function可以选择哪一个facial part要color transfer
#       4)做一个interface?
#       5)整理code 和code file(DONE)


from test import evaluate
from crop import crop 
from reinhard_color_transfer import color_transfer_method
from merge import merge

import cv2



if __name__ == "__main__":

    # Variables
    anno_mask_path = './image_folder/anno_mask_img/'
    img_folder_path = './image_folder/test-img/'
    crop_save_path = './image_folder/crop_facial_parts/'
    res_save_path = './image_folder/res-img/result.png'
    source_img_name = '116' 
    dest_img_name = '5'


    # Run evaluate function to get all of the facial segmentation color mapping of the pictures in the dspth folder 
    evaluate(respth = anno_mask_path, dspth = img_folder_path, cp = '79999_iter.pth')

    # Refer to facial-parts-index-notes.txt to add in the color of required facial parts
    #  
    target_color = [ [255, 255, 85], [170, 0, 255], [0, 85, 255]]

    # Destination Image
    img = cv2.imread(img_folder_path + dest_img_name + '.jpg')
    dest_img = cv2.resize(img, (512,512), interpolation = cv2.INTER_AREA)

    # For each facial features (based on target color)
    for facial_anno_color in target_color:

        # Crop out facial features of SOURCE image
        source = crop(img_name = source_img_name, img_path = img_folder_path, anno_path = anno_mask_path,  save_path = crop_save_path + 'source.png', target_color = facial_anno_color)
        # Crop out facial features of DEST image
        dest = crop(img_name = dest_img_name, img_path = img_folder_path, anno_path = anno_mask_path,  save_path = crop_save_path + 'dest.png', target_color = facial_anno_color)

        # Perform Color Transfer
        result_img_parts = color_transfer_method(source_img=source,dest_img=dest)

        # Merge color transferred facial features back to the dest image
        dest_img = merge(img = dest_img, img_parts = result_img_parts, img_anno_path = anno_mask_path + dest_img_name + '_anno.png', result_save_path = res_save_path, target_color = facial_anno_color)


    # For showing initial source and dest img
    src_img = cv2.imread(img_folder_path + source_img_name + '.jpg')
    src_img = cv2.resize(src_img, (512,512), interpolation = cv2.INTER_AREA)
    img = cv2.resize(img, (512,512), interpolation = cv2.INTER_AREA)
    cv2.imshow(f"Source Image ", src_img)
    cv2.imshow(f"Destination Image ", img)

    # Final Result Showcase
    cv2.imshow(f"Final Result ", dest_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


