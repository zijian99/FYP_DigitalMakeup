'''
Author : ZJ

Color Transfer Algorithm 

'''
# ============================================================================================================================
# REINHARD COLOR TRANSFER METHOD
# 
# 
# ----------------------------------
#  STEPS FOR ALGORITHM             |
# ----------------------------------
# 
# 1. Read source and destination image via OpenCV2
# 2. Get the mean and standard deviation of all(RGB) color space from both source and destination image
# 3. Split destination image into R,G,B color space 
# 4. For each color space, do the following calculation :
#       RESULT = (DEST_COLOR_SPACE - DEST_COLOR_SPACE_MEAN)*(SRC_COLOR_SPACE_STD/DEST_COLOR_SPACE_STD) + SRC_COLOR_SPACE_MEAN
# 5. Combine the result color space R,G,B back into image and return the image
# 
# ============================================================================================================================


# Import Modules
import numpy as np
import cv2

# ----------------------------------------------------------------------------------------------------------------------------
# FUNCTION PART        
# ----------------------------------------------------------------------------------------------------------------------------

# Get the mean and standard deviation of respective color space of an image
def getColorSpaceMeanStdDev(image):
    
    # OpenCV2 function to get the mean and std dev
    x_mean,x_std = cv2.meanStdDev(image)

    # Split the array into respective color space var for easier usage
    r_mean, r_std = x_mean[0],x_std[0] 
    g_mean, g_std = x_mean[1],x_std[1] 
    b_mean, b_std = x_mean[2],x_std[2]

    return (r_mean, r_std, g_mean, g_std, b_mean, b_std)


# Main function to do the color transfer method from source image to destination image
def color_transfer(source, destination):

    # Convert source and destination image from BGR to RGB(for easier usage) and uint8 to float32(for doing calculation)
    source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB).astype("float32")
    destination = cv2.cvtColor(destination, cv2.COLOR_BGR2RGB).astype("float32")

    # Get the mean and std dev of all color space from the source and destination images
    (r_mean_src, r_std_src, g_mean_src, g_std_src, b_mean_src, b_std_src) = getColorSpaceMeanStdDev(source)
    (r_mean_dest, r_std_dest, g_mean_dest, g_std_dest, b_mean_dest, b_std_dest) = getColorSpaceMeanStdDev(destination)

    # Split the destination image into respective color space(RGB)
    r, g, b = cv2.split(destination)

    # Formula
    # RESULT = (DEST_COLOR_SPACE - DEST_COLOR_SPACE_MEAN)*(SRC_COLOR_SPACE_STD/DEST_COLOR_SPACE_STD) + SRC_COLOR_SPACE_MEAN

    # Subtract by mean of destination image
    r -= r_mean_dest
    g -= g_mean_dest
    b -= b_mean_dest

    # Scale by the standard deviations
    r = (r_std_src / r_std_dest) * r
    g = (g_std_src / g_std_dest) * g
    b = (b_std_src / b_std_dest) * b

    # Add the source mean
    r += r_mean_src
    g += g_mean_src
    b += b_mean_src

    # Round the pixel intensity
    r = np.round(r)
    g = np.round(g)
    b = np.round(b)
    
    # Clip the pixel intensities to [0, 255] if they fall outside this range
    r = np.clip(r, 0, 255)
    g = np.clip(g, 0, 255)
    b = np.clip(b, 0, 255)

    # Merge the color space back into image 
    result = cv2.merge([r, g, b])
    # Convert float32 back to uint8 format and RGB to BGR
    result = cv2.cvtColor(result.astype("uint8"), cv2.COLOR_RGB2BGR)

    # Return the result image
    return result


# ----------------------------------------------------------------------------------------------------------------------------



# ----------------------------------------------------------------------------------------------------------------------------
# MAIN CODE
# ----------------------------------------------------------------------------------------------------------------------------


def color_transfer_method(source_img,dest_img):
    # *************CHANGE THE PATH TO THE IMAGE HERE****************
    # source_path = "./image_folder/crop_facial_parts/source.png"
    # dest_path = "./image_folder/crop_facial_parts/dest.png"
    # **************************************************************

    # (Step 1) Read the source and destination image via OpenCV2
    # source = cv2.imread(source_path)
    # destination = cv2.imread(dest_path)
    

    # (Step 2,3,4,5) Transfer the color from source to destination
    result = color_transfer(source_img, dest_img)

    # # Show the source and destination image
    # cv2.imshow("Source Image", source)
    # cv2.imshow("Destination Image", destination)
    # # Show the result image for verification
    # cv2.imshow("Output Image", result)

    # # Waits for user to press any key
    # # (This is necessary to avoid Python kernel form crashing)
    # cv2.waitKey(0)
    
    # # Closing all open windows(After the key is pressed)
    # cv2.destroyAllWindows()




    # Save the result image for report
    output_name = "output.png"
    path_r = "./image_folder/color_transferred_img/"
    cv2.imwrite(path_r+output_name,result)

    return result
# ----------------------------------------------------------------------------------------------------------------------------
    
if __name__ == "__main__":
    source_path = "./image_folder/scenery/z.jpg"
    dest_path = "./image_folder/scenery/c.jpg"
    source = cv2.imread(source_path)
    destination = cv2.imread(dest_path)

    color_transfer_method(source, destination)