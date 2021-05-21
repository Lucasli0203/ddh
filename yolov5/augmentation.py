import os, cv2, shutil
import numpy as np
from PIL import Image
from PIL import ImageFilter

#path1 = r"C:\Users\Lucas\Documents\yolov5\training\data\originals\images" #load img
#path2 = r"C:\Users\Lucas\Documents\yolov5\training\data\originals\labels" #load label
#path3 = r"C:\Users\Lucas\Documents\yolov5\training\data\augmented\images" #save img
#path4 = r"C:\Users\Lucas\Documents\yolov5\training\data\augmented\labels" #save label

path1 = "/users/lgaray/yolov5/training/data/images/train" #load img
path2 = "/users/lgaray/yolov5/training/data/labels/train" #load label


listing = os.listdir(path1)
for imagefile in listing: 
    os.chdir(path1)
    label = os.path.splitext(imagefile)[0]
    '''
    PIL

    im=Image.open(imagefile)
    im=im.convert("RGB")
    if im is None:
        print ('Error opening image')
        break

    r,g,b=im.split()
    r=r.convert("RGB")
    g=g.convert("RGB")
    b=b.convert("RGB")
    im_blur=im.filter(ImageFilter.GaussianBlur)
    im_unsharp=im.filter(ImageFilter.UnsharpMask)

    r.save('r_'+imagefile)
    g.save('g_'+imagefile)
    b.save('b_'+imagefile)
    im_blur.save('bl_'+imagefile)
    im_unsharp.save('un_'+imagefile)
    '''

    '''
    CV2
    '''
    img = cv2.imread(imagefile,0)
    kernel = np.ones((5,5),np.uint8)
    if img is None:
        print('Error opening image')
        break

    # Filters    
    edges = cv2.Canny(img,100,200)
    erosion = cv2.erode(img,kernel,iterations = 1)
    gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

    # Save imgs
    #os.chdir(path3)
    cv2.imwrite('edge_'+imagefile, edges)
    cv2.imwrite('eros_'+imagefile, erosion)
    cv2.imwrite('grad_'+imagefile, gradient)

    # Save labels
    os.chdir(path2)
    shutil.copyfile(label+'.txt', 'edge_' + label + ".txt")
    shutil.copyfile(label+'.txt', 'eros_' + label + ".txt")
    shutil.copyfile(label+'.txt', 'grad_' + label + ".txt")