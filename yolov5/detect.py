import argparse
import time
from pathlib import Path

import cv2, imutils
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, non_max_suppression, apply_classifier, scale_coords, \
    xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from utils.overdetect_selection import select_highest_accuracy

from numpy import ones,vstack
from numpy.linalg import lstsq
import math
import pickle


def angle2(p1, p2, p3):
  x1, y1 = p1[0], p1[1]
  x2, y2 = p2[0], p2[1]
  x3, y3 = p3[0], p3[1]   
  if (x1==x2==x3 or y1==y2==y3):
    return 180
  else:
    dx1 = x2-x1
    dy1 = y2-y1
    dx2 = x3-x2
    dy2 = y3-y2
    if x1==x2:
      a1=90
    else:
      m1=dy1/dx1
      a1=math.degrees(math.atan(m1))
    if x2==x3:
      a2=90
    else:
      m2=dy2/dx2
      a2=math.degrees(math.atan(m2))
    angle = abs(a2-a1)
    return angle


def dot(vA, vB):
    return vA[0]*vB[0]+vA[1]*vB[1]
def ang(lineA, lineB):
    # Get nicer vector form
    vA = [(lineA[0][0]-lineA[1][0]), (lineA[0][1]-lineA[1][1])]
    vB = [(lineB[0][0]-lineB[1][0]), (lineB[0][1]-lineB[1][1])]
    # Get dot prod
    dot_prod = dot(vA, vB)
    # Get magnitudes
    magA = dot(vA, vA)**0.5
    magB = dot(vB, vB)**0.5
    # Get cosine value
    cos_ = dot_prod/magA/magB
    # Get angle in radians and then convert to degrees
    angle = math.acos(dot_prod/magB/magA)
    # Basically doing angle <- angle mod 360
    ang_deg = math.degrees(angle)%360

    if ang_deg-180>=0:
        # As in if statement
        return 360 - ang_deg
    else: 

        return ang_deg


def getAngle(a, b, c):
    
    #ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    #return ang + 360 if ang < 0 else ang
    
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)

def find_formula(p1,p2):
    points = [p1, p2]
    x_coords, y_coords = zip(*points)
    A = vstack([x_coords,ones(len(x_coords))]).T
    m, c = lstsq(A, y_coords, rcond=None)[0]
    return(m,c)

def find_intersection(m1,m2, b1, b2):
    xi = (b1-b2) // (m2-m1)
    yi = m1 * xi + b1
    return(int(xi),int(yi))

def line_ec(x,m,b):
    return(m*x + b)


def trans_img(imgg):

    kernel = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(imgg, cv2.MORPH_OPEN, kernel)
    Z = opening.reshape((-1,3))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((imgg.shape))


    # convert img to grayscale
    gray = cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)


    # do morphology gradient
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT , (2,2))
    morph = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)

    # apply gain
    morph = cv2.multiply(morph, 5)
    ret2,th = cv2.threshold(morph,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    
    #edges = cv2.Canny(th,100,200)

    return th


### function to find slope 
def slope(p1,p2):
    x1,y1=p1
    x2,y2=p2
    if x2!=x1:
        return((y2-y1)/(x2-x1))
    else:
        return 'NA'

### main function to draw lines between two points
def drawLine(image,p1,p2, color):
    x1,y1=p1
    x2,y2=p2
    ### finding slope
    m=slope(p1,p2)
    ### getting image shape
    h,w=image.shape[:2]

    if m!='NA':
        ### here we are essentially extending the line to x=0 and x=width
        ### and calculating the y associated with it
        ##starting point
        px=0
        py=-(x1-0)*m+y1
        ##ending point
        qx=w
        qy=-(x2-w)*m+y2
    else:
    ### if slope is zero, draw a line with x=x1 and y=0 and y=height
        px,py=x1,0
        qx,qy=x1,h
    cv2.line(image, (int(px), int(py)), (int(qx), int(qy)), color, 1)

#c of the for [ [[a1, b1]], [[a2, b2]], ..]
#y es el numero con el que queremos emparezar dado x fijo
def find_matching_pair(c, x, y):
    flat_list = [item for sublist in c for item in sublist.tolist()]
    if [x,y] in flat_list:
        return (x,y)
    else:
        return find_matching_pair(c, x, y-1)


def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    #colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    colors = [[156,39,176], [33,150,243], [76,175,88], [0,0,255], [244,67,54], [212,188,0], [255, 0, 255], [176,156,39]]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        cv2img = cv2.imread(path)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Delete duplicates selecting the one with the highest accuracy 
                det = select_highest_accuracy(det)
                
                for *xyxy, conf, cls in reversed(det):

                    xywh_values = [t.tolist() for t in xyxy]
                    x = int(xywh_values[0])
                    y = int(xywh_values[1])
                    w = int(xywh_values[2])
                    h = int(xywh_values[3])

                    # 骨软骨交界面 Osteochondral
                    if int(cls) == 4:
                        s4_x = x
                        s4_w = w
                    

                    # 软骨膜 Perichondrium
                    if int(cls) == 2:
                        
                        cropped = cv2img[y:h, x:w]

                        '''

                        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                        image = cv2.equalizeHist(gray)
                        kernel = np.ones((5, 5),np.uint8)
                        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=2)
                        image = cv2.dilate(image,kernel,iterations = 1)

                        image = cv2.GaussianBlur(image, (13, 13), 3)

                        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(image)
                        intensity = maxVal-minVal
                        #print('intensity = {}'.format(intensity))
                        #thres_val = intensity*0.85

                        if intensity == 196  or intensity == 227 or intensity == 217:
                            thres_val = intensity*0.77
                        else:
                            thres_val = intensity*0.85

                        ret,thresh1 = cv2.threshold(image, thres_val, 255, cv2.THRESH_BINARY)

                        '''

                        #th = thresh1
                        th = trans_img(cropped) #si no quiero el procesamiento anterior, directamente pasar cropped como input
                        cnts = cv2.findContours(th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cnts = imutils.grab_contours(cnts) 
                        c = max(cnts, key=len) #cnts[-1] #max(cnts, key=cv2.contourArea)



                        extLeft = tuple(c[c[:, :, 0].argmin()][0])
                        shifted_extLeft = tuple([extLeft[0] + x, h])# extLeft[1] + y]) #h
                        c2 = shifted_extLeft
                        c7 = (shifted_extLeft[0] + 30, shifted_extLeft[1])

                        cv2.circle(im0, shifted_extLeft, 4, (183, 15, 245), -1)


                    # 髂骨最低点 Lowest_ilium
                    if int(cls) == 5:

                        cropped = cv2img[y:h, x:w]
                        th = trans_img(cropped)
                        #gray = cv2.cvtColor(cropped,cv2.COLOR_BGR2GRAY)
                        #ret2,th = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                        
                        cnts = cv2.findContours(th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cnts = imutils.grab_contours(cnts)
                        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)
                        c = max(cnts, key=len)

                        #cv2.imshow('res2',th)
                        #cv2.waitKey(0)
                        #cv2.destroyAllWindows()

                        extRight = tuple(c[c[:, :, 0].argmax()][0])
                        shifted_extRight = tuple([extRight[0] + x, extRight[1] + y])

                        c5 = shifted_extRight

                        cv2.circle(im0, shifted_extRight, 4, (0, 0, 255), -1)


                    # 髂骨平面 Ilium_plane
                    if int(cls) == 3:
                        
                        cropped = cv2img[y:h, x:w]


                        '''
                        '''
                        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                        image = cv2.equalizeHist(gray)
                        kernel = np.ones((5, 5),np.uint8)
                        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=2)
                        image = cv2.dilate(image,kernel,iterations = 1)
                        image = cv2.GaussianBlur(image, (13, 13), 3)

                        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(image)
                        intensity = maxVal-minVal
                        #print('intensity = {}'.format(intensity))
                        thres_val = intensity*0.99

                        ret,thresh1 = cv2.threshold(image, thres_val, 255, cv2.THRESH_BINARY)

                        th = thresh1 #trans_img(cropped) #si no quiero el procesamiento anterior, directamente pasar cropped como input
                        cnts = cv2.findContours(th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cnts = imutils.grab_contours(cnts) 
                        c = max(cnts, key=len)
                        '''
                        '''

                        #extRight = tuple(c[c[:, :, 0].argmax()][0])
                        extTop = tuple(c[c[:, :, 1].argmin()][0])

                        rightUpper = (extRight[0], extTop[1])
                        #cv2.circle(cropped, extRight, 4, (0, 0, 255), -1)
                        #right shitef point +20 ?
                        new_top = tuple([extTop[0], extTop[1]])
                        aux = new_top[0] + x
                        shifted_extTop = tuple([ aux + (w - aux) + 30, new_top[1] + y])
                        #shifted_rightUpper = (rightUpper[0] + x, rightUpper[1] + y)
                        c3 = shifted_extTop


                        #cv2.circle(im0, c3, 4, (209, 153, 229), -1)

                        #cv2.imshow('res2',thresh1)
                        #cv2.waitKey(0)
                        #cv2.destroyAllWindows()


                        th = trans_img(cropped)
                        #gray = cv2.cvtColor(cropped,cv2.COLOR_BGR2GRAY)
                        #ret2,th = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                        
                        cnts = cv2.findContours(th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cnts = imutils.grab_contours(cnts)
                        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)
                        c = max(cnts, key=len)

                        
                        # support line for line n1
                        p1 = tuple(c[c[:, :, 0].argmax()][0])[0]
                        p2 = (h-y) // 2
                        #p11, p22 = find_matching_pair(c, p1-1, p2)
                        
                        c6 = (p1 + x, p2 + y)


                        #cv2.circle(im0, c6, 4, (122, 0, 255), -1)

                        #cv2.imshow('res2',th)
                        #cv2.waitKey(0)
                        #cv2.destroyAllWindows()
                        


                    # 盂唇 Labrum
                    if int(cls) == 0:
                        
                        cropped = cv2img[y:h, x:w]
                        th = trans_img(cropped)
                        #gray = cv2.cvtColor(cropped,cv2.COLOR_BGR2GRAY)
                        #ret2,th = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                        
                        cnts = cv2.findContours(th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cnts = imutils.grab_contours(cnts)
                        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)
                        c = max(cnts, key=len)

                        #cv2.imshow('res2',th)
                        #cv2.waitKey(0)
                        #cv2.destroyAllWindows()

                        extLeft = tuple(c[c[:, :, 0].argmin()][0])
                        extRight = tuple(c[c[:, :, 0].argmax()][0])
                        extTop = tuple(c[c[:, :, 1].argmin()][0])
                        extBot = tuple(c[c[:, :, 1].argmax()][0])

                        #a = extRight[0] - extLeft[0]
                        #b = extBot[1] - extTop[1]

                        a = (w-x)//2 
                        b = (h-y)//2

                        middlex = x + (a//2)#extLeft[0] + (a//2)
                        middle = (middlex, y + (b//2)) #tuple([middlex, extTop[1] + (b//2)])

                        # projection over the original pic
                        # shifted 5pxl right and down
                        shifted_middle = tuple([a + x + 2, b + y + 2]) 
                        c1 = shifted_middle
                        # then draw it on im0 instead of cropped

                        cv2.circle(im0, shifted_middle, 4, (255, 0, 0), -1)

                    
                    if int(cls) == 3:
                        
                        cropped = cv2img[y:h, x:w]
                        th = trans_img(cropped)
                        #gray = cv2.cvtColor(cropped,cv2.COLOR_BGR2GRAY)
                        #ret2,th = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                        
                        cnts = cv2.findContours(th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cnts = imutils.grab_contours(cnts)
                        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)
                        c = max(cnts, key=len)

                        extBot = tuple(c[c[:, :, 1].argmax()][0])
                        extRight = tuple(c[c[:, :, 0].argmax()][0])

                        extBot = tuple([extRight[0] ,extBot[1]])
                        shifted_extBot = tuple([extBot[0] + x, extBot[1] + y])

                        c4 = shifted_extBot

                        cv2.circle(im0, shifted_extBot, 4, (0, 255, 0), -1)

                        #cv2.imshow("ROI", th)
                        #cv2.waitKey()
                        cv2.imwrite("ROI.jpg", trans_img(cv2img))


                        

                    #im0 = cv2img
                
                    '''
                    if int(cls) == 0:
                        xywh_values = [t.tolist() for t in xyxy]
                        x = int(xywh_values[0])
                        y = int(xywh_values[1])
                        w = int(xywh_values[2])
                        h = int(xywh_values[3])
                        
                        cropped = cv2img[y:h, x:w]
                        gray = cv2.cvtColor(cropped,cv2.COLOR_BGR2GRAY)
                        ret2,th = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                        
                        cnts = cv2.findContours(th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cnts = imutils.grab_contours(cnts)
                        c = max(cnts, key=cv2.contourArea)

                        extLeft = tuple(c[c[:, :, 0].argmin()][0])
                        extRight = tuple(c[c[:, :, 0].argmax()][0])
                        extTop = tuple(c[c[:, :, 1].argmin()][0])
                        extBot = tuple(c[c[:, :, 1].argmax()][0])

                        #cir_position = (x//2 + w//2, y//2 + h//2)
                        #cir_color = (255, 0, 0)
                        #cv2.circle(cropped, cir_position, 5, cir_color, -1)

                        #cv2.drawContours(cropped, [c], -1, (0, 255, 255), 2)
                        cv2.circle(cropped, extLeft, 4, (0, 0, 255), -1)
                        cv2.circle(cropped, extRight, 4, (0, 255, 0), -1)
                        cv2.circle(cropped, extTop, 4, (255, 0, 0), -1)
                        cv2.circle(cropped, extBot, 4, (255, 255, 0), -1)

                        #cv2.imshow("ROI", cropped)
                        cv2.imwrite("ROI.jpg", cv2img)
                        #cv2.waitKey()
                    '''


                '''
                BETA ANGLE
                '''
                ##### Add bias in the x-axis to make the line longer
                projected_c3 = (s4_x + (s4_w-s4_x)//2, c3[1])
                cv2.line(im0, c2, projected_c3, (0, 255, 255), 2) #===================================> L3
                ##########################################################################################################
                ##### Still adding bias, but this time it's more complex, we need to calculate the line equation #########
                m, b = find_formula(c4,c1)
                new_y = line_ec(c5[0]+30, m, b)
                projected_c1 = (c5[0]+30, int(new_y))
                cv2.line(im0, c4, projected_c1, (0, 255, 255), 2) #====================================> L2


                ###########################################################################################################
                ############################################ Get angle ####################################################
                m1, b1 = find_formula(c2,  projected_c3)
                m2, b2 = find_formula(c4,  projected_c1)

                (xi, yi) = find_intersection(m1, m2, b1, b2)
                #################################################cv2.circle(im0, (xi, yi), 4, (255, 0, 255), -1)

                #shift = int(math.hypot(xi - c1[0], yi - c1[1]))
                shift = 30

                nyi = line_ec(xi+shift, m1, b1)
                projected_point = (xi+shift, int(nyi))
                #################################################cv2.circle(im0, projected_point, 4, (255, 0, 255), -1)


                line1 = (projected_c1, (xi, yi))
                line2 = (projected_point, (xi, yi))
                angle = ang(line1, line2)
                #angle = angle2(projected_point, (xi, yi), c1)
                #angle = getAngle(projected_point, (xi, yi),  c1)
                print('\nThe angle beta is {}'.format(angle))


                '''
                ALPHA ANGLE
                '''
                drawLine(im0,c2, c7, (255, 255, 255))

                # Project the line
                m, b = find_formula(c5,c6)
                new_y = line_ec(c2[0], m, b)
                projected_c6 = (c2[0], int(new_y))

                cv2.line(im0, projected_c6, c5, (0, 255, 255), 2)  #=================================> L1

                ###########################################################################################################
                ############################################ Get angle ####################################################
                m1, b1 = find_formula(c2, c3)
                m2, b2 = find_formula(projected_c6, c5)


                (xi, yi) = find_intersection(m1, m2, b1, b2)
                #cv2.circle(im0, (xi, yi), 4, (255, 0, 255), -1)


                #shift = int(math.hypot(xi - c2[0], yi - c2[1]))

                # Project over x-axis
                #nyi = line_ec(xi-shift, m2, b2)
                #projected_point2 = (xi-shift, int(nyi))
                #cv2.circle(im0, projected_point2, 4, (255, 0, 255), -1)


                line1 = ((xi, yi), projected_c6)
                line2 = ((xi, yi), c2)
                angle = ang(line1, line2)
                #angle = angle2(projected_point2, (xi, yi), c2)
                #angle = getAngle(projected_point2, (xi, yi), c2)
                print('\nThe angle alpha is {}'.format(angle))

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    '''
                    Draw lines
                    '''
                    '''
                    drawLine(im0,cir_position3, cir_position4)
                    drawLine(im0,cir_position1, cir_position4)
                    drawLine(im0,cir_position2, cir_auxposition)
                    '''

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format                       
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image

                        label = f'{names[int(cls)]} {conf:.2f}'
                        #plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1) #line thickness

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    check_requirements()

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
