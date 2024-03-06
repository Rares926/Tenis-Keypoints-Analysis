import time
import torch
import cv2
import numpy as np
from torchvision import transforms
import os
from utils.datasets import letterbox
from utils.general  import non_max_suppression_kpt
from utils.plots    import output_to_keypoint, plot_skeleton_kpts, plot_one_box, xywh2xyxy

from TennisCourtDetector.tracknet import BallTrackerNet
import torch.nn.functional as F
from TennisCourtDetector.postprocess import postprocess, refine_kps
from TennisCourtDetector.homography import get_trans_matrix, refer_kps

import time
from typing import Tuple, List
import copy
import csv
import math

import numpy as np

def load_model(
        model_usage: str,
        device: str,
        weights_path: str
    ):
    """
    Return model for specific task, 
    either keypoints detection or court detection. 

    Args:
        model_usage (str): Task of the model.
        device (str): Either Cuda or Cpu.
        weights_path (str): Path to saved weights.
            Either model itself or stored weights dict.

    Returns:
        _type_: _description_
    """
    if model_usage == 'pose_player':
        weights = torch.load(weights_path, map_location=device)
        model = weights['model']
        model.float().eval()
        model.to(device)
    elif model_usage == 'court_detection':
        model = BallTrackerNet(out_channels=15)
        model.load_state_dict(
            torch.load(
                weights_path,
                map_location=device
                )
            )
        model.eval()
        model = model.to(device)

    return model

def get_scaled_point(
        points,
        index: int,
        default_point: List,
        scale_width: float,
        scale_height: float
        ):
    """
    Scale a point from 'points' using the given scale factors for width and height,
        or return a default point if the predicted point is None.
    """
    point = points[index]
    if point[0] is None or point[1] is None:
        return default_point
    else:
        return [point[0] * scale_width, point[1] * scale_height]

    
def scale_extract_line_coords(
        points,
        line_up_indices:Tuple = (4, 6),
        line_down_indices:Tuple = (5, 7),
        scale_width: float = 2.0,
        scale_height: float = 2.0
        ):
    """
    Extracts coordinates and scales them for two court lines

    Args:
        points: The list of points to process.
        line_up_indices(Tuple): Indices for 'line up' points.
        line_down_indices(Tuple): Indices for 'line down' points.
        scale_width(float): The scale factor for width.
        scale_height(float): The scale factor for height.
    Returns:
        (x,y,x,y): A tuple of scaled line coordinates.
    """
    xylup = get_scaled_point(points, line_down_indices[0], [390, 686], scale_width, scale_height)
    xyrup = get_scaled_point(points, line_down_indices[1], [1232, 705], scale_width, scale_height)
    xyldown = get_scaled_point(points, line_up_indices[0], [573, 264], scale_width, scale_height)
    xyrdown = get_scaled_point(points, line_up_indices[1], [1036, 264], scale_width, scale_height)

    return xylup, xyrup, xyldown, xyrdown

def initialize_video_capture_and_writer(file_name, base_path):
    """
    Initializes video capture for a video and writer for the output
      video containing player keypoints and court lines.

    Args:
        file_name(str): Name of the video file (without extension).
        base_path(str): Base path where the video file is located.
    Returns:
        (VideoCapture,VideoWriter): A tuple containing the VideoCapture and VideoWriter objects.
    """
    vid_path = f'{base_path}/{file_name}.mp4'

    # Initialize video capture
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {vid_path}")

    # Get frames per second and read the first frame
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    ret, frame = cap.read()
    if not ret:
        raise IOError("Cannot read the first frame from video")

    h, w, _ = frame.shape

    # Initialize video writer
    out = cv2.VideoWriter(f'{base_path}/{file_name}_all_one_script.mp4', 
                          cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 
                          fps, (w, h))

    return cap, out

def compute_closest_point(x0, y0, x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    if dx == dy == 0:  # The line segment is a point
        return x1, y1
    t = ((x0 - x1) * dx + (y0 - y1) * dy) / (dx**2 + dy**2)
    t = max(0, min(1, t))  # Clamp t to the [0, 1] range
    x = x1 + t * dx
    y = y1 + t * dy
    return x, y

def distance_point_to_line(x0, y0, x1, y1, x2, y2):
    """
    Calculate the shortest distance from a point to a line segment.

    The line is defined by two points (x1, y1) and (x2, y2).
    The point is given by the coordinates (x0, y0).

    Parameters:
    x0, y0 (float): Coordinates of the point.
    x1, y1, x2, y2 (float): Coordinates of the two points defining the line.

    Returns:
    float: The shortest distance from the point to the line.
    """
    numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    denominator = math.sqrt((y2 - y1)**2 + (x2 - x1)**2)
    distance = numerator / denominator
    return distance

def distance_point_to_point(point1, point2):
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

def distance_point_to_line_and_closest_point(x0, y0, x1, y1, x2, y2):
    """
    Calculate the shortest distance from a point to a line segment and find the closest point on the line.

    The line is defined by two points (x1, y1) and (x2, y2).
    The point is given by the coordinates (x0, y0).

    Parameters:
    x0, y0 (float): Coordinates of the point.
    x1, y1, x2, y2 (float): Coordinates of the two points defining the line.

    Returns:
    float, (float, float): The shortest distance from the point to the line, and the coordinates of the closest point on the line.
    """
    # Calculate the line parameters
    A = y2 - y1
    B = x1 - x2
    C = x2*y1 - x1*y2

    # Calculate the perpendicular projection of the point onto the extended line
    denom = A*A + B*B
    x = (B*(B*x0 - A*y0) - A*C) / denom
    y = (A*(-B*x0 + A*y0) - B*C) / denom

    # Check if the projected point is within the line segment
    dx1, dx2 = x - x1, x - x2
    dy1, dy2 = y - y1, y - y2
    if dx1 * dx2 <= 0 and dy1 * dy2 <= 0:
        # The projected point is within the line segment
        closest_point = (x, y)
    else:
        # The closest point is one of the endpoints
        dist1 = math.sqrt((x0 - x1)**2 + (y0 - y1)**2)
        dist2 = math.sqrt((x0 - x2)**2 + (y0 - y2)**2)
        if dist1 < dist2:
            closest_point = (x1, y1)
        else:
            closest_point = (x2, y2)

    # Calculate the distance from the point to the closest point on the line
    distance = math.sqrt((x0 - closest_point[0])**2 + (y0 - closest_point[1])**2)

    return distance, closest_point

def get_box_line_down_midpoint(box, as_int=True):

    x_top_left = box[0]
    x_bottom_right = box[2]
    y_bottom_right = box[3]
    midx = (x_top_left + x_bottom_right) / 2
    midy = y_bottom_right

    if as_int:
        return (int(midx), int(midy))
    else:
        return (midx, midy)

def pose_video(model, frame, l_up, l_down):
    # line_up, line_down scaled to 720,1280 (orig image size)
    mapped_img = frame.copy()
    # Letterbox resizing.
    # frame = frame
    img = letterbox(frame, (input_size, input_size), stride=64, auto=True)[0]
    # the img will be letterboxed up and down
    # transforming the img size from (720,1280) to (764,1280)
    # 24 gray pixels will be added to the first dimension up and down
    # 
    print(img.shape)
    # Scale coordinates with respect to letterbox resize 
    add_pixels = (img.shape[0]-frame.shape[0])/2

    l_up[0][1] = l_up[0][1] + add_pixels
    l_up[1][1] = l_up[1][1] + add_pixels
    l_down[0][1] = l_down[0][1] + add_pixels
    l_down[1][1] = l_down[1][1] + add_pixels

    mapped_img = cv2.line(
            mapped_img,
            (int(l_up[0][0]),int(l_up[0][1])),
            (int(l_up[1][0]),int(l_up[1][1])),
            color=(0, 255, 0),
            thickness=4)
        
    mapped_img = cv2.line(
            mapped_img,
            (int(l_down[0][0]),int(l_down[0][1])),
            (int(l_down[1][0]),int(l_down[1][1])),
            color=(0, 255, 0),
            thickness=4)

    # Convert the array to 4D.
    img = transforms.ToTensor()(img)
    # Convert the array to Tensor.
    img = torch.tensor(np.array([img.numpy()]))
    # Load the image into the computation device.
    img = img.to(device)
    
    # Gradients are stored during training, not required while inference.
    with torch.no_grad():
        t1 = time.time()
        output, lk = model(img)
        t2 = time.time()
        fps = 1/(t2 - t1)
        output = non_max_suppression_kpt(output, 
                                         0.02,    # Conf. Threshold. [0.02 before]
                                         0.7,    # IoU Threshold.
                                         nc=1,   # Number of classes.
                                         nkpt=17, # Number of keypoints.
                                         kpt_label=True)
        
        output_kpt = output_to_keypoint(output)


    # Change format [b, c, h, w] to [h, w, c] for displaying the image.
    nimg = img[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)

    min_dis_line_one = 1200
    min_dis_line_two = 1200

    idx_player1 = 0
    idx_player2 = 0

    line_midpoint_to_box_1 = None
    line_midpoint_to_box_2 = None
    
    for idx in range(output_kpt.shape[0]):
        # here use box coordinates to filter keypoints
        box = list(*xywh2xyxy(np.array(output_kpt[idx, 2:6].T)[None]))

        # if box bottom line midlle not closest to line don't show
        # the two computed lines for this frame
        line1 = np.array([l_up], dtype=np.float32)
        line2 = np.array([l_down], dtype=np.float32)

        midpoint = get_box_line_down_midpoint(box, as_int = False)


        # ! plot boxes to check if the boxes and the lines are compared on the same scale
        plot_one_box(box, mapped_img, color=(255, 0, 0))
        cv2.circle(mapped_img, (int(midpoint[0]), int(midpoint[1])), 5, (0,0,255), -1)

        # distance_to_line1_cv = abs(cv2.pointPolygonTest(line1, midpoint, True))
        # distance_to_line1_cv = abs(distance_point_to_line(
        #     midpoint[0], midpoint[1],l_up[0][0],l_up[0][1],l_up[1][0],l_up[1][1]))
        distance_to_line1_cv, line_seg_1 = distance_point_to_line_and_closest_point(
            midpoint[0], midpoint[1],l_up[0][0],l_up[0][1],l_up[1][0],l_up[1][1])
        
        # ! plot all shortest lines from midpoint to line 1
        cv2.circle(mapped_img, (int(line_seg_1[0]),int(line_seg_1[1])), 5, (0,0,255), -1)
        cv2.line(mapped_img, (int(midpoint[0]),int(midpoint[1])), (int(line_seg_1[0]),int(line_seg_1[1])), (255, 0, 255), 3) 

        if abs(distance_to_line1_cv) < min_dis_line_one:
            min_dis_line_one = abs(distance_to_line1_cv)
            idx_player1 = idx
            line_midpoint_to_box_1 = line_seg_1
            box_player_1_midpoint = midpoint

        # distance_to_line2_cv = abs(cv2.pointPolygonTest(line2, midpoint, True))
        # distance_to_line2_cv = abs(distance_point_to_line(
        #     midpoint[0], midpoint[1],l_down[0][0],l_down[0][1],l_down[1][0],l_down[1][1]))
        distance_to_line2_cv, line_seg_2 = distance_point_to_line_and_closest_point(
            midpoint[0], midpoint[1],l_down[0][0],l_down[0][1],l_down[1][0],l_down[1][1])
        
        # ! plot all shortest lines from midpoint to line 2
        cv2.circle(mapped_img, (int(line_seg_2[0]),int(line_seg_2[1])), 5, (0,0,255), -1)
        cv2.line(mapped_img, (int(midpoint[0]),int(midpoint[1])), (int(line_seg_2[0]),int(line_seg_2[1])), (0, 255, 255), 3) 


        if abs(distance_to_line2_cv) < min_dis_line_two:
            min_dis_line_two = abs(distance_to_line2_cv)
            idx_player2 = idx
            line_midpoint_to_box_2 = line_seg_2
            box_player_2_midpoint = midpoint

    # cv2.circle(mapped_img, (int(line_midpoint_to_box_1[0]),int(line_midpoint_to_box_1[1])), 5, (0,0,255), -1)
    # cv2.line(mapped_img, (int(box_player_1_midpoint[0]),int(box_player_1_midpoint[1])), (int(line_midpoint_to_box_1[0]),int(line_midpoint_to_box_1[1])), (255, 0, 255), 3) 
    

    # cv2.circle(mapped_img, (int(line_midpoint_to_box_2[0]),int(line_midpoint_to_box_2[1])), 5, (0,0,255), -1)
    # cv2.line(mapped_img, (int(box_player_2_midpoint[0]),int(box_player_2_midpoint[1])), (int(line_midpoint_to_box_2[0]),int(line_midpoint_to_box_2[1])), (255, 0, 255), 3) 
    
    # ! show the test   
    cv2.imshow('Check distance on cor image', mapped_img)

    # here rescale the coordinates for box and keypoints and return them 
    # use plot_one_box and plot_skeleton_points  in main

    # plot player one
    p1_box = list(*xywh2xyxy(np.array(output_kpt[idx_player1, 2:6].T)[None]))
    #! box coords rescaled to orig img size before returning
    # plot_one_box(p1_box, nimg, color=(0, 0, 255))  
    #! output_kpt rescaled to orig img size before returning 
    # plot_skeleton_kpts(nimg, mapped_img, input_size, output_kpt[idx_player1, 7:].T, 3)
    p1_box_rescaled = [
        p1_box[0],
        p1_box[1] - add_pixels,
        p1_box[2],
        p1_box[3] - add_pixels]
    
    output_kpt[idx_player1, 7:].T[1::3] -= add_pixels

    # plot player two
    p2_box = list(*xywh2xyxy(np.array(output_kpt[idx_player2, 2:6].T)[None]))
    # plot_one_box(p2_box, nimg, color=(0, 0, 255))
    # plot_skeleton_kpts(nimg, mapped_img, input_size, output_kpt[idx_player2, 7:].T, 3)
    p2_box_rescaled = [
        p2_box[0],
        p2_box[1] - add_pixels,
        p2_box[2],
        p2_box[3] - add_pixels
        ]

    boxes = [p1_box_rescaled, p2_box_rescaled]

    output_kpt[idx_player2, 7:].T[1::3] -= add_pixels

    return boxes, [output_kpt[idx_player1, 7:].T, output_kpt[idx_player2, 7:].T], fps

def extract_court_points(mth_frame, court_model):

    mth_frame = cv2.resize(mth_frame, (640, 360))

    # frame = cv2.resize(frame, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
    inp = (mth_frame.astype(np.float32) / 255.)
    inp = torch.tensor(np.rollaxis(inp, 2, 0))
    inp = inp.unsqueeze(0)

    t_start = time.time()
    out_m = court_model(inp.float().to(device))[0]
    fps = 1/(time.time()-t_start)

    pred = F.sigmoid(out_m).detach().cpu().numpy()

    points = []
    for kps_num in range(15):
        heatmap = (pred[kps_num] * 255).astype(np.uint8)
        x_pred, y_pred = postprocess(heatmap, low_thresh=170, max_radius=25, scale=1)
        if use_refine_kps and kps_num not in [8, 12, 9] and x_pred and y_pred:
            x_pred, y_pred = refine_kps(mth_frame, int(y_pred), int(x_pred))
        points.append((x_pred, y_pred))

    if use_homography:
        matrix_trans = get_trans_matrix(points)
        if matrix_trans is not None:
            points = cv2.perspectiveTransform(refer_kps, matrix_trans)
            points = [np.squeeze(x) for x in points]

    return points, fps

def save_datapoint_csv(
        all_frames_players_kp,
        out_name,
        base_path,
        kp_confidence=0.5):
    
    with open(base_path + "/" + str(out_name) + ".csv", 'w', newline='') as file:
        writer = csv.writer(file)
        for players_kp_frame in all_frames_players_kp:
            writer.writerow(players_kp_frame[0].reshape(-1, 3)[:, :2].flatten())

        writer.writerow([0.0]*34)

        for players_kp_frame in all_frames_players_kp:
            writer.writerow(players_kp_frame[1].reshape(-1, 3)[:, :2].flatten())

# - Arguments
# Change forward pass input size. (multiple of 32)
# input_size = (1280, 720) # ! trace error for this 
input_size = 1280 # 720 is not multiple of stride(34 or 64)
# so we will letterbox the image to be 1290 764

file_name = "0102"
base_path = 'Z:/Tenis-Keypoints-Analysis/data/New_set/V010.mp4/Combined/0102/'

use_refine_kps = True
use_homography = True

OUTPUT_WIDTH = 1280
OUTPUT_HEIGHT = 720
debug_tmp = 0
box_frame_distance_threshold = 35.0 # this is a tricky threshold 

if __name__ == '__main__':

    start_time = time.time()
    #---------------------------INITIALIZATIONS------------------------------------#

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print('Selected Device : ', device)

    pose_model = load_model(
        "pose_player",
        device,
        'yolov7/yolov7-w6-pose.pt'
        )
    cap, out = initialize_video_capture_and_writer(
        file_name, base_path
        )

    court_model = load_model(
        "court_detection",
        device,
        "yolov7/TennisCourtDetector/weights/model_tennis_court_det.pt"
        )

    all_processed_kp = []

    #------------------------------Script Processing--------------------------------#
    frame_number = 0
    anterior_player_boxes = []
    anterior_player_kp = []
    tmp_list_for_mid_1 = []
    tmp_list_for_mid_2 = []
    while True:

        # ! frame should be the output on which we visualize results 
        ret, frame = cap.read()
        # frame.shape (720, 1280, 3)
        if not ret:
            print('Unable to read frame. Exiting ..')
            break

        
        input_court_frame = frame.copy()
        # frame_copy.shape (720, 1280, 3)

        points, court_fps = extract_court_points(
            input_court_frame,
            court_model
            )
        
        xylup, xyrup, xyldown, xyrdown = scale_extract_line_coords(
            points,
            line_up_indices=(0,1),
            line_down_indices=(2,3),
            scale_width=frame.shape[1]/640,
            scale_height=frame.shape[0]/360
            )

        # - these line points are relative to 1280-720 image size 
        # - scale them accordingly in the future 
        line_up = [xylup, xyrup]
        line_down = [xyldown, xyrdown]
        
        ################# extract player boxes and keypoints

        tmp_line_up = copy.deepcopy(line_up)
        tmp_line_down = copy.deepcopy(line_down)
    
        # if debug_tmp in list(range(70)):
        #     print("hehe")

        # if debug_tmp == 80:
        #     print("hehe")


        debug_tmp += 1
        player_boxes, player_kp, pose_fps = pose_video(
            pose_model,
            input_court_frame,
            tmp_line_up,
            tmp_line_down)

        # * here add an edge case if anterior box mid is too distant to new box mid 
        # * new box = old box 

        if len(anterior_player_boxes) == 0:
            # * first frame we do not have an anterior box
            anterior_player_boxes = player_boxes
            # anterior_player_boxes[1] = player_boxes[1]

            anterior_player_kp = player_kp
            # anterior_player_kp[1] = player_kp[1]
        else:
            # for first player 
            mid_anterior_box_pl_1 = get_box_line_down_midpoint(anterior_player_boxes[0])
            mid_new_box_pl_1 = get_box_line_down_midpoint(player_boxes[0])
            if abs(distance_point_to_point(mid_anterior_box_pl_1, mid_new_box_pl_1)) > box_frame_distance_threshold:
                player_boxes[0] = anterior_player_boxes[0]
                player_kp[0] = anterior_player_kp[0]
            # tmp_list_for_mid_1.append(abs(distance_point_to_point(mid_anterior_box_pl_1, mid_new_box_pl_1)))

            mid_anterior_box_pl_2 = get_box_line_down_midpoint(anterior_player_boxes[1])
            mid_new_box_pl_2 = get_box_line_down_midpoint(player_boxes[1])
            if abs(distance_point_to_point(mid_anterior_box_pl_2, mid_new_box_pl_2)) > box_frame_distance_threshold:
                player_boxes[1] = anterior_player_boxes[1]
                player_kp[1] = anterior_player_kp[1]
            # tmp_list_for_mid_2.append(abs(distance_point_to_point(mid_anterior_box_pl_2, mid_new_box_pl_2)))

        anterior_player_boxes = player_boxes
        anterior_player_kp = player_kp
        # ! TEMPORARY DISPLAY BOXES MIDPOINT TO CHECK IF THEY ARE COMPUTED CORRECTLY

        player_1_box_mp = get_box_line_down_midpoint(player_boxes[0])
        cv2.circle(frame, player_1_box_mp, 8, (0, 0, 255), -1)

        player_2_box_mp = get_box_line_down_midpoint(player_boxes[1])
        cv2.circle(frame, player_2_box_mp, 8, (0, 0, 255), -1)
        

##########################################################
        frame = cv2.line(
            frame,
            (int(xylup[0]),int(xylup[1])),
            (int(xyrup[0]),int(xyrup[1])),
            color=(0, 255, 0),
            thickness=4)
        
        frame = cv2.line(
            frame,
            (int(xyldown[0]),int(xyldown[1])),
            (int(xyrdown[0]),int(xyrdown[1])),
            color=(0, 255, 0),
            thickness=4)
        
        plot_one_box(player_boxes[0], frame, color=(255, 0, 0))
        plot_one_box(player_boxes[1], frame, color=(255, 0, 0))

        plot_skeleton_kpts(frame, player_kp[0], steps = 3, kp_confidence=0.0)
        plot_skeleton_kpts(frame, player_kp[1], steps = 3, kp_confidence=0.0)

        all_processed_kp.append(player_kp)
##############################################################        
        cv2.putText(frame,
                    'YoloV7 FPS : {:.2f}'.format(pose_fps),
                    (700, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, 
                    (0,255,0), 
                    2, 
                    cv2.LINE_AA)

        cv2.putText(frame,
                    'TrackNet FPS : {:.2f}'.format(court_fps),
                    (700, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, 
                    (0,255,0), 
                    2, 
                    cv2.LINE_AA)
        
        cv2.imshow('Proccesing video', frame)

        out.write(frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

        frame_number += 1
##############################################################   
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # ! save all the keipoints for each player nomatter confidence?
    save_datapoint_csv(
        all_processed_kp,
        kp_confidence=0.0,
        base_path=base_path,
        out_name=file_name)

    print("Video was processed in {} seconds".format(int(time.time()-start_time)))
