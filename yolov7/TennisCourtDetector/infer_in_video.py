import os
import cv2
import numpy as np
import torch
from tracknet import BallTrackerNet
import torch.nn.functional as F
from tqdm import tqdm
from postprocess import postprocess, refine_kps
from homography import get_trans_matrix, refer_kps
import argparse
import copy 


if __name__ == '__main__':

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--model_path', type=str, help='path to model')
    # parser.add_argument('--input_path', type=str, help='path to input video')
    # parser.add_argument('--output_path', type=str, help='path to output video')
    # parser.add_argument('--use_refine_kps', action='store_true', help='whether to use refine kps postprocessing')
    # parser.add_argument('--use_homography', action='store_true', help='whether to use homography postprocessing')
    # args = parser.parse_args()

    use_refine_kps = True
    use_homography = True

    model_path = "yolov7/TennisCourtDetector/weights/model_tennis_court_det.pt"
    input_path = "Z:/Disertatie/data/New_set/V010.mp4/Combined/0028/0028.mp4"
    output_path = "Z:/Disertatie/data/New_set/V010.mp4/Combined/0028/0028_line_script.mp4"

    model = BallTrackerNet(out_channels=15)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    OUTPUT_WIDTH = 1280
    OUTPUT_HEIGHT = 720
    
    # Write video loader
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    ret, frame = cap.read()
    h, w, _ = frame.shape

    # Write video writer
    video_writer = cv2.VideoWriter(output_path, 
                          cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 
                          fps, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
    frame_id = 1
    while True:
        ret, frame = cap.read()
        # original_image = copy.deepcopy(frame)
        if not ret:
            print('Unable to read frame. Exiting ..')
            break

        input_frame = frame.copy()
        input_frame = cv2.resize(input_frame, (640, 360))

        # frame = cv2.resize(frame, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
        inp = (input_frame.astype(np.float32) / 255.)
        inp = torch.tensor(np.rollaxis(inp, 2, 0))
        inp = inp.unsqueeze(0)

        out = model(inp.float().to(device))[0]
        pred = F.sigmoid(out).detach().cpu().numpy()

        points = []
        for kps_num in range(15):
            heatmap = (pred[kps_num] * 255).astype(np.uint8)
            # ! THIS DOES NOT WORK WELL WHEN SCALE 2 
            # ALTOUGHT IN THE FUTURE IS BETTER TO RESCALE POINTS THAN
            # TO RESIZE THE IMAGE IN ORDER TO NOT LOSE IMAGE QUALITY
            x_pred, y_pred = postprocess(heatmap, low_thresh=170, max_radius=25, scale=1)
            if use_refine_kps and kps_num not in [8, 12, 9] and x_pred and y_pred:
                x_pred, y_pred = refine_kps(input_frame, int(y_pred), int(x_pred))
            points.append((x_pred, y_pred))

        if use_homography:
            matrix_trans = get_trans_matrix(points)
            if matrix_trans is not None:
                points = cv2.perspectiveTransform(refer_kps, matrix_trans)
                points = [np.squeeze(x) for x in points]

        for j in range(len(points)):
            if points[j][0] is not None:
                # here maybe use a resized version of the frame scale the coordinates 
                input_frame = cv2.circle(input_frame, (int(points[j][0]), int(points[j][1])),
                                  radius=0, color=(0, 0, 255), thickness=3)
        
        # draw lines 5-7, 4-6
        input_frame = cv2.line(
            input_frame,
            (int(points[5][0]),int(points[5][1])),
            (int(points[7][0]),int(points[7][1])),
            color=(0, 255, 0))
        input_frame = cv2.line(
            input_frame,
            (int(points[4][0]),int(points[4][1])),
            (int(points[6][0]),int(points[6][1])),
            color=(0, 255, 0) )

        cv2.imshow('Model input frame with court points', input_frame)  # stop showing it

        frame = cv2.line(
            frame,
            (int(points[5][0]*2),int(points[5][1]*2)),
            (int(points[7][0]*2),int(points[7][1]*2)),
            color=(0, 255, 0),
            thickness=4)
        frame = cv2.line(
            frame,
            (int(points[4][0]*2),int(points[4][1]*2)),
            (int(points[6][0]*2),int(points[6][1]*2)),
            color=(0, 255, 0),
            thickness=4)

        cv2.imshow('Original frame with court points', frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

        # if scale is one we have to resize it
        video_writer.write(frame)
        # video_writer.write(cv2.resize(frame, (OUTPUT_WIDTH, OUTPUT_HEIGHT)))
        print("Processed frame nr {}".format(frame_id))
        frame_id += 1

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

    # write_video(frames_upd, fps, output_path)
