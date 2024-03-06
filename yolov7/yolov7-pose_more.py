import time
import torch
import cv2
import numpy as np
from torchvision import transforms
import os
from utils.datasets import letterbox
from utils.general  import non_max_suppression_kpt
from utils.plots    import output_to_keypoint, plot_skeleton_kpts


def pose_video(frame):
    mapped_img = frame.copy()
    # Letterbox resizing.
    # frame = frame
    img = letterbox(frame, input_size, stride=64, auto=True)[0]
    print(img.shape)
    img_ = img.copy()
    # Convert the array to 4D.
    img = transforms.ToTensor()(img)
    # Convert the array to Tensor.
    img = torch.tensor(np.array([img.numpy()]))
    # Load the image into the computation device.
    img = img.to(device)
    
    # Gradients are stored during training, not required while inference.
    with torch.no_grad():
        t1 = time.time()
        output, _ = model(img)
        t2 = time.time()
        fps = 1/(t2 - t1)
        output = non_max_suppression_kpt(output, 
                                         0.02,    # Conf. Threshold.
                                         0.65,    # IoU Threshold.
                                         nc=1,   # Number of classes.
                                         nkpt=17, # Number of keypoints.
                                         kpt_label=True)
        
        output = output_to_keypoint(output)

    # Change format [b, c, h, w] to [h, w, c] for displaying the image.
    nimg = img[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)

    for idx in range(output.shape[0]):
        # plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)
        plot_skeleton_kpts(nimg, mapped_img, input_size, output[idx, 7:].T, 3)
        
    return mapped_img, nimg, fps


#------------------------------------------------------------------------------#
# Change forward pass input size.
input_size = 896

#---------------------------INITIALIZATIONS------------------------------------#

# Select the device based on hardware configs.
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
print('Selected Device : ', device)

# Load keypoint detection model.
weights = torch.load('yolov7/yolov7-w6-pose.pt', map_location=device)
model = weights['model']
# Load the model in evaluation mode.
_ = model.float().eval()
# Load the model to computation device [cpu/gpu/tpu]
model.to(device)


#--------------------------------------------------------------------------------#
# # Video capture and writer init.
# file_name = "0048_h" 
# vid_path = 'Z:/Disertatie/data/New_set/V010.mp4/Datapoints/datapoint_22/' + file_name + '.mp4'

# cap = cv2.VideoCapture(vid_path)
# fps = int(cap.get(cv2.CAP_PROP_FPS))
# ret, frame = cap.read()
# h, w, _ = frame.shape

# # May need to change the w, h as letterbox function reshapes the image.
# out = cv2.VideoWriter('Z:/Disertatie/data/New_set/V010.mp4/Datapoints/datapoint_22/' + file_name +"_kp"+'.mp4', 
#                        cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 
#                        fps, (w, h))
#-------------------------------------------------------------------------------#

if __name__ == '__main__':

    #--------------------------------------------------------------------------------#
    # Video capture and writer init.

    datapoints_path = "Z:/Disertatie/data/New_set/V010.mp4/Datapoints/"
    all_dp = os.listdir(datapoints_path)


    for dp_in in all_dp:
        for fn in os.listdir(os.path.join(datapoints_path,dp_in)):

            cap = cv2.VideoCapture(os.path.join(datapoints_path, dp_in, fn))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            ret, frame = cap.read()
            h, w, _ = frame.shape

            # May need to change the w, h as letterbox function reshapes the image.
            out = cv2.VideoWriter(os.path.join(datapoints_path, dp_in, fn[0:-4] + "_kp"+'.mp4') , 
                                cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 
                                fps, (w, h))
            #-------------------------------------------------------------------------------#

            while True:

                ret, frame = cap.read()
                
                if not ret:
                    print('Unable to read frame. Exiting ..')
                    break

                oimg, nimg, fps_ = pose_video(frame)

                cv2.putText(nimg,
                            'FPS : {:.2f}'.format(fps_),
                            (200, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, 
                            (0,255,0), 
                            2, 
                            cv2.LINE_AA)

                # cv2.imshow('Output', nimg[...,::-1])  #stop showing it
                out.write(cv2.resize(nimg[...,::-1], oimg.shape[0:2][::-1]))
                # key = cv2.waitKey(1)
                # if key == ord('q'):
                    # break

            cap.release()
            out.release()
            cv2.destroyAllWindows()