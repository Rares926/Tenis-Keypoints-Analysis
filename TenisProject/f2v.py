import os
import glob
import cv2
from tqdm import tqdm

def frames_to_video(frames_dir, video_path, fps=30):
    """
    Generates a .mp4 video from a directory of frames

    Args:
        frames_dir (str): the directory containing the frames
                          note that this and any subdirs be looked through recursively
        video_path (str): path to save the video
        fps (int): the frames per second to make the output video (default is 30)

    Returns:
        str: the output video path, or None if error
    """

    frames_dir = os.path.normpath(frames_dir)  # make the paths OS (Windows) compatible
    video_path = os.path.normpath(video_path)  # make the paths OS (Windows) compatible

    # add the .mp4 extension if it isn't already there
    if video_path[-4:] != ".mp4":
        video_path += ".mp4"

    # get the frame file paths
    for ext in [".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"]:
        files = glob.glob(frames_dir + "/**/*" + ext, recursive=True)
        if len(files) > 0:
            break

    # couldn't find any images
    if not len(files) > 0:
        print("Couldn't find any files in {}".format(frames_dir))
        return None

    # get first file to check frame size
    image = cv2.imread(files[0])
    height, width, _ = image.shape  # need to get the shape of the frames

    # sort the files alphabetically assuming this will do them in the correct order
    files.sort()

    # create the videowriter - will create an .mp4
    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (width, height))

    # load and write the frames to the video
    for filename in tqdm(files, desc="Generating Video {}".format(video_path)):
        image = cv2.imread(filename)  # load the frame
        video.write(image)  # write the frame to the video

    video.release()  # release the video

    return video_path


def main():
    print("Video to Images")
    frames_to_video(frames_dir="Z:/Disertatie/data/frames/V009.mp4/Points/0000006465/",
                    video_path="Z:/Disertatie/data/frames/V009.mp4/Points/0000006465",
                    fps=30)

if __name__ == '__main__':
    main()