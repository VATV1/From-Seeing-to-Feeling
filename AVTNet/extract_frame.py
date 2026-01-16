""" The extract_frame.py is part of the data preprocessing of
our dataset that aims to transfer raw videos for both vision and
touch into static frames to support further applications. Specifically,
Two folders "video_frames" and "gelsight_frame" will be generated.
To run this code, please change the dir in line 57 to the path of our dataset.
"""

import os
from pathlib import Path
import numpy as np
import cv2

def getfiles(dir):
    filenames=os.listdir(dir)
    return filenames

def merge(dir):
    video1_dir = str(dir) + '/' + 'video.mp4'
    video2_dir = str(dir) + '/' + 'gelsight.mp4'

    cap1 = cv2.VideoCapture(video1_dir)
    frame_number1 = int(cap1.get(7))

    cap2 = cv2.VideoCapture(video2_dir)
    frame_number2 = int(cap2.get(7))
    
    frame_number1 = min(frame_number1, frame_number2)

    # if processed previously, skip this video
    if os.path.exists(str(dir) + '/video_frame'):
        return

    if not os.path.exists(str(dir) + '/video_frame'):
        os.makedirs(str(dir) + '/video_frame')
    if not os.path.exists(str(dir) + '/gelsight_frame'):
        os.makedirs(str(dir) + '/gelsight_frame')

    for i in range(frame_number1):
        cap1.set(cv2.CAP_PROP_POS_FRAMES, i)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, i)
        _, frame1 = cap1.read()
        _, frame2 = cap2.read()
        cv2.imwrite(str(dir) + '/video_frame/' + str(i).rjust(10,'0') + '.jpg', frame1)
        cv2.imwrite(str(dir) + '/gelsight_frame/' + str(i).rjust(10,'0') + '.jpg', frame2)

    cap1.release()
    cap2.release()


def main():
    dir = Path('/home/yzk/下载/touch_and_go/') # Path to dataset
    files = getfiles(dir)
    if '.DS_Store' in files:
        files.remove('.DS_Store')

    for i in range(len(files)):
        sub_dir = Path(str(dir) + '/' + (files[i]))
        print(str(sub_dir) + " Start!")
        merge(sub_dir)
        print(str(sub_dir) + " Finished!")


if __name__ == '__main__':
    main()
