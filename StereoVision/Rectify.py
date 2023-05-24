import cv2
from matplotlib import pyplot as plt
import os
from StereoVision import Configuration as CF
# camera parameters to undistort and rectify
file_name = "stereo_calib_results.xml"
filepath = os.path.join(CF.Data.results_path, file_name)
cv_file = cv2.FileStorage()
cv_file.open(filepath, cv2.FILE_STORAGE_READ)

Left_Stereo_Map_x = cv_file.getNode("stereoMapL_x").mat()
Left_Stereo_Map_y = cv_file.getNode("stereoMapL_y").mat()
Right_Stereo_Map_x = cv_file.getNode("stereoMapR_x").mat()
Right_Stereo_Map_y = cv_file.getNode("stereoMapR_y").mat()

# cv_file.release()

# Undistort and rectify images
def undistortRectify(frame_1, frame_2, show=False):
    undistorted_img1 = cv2.remap(frame_1, Left_Stereo_Map_x,
                                 Left_Stereo_Map_y, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 0)

    undistorted_img2 = cv2.remap(frame_2, Right_Stereo_Map_x,
                                 Right_Stereo_Map_y, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 0)

    # Display two rectified images ####################################################################################
    if show:
        # plt.figure(figsize=(10, 5))
        # plt.subplot(1, 1, 1)
        # plt.imshow(frame_1, 'gray')
        # plt.subplot(1, 1, 2)
        # plt.imshow(frame_2, 'gray')
        # plt.suptitle("Original Images")
        # plt.show()

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        # plt.suptitle("Original Images", verticalalignment='top')

        axes[0, 0].set_title("Original Images")
        axes[0, 0].imshow(frame_1, cmap="gray")
        axes[0, 1].imshow(frame_2, cmap="gray")
        axes[0, 0].axhline(2000)
        axes[0, 1].axhline(2000)
        axes[0, 0].axhline(2700)
        axes[0, 1].axhline(2700)

        axes[1, 0].set_title("Rectified Images")
        axes[1, 0].imshow(undistorted_img1, cmap="gray")
        axes[1, 1].imshow(undistorted_img2, cmap="gray")
        axes[1, 0].axhline(2000)
        axes[1, 1].axhline(2000)
        axes[1, 0].axhline(2700)
        axes[1, 1].axhline(2700)
        plt.suptitle("Stereo Images")
        plt.show()

    return undistorted_img1, undistorted_img2
