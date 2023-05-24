import glob
import os
import pickle
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from natsort import natsorted

from StereoVision import Configuration as CF


def readImages(path):
    images = []
    file_names = []
    files = glob.glob(path + '/*.png')
    files = natsorted(files)
    for file in files:
        images.append(cv2.imread(file))
        file_name = Path(file).stem
        file_names.append(file_name)
        # filename = ntpath.basename(file)
        # filename = os.path.splitext(filename)
    return file_names, images


def loadCameraParameters(file_name):
    file_path = os.path.join(CF.Data.results_path, file_name)
    file = open(file_path, 'rb')
    object_file = pickle.load(file)
    file.close()
    return object_file


def saveCameraParameters(file_name, results):
    filepath = os.path.join(CF.Data.results_path, file_name)
    file_handler = open((filepath + ".obj"), "wb")
    pickle.dump(results, file_handler)
    file_handler.close()


def saveCamResults(file_name, cam_1_calib_para, cam_2_calib_para):
    cam_calib_results = {
        "C1_camera_matrix": cam_1_calib_para.result['camera_matrix'],
        "C1_optimal_Camera_matrix": cam_1_calib_para.result['optimal_Camera_Matrix'],
        "C1_distortion_coeff": cam_1_calib_para.result['dist'],
        "C1_reprojection_error": cam_1_calib_para.result['calibration_error'],
        "C2_camera_matrix": cam_2_calib_para.result['camera_matrix'],
        "C2_optimal_Camera_matrix": cam_2_calib_para.result['optimal_Camera_Matrix'],
        "C2_distortion_coeff": cam_2_calib_para.result['dist'],
        "C2_reprojection_error": cam_2_calib_para.result['calibration_error']
    }
    filepath = os.path.join(CF.Data.results_path, file_name)
    with open((filepath + ".txt"), "w") as cf:
        cf.write(repr(cam_calib_results) + '\n')
        cf.write("\n")

    saveCameraParameters(file_name, cam_calib_results)

    return cam_calib_results


def saveStereoResults(file_name, stereo_calib_parameters, stereo_rect_parameters):
    stereo_calib_results = {
        "stereo_reprojection_error": stereo_calib_parameters.result['Stereo_Error'],
        "stereo_rotation_C1_C2": stereo_calib_parameters.result['rot'],
        "stereo_translation_C1_C2": stereo_calib_parameters.result['trans'],
        "Stereo_camera_essential_matrix": stereo_calib_parameters.result['essentialMatrix'],
        "Stereo_camera_fundamental_matrix": stereo_calib_parameters.result['fundamentalMatrix'],
        "Stereo_camera_disparity_to_depth_mapping_matrix": stereo_rect_parameters.results[
            'disparity_to_depth_mapping_matrix'],
        "Rectification_transform_R_C1": stereo_rect_parameters.results['rectification_transform_R_C1'],
        "Rectification_transform_R_C2": stereo_rect_parameters.results['rectification_transform_R_C2'],
        "Rectified_projection_matrix_P1": stereo_rect_parameters.results['rectified_projection_matrix_P1'],
        "Rectified_projection_matrix_P2": stereo_rect_parameters.results['rectified_projection_matrix_P2']
    }
    filepath = os.path.join(CF.Data.results_path, file_name)
    with open((filepath + ".txt"), "w") as sf:
        sf.write(repr(stereo_calib_results))

    saveCameraParameters(file_name, stereo_calib_results)

    cv2_file = cv2.FileStorage(filepath + '.xml', cv2.FILE_STORAGE_WRITE)
    stereoMapL = stereo_rect_parameters.results['stereoMapL']
    stereoMapR = stereo_rect_parameters.results['stereoMapR']
    cv2_file.write('stereoMapL_x', stereoMapL[0])
    cv2_file.write('stereoMapL_y', stereoMapL[1])
    cv2_file.write('stereoMapR_x', stereoMapR[0])
    cv2_file.write('stereoMapR_y', stereoMapR[1])
    cv2_file.release()

    return stereo_calib_results


def sharpen(image, show=False):
    kernel = np.array([[-2, -1, -2], [-2, 15, -2], [-2, -1, -2]])
    image_sharp = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
    if show:
        fig, axes = plt.subplots(1, 2, figsize=(15, 10))
        axes[0].set_title("Original Image")
        axes[0].imshow(image, cmap="gray")
        axes[1].set_title("Sharpened Image")
        axes[1].imshow(image_sharp, cmap="gray")
        plt.show()
    return image_sharp


def resize(image, height=None, width=None, interMethod=cv2.INTER_AREA):
    if height is None and width is None:
        return image

    (h, w) = image.shape[:2]

    if height is not None:
        ratio = height / float(h)
        dimension = (int(ratio * w), height)
    else:
        ratio = width / float(w)
        dimension = (width, int(ratio * h))

    resizeimage = cv2.resize(image, dimension, interpolation=interMethod)

    return resizeimage


# import cv2
# import os
# file_name = 'stereo_calib_results'
# filepath = os.path.join(CF.Data.results_path, file_name)
# cv2_file = cv2.FileStorage(filepath + '.xml', cv2.FILE_STORAGE_WRITE)
# cv2_file.release()