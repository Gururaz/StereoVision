import cv2
import numpy as np
from matplotlib import pyplot as plt


class StereoCalibration:
    # This step is performed to transformation between the two cameras and calculate Essential and Fundamental matrix

    def __init__(self, Data, cam_1, cam_2, calibration_data):
        self.data = Data
        self.calibration_data = calibration_data
        self.cam_1 = cam_1
        self.cam_2 = cam_2
        self.stereo_filtering_criteria = 1
        self._flags = 0
        self._flags |= cv2.CALIB_FIX_INTRINSIC # cv2.CALIB_USE_INTRINSIC_GUESS
        self.criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
        self.result = {
            'Stereo_Error': None,
            'rot': None,
            'trans': None,
            'essentialMatrix': None,
            'fundamentalMatrix': None
        }

    def stereo_calibrate(self, obj_points, cam1_img_points, cam2_img_points, gray_image_shape):

        retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix, rvecs, tvecs, perViewErrors = cv2.stereoCalibrateExtended(
            obj_points,
            cam1_img_points,
            cam2_img_points,
            self.calibration_data['C1_camera_matrix'],
            self.calibration_data['C1_distortion_coeff'],
            self.calibration_data['C2_camera_matrix'],
            self.calibration_data['C2_distortion_coeff'],
            gray_image_shape,  # .shape[::-1],
            self._flags,
            self.criteria_stereo)

        return retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix, perViewErrors

    def calibration_with_checkerboard(self, cam1_file_names, cam1_images, cam2_file_names, cam2_images,
                                      checker_board_size=(8, 5),
                                      size_chessboard_squares_mm=14, show=False):

        print("\nPerforming stereo calibration using checker board")

        # Creating vector to store vectors of 3D points for each checkerboard image
        obj_points = []

        # Creating vector to store vectors of 2D points for each checkerboard image
        cam1_img_points = []
        cam2_img_points = []

        # Defining the world coordinates for 3D points
        objp = np.zeros((1, checker_board_size[0] * checker_board_size[1], 3), np.float32)
        objp[0, :, :2] = np.mgrid[0:checker_board_size[0], 0:checker_board_size[1]].T.reshape(-1, 2)

        # specify the scale for each square in mm
        objp = objp * size_chessboard_squares_mm

        reprojection_error_per_Image = []

        for count, (frame1, frame2) in enumerate(zip(cam1_images, cam2_images)):
            gray_image1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray_image2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret1, corners1 = cv2.findChessboardCorners(gray_image1, checker_board_size,
                                                       cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
            ret2, corners2 = cv2.findChessboardCorners(gray_image2, checker_board_size,
                                                       cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

            # If desired number of corners are found in the image then ret = true
            if ret1 and ret2:
                # refining pixel coordinates for given 2d points.
                # Convolution size used to improve corner detection.
                conv_size = (13, 13)

                corners1 = cv2.cornerSubPix(gray_image1, corners1, conv_size, (-1, -1), self.criteria_stereo)
                corners2 = cv2.cornerSubPix(gray_image2, corners2, conv_size, (-1, -1), self.criteria_stereo)

                if show:
                    save_text = "Press key s to consider the image for calibration or Esc to skip the image"

                    frame1_draw = cv2.drawChessboardCorners(frame1.copy(), checker_board_size, corners1, ret1)
                    frame2_draw = cv2.drawChessboardCorners(frame2.copy(), checker_board_size, corners2, ret2)
                    # Draw and display the corners
                    img1 = cv2.resize(frame1_draw, (960, 540))
                    img2 = cv2.resize(frame2_draw, (960, 540))
                    combined = np.hstack((img1, img2))
                    cv2.putText(img=combined, text=save_text, org=(0, 500), fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                                fontScale=1, color=(0, 255, 0), thickness=5)
                    cv2.imshow("Left : " + self.cam_1 + " " + "Right : " + self.cam_2 + " - " + cam1_file_names[count],
                               combined)
                    # cv2.imshow(self.cam_1 + cam1_file_names[count], img1)
                    # cv2.imshow(self.cam_2 + cam2_file_names[count], img2)
                    k = cv2.waitKey(0)
                    if k == ord("s"):
                        cam1_img_points.append(corners1)
                        cam2_img_points.append(corners2)
                        obj_points.append(objp)
                    elif k == 27:
                        pass
                    cv2.destroyAllWindows()
                else:
                    cam1_img_points.append(corners1)
                    cam2_img_points.append(corners2)
                    obj_points.append(objp)

        print("\nEstimating stereo calibration...")
        print('\nImages available for stereo calibration : ', len(cam1_images))

        if (len(cam1_images) > 0):
            gray_image_shape = gray_image1.shape[::-1]
            retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix, perViewErrors = self.stereo_calibrate(
                obj_points, cam1_img_points, cam2_img_points, gray_image_shape)

            print("Stereo calibration RMSE :", retStereo)

            # Filtering inaccurate image points based on criteria and recalculating stereo calibration
            updated_cam1_img_points = []
            updated_cam2_img_points = []
            updated_obj_points = []
            updated_img_list = []

            filter_img_indices = np.argwhere(np.all(perViewErrors < self.stereo_filtering_criteria, axis=1)).tolist()
            filter_img_indices = [item for sublist in filter_img_indices for item in sublist]
            for i in filter_img_indices:
                updated_cam1_img_points.append(cam1_img_points[i])
                updated_cam2_img_points.append(cam2_img_points[i])
                updated_obj_points.append(obj_points[i])
                updated_img_list.append(cam1_file_names[i])

            print('\nImages considered for stereo calibration post filtering : ', len(updated_img_list))
            if (len(updated_img_list)>0):
                newRetStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, newRot, newTrans, essentialMatrix, fundamentalMatrix, newPerViewErrors = self.stereo_calibrate(
                    updated_obj_points, updated_cam1_img_points, updated_cam2_img_points, gray_image_shape)

                for error in newPerViewErrors:
                    reprojection_error_per_Image.append(error)

                reprojection_error_per_Image = list(map(tuple, reprojection_error_per_Image))
                cam1_reprojection_error_per_Image = [x for (x, y) in reprojection_error_per_Image]
                cam2_reprojection_error_per_Image = [y for (x, y) in reprojection_error_per_Image]
                X_axis = np.arange(len(updated_img_list))
                plt.figure(figsize=(10, 5))
                plt.bar(X_axis - 0.1, cam1_reprojection_error_per_Image, 0.2, label='Cam1')
                plt.bar(X_axis + 0.1, cam2_reprojection_error_per_Image, 0.2, label='Cam2')
                plt.axhline(y=newRetStereo, color='r', linestyle='--')
                # handles = ['Overall Mean error ' + str(round(newRetStereo, 2)), 'Reprojection error per image']
                plt.xticks(X_axis, updated_img_list, rotation=45, ha="right")
                plt.xlabel("Calibration images")
                plt.ylabel("Reprojection error (RMSE)")
                plt.title("Stereo calibration result")
                plt.legend()
                plt.show()

                print("Stereo calibration RMSE post filtering :", newRetStereo)
                print("\nRotation matrix from C1 to C2 : \n", newRot)
                print("\nTranslation vector from C1 to C2 : \n", newTrans)

                self.result.update(Stereo_Error=newRetStereo, rot=newRot, trans=newTrans, essentialMatrix=essentialMatrix,
                                   fundamentalMatrix=fundamentalMatrix)

                self.data.image_size = gray_image_shape

            else:
                print("No images available above threshold for calibration, increase threshold")
                quit()
        else:
            print("Images not found for calibration, check the path / extension in the configuration.py file")
            quit()