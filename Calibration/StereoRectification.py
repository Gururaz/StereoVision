import cv2
import numpy as np
from StereoVision.Read_Write import resize


class StereoRectification:
    def __init__(self, Data, calibration_para, stereo_calib_para):
        self._data = Data
        self._calibration_para = calibration_para
        self._stereo_calib_para = stereo_calib_para
        self.results = {
            'stereoMapL': None,
            'stereoMapR': None,
            'disparity_to_depth_mapping_matrix': [],
            'rectification_transform_R_C1': None,
            'rectification_transform_R_C2': None,
            'rectified_projection_matrix_P1': None,
            'rectified_projection_matrix_P2': None
        }
        self._rectify()

    def _rectify(self):
        rectifyScale = 0.25
        flags = cv2.CALIB_ZERO_DISPARITY
        rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R = cv2.stereoRectify(
            self._calibration_para['C1_camera_matrix'],
            self._calibration_para['C1_distortion_coeff'],
            self._calibration_para['C2_camera_matrix'],
            self._calibration_para['C2_distortion_coeff'],
            self._data.image_size,
            self._stereo_calib_para.result['rot'],
            self._stereo_calib_para.result['trans'],
            flags,
            rectifyScale, (0, 0))

        stereoMapL = cv2.initUndistortRectifyMap(self._calibration_para['C1_camera_matrix'],
                                                 self._calibration_para['C1_distortion_coeff'],
                                                 rectL,
                                                 projMatrixL,
                                                 self._data.image_size,
                                                 cv2.CV_32FC1)  # cv2.CV_16SC2

        stereoMapR = cv2.initUndistortRectifyMap(self._calibration_para['C2_camera_matrix'],
                                                 self._calibration_para['C2_distortion_coeff'],
                                                 rectR,
                                                 projMatrixR,
                                                 self._data.image_size,
                                                 cv2.CV_32FC1)  # cv2.CV_16SC2

        self.results.update(stereoMapL=stereoMapL, stereoMapR=stereoMapR, disparity_to_depth_mapping_matrix=Q,
                            rectification_transform_R_C1=rectL, rectification_transform_R_C2=rectR,
                            rectified_projection_matrix_P1=projMatrixL, rectified_projection_matrix_P2=projMatrixR)

    def rectification_check(self, Left_img, Right_img, boardSize, size_chessboard_squares_mm):

        Left_img = cv2.remap(Left_img, self.results['stereoMapL'][0], self.results['stereoMapL'][1], cv2.INTER_LINEAR,
                             cv2.BORDER_CONSTANT, 0)

        Right_img = cv2.remap(Right_img, self.results['stereoMapR'][0], self.results['stereoMapR'][1], cv2.INTER_LINEAR,
                              cv2.BORDER_CONSTANT, 0)

        grayL = cv2.cvtColor(Left_img, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(Right_img, cv2.COLOR_BGR2GRAY)

        font = cv2.FONT_HERSHEY_PLAIN
        fontScale = 4

        # Find all chessboard corners at subpixel accuracy
        boardSize = boardSize
        subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 10e-06)
        winSize = (11, 11)

        objp = np.zeros((1, boardSize[0] * boardSize[1], 3), np.float32)
        objp[0, :, :2] = np.mgrid[0:boardSize[0], 0:boardSize[1]].T.reshape(-1, 2)
        objp = objp * size_chessboard_squares_mm
        objectPoints = []
        imagePointsL = []
        imagePointsR = []
        slopes = []

        retL, cornersL = cv2.findChessboardCorners(grayL, boardSize,
                                                   cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        retR, cornersR = cv2.findChessboardCorners(grayR, boardSize,
                                                   cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

        if retR is True and retL is True:
            objectPoints.append(objp)
            cornersR = cv2.cornerSubPix(grayR, cornersR, winSize, (-1, -1), subpix_criteria)
            cornersL = cv2.cornerSubPix(grayL, cornersL, winSize, (-1, -1), subpix_criteria)
            imagePointsR.append(cornersR)
            imagePointsL.append(cornersL)

            # Get points in 4th row (vertical centre) and display them
            vis = np.concatenate((Left_img, Right_img), axis=1)
            cv2.line(vis, (0, 2500), (5328 * 2, 2500), (0, 255, 255), 20)
            for i in range(15, 20):
                x_l = int(round(imagePointsL[0][i][0][0]))
                y_l = int(round(imagePointsL[0][i][0][1]))
                cv2.circle(vis, (x_l, y_l), 7, (0, 255, 255), -1)
                x_r = int(round(imagePointsR[0][i][0][0] + Left_img.shape[1]))
                y_r = int(round(imagePointsR[0][i][0][1]))
                cv2.circle(vis, (x_r, y_r), 7, (0, 255, 255), -1)
                slope = (y_l - y_r) / (x_r - x_l)
                slopes.append(slope)
                cv2.line(vis, (x_l, y_l), (x_r, y_r), (0, 255, 255), 2)
                avg = sum(slopes) / len(slopes)
                print("Rectification check - remapped images slope : ", avg)
                cv2.putText(vis, 'Average slope ' + str(avg),
                            (vis.shape[1] // 3 - 500, (vis.shape[0] // 5) * 2 + i * 75), font, fontScale,
                            (0, 255, 255), 2, cv2.LINE_AA)
                cropped_vis = resize(vis, 800)
                cv2.imshow('Rectification check - remapped images', cropped_vis)
                k = cv2.waitKey(0)
                if k == ord("q"):
                    break
            cv2.destroyAllWindows()
        else:
            print("Rectification check is available only for chessboard pattern or corners not detected")
