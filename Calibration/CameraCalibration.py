import cv2
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable

from StereoVision import Read_Write as RW


class CameraCalibration:

    def __init__(self, Data, camera):
        # Termination criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.camera = camera
        self.data = Data
        self._filtering_criteria = 0.6
        self.result = {
            'calibration_error': None,
            'camera_matrix': None,
            'optimal_Camera_Matrix': None,
            'dist': None,
            'ROI': None
        }

    def camera_calibration(self, obj_points, img_points, gray_image_shape):

        ret, cameraMatrix, dist, rvecs, tvecs, _, _, perViewErrors = cv2.calibrateCameraExtended(obj_points, img_points,
                                                                                                 gray_image_shape, None,
                                                                                                 None)

        height, width = gray_image_shape[:2]  # gray_image.shape[:2]
        optimal_Camera_Matrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, imageSize=(width, height),
                                                                   alpha=0.25, newImgSize=(width, height))
        # dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

        return ret, cameraMatrix, dist, rvecs, tvecs, _, _, perViewErrors, optimal_Camera_Matrix, roi

    def calibration_Checker_Board(self, file_names, images, checker_board_size, size_chessboard_squares_mm,
                                  show=False):

        print("\nPerforming camera calibration for " + self.camera + " using checker board")
        print("\nChess board corners :", checker_board_size)
        print("size of each squares :", size_chessboard_squares_mm)

        # Creating vector to store vectors of 3D points for each checkerboard image
        obj_points = []
        # Creating vector to store vectors of 2D points for each checkerboard image
        img_points = []

        reprojection_error_per_Image = []
        sharpness_error_per_Image = []
        # Defining the world coordinates for 3D points
        objp = np.zeros((1, checker_board_size[0] * checker_board_size[1], 3), np.float32)
        objp[0, :, :2] = np.mgrid[0:checker_board_size[0], 0:checker_board_size[1]].T.reshape(-1, 2)

        # specify the scale for each square in mm
        objp = objp * size_chessboard_squares_mm

        img_list = []

        for count, frame in enumerate(images):
            # assert frame.any() == 0, "file could not be read, check with os.path.exists()"
            # ret_CB = cv2.checkChessboard(frame, checker_board_size)
            # print(ret_CB)
            gray_img = frame  # RW.sharpen(img, show=True)
            gray_image = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)
            gray_image_shape = gray_image.shape[::-1]
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray_image, checker_board_size,
                                                     cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
            # cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FILTER_QUADS)
            # If desired number of corners are found in the image then ret = true
            if ret:
                # refining pixel coordinates for given 2d points.
                # Convolution size used to improve corner detection.
                conv_size = (13, 13)  # (5, 5)

                corners = cv2.cornerSubPix(gray_image, corners, conv_size, (-1, -1), self.criteria)
                # Try corners = cv2.find4QuadCornerSubpix(gray_image, corners)
                retval, sharpness = cv2.estimateChessboardSharpness(gray_img, checker_board_size, corners)
                # print(file_names[count] + " sharpness", retval[0])
                if show:
                    # Try corners = cv2.find4QuadCornerSubpix(gray_image, corners)
                    img_draw = cv2.drawChessboardCorners(gray_img, checker_board_size, corners)
                    # plt.imshow(img_draw, cmap="gray")
                    # plt.show()
                    save_text = "Press key s to consider the image for calibration or Esc to skip the image"
                    sharp_text = str(file_names[count]) + " sharpness : " + str(retval[0])
                    # Draw and display the corners
                    win_name = self.camera + " " + file_names[count]
                    cv2.putText(img=img_draw, text=save_text, org=(0, 500), fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                                fontScale=3,
                                color=(0, 255, 0), thickness=5)
                    cv2.putText(img=img_draw, text=sharp_text, org=(0, 700), fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                                fontScale=3,
                                color=(0, 255, 0), thickness=5)
                    # cv2.moveWindow(winname, 100, 30)
                    img_draw = RW.resize(img_draw, height=900, width=None, interMethod=cv2.INTER_LINEAR)
                    cv2.imshow(win_name, img_draw)
                    k = cv2.waitKey(0)
                    if k == ord("s"):
                        img_points.append(corners)
                        obj_points.append(objp)
                        img_list.append(file_names[count])
                        sharpness_error_per_Image.append(retval[0])
                    elif k == 27:
                        pass
                    cv2.destroyAllWindows()
                else:
                    img_points.append(corners)
                    obj_points.append(objp)
                    img_list.append(file_names[count])
                    sharpness_error_per_Image.append([retval[0]])

        print("\nEstimating camera calibration for " + self.camera + "...")
        print('\nImages available for calibration for ' + self.camera + ' : ', len(images))
        if len(images) > 0:
            print('\nImage size ', gray_image_shape)

            ret, cameraMatrix, dist, rvecs, tvecs, _, _, perViewErrors, optimal_Camera_Matrix, roi = self.camera_calibration(
                obj_points, img_points, gray_image_shape)

            print("camera calibration RMSE " + self.camera + " :", ret)

            # Filtering inaccurate image points based on criteria and recalculating camera calibration
            updated_img_points = []
            updated_obj_points = []
            updated_img_list = []
            updated_sharpness_error_per_Image = []

            filter_img_indices = np.argwhere(np.all(perViewErrors <= self._filtering_criteria, axis=1)).tolist()
            filter_img_indices = [item for sublist in filter_img_indices for item in sublist]

            for i in filter_img_indices:
                updated_img_points.append(img_points[i])
                updated_obj_points.append(obj_points[i])
                updated_img_list.append(file_names[i])
                updated_sharpness_error_per_Image.append(sharpness_error_per_Image[i])

            newRet, cameraMatrix, dist, rvecs, tvecs, _, _, perViewErrors, optimal_Camera_Matrix, roi = self.camera_calibration(
                updated_obj_points, updated_img_points, gray_image_shape)
            print("\nThreshold for reprojection error : " + str(self._filtering_criteria))
            print("\nImages considered for calibration post filtering, " + self.camera + " :", len(updated_img_list))
            print("camera calibration RMSE post filtering, " + self.camera + " :", newRet)

            for error in perViewErrors:
                reprojection_error_per_Image.append(error)

            print("Camera Matrix \n", cameraMatrix)
            print("Distortion Coefficient : ", dist)

            updated_sharpness_error_per_Image = [item for sublist in updated_sharpness_error_per_Image for item in
                                                 sublist]

            # Print of Reprojection error per image using PrettyTable
            print("\nReprojection error in pixels and sharpness value per image")
            ReprojTable = [updated_img_list, list(np.around(np.array(reprojection_error_per_Image), 2)),
                           list(np.around(np.array(updated_sharpness_error_per_Image), 2))]

            ReprojTab = PrettyTable(ReprojTable[0])
            ReprojTab.add_rows(ReprojTable[1:])
            # leftcol = ["Images", "Reproj", "Sharpness"]
            print(ReprojTab)

            # x = PrettyTable()
            # x.field_names = ["Images", ReprojTable[0]]
            # x.add_column(["Reproj", ReprojTable[1]])
            # x.add_column(["Sharpness", ReprojTable[2]])
            # print(x)

            error = np.concatenate(reprojection_error_per_Image)
            plt.figure(figsize=(10, 5))
            plt.bar(updated_img_list, error)  # color='b'
            plt.axhline(y=newRet, color='r', linestyle='--')
            handles = ['Overall Mean error ' + str(round(newRet, 2)), 'Reprojection error per image']
            plt.legend(handles)
            plt.xticks(rotation=45, ha="right")
            plt.xlabel("Calibration images")
            plt.ylabel("Reprojection error (RMSE)")
            plt.title("Calibration result for " + self.camera)
            plt.show()

            self.result.update(calibration_error=newRet,
                               camera_matrix=cameraMatrix,
                               optimal_Camera_Matrix=optimal_Camera_Matrix, dist=dist, ROI=roi)
            self.data.image_size = gray_image_shape
        else:
            print("No images available or unable to detect chessboard corners")

    def calibration_Radon(self, file_names, images, checker_board_size=(8, 5), size_chessboard_squares_mm=12.1,
                          show=False):
        print("\nPerforming camera calibration for " + self.camera + " using checker board")
        print("\nChess board corners :", checker_board_size)
        print("size of each squares :", size_chessboard_squares_mm)

        # Creating vector to store vectors of 3D points for each checkerboard image
        obj_points = []
        # Creating vector to store vectors of 2D points for each checkerboard image
        img_points = []

        reprojection_error_per_Image = []
        # sharpness_error_per_Image = []
        # Defining the world coordinates for 3D points
        objp = np.zeros((1, checker_board_size[0] * checker_board_size[1], 3), np.float32)
        objp[0, :, :2] = np.mgrid[0:checker_board_size[0], 0:checker_board_size[1]].T.reshape(-1, 2)

        # specify the scale for each square in mm
        objp = objp * size_chessboard_squares_mm

        img_list = []

        for count, frame in enumerate(images):
            # assert frame.any() == 0, "file could not be read, check with os.path.exists()"
            # ret_CB = cv2.checkChessboard(frame, checker_board_size)
            # print(ret_CB)
            gray_img = frame  # RW.sharpen(img, show=True)
            gray_image = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)
            gray_image_shape = gray_image.shape[::-1]
            # Find the chess board corners
            ret2, corners = cv2.findChessboardCornersSB(gray_image, checker_board_size,
                                                        cv2.CALIB_CB_LARGER + cv2.CALIB_CB_MARKER)
            # If desired number of corners are found in the image then ret = true
            if ret2:
                # refining pixel coordinates for given 2d points.
                # retval, sharpness = cv2.estimateChessboardSharpness(gray_img, checker_board_size, corners)
                # # print(file_names[count] + " sharpness", retval[0])
                if show:
                    # Try corners = cv2.find4QuadCornerSubpix(gray_image, corners)
                    img_draw = cv2.drawChessboardCorners(gray_img, checker_board_size, corners, ret2)
                    # plt.imshow(img_draw, cmap="gray")
                    # plt.show()
                    save_text = "Press key s to consider the image for calibration or Esc to skip the image"
                    # sharp_text = str(file_names[count]) + " sharpness : " + str(retval[0])
                    # Draw and display the corners
                    winname = self.camera + " " + file_names[count]
                    cv2.putText(img=img_draw, text=save_text, org=(0, 500), fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                                fontScale=3,
                                color=(0, 255, 0), thickness=5)
                    # # cv2.putText(img=img_draw, text=sharp_text, org=(0, 700), fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                    #             fontScale=3,
                    #             color=(0, 255, 0), thickness=5)
                    # cv2.moveWindow(winname, 100, 30)
                    img_draw = RW.resize(img_draw, height=900, width=None, interMethod=cv2.INTER_LINEAR)
                    cv2.imshow(winname, img_draw)
                    k = cv2.waitKey(0)
                    if k == ord("s"):
                        img_points.append(corners)
                        obj_points.append(objp)
                        img_list.append(file_names[count])
                        # sharpness_error_per_Image.append([retval[0]])
                    elif k == 27:
                        pass
                    cv2.destroyAllWindows()
                else:
                    img_points.append(corners)
                    obj_points.append(objp)
                    img_list.append(file_names[count])
                    # sharpness_error_per_Image.append([retval[0]])
        else:
            pass

        print("\nEstimating camera calibration for " + self.camera + "...")
        print('\nImages available for calibration for ' + self.camera + ' : ', len(images))
        if len(images) > 0:
            print('\nImage size ', gray_image_shape)

            ret, cameraMatrix, dist, rvecs, tvecs, _, _, perViewErrors, optimal_Camera_Matrix, roi = self.camera_calibration(
                obj_points, img_points, gray_image_shape)

            print("camera calibration RMSE " + self.camera + " :", ret)

            # Filtering inaccurate image points based on criteria and recalculating camera calibration
            updated_img_points = []
            updated_obj_points = []
            updated_img_list = []
            updated_sharpness_error_per_Image = []
            filter_img_indices = np.argwhere(np.all(perViewErrors <= self._filtering_criteria, axis=1)).tolist()
            filter_img_indices = [item for sublist in filter_img_indices for item in sublist]
            for i in filter_img_indices:
                updated_img_points.append(img_points[i])
                updated_obj_points.append(obj_points[i])
                updated_img_list.append(file_names[i])
                # updated_sharpness_error_per_Image.append(sharpness_error_per_Image[i])

            newRet, cameraMatrix, dist, rvecs, tvecs, _, _, perViewErrors, optimal_Camera_Matrix, roi = self.camera_calibration(
                updated_obj_points, updated_img_points, gray_image_shape)
            print("\nThreshold for reprojection error : " + str(self._filtering_criteria))
            print("\nImages considered for calibration post filtering, " + self.camera + " :", len(updated_img_list))
            print("camera calibration RMSE post filtering, " + self.camera + " :", newRet)

            for error in perViewErrors:
                reprojection_error_per_Image.append(error)

            print("Camera Matrix \n", cameraMatrix)
            print("Distortion Coefficient : ", dist)

            # updated_sharpness_error_per_Image = [item for sublist in updated_sharpness_error_per_Image for item in
            #                                      sublist]

            # Print of Reprojection error per image using PrettyTable
            print("\nReprojection error in pixels and sharpness value per image")
            ReprojTable = [updated_img_list, list(np.around(np.array(reprojection_error_per_Image), 2))]
            # list(np.around(np.array(updated_sharpness_error_per_Image), 2))]
            ReprojTab = PrettyTable(ReprojTable[0])
            ReprojTab.add_rows(ReprojTable[1:])
            # leftcol = ["Images", "Reproj", "Sharpness"]
            print(ReprojTab)

            # x = PrettyTable()
            # x.field_names = ["Images", ReprojTable[0]]
            # x.add_column(["Reproj", ReprojTable[1]])
            # x.add_column(["Sharpness", ReprojTable[2]])
            # print(x)

            error = np.concatenate(reprojection_error_per_Image)
            plt.figure(figsize=(10, 5))
            plt.bar(updated_img_list, error)  # color='b'
            plt.axhline(y=newRet, color='r', linestyle='--')
            handles = ['Overall Mean error ' + str(round(newRet, 2)), 'Reprojection error per image']
            plt.legend(handles)
            plt.xticks(rotation=45, ha="right")
            plt.xlabel("Calibration images")
            plt.ylabel("Reprojection error (RMSE)")
            plt.title("Calibration result for " + self.camera)
            plt.show()

            self.result.update(calibration_error=newRet,
                               camera_matrix=cameraMatrix,
                               optimal_Camera_Matrix=optimal_Camera_Matrix, dist=dist, ROI=roi)
            self.data.image_size = gray_image_shape
        else:
            print("No images available or unable to detect chessboard corners")

    def calibration_ChAruCo_pattern(self, file_names, images, rows, columns):
        # Chessboard configuration
        CHARUCOBOARD_ROWCOUNT = columns
        CHARUCOBOARD_COLCOUNT = rows
        aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_1000)

        # create constants to be passed into openCV and Aruco methods
        board = cv2.aruco.CharucoBoard_create(squaresX=CHARUCOBOARD_ROWCOUNT,
                                              squaresY=CHARUCOBOARD_COLCOUNT,
                                              squareLength=0.011,
                                              markerLength=0.009,
                                              dictionary=aruco_dict)

        # check if the board is correct
        image = board.draw((1280, 720))

        plt.figure()
        plt.imshow(image, cmap='gray')
        plt.title('8x12 ChAruco pattern')
        plt.show()

        # # Input images capturing the chessboard above
        # input_files = '../data/charuco/*.jpg'

        parameters = cv2.aruco.DetectorParameters_create()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR

        # Create the arrays and variables for corners and image names

        all_corners = []  # Corners detected in all images
        all_ids = []  # Aruco ids corresponding to corners detected
        imsize = None

        for count, frame in enumerate(images):
            assert frame.any() != 0, "file could not be read, check with os.path.exists()"
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Find aruco markers in the image
            corners, ids, rejected_points = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

            # outline the aruco markers found
            frame = cv2.aruco.drawDetectedMarkers(image=frame, corners=corners)

            # Get charuco corners and ids from detected aruco markers
            if len(corners) > 0:
                ret, c_corners, c_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
                print(str(file_names[count]) + ' found ' + str(ret) + ' corners')
                if ret > 0:
                    all_corners.append(c_corners)
                    all_ids.append(c_ids)

                    frame = cv2.aruco.drawDetectedCornersCharuco(
                        image=frame,
                        charucoCorners=c_corners,
                        charucoIds=c_ids
                    )
                    imsize = (gray.shape[1], gray.shape[0])

                    # proportion = max(frame.shape) / 1000
                    # img = cv2.resize(frame, (int(frame.shape[1] / proportion), int(frame.shape[0] / proportion)))

                    img_draw = RW.resize(img_draw, height=600, width=None, interMethod=cv2.INTER_LINEAR)

                    cv2.imshow('Charuco Board', img_draw)
                    cv2.waitKey(0)
                else:
                    print("Unable to detect a charuco board in the image: {}", format(count))

        cv2.destroyAllWindows()

        if (len(images) < 1):
            print(
                "Calibration failed. No images of charuco boards were found in the folder. Add images or recheck the naming conventions used in this file")
            exit()

        ret, cameraMatrix, dist, rvec, tvec, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors = cv2.aruco.calibrateCameraCharucoExtended(
            all_corners, all_ids, board, imsize, None, None,
            flags=cv2.CALIB_FIX_ASPECT_RATIO + cv2.CALIB_RATIONAL_MODEL)

        print("Reprojection error = ", ret)
        print("Intrinsic parameter K = ", cameraMatrix)
        print("Distortion parameters d = (k1, k2, p1, p2, k3, k4, k5, k6) = ", dist)

        assert ret < 1.0

    def calibration_circular_grid(self, file_names, images, grid_size, spacing_in_mm=15, show=False):
        # https://github.com/LongerVision/Examples_OpenCV/blob/master/01_internal_camera_calibration/circle_grid.py

        # #######################################Blob Detector##############################################

        # Setup SimpleBlobDetector parameters.
        blobParams = cv2.SimpleBlobDetector_Params()

        # Change thresholds
        blobParams.minThreshold = 1
        blobParams.maxThreshold = 255

        # Filter by Area.
        blobParams.filterByArea = True
        blobParams.minArea = 10  # minArea may be adjusted to suit for your experiment
        blobParams.maxArea = 10000  # maxArea may be adjusted to suit for your experiment

        # Filter by Circularity
        blobParams.filterByCircularity = True
        blobParams.minCircularity = 0.1

        # # Filter by Convexity
        # blobParams.filterByConvexity = True
        # blobParams.minConvexity = 0.87

        # Filter by Inertia
        blobParams.filterByInertia = False
        blobParams.minInertiaRatio = 0.01

        blobParams.minDistBetweenBlobs = 10

        # Create a detector with the parameters
        blobDetector = cv2.SimpleBlobDetector_create(blobParams)
        ###################################################################################################

        # Prepare object points (0,0,0), (1,0,0), (2,0,0) ...., (8,5,0)
        objp = np.zeros((grid_size[0] * grid_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:grid_size[0], 0:grid_size[1]].T.reshape(-1, 2)

        # specify the scale for each square in mm
        objp = objp * spacing_in_mm

        # Create arrays to store object points and image points from all the images
        obj_points = []  # 3D points in real world space
        img_points = []  # 2D points in image plane

        for frame in images:

            # Read the image
            # assert frame.any() == 0, "file could not be read, check with os.path.exists()"

            # Convert to grayscale
            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            keypoints = blobDetector.detect(gray_image)  # Detect blobs.

            # Draw detected blobs as red circles. This helps cv2.findCirclesGrid() .
            im_with_keypoints = cv2.drawKeypoints(gray_image, keypoints, np.array([]), (0, 255, 0),
                                                  cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            im_with_keypoints_gray = cv2.cvtColor(im_with_keypoints, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findCirclesGrid(im_with_keypoints_gray, grid_size, None,
                                               flags=cv2.CALIB_CB_SYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING)  # Find the circle grid

            if ret:
                obj_points.append(objp)  # Certainly, every loop objp is the same, in 3D.

                corners2 = cv2.cornerSubPix(im_with_keypoints_gray, corners, (11, 11), (-1, -1),
                                            self.criteria)  # Refines the corner locations.
                img_points.append(corners2)

                # Draw and display the corners.
                im_with_keypoints = cv2.drawChessboardCorners(frame, grid_size, corners2, ret)

                if show:
                    plt.imshow(im_with_keypoints, cmap="gray")
                    plt.show()

        gray_image_shape = gray_image.shape[::-1]

        ret, cameraMatrix, dist, rvecs, tvecs, _, _, perViewErrors, optimal_Camera_Matrix, roi = self.camera_calibration(
            obj_points, img_points, gray_image_shape)

        # Print the camera matrix and distortion coefficients
        print("Camera Matrix:\n", cameraMatrix)
        print("\nDistortion Coefficients:\n", dist)
        print("Reprojection Error :", ret)
