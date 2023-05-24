import cv2
import numpy as np
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import pandas as pd
import os

import Calibration.CameraCalibration as camCalib
import Calibration.StereoCalibration as stereoCalib
import Calibration.StereoRectification as stereoRect
from StereoVision import Triangulation, Rectify, Read_Write, Configuration, \
    Camera_World_Calibration, Camera, Stereo_Correspondence

if __name__ == '__main__':
    camera_available = False
    camera_calibration_required = False
    stereo_calibration_required = False
    depth_estimation = True
    saveWorldCoordinateFrame = False

    Data = Configuration.Data
    RW = Read_Write

    if camera_available:
        set_parameters = False

        # Camera initialization and operations
        if set_parameters:
            Data.cam_1_parameters.update(gain=1, gamma=1, exposure_time=42631, digital_shift=0,
                                         AcquisitionFrameRate=100)
            Data.cam_2_parameters.update(gain=1, gamma=1, exposure_time=42631, digital_shift=0,
                                         AcquisitionFrameRate=100)

        Camera_Basler = Camera.Camera(Data, set_parameters)
        camera_1 = Camera_Basler.connected_camera[0]
        camera_2 = Camera_Basler.connected_camera[1]

        Data.connected_camera.update(cam_1=camera_1, cam_2=camera_2)

    else:
        Data.connected_camera.update(cam_1="camera_1", cam_2="camera_2")
        camera_1 = Data.connected_camera['cam_1']
        camera_2 = Data.connected_camera['cam_2']
    ########################################################################################################q###########
    if camera_calibration_required:
        Calibration_Images_available = False

        # Chose Calibration type :
        # Checkerboard = 1, Charuco board = 2, Radon = 3
        calibration = 1

        if not Calibration_Images_available:
            if calibration == 1:
                left_camera_path = Data.left_image_path
                right_camera_path = Data.right_image_path
            elif calibration == 2:
                left_camera_path = Data.CH_left_image_path
                right_camera_path = Data.CH_right_image_path
            elif calibration == 3:
                left_camera_path = Data.CG_left_image_path
                right_camera_path = Data.CG_right_image_path

            Camera_Basler.stereo_stream(left_camera_path, right_camera_path, save_image=True)
            print("Images saved to the designated folders")
        # Individual camera calibration with checkerboard ##############################################################
        if calibration == 1:
            checker_board_size = (8, 5)
            size_chessboard_squares_mm = 13.95

            cam_1_calibrate = camCalib.CameraCalibration(Data, camera_1)
            cam_1_file_names, cam_1_images = RW.readImages(Data.left_image_path)
            cam_1_calibrate.calibration_Checker_Board(cam_1_file_names, cam_1_images,
                                                      checker_board_size,
                                                      size_chessboard_squares_mm, show=False)

            cam_2_calibrate = camCalib.CameraCalibration(Data, camera_2)
            cam_2_file_names, cam_2_images = RW.readImages(Data.right_image_path)
            cam_2_calibrate.calibration_Checker_Board(cam_2_file_names, cam_2_images,
                                                      checker_board_size,
                                                      size_chessboard_squares_mm, show=False)

            cam_parameter_file = "Cam_Calib_Checkerboard_Results"
            camera_calibration_data = RW.saveCamResults(cam_parameter_file, cam_1_calibrate, cam_2_calibrate)

            print("\nCamera calibration for cameras executed successfully")
            print("Results saved in a file named " + cam_parameter_file + ".txt to the directory")

        # Individual camera calibration with Charuco Board ##############################################################

        elif calibration == 2:

            # Charuco Board configuration
            rows = 8
            columns = 12
            cam_1_calibrate = camCalib.CameraCalibration(Data, camera_1)
            cam_1_file_names, cam_1_images = RW.readImages(Data.CH_left_image_path)
            cam_1_calibrate.calibration_ChAruCo_pattern(cam_1_file_names, cam_1_images, rows, columns)

            cam_2_calibrate = camCalib.CameraCalibration(Data, camera_2)
            cam_2_file_names, cam_2_images = RW.readImages(Data.CH_right_image_path)
            cam_2_calibrate.calibration_ChAruCo_pattern(cam_2_file_names, cam_2_images, rows, columns)
            #
            cam_parameter_file = "Cam_Calib_ChArUco_Results"
            camera_calibration_data = RW.saveCamResults(cam_parameter_file, cam_1_calibrate, cam_2_calibrate)
            #
            print("\nCamera calibration for cameras executed successfully")
            print("Results saved in a file named " + cam_parameter_file + ".txt to the directory")

        # Individual camera calibration with Circular Grid ##############################################################

        elif calibration == 3:
            # Circular grid configuration
            rows = 5
            columns = 6

            cam_1_calibrate = camCalib.CameraCalibration(Data, camera_1)
            cam_1_file_names, cam_1_images = RW.readImages(Data.CG_left_image_path)
            cam_1_calibrate.calibration_circular_grid(cam_1_file_names, cam_1_images, (rows, columns), spacing_in_mm=15,
                                                      show=True)

            cam_2_calibrate = camCalib.CameraCalibration(Data, camera_2)
            cam_2_file_names, cam_2_images = RW.readImages(Data.CG_right_image_path)
            cam_2_calibrate.calibration_circular_grid(cam_2_file_names, cam_2_images, (rows, columns),
                                                      spacing_in_mm=15, show=False)
            #
            cam_parameter_file = "Cam_Calib_CircularGrid_Results"
            camera_calibration_data = RW.saveCamResults(cam_parameter_file, cam_1_calibrate, cam_2_calibrate)
            #
            print("\nCamera calibration for cameras executed successfully")
            print("Results saved in a file named " + cam_parameter_file + ".txt to the directory")

    # Stereo Calibration ###############################################################################################
    if stereo_calibration_required:
        camera_parameter_file = "Cam_Calib_Checkerboard_Results.obj"

        if True:  # os.path.isfile(camera_parameter_file):
            camera_calibration = RW.loadCameraParameters(camera_parameter_file)

            cam_1_file_names, cam_1_images = RW.readImages(Data.left_image_path)
            cam_2_file_names, cam_2_images = RW.readImages(Data.right_image_path)
            stereo_calibrate = stereoCalib.StereoCalibration(Data, camera_1, camera_2, camera_calibration)
            stereo_calibrate.calibration_with_checkerboard(cam_1_file_names, cam_1_images, cam_2_file_names,
                                                           cam_2_images, checker_board_size=(8, 5),
                                                           size_chessboard_squares_mm=13.95, show=False)

            # Stereo Rectification #####################################################################################

            stereo_rectification = stereoRect.StereoRectification(Data, camera_calibration, stereo_calibrate)
            stereo_rectification.rectification_check(cam_1_images[0], cam_2_images[0], boardSize=(8, 5),
                                                     size_chessboard_squares_mm=13.95)

            # Saving stereo calibration and rectification parameters####################################################
            stereo_parameter_file = "stereo_calib_results"
            calibration_data = RW.saveStereoResults(stereo_parameter_file, stereo_calibrate,
                                                    stereo_rectification)

            print("\nStereo calibration along with stereo rectification executed successfully")
            print("Results saved in a file named " + stereo_parameter_file + " to the directory")

        else:
            print("\nStereo calibration not possible, camera calibration parameter file missing!!!")
            quit()

    if depth_estimation:
        # Stereo pair Triangulation ###################################################################################
        # img1, img2 = Camera_Basler.get_frame(save=True)

        # Input images of scene from stereo cameras ####################################################################

        img1 = cv2.imread("F:\TUHH\MasterThesis\Code\hyfas_calibration\Images\TemplateImage\TestObject_863_Img1.bmp")
        # "F:\TUHH\MasterThesis\Code\hyfas_calibration\Images\CheckerBoard\StereoLeft\Img_5.png"
        img2 = cv2.imread("F:\TUHH\MasterThesis\Code\hyfas_calibration\Images\TemplateImage\TestObject_864_Img1.bmp")
        # "F:\TUHH\MasterThesis\Code\hyfas_calibration\Images\CheckerBoard\StereoRight\Img_5.png"

        # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Input camera and stereo pair calibration parameters###########################################################
        camera_parameter_file = "Cam_Calib_Checkerboard_Results.obj"
        stereo_parameter_file = "stereo_calib_results.obj"
        camera_calibration_parameters = RW.loadCameraParameters(camera_parameter_file)
        Stereo_calibration_parameters = RW.loadCameraParameters(stereo_parameter_file)

        # dst = cv2.undistort(img1, camera_calibration_parameters['C1_camera_matrix'],
        #                     camera_calibration_parameters['C1_distortion_coeff'], None,
        #                     camera_calibration_parameters['C1_optimal_Camera_matrix'])
        # plt.imshow(dst, cmap="gray")
        # plt.show()

        # Image Rectification ##########################################################################################

        frame_1, frame_2 = Rectify.undistortRectify(img1, img2, show=False)

        # Stereo Correspondence ########################################################################################

        # Validating with stereo matcher
        #
        # F, E, R, t, RPose, left_inliers, right_inliers = Stereo_Correspondence.stereo_matcher(frame_1, frame_2,
        #                                                                                       camera_calibration_parameters,
        #                                                                                       show=False)

        # Disparity Map

        # disparity = Stereo_Correspondence.show_disparity_with_blockMatching(frame_1, frame_2, show=True)
        # disparity = correspondenseTrial.ssd_correspondence(frame_1, frame_2, F)

        # Input the specific point in the left image
        # # Finding the correspondence point from the right image
        # L_point_x, L_point_y = Stereo_Correspondence.mark_points(frame_1)
        # R_point_x, R_point_y = Stereo_Correspondence.detect_corresponding_point_old(L_point_x, L_point_y, frame_1, frame_2)

        # Read the Excel file into a pandas DataFrame
        file = os.path.join(Data.results_path, "Measurements.xlsx")
        df = pd.read_excel(file, header=0)
        print(df.head())
        total = 6
        columns = ['Target', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2']
        df1 = pd.DataFrame(columns=columns)
        for n in range(len(df)):
            # Manual addition of points
            L_pt1_x, L_pt1_y = df.iloc[n][0], df.iloc[n][
                1]  # 2727,2968 # 2445,3296 # 2735.6,3207.1  # 2456.6,2326.2  #  # 2434.1,3212.6  #
            R_pt1_x, R_pt1_y = df.iloc[n][2], df.iloc[n][
                3]  # 1515,2966 # 1201.8,3295.9 # 1465.8,3206.5 # 1434.1,2323.9 #  1163.3,3212.6  #   #

            L_pt2_x, L_pt2_y = df.iloc[n][4], df.iloc[n][
                5]  # 2435,3211 # 3827.3,3265 # 2735.6,3291.9  # 3838.6,3231.3 # 3596.6,2266.5 #   #
            R_pt2_x, R_pt2_y = df.iloc[n][6], df.iloc[n][
                7]  # 1165,3210 # 2574.1,3264.9 # 1494.7,3291.5  # 2574.6,3231.0 # 2567.8,2266.5 #   #

            # Stereo_Correspondence.display_corresponding_points(frame_1, frame_2, (L_pt1_x, L_pt1_y), (R_pt1_x, R_pt1_y),
            #                                                    (L_pt2_x, L_pt2_y),
            #                                                    (R_pt2_x, R_pt2_y), camera_calibration_parameters)

            # Triangulation ################################################################################################

            triangulation = Triangulation.Triangulation(camera_calibration_parameters, Stereo_calibration_parameters)

            L_pt1, R_pt1 = triangulation.corner2center_transform(L_pt1_x, L_pt1_y, R_pt1_x, R_pt1_y)
            L_pt2, R_pt2 = triangulation.corner2center_transform(L_pt2_x, L_pt2_y, R_pt2_x, R_pt2_y)

            triangulation_methods = ['linear_triangulation', 'linear_eigen_triangulation', 'linear_LS_triangulation',
                                     'polynomial_triangulation', 'Iterative_LS_triangulation', 'dlt']
            lengths_compare = []
            error = []
            reprojection_error_compare = []
            actual_length_cm = df.iloc[n][8]  # 2.78 # 9.0001 # 0.9107 # 12.73 #
            print("True_Length : ", actual_length_cm)
            for method_name in triangulation_methods:
                # Get the method from the triangulator instance
                # print("Triangulation method : ", method_name)
                method = getattr(triangulation, method_name)
                point1 = method(L_pt1, R_pt1)
                point2 = method(L_pt2, R_pt2)
                # print(point1/10)
                # print(point2/10)
                length = triangulation.measure_length(point1, point2)
                reprojection_error = triangulation.reproject(frame_1, point1, L_pt1, R_pt1)
                lengths_compare.append(length / 10.0)
                error.append(actual_length_cm - (length / 10.0))

            world_data = [[actual_length_cm] + point1.tolist() + point2.tolist()]
            world_data = [item for sublist in world_data for item in sublist]
            df1.loc[len(df1)] = world_data

            # Print of Reprojection error per image using PrettyTable
            print("\n Length between the two selected points with multiple approaches in cm")
            ReprojTable = [triangulation_methods, list(np.around(np.array(lengths_compare), 8)),
                           list(np.around(np.array(error), 8))]
            ReprojTab = PrettyTable(ReprojTable[0])
            ReprojTab.add_rows(ReprojTable[1:])
            print(ReprojTab)

        # Save the DataFrame to an Excel file
        World_coordinates_file = os.path.join(Data.results_path, "World.xlsx")
        df1.to_excel(World_coordinates_file, index=False)
    # Coordinate transformation #####################################################################################

    if saveWorldCoordinateFrame:

        # img1, img2 = Camera_Basler.get_frame()

        # Input camera and stereo pair calibration parameters###########################################################
        camera_parameter_file = "Cam_Calib_Checkerboard_Results.obj"
        stereo_parameter_file = "stereo_calib_results.obj"
        camera_calibration_parameters = RW.loadCameraParameters(camera_parameter_file)
        Stereo_calibration_parameters = RW.loadCameraParameters(stereo_parameter_file)

        CW_calibration = Camera_World_Calibration.Camera_World_Calibration(camera_calibration_parameters)
        # Assuming world frame at Left camera or camera0
        method = 1

        # Method 1 : Define world coordinate frame through aruco marker
        if method == 1:
            img = cv2.imread(
                "F:\TUHH\MasterThesis\Code\hyfas_calibration\Images\TemplateImage\864_downsampled_Img3.bmp")
            # Definition of Aruco Marker
            aruco_dict_type = cv2.aruco.DICT_5X5_250
            marker_id = 124
            marker_size_cm = 5

            # marker_image = CW_calibration.generateAruCoMarker(aruco_dict_type, marker_id, marker_size_cm, save=False)
            R_W0, T_W0 = CW_calibration.get_world_space_origin_ArucoMarker(img, aruco_dict_type,
                                                                           camera_calibration_parameters,
                                                                           marker_size_cm)

        elif method == 2:
            img = cv2.imread(
                "F:\TUHH\MasterThesis\Code\hyfas_calibration\Images\TemplateImage\864_downsampled_Img3.bmp")
            # Method 2 :  Define world coordinate frame through checkerboard
            R_W0, T_W0 = Camera_World_Calibration.get_world_space_origin_checkerboard(img,
                                                                                      camera_calibration_parameters)
            R_W1, T_W1 = Camera_World_Calibration.get_cam1_to_world_transforms(camera_calibration_parameters, R_W0,
                                                                               T_W0, R1, T1, frame_1, frame_2)

        elif method == 3:

            img = cv2.imread("F:/TUHH/MasterThesis/Code/hyfas_calibration/Images/TemplateImage/bedPlateSample1_863.bmp")
            # Method 3 : Define world coordinate frame through PnP solve
            world_points = []
            Image_points = CW_calibration.marker_detection(img)
            # R_W0, T_W0 = CW_calibration.pnpSolver(img, Image_points,world_points, camera_calibration_parameters)

        # print(R_W0, T_W0)
        # cam_world_coordinate = dict(R_C_W0=R_W0, T_C_W0=T_W0)
        # RW.saveCameraParameters("WorldCoordinateFrame", cam_world_coordinate)

    else:
        # camera_parameter_file = "Cam_Calib_Checkerboard_Results.obj"
        # camera_calibration_parameters = RW.loadCameraParameters(camera_parameter_file)
        # camera0 rotation and translation is identity matrix and zeros vector
        # R0 = np.eye(3, dtype=np.float32)
        # T0 = np.zeros((3, 1))
        #
        # R1 = Stereo_calibration_parameters['stereo_rotation_C1_C2']
        # T1 = Stereo_calibration_parameters['stereo_translation_C1_C2']
        # # to avoid confusion, camera1 R and T are labeled R1 and T1

        # Camera_World_Calibration.marker_detection()

        # Transform detected point to world coordinate frame#################################################################
        # cam_world_coordinate = RW.loadCameraParameters("WorldCoordinateFrame.obj")
        # # Create the 4x4 transformation matrix that combines rotation and translation
        # extrinsic_matrix = np.zeros((4, 4), np.float32)
        # extrinsic_matrix[:3, :3] = R0
        # extrinsic_matrix[:3, 3] = T0
        # extrinsic_matrix[3, 3] = 1
        # #
        # # # Define the point in camera coordinate frame
        # point_camera = np.array([X, Y, Z, 1], np.float32)
        # point_camera = np.array([X, Y, Z], np.float32)
        # # Transform the point from camera coordinate frame to world coordinate frame
        # point_world = np.dot(extrinsic_matrix, point_camera)
        # #
        # # # Normalize the homogeneous coordinates to get the final 3D point in world coordinate frame
        # point_world = point_world[:3] / point_world[3]
        #
        # # camera coordinate to world coordinate
        # world_point = cam_world_coordinate['R_C_W0'] ** -1 * (point_camera - cam_world_coordinate['R_C_W0'])
        print("Good bye!!!!")
