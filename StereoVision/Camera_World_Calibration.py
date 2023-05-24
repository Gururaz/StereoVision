import os

import cv2
import numpy as np

from StereoVision import Configuration as CF
from StereoVision import Read_Write


class Camera_World_Calibration:

    def __init__(self, camera_calibration_parameters):

        # Camera 1 parameters
        self.K1 = camera_calibration_parameters['C1_camera_matrix']
        self.d1 = camera_calibration_parameters['C1_distortion_coeff']
        self.transformation = {
            'Rvec': [],
            'Tvec': []
        }

    def pnpSolver(self, img, imagePoints, objectPoints, camera_calibration_parameters):
        flag = cv2.SOLVEPNP_AP3P
        retval, rvec, tvec = cv2.solvePnP(objectPoints, imagePoints, self.K1,
                                          self.d1, False, flag)
        cv2.drawFrameAxes(img, self.K1, self.d1,rvec,tvec, 50, 5)
        self.transformation.update(Rvec=rvec, Tvec=tvec)
        return rvec, tvec

    def marker_detection(self, im):
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        crop_gray = Read_Write.resize(gray, 800)
        cv2.imshow("Originial Image", crop_gray)
        cv2.waitKey(0)

        # create a mask
        mask = np.zeros(gray.shape[:2], np.uint8)
        # Rectangular marker starting at Top left corner
        x = 1800
        y = 1500
        w = 2700
        h = 2500
        mask[y:y + h, x: x + w] = 255
        masked_img = cv2.bitwise_and(gray, gray, mask=mask)
        crop_mask = Read_Write.resize(masked_img, 800)
        cv2.imshow("Masked", crop_mask)
        cv2.waitKey(0)

        # Apply binary thresholding to the image
        thresh = cv2.threshold(masked_img, 14, 255, cv2.THRESH_BINARY_INV)[1]
        mask = np.zeros(thresh.shape[:2], np.uint8)
        mask[y:y + h, x:x + w] = thresh[y:y + h, x:x + w]

        # Improve thresholding technique with Morphological Transformation : Closing
        kernel = np.ones((50, 50), np.uint8)
        closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        crop_binary = Read_Write.resize(closing, 800)
        cv2.imshow("Masked", crop_binary)
        cv2.waitKey(0)

        # Find contours, filter using contour threshold area, and draw rectangle
        cnts = cv2.findContours(closing, None, cv2.CHAIN_APPROX_TC89_L1)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        required_contours = []
        set_pixel_threshold = 100000
        marker_centroid = []
        for c in cnts:
            area = cv2.contourArea(c)
            if area > set_pixel_threshold:  # and area < 400000:
                # Additional contour approximation method for smooth contour (cv2.approxPolyDP)
                # epsilon = 0.0035 * cv2.arcLength(c, True)
                # approx = cv2.approxPolyDP(c, epsilon, True)
                required_contours.append(c)

        if len(required_contours) > 0:
            print("Available contours ", len(required_contours))
            crop = cv2.drawContours(image=im.copy(), contours=required_contours, contourIdx=-1, color=(0, 255, 0),
                                    thickness=3, lineType=cv2.LINE_AA)
            for i in required_contours:
                M = cv2.moments(i)
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                marker_centroid.append((cx, cy))
                cv2.drawMarker(crop, (cx, cy), (255,128, 0), cv2.MARKER_CROSS, 75, 5)

            crop_contour = Read_Write.resize(crop, 800)
            cv2.imshow("Masked", crop_contour)
            cv2.waitKey(0)

        else:
            print("No contours detected")

        cv2.destroyAllWindows()
        return marker_centroid

    def get_world_space_origin_checkerboard(self, camera_calibration_parameters, frame):
        cmtx = camera_calibration_parameters['C1_camera_matrix']
        dist = camera_calibration_parameters['C1_distortion_coeff']

        # calibration pattern settings
        rows = 8
        columns = 8
        world_scaling = 14

        # coordinates of squares in the checkerboard world space
        objp = np.zeros((rows * columns, 3), np.float32)
        objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
        objp = world_scaling * objp

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (rows, columns), None)

        cv2.drawChessboardCorners(frame, (rows, columns), corners, ret)
        cv2.putText(frame, "If you don't see detected points, try with a different image", (50, 50),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1, (0, 0, 255), 1)
        cv2.imshow('img', frame)
        cv2.waitKey(0)

        ret, rvec, tvec = cv2.solvePnP(objp, corners, cmtx, dist)
        R, _ = cv2.Rodrigues(rvec)  # rvec is Rotation matrix in Rodrigues vector form

        return R, tvec

    def get_world_space_origin_ArucoMarker(self, frame, aruco_dict_type, camera_calibration_parameters, marker_size):
        print("Reading the aruco marker from image for pose estimation of the marker")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Visualise image
        # crop_gray = Read_Write.resize(gray, 800)
        # cv2.imshow("Originial Image", crop_gray)
        # cv2.waitKey(0)

        # cameraMatrix=cmtx, distCoeff=dist)
        cmtx = camera_calibration_parameters['C1_camera_matrix']
        dist = camera_calibration_parameters['C1_distortion_coeff']

        marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                                  [marker_size / 2, marker_size / 2, 0],
                                  [marker_size / 2, -marker_size / 2, 0],
                                  [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)

        detectorParams = cv2.aruco.DetectorParameters()
        dictionary = cv2.aruco.getPredefinedDictionary(dict=aruco_dict_type)
        detector = cv2.aruco.ArucoDetector(dictionary, detectorParams)

        ids = []
        corners = []
        rvecs = []
        tvecs = []
        rotation = []

        # corners, ids, rejected_img_points = detector.detectMarkers(frame, aruco_dict_type, parameters=detectorParams)
        try:
            corners, ids, rejected_img_points = detector.detectMarkers(gray, corners)
            if len(ids) == None:
                raise Exception
            # If markers are detected
            if len(ids) > 0:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)

                # Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
                for c in corners:
                    _, r, t = cv2.solvePnP(marker_points, c[0], cmtx, dist, False, cv2.SOLVEPNP_IPPE_SQUARE)
                    rvecs.append(r)
                    tvecs.append(t)
                    cv2.drawFrameAxes(frame, cmtx, dist, r, t, 3)
                    distance_to_marker = np.round(np.linalg.norm(tvecs), 2)
                print("Distance from camera frame to marker frame in mm: ", distance_to_marker)
                crop_frame = Read_Write.resize(frame, 800)
                cv2.imshow('Estimated Pose', crop_frame)
                cv2.waitKey(0)
                rotation, _ = cv2.Rodrigues(rvecs[0])

        except Exception as e:
            print(e)

        except Exception:
            print("No Aruco marker found in the image. Check the image or Aruco Marker id")

        finally:
            return rotation, tvecs

    def get_cam1_to_world_transforms(self, camera_calibration_parameters, R_W0, T_W0,
                                     R_01, T_01,
                                     frame0,
                                     frame1):
        cmtx0 = camera_calibration_parameters['C1_camera_matrix']
        dist0 = camera_calibration_parameters['C1_distortion_coeff']
        cmtx1 = camera_calibration_parameters['C2_camera_matrix']
        dist1 = camera_calibration_parameters['C2_distortion_coeff']

        unitv_points = 5 * np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype='float32').reshape((4, 1, 3))
        # axes colors are RGB format to indicate XYZ axes.
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]

        # project origin points to frame 0
        points, _ = cv2.projectPoints(unitv_points, R_W0, T_W0, cmtx0, dist0)
        points = points.reshape((4, 2)).astype(np.int32)
        origin = tuple(points[0])
        for col, _p in zip(colors, points[1:]):
            _p = tuple(_p.astype(np.int32))
            cv2.line(frame0, origin, _p, col, 2)

        # project origin points to frame1
        R_W1 = R_01 @ R_W0
        T_W1 = R_01 @ T_W0 + T_01
        points, _ = cv2.projectPoints(unitv_points, R_W1, T_W1, cmtx1, dist1)
        points = points.reshape((4, 2)).astype(np.int32)
        origin = tuple(points[0])
        for col, _p in zip(colors, points[1:]):
            _p = tuple(_p.astype(np.int32))
            cv2.line(frame1, origin, _p, col, 2)

        cv2.imshow('frame0', frame0)
        cv2.imshow('frame1', frame1)
        cv2.waitKey(0)

        return R_W1, T_W1

    def generateAruCoMarker(self, aruco_dict_type, marker_id, marker_size_cm=5, save=False):
        print("Performing Aruco Marker of type " + str(aruco_dict_type) + " with id " + str(marker_id))
        try:
            dictionary = cv2.aruco.getPredefinedDictionary(dict=aruco_dict_type)
            marker_image = cv2.aruco.generateImageMarker(dictionary, id=marker_id, sidePixels=250)

            # Define the marker size in centimeters
            marker_size = marker_size_cm

            # Define the dpi (dots per inch) of the display or image
            dpi = 300

            # Define the physical size of the display or image in pixels
            display_size_px = (1920, 1080)
            inch_to_cm = 2.54
            # Convert the marker size in centimeters to pixels
            marker_size_px = dpi * marker_size / inch_to_cm

            # marker_size_px = (marker_size_cm / display_size_px[0]) * dpi * display_size_px[0]
            # print(marker_size_px)
            print("Aruco Marker of specified dictionary type exists, enable save to write the image to the folder!!")
            # Display the marker image
            img = Read_Write.resize(marker_image)
            cv2.imshow("Aruco Marker", img)
            cv2.waitKey(0)
            if save:
                file = "Marker " + str(marker_id) + ".png"
                cv2.imwrite(os.path.join(CF.Data.pattern_path, file), marker_image)
                print("File saved to the local folder")
            cv2.destroyAllWindows()

            return marker_image

        except Exception:
            print("No such Aruco marker with the specified dictionary or marker id found in Opencv dictionary!!")

    if __name__ == '__main__':
        aruco_dict_type = cv2.aruco.DICT_5X5_250
        marker_id = 124  # 135
        marker_size_cm = 5
        # generateAruCoMarker(aruco_dict_type, marker_id, marker_size_cm = 5, save=False)

        try:
            frame = cv2.imread(
                "F:\TUHH\MasterThesis\Code\hyfas_calibration\Images\TemplateImage\864_downsampled_Img3.bmp")
            camera_parameter_file = "Cam_Calib_Checkerboard_Results.obj"
            stereo_parameter_file = "stereo_calib_results.obj"
            camera_calibration_parameters = Read_Write.loadCameraParameters(camera_parameter_file)
            Stereo_calibration_parameters = Read_Write.loadCameraParameters(stereo_parameter_file)
        except FileNotFoundError as e:
            print(e)

        rotation, translation = get_world_space_origin_ArucoMarker(frame, aruco_dict_type,
                                                                   camera_calibration_parameters, marker_size_cm)
        print("Rotation", rotation)
        print("\nTranslation", translation)
