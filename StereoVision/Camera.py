import os
import time
from texttable import Texttable

import cv2
import numpy as np
from pypylon import pylon
from StereoVision import Configuration as CF

class Camera:

    def __init__(self, data, set_parameters=False):
        self._data = data
        self.devices = None
        self._converter = None
        self._tlFactory = None
        self._cameras = None
        self.connected_camera = []
        self._initialize_camera(set_parameters)

    def _initialize_camera(self, set_parameters):
        print("Camera Initialization...")
        os.environ["PYLON_CAMEMU"] = "2"

        # Get the transport layer factory
        self._tlFactory = pylon.TlFactory.GetInstance()
        # Get all attached devices and exit application if no device is found.
        devices = self._tlFactory.EnumerateDevices()
        if len(devices) == 0:
            raise pylon.RuntimeException("No camera present.")
        self.devices = devices

        # Create an array of instant cameras for the found devices and avoid exceeding a maximum number of devices.
        self._cameras = pylon.InstantCameraArray(min(len(devices), self._data.max_Cameras_To_Use))

        # Create and attach all Pylon Devices.
        for idx, cam in enumerate(self._cameras):
            cam.Attach(self._tlFactory.CreateDevice(self.devices[idx]))
            cameraType = cam.GetDeviceInfo().GetModelName() + " (" + cam.GetDeviceInfo().GetSerialNumber() + ")"
            self.connected_camera.append(cameraType)
            print("Connected cameras : ", cameraType)

        # converting to opencv bgr format
        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        self._cameras.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        self._cameras.Open()
        if set_parameters:
            self.set_camera_parameters(0, self._data.cam_1_parameters)
            self.set_camera_parameters(1, self._data.cam_2_parameters)

        print("Camera Initialization complete")

    def set_camera_parameters(self, camera, cam_parameters):
        self._cameras[camera].Gain.SetValue(cam_parameters['gain'])
        self._cameras[camera].Gamma.SetValue(cam_parameters['gamma'])
        self._cameras[camera].ExposureTime.SetValue(cam_parameters['exposure_time'])
        self._cameras[camera].DigitalShift.SetValue(cam_parameters['digital_shift'])

    def read_camera_parameters(self):
        my_table = Texttable()
        # Adding rows to our tabular table
        my_table.add_rows(
            [["Camera Parameters", self.connected_camera[0], self.connected_camera[1]],
             ["Gain", self._cameras[0].Gain.GetValue(), self._cameras[1].Gain.GetValue()],
             ["Gamma", self._cameras[0].Gamma.GetValue(), self._cameras[1].Gamma.GetValue()],
             ["Exposure Time", self._cameras[0].ExposureTime.GetValue(), self._cameras[1].ExposureTime.GetValue()],
             ["Digital Shift", self._cameras[0].DigitalShift.GetValue(), self._cameras[1].DigitalShift.GetValue()]])

        print('Camera parameters')
        print(my_table.draw())

    def save_image(self, cameraType, savePath, count: int):
        """
        :param cameras:
        :param cameraType:
        :param savePath:
        :param count:
        :return:
        """

        img = pylon.PylonImage()
        ipo = pylon.ImagePersistenceOptions()
        quality = 100
        ipo.SetQuality(quality)

        grabResult = self._cameras[cameraType].RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        cameraContextValue = grabResult.GetCameraContext()
        # print("Camera ", cameraContextValue + 1, ": ", self._cameras[cameraContextValue].GetDeviceInfo().GetModelName())

        img.AttachGrabResultBuffer(grabResult)
        base_filename = "Img_%d.png" % count
        dir_path = os.path.join(savePath, base_filename)
        img.Save(pylon.ImageFileFormat_Png, dir_path, None)
        grabResult.Release()

    def get_frame(self,save=False):
        # self._cameras.StartGrabbing(pylon.GrabStrategy_LatestImageOnly,
        #                             pylon.GrabLoop_ProvidedByUser)
        # self._cameras.Open()
        # while self._cameras.IsGrabbing():
        grabResult1 = self._cameras[0].RetrieveResult(5000,
                                                      pylon.TimeoutHandling_ThrowException)

        grabResult2 = self._cameras[1].RetrieveResult(5000,
                                                      pylon.TimeoutHandling_ThrowException)

        # k = cv2.waitKey(10)

        img1 = grabResult1.GetArray()
        img2 = grabResult2.GetArray()
        # print("img1", img1)
        # print("img2", img2)
        cv2.namedWindow('Acquisition', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Acquisition', 1280, 512)
        cv2.imshow('Acquisition', np.hstack([img1, img2]))
        cv2.waitKey(0)
        grabResult1.Release()
        grabResult2.Release()

        k = cv2.waitKey(1)

        if save and k == ord("s"):
            # Right Camera
            self.save_image(0, CF.Data.pattern_path, 1)
            # Left camera
            self.save_image(1, CF.Data.pattern_path, 1)
        # cv2.destroyAllWindows()
        self._cameras.StopGrabbing()
        self._cameras.Close()
        cv2.destroyAllWindows()
        return img1, img2

    def read_corners(self, gray_img):
        gray_image = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        checker_board_size = (8, 5)
        ret, corners = cv2.findChessboardCorners(gray_image, checker_board_size,
                                                 cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret:
            conv_size = (13, 13)  # (5, 5)
            corners = cv2.cornerSubPix(gray_image, corners, conv_size, (-1, -1), criteria)
            img_draw = cv2.drawChessboardCorners(gray_img, checker_board_size, corners, ret)
            return img_draw
        else:
            return gray_img

    def stereo_stream(self, left_camera_path, right_camera_path, save_image=False):
        print("Starting Live stream....")

        # used to record the time when we processed last frame
        prev_frame_time = 0
        # used to record the time at which we processed current frame
        new_frame_time = 0
        self.read_camera_parameters()
        count = 1
        while self._cameras.IsGrabbing():
            new_frame_time = time.time()

            Cam1 = self._cameras[0].RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            Cam2 = self._cameras[1].RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

            if Cam1.GrabSucceeded() and Cam2.GrabSucceeded():
                fps = 1 / (new_frame_time - prev_frame_time)
                prev_frame_time = new_frame_time
                fps = str(int(fps))
                leftImage = self.converter.Convert(Cam1)
                leftImg = leftImage.GetArray()

                rightImage = self.converter.Convert(Cam2)
                rightImg = rightImage.GetArray()

                # img_h = cv2.hconcat([leftImg, rightImg])
                # cv2.resizeWindow(img_h, 640, 480)
                # cv2.imshow('Stereo Camera', img_h)
                # putting the FPS count on the frame

                windowNameLeft = "Camera 1"
                cv2.namedWindow(windowNameLeft, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(windowNameLeft, 640, 480)
                #L_img_draw = self.read_corners(leftImg)
                cv2.putText(leftImg, fps, (250, 250), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 10, cv2.LINE_AA)
                cv2.imshow(windowNameLeft, leftImg)

                windowNameRight = "Camera 2"
                cv2.namedWindow(windowNameRight, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(windowNameRight, 640, 480)
                #R_img_draw = self.read_corners(rightImg)
                cv2.putText(rightImg, fps, (250, 250), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 10, cv2.LINE_AA)
                cv2.imshow(windowNameRight, rightImg)

                Cam1.Release()
                Cam2.Release()

                k = cv2.waitKey(1)

                if save_image and k == ord("s"):
                    # Right Camera
                    self.save_image(0, left_camera_path, count)
                    # Left camera
                    self.save_image(1, right_camera_path, count)
                    print('Image captured ', count)
                    count += 1

                if count - 1 == self._data.countOfImagesToGrab or k == 27:
                    print("Live stream ended...")
                    # print("Images saved to the folder : %d" % (count - 1))
                    break

        self._cameras.StopGrabbing()
        self._cameras.Close()
        cv2.destroyAllWindows()

    def readFromWebCam(path):
        # Open a connection to the webcam
        cap = cv2.VideoCapture(1)
        count = 1
        while count < 15:
            # Capture a frame from the webcam
            ret, frame = cap.read()

            if ret > 0:
                # Display the frame
                cv2.imshow("Webcam", frame)

                # Wait for the user to press the 's' key
                key = cv2.waitKey(1) & 0xFF
                if key == ord("s"):
                    # Save the frame to the specified directory
                    base_filename = "Img_%d.png" % count
                    dir_path = os.path.join(path, base_filename)
                    cv2.imwrite(dir_path, frame)
                    count = count + 1
                # Exit the loop if the 'q' key is pressed
                if key == ord("q"):
                    quit()
                    break

        # Release the webcam and close the display window
        cap.release()
        cv2.destroyAllWindows()
