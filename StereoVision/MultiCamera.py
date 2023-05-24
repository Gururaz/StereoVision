import cv2
from pypylon import genicam
from pypylon import pylon
import os

# class MultiCamera:
os.environ["PYLON_CAMEMU"] = "2"
maxCamerasToUse = 2
countOfImagesToGrab = 5
leftCameraImageSavePath = "F:/TUHH/MasterThesis/Code/Python/Images/StereoLeft/"
rightCamraImageSavePath = "F:/TUHH/MasterThesis/Code/Python/Images/StereoRight/"

converter = pylon.ImageFormatConverter()
# converting to opencv bgr format
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

img2 = pylon.PylonImage()
ipo = pylon.ImagePersistenceOptions()
quality = 100
ipo.SetQuality(quality)
# The exit code of the sample application.
exitCode = 0


def saveImages(CameraType, path, count):
    grabResult = cameras[CameraType].RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    # Print the index and the model name of the camera.
    cameraContextValue = grabResult.GetCameraContext()
    print("Camera ", cameraContextValue, ": ", cameras[cameraContextValue].GetDeviceInfo().GetModelName())
    #image = converter.Convert(grabResult)
    # img = image.GetArray()

    img2.AttachGrabResultBuffer(grabResult)
    base_filename = "Img_%d.png" % count
    dir_path = os.path.join(path, base_filename)
    print(dir_path)
    # Frame_data = img2.GetArray()
    # pylon.DisplayImage(1, grabResult);
    img2.Save(pylon.ImageFileFormat_Png, dir_path,None)
    grabResult.Release()

# Get the transport layer factory
tlFactory = pylon.TlFactory.GetInstance()

# Get all attached devices and exit application if no device is found.
devices = tlFactory.EnumerateDevices()
if len(devices) == 0:
    raise pylon.RuntimeException("No camera present.")

# Create an array of instant cameras for the found devices and avoid exceeding a maximum number of devices.
cameras = pylon.InstantCameraArray(min(len(devices), maxCamerasToUse))

# Create and attach all Pylon Devices.
for idx, cam in enumerate(cameras):
    cam.Attach(tlFactory.CreateDevice(devices[idx]))
    print("Using device ", cam.GetDeviceInfo().GetModelName())

cameras.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
count = 1
# Grab c_countOfImagesToGrab from the cameras.
print("Starting Live stream....")
while cameras.IsGrabbing():
    grabResult = cameras[0].RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)
    grabResult1 = cameras[1].RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)
    if grabResult.GrabSucceeded():
        # cameraContextValue = grabResult.GetCameraContext()
        # Now, the image data can be processeds.
        # Access the image datas
        image = converter.Convert(grabResult)
        img = image.GetArray()
        # window_name = f'Camera-{cameraContextValue:03}'
        window_name = "Left Camera"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 640, 480)
        cv2.imshow(window_name, img)

        image11 = converter.Convert(grabResult1)
        img11 = image11.GetArray()
        window_name11 = "Right Camera"
        cv2.namedWindow(window_name11, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name11, 640, 480)
        cv2.imshow(window_name11, img11)
        grabResult.Release()

    k = cv2.waitKey(1)

    if k == 27:
        break

    if k == ord("s"):
        # Right Camera
        saveImages(0, rightCamraImageSavePath, count)
        # Left camera
        saveImages(1, leftCameraImageSavePath, count)
        print('Image captured ', count)
        count += 1

    if count - 1 == countOfImagesToGrab:
        print("All images saved to the folder")
        cameras.StopGrabbing()
        cameras.Close()
        break

# except genicam.GenericException as e:
#     # Error handling
#     print("An exception occurred.", e.GetDescription())
#     exitCode = 1

print('done')
