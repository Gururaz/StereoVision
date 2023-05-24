# Import necessary libraries
import math
import os

import cv2
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split

from StereoVision import Read_Write as RW
from StereoVision import Rectify
from StereoVision import Triangulation
from StereoVision.Configuration import Data


def read_data_checkerBoard(frame_1, frame_2):
    checker_board_size = (5,8)
    size_chessboard_squares_mm = 14
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    img_grayL = cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY)
    img_grayR = cv2.cvtColor(frame_2, cv2.COLOR_BGR2GRAY)

    # Creating vector to store vectors of 2D points for each checkerboard image
    cornersL = []
    cornersR = []
    objp = np.zeros((1, checker_board_size[0] * checker_board_size[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:checker_board_size[0], 0:checker_board_size[1]].T.reshape(-1, 2)
    # specify the scale for each square in mm
    objp = objp * size_chessboard_squares_mm

    retL, cornersL = cv2.findChessboardCorners(img_grayL, checker_board_size,
                                               cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    retR, cornersR = cv2.findChessboardCorners(img_grayR, checker_board_size,
                                               cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if retL and retR:
        conv_size = (13, 13)  # (5, 5)
        cornersL = cv2.cornerSubPix(img_grayL, cornersL, conv_size, (-1, -1), criteria)
        cornersR = cv2.cornerSubPix(img_grayR, cornersR, conv_size, (-1, -1), criteria)
        img_drawL = cv2.drawChessboardCorners(frame_1, checker_board_size, cornersL, retL)
        img_drawR = cv2.drawChessboardCorners(frame_2, checker_board_size, cornersR, retR)
        pt = cornersL[0].flatten()
        cv2.putText(img=img_drawL, text="(0,0)", org=(int(pt[0]+50),int(pt[1])), fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                    fontScale=3,
                    color=(0, 255, 0), thickness=5)
        img_drawL = RW.resize(img_drawL, height=600, width=None, interMethod=cv2.INTER_LINEAR)
        img_drawR = RW.resize(img_drawR, height=600, width=None, interMethod=cv2.INTER_LINEAR)
        cv2.imshow("Cam 1", img_drawL)
        cv2.imshow("Cam 2", img_drawR)
        k = cv2.waitKey(0)
        return cornersL, cornersR, objp

    return cornersL, cornersR, objp


def triangulate_corners(cornersL, cornersR):
    world_points = []
    for i in range(len(cornersL)):
        L_pt1, R_pt1 = triangulation.corner2center_transform(cornersL[i][0][0], cornersL[i][0][1], cornersR[i][0][0],
                                                             cornersR[i][0][1])
        pt = triangulation.linear_eigen_triangulation(L_pt1, R_pt1)
        world_points.append(pt)
    return world_points


def calculate_distances(world_points_left, actual_corners):
    distances = []
    for i in range(len(world_points_left)):
        for j in range(i + 1, len(world_points_left)):
            dist_estimate = triangulation.measure_length(world_points_left[i], world_points_left[j])
            distances.append(dist_estimate)
            dist_actual = triangulation.measure_length(actual_corners[0][i],actual_corners[0][j])
            df.loc[len(df)] = [dist_actual, dist_estimate, world_points_left[i][0],world_points_left[i][1], world_points_left[i][2],world_points_left[j][0],world_points_left[j][1], world_points_left[j][2]]
    return distances


def optimize_triangulation(data):
    # Split data into features (X) and target variable (y)
    X = data.iloc[:, 2:].values
    y = data.iloc[:, 0].values

    # Split data into training, validation, and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    # Define the model
    model = Sequential()
    model.add(Dense(6, input_dim=6, activation='sigmoid'))
    model.add(Dense(5, activation='sigmoid'))
    model.add(Dense(5, activation='sigmoid'))
    model.add(Dense(1, activation='linear'))

    # Compile the model
    optimizer = keras.optimizers.Adam(learning_rate=0.1)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics='mse')

    # Fit the model to the training data
    epochs = 200
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=2, validation_data=(X_val, y_val))

    # Evaluate the model on the test data
    loss = model.evaluate(X_test, y_test)
    print("Test loss:", loss)

    # Make predictions on the test data
    predictions = model.predict(X_test)
    print(y_test)
    # Print predictions
    print(predictions)
    # accuracy = accuracy_score(y_test[0],predictions[0])
    # print(accuracy)

    # Plot the validation loss over epochs
    plt.plot(history.history['val_loss'])
    plt.title('Validation loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Validation loss')
    plt.show()

    acc = history.history['mse']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(epochs)

    plt.figure(figsize=(12, 12))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


if __name__ == '__main__':
    # # Load data from CSV file : Reading the real - component points
    data = pd.read_excel("F:\TUHH\MasterThesis\Code\hyfas_calibration\Results\WorldCB.xlsx")
    optimize_triangulation(data)
    read_features = False

    if read_features:
        imgL = cv2.imread("F:\TUHH\MasterThesis\Code\hyfas_calibration\Images\CheckerBoard\StereoLeft\Img_3.png")
        imgR = cv2.imread("F:\TUHH\MasterThesis\Code\hyfas_calibration\Images\CheckerBoard\StereoRight\Img_3.png")

        camera_parameter_file = "Cam_Calib_Checkerboard_Results.obj"
        stereo_parameter_file = "stereo_calib_results.obj"

        camera_calibration_parameters = RW.loadCameraParameters(camera_parameter_file)
        Stereo_calibration_parameters = RW.loadCameraParameters(stereo_parameter_file)

        triangulation = Triangulation.Triangulation(camera_calibration_parameters, Stereo_calibration_parameters)

        frame_1, frame_2 = Rectify.undistortRectify(imgL, imgR, show=False)

        cornersL, cornersR, actual_corners = read_data_checkerBoard(frame_1, frame_2)
        world_points_left = triangulate_corners(cornersL, cornersR)

        reprojection_point_check = 1
        triangulation.reproject(frame_1, world_points_left[reprojection_point_check], cornersL[reprojection_point_check][0],
                                cornersR[reprojection_point_check][0])

        columns = ['Target', 'estimated', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2']
        df = pd.DataFrame(columns=columns)
        distances = calculate_distances(world_points_left, actual_corners)
        print("TILL HERE")

        # Save the DataFrame to an Excel file
        # World_points_CB_file = os.path.join(Data.results_path, "WorldCB.xlsx")
        # df.to_excel(World_points_CB_file, index=False)