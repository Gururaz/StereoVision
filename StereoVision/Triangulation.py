import cv2
import numpy as np
import math
import numpy.linalg as npla
from scipy import linalg
from StereoVision import Read_Write


class Triangulation:

    def __init__(self, camera_calibration_parameters, Stereo_calibration_parameters):
        # P1 = cv2.projectionFromKRt(camera_calibration_parameters['C1_camera_matrix'], np.eye(3, 3), np.zeros((3, 1)))
        # P2 = cv2.projectionFromKRt(camera_calibration_parameters['C2_camera_matrix'],
        #                            Stereo_calibration_parameters['stereo_rotation_C1_C2'],
        #                            Stereo_calibration_parameters['stereo_translation_C1_C2'])

        # Camera 1 parameters
        self.K1 = camera_calibration_parameters['C1_camera_matrix']
        self.d1 = camera_calibration_parameters['C1_distortion_coeff']
        self.OK1 = camera_calibration_parameters['C1_optimal_Camera_matrix']
        # Camera 2 parameters
        self.K2 = camera_calibration_parameters['C2_camera_matrix']
        self.d2 = camera_calibration_parameters['C2_distortion_coeff']
        self.OK2 = camera_calibration_parameters['C2_optimal_Camera_matrix']

        # Stereo parameters
        self.R1 = np.eye(3, 3)  # Rotation matrix of camera 1
        self.t1 = np.zeros((3, 1))  # Translation vector of camera 1
        self.R12 = Stereo_calibration_parameters['Rectification_transform_R_C2'] #Stereo_calibration_parameters['stereo_rotation_C1_C2']
        # self.t12 = Stereo_calibration_parameters['stereo_translation_C1_C2']

        # Rectification parameters
        self.RRC1 = Stereo_calibration_parameters['Rectification_transform_R_C1']
        self.RRC2 = Stereo_calibration_parameters['Rectification_transform_R_C2']
        self.projMat1 = Stereo_calibration_parameters['Rectified_projection_matrix_P1']
        self.projMat2 = Stereo_calibration_parameters['Rectified_projection_matrix_P2']
        self.RK1 = cv2.decomposeProjectionMatrix(self.projMat1)[0]
        self.RK2 = cv2.decomposeProjectionMatrix(self.projMat2)[0]
        self.Rt12 = cv2.decomposeProjectionMatrix(self.projMat2)[2]
        self.Rt12 = (self.Rt12 [:3, :] / self.Rt12[3, :]).T
        print("")

        # Compute the projection matrix
        # self.projMat1 = self.K1 @ cv2.hconcat([self.R1, self.t1])  # Cam1 is the origin
        # self.projMat2 = self.K2 @ cv2.hconcat([self.R12, self.t12])  # R, T from stereoCalibrate

        # P1 = np.hstack([self.R1, self.R1.dot(self.t1)])
        # P2 = np.hstack([self.R12, self.R12.dot(self.t12)])
        # self.projMat1 = self.K1.dot(P1)
        # self.projMat2 = self.K2.dot(P2)

        # self.projMat1 = np.hstack((np.eye(3, 3), np.zeros((3, 1))))
        # self.projMat1 = np.dot(self.K1, self.projMat1)
        # self.projMat2 = np.hstack((self.R12, self.t12))
        # self.projMat2 = np.dot(self.K2, self.projMat2)

    # def pixel2cam_normalize(self, pt, K):
    #     u = (pt[0] - K[0][2]) / K[0][0]
    #     v = (pt[1] - K[1][2]) / K[1][1]
    #     return np.array([u, v], dtype=np.float32)

    def corner2center_transform(self, L_point_x, L_point_y, R_point_x, R_point_y):
        # pt1 = np.asarray((L_point_x, L_point_y)).reshape(1, 1, 2)
        # pt2 = np.asarray((R_point_x, R_point_y)).reshape(1, 1, 2)
        # pts1, pts2 = cv2.correctMatches(F, pt1, pt2)
        #
        #
        # # The transformation of coordinates from the top left end (P) to the centre of the image
        #
        # # Method 1 : Using principle points of the camera (Center)
        # L_point_x, L_point_y = (pts1[0][0][0] - self.K1[0][2]), (self.K1[1][2] - pts1[0][0][1])
        # # camera_calibration_parameters['C1_camera_matrix'][0][0], camera_calibration_parameters['C1_camera_matrix'][1][1]
        # R_point_x, R_point_y = (pts2[0][0][0] - self.K2[0][2]), (self.K2[1][2] - pts2[0][0][1])
        # # camera_calibration_parameters['C2_camera_matrix'][0][0],camera_calibration_parameters['C2_camera_matrix'][1][1]

        L_point_x, L_point_y = (L_point_x - self.RK1[0][2]), (L_point_y - self.RK1[1][2])
        R_point_x, R_point_y = (R_point_x - self.RK2[0][2]), (R_point_y - self.RK2[1][2])

        L_pt1, R_pt1 = (L_point_x, L_point_y), (R_point_x, R_point_y)
        # L_pt2, R_pt2 = (R_point_x, R_point_y)

        return L_pt1, R_pt1


    def measure_length(self, p1, p2):

        dist = math.sqrt(math.pow(p2[0] - p1[0], 2) +
                      math.pow(p2[1] - p1[1], 2) +
                      math.pow(p2[2] - p1[2], 2) * 1.0)
        # print("Distance in cm is ", dist/10.0)
        return dist

    def reproject(self,frame_1, point1, L_pt1, R_pt1):
        #, L_pt2, R_pt2):
        # Reproject back into the Camera 0
        worldPoint_1 = np.array(point1)
        # worldPoint_2 = np.array(point2)

        rvec1, _ = cv2.Rodrigues(self.R1.T)

        p1, _ = cv2.projectPoints(worldPoint_1, rvec1, self.t1, self.RK1, None)
        # p2, _ = cv2.projectPoints(worldPoint_2, rvec1, self.t1, self.RK1, None)

        p1 = p1.flatten()
        # p2 = p2.flatten()
        # x = np.array(L_pt1,R_pt1)
        # # y = np.array(L_pt2, R_pt2)
        # # measure difference between original image point and reporjected image point
        # reprojection_error1 = np.linalg.norm(x - p1[:])
        # # reprojection_error2 = np.linalg.norm(y - p2[:])

        p1[0] = p1[0] + self.RK1[0][2]
        p1[1] = p1[1] + self.RK1[1][2]
        pt1 = p1.astype(int)

        # p2[0] = p2[0] + self.RK1[0][2]
        # p2[1] = p2[1] + self.RK1[1][2]
        # pt2 = p2.astype(int)

        # print("reprojection_error1 : ", reprojection_error1)
        # print("reprojection_error2 : ", reprojection_error2)

        x1 = int(L_pt1[0] + self.RK1[0][2])
        y1 = int(L_pt1[1] + self.RK1[1][2])
        # x2 = int(L_pt2[0] + self.RK1[0][2])
        # y2 = int(L_pt2[1] + self.RK1[1][2])

        color1 = (list(np.random.choice(range(256), size=3)))
        color = [int(color1[0]), int(color1[1]), int(color1[2])]

        # cv2.drawMarker(frame_1, (x1,y1), (255, 0, 0), cv2.MARKER_CROSS, 30, 10)
        cv2.drawMarker(frame_1, pt1, color, cv2.MARKER_CROSS, 50, 10)

        # cv2.drawMarker(frame_1, (x2,y2), (255, 0, 0), cv2.MARKER_STAR, 30, 10)
        # # cv2.drawMarker(frame_1, pt2, color, cv2.MARKER_STAR, 30, 10)

        crop = Read_Write.resize(frame_1, 800)
        cv2.imshow("Reprojected triangulated point", crop)
        cv2.waitKey(0)

        return 0 # reprojection_error1 #,reprojection_error2

    def estimate_depth(self, u, v, frame_left, frame_right, baseline, alpha=61.2):
        # CONVERT FOCAL LENGTH f FROM [mm] TO [pixel]:
        height_right, width_right = frame_right.shape
        height_left, width_left = frame_left.shape

        f_pixel = self.K1[0][0]
        # if width_right == width_left:
        #     f_pixel = (width_right * 0.5) / np.tan(alpha * 0.5 * np.pi / 180)
        #
        # else:
        #     print('Left and right camera frames do not have the same pixel width')

        # x_right = v[0]-0.5 * width_right
        # y_right = 0.5 * height_right - v[1]
        #
        # x_left = u[0]-0.5 * width_left
        # y_left = 0.5*height_left - u[1]

        x_left = u[0]
        x_right = v[0]

        y_left = u[1]
        # y_right = v[1]

        # CALCULATE THE DISPARITY:
        disparity = x_left - x_right  # Displacement between left and right frames [pixels]

        # CALCULATE DEPTH z:
        zDepth = (-baseline * f_pixel) / disparity  # Depth in [cm]
        # X = baseline * x_left / disparity
        # Y = baseline * y_left / disparity

        X = x_left * zDepth / f_pixel
        Y = y_left * zDepth / f_pixel

        # X = (baseline * x_left - (self.K1[0][2])) / disparity
        # Y = (baseline * ((y_left + y_right) / 2) - (self.K1[1][2])) / disparity

        return zDepth, Y, X

    def linear_triangulation(self, u1, u2):

        """
            Linear triangulation (Hartley ch 12.2 pg 312) to find the 3D point X
            where u = p1 * X and v = p2 * X. Solve AX = 0.
            :param u1, u2: 2D points in homo. or catesian coordinates. Shape (3 x n)
            :param projMat1, projMat2: Camera matrices associated with u and v. Shape (3 x 4)
            :returns: 4 x n homogenous 3d triangulated points
        """
        u1 = np.array(u1, dtype=np.float32)
        u1 = np.reshape(u1, [1, -1]).T
        u2 = np.array(u2, dtype=np.float32)
        u2 = np.reshape(u2, [1, -1]).T

        num_points = u1.shape[1]
        res = np.ones((4, num_points))

        # P1 = self.K1 @ cv2.hconcat([np.eye(3), np.zeros((3, 1))])
        # P2 = self.K2 @ cv2.hconcat([R, t])

        for i in range(num_points):
            A = np.asarray([
                (u1[0, i] * self.projMat1[2, :] - self.projMat1[0, :]),
                (u1[1, i] * self.projMat1[2, :] - self.projMat1[1, :]),
                (u2[0, i] * self.projMat2[2, :] - self.projMat2[0, :]),
                (u2[1, i] * self.projMat2[2, :] - self.projMat2[1, :])
            ])

            _, _, V = np.linalg.svd(A)
            X = V[-1, :4]
            res[:, i] = X / X[3]

        return res[:3].flatten()

    def linear_eigen_triangulation(self, u1, u2):
        #
        u1 = np.array(u1, dtype=np.float32)
        u1 = np.reshape(u1, [1, 1, -1])
        u2 = np.array(u2, dtype=np.float32)
        u2 = np.reshape(u2, [1, 1, -1])

        arr1 = np.array(u1).reshape((2, 1))
        arr2 = np.array(u2).reshape((2, 1))

        # P1 = self.K1 @ cv2.hconcat([np.eye(3), np.zeros((3, 1))])
        # P2 = self.K2 @ cv2.hconcat([R, t])

        # u1 = np.array([[u1[0]], [u1[1]]])
        # u2 = np.array([[u2[0]], [u2[1]]])

        # Lx_norm = (u1[0] - (self.K1[0][2])) / \
        #           self.K1[0][0]
        # Ly_norm = (u1[1] - (self.K1[1][2])) / \
        #           self.K1[1][1]
        # Rx_norm = (u2[0] - (self.K2[0][2])) / \
        #           self.K2[0][0]
        # Ry_norm = (u2[1] - (self.K2[1][2])) / \
        #           self.K2[1][1]
        # pts1 = np.asarray((Lx_norm, Ly_norm))
        # pts2 = np.asarray((Rx_norm, Ry_norm))
        #
        # #x = cv2.triangulatePoints(self.projMat1, self.projMat2, pts1, pts2)  # OpenCV's Linear-Eigen triangl
        # x = cv2.triangulatePoints(self.projMat1, self.projMat2, pts1.T, pts2.T)  # OpenCV's Linear-Eigen triangl
        # x[0:3, :] = x[0:3, :] / x[3:4, :]  # normalize coordinates

        # cam_pts_1 = self.pixel2cam_normalize(u1, self.K1).reshape(2, 1)
        # cam_pts_2 = self.pixel2cam_normalize(u2, self.K2).reshape(2, 1)

        points4d = cv2.triangulatePoints(self.projMat1, self.projMat2, arr1, arr2)  # points1u.T, points2u.T)
        points3d = (points4d[:3, :] / points4d[3, :]).T
        # R_pt = self.R12.dot(points3d.flatten() - self.Rt12.flatten().T)
        # print("Right point", R_pt)
        return points3d.flatten()

    def linear_LS_triangulation(self, u1, u2):
        """
        Linear Least Squares based triangulation.
        Relative speed: 0.1

        (u1, P1) is the reference pair containing normalized image coordinates (x, y) and the corresponding camera matrix.
        (u2, P2) is the second pair.

        u1 and u2 are matrices: amount of points equals #rows and should be equal for u1 and u2.

        The status-vector will be True for all points.
        """

        # Initialize consts to be used in linear_LS_triangulation()
        linear_LS_triangulation_C = -np.eye(2, 3)

        u1 = np.array(u1, dtype=np.float32)
        u1 = np.reshape(u1, [1, 1, -1])
        u2 = np.array(u2, dtype=np.float32)
        u2 = np.reshape(u2, [1, 1, -1])

        A = np.zeros((4, 3))
        b = np.zeros((4, 1))

        # Create array of triangulated points
        x = np.zeros((3, len(u1)))

        # Initialize C matrices
        C1 = np.array(linear_LS_triangulation_C)
        C2 = np.array(linear_LS_triangulation_C)

        for i in range(len(u1)):
            # Derivation of matrices A and b:
            # for each camera following equations hold in case of perfect point matches:
            #     u.x * (P[2,:] * x)     =     P[0,:] * x
            #     u.y * (P[2,:] * x)     =     P[1,:] * x
            # and imposing the constraint:
            #     x = [x.x, x.y, x.z, 1]^T
            # yields:
            #     (u.x * P[2, 0:3] - P[0, 0:3]) * [x.x, x.y, x.z]^T     +     (u.x * P[2, 3] - P[0, 3]) * 1     =     0
            #     (u.y * P[2, 0:3] - P[1, 0:3]) * [x.x, x.y, x.z]^T     +     (u.y * P[2, 3] - P[1, 3]) * 1     =     0
            # and since we have to do this for 2 cameras, and since we imposed the constraint,
            # we have to solve 4 equations in 3 unknowns (in LS sense).

            # Build C matrices, to construct A and b in a concise way
            C1[:, 2] = u1[i, :]
            C2[:, 2] = u2[i, :]

            # Build A matrix:
            # [
            #     [ u1.x * P1[2,0] - P1[0,0],    u1.x * P1[2,1] - P1[0,1],    u1.x * P1[2,2] - P1[0,2] ],
            #     [ u1.y * P1[2,0] - P1[1,0],    u1.y * P1[2,1] - P1[1,1],    u1.y * P1[2,2] - P1[1,2] ],
            #     [ u2.x * P2[2,0] - P2[0,0],    u2.x * P2[2,1] - P2[0,1],    u2.x * P2[2,2] - P2[0,2] ],
            #     [ u2.y * P2[2,0] - P2[1,0],    u2.y * P2[2,1] - P2[1,1],    u2.y * P2[2,2] - P2[1,2] ]
            # ]
            A[0:2, :] = C1.dot(self.projMat1[0:3, 0:3])  # C1 * R1
            A[2:4, :] = C2.dot(self.projMat2[0:3, 0:3])  # C2 * R2

            # Build b vector:
            # [
            #     [ -(u1.x * P1[2,3] - P1[0,3]) ],
            #     [ -(u1.y * P1[2,3] - P1[1,3]) ],
            #     [ -(u2.x * P2[2,3] - P2[0,3]) ],
            #     [ -(u2.y * P2[2,3] - P2[1,3]) ]
            # ]
            b[0:2, :] = C1.dot(self.projMat1[0:3, 3:4])  # C1 * t1
            b[2:4, :] = C2.dot(self.projMat2[0:3, 3:4])  # C2 * t2
            b *= -1

            # Solve for x vector
            cv2.solve(A, b, x[:, i:i + 1], cv2.DECOMP_SVD)

        return x.T.astype(float).flatten()

    def polynomial_triangulation(self, u1, u2):
        """
        Polynomial (Optimal) triangulation.
        Uses Linear-Eigen for final triangulation.
        Relative speed: 0.1

        (u1, P1) is the reference pair containing normalized image coordinates (x, y) and the corresponding camera matrix.
        (u2, P2) is the second pair.

        u1 and u2 are matrices: amount of points equals #rows and should be equal for u1 and u2.

        The status-vector is based on the assumption that all 3D points have finite coordinates.
        """

        P1_full = np.eye(4)
        P1_full[0:3, :] = self.projMat1[0:3, :]  # convert to 4x4
        P2_full = np.eye(4)
        P2_full[0:3, :] = self.projMat2[0:3, :]  # convert to 4x4
        P_canon = P2_full.dot(cv2.invert(P1_full)[1])  # find canonical P which satisfies P2 = P_canon * P1

        # "F = [t]_cross * R" [HZ 9.2.4]; transpose is needed for numpy
        F = np.cross(P_canon[0:3, 3], P_canon[0:3, 0:3], axisb=0).T

        # Other way of calculating "F" [HZ (9.2)]
        # op1 = (P2[0:3, 3:4] - P2[0:3, 0:3] .dot (cv2.invert(P1[0:3, 0:3])[1]) .dot (P1[0:3, 3:4]))
        # op2 = P2[0:3, 0:4] .dot (cv2.invert(P1_full)[1][0:4, 0:3])
        # F = np.cross(op1.reshape(-1), op2, axisb=0).T

        u1 = np.array(u1, dtype=np.float32)
        u1 = np.reshape(u1, [1, 1, -1])
        u2 = np.array(u2, dtype=np.float32)
        u2 = np.reshape(u2, [1, 1, -1])

        # Project 2D matches to closest pair of epipolar lines
        u1_new, u2_new = cv2.correctMatches(F, u1.reshape(1, len(u1), 2), u2.reshape(1, len(u1), 2))

        # For a purely sideways trajectory of 2nd cam, correctMatches() returns NaN for all possible points!
        if np.isnan(u1_new).all() or np.isnan(u2_new).all():
            F = cv2.findFundamentalMat(u1, u2, cv2.FM_8POINT)[0]  # so use a noisy version of the fund mat
            u1_new, u2_new = cv2.correctMatches(F, u1.reshape(1, len(u1), 2), u2.reshape(1, len(u1), 2))

        # Triangulate using the refined image points
        return self.linear_LS_triangulation(u1_new[0], u2_new[0]).flatten()
        # TODO: replace with linear_LS: better results for points not at Inf

    def Iterative_LS_triangulation(self, u1, u2, tolerance=3.e-5):
        """
        Iterative (Linear) Least Squares based triangulation.
        From "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997.
        Relative speed: 0.025

        (u1, P1) is the reference pair containing normalized image coordinates (x, y) and the corresponding camera matrix.
        (u2, P2) is the second pair.
        "tolerance" is the depth convergence tolerance.

        Additionally returns a status-vector to indicate outliers:
            1: inlier, and in front of both cameras
            0: outlier, but in front of both cameras
            -1: only in front of second camera
            -2: only in front of first camera
            -3: not in front of any camera
        Outliers are selected based on non-convergence of depth, and on negativity of depths (=> behind camera(s)).

        u1 and u2 are matrices: amount of points equals #rows and should be equal for u1 and u2.
        """
        # Initialize consts to be used in iterative_LS_triangulation()
        iterative_LS_triangulation_C = -np.eye(2, 3)

        u1 = np.array(u1, dtype=np.float32)
        u1 = np.reshape(u1, [1, 1, -1])
        u2 = np.array(u2, dtype=np.float32)
        u2 = np.reshape(u2, [1, 1, -1])

        A = np.zeros((4, 3))
        b = np.zeros((4, 1))

        # Create array of triangulated points
        x = np.empty((4, len(u1)))
        x[3, :].fill(1)  # create empty array of homogenous 3D coordinates
        x_status = np.empty(len(u1), dtype=int)

        # Initialize C matrices
        C1 = np.array(iterative_LS_triangulation_C)
        C2 = np.array(iterative_LS_triangulation_C)

        for xi in range(len(u1)):
            # Build C matrices, to construct A and b in a concise way
            C1[:, 2] = u1[xi, :]
            C2[:, 2] = u2[xi, :]

            # Build A matrix
            A[0:2, :] = C1.dot(self.projMat1[0:3, 0:3])  # C1 * R1
            A[2:4, :] = C2.dot(self.projMat2[0:3, 0:3])  # C2 * R2

            # Build b vector
            b[0:2, :] = C1.dot(self.projMat1[0:3, 3:4])  # C1 * t1
            b[2:4, :] = C2.dot(self.projMat2[0:3, 3:4])  # C2 * t2
            b *= -1

            # Init depths
            d1 = d2 = 1.
            i = 0
            for i in range(10):  # Hartley suggests 10 iterations at most
                # Solve for x vector
                # x_old = np.array(x[0:3, xi])    # TODO: remove
                cv2.solve(A, b, x[0:3, xi:xi + 1], cv2.DECOMP_SVD)

                # Calculate new depths
                d1_new = self.projMat1[2, :].dot(x[:, xi])
                d2_new = self.projMat2[2, :].dot(x[:, xi])

                # Convergence criterium
                # print i, d1_new - d1, d2_new - d2, (d1_new > 0 and d2_new > 0)    # TODO: remove
                # print i, (d1_new - d1) / d1, (d2_new - d2) / d2, (d1_new > 0 and d2_new > 0)    # TODO: remove
                # print i, np.sqrt(np.sum((x[0:3, xi] - x_old)**2)), (d1_new > 0 and d2_new > 0)    # TODO: remove
                # print i, u1[xi, :] - P1[0:2, :].dot(x[:, xi]) / d1_new, u2[xi, :] - P2[0:2, :].dot(x[:, xi]) / d2_new
                # print bool(i) and ((d1_new - d1) / (d1 - d_old), (d2_new - d2) / (d2 - d1_old), (d1_new > 0 and d2_new > 0))    # TODO: remove
                # if abs(d1_new - d1) <= tolerance and abs(d2_new - d2) <= tolerance: print "Orig cond met"
                if abs(d1_new - d1) <= tolerance and \
                        abs(d2_new - d2) <= tolerance:
                    # if i and np.sum((x[0:3, xi] - x_old)**2) <= 0.0001**2:
                    # if abs((d1_new - d1) / d1) <= 3.e-6 and \
                    # abs((d2_new - d2) / d2) <= 3.e-6: #and \
                    # abs(d1_new - d1) <= tolerance and \
                    # abs(d2_new - d2) <= tolerance:
                    # if i and 1 - abs((d1_new - d1) / (d1 - d_old)) <= 1.e-2 and \    # TODO: remove
                    # 1 - abs((d2_new - d2) / (d2 - d1_old)) <= 1.e-2 and \    # TODO: remove
                    # abs(d1_new - d1) <= tolerance and \    # TODO: remove
                    # abs(d2_new - d2) <= tolerance:    # TODO: remove
                    break

                # Re-weight A matrix and b vector with the new depths
                A[0:2, :] *= 1 / d1_new
                A[2:4, :] *= 1 / d2_new
                b[0:2, :] *= 1 / d1_new
                b[2:4, :] *= 1 / d2_new

                # Update depths
                # d_old = d1    # TODO: remove
                # d1_old = d2    # TODO: remove
                d1 = d1_new
                d2 = d2_new

            # Set status
            x_status[xi] = (i < 10 and  # points should have converged by now
                            (d1_new > 0 and d2_new > 0))  # points should be in front of both cameras
            if d1_new <= 0:
                x_status[xi] -= 1
            if d2_new <= 0:
                x_status[xi] -= 2

        return x[0:3, :].T.astype(float).flatten()

    def midpoint_triangulate(self, x, cam):
        """
        Args:
            x:   Set of 2D points in homogeneous coords, (3 x n) matrix
            cam: Collection of n objects, each containing member variables
                     cam.P - 3x4 camera matrix
                     cam.R - 3x3 rotation matrix
                     cam.T - 3x1 translation matrix
        Returns:
            midpoint: 3D point in homogeneous coords, (4 x 1) matrix
        """

        n = len(cam)  # No. of cameras

        I = np.eye(3)  # 3x3 identity matrix
        A = np.zeros((3, n))
        B = np.zeros((3, n))
        sigma2 = np.zeros((3, 1))

        for i in range(n):
            a = -np.transpose(cam[i].R).dot(cam[i].T)  # ith camera position
            A[:, i, None] = a

            b = npla.pinv(cam[i].P).dot(x[:, i])  # Directional vector
            b = b / b[3]
            b = b[:3, None] - a
            b = b / npla.norm(b)
            B[:, i, None] = b

            sigma2 = sigma2 + b.dot(b.T.dot(a))

        C = (n * I) - B.dot(B.T)
        Cinv = npla.inv(C)
        sigma1 = np.sum(A, axis=1)[:, None]
        m1 = I + B.dot(np.transpose(B).dot(Cinv))
        m2 = Cinv.dot(sigma2)

        midpoint = (1 / n) * m1.dot(sigma1) - m2
        return np.vstack((midpoint, 1))

    def DLT_compute(self, P1, P2, point1, point2):

        A = [point1[1] * P1[2, :] - P1[1, :],
             P1[0, :] - point1[0] * P1[2, :],
             point2[1] * P2[2, :] - P2[1, :],
             P2[0, :] - point2[0] * P2[2, :]
             ]
        A = np.array(A).reshape((4, 4))
        # print('A: ')
        # print(A)

        B = A.transpose() @ A

        U, s, Vh = linalg.svd(B, full_matrices=False)
        return Vh[3, 0:3] / Vh[3, 3]

    def dlt(self, u1, u2):

        # uvs1 = [[458, 86], [451, 164], [287, 181],
        #         [196, 383], [297, 444], [564, 194]]
        #
        # uvs2 = [[540, 311], [603, 359], [542, 378],
        #         [525, 507], [485, 542], [691, 352]]

        uvs1 = [[u1[0], u1[1]]]
        uvs2 = [[u2[0], u2[1]]]
        uvs1 = np.array(uvs1)
        uvs2 = np.array(uvs2)

        p3ds = []
        for uv1, uv2 in zip(uvs1, uvs2):
            _p3d = self.DLT_compute(self.projMat1, self.projMat2, uv1, uv2)
            p3ds.append(_p3d)

        p3ds = np.array(p3ds)
        return p3ds.flatten()
