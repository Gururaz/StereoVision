# from asw import asw
# from pydensecrf import densecrf
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Cursor


# def show_disparity_with_CRF(left_img, right_img, show=False):
#     # source code : https://github.com/lucasb-eyer/pydensecrf
#
#     left_img = np.uint8(255 * left_img)
#     right_img = np.uint8(255 * right_img)
#
#     # Compute cost volume
#     cost_volume = np.abs(left_img[:, :, np.newaxis] - right_img[:, :, np.newaxis])  # .transpose(1, 0, 2)))
#
#     # Initialize dense CRF
#     d = densecrf.DenseCRF(left_img.shape[1] * left_img.shape[0], 2)
#
#     # Set unary potentials
#     U = cost_volume.reshape((-1, 2))
#     d.setUnaryEnergy(U)
#
#     # Set pairwise potentials
#     pairwise_energy = densecrf.create_pairwise_bilateral(sdims=(10, 10), schan=(0.01,), img=left_img, chdim=2)
#     d.addPairwiseEnergy(pairwise_energy, compat=1)
#
#     # Infer the optimal label map
#     Q = d.inference(5)
#     disparity = np.argmax(Q, axis=1).reshape(left_img.shape[:2])
#     # plot the result
#
#     if show:
#         plt.imshow(disparity)
#         plt.colorbar()
#         plt.axis('off')
#         plt.show()
#
#     return disparity

def display_corresponding_points(frame_1, frame_2, Lpt1, Rpt1, Lpt2, Rpt2, camera_parameters):
    fig, axes = plt.subplots(1, 2, figsize=(15, 10))
    # plt.suptitle("Original Images", verticalalignment='top')

    axes[0].imshow(frame_1, cmap="gray")
    axes[1].imshow(frame_2, cmap="gray")
    axes[0].axhline(camera_parameters['C1_camera_matrix'][1][2])
    axes[0].axvline(camera_parameters['C1_camera_matrix'][0][2])
    axes[1].axhline(camera_parameters['C2_camera_matrix'][1][2])
    axes[1].axvline(camera_parameters['C2_camera_matrix'][0][2])
    axes[0].plot(Lpt1[0], Lpt1[1], 'r*')
    axes[0].plot(Lpt2[0], Lpt2[1], 'r*')
    axes[1].plot(Rpt1[0], Rpt1[1], 'r*')
    axes[1].plot(Rpt2[0], Rpt2[1], 'r*')
    plt.suptitle("Keypoint Match")
    plt.show()


def disparity_loop(img1, img2):
    cv2.namedWindow('disp', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('disp', 600, 600)

    cv2.createTrackbar('numDisparities', 'disp', 1, 17, nothing)
    cv2.createTrackbar('blockSize', 'disp', 5, 50, nothing)
    cv2.createTrackbar('preFilterType', 'disp', 1, 1, nothing)
    cv2.createTrackbar('preFilterSize', 'disp', 2, 25, nothing)
    cv2.createTrackbar('preFilterCap', 'disp', 5, 62, nothing)
    cv2.createTrackbar('textureThreshold', 'disp', 10, 100, nothing)
    cv2.createTrackbar('uniquenessRatio', 'disp', 15, 100, nothing)
    cv2.createTrackbar('speckleRange', 'disp', 0, 100, nothing)
    cv2.createTrackbar('speckleWindowSize', 'disp', 3, 25, nothing)
    cv2.createTrackbar('disp12MaxDiff', 'disp', 5, 25, nothing)
    cv2.createTrackbar('minDisparity', 'disp', 5, 25, nothing)

    # Creating an object of StereoBM algorithm
    stereo = cv2.StereoBM_create()

    while True:

        # Updating the parameters based on the trackbar positions
        numDisparities = cv2.getTrackbarPos('numDisparities', 'disp') * 16
        blockSize = cv2.getTrackbarPos('blockSize', 'disp') * 2 + 5
        preFilterType = cv2.getTrackbarPos('preFilterType', 'disp')
        preFilterSize = cv2.getTrackbarPos('preFilterSize', 'disp') * 2 + 5
        preFilterCap = cv2.getTrackbarPos('preFilterCap', 'disp')
        textureThreshold = cv2.getTrackbarPos('textureThreshold', 'disp')
        uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio', 'disp')
        speckleRange = cv2.getTrackbarPos('speckleRange', 'disp')
        speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize', 'disp') * 2
        disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff', 'disp')
        minDisparity = cv2.getTrackbarPos('minDisparity', 'disp')

        # Setting the updated parameters before computing disparity map
        stereo.setNumDisparities(numDisparities)
        stereo.setBlockSize(blockSize)
        stereo.setPreFilterType(preFilterType)
        stereo.setPreFilterSize(preFilterSize)
        stereo.setPreFilterCap(preFilterCap)
        stereo.setTextureThreshold(textureThreshold)
        stereo.setUniquenessRatio(uniquenessRatio)
        stereo.setSpeckleRange(speckleRange)
        stereo.setSpeckleWindowSize(speckleWindowSize)
        stereo.setDisp12MaxDiff(disp12MaxDiff)
        stereo.setMinDisparity(minDisparity)

        # Calculating disparity using the StereoBM algorithm
        disparity = stereo.compute(img2, img1)
        # NOTE: Code returns a 16bit signed single channel image,
        # CV_16S containing a disparity map scaled by 16. Hence it
        # is essential to convert it to CV_32F and scale it down 16 times.

        # Converting to float32
        disparity = disparity.astype(np.float32)

        # Scaling down the disparity values and normalizing them
        disparity = (disparity / 16.0 - minDisparity) / numDisparities

        # Displaying the disparity map
        cv2.imshow("disp", disparity)

        # Close window using esc key
        if cv2.waitKey(1) == 27:
            break
    return disparity


#
# def show_disparity_with_SGblockMatching(imgL, imgR, show=False):
#     # Source : https://github.com/aliyasineser/stereoDepth/blob/master/stereo_depth.py
#     """ Depth map calculation. Works with SGBM and WLS. Need rectified images, returns depth map ( left to right disparity ) """
#     # SGBM Parameters -----------------
#     window_size = 15  # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
#
#     left_matcher = cv2.StereoSGBM_create(
#         minDisparity=-1,
#         numDisparities=1 * 16,  # max_disp has to be dividable by 16 f. E. HH 192, 256
#         blockSize=window_size)
#     # P1 = 8 * 3 * window_size,
#     # # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
#     # P2 = 32 * 3 * window_size,
#     # disp12MaxDiff = 12,
#     # uniquenessRatio = 10,
#     # speckleWindowSize = 50,
#     # speckleRange = 32,
#     # preFilterCap = 63,
#     # mode = cv2.STEREO_SGBM_MODE_SGBM_3WAY
#     right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
#     # FILTER Parameters
#     lmbda = 80000
#     sigma = 0.08
#     visual_multiplier = 5
#
#     wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
#     wls_filter.setLambda(lmbda)
#
#     wls_filter.setSigmaColor(sigma)
#     displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
#     dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
#     displ = np.int16(displ)
#     dispr = np.int16(dispr)
#     filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!
#
#     filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
#     filteredImg = np.uint8(filteredImg)
#
#     if show:
#         plt.imshow(filteredImg)
#         plt.colorbar()
#         plt.axis('off')
#         plt.show()
#
#     return filteredImg


def nothing(x):
    pass


# def show_disparity_with_adaptiveSupportWeight(img1, img2):
# #     # Compute disparity map using ASW algorithm
# #     disp = asw(img1, img2, max_disp=64, r=5, e=5)
# #
# #     # Display the disparity map
# #     cv2.imshow('Disparity Map', disp)
# #     cv2.waitKey(0)
# #     cv2.destroyAllWindows()


def show_disparity_with_blockMatching(img1, img2, show=False):
    stereo = cv2.StereoSGBM_create(minDisparity=-1,
                                   numDisparities=1 * 16,  # max_disp has to be dividable by 16 f. E. HH 192, 256
                                   blockSize=15)

    # Compute the disparity image
    disparity = stereo.compute(img1, img2).astype(np.float32) / 16

    disparityImg = cv2.normalize(src=disparity, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                 dtype=cv2.CV_8UC1)

    # disparityImg = cv2.applyColorMap(disparityImg, cv2.COLORMAP_JET)

    # plot the result
    if show:
        plt.imshow(disparityImg)
        plt.colorbar()
        plt.axis('off')
        plt.show()

    return disparity


def find_corresponding_point(frame_1, frame_2, disparity, L_point_x, L_point_y, Q):
    # Approach 1
    # Get the corresponding point in the right image using the disparity map
    # R_point_x = L_point_x - disparity[L_point_y, L_point_x]
    # R_point_y = L_point_y - disparity[L_point_y, L_point_x]

    # Approach 2
    # Create a 3D point from the disparity map
    points_3D = cv2.reprojectImageTo3D(disparity, Q)

    # Get the x, y, and z coordinates of the 3D point
    x, y, z = points_3D[L_point_y, L_point_x]

    # Project the 3D point onto the right image plane
    u, v, w, _ = Q @ [x, y, z, 1]

    # Normalize the coordinates by the w value
    u /= w
    v /= w

    # The (u, v) coordinates are the corresponding point on the right image
    R_point_x, R_point_y = int(u), int(v)

    # Print the corresponding point
    print("Corresponding point in the right image: (", R_point_x, ", ", R_point_y, ")")

    # Visualization of the pixels in the image pair
    cv2.putText(frame_1, 'x', (L_point_x, L_point_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
    cv2.putText(frame_2, 'x', (R_point_x, L_point_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
    result = cv2.hconcat([frame_1, frame_2])
    result = cv2.resize(result, (1920, 1080))
    cv2.imshow("image", result)
    cv2.waitKey(0)

    return R_point_x, R_point_y


def mark_points(img):
    # Show the image
    fig, ax = plt.subplots()
    ax.imshow(img, cmap="gray")

    # Create a cursor object
    cursor = Cursor(ax, useblit=True, color='red', linewidth=1)

    # Pick a pixel by clicking on the image
    point = plt.ginput(n=1, timeout=0, show_clicks=True, mouse_add=1, mouse_pop=3, mouse_stop=2)
    point = np.array(point)
    # Print the pixel coordinates

    plt.close('all')
    return point[0][0], point[0][1]


def find_corresponding_point_sgm(left_img, right_img, u, v, F, window_size=5, num_disparities=32):
    # Compute disparity map using SGM
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=window_size,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )
    disparity = stereo.compute(left_img, right_img).astype(np.float32) / 16.0

    # Extract corresponding epipolar line in right image
    # F, _ = cv2.findFundamentalMat(np.array([[u, v]], dtype=np.float32), None, cv2.FM_RANSAC)
    epipolar_line = cv2.computeCorrespondEpilines(np.array([[u, v]], dtype=np.float32), 1, F)
    epipolar_line = np.squeeze(epipolar_line)

    # Define search range on epipolar line
    search_range = (int(max(0, -epipolar_line[2] / epipolar_line[0] - num_disparities)),
                    int(min(disparity.shape[1], -epipolar_line[2] / epipolar_line[0] + num_disparities)))

    # Compute matching cost for each pixel in search range
    matching_cost = np.zeros((disparity.shape[1],), dtype=np.float32)
    for i in range(search_range[1], search_range[0]):
        x = i + num_disparities
        if x < 0 or x >= disparity.shape[0]:
            continue
        block_left = left_img[v - window_size // 2:v + window_size // 2 + 1,
                     i - window_size // 2:i + window_size // 2 + 1]
        for j in range(window_size // 2, disparity.shape[0] - window_size // 2):
            block_right = right_img[j - window_size // 2:j + window_size // 2 + 1,
                          i - window_size // 2:i + window_size // 2 + 1]
            matching_cost[x] += np.sum(np.abs(block_left - block_right))

    # Find pixel on epipolar line with minimum matching cost
    best_match = np.argmin(matching_cost[search_range[0] + num_disparities:search_range[1] + num_disparities])
    point_right = np.array([best_match + search_range[0], v], dtype=np.float32)

    # # Extract row and column of point in left image
    # row, col = point_left[0, 0]
    #
    # # Extract epipolar line from line_right
    # a, b, c = line_right.squeeze()
    # x1, y1 = 0, int((-c - a * 0) / b)  # Point on the left edge of image
    # x2, y2 = right_gray.shape[1] - 1, int((-c - a * (right_gray.shape[1] - 1)) / b)  # Point on the right edge of image
    #
    # # Compute SSD cost for each pixel on the epipolar line
    # costs = []
    # for x in range(num_disparities):
    #     if x < col:
    #         # Pixel is to the left of the point in the left image
    #         cost = np.inf
    #     else:
    #         # Extract window around corresponding pixel in right image
    #         window = right_gray[y1:y2 + 1, x - window_size:x + window_size + 1]
    #
    #         # Compute SSD cost for the window
    #         if window.shape == (window_size * 2 + 1, window_size * 2 + 1):
    #             left_window = left_gray[row - window_size:row + window_size + 1,
    #                           col - window_size:col + window_size + 1]
    #             cost = np.sum((left_window - window) ** 2)
    #         else:
    #             cost = np.inf
    #
    #     costs.append(cost)
    #
    # # Find disparity with smallest SSD cost
    # best_disparity = np.argmin(costs)
    #
    # # Compute corresponding point in right image
    # point_right = np.array([col - best_disparity, row]).astype(np.float32).reshape(1, 1, 2)

    return point_right


def detect_corresponding_point(u, v, left_img, right_img, F):
    # # Detect SIFT features in left image
    # sift = cv2.SIFT_create()
    # kp1, des1 = sift.detectAndCompute(left_img, None)
    #
    # # Detect SIFT features in right image
    # kp2, des2 = sift.detectAndCompute(right_img, None)
    #
    # # Match SIFT features
    # FLANN_INDEX_KDTREE = 1
    # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    # search_params = dict(checks=50)
    # flann = cv2.FlannBasedMatcher(index_params, search_params)
    # matches = flann.knnMatch(des1, des2, k=2)
    # # Filter matches using Lowe's ratio test
    # good_matches = []
    # for m, n in matches:
    #     if m.distance < 0.8 * n.distance:
    #         good_matches.append(m)
    #
    # # Compute fundamental matrix from good matches
    # pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    # pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    # F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)

    # Select specific point in left image
    point_left = np.array([u, v]).reshape(1, 1, 2)

    # Compute epipolar line in right image for point in left image
    line_right = cv2.computeCorrespondEpilines(point_left, 1, F)
    point_left = np.reshape(point_left, [1, 2])
    point_right = find_corresponding_point_sgm(left_img, right_img, point_left[0][0], point_left[0][1], F,
                                               window_size=5,
                                               num_disparities=16)

    # Alternate
    point_right_refined1 = np.reshape(point_right, [1, 2])
    lines2 = line_right.reshape(-1, 3)

    img3, img4 = drawlines(right_img, left_img, lines2, point_right_refined1, point_left)
    plt.subplot(121), plt.imshow(img4)
    plt.subplot(122), plt.imshow(img3)
    plt.show()

    return point_right[0], point_right[1]


def detect_corresponding_point_old(u, v, left_img, right_img):
    # Detect SIFT features in right image
    # sift = cv2.xfeatures2d.SIFT_create()
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(left_img, None)

    # Detect SIFT features in right image
    kp2, des2 = sift.detectAndCompute(right_img, None)

    # Match SIFT features
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    pts1 = []
    pts2 = []

    # Filter matches using Lowe's ratio test
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
    # print("F_old", F)
    # Select specific point in left image
    point_left = (u, v)
    point_left = np.int32(point_left)
    point_left = np.array(point_left)

    # Compute epipolar line in right image for point in left image
    line_right_1 = cv2.computeCorrespondEpilines(point_left.reshape(1, 1, 2), 1, F)
    line_right = line_right_1.reshape(3)

    # Compute point on epipolar line in right image that is closest to point in left image
    x = np.arange(0, right_img.shape[1])
    y = -(line_right[2] + line_right[0] * x) / line_right[1]
    closest_point = (x[np.argmin(np.abs(y - point_left[1]))], y[np.argmin(np.abs(y - point_left[1]))])
    closest_point = np.int32(closest_point)
    closest_point = np.array(closest_point)
    closest_point = np.reshape(closest_point, [1, 2])
    point_left = np.reshape(point_left, [1, 2])
    closest_point = np.array((718, 688))
    closest_point = np.reshape(closest_point, [1, 2])

    # Alternat
    lines2 = line_right_1.reshape(-1, 3)
    img3, img4 = drawlines(right_img, left_img, lines2, closest_point, point_left)
    plt.subplot(121), plt.imshow(img4)
    plt.subplot(122), plt.imshow(img3)
    plt.show()

    # # Print the corresponding point
    # print("Corresponding point in the right image: (x", closest_point[0], ", y: ", closest_point[1], ")")

    # fig, axes = plt.subplots(1, 2, figsize=(15, 10))
    # axes[0].imshow(left_img, cmap="gray")
    # axes[1].imshow(right_img, cmap="gray")
    # axes[0].scatter(u, v, color='red', marker='x')
    # axes[1].scatter(closest_point[0], closest_point[1], color='red', marker='x') #s=200, alpha=0.5
    # plt.suptitle("Stereo Matched Images")
    # plt.show()

    return closest_point[0][0], closest_point[0][1]


def drawlines(img1, img2, lines, pts1, pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, cv2.FILLED)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, cv2.FILLED)
    return img1, img2


def pixel2cam_normalize(pt, K):
    u = (pt[0] - K[0][2]) / K[0][0]
    v = (pt[1] - K[1][2]) / K[1][1]
    return np.array([u, v], dtype=np.float32)


def stereo_matcher(img1, img2, camera_parameters, show=False):
    # Detect SIFT features
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Match SIFT features
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    pts1 = []
    pts2 = []

    # Filter matches using Lowe's ratio test
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)

    # print("F_new", F)
    # for i in range(len(pts1)):
    #     error = cv2.sampsonDistance(pts1[i].tolist(), pts2[i].tolist(), F)
    #     print(error)

    # We select only inlier points
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    pt1 = pts1.astype(np.float32).reshape((-1, 1, 2))
    pt2 = pts2.astype(np.float32).reshape((-1, 1, 2))

    dist1 = camera_parameters['C1_distortion_coeff'].flatten()
    dist2 = camera_parameters['C2_distortion_coeff'].flatten()
    E, E_mask = cv2.findEssentialMat(points1=pt1, points2=pt2, cameraMatrix1=camera_parameters['C1_camera_matrix'],
                                     distCoeffs1=dist1,
                                     cameraMatrix2=camera_parameters['C2_camera_matrix'],
                                     distCoeffs2=dist2,
                                     method=cv2.LMEDS, prob=0.99, threshold=0.05)
    RPose, R, t, RP_mask = cv2.recoverPose(E, pt1, pt2, camera_parameters['C1_camera_matrix'],
                                           E_mask)  # distanceThresh=3
    # triangulatedPoints
    left_inliers = pts1[RP_mask.ravel() == 255]
    right_inliers = pts2[RP_mask.ravel() == 255]

    if show:
        print("Stereo match for validating image rectification")
        # Find epilines corresponding to points in right image (second image) and
        # drawing its lines on left image
        lines1 = cv2.computeCorrespondEpilines(right_inliers.reshape(-1, 1, 2), 2, F)  # pts2
        lines1 = lines1.reshape(-1, 3)
        img5, img6 = drawlines(img1, img2, lines1, left_inliers, right_inliers)  # pts1, pts2
        # plt.subplot(121), plt.imshow(img5)
        # plt.subplot(122), plt.imshow(img6)

        # Find epilines corresponding to points in left image (first image) and
        # drawing its lines on right image
        lines2 = cv2.computeCorrespondEpilines(left_inliers.reshape(-1, 1, 2), 1, F)  # pts1
        lines2 = lines2.reshape(-1, 3)
        img3, img4 = drawlines(img2, img1, lines2, right_inliers, left_inliers)  # pts2, pts1

        plt.subplot(121), plt.imshow(img4)
        plt.subplot(122), plt.imshow(img3)
        plt.show()

    left_norm = pixel2cam_normalize(left_inliers[0], camera_parameters['C1_camera_matrix'] )
    right_norm = pixel2cam_normalize(right_inliers[0], camera_parameters['C2_camera_matrix'] )

    # Triangulate the 3D position of the inlier points using cv2.triangulatePoints()
    P1 = camera_parameters['C1_camera_matrix'].dot(np.hstack((np.eye(3), np.zeros((3, 1)))))
    P2 = camera_parameters['C2_camera_matrix'].dot(np.hstack((R, t)))
    points_4d_hom = cv2.triangulatePoints(P1, P2, left_norm.T, right_norm.T)
    points_3d = points_4d_hom / points_4d_hom[3]

    # Extract the 3D coordinates of the inlier points
    inlier_3d_points = points_3d[:3].T
    # print(inlier_3d_points)
    return F, E, R, t, RPose, left_inliers, right_inliers
