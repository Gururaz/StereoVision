class Data:
    # dir_name = "Images"
    # isdir = os.path.isdir(dir_name)
    # if isdir:
    #     pass
    # else:
    #     os.mkdir(dir_name)
    #     sub_folder_names = ['StereoLeft', 'StereoRight', 'RectifiedLeft', 'RectifiedRight', 'TemplateImage']
    #     for sub_folder_name in sub_folder_names:
    #         os.makedirs(os.path.join(dir_name, sub_folder_name))

    # Images save paths

    # Calibration path for Checker Board
    # old camera images
    # left_image_path = 'F:\TUHH\MasterThesis\Code\hyfas_calibration\Archive\Images\CalibrationImages\StereoLeft'
    # right_image_path = 'F:\TUHH\MasterThesis\Code\hyfas_calibration\Archive\Images\CalibrationImages\StereoRight'

    # New camera images (Down - sampled)
    # left_image_path = 'F:\TUHH\MasterThesis\Code\hyfas_calibration\Images\CBwithPylonDownsampled\Left'
    # right_image_path = 'F:\TUHH\MasterThesis\Code\hyfas_calibration\Images\CBwithPylonDownsampled\Right'

    left_image_path = 'F:\TUHH\MasterThesis\Code\hyfas_calibration\Images\CheckerBoard\StereoLeft'
    right_image_path = 'F:\TUHH\MasterThesis\Code\hyfas_calibration\Images\CheckerBoard\StereoRight'

    # left_Undistored_image_path = 'Images/RectifiedLeft'
    # right_Undistored_image_path = 'Images/RectifiedRight'

    # Calibration path for Radon
    R_left_image_path = 'F:\TUHH\MasterThesis\Code\hyfas_calibration\Images\Radon\StereoLeft'
    R_right_image_path = 'F:\TUHH\MasterThesis\Code\hyfas_calibration\Images\Radon\StereoRight'

    # Calibration path for aruco boards

    CH_left_image_path = 'F:\TUHH\MasterThesis\Code\hyfas_calibration\Images\CharucoBoard\StereoLeft'
    CH_right_image_path = 'F:\TUHH\MasterThesis\Code\hyfas_calibration\Images\CharucoBoard\StereoRight'

    # Calibration path for Circular Grid

    CG_left_image_path = 'F:\TUHH\MasterThesis\Code\hyfas_calibration\Images\CircularGrid\StereoLeft'
    CG_right_image_path = 'F:\TUHH\MasterThesis\Code\hyfas_calibration\Images\CircularGrid\StereoRight'

    # Results save path
    results_path = 'F:\TUHH\MasterThesis\Code\hyfas_calibration\Results'

    # Path for test object, Aruco pattern
    pattern_path = 'F:\TUHH\MasterThesis\Code\hyfas_calibration\Images\TemplateImage'

    # Camera definition
    countOfImagesToGrab = 60
    max_Cameras_To_Use = 2
    connected_camera = {
        'cam_1': None,
        'cam_2': None
    }
    image_size = ()
    obj_points = []
    # cam_parameters = {
    #     'gain': float,
    #     'gamma': float,
    #     'exposure_time': float,
    #     'digital_shift': float
    # }

    cam_1_parameters = {
        'gain': 0,
        'gamma': 1,
        'exposure_time': 24783,
        'digital_shift': 0,
        'AcquisitionFrameRate': 100
    }

    cam_2_parameters = {
        'gain': 0,
        'gamma': 1,
        'exposure_time': 25315.0,
        'digital_shift': 0,
        'AcquisitionFrameRate': 100
    }
