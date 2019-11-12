import time
import logging
import cv2

class VideoFeed:
    def __init__(self, config, width=1280, height=720, ncams=2):
        self.width = width
        self.height = height
        self.ncams = ncams  
        self.cam_ids = [] 
        self.cap1 = None
        self.cap2 = None
        self._init_video_cap_from_file(config)
        self._check_setup()
        self._init_cap()    
        self._init_generator()           

    def _init_video_cap_from_file(self, config):
        c1 = config.get('cam1_video_capture_from_file')
        c2 = config.get('cam2_video_capture_from_file')
        assert not (bool(c1) ^ bool(c2)), "Invalid video capture configuration"

        self.cam1_video_cap_from_file = c1
        self.cam2_video_cap_from_file = c2

    def _check_setup(self):
        self.cam_ids = self._get_camera_ids(20)
        while len(self.cam_ids) < self.ncams:
            #print("Not enough cameras connected. Detected: %d. Expected %d." % (len(self.cam_ids), self.ncams))
            logging.warning("Not enough cameras connected. Detected: %d. Expected %d." % (len(self.cam_ids), self.ncams))
            self.cam_ids = self._get_camera_ids(20)
            time.sleep(2)
        logging.info("Video feed test passed... Found %d. Expected %d" % ((len(self.cam_ids), self.ncams)))

    def _init_cap(self):
        cap_src = self.cam1_video_cap_from_file or self.cam_ids[0]
        self.cap1 = cv2.VideoCapture(cap_src)
        self.cap1.set(3, self.width)
        self.cap1.set(4, self.height)
        if self.ncams == 2:
            cap_src = self.cam2_video_cap_from_file or self.cam_ids[1]
            self.cap2 = cv2.VideoCapture(cap_src)
            self.cap2.set(3, self.width)
            self.cap2.set(4, self.height)
        logging.info("VideoCapture initialized.")

    def _release(self):
        if self.cap1 is not None and self.cap1.isOpened():
            self.cap1.release()
        if self.cap2 is not None and self.cap2.isOpened():
            self.cap2.release()

    def __del__(self):
        self._release()

    def _check_camera_ids(self, i):
        cap = cv2.VideoCapture(self._get_vid_cap_src(i))
        ret, _ = cap.read()
        #print(i, ret)
        cap.release()

        return(ret, i)

    def _get_vid_cap_src(self, i):
        # real camera
        if not self._is_cameras_video_capture_from_file():
            return i

        # capture from file
        assert i in {0, 1}, f"Error: invalid cap index: {i}"
        video_srcs = [self.cam1_video_cap_from_file,
                      self.cam2_video_cap_from_file]
        return video_srcs[i]

    def _get_camera_ids(self, max_id):
        if self._is_cameras_video_capture_from_file():
            return 0, 1
        ids = [ i for (ret,i) in [self._check_camera_ids(id) for id in range(0, max_id)] if ret ]
        return(ids)

    def _is_cameras_video_capture_from_file(self):
        return bool(self.cam1_video_cap_from_file)


    def _gen_frames(self):        
        logging.info("Initializing frame generator...")
        id1 = self.cam_ids[0]
        id2 = self.cam_ids[1]
        try:            
            if self.ncams == 1:
                while self.cap1.isOpened():
                    has_frames1, image1 = self.cap1.read()
                    if has_frames1:
                        yield({id1: image1, id2: None})
                    else:
                        logging.warning("No more frames.")
                        self._release()
            
            if self.ncams == 2:
                while self.cap1.isOpened() and self.cap2.isOpened():        
                    has_frames1, image1 = self.cap1.read()
                    has_frames2, image2 = self.cap2.read()
                    if has_frames1 and has_frames2:
                        yield({id1: image1, id2: image2})
                    else:
                        logging.warning("No more frames.")
                        self._release()
        except:
            logging.error("Exception occurred", exc_info=True)
            self._release()

    def _init_generator(self):
        self.generator = self._gen_frames()
    
    def get_frame(self):
        return(next(self.generator))

    @property
    def fps(self):
        ret = [None, None]
        if self.cap1:
            ret[0] = self.cap1.get(cv2.CAP_PROP_FPS)
        if self.cap2:
            ret[1] = self.cap2.get(cv2.CAP_PROP_FPS)

        return ret