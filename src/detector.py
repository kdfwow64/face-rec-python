import time
import atexit
from collections import Counter
from dataclasses import dataclass
from typing import List

from facenet_pytorch import InceptionResnetV1, extract_face, prewhiten
import logging
import datetime
# import tensorflow as tf
import numpy as np
import imutils
from imutils.video import FPS
import cv2
from PIL import Image, ImageDraw, ImageFont
# from IPython import display

from fastdetector import TensoflowFaceDector, PATH_TO_CKPT

import videofeed
import gps
import db
import config
import sys
import os
import glob
import signal

output_stream = sys.stdout
clear = lambda: os.system('clear')
start_time = datetime.datetime.now()

shutdown_reason = "orderly"
if not os.path.exists('shutdown.txt'):
    os.mknod('shutdown.txt')
else:
    shutdown_reason = "anomalous"

list_of_files = glob.glob('./data/*.csv') # * means all if need specific format then *.csv
latest_file = max(list_of_files, key=os.path.getctime)
last_close_time = datetime.datetime.fromtimestamp(os.path.getmtime(latest_file)).strftime("%Y-%m-%d %H:%M:%S")
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print('Running on device: {}'.format(device))

# mtcnn = MTCNN(keep_all=True, device=device, min_face_size=100, thresholds=[0.8, 0.8, 0.8], image_size=160)
fast_det = TensoflowFaceDector(PATH_TO_CKPT)

resnet = InceptionResnetV1(pretrained='vggface2').eval()

def exit_handler():
    if os.path.isfile("shutdown.txt"):
        os.remove("shutdown.txt")

atexit.register(exit_handler)



def bb_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def abs_box(box, width, height):
    return ([int(box[1] * width), int(box[0] * height), int(box[3] * width), int(box[2] * height)])


def scale_bbox(bbox, factor=2):
    ini_w = bbox["right"] - bbox["left"]
    ini_h = bbox["bottom"] - bbox["top"]
    new_w = ini_w * factor
    new_h = ini_h * factor
    pos_x = bbox["left"] + ini_w / 2
    pos_y = bbox["top"] + ini_h / 2

    new_left = max(pos_x - new_w / 2, 0)
    new_top = max(pos_y - new_h / 2, 0)

    new_bbox = {
        "left": new_left,
        "top": new_top,
        "right": new_left + new_w,  # no max width check
        "bottom": new_top + new_h,  # no max height check
        "label": bbox["label"]
    }

    return new_bbox


def avg_dist(emb, embs):
    distances = [(emb - x).norm().item() for x in embs]
    min_dist = np.min(distances)
    # logging.debug(min_dist)
    return min_dist


def propose_pid_emb(emb, id_dict, thresh=1.1):
    prop_pid = DETECTING
    min_avg_dist = thresh
    dict_items = (x for x in id_dict.items() if not x[1]["detecting"])
    for (pid, vals) in dict_items:
        dist = avg_dist(emb, vals["embeddings"])
        if dist < min_avg_dist:
            prop_pid = pid
            min_avg_dist = dist
    return prop_pid


def _propose_person_emb(emb, id_list, thresh=1.1):
    prop_person = None
    min_avg_dist = thresh
    for person in id_list:
        dist = avg_dist(emb, person["embeddings"])
        if dist < min_avg_dist:
            prop_person = person
            min_avg_dist = dist
    return prop_person


def propose_pid_iou(box, current_frame_id, id_dict):
    prop_pid = DETECTING
    max_iou = 0.4
    for (pid, vals) in id_dict.items():
        if current_frame_id - vals["identity_frame"] < 8:  # FIXME(haim): normalize for camera fps!
            iou = bb_iou(box, vals["last_box"])
            if iou > max_iou:
                vals["identity_frame"] = current_frame_id  # try keeping track of the person in the followup frames
                prop_pid = pid
                max_iou = iou
    # logging.debug("MAX IOU: %s Frame Diff: %s", max_iou, current_frame_id - vals["identity_frame"])
    return prop_pid


def _get_emb(image, box):
    """Return facial embeddings from given image inside the box."""
    cropped_face = extract_face(image, box)
    cropped_face = prewhiten(cropped_face)

    return resnet(cropped_face.unsqueeze(0))[0].detach()


def get_emb(emb_state, image, box):
    if emb_state is not None:
        return (emb_state)
    else:
        cropped_face = extract_face(image, box)
        cropped_face = prewhiten(cropped_face)
        emb = resnet(cropped_face.unsqueeze(0))[0].detach()  # .numpy().reshape(1, 512)
        return (emb)


def box_area(box):
    return abs(box[0] - box[2]) * abs(box[1] - box[3])


def offset_box(x, y, box):
    box[0] += x
    box[2] += x
    box[1] += y
    box[3] += y


@dataclass
class CameraStats:
    n_detected: int = 0
    n_identified: int = 0
    fps: float = 0.


@dataclass
class DetectorStats:
    cam_stats: List[CameraStats]
    fps: FPS = FPS()
    person_count = 0


DETECTING = "Detecting..."


class Detector:
    fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMonoBold.ttf', 27)

    def __init__(self, config):
        self.config = config
        self.use_gps = config["use_gps"]
        self.detector_sensibility = config["detector_sensibility"]
        self.min_decision_frames = config["min_decision_frames"]
        self.DB = db.DB("tst", config["save_dir"])
        self.DB.connect()
        self.session = self.DB.get_session()
        self.person_id = 0
        self.frame_id = 0
        self.detecting_id_cam1 = 0
        self.detecting_id_cam2 = 0
        self.identity_dict = {}
        self.identity_list = []
        self.verify_dict = {}
        self.vidsource = videofeed.VideoFeed(config)
        self.cam1 = 0
        self.cam2 = 1
        self.cam1_rotate_deg = config.get('cam1_rotate_deg', 0)
        self.cam2_rotate_deg = config.get('cam2_rotate_deg', 0)
        self.stats = DetectorStats(
            cam_stats=[CameraStats(fps=self.vidsource.fps[0]),
                       CameraStats(fps=self.vidsource.fps[1])])

        if config.get('switch_cameras_on_start', False):
            self._switch_cams()

        self.GPS = None
        if self.use_gps:
            self.GPS = gps.GPS()
            self.GPS.start()
            # logging.info("Detector created successfuly...")

    def switch_cams(self):
        self._switch_cams()
        self.config['switch_cameras_on_start'] = not self.config.get(
            'switch_cameras_on_start', False)
        config.store(self.config)

    def _switch_cams(self):
        self.cam1, self.cam2 = self.cam2, self.cam1

    def clean_old_attempts(self):
        # clean detection attempts

        kvs_id = [(k, v) for (k, v) in self.identity_dict.items()]
        kvs_ve = [(k, v) for (k, v) in self.verify_dict.items()]

        # FIXME(haim): this should be translated to sec func(fps)
        for (k, v) in kvs_id:
            if v["detecting"] and self.frame_id - v["identity_frame"] > 20:
                del self.identity_dict[k]

        for (k, v) in kvs_ve:
            if v["detecting"] and self.frame_id - v["identity_frame"] > 20:
                del self.verify_dict[k]

    def make_entry(self, box, emb, frame_id, detecting):
        vals = {
            "session": self.session,
            "last_box": box,
            "embeddings": [emb],
            "identity_frame": frame_id,
            "detecting": detecting,
            "get_in_latitude": 0.0,
            "get_in_longitude": 0.0,
            "get_off_latitude": 0.0,
            "get_off_longitude": 0.0,
            "first_seen": "",
            "last_seen": ""
        }
        return (vals)

    def detect_faces_incoming(self, frame):
        # no bus driver
        ROW_START = 250
        ROW_END = 500

        # just the "onboarding" aread
        COL_START = 600
        # COL_START = 500
        COL_END = 750

        MIN_BOX_AREA = 228
        self.frame_id += 1
        # boxes, prob = mtcnn.detect(frame)
        frame = cv2.flip(frame, 1)
        # set search area, to avoid driver, and lateral faces' areas in image.
        # the dimension are rows , columns
        search_area = frame[ROW_START:ROW_END, COL_START:COL_END]
        # cv2.imshow('debug', search_area)
        # cv2.waitKey(0)
        (boxes, scores, _, _) = fast_det.run(search_area)
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        boxes = (b for (s, b) in zip(scores, boxes) if s > self.detector_sensibility)
        # normalize the box size to search area
        boxes = (abs_box(b, search_area.shape[1], search_area.shape[0]) for b in boxes)
        # filter distant faces
        boxes = (b for b in boxes if box_area(b) > MIN_BOX_AREA)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        frame_draw = image.copy()
        draw = ImageDraw.Draw(frame_draw)

        i = 0
        for i, box in enumerate(boxes):
            # offset the box to original frame coordinates
            offset_box(COL_START, ROW_START, box)
            # pdb.set_trace()
            emb = _get_emb(image, box)
            self.detecting_id_cam1 += 1
            current_person = DETECTING
            # DB is empty - initiate first entry
            if not self.identity_list:
                new_person = self._create_new_person(emb, box)
                self.identity_list.append(new_person)
                current_person = new_person["name"]
            else:
                winner = _propose_person_emb(emb, self.identity_list, thresh=0.8)
                if winner:
                    self.stats.cam_stats[0].n_identified += 1
                    winner["embeddings"].append(emb)
                    if winner["detecting"]:
                        winner["name"] = f"Person_{self.person_id}"
                        winner["detecting"] = False
                        self.person_id += 1
                        self.stats.person_count += 1
                    current_person = winner["name"]
                    self.stats.cam_stats[0].n_identified += 1
                    # logging.debug('winner: %s', winner["name"])
                else:
                    new_person = self._create_new_person(emb, box)
                    self.identity_list.append(new_person)
                    current_person = new_person["name"]

            draw.rectangle(box, outline=(255, 0, 0), width=6)
            draw.text((box[0], box[3]), current_person, font=self.fnt, fill=(255, 255, 0), )
        # draw the search area
        draw.rectangle(((COL_START, ROW_END), (COL_END, ROW_START)),
                       outline=(0, 0, 255), width=6)
        del draw
        return frame_draw, i

    def _create_new_person(self, emb_state, box):
        new_person = self.make_entry(box, emb_state, self.frame_id, True)
        new_person["first_seen"] = datetime.datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S")
        new_person["name"] = f"Detecting_{self.detecting_id_cam1}"
        if self.GPS:
            new_person["get_in_latitude"] = self.GPS.latitude
            new_person["get_in_longitude"] = self.GPS.longitude

        return new_person

    def detect_faces_departuring(self, frame):
        # no bus driver
        ROW_START = 50
        ROW_END = 400

        # just the "departure" aread
        COL_START = 200
        COL_END = 950

        MIN_BOX_AREA = 228
        self.frame_id += 1
        frame = cv2.flip(frame, 1)
        # set search area, to avoid driver, and lateral faces' areas in image.
        # the dimension are rows , columns
        search_area = frame[ROW_START:ROW_END, COL_START:COL_END]
        (boxes, scores, _, _) = fast_det.run(search_area)
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        boxes = (b for (s, b) in zip(scores, boxes) if s > self.detector_sensibility)
        # normalize the box size to search area
        boxes = (abs_box(b, search_area.shape[1], search_area.shape[0]) for b in boxes)
        # filter distant faces
        boxes = (b for b in boxes if box_area(b) > MIN_BOX_AREA)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        frame_draw = image.copy()
        draw = ImageDraw.Draw(frame_draw)

        i = 0
        for i, box in enumerate(boxes):
            if not self.identity_list:
                break
            # offset the box to original frame coordinates
            offset_box(COL_START, ROW_START, box)
            # pdb.set_trace()
            emb = _get_emb(image, box)
            self.detecting_id_cam2 += 1
            current_person = DETECTING
            # DB is empty - initiate first entry
            winner = _propose_person_emb(emb, self.identity_list, thresh=0.8)
            if winner:
                self.stats.cam_stats[1].n_identified += 1
                winner["embeddings"].append(emb)
                # if winner["detecting"]:
                #     winner["name"] = f"Person_{self.person_id}"
                #     winner["detecting"] = False
                #     self.person_id += 1
                if self.GPS:
                    winner["get_off_latitude"] = self.GPS.latitude
                    winner["get_off_longitude"] = self.GPS.longitude
                winner['last_seen'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                current_person = winner["name"]
                # logging.debug('winner: %s', winner["name"])
            else:
                # logging.warning('no winner')
                pass

            draw.rectangle(box, outline=(255, 0, 0), width=6)
            draw.text((box[0], box[3]), current_person, font=self.fnt, fill=(255, 255, 0), )
        # draw the search area
        draw.rectangle(((COL_START, ROW_END), (COL_END, ROW_START)),
                       outline=(0, 0, 255), width=6)
        del draw
        return frame_draw, i

    def verify_faces(self, frame):
        emb_state = None
        # boxes, prob = mtcnn.detect(frame)
        frame = cv2.flip(frame, 1)
        (boxes, scores, _, _) = fast_det.run(frame)
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        boxes = [b for (s, b) in zip(scores, boxes) if s > self.detector_sensibility]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)

        frame_draw = image.copy()
        draw = ImageDraw.Draw(frame_draw)

        if boxes is not None:
            for box in boxes:

                box = abs_box(box, frame.shape[1], frame.shape[0])

                current_person = DETECTING
                # DB is empty - initiate first entry
                if len(self.verify_dict) == 0:
                    continue
                else:
                    current_person = propose_pid_iou(box, self.frame_id, self.verify_dict)
                    # logging.debug("Cam 2: " + current_person)

                    if current_person == DETECTING:
                        current_person = f"Detecting_{self.detecting_id_cam2}"
                        self.detecting_id_cam2 += 1
                        emb_state = get_emb(emb_state, image, box)
                        self.verify_dict[current_person] = self.make_entry(box, emb_state, self.frame_id, True)
                    elif self.verify_dict[current_person]["detecting"] and len(
                            self.verify_dict[current_person]["embeddings"]) > self.min_decision_frames:
                        emb_state = get_emb(emb_state, image, box)
                        self.verify_dict[current_person]["embeddings"].append(emb_state)
                        self.verify_dict[current_person]["last_box"] = box
                        votes = [propose_pid_emb(e, self.verify_dict) for e in
                                 self.verify_dict[current_person]["embeddings"]]
                        (winner, nvotes) = Counter(votes).most_common(1)[0]
                        if winner != DETECTING:
                            self.verify_dict[winner]["embeddings"] = self.verify_dict[winner]["embeddings"] + \
                                                                     self.verify_dict[current_person]["embeddings"]
                            self.verify_dict[winner]["last_box"] = box
                            self.verify_dict[winner]["last_seen"] = datetime.datetime.now().strftime(
                                "%Y-%m-%d %H:%M:%S")
                            self.verify_dict[winner]["name"] = winner
                            if self.GPS is not None:
                                self.verify_dict[winner]["get_off_latitude"] = self.GPS.latitude
                                self.verify_dict[winner]["get_off_longitude"] = self.GPS.longitude
                            del self.verify_dict[current_person]
                            current_person = winner
                            self.stats.cam_stats[1].n_identified += 1
                            self.DB.insert_person(self.verify_dict[winner])

                        # logging.debug((winner, nvotes))
                    elif self.verify_dict[current_person]["detecting"]:
                        emb_state = get_emb(emb_state, image, box)
                        self.verify_dict[current_person]["embeddings"].append(emb_state)
                        self.verify_dict[current_person]["last_box"] = box
                    elif len(self.verify_dict[current_person]["embeddings"]) < 40:
                        emb_state = get_emb(emb_state, image, box)
                        self.verify_dict[current_person]["embeddings"].append(emb_state)

                    self.verify_dict[current_person]["last_box"] = box
                    self.verify_dict[current_person]["identity_frame"] = self.frame_id

                emb_state = None
                draw.rectangle(box, outline=(255, 0, 0), width=6)
                draw.text((box[0], box[3]), current_person, font=self.fnt, fill=(255, 255, 0), )

        return frame_draw, len(boxes)

    def print_information(self):
        gps = 0, 0
        if self.GPS:
            gps = self.GPS.latitude, self.GPS.longitude

        detector_stats = self.stats
        cam_stats = detector_stats.cam_stats

        clear()

        output_stream.write('----------------------------------------------------------------\n')
        output_stream.write('App Start DateTime: {}\n'.format(start_time.strftime("%Y-%m-%d %H:%M:%S")))
        output_stream.write('----------------------------------------------------------------\n')
        output_stream.write('Last App Shutdown DateTime: {}\n'.format(last_close_time))
        output_stream.write('----------------------------------------------------------------\n')
        output_stream.write('Last App Shutdown Type: {}\n'.format(shutdown_reason))
        output_stream.write('----------------------------------------------------------------\n')
        output_stream.write('Now Time: {}\n'.format(datetime.datetime.now().strftime("%H:%M:%S")))
        output_stream.write('----------------------------------------------------------------\n')
        output_stream.write('LAT: {}   LON: {}\n'.format(gps[0], gps[1]))
        output_stream.write('----------------------------------------------------------------\n')
        output_stream.write('FRONT: Detected: {}  |  Identified : {}\n'.format(cam_stats[0].n_detected, cam_stats[0].n_identified))
        output_stream.write('----------------------------------------------------------------\n')
        output_stream.write('BACK: Detected: {}  |  Identified : {}\n'.format(cam_stats[1].n_detected, cam_stats[1].n_identified))
        output_stream.write('----------------------------------------------------------------\n')
        output_stream.write('Station Number and name\n')
        output_stream.write('----------------------------------------------------------------\n')

        # output_stream.flush()



        # output_stream.write('\n')


        # logging.info('----------------------------------------------------------------')
        # logging.info('App Start DateTime: %s', start_time.strftime("%Y-%m-%d %H:%M:%S"))
        # logging.info('----------------------------------------------------------------')
        # logging.info('Last App Shutdown DateTime: %s', start_time.strftime("%Y-%m-%d %H:%M:%S"))
        # logging.info('----------------------------------------------------------------')
        # logging.info('Last App Shutdown Type: %s', "orderly")
        # logging.info('----------------------------------------------------------------')
        # logging.info('Now Time: %s', datetime.datetime.now().strftime("%H:%M:%S"))
        # logging.info('----------------------------------------------------------------')
        # logging.info('LAT: %s   LON: %s', gps[0], gps[1])
        # logging.info('----------------------------------------------------------------')
        # logging.info('FRONT: Detected: %s  |  Identified : %s', cam_stats[0].n_detected, cam_stats[0].n_identified)
        # logging.info('----------------------------------------------------------------')
        # logging.info('BACK: Detected: %s  |  Identified : %s', cam_stats[1].n_detected, cam_stats[1].n_identified)
        # logging.info('----------------------------------------------------------------')
        # logging.info('Station Number and name')
        # logging.info('----------------------------------------------------------------')
        return

    def detect(self):
        frame_pair = self.vidsource.get_frame()
        if not self.stats.fps._start:
            self.stats.fps.start()
        self.stats.fps.update()
        cam1_frame = frame_pair[self.vidsource.cam_ids[self.cam1]]
        cam2_frame = frame_pair[self.vidsource.cam_ids[self.cam2]]

        if self.cam1_rotate_deg:
            cam1_frame = imutils.rotate_bound(cam1_frame, self.cam1_rotate_deg)

        if self.cam2_rotate_deg:
            cam2_frame = imutils.rotate_bound(cam2_frame, self.cam2_rotate_deg)

        start = time.time()
        cam1_res_frame, cam1_n_detected = self.detect_faces_incoming(cam1_frame)
        self.print_information()
        cam2_res, cam2_n_detected = self.detect_faces_departuring(cam2_frame)

        cam_stats = self.stats.cam_stats
        cam_stats[0].n_detected += cam1_n_detected
        cam_stats[1].n_detected += cam2_n_detected

        # FIXME(haim): this should be translated to sec func(fps)
        if self.frame_id % 100 == 0:
            self.clean_old_attempts()
            self.DB.write_csv(self.verify_dict)

        return (cam1_res_frame, cam2_res)

    def __del__(self):
        self.DB.close()