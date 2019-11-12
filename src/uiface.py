#! /usr/bin/env python
import os
import time
import tkinter
import logging
from logging import handlers

import config
import videofeed
import detector
import sys, signal
import PIL.Image, PIL.ImageTk

CURDIR = os.path.realpath(os.path.dirname(__file__))

class App:
    def __init__(self, window, window_title, conf):
        self.window = window
        self.window.title(window_title)


        self.feed_frame = tkinter.Frame(self.window)
        self.control_frame = tkinter.Frame(self.window)
        self.feed_frame.pack()
        self.control_frame.pack(side = "bottom")

        logging.info("Started UI.")
        self.detector = detector.Detector(conf)

        self.canvas1 = tkinter.Canvas(self.feed_frame, width = 900 * 2 + 40, height = 550)
        #self.canvas1 = tkinter.Canvas(self.feed_frame)
        self.canvas1.pack()

        self.sw_cam_button = tkinter.Button(self.control_frame, pady = 15, text = "Switch", command = self.detector.switch_cams)
        self.sw_cam_button.pack(side = "bottom")
        self.report_label = tkinter.Label(self.control_frame, padx=10, pady=10,
                                          font=("Helvetica", 14),
                                          justify=tkinter.LEFT,
                                          text="cam fps: 0\nDetected Persons: 0\nLatitude: 0\nLongitude: 0\n")
        self.report_label.pack(side="left")

        with open(os.path.join(CURDIR, os.path.pardir, 'version')) as ver_file:
            self.version_label = tkinter.Label(self.control_frame, padx=10,
                                               pady=0, font=("Helvetica", 14),
                                               justify=tkinter.LEFT,
                                               text="ver: {}".format(
                                                   ver_file.read()))
            self.version_label.pack(side="left")

        self.delay = 1
        self.update()


        self.window.mainloop()

    def update(self):
        try:
            (im1, im2) = self.detector.detect()
            im1 = im1.resize((900, 500), PIL.Image.ANTIALIAS)
            im2 = im2.resize((900, 500), PIL.Image.ANTIALIAS)
            self.photo1 = PIL.ImageTk.PhotoImage(image = im1)
            self.photo2 = PIL.ImageTk.PhotoImage(image = im2)

            self.canvas1.create_image(20, 20, image = self.photo1, anchor = tkinter.NW)
            self.canvas1.create_image(900 + 40, 20, image = self.photo2, anchor = tkinter.NW)
            self.report_label.config(text=self._get_report_label_text())
            self.window.after(self.delay, self.update)
        except StopIteration:
            logging.error("Camera disconnected or used by another process... please check the status of your cameras.")
            self.detector.vidsource = videofeed.VideoFeed(get_config())
            time.sleep(2)
            self.update()

    def _get_report_label_text(self):
        gps = 0, 0
        if self.detector.GPS:
            gps = self.detector.GPS.latitude, self.detector.GPS.longitude

        detector_stats = self.detector.stats
        cam_stats = detector_stats.cam_stats
        report_label_fmt = "cam fps: {cam1_fps:05.3f},{cam2_fps:05.3f}\n" \
                           "cam detected persons: {cam1_n_detected}, {cam2_n_detected}\n" \
                           "cam identified: {cam1_n_identified} {cam2_n_identified}\n" \
                           "detector persons: {detector_persons}\n" \
                           "gps: {lat}, {long}"
        return report_label_fmt.format(cam1_fps=cam_stats[0].fps,
                                       cam2_fps=cam_stats[1].fps,
                                       cam1_n_detected=cam_stats[0].n_detected,
                                       cam2_n_detected=cam_stats[1].n_detected,
                                       cam1_n_identified=cam_stats[0].n_identified,
                                       cam2_n_identified=cam_stats[1].n_identified,
                                       detector_persons=detector_stats.person_count,
                                       lat=gps[0], long=gps[1])


def configure_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "face-id:%(asctime)s:%(levelname)s:%(filename)s:%(funcName)s:%(lineno)d:%(message)s",
        datefmt='%Y-%m-%d %H:%M:%S')
    syslog_handler = handlers.SysLogHandler(address ='/dev/log')
    syslog_handler.setFormatter(formatter)
    logger.addHandler(syslog_handler)


def get_config():
    conf = config.load()
    logging.info(conf)
    return conf


def handler(signum = None, frame = None):
    if os.path.isfile("shutdown.txt"):
        os.remove("shutdown.txt")
    sys.exit(0)

for sig in [signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGQUIT]:
    signal.signal(sig, handler)

def main():
    configure_logging()
    app = None
    try:
        app = App(tkinter.Tk(), "Face Detection and Verification Software",
                  conf=get_config())
    finally:
        fps = app.detector.stats.fps
        fps.stop()
        logging.info(f'elapsed: {fps.elapsed()} fps: {fps.fps()}')


if __name__ == '__main__':
    main()
