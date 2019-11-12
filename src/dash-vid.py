import dash
import dash_core_components as dcc
import dash_html_components as html

from flask import Flask, Response

import io
import cv2
from PIL import Image
import logging
import detector

frmt = "%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s:%(lineno)d - %(message)s"
logging.basicConfig(level=logging.INFO, filename="log/uiface.log", format=frmt)

class TestClass:
    def __init__(self):
        self.det = detector.Detector()
        self.fcam1 = None
        self.fcam2 = None

    def cam1(self):
        return self.det.detect()[0]


def gen(det):
    while True:
        frame = det.cam1()
        buf = io.BytesIO()
        frame.save(buf, format='JPEG')
        frame = buf.getvalue()
        #yield byte_im
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

server = Flask(__name__)
app = dash.Dash(__name__, server=server)

@server.route('/video_feed_0')
def video_feed_0():
    return Response(gen(TestClass()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@server.route('/video_feed_1')
def video_feed_1():
    return Response(gen(VideoCamera(2)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

app.layout = html.Div(
    children = [html.H1("Cam 1 Test"), html.Img(src="/video_feed_0")])

if __name__ == '__main__':
    app.run_server(debug=False)