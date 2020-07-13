from sanic import Sanic
from sanic.response import HTTPResponse
import ujson
import cv2
import pandas as pd
from tqdm import tqdm
import base64

app = Sanic(__name__)


@app.route('/get_video', methods=['POST'])
async def get_video(request):
    frames, boxes = [], []
    videoname = request.json['videoname']

    cap = cv2.VideoCapture(videoname + '.mp4')
    frame_shape = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3)
    anno = pd.read_csv(videoname + '.csv')

    for i_frame, frame_data in tqdm(anno.groupby('i_frame'), desc=f'Read {videoname}'):
        # read specific frame
        if i_frame != cap.get(cv2.CAP_PROP_POS_FRAMES):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i_frame)
        _, frame = cap.read()

        # bytes to utf8 string
        frames.append(base64.b64encode(frame.tobytes()).decode())

        # read all boxes in the frame
        frame_boxes = []
        for idx in frame_data.index:
            frame_boxes.append(frame_data.loc[idx].bbox)
        boxes.append(frame_boxes)

    return HTTPResponse(ujson.dumps({'frames': frames,
                                     'boxes': boxes,
                                     'frame_shape': frame_shape}), status=201)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=2299, debug=False)
