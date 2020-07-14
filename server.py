from sanic import Sanic
from sanic.response import HTTPResponse
import ujson
import cv2
import pandas as pd
from tqdm import tqdm
import numpy as np
import uuid

app = Sanic(__name__)
TARGET_WIDTH = 1080


@app.route('/get_video', methods=['POST'])
async def get_video(request):
    boxes = []
    videoname = request.json['videoname']

    cap = cv2.VideoCapture(videoname + '.mp4')
    anno = pd.read_csv(videoname + '.csv')

    scale = TARGET_WIDTH / cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    target_height = int(np.round((cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale)))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    result_video = str(uuid.uuid4()) + '.mp4'

    vw = cv2.VideoWriter(result_video, fourcc, fps, (TARGET_WIDTH, target_height))

    for i_frame, frame_data in tqdm(anno.groupby('i_frame'), desc=f'Read {videoname}'):
        # read specific frame
        if i_frame != cap.get(cv2.CAP_PROP_POS_FRAMES):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i_frame)
        _, frame = cap.read()
        frame = cv2.resize(frame, (TARGET_WIDTH, target_height), interpolation=cv2.INTER_NEAREST)
        vw.write(frame)

        # read all boxes in the frame
        frame_boxes = []
        for idx in frame_data.index:
            frame_boxes.append(frame_data.loc[idx].bbox)
        boxes.append(frame_boxes)

    vw.release()
    return HTTPResponse(ujson.dumps({'boxes': boxes,
                                     'scale': scale,
                                     'result_video': result_video}), status=201)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=2299, debug=False)
