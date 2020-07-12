import numpy as np
import cv2
import requests
import base64
from timeit import default_timer
from tqdm import tqdm

videoname = 'SIZ_2_Work_Area_View_13_2020-06-04_05%3A15%3A51.209912'
response = requests.post('http://127.0.0.1:2299/get_video', json={'videoname': videoname})
response = response.json()

visualization, decode_time = True, 0
for frame, frame_boxes in tqdm(list(zip(response['frames'], response['boxes']))):
    t = default_timer()
    # utf-8 string to bytes
    frame = base64.decodebytes(frame.encode())
    # bytes to encoded jpg img
    frame = np.frombuffer(frame, np.uint8)
    # decode jpg image
    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
    decode_time += default_timer() - t

    # draw boxes and scores
    if visualization:
        for box in frame_boxes:
            score = box.split(',')[-1]
            x1, y1, x2, y2 = [int(s) for s in box.split(',')[:-1]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, score, (x1, y1 - 5), 1, 1, (255, 0, 0), 2)

        cv2.imshow(videoname, cv2.resize(frame, (0, 0), fx=1/3, fy=1/3))
        cv2.waitKey(1)

decode_time = decode_time * 1000 / len(response["frames"])
fps = 1000 / decode_time
print(f'average decode time: {decode_time} ms\nframes per second: {fps}')

