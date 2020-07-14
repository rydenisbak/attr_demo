import cv2
import requests
from timeit import default_timer
from tqdm import tqdm

videoname = 'SIZ_2_Work_Area_View_13_2020-06-04_05%3A15%3A51.209912'
response = requests.post('http://127.0.0.1:2299/get_video', json={'videoname': videoname})
response = response.json()

cap = cv2.VideoCapture(response['result_video'])
assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == len(response['boxes'])
visualization, decode_time = True, 0
for frame_boxes in tqdm(response['boxes']):
    t = default_timer()
    status, frame = cap.read()
    if not status:
        raise IOError(f'Video file {response["result_video"]} is broken')
    decode_time += default_timer() - t

    # draw boxes and scores
    if visualization:
        for box in frame_boxes:
            score = box.split(',')[-1]
            x1, y1, x2, y2 = [int(float(s) * response['scale']) for s in box.split(',')[:-1]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, score, (x1, y1 - 5), 1, 1, (255, 0, 0), 2)

        cv2.imshow(videoname, frame)
        cv2.waitKey(1)

decode_time = decode_time * 1000 / len(response["boxes"])
fps = 1000 / decode_time
print(f'average decode time: {decode_time} ms\nframes per second: {fps}')

