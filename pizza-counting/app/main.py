from onnxruntime import InferenceSession
import os
import cv2
import supervision as sv
import numpy as np
from tqdm import tqdm

model = InferenceSession("/app/best1.onnx",providers=["CPUExecutionProvider"])
input_name = model.get_inputs()[0].name

SOURCE_VIDEO_PATH = os.getenv("INPUT_VIDEO", "/videos/1462_CH03_20250607192844_202844.mp4")
TARGET_VIDEO_PATH = os.getenv("OUTPUT_VIDEO", "/videos/output107.avi")

W, H = 1280, 720
video_info = sv.VideoInfo(
    width=W,
    height=H,
    fps=24,
    total_frames=sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH).total_frames
)

START_FRAME = 32000
END_FRAME = 35000


# Initialize annotators and tracker
box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=0.5, text_color=sv.Color.BLACK)
trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=50)
line_zone_annotator = sv.LineZoneAnnotator(thickness=1, text_thickness=1, text_scale=0.5)

byte_tracker = sv.ByteTrack(
    track_activation_threshold=0.25,
    lost_track_buffer=90,
    minimum_matching_threshold=0.8,
    frame_rate=24,
    minimum_consecutive_frames=3
)
byte_tracker.reset()


CLASS_NAMES_DICT = {0: "piece", 1:"whole"} 

# Define line zones
LINE_START_ = sv.Point(int(100 * W / 640), int(140 * H / 640))
LINE_END_ = sv.Point(int(350 * W / 640), int(115 * H / 640))

line_zone_ = sv.LineZone(start=LINE_START_, end=LINE_END_)

triggered_ids_line_zone_ = set()


generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH, start=START_FRAME, end=END_FRAME, stride=1)

# Process and save video
with sv.VideoSink(target_path=TARGET_VIDEO_PATH, video_info=video_info) as sink:
    for frame in tqdm(generator, total=END_FRAME - START_FRAME + 1):
        resized_frame = cv2.resize(frame, (640, 640))
        resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        resized_frame = resized_frame.transpose(2, 0, 1)
        resized_frame = resized_frame.reshape(1, 3, 640, 640).astype(np.float32)/255.0
        
        prediction = model.run(
            None,
           {"images": resized_frame}
        )[0]
        prediction = prediction[0]
        prediction = prediction[prediction[...,4]>=0.25]
        # print(prediction[prediction[...,4]!=0])
        # print(prediction)
        
        xyxy = prediction[...,0:4]
        xyxy[..., 0] *= W / 640
        xyxy[..., 1] *= H / 640
        xyxy[..., 2] *= W / 640
        xyxy[..., 3] *= H / 640
        
        detections = sv.Detections(
            xyxy=xyxy,
            confidence=prediction[...,4],
            class_id=prediction[...,5].astype(int)
        )

        tracked_detections = byte_tracker.update_with_detections(detections)

        labels = [
            f'{CLASS_NAMES_DICT.get(class_id, "unknown")} ID:{track_id}'
            for _, _, _, class_id, track_id, _ in tracked_detections
        ]

        new_triggers_zone_ = line_zone_.trigger(tracked_detections)
        new_triggers_zone_ = np.array(new_triggers_zone_).flatten()

        if len(new_triggers_zone_) == len(tracked_detections.tracker_id):
            for triggered, track_id in zip(new_triggers_zone_, tracked_detections.tracker_id):
                if triggered.item() and track_id not in triggered_ids_line_zone_:
                    triggered_ids_line_zone_.add(track_id)
                    line_zone_.in_count += 1

        annotated_frame = cv2.resize(frame, (W, H))
        annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=tracked_detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=tracked_detections, labels=labels)
        annotated_frame = line_zone_annotator.annotate(annotated_frame, line_counter=line_zone_)
        # annotated_frame = line_zone_annotator.annotate(annotated_frame, line_counter=line_zone)

        sink.write_frame(annotated_frame)
        # break
