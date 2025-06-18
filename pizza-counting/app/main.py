import os
import cv2
import supervision as sv
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

# Load YOLO model
model = YOLO("/app/best1.pt")

# Input/output video paths from environment variables
SOURCE_VIDEO_PATH = os.getenv("INPUT_VIDEO", "./videos/input107.mp4")
TARGET_VIDEO_PATH = os.getenv("OUTPUT_VIDEO", "./videos/output107.avi")


# Video resolution and info
W, H = 1280, 720
video_info = sv.VideoInfo(
    width=W,
    height=H,
    fps=24,
    total_frames=sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH).total_frames
)

# Frame range for processing
START_FRAME = 32000
END_FRAME = 32100

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

triggered_ids_line_zone = set()
triggered_ids_line_zone_ = set()

# Video frame generator
generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH, start=START_FRAME, end=END_FRAME, stride=1)

# Process and save video
with sv.VideoSink(target_path=TARGET_VIDEO_PATH, video_info=video_info) as sink:
    for frame in tqdm(generator, total=END_FRAME - START_FRAME + 1):
        resized_frame = cv2.resize(frame, (640, 640)).astype("uint8")
        prediction = model(resized_frame, conf=0.2, iou=0.5, verbose=True)[0]

        # Scale detections back to original frame size
        xyxy = prediction.boxes.xyxy.cpu().numpy()
        xyxy[..., 0] *= W / 640
        xyxy[..., 1] *= H / 640
        xyxy[..., 2] *= W / 640
        xyxy[..., 3] *= H / 640

        detections = sv.Detections(
            xyxy=xyxy,
            confidence=prediction.boxes.conf.cpu().numpy(),
            class_id=prediction.boxes.cls.cpu().numpy().astype(int)
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


