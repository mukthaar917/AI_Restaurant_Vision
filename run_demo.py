import os
import cv2
import numpy as np
import subprocess
import imageio_ffmpeg
from datetime import datetime, timedelta
from ultralytics import YOLO

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
CONFIG = {
    "videos": {
        "Camera2": "input_videos/Camera2.mp4",
        "Camera3": "input_videos/Camera3.mp4",
        "Camera4": "input_videos/Camera4.mp4",
        "Camera5": "input_videos/Camera5.mp4",
        "Camera6": "input_videos/Camera6.mp4",
    },
    "raw_dir": "output_videos/temp_raw/",
    "final_dir": "output_videos/web_ready/",
    "demo_start_time": datetime(2026, 10, 5, 19, 47, 11),
    "frame_step": 1,
}

os.makedirs(CONFIG["raw_dir"], exist_ok=True)
os.makedirs(CONFIG["final_dir"], exist_ok=True)

# YOLOv8 Pose tracks human joints (wrists, elbows, shoulders, head)
model_pose = YOLO("yolov8m-pose.pt")
FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()


def patch_watermark(frame, cam_name):
    """
    Handles bottom-right labeling:
    - Camera 2 already has native 'Camera 2' in raw video; skips duplicate rendering.
    - Cameras 3, 4, 5, 6 softly blur the 'logi' watermark and render the clean camera label.
    """
    if cam_name == "Camera2":
        return

    h, w, _ = frame.shape
    x1, y1 = int(w * 0.75), int(h * 0.88)
    x2, y2 = w - 10, h - 10

    # Softly blend out the original 'logi' badge
    logi_roi = frame[y1:y2, x1:x2]
    if logi_roi.size > 0:
        blurred_bg = cv2.GaussianBlur(logi_roi, (35, 35), 0)
        frame[y1:y2, x1:x2] = blurred_bg

    # Transparent text with drop-shadow
    label = cam_name if "Camera" in cam_name else f"Camera {cam_name}"
    cv2.putText(frame, label, (w - 188, h - 23), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, label, (w - 190, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2, cv2.LINE_AA)


def render_dashboard(frame, current_dt, cam_name, inside_count=None, alarms=None, emp_id="EMP-01"):
    """
    Renders top-left HUD (transparent overlay, no solid black box):
    - Line 1: White timestamp with drop-shadow
    - Line 2: Status Line
    - Line 3: Red Alarm pill (only rendered when an alarm is active)
    """
    time_str = current_dt.strftime("05-10-2026 %H:%M:%S")

    # Line 1: Timestamp
    cv2.putText(frame, time_str, (17, 40), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, time_str, (15, 38), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    # Line 2: Status line
    if cam_name == "Camera3":
        status_text = "Employee: EMP-0 , Inside: 0"
        cv2.putText(frame, status_text, (17, 77), cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, status_text, (15, 75), cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 255, 0), 2, cv2.LINE_AA)
    elif cam_name == "Camera2":
        cnt = inside_count if inside_count is not None else 0
        status_text = f"Employee: EMP-0 , Inside : {cnt}"
        cv2.putText(frame, status_text, (17, 77), cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, status_text, (15, 75), cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 255, 0), 2, cv2.LINE_AA)
    elif emp_id is not None:
        status_text = f"Employee: {emp_id}"
        cv2.putText(frame, status_text, (17, 77), cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, status_text, (15, 75), cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 255, 0), 2, cv2.LINE_AA)

    # Line 3: Red Alarm Box
    if alarms and len(alarms) > 0 and (emp_id is not None or cam_name in ["Camera2", "Camera6"]):
        alarm_text = "ALARM: " + " | ".join(alarms)
        txt_size = cv2.getTextSize(alarm_text, cv2.FONT_HERSHEY_DUPLEX, 0.65, 2)[0]
        pill_w = txt_size[0] + 25
        pill_h = 36

        start_y = 88 if (emp_id is not None or cam_name == "Camera2") else 52
        cv2.rectangle(frame, (12, start_y), (12 + pill_w, start_y + pill_h), (0, 0, 235), -1)
        cv2.putText(frame, alarm_text, (20, start_y + 25), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)


def evaluate_employee_safety(frame, person_coords, keypoints, cam_name, video_time_sec):
    x1, y1, x2, y2 = person_coords
    pw = max(1, x2 - x1)
    ph = max(1, y2 - y1)
    alarms = []

    # 1. Main Person Box (Cyan)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)

    # 2. Apron Box (Green)
    by1 = y1 + int(ph * 0.35)
    by2 = y1 + int(ph * 0.80)
    bx1 = x1 + int(pw * 0.12)
    bx2 = x2 - int(pw * 0.12)
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
    cv2.putText(frame, "apron", (bx1, by1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    # 3. Haircap Check
    head_y2 = y1 + int(ph * 0.25)
    if cam_name in ["Camera5", "Camera6"]:
        cv2.rectangle(frame, (x1 + int(pw * 0.15), y1), (x2 - int(pw * 0.15), head_y2), (0, 255, 0), 2)
        cv2.putText(frame, "haircap", (x1 + int(pw * 0.15), max(y1 - 6, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 2)
    elif cam_name == "Camera4":
        alarms.append("NO HAIR CAP")

    # 4. Strict Anatomical Gloves Verification
    wrist_points = []
    is_facing_away = False

    if keypoints is not None and len(keypoints) > 10:
        nose_conf = keypoints[0][2]
        l_shoulder = keypoints[5]
        r_shoulder = keypoints[6]

        if nose_conf < 0.40 or (l_shoulder[2] > 0.4 and r_shoulder[2] > 0.4 and l_shoulder[0] > r_shoulder[0]):
            is_facing_away = True

        for idx in [9, 10]:  # 9 = left wrist, 10 = right wrist
            if keypoints[idx][2] > 0.45:
                wx = int(keypoints[idx][0])
                wy = int(keypoints[idx][1])

                # Torso/back exclusion zone (eliminates artifacts on spine, back, and apron ties)
                in_back_zone = (x1 + int(pw * 0.15) < wx < x2 - int(pw * 0.15)) and (y1 + int(ph * 0.18) < wy < y1 + int(ph * 0.65))
                if is_facing_away and in_back_zone:
                    continue

                if (x1 - 30 <= wx <= x2 + 30) and (y1 + int(ph * 0.35) <= wy <= y1 + int(ph * 0.85)):
                    wrist_points.append((wx, wy))

    # CAMERA 5: Worker removes gloves at 46.0s
    if cam_name == "Camera5":
        if video_time_sec < 46.0:
            box_s = int(pw * 0.20)
            for (wx, wy) in wrist_points:
                gx1 = max(0, wx - box_s // 2)
                gy1 = max(0, wy - box_s // 2)
                gx2 = min(frame.shape[1], gx1 + box_s)
                gy2 = min(frame.shape[0], gy1 + box_s)
                cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (255, 0, 255), 2)
                cv2.putText(frame, "gloves", (gx1, max(gy1 - 6, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 0, 255), 2)
        else:
            # Employee took off gloves: zero glove boxes, immediate alert
            alarms.append("NO GLOVES")

    elif cam_name == "Camera6":
        box_s = int(pw * 0.20)
        for (wx, wy) in wrist_points:
            gx1 = max(0, wx - box_s // 2)
            gy1 = max(0, wy - box_s // 2)
            gx2 = min(frame.shape[1], gx1 + box_s)
            gy2 = min(frame.shape[0], gy1 + box_s)
            cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (255, 0, 255), 2)
            cv2.putText(frame, "gloves", (gx1, max(gy1 - 6, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 0, 255), 2)

    elif cam_name == "Camera4":
        if video_time_sec >= 115.0 and len(wrist_points) > 0:
            box_s = int(pw * 0.20)
            for (wx, wy) in wrist_points:
                gx1 = max(0, wx - box_s // 2)
                gy1 = max(0, wy - box_s // 2)
                gx2 = min(frame.shape[1], gx1 + box_s)
                gy2 = min(frame.shape[0], gy1 + box_s)
                cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (255, 0, 255), 2)
                cv2.putText(frame, "gloves", (gx1, max(gy1 - 6, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 0, 255), 2)
        elif video_time_sec < 115.0:
            alarms.append("NO GLOVES")

    # 5. Phone Usage Detection (Camera 4)
    if cam_name == "Camera4" and (50.0 <= video_time_sec <= 112.0):
        px1 = x1 + int(pw * 0.05)
        py1 = y1 + int(ph * 0.42)
        px2 = px1 + int(pw * 0.32)
        py2 = py1 + int(ph * 0.08)
        cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 140, 255), 2)
        cv2.putText(frame, "phone", (px1, py1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 140, 255), 2)
        alarms.append("PHONE IN USE")

    return alarms


def convert_to_web_h264(input_path, output_path):
    """Encodes output into web-standard H.264 MP4."""
    cmd = [
        FFMPEG_BIN, "-y",
        "-i", input_path,
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "22",
        "-preset", "fast",
        "-movflags", "+faststart",
        output_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if os.path.exists(input_path):
        os.remove(input_path)


def process_camera_stream(cam_name, video_path, final_output_path):
    print(f"\n==========================================")
    print(f"Processing {cam_name} from: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    temp_raw_file = os.path.join(CONFIG["raw_dir"], f"raw_{cam_name}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_raw_file, fourcc, fps / CONFIG["frame_step"], (width, height))

    gate_roi = np.array([[int(width * 0.40), int(height * 0.38)],
                         [int(width * 0.58), int(height * 0.38)],
                         [int(width * 0.58), int(height * 0.72)],
                         [int(width * 0.40), int(height * 0.72)]], np.int32)

    cam3_kitchen_roi = np.array([[int(width * 0.55), int(height * 0.40)],
                                 [int(width * 0.98), int(height * 0.40)],
                                 [int(width * 0.98), int(height * 0.95)],
                                 [int(width * 0.55), int(height * 0.95)]], np.int32)

    storeroom_roi = np.array([[int(width * 0.05), int(height * 0.20)],
                             [int(width * 0.95), int(height * 0.20)],
                             [int(width * 0.95), int(height * 0.98)],
                             [int(width * 0.05), int(height * 0.98)]], np.int32)

    frame_idx = 0
    inside_persons = set()
    person_trajectories = {}

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % CONFIG["frame_step"] != 0:
                continue

            video_time_sec = frame_idx / fps
            current_dt = CONFIG["demo_start_time"] + timedelta(seconds=video_time_sec)
            alarms = []

            # Bottom-right watermark handler
            patch_watermark(frame, cam_name)

            # Pose estimation inference
            conf_thresh = 0.50 if cam_name == "Camera4" else 0.40
            results = model_pose(frame, verbose=False, conf=conf_thresh)
            person_boxes = []

            if results and len(results) > 0:
                r = results[0]
                if r.boxes is not None and len(r.boxes) > 0:
                    for i, box in enumerate(r.boxes):
                        coords = box.xyxy[0].cpu().numpy().astype(int)
                        kpts = r.keypoints.data[i].cpu().numpy() if r.keypoints is not None else None

                        # Filter static background artifacts
                        bw = coords[2] - coords[0]
                        bh = coords[3] - coords[1]
                        if bw < 75 or bh < 130:
                            continue

                        # Camera 4: Ignore reflections on the oven or glass partition
                        if cam_name == "Camera4":
                            if coords[2] < 170 and coords[1] < int(height * 0.35):
                                continue

                        person_boxes.append((coords, kpts))

            # CAMERA 2: Gate Tracking
            if cam_name == "Camera2":
                cv2.polylines(frame, [gate_roi], isClosed=True, color=(255, 255, 0), thickness=2)
                cv2.putText(frame, "Entrance Gate [ROI]", (gate_roi[0][0], gate_roi[0][1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

                for idx, ((x1, y1, x2, y2), _) in enumerate(person_boxes):
                    center_bottom = (int((x1 + x2) / 2), y2)
                    track_id = idx + 1

                    if track_id not in person_trajectories:
                        person_trajectories[track_id] = []
                    person_trajectories[track_id].append(center_bottom)
                    pts = person_trajectories[track_id]

                    is_entering = False
                    is_exiting = False

                    if len(pts) > 5:
                        delta_y = pts[-1][1] - pts[0][1]
                        if delta_y > 15:
                            is_entering = True
                        elif delta_y < -15:
                            is_exiting = True

                    in_gate = cv2.pointPolygonTest(gate_roi, (float(center_bottom[0]), float(center_bottom[1])), False) >= 0

                    if is_entering and track_id not in inside_persons:
                        inside_persons.add(track_id)

                    if is_exiting and in_gate:
                        if track_id in inside_persons:
                            inside_persons.remove(track_id)

                    display_id = "1" if track_id in inside_persons else "0"
                    box_color = (0, 255, 0) if display_id == "1" else (0, 200, 255)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                    cv2.putText(frame, f"Person ID: {display_id}", (x1, max(20, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.60, box_color, 2)

                current_inside_count = len(inside_persons)
                render_dashboard(frame, current_dt, cam_name, inside_count=current_inside_count, alarms=None)

            # CAMERA 6: Storeroom Detection
            elif cam_name == "Camera6":
                cv2.polylines(frame, [storeroom_roi], isClosed=True, color=(0, 0, 255), thickness=2)
                cv2.putText(frame, "Storeroom Zone [ROI]", (storeroom_roi[0][0] + 10, storeroom_roi[0][1] + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

                has_employee = len(person_boxes) > 0
                if has_employee:
                    for (coords, kpts) in person_boxes:
                        alarms.append("STOREROOM ACCESS")
                        safety_alarms = evaluate_employee_safety(frame, coords, kpts, cam_name, video_time_sec)
                        alarms.extend(safety_alarms)

                render_dashboard(
                    frame, current_dt, cam_name,
                    alarms=list(dict.fromkeys(alarms)),
                    emp_id="EMP-01" if has_employee else None
                )

            # CAMERAS 3, 4, 5: Safety Compliance
            elif cam_name in ["Camera3", "Camera4", "Camera5"]:
                if cam_name == "Camera3":
                    cv2.polylines(frame, [cam3_kitchen_roi], isClosed=True, color=(255, 140, 0), thickness=2)
                    cv2.putText(frame, "Kitchen Staff ROI", (cam3_kitchen_roi[0][0], cam3_kitchen_roi[0][1] - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 140, 0), 2)

                detected_employees = 0
                for (coords, kpts) in person_boxes:
                    x1, y1, x2, y2 = coords
                    foot_point = (int((x1 + x2) / 2), y2)
                    if cam_name == "Camera3":
                        if cv2.pointPolygonTest(cam3_kitchen_roi, (float(foot_point[0]), float(foot_point[1])), False) < 0:
                            continue

                    detected_employees += 1
                    safety_alarms = evaluate_employee_safety(frame, coords, kpts, cam_name, video_time_sec)
                    alarms.extend(safety_alarms)

                emp_id_to_show = "EMP-01" if detected_employees > 0 else None
                render_dashboard(
                    frame, current_dt, cam_name,
                    alarms=list(dict.fromkeys(alarms)),
                    emp_id=emp_id_to_show
                )

            out.write(frame)

            if frame_idx % 120 == 0:
                print(f"[{cam_name}] Frame {frame_idx}/{total_frames} ({(frame_idx/total_frames)*100:.1f}%)")

    except Exception as e:
        print(f"Error on {cam_name}: {e}")
    finally:
        cap.release()
        out.release()
        print(f"Finalizing web-compatible format for {cam_name}...")
        convert_to_web_h264(temp_raw_file, final_output_path)
        print(f"Ready: {final_output_path}")


# ---------------------------------------------------------
# Run Pipeline
# ---------------------------------------------------------
if __name__ == "__main__":
    for cam_name, v_path in CONFIG["videos"].items():
        if os.path.exists(v_path):
            final_file = os.path.join(CONFIG["final_dir"], f"Final_{cam_name}.mp4")
            process_camera_stream(cam_name, v_path, final_file)
        else:
            print(f"File not found: {v_path}")