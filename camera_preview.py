"""Camera preview: shows the webcam with face detection, FPS and brightness.

Use it to check the camera works, position yourself and check lighting before benchmarking.
Keys: q / Esc = quit, m = toggle mirror.
"""

import os
import sys
import time

os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

import cv2  # noqa: E402
import numpy as np  # noqa: E402


def main():
    camera_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    print(f"Opening camera {camera_index} (this can take a few seconds)...")
    cap = cv2.VideoCapture(camera_index, cv2.CAP_MSMF)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("ERROR: could not open the camera. Close other apps using it (Camera, Zoom, Teams, browser).")
        input("Press Enter to exit...")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    detector = None
    try:
        import onnxruntime as ort

        ort.preload_dlls()
        from insightface.app import FaceAnalysis

        detector = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"],
                                providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        detector.prepare(ctx_id=0, det_size=(640, 640))
    except Exception as e:
        print(f"Face detection unavailable ({e}); showing camera only.")

    window = "Camera Preview - q to quit, m to mirror"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 960, 720)

    mirror = True
    failures = 0
    times = []
    while True:
        ok, frame = cap.read()
        if not ok:
            failures += 1
            if failures > 30:
                print("ERROR: camera stopped sending frames.")
                break
            continue
        failures = 0

        now = time.perf_counter()
        times = [t for t in times if now - t < 1.0] + [now]

        if mirror:
            frame = cv2.flip(frame, 1)

        brightness = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()
        faces = detector.get(frame) if detector is not None else []
        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{face.det_score:.2f}", (x1, max(y1 - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if detector is None:
            status, color = "Detection off", (200, 200, 200)
        elif faces:
            status, color = f"FACE DETECTED ({len(faces)})", (0, 220, 0)
        else:
            status, color = "NO FACE - move into view / add light", (0, 0, 255)
        light = "too dark" if brightness < 60 else "too bright" if brightness > 200 else "ok"

        overlay = np.zeros((70, frame.shape[1], 3), np.uint8)
        frame[:70] = cv2.addWeighted(frame[:70], 0.4, overlay, 0.6, 0)
        cv2.putText(frame, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        cv2.putText(frame, f"FPS {len(times):d}   brightness {brightness:.0f} ({light})   {frame.shape[1]}x{frame.shape[0]}",
                    (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        cv2.imshow(window, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key == ord("m"):
            mirror = not mirror

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
