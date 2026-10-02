"""
Deep Face Net - Main Module
Contains all CLI and GUI logic
"""

import sys
import argparse
import logging
from pathlib import Path

from core.image_processor import process_image


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Deep Face Net - Real-time Face Swapping",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch GUI application (default)
  python run.py

  # Process image file
  python run.py --source face.jpg --target photo.jpg --output result.jpg

  # Several photos (or a folder) of the source person, averaged for a stabler identity
  python run.py --source C:/photos/person/ --webcam

  # Live webcam mode (CLI)
  python run.py --source face.jpg --webcam

  # Live webcam with specific camera
  python run.py --source face.jpg --webcam --camera-index 1
        """,
    )

    # Input/Output arguments
    parser.add_argument(
        "-s", "--source", nargs="+",
        help="Source photo(s) or folder(s) of the face to swap from; several photos are averaged",
    )

    parser.add_argument("-t", "--target", type=str, help="Path to target image file")

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Path to output file (required for image processing)",
    )

    # Webcam mode
    parser.add_argument(
        "--webcam",
        action="store_true",
        help="Use webcam for live face swapping (CLI mode)",
    )

    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Camera index for webcam mode (default: 0)",
    )

    return parser.parse_args()


def launch_gui():
    """Launch the Qt GUI application"""
    print("Launching Deep Face Net GUI...")
    from app.deepfake_app import main as gui_main

    gui_main()


def run_cli(args):
    """Run CLI mode based on arguments"""
    from core.face_analyser import get_face_analyser, get_live_face_analyser

    # Validate source image
    if not args.source:
        print("Error: --source argument is required for CLI mode")
        sys.exit(1)

    from core.source_faces import build_source_identity

    print(f"Loading source photo(s): {', '.join(args.source)}")
    try:
        identity = build_source_identity(args.source, get_face_analyser())
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    source_face = identity.face
    print(f"✓ Source face: {identity.summary()}")

    # Target frames only need detection + landmarks
    face_analyser = get_live_face_analyser()

    # Route to appropriate mode
    if args.webcam:
        run_webcam_mode(source_face, face_analyser, args)
    elif args.target:
        run_file_mode(source_face, face_analyser, args)
    else:
        print("Error: Either --target or --webcam must be specified")
        sys.exit(1)


def run_webcam_mode(source_face, face_analyser, args):
    """Run live webcam face swapping in CLI mode"""
    import cv2
    from core.engine.face_swapper import swap_face

    print(f"Starting webcam (camera index: {args.camera_index})...")
    print("Press 'q' to quit, 's' to save screenshot")

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print(f"Error: Failed to open camera {args.camera_index}")
        sys.exit(1)

    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    screenshot_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame from camera")
                break

            # Mirror effect
            frame = cv2.flip(frame, 1)

            # Detect faces and swap
            faces = face_analyser.get(frame)
            for face in faces:
                frame = swap_face(source_face, face, frame)

            # Display
            cv2.imshow("Deep Face Net - Webcam (Press 'q' to quit, 's' to save)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                screenshot_count += 1
                filename = f"screenshot_{screenshot_count:03d}.jpg"
                cv2.imwrite(filename, frame)
                print(f"✓ Screenshot saved: {filename}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Webcam closed")


def run_file_mode(source_face, face_analyser, args):
    """Process image or video file"""

    target_path = Path(args.target)
    if not target_path.exists():
        print(f"Error: Target file not found: {args.target}")
        sys.exit(1)

    # Determine target format
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    target_ext = target_path.suffix.lower()

    if target_ext in image_extensions:
        process_image(source_face, face_analyser, args)
    else:
        print(
            f"Error: Unsupported file format: {target_ext}. Only images are supported in CLI."
        )
        sys.exit(1)

def setup_logging():
    """Log to the console and to a rotating file at ~/.deepfacenet/deepfacenet.log"""
    from logging.handlers import RotatingFileHandler

    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    try:
        log_dir = Path.home() / ".deepfacenet"
        log_dir.mkdir(exist_ok=True)
        handlers.append(RotatingFileHandler(log_dir / "deepfacenet.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8"))
    except OSError:
        pass
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)


def main():
    """Main entry point - routes to GUI or CLI based on arguments"""
    # Windows consoles/pipes default to cp1252, which can't print "✓" and would crash the CLI
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    setup_logging()

    args = parse_arguments()

    # Determine mode: GUI or CLI
    if len(sys.argv) == 1:
        # No arguments provided - launch GUI
        launch_gui()
    else:
        # Arguments provided - run CLI mode
        run_cli(args)
