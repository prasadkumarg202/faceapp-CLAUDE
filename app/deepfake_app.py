"""
Qt Deepfake Application - Real-time Face Swapping
A professional PyQt6 application for live deepfake face swapping
"""

import sys
import logging
import cv2
import numpy as np
import json
from pathlib import Path

# Try to import pyvirtualcam (optional dependency)
try:
    import pyvirtualcam
    VIRTUAL_CAM_AVAILABLE = True
except ImportError:
    VIRTUAL_CAM_AVAILABLE = False
    logging.getLogger(__name__).info("pyvirtualcam not available - virtual camera feature disabled")
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QMessageBox,
    QGroupBox,
    QStatusBar,
    QFrame,
    QCheckBox,
    QTabWidget,
    QProgressBar,
    QDialog,
    QLineEdit,
    QComboBox,
    QScrollArea,
    QSlider,
)
from PyQt6.QtCore import Qt, QUrl, QThread, pyqtSignal, PYQT_VERSION_STR, qVersion
from PyQt6.QtGui import QImage, QPixmap, QFont, QDesktopServices, QIcon

from app.video_thread import VideoThread
from core.face_analyser import get_face_analyser
from download_models import MODELS, DownloadCancelled, check_model_status, fetch_model, format_size

logger = logging.getLogger(__name__)


def model_size_text(model_info, is_downloaded, current_size, path):
    """Size line for a model card. Zip packs are bigger once unpacked, so label both sizes."""
    expected = format_size(model_info.get("size", 0))
    if model_info.get("location") == "insightface":
        text = f"Download: {expected} (zip)"
        if is_downloaded:
            text += f"  |  Installed: {format_size(current_size)}"
    else:
        text = f"Size: {expected}"
        if is_downloaded:
            text += f"  |  Downloaded: {format_size(current_size)}"
    if is_downloaded and path:
        text += f"  |  {path}"
    return text


class ModelDownloadThread(QThread):
    """Background thread for downloading a model file"""

    progress_update = pyqtSignal(int)  # percentage 0-100
    download_complete = pyqtSignal(str)  # model_name
    download_error = pyqtSignal(str, str)  # model_name, error_msg
    status_update = pyqtSignal(str)  # status text

    def __init__(self, model_name, model_info):
        super().__init__()
        self.model_name = model_name
        self.model_info = model_info
        self._cancelled = False

    def run(self):
        try:
            fetch_model(
                self.model_name,
                progress=self._on_progress,
                is_cancelled=lambda: self._cancelled,
                status=self.status_update.emit,
            )
            self.download_complete.emit(self.model_name)
        except DownloadCancelled:
            pass  # the .part file is kept so the next attempt resumes
        except Exception as e:
            self.download_error.emit(self.model_name, str(e))

    def _on_progress(self, done, total):
        if total > 0:
            self.progress_update.emit(int(done * 100 / total))

    def cancel(self):
        self._cancelled = True


class CameraDetectionThread(QThread):
    """Background thread for detecting available cameras"""

    cameras_detected = pyqtSignal(list)  # Emits list of camera indices

    def __init__(self, max_cameras=6):
        super().__init__()
        self.max_cameras = max_cameras

    def run(self):
        """Detect available cameras in background"""
        available = []
        # Probing an index with no camera makes OpenCV try every backend (FFMPEG, depth-camera
        # "obsensor", ...) and print warnings/errors; on Windows only Media Foundation is relevant.
        backend = cv2.CAP_MSMF if sys.platform == "win32" else cv2.CAP_ANY
        log_level = cv2.utils.logging.getLogLevel()
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
        try:
            # Test camera indices 0-5 (covers most cases, faster than 0-10)
            for i in range(self.max_cameras):
                cap = cv2.VideoCapture(i, backend)
                if cap.isOpened():
                    # Try to read a frame to verify camera works
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    ret, _ = cap.read()
                    if ret:
                        available.append(i)
                cap.release()
        finally:
            cv2.utils.logging.setLogLevel(log_level)

        self.cameras_detected.emit(available)


class SettingsDialog(QDialog):
    """Settings dialog for configuring app preferences"""

    def __init__(self, current_working_dir, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.setMinimumWidth(500)

        self.working_dir = current_working_dir

        layout = QVBoxLayout()

        # Working Directory Section
        dir_group = QGroupBox("Working Directory")
        dir_layout = QVBoxLayout()

        info_label = QLabel("All file browsers will start from this directory:")
        info_label.setStyleSheet("color: #aaa; font-size: 11px;")
        dir_layout.addWidget(info_label)

        # Directory display and browse
        dir_row = QHBoxLayout()
        self.dir_input = QLineEdit(self.working_dir)
        self.dir_input.setReadOnly(True)
        dir_row.addWidget(self.dir_input)

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_directory)
        dir_row.addWidget(browse_btn)

        dir_layout.addLayout(dir_row)
        dir_group.setLayout(dir_layout)
        layout.addWidget(dir_group)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.accept)
        save_btn.setStyleSheet("background-color: #4CAF50; min-width: 80px;")
        button_layout.addWidget(save_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        cancel_btn.setStyleSheet("background-color: #555; min-width: 80px;")
        button_layout.addWidget(cancel_btn)

        layout.addLayout(button_layout)
        self.setLayout(layout)

        # Apply dark theme
        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e1e;
            }
            QGroupBox {
                font-weight: bold;
                color: #ffffff;
                border: 2px solid #444;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #2a2a2a;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QLabel { color: #ffffff; }
            QLineEdit {
                background-color: #2a2a2a;
                color: #ffffff;
                border: 1px solid #444;
                border-radius: 4px;
                padding: 5px;
            }
        """)

    def browse_directory(self):
        """Open directory browser"""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Working Directory",
            self.working_dir
        )
        if directory:
            self.working_dir = directory
            self.dir_input.setText(directory)

    def get_working_dir(self):
        """Return the selected working directory"""
        return self.working_dir


class DeepfakeApp(QMainWindow):
    """Main application window for deepfake face swapping"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Deep Face Net - Advanced Face Swapping")
        self.setGeometry(100, 100, 1200, 800)

        # Set app icon
        icon_path = Path(__file__).parent.parent / "assets" / "icon.png"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

        # Application state
        self.source_image = None
        self.source_face = None
        self.video_thread = None
        self.is_capturing = False
        from core.config import DEFAULT_CAMERA_INDEX
        self.selected_camera_index = DEFAULT_CAMERA_INDEX
        self.available_cameras = []
        self.virtual_cam = None  # Virtual camera instance
        self.virtual_cam_enabled = False  # Virtual camera state

        # Model download state
        self._active_downloads = {}  # model_name -> ModelDownloadThread
        self._model_cards = {}  # model_name -> dict of widgets

        # Settings
        self.settings_file = Path.home() / ".deepfacenet_settings.json"
        self.working_dir = self.load_settings()

        # Initialize UI
        self.init_ui()

        # Apply modern styling
        self.apply_styles()

    def init_ui(self):
        """Initialize the user interface"""
        # Central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Create Tab Widget
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Tab 1: Live Camera
        self.live_tab = QWidget()
        self.setup_live_tab()
        self.tabs.addTab(self.live_tab, "Live Camera")

        # Tab 2: Models
        # Tab 3: Models
        self.models_tab = QWidget()
        self.setup_models_tab()
        self.models_tab_index = self.tabs.addTab(self.models_tab, "Models")
        self.check_models_on_startup()

        # Tab 4: About
        self.about_tab = QWidget()
        self.setup_about_tab()
        self.tabs.addTab(self.about_tab, "About")

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_label = QLabel("Ready")
        self.fps_label = QLabel("FPS: 0")
        self.face_count_label = QLabel("Faces: 0")
        self.device_label = QLabel("")
        self.status_bar.addWidget(self.status_label, stretch=1)
        self.status_bar.addPermanentWidget(self.device_label)
        self.status_bar.addPermanentWidget(self.face_count_label)
        self.status_bar.addPermanentWidget(self.fps_label)

        # Add Settings button to status bar
        settings_btn = QPushButton("⚙ Settings")
        settings_btn.clicked.connect(self.open_settings)
        settings_btn.setStyleSheet("""
            QPushButton {
                background-color: #555;
                padding: 5px 15px;
                margin-left: 10px;
            }
            QPushButton:hover { background-color: #666; }
        """)
        self.status_bar.addPermanentWidget(settings_btn)

        # Detect and populate available cameras
        self.detect_and_populate_cameras()

    def setup_live_tab(self):
        """Setup the Live Camera tab"""
        layout = QHBoxLayout(self.live_tab)
        layout.setSpacing(15)
        layout.setContentsMargins(15, 15, 15, 15)

        # Left panel - Video display
        left_panel = self.create_video_panel()
        layout.addWidget(left_panel, stretch=3)

        # Right panel - Controls
        right_panel = self.create_control_panel()
        layout.addWidget(right_panel, stretch=1)

    def setup_models_tab(self):
        """Setup the Models download tab"""
        layout = QVBoxLayout(self.models_tab)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # Header
        header = QLabel("Model Manager")
        header.setStyleSheet("font-size: 18px; font-weight: bold; color: #fff; margin-bottom: 5px;")
        layout.addWidget(header)

        subtitle = QLabel("Download and manage the AI models required for face swapping.")
        subtitle.setStyleSheet("color: #aaa; font-size: 12px; margin-bottom: 10px;")
        layout.addWidget(subtitle)

        # Scroll area for model cards
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        cards_widget = QWidget()
        cards_layout = QVBoxLayout(cards_widget)
        cards_layout.setSpacing(12)

        # Show all models
        for model_name, model_info in MODELS.items():
            card = self._create_model_card(model_name, model_info)
            cards_layout.addWidget(card)

        cards_layout.addStretch()
        scroll.setWidget(cards_widget)
        layout.addWidget(scroll)

        # Refresh button
        refresh_btn = QPushButton("Refresh Status")
        refresh_btn.setFixedHeight(36)
        refresh_btn.setStyleSheet("background-color: #555; max-width: 200px;")
        refresh_btn.clicked.connect(self.refresh_model_status)
        layout.addWidget(refresh_btn, alignment=Qt.AlignmentFlag.AlignLeft)

    def _create_model_card(self, model_name, model_info):
        """Create a single model card widget"""
        is_downloaded, path, current_size = check_model_status(model_name)
        is_required = model_info.get("required", False)

        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background-color: #2a2a2a;
                border: 2px solid #444;
                border-radius: 8px;
                padding: 4px;
            }
        """)
        card_layout = QVBoxLayout(card)
        card_layout.setSpacing(2)
        card_layout.setContentsMargins(10, 8, 10, 8)

        # Row 1: Name + status icon + required badge
        top_row = QHBoxLayout()

        if is_downloaded:
            status_icon = QLabel("[OK]")
            status_icon.setStyleSheet("font-size: 12px; font-weight: bold; color: #4CAF50; border: none;")
        else:
            status_icon = QLabel("[!!]")
            status_icon.setStyleSheet("font-size: 12px; font-weight: bold; color: #FF9800; border: none;")
        status_icon.setMinimumWidth(40)
        top_row.addWidget(status_icon)

        name_label = QLabel(model_name)
        name_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #fff; border: none;")
        top_row.addWidget(name_label)

        if is_required:
            req_badge = QLabel("REQUIRED")
            req_badge.setStyleSheet("""
                background-color: #f44336;
                color: white;
                font-size: 10px;
                font-weight: bold;
                padding: 2px 8px;
                border-radius: 4px;
                border: none;
            """)
            req_badge.setFixedHeight(20)
            top_row.addWidget(req_badge)
        else:
            opt_badge = QLabel("OPTIONAL")
            opt_badge.setStyleSheet("""
                background-color: #555;
                color: #ccc;
                font-size: 10px;
                font-weight: bold;
                padding: 2px 8px;
                border-radius: 4px;
                border: none;
            """)
            opt_badge.setFixedHeight(20)
            top_row.addWidget(opt_badge)

        top_row.addStretch()
        card_layout.addLayout(top_row)

        # Row 2: Description + size
        desc_label = QLabel(model_info.get("description", ""))
        desc_label.setStyleSheet("color: #aaa; font-size: 11px; border: none;")
        card_layout.addWidget(desc_label)

        size_label = QLabel(model_size_text(model_info, is_downloaded, current_size, path))
        size_label.setStyleSheet("color: #777; font-size: 10px; border: none;")
        size_label.setWordWrap(True)
        card_layout.addWidget(size_label)

        # Row 3: Progress bar (hidden by default)
        progress = QProgressBar()
        progress.setValue(0)
        progress.setVisible(False)
        progress.setFixedHeight(18)
        progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #555;
                border-radius: 4px;
                text-align: center;
                font-size: 10px;
                color: #fff;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
            }
        """)
        card_layout.addWidget(progress)

        # Row 4: Status text (for download progress)
        status_text = QLabel("")
        status_text.setStyleSheet("color: #4CAF50; font-size: 11px; border: none;")
        status_text.setVisible(False)
        card_layout.addWidget(status_text)

        # Row 5: Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        if is_downloaded:
            dl_btn = QPushButton("Re-download")
            dl_btn.setStyleSheet("background-color: #555; max-width: 150px;")
        else:
            dl_btn = QPushButton("Download")
            dl_btn.setStyleSheet("background-color: #2196F3; max-width: 150px;")

        dl_btn.setFixedHeight(32)
        dl_btn.clicked.connect(lambda checked, mn=model_name: self.start_model_download(mn))
        btn_row.addWidget(dl_btn)

        card_layout.addLayout(btn_row)

        # Store widget references for later updates
        self._model_cards[model_name] = {
            "card": card,
            "status_icon": status_icon,
            "size_label": size_label,
            "progress": progress,
            "status_text": status_text,
            "dl_btn": dl_btn,
        }

        return card

    def check_models_on_startup(self):
        """Check required models and show warning on tab if missing"""
        missing_required = []
        for model_name, model_info in MODELS.items():
            if model_info.get("required", False):
                is_downloaded, _, _ = check_model_status(model_name)
                if not is_downloaded:
                    missing_required.append(model_name)

        if missing_required:
            self.tabs.setTabText(self.models_tab_index, "[!] Models")
            self.tabs.tabBar().setTabTextColor(
                self.models_tab_index,
                self.tabs.tabBar().tabTextColor(self.models_tab_index)
            )

    def start_model_download(self, model_name):
        """Start downloading a model in the background"""
        if model_name in self._active_downloads:
            return  # Already downloading

        model_info = MODELS.get(model_name)
        if not model_info:
            return

        widgets = self._model_cards.get(model_name)
        if not widgets:
            return

        # Update UI
        widgets["progress"].setValue(0)
        widgets["progress"].setVisible(True)
        widgets["status_text"].setText("Downloading...")
        widgets["status_text"].setVisible(True)
        widgets["status_text"].setStyleSheet("color: #2196F3; font-size: 11px; border: none;")
        widgets["dl_btn"].setEnabled(False)
        widgets["dl_btn"].setText("Downloading...")

        # Start download thread
        thread = ModelDownloadThread(model_name, model_info)
        thread.progress_update.connect(
            lambda pct, mn=model_name: self._on_download_progress(mn, pct)
        )
        thread.download_complete.connect(self._on_download_complete)
        thread.download_error.connect(self._on_download_error)
        thread.status_update.connect(
            lambda status, mn=model_name: self._on_download_status(mn, status)
        )
        thread.start()

        self._active_downloads[model_name] = thread

    def _on_download_progress(self, model_name, pct):
        widgets = self._model_cards.get(model_name)
        if widgets:
            widgets["progress"].setValue(pct)

    def _on_download_status(self, model_name, status):
        widgets = self._model_cards.get(model_name)
        if widgets:
            widgets["status_text"].setText(status)

    def _on_download_complete(self, model_name):
        widgets = self._model_cards.get(model_name)
        if widgets:
            widgets["progress"].setValue(100)
            widgets["status_text"].setText("Download complete!")
            widgets["status_text"].setStyleSheet("color: #4CAF50; font-size: 11px; border: none;")
            widgets["status_icon"].setText("[OK]")
            widgets["status_icon"].setStyleSheet("font-size: 12px; font-weight: bold; color: #4CAF50; border: none;")
            widgets["dl_btn"].setEnabled(True)
            widgets["dl_btn"].setText("Re-download")
            widgets["dl_btn"].setStyleSheet("background-color: #555; max-width: 150px;")

            # Update size label
            is_downloaded, path, current_size = check_model_status(model_name)
            model_info = MODELS.get(model_name, {})
            widgets["size_label"].setText(model_size_text(model_info, is_downloaded, current_size, path))

        self._active_downloads.pop(model_name, None)

        # Update tab warning
        self._update_models_tab_badge()

    def _on_download_error(self, model_name, error_msg):
        widgets = self._model_cards.get(model_name)
        if widgets:
            widgets["progress"].setVisible(False)
            widgets["status_text"].setText(f"Error: {error_msg}")
            widgets["status_text"].setStyleSheet("color: #f44336; font-size: 11px; border: none;")
            widgets["dl_btn"].setEnabled(True)
            widgets["dl_btn"].setText("Retry")
            widgets["dl_btn"].setStyleSheet("background-color: #f44336; max-width: 150px;")

        self._active_downloads.pop(model_name, None)

    def _update_models_tab_badge(self):
        """Update the models tab title based on current model status"""
        has_missing = False
        for model_name, model_info in MODELS.items():
            if model_info.get("required", False):
                is_downloaded, _, _ = check_model_status(model_name)
                if not is_downloaded:
                    has_missing = True
                    break

        self.tabs.setTabText(
            self.models_tab_index,
            "[!] Models" if has_missing else "Models"
        )

    def refresh_model_status(self):
        """Refresh status of all model cards"""
        for model_name in list(self._model_cards.keys()):
            if model_name in self._active_downloads:
                continue  # Don't refresh while downloading
            widgets = self._model_cards[model_name]
            model_info = MODELS.get(model_name, {})
            is_downloaded, path, current_size = check_model_status(model_name)

            if is_downloaded:
                widgets["status_icon"].setText("[OK]")
                widgets["status_icon"].setStyleSheet("font-size: 12px; font-weight: bold; color: #4CAF50; border: none;")
            else:
                widgets["status_icon"].setText("[!!]")
                widgets["status_icon"].setStyleSheet("font-size: 12px; font-weight: bold; color: #FF9800; border: none;")

            widgets["size_label"].setText(model_size_text(model_info, is_downloaded, current_size, path))

            if is_downloaded:
                widgets["dl_btn"].setText("Re-download")
                widgets["dl_btn"].setStyleSheet("background-color: #555; max-width: 150px;")
            else:
                widgets["dl_btn"].setText("Download")
                widgets["dl_btn"].setStyleSheet("background-color: #2196F3; max-width: 150px;")

            widgets["progress"].setVisible(False)
            widgets["status_text"].setVisible(False)

        self._update_models_tab_badge()
        self.status_label.setText("Model status refreshed")

    # --- Live Tab Methods ---

    def create_video_panel(self):
        """Create the video display panel"""
        group = QGroupBox("Live Video Feed")
        layout = QVBoxLayout()

        # Minimal clear button in top-right
        header = QHBoxLayout()
        header.addStretch()

        # Clear button
        self.clear_feed_btn = QPushButton("✕")
        self.clear_feed_btn.setFixedSize(20, 20)
        self.clear_feed_btn.setToolTip("Clear feed")
        self.clear_feed_btn.clicked.connect(self.clear_video_feed)
        self.clear_feed_btn.setStyleSheet("""
            QPushButton {
                background-color: #444;
                color: #fff;
                border-radius: 10px;
                font-weight: bold;
                padding: 0;
            }
            QPushButton:hover { background-color: #666; }
        """)
        header.addWidget(self.clear_feed_btn)
        layout.addLayout(header)

        # Alert banner (hidden by default)
        self.missing_model_banner = QLabel("⚠️ Face swapping disabled: 'buffalo_l' detection model is missing. Download it in the Models tab.")
        self.missing_model_banner.setStyleSheet("""
            QLabel {
                background-color: #3a2e00;
                color: #ffcc00;
                border: 1px solid #cca300;
                border-radius: 4px;
                padding: 8px;
                font-weight: bold;
            }
        """)
        self.missing_model_banner.setVisible(False)
        self.missing_model_banner.setWordWrap(True)
        layout.addWidget(self.missing_model_banner)

        # Video display label
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet(
            "QLabel { background-color: #1a1a1a; border: 2px solid #333; border-radius: 8px; }"
        )
        self.video_label.setText("Camera feed will appear here")
        layout.addWidget(self.video_label)

        group.setLayout(layout)
        return group

    def create_control_panel(self):
        """Create the control panel"""
        # Wrap in scroll area for cross-platform compatibility
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollArea > QWidget > QWidget {
                background-color: transparent;
            }
        """)

        widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(5, 5, 5, 5)

        # Source image section
        source_group = QGroupBox("Source Image")
        source_layout = QVBoxLayout()

        # Header with Clear Button
        source_header = QHBoxLayout()
        source_header.addStretch()
        self.btn_clear_live_source = QPushButton("✕")
        self.btn_clear_live_source.setFixedSize(24, 24)
        self.btn_clear_live_source.setToolTip("Clear source selection")
        self.btn_clear_live_source.setEnabled(False)
        self.btn_clear_live_source.clicked.connect(self.clear_live_source)
        self.btn_clear_live_source.setStyleSheet("""
            QPushButton {
                background-color: #444; color: #fff; border-radius: 12px; font-weight: bold; padding: 0;
            }
            QPushButton:hover { background-color: #f44336; }
            QPushButton:disabled { background-color: #333; color: #555; }
        """)
        source_header.addWidget(self.btn_clear_live_source)
        source_layout.addLayout(source_header)

        self.source_preview = QLabel()
        self.source_preview.setFixedSize(200, 200)
        self.source_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.source_preview.setStyleSheet(
            "QLabel { background-color: #2a2a2a; border: 2px solid #444; border-radius: 8px; }"
        )
        self.source_preview.setText("No image selected")
        source_layout.addWidget(
            self.source_preview, alignment=Qt.AlignmentFlag.AlignCenter
        )

        self.select_source_btn = QPushButton("Select Source Image")
        self.select_source_btn.clicked.connect(self.select_source_image)
        source_layout.addWidget(self.select_source_btn)

        self.source_status = QLabel("No source loaded")
        self.source_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.source_status.setStyleSheet("color: #888;")
        source_layout.addWidget(self.source_status)

        source_layout.setContentsMargins(10, 25, 10, 10)
        source_group.setLayout(source_layout)
        layout.addWidget(source_group)

        # Camera controls section
        camera_group = QGroupBox("Camera Controls")
        camera_layout = QVBoxLayout()
        camera_layout.setContentsMargins(10, 25, 10, 10)
        camera_layout.setSpacing(8)

        # Camera selection
        camera_select_label = QLabel("Select Camera:")
        camera_select_label.setStyleSheet("color: #ffffff; font-weight: bold; margin-top: 5px;")
        camera_layout.addWidget(camera_select_label)

        camera_select_row = QHBoxLayout()
        self.camera_combo = QComboBox()
        self.camera_combo.setMinimumHeight(30)
        self.camera_combo.currentIndexChanged.connect(self.on_camera_selection_changed)
        self.camera_combo.setStyleSheet("""
            QComboBox {
                background-color: #2a2a2a;
                color: #ffffff;
                border: 2px solid #444;
                border-radius: 5px;
                padding: 5px;
            }
            QComboBox:hover {
                border-color: #4CAF50;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: #2a2a2a;
                color: #ffffff;
                selection-background-color: #4CAF50;
            }
        """)
        camera_select_row.addWidget(self.camera_combo)

        self.refresh_cameras_btn = QPushButton("🔄")
        self.refresh_cameras_btn.setFixedSize(30, 30)
        self.refresh_cameras_btn.setToolTip("Refresh camera list")
        self.refresh_cameras_btn.clicked.connect(self.refresh_cameras)
        self.refresh_cameras_btn.setStyleSheet("""
            QPushButton {
                background-color: #555;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #666;
            }
        """)
        camera_select_row.addWidget(self.refresh_cameras_btn)
        camera_layout.addLayout(camera_select_row)

        self.start_btn = QPushButton("Start Camera")
        self.start_btn.clicked.connect(self.toggle_camera)
        self.start_btn.setFixedHeight(40)
        camera_layout.addWidget(self.start_btn)

        self.swap_btn = QPushButton("Enable Face Swap")
        self.swap_btn.clicked.connect(self.toggle_swap)
        self.swap_btn.setEnabled(False)
        self.swap_btn.setFixedHeight(40)
        camera_layout.addWidget(self.swap_btn)

        # Advanced masking section
        self.advanced_group = QGroupBox("Advanced Settings")
        advanced_layout = QVBoxLayout()
        advanced_layout.setSpacing(6)

        # Opacity slider
        opacity_layout = QHBoxLayout()
        self.opacity_label = QLabel("Opacity: 100%")
        self.opacity_label.setStyleSheet("color: white;")
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(100)
        self.opacity_slider.setEnabled(False)
        def _update_opacity(val):
            self.opacity_label.setText(f"Opacity: {val}%")
            from core import config
            config.OPACITY = val / 100.0
        self.opacity_slider.valueChanged.connect(_update_opacity)
        opacity_layout.addWidget(self.opacity_label)
        opacity_layout.addWidget(self.opacity_slider)
        advanced_layout.addLayout(opacity_layout)

        # Mouth mask checkbox
        self.mouth_mask_checkbox = QCheckBox("Enable Mouth Mask")
        self.mouth_mask_checkbox.setEnabled(False)
        def _toggle_mouth_mask(state):
            from core import config
            config.MOUTH_MASK_ENABLED = (state == Qt.CheckState.Checked.value)
        self.mouth_mask_checkbox.stateChanged.connect(_toggle_mouth_mask)
        self.mouth_mask_checkbox.setStyleSheet("QCheckBox { color: #ffffff; }")
        advanced_layout.addWidget(self.mouth_mask_checkbox)

        # Eyes mask checkbox
        self.eyes_mask_checkbox = QCheckBox("Enable Eyes Mask")
        self.eyes_mask_checkbox.setEnabled(False)
        def _toggle_eyes_mask(state):
            from core import config
            config.EYES_MASK_ENABLED = (state == Qt.CheckState.Checked.value)
        self.eyes_mask_checkbox.stateChanged.connect(_toggle_eyes_mask)
        self.eyes_mask_checkbox.setStyleSheet("QCheckBox { color: #ffffff; }")
        advanced_layout.addWidget(self.eyes_mask_checkbox)

        # Eyebrows mask checkbox
        self.eyebrows_mask_checkbox = QCheckBox("Enable Eyebrows Mask")
        self.eyebrows_mask_checkbox.setEnabled(False)
        def _toggle_eyebrows_mask(state):
            from core import config
            config.EYEBROWS_MASK_ENABLED = (state == Qt.CheckState.Checked.value)
        self.eyebrows_mask_checkbox.stateChanged.connect(_toggle_eyebrows_mask)
        self.eyebrows_mask_checkbox.setStyleSheet("QCheckBox { color: #ffffff; }")
        advanced_layout.addWidget(self.eyebrows_mask_checkbox)

        # Poisson Blending checkbox
        self.poisson_blend_checkbox = QCheckBox("Enable Seamless Blending")
        self.poisson_blend_checkbox.setEnabled(False)
        def _toggle_poisson(state):
            from core import config
            config.POISSON_BLEND_ENABLED = (state == Qt.CheckState.Checked.value)
        self.poisson_blend_checkbox.stateChanged.connect(_toggle_poisson)
        self.poisson_blend_checkbox.setStyleSheet("QCheckBox { color: #ffffff; }")
        advanced_layout.addWidget(self.poisson_blend_checkbox)

        # Occlusion mask checkbox
        self.occlusion_checkbox = QCheckBox("Keep Hands/Objects in Front of Face (slower)")
        self.occlusion_checkbox.setToolTip(
            "Hands, microphones, cups or phones in front of the face are not painted over. "
            "Costs ~26 ms per frame; needs the occlusion model (Models tab)."
        )
        from core import config as _cfg
        self.occlusion_checkbox.setChecked(_cfg.OCCLUSION_MASK_ENABLED)
        def _toggle_occlusion(state):
            from core import config
            config.OCCLUSION_MASK_ENABLED = (state == Qt.CheckState.Checked.value)
            if config.OCCLUSION_MASK_ENABLED:
                from core.engine.occlusion import get_occlusion_model
                if get_occlusion_model() is None:
                    self.status_label.setText("Occlusion model not downloaded - get it in the Models tab")
        self.occlusion_checkbox.stateChanged.connect(_toggle_occlusion)
        self.occlusion_checkbox.setStyleSheet("QCheckBox { color: #ffffff; }")
        advanced_layout.addWidget(self.occlusion_checkbox)

        # GFPGAN enhancement checkbox
        self.enhance_checkbox = QCheckBox("Enhance Face (GFPGAN, slower)")
        self.enhance_checkbox.setEnabled(False)
        self.enhance_checkbox.setToolTip("Restores face detail after the swap. Roughly halves FPS.")
        def _toggle_enhance(state):
            from core import config
            from core.engine.face_enhancer import is_enhancer_ready, warm_up_async
            config.ENHANCE_ENABLED = (state == Qt.CheckState.Checked.value)
            if config.ENHANCE_ENABLED and not is_enhancer_ready():
                warm_up_async()
                self.status_label.setText("Preparing face enhancer (first time can take a minute)...")
            elif config.ENHANCE_ENABLED:
                self.status_label.setText("Face enhancer on")
            else:
                self.status_label.setText("Face enhancer off")
        self.enhance_checkbox.stateChanged.connect(_toggle_enhance)
        self.enhance_checkbox.setStyleSheet("QCheckBox { color: #ffffff; }")
        advanced_layout.addWidget(self.enhance_checkbox)

        self.advanced_group.setLayout(advanced_layout)
        camera_layout.addWidget(self.advanced_group)

        # Virtual camera checkbox (only if available)
        if VIRTUAL_CAM_AVAILABLE:
            self.virtual_cam_checkbox = QCheckBox("Enable Virtual Camera")
            self.virtual_cam_checkbox.setEnabled(False)
            self.virtual_cam_checkbox.stateChanged.connect(self.toggle_virtual_camera)
            self.virtual_cam_checkbox.setStyleSheet(
                "QCheckBox { color: #ffffff; padding: 8px 5px; }"
                "QCheckBox::indicator { width: 18px; height: 18px; }"
            )
            self.virtual_cam_checkbox.setToolTip("Stream to virtual webcam for OBS/Zoom/Discord")
            camera_layout.addWidget(self.virtual_cam_checkbox)

        camera_group.setLayout(camera_layout)
        layout.addWidget(camera_group)

        # ── Model status (Live tab) ───────────────────────────────────────────
        from core.config import ENHANCER_MODEL, SWAPPER_MODEL, ANALYSIS_MODEL
        live_models_group = QGroupBox("Model Status")
        live_models_layout = QVBoxLayout()
        live_models_layout.setContentsMargins(10, 20, 10, 10)
        live_models_layout.setSpacing(5)

        analyser_ok, _, _ = check_model_status(ANALYSIS_MODEL)

        _models_to_show = [
            ("Detect (buffalo_l)", analyser_ok),
            ("Swap (inswapper)", SWAPPER_MODEL.exists()),
            ("Enhance (GFPGAN)", ENHANCER_MODEL.exists()),
            ("Occlusion (XSeg)", check_model_status("xseg_2.onnx")[0]),
        ]
        for name, ok in _models_to_show:
            row = QHBoxLayout()
            dot = QLabel("●")
            dot.setFixedWidth(16)
            dot.setStyleSheet(f"color: {'#4CAF50' if ok else '#f44336'}; font-size: 14px; border: none;")
            lbl = QLabel(name)
            lbl.setStyleSheet(f"color: {'#ccc' if ok else '#888'}; font-size: 11px; border: none;")
            status = QLabel("Ready" if ok else "Not downloaded")
            status.setStyleSheet(f"color: {'#4CAF50' if ok else '#f44336'}; font-size: 11px; border: none;")
            row.addWidget(dot)
            row.addWidget(lbl)
            row.addStretch()
            row.addWidget(status)
            live_models_layout.addLayout(row)

        goto_btn = QPushButton("→ Go to Models Tab")
        goto_btn.setStyleSheet("""
            QPushButton { background-color: #333; color: #58a6ff; font-size: 11px;
                          border: 1px solid #444; border-radius: 4px; padding: 4px 8px; }
            QPushButton:hover { background-color: #444; }
        """)
        goto_btn.clicked.connect(lambda: self.tabs.setCurrentIndex(self.models_tab_index))
        live_models_layout.addWidget(goto_btn)
        live_models_group.setLayout(live_models_layout)
        layout.addWidget(live_models_group)

        # Info section
        info_group = QGroupBox("Information")
        info_layout = QVBoxLayout()
        info_layout.setContentsMargins(10, 25, 10, 10)

        info_text = QLabel(
            "1. Select a source image\n"
            "2. Start the camera\n"
            "3. Enable face swap\n\n"
            "The source face will be\n"
            "swapped onto detected faces\n"
            "in the live video feed."
        )
        info_text.setWordWrap(True)
        info_text.setStyleSheet("color: #aaa; font-size: 11px;")
        info_layout.addWidget(info_text)

        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        layout.addStretch()
        widget.setLayout(layout)
        scroll_area.setWidget(widget)
        return scroll_area

    def detect_and_populate_cameras(self):
        """Detect cameras in background and populate the combo box"""
        self.status_label.setText("Detecting cameras...")
        self.camera_combo.setEnabled(False)
        self.refresh_cameras_btn.setEnabled(False)

        # Start background detection thread
        self.camera_detection_thread = CameraDetectionThread(max_cameras=6)
        self.camera_detection_thread.cameras_detected.connect(self.on_cameras_detected)
        self.camera_detection_thread.start()

    def on_cameras_detected(self, available_cameras):
        """Handle camera detection results from background thread"""
        self.available_cameras = available_cameras

        self.camera_combo.clear()
        if self.available_cameras:
            for cam_idx in self.available_cameras:
                self.camera_combo.addItem(f"Camera {cam_idx}", cam_idx)
            self.status_label.setText(f"Found {len(self.available_cameras)} camera(s)")
            # Set default selection
            if self.selected_camera_index in self.available_cameras:
                idx = self.available_cameras.index(self.selected_camera_index)
                self.camera_combo.setCurrentIndex(idx)
        else:
            self.camera_combo.addItem("No cameras found", -1)
            self.selected_camera_index = -1
            self.status_label.setText("No cameras detected")

        self.camera_combo.setEnabled(True)
        self.refresh_cameras_btn.setEnabled(True)

    def refresh_cameras(self):
        """Refresh the camera list"""
        if self.is_capturing:
            QMessageBox.warning(
                self,
                "Camera Active",
                "Please stop the camera before refreshing the camera list."
            )
            return
        self.detect_and_populate_cameras()

    def on_camera_selection_changed(self, index):
        """Handle camera selection change"""
        if index >= 0:
            self.selected_camera_index = self.camera_combo.itemData(index)
            if self.selected_camera_index >= 0:
                self.status_label.setText(f"Selected Camera {self.selected_camera_index}")

    def select_source_image(self):
        """Select one or more photos of the source person for live mode.

        Several photos (e.g. front, left, right, up, down) are combined into one averaged
        identity, which is more stable than a single photo; photos without a face or that
        don't match the others are skipped.
        """
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Source Photo(s) - select several photos of the same person for a better match",
            self.working_dir,
            "Image Files (*.png *.jpg *.jpeg *.bmp *.webp)",
        )

        if file_paths:
            try:
                from core.source_faces import build_source_identity

                self.status_label.setText(f"Analyzing {len(file_paths)} photo(s)...")
                QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
                QApplication.processEvents()  # Keep UI responsive

                try:
                    identity = build_source_identity(file_paths, get_face_analyser())
                except ValueError as e:
                    QApplication.restoreOverrideCursor()
                    QMessageBox.warning(self, "No Face Detected", f"{e} Please choose photos with a clear face.")
                    self.status_label.setText("Ready")
                    return

                self.source_identity = identity
                self.source_face = identity.face
                self.source_image = cv2.imdecode(np.fromfile(str(identity.preview_path), np.uint8), cv2.IMREAD_COLOR)

                # Display preview
                self.display_source_preview(self.source_image, self.source_preview)
                used = len(identity.used)
                self.source_status.setText(
                    "✓ Face detected" if len(identity.photos) == 1 else f"✓ {used} of {len(identity.photos)} photos used"
                )
                self.source_status.setToolTip(identity.summary())
                self.source_status.setStyleSheet("color: #4CAF50;")
                self.btn_clear_live_source.setEnabled(True)
                logger.info("Source identity: %s", identity.summary())

                # Update video thread if running
                if self.video_thread is not None:
                    self.video_thread.set_source_face(self.source_face)
                    self.swap_btn.setEnabled(True)

                self.status_label.setText(identity.summary())
                self.update_device_label()
                QApplication.restoreOverrideCursor()

            except Exception as e:
                QApplication.restoreOverrideCursor()
                QMessageBox.critical(
                    self, "Error", f"Failed to process image: {str(e)}"
                )
                self.status_label.setText("Ready")

    def clear_live_source(self):
        """Clear the live source image selection"""
        self.source_image = None
        self.source_face = None
        self.source_preview.clear()
        self.source_preview.setText("No image selected")
        self.source_status.setText("No source loaded")
        self.source_status.setStyleSheet("color: #888;")
        self.btn_clear_live_source.setEnabled(False)

        # Update video thread if running
        if self.video_thread is not None:
            self.video_thread.set_source_face(None)
            self.swap_btn.setEnabled(False)
            # If swap was enabled, disable it
            if self.video_thread.swap_enabled:
                self.toggle_swap()

        self.status_label.setText("Source image cleared")

    def display_source_preview(self, image, label_widget):
        """Display source image preview on a specific label"""
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize to fit preview
        h, w = rgb_image.shape[:2]
        aspect = w / h
        if aspect > 1:
            new_w = 200
            new_h = int(200 / aspect)
        else:
            new_h = 200
            new_w = int(200 * aspect)

        resized = cv2.resize(rgb_image, (new_w, new_h))

        # Convert to QPixmap
        q_image = QImage(
            resized.data, new_w, new_h, new_w * 3, QImage.Format.Format_RGB888
        )
        pixmap = QPixmap.fromImage(q_image)

        label_widget.setPixmap(pixmap)

    def toggle_camera(self):
        """Start or stop camera capture"""
        if not self.is_capturing:
            self.start_camera()
        else:
            self.stop_camera()

    def start_camera(self):
        """Start video capture"""
        try:
            # Validate camera selection
            if self.selected_camera_index < 0:
                QMessageBox.warning(
                    self,
                    "No Camera",
                    "No camera selected. Please select a valid camera from the dropdown."
                )
                return

            # Create and start video thread with selected camera
            self.video_thread = VideoThread(camera_index=self.selected_camera_index)
            self.video_thread.frame_ready.connect(self.update_frame)
            self.video_thread.fps_update.connect(self.update_fps)
            self.video_thread.face_count_update.connect(self.update_face_count)
            self.video_thread.error_occurred.connect(self.handle_error)
            self.video_thread.virtual_cam_error.connect(self.handle_virtual_cam_error)

            # Set source face if available
            if self.source_face is not None:
                self.video_thread.set_source_face(self.source_face)
                self.swap_btn.setEnabled(True)

            self.video_thread.start()

            self.is_capturing = True
            self.start_btn.setText("Stop Camera")
            self.start_btn.setStyleSheet("background-color: #f44336;")
            self.camera_combo.setEnabled(False)  # Disable camera selection while running
            self.refresh_cameras_btn.setEnabled(False)

            # Enable virtual camera checkbox if available
            if VIRTUAL_CAM_AVAILABLE:
                self.virtual_cam_checkbox.setEnabled(True)

            # Show a subtle banner if the analyser model is missing
            from core.config import ANALYSIS_MODEL
            from download_models import check_model_status
            is_downloaded, _, _ = check_model_status(ANALYSIS_MODEL)
            self.missing_model_banner.setVisible(not is_downloaded)

            self.status_label.setText(f"Camera {self.selected_camera_index} started")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start camera: {str(e)}")

    def stop_camera(self):
        """Stop video capture"""
        self.missing_model_banner.setVisible(False)
        if self.video_thread is not None:
            self.video_thread.stop()
            self.video_thread = None

        self.is_capturing = False
        self.start_btn.setText("Start Camera")
        self.start_btn.setStyleSheet("QPushButton { background-color: #4CAF50; } QPushButton:hover { background-color: #45a049; } QPushButton:disabled { background-color: #555; color: #888; }")
        self.camera_combo.setEnabled(True)  # Re-enable camera selection
        self.refresh_cameras_btn.setEnabled(True)
        self.swap_btn.setEnabled(False)
        self.swap_btn.setText("Enable Face Swap")
        self.swap_btn.setStyleSheet("QPushButton { background-color: #4CAF50; } QPushButton:hover { background-color: #45a049; } QPushButton:disabled { background-color: #555; color: #888; }")
        self.mouth_mask_checkbox.setEnabled(False)
        self.eyes_mask_checkbox.setEnabled(False)
        self.eyebrows_mask_checkbox.setEnabled(False)
        self.poisson_blend_checkbox.setEnabled(False)
        self.enhance_checkbox.setEnabled(False)
        self.opacity_slider.setEnabled(False)
        self.mouth_mask_checkbox.setChecked(False)
        self.eyes_mask_checkbox.setChecked(False)
        self.eyebrows_mask_checkbox.setChecked(False)
        self.poisson_blend_checkbox.setChecked(False)
        self.enhance_checkbox.setChecked(False)

        # Stop and disable virtual camera
        if VIRTUAL_CAM_AVAILABLE:
            if self.virtual_cam_enabled:
                self.virtual_cam_checkbox.setChecked(False)
            self.virtual_cam_checkbox.setEnabled(False)

        # Automatically clear the video feed
        self.clear_video_feed()

        self.status_label.setText("Camera stopped")
        self.fps_label.setText("FPS: 0")
        self.face_count_label.setText("Faces: 0")

    def clear_video_feed(self):
        """Clear the video feed display"""
        # Remove any pixmap and force text display
        self.video_label.setPixmap(QPixmap())
        self.video_label.clear()
        self.video_label.setText("Camera feed will appear here")
        self.video_label.repaint()  # Force immediate repaint

    def toggle_swap(self):
        """Toggle face swapping on/off"""
        if self.video_thread is not None:
            current_state = self.video_thread.swap_enabled
            new_state = not current_state
            self.video_thread.enable_swap(new_state)

            if new_state:
                self.swap_btn.setText("Disable Face Swap")
                self.swap_btn.setStyleSheet("background-color: #FF9800;")
                self.status_label.setText("Face swapping enabled")
                self.mouth_mask_checkbox.setEnabled(True)
                self.eyes_mask_checkbox.setEnabled(True)
                self.eyebrows_mask_checkbox.setEnabled(True)
                self.poisson_blend_checkbox.setEnabled(True)
                self.enhance_checkbox.setEnabled(True)
                self.opacity_slider.setEnabled(True)
            else:
                self.swap_btn.setText("Enable Face Swap")
                self.swap_btn.setStyleSheet("QPushButton { background-color: #4CAF50; } QPushButton:hover { background-color: #45a049; } QPushButton:disabled { background-color: #555; color: #888; }")
                self.status_label.setText("Face swapping disabled")
                self.mouth_mask_checkbox.setEnabled(False)
                self.eyes_mask_checkbox.setEnabled(False)
                self.eyebrows_mask_checkbox.setEnabled(False)
                self.poisson_blend_checkbox.setEnabled(False)
                self.enhance_checkbox.setEnabled(False)
                self.opacity_slider.setEnabled(False)
                self.mouth_mask_checkbox.setChecked(False)
                self.eyes_mask_checkbox.setChecked(False)
                self.eyebrows_mask_checkbox.setChecked(False)
                self.poisson_blend_checkbox.setChecked(False)
                self.enhance_checkbox.setChecked(False)

    def toggle_virtual_camera(self, state):
        """Toggle virtual camera output"""
        if not VIRTUAL_CAM_AVAILABLE:
            return

        enabled = state == Qt.CheckState.Checked.value

        if enabled:
            # Check if camera is running
            if not self.is_capturing or self.video_thread is None:
                self.virtual_cam_checkbox.setChecked(False)
                QMessageBox.warning(
                    self,
                    "Camera Not Running",
                    "Please start the camera before enabling virtual camera."
                )
                return

            try:
                # Get actual camera resolution from video thread
                if hasattr(self.video_thread, 'cap') and self.video_thread.cap is not None:
                    width = int(self.video_thread.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(self.video_thread.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                else:
                    # Fallback to common resolution
                    width, height = 640, 480

                # Create virtual camera with actual camera resolution
                self.virtual_cam = pyvirtualcam.Camera(width=width, height=height, fps=30, fmt=pyvirtualcam.PixelFormat.BGR)
                self.virtual_cam_enabled = True
                self.video_thread.set_virtual_camera(self.virtual_cam)
                self.status_label.setText(f"Virtual camera started: {self.virtual_cam.device} ({width}x{height})")
            except Exception as e:
                self.virtual_cam_enabled = False
                self.virtual_cam_checkbox.setChecked(False)
                QMessageBox.critical(
                    self,
                    "Virtual Camera Error",
                    f"Failed to start virtual camera:\n{str(e)}\n\nOn Linux, install v4l2loopback:\nsudo modprobe v4l2loopback"
                )
        else:
            if self.video_thread is not None:
                self.video_thread.set_virtual_camera(None)
            if self.virtual_cam is not None:
                self.virtual_cam.close()
                self.virtual_cam = None
            self.virtual_cam_enabled = False
            self.status_label.setText("Virtual camera stopped")

    def update_frame(self, frame):
        """Update video display with new frame"""
        # Convert BGR to RGB for display
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to QImage
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        q_image = QImage(
            rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
        )

        # Scale to fit label while maintaining aspect ratio
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        self.video_label.setPixmap(scaled_pixmap)

        # Virtual camera frames are sent from the video thread; tell it this frame was drawn
        if self.video_thread is not None:
            self.video_thread.frame_displayed()

    def handle_virtual_cam_error(self, error_msg):
        """Virtual camera failed in the video thread: turn the checkbox off"""
        logger.warning("Virtual camera error: %s", error_msg)
        if VIRTUAL_CAM_AVAILABLE:
            self.virtual_cam_checkbox.setChecked(False)

    def update_fps(self, fps):
        """Update FPS display"""
        self.fps_label.setText(f"FPS: {fps:.1f}")
        self.update_device_label()
        if self.status_label.text().startswith("Preparing face enhancer"):
            from core.engine.face_enhancer import is_enhancer_ready
            if is_enhancer_ready():
                self.status_label.setText("Face enhancer ready")

    def update_device_label(self):
        """Show whether inference is running on GPU or has fallen back to CPU"""
        from core.runtime import gpu_status
        status, details = gpu_status()
        if status is None:
            return
        colors = {"GPU": "#4CAF50", "Mixed": "#FF9800", "CPU": "#f44336"}
        text = {"GPU": "GPU", "Mixed": "GPU+CPU", "CPU": "CPU ONLY"}[status]
        self.device_label.setText(text)
        self.device_label.setStyleSheet(f"color: {colors[status]}; font-weight: bold; padding: 0 8px;")
        self.device_label.setToolTip(details)

    def update_face_count(self, count):
        """Update face count display"""
        self.face_count_label.setText(f"Faces: {count}")

    def handle_error(self, error_msg):
        """Handle errors from video thread"""
        QMessageBox.critical(self, "Error", error_msg)
        self.stop_camera()

    def apply_styles(self):
        """Apply modern styling to the application"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QTabWidget::pane {
                border: 1px solid #444;
                background-color: #1e1e1e;
            }
            QTabBar::tab {
                background-color: #2a2a2a;
                color: #aaa;
                padding: 10px 20px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #3a3a3a;
                color: #fff;
                font-weight: bold;
                border-bottom: 2px solid #4CAF50;
            }
            QGroupBox {
                font-weight: bold;
                font-size: 13px;
                color: #ffffff;
                border: 2px solid #444;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 18px;
                background-color: #2a2a2a;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                padding: 0 5px;
                top: 2px;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton:disabled {
                background-color: #555;
                color: #888;
            }
            QLabel {
                color: #ffffff;
            }
            QStatusBar {
                background-color: #2a2a2a;
                color: #ffffff;
            }
            QStatusBar QLabel {
                color: #ffffff;
                padding: 2px 8px;
            }
            QCheckBox {
                color: white;
            }
        """)

    def load_settings(self):
        """Load settings from JSON file"""
        try:
            if self.settings_file.exists():
                with open(self.settings_file, 'r') as f:
                    settings = json.load(f)
                    return settings.get('working_dir', str(Path.home()))
        except Exception as e:
            logger.warning("Error loading settings: %s", e)
        return str(Path.home())

    def save_settings(self):
        """Save settings to JSON file"""
        try:
            settings = {
                'working_dir': self.working_dir
            }
            with open(self.settings_file, 'w') as f:
                json.dump(settings, f, indent=2)
        except Exception as e:
            logger.warning("Error saving settings: %s", e)

    def open_settings(self):
        """Open settings dialog"""
        dialog = SettingsDialog(self.working_dir, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.working_dir = dialog.get_working_dir()
            self.save_settings()
            self.status_label.setText(f"Working directory updated: {Path(self.working_dir).name}")

    def setup_about_tab(self):
        """Setup the About tab with app information"""
        layout = QVBoxLayout(self.about_tab)
        layout.setContentsMargins(0, 0, 0, 0)

        # Scroll area so it works on small screens
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        content = QWidget()
        inner = QVBoxLayout(content)
        inner.setSpacing(16)
        inner.setContentsMargins(40, 40, 40, 40)

        # Centre everything
        inner.addStretch()

        # App icon
        icon_path = Path(__file__).parent.parent / "assets" / "icon.png"
        if icon_path.exists():
            icon_label = QLabel()
            pixmap = QPixmap(str(icon_path)).scaled(
                96, 96, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            icon_label.setPixmap(pixmap)
            icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            icon_label.setStyleSheet("border: none;")
            inner.addWidget(icon_label)

        # App name
        name_label = QLabel("Deep Face Net")
        name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        name_label.setStyleSheet(
            "font-size: 28px; font-weight: bold; color: #ffffff;"
        )
        inner.addWidget(name_label)

        # Version badge
        version_label = QLabel("v1.0.0")
        version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        version_label.setStyleSheet(
            "font-size: 14px; color: #4CAF50; font-weight: bold;"
        )
        inner.addWidget(version_label)

        # Description
        desc_label = QLabel("Advanced Real-time Face Swapping Application")
        desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("font-size: 13px; color: #aaa;")
        inner.addWidget(desc_label)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFixedWidth(300)
        sep.setStyleSheet("background-color: #444; max-height: 1px;")
        inner.addWidget(sep, alignment=Qt.AlignmentFlag.AlignCenter)

        # Info card
        info_card = QFrame()
        info_card.setFixedWidth(360)
        info_card.setStyleSheet("""
            QFrame {
                background-color: #2a2a2a;
                border: 1px solid #444;
                border-radius: 10px;
                padding: 6px;
            }
        """)
        info_layout = QVBoxLayout(info_card)
        info_layout.setSpacing(8)
        info_layout.setContentsMargins(20, 16, 20, 16)

        for key_text, val_text in [
            ("Developed", "PolyNet Org"),
            ("Python", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"),
            ("Qt", qVersion()),
            ("PyQt6", PYQT_VERSION_STR),
            ("License", "AGPL-3.0"),
        ]:
            row = QHBoxLayout()
            key = QLabel(key_text)
            key.setStyleSheet("color: #888; font-size: 12px; border: none;")
            key.setFixedWidth(80)
            val = QLabel(val_text)
            val.setStyleSheet("color: #fff; font-size: 12px; font-weight: bold; border: none;")
            row.addWidget(key)
            row.addWidget(val)
            row.addStretch()
            info_layout.addLayout(row)

        inner.addWidget(info_card, alignment=Qt.AlignmentFlag.AlignCenter)

        # Buttons layout
        btn_layout = QVBoxLayout()
        btn_layout.setSpacing(12)
        btn_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Premium button
        premium_btn = QPushButton("Try Premium")
        premium_btn.setFixedWidth(260)
        premium_btn.setFixedHeight(40)
        premium_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        premium_btn.clicked.connect(
            lambda: QDesktopServices.openUrl(
                QUrl("https://www.polynet.online")
            )
        )
        premium_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
        """)
        btn_layout.addWidget(premium_btn)

        # GitHub button
        github_btn = QPushButton("  GitHub Repository")
        github_btn.setFixedWidth(260)
        github_btn.setFixedHeight(40)
        github_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        github_btn.clicked.connect(
            lambda: QDesktopServices.openUrl(
                QUrl("https://github.com/polynet-org/Deep-Face-Net")
            )
        )
        github_btn.setStyleSheet("""
            QPushButton {
                background-color: #333;
                color: #58a6ff;
                border: 1px solid #444;
                border-radius: 8px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #444;
                border-color: #58a6ff;
            }
        """)
        btn_layout.addWidget(github_btn)

        inner.addLayout(btn_layout)

        inner.addStretch()

        scroll.setWidget(content)
        layout.addWidget(scroll)

    def closeEvent(self, event):
        """Handle application close event"""
        if self.is_capturing:
            self.stop_camera()
        # Cancel active model downloads
        for thread in self._active_downloads.values():
            thread.cancel()
        event.accept()


def main():
    """Main application entry point"""
    # Enable High DPI scaling for Windows/multi-monitor setups
    try:
        from PyQt6.QtCore import Qt as QtCore_Qt
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            QtCore_Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
    except AttributeError:
        pass  # Older PyQt6 versions handle this automatically

    app = QApplication(sys.argv)

    # Set application-wide font with cross-platform fallbacks
    font = QFont()
    font.setFamilies(["Segoe UI", "SF Pro Display", "Ubuntu", "Noto Sans", "sans-serif"])
    font.setPointSize(10)
    app.setFont(font)

    window = DeepfakeApp()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
