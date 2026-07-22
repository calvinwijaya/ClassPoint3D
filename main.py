import sys
import subprocess
import numpy as np
import os
try:
    import laspy
except ImportError:
    laspy = None

from PyQt6.QtWidgets import (QApplication, QWidget, QPushButton, QVBoxLayout, 
                             QFileDialog, QTextEdit, QLabel, QHBoxLayout, QLineEdit, 
                             QSpinBox, QProgressBar, QMessageBox, QGroupBox, 
                             QSplitter, QTabWidget, QComboBox) # <-- Tambahan QComboBox
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QIcon, QPixmap, QCursor
import pyqtgraph.opengl as gl


class ProcessThread(QThread):
    output_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool)
    progress_signal = pyqtSignal(int)

    # Tambahkan parameter script_path
    def __init__(self, point_clouds, model, output_dir, batch_size, script_path):
        super().__init__()
        self.point_clouds = point_clouds 
        self.model = model
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.script_path = script_path # Menampung script yang dipilih (DGCNN atau XGBoost)

    def run(self):
        try:
            total_files = len(self.point_clouds)
            
            for file_idx, point_cloud_file in enumerate(self.point_clouds):
                filename = os.path.basename(point_cloud_file)
                self.output_signal.emit(f"\n========================================")
                self.output_signal.emit(f"Processing File [{file_idx + 1}/{total_files}]: {filename}")
                self.output_signal.emit(f"========================================\n")
                
                # Gunakan self.script_path yang dinamis
                command = [
                    'python', self.script_path,
                    '--batch_size', str(int(self.batch_size)),
                    '--model', self.model,
                    '--point_cloud', point_cloud_file,
                    '--output_dir', self.output_dir
                ]
                
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

                for line in process.stdout:
                    text = line.strip()
                    if text.startswith("PROGRESS:"):
                        try:
                            # Hitung progres file saat ini
                            numbers = text.split(":")[1].split("/")
                            current = int(numbers[0])
                            total = int(numbers[1])
                            file_fraction = current / total
                            
                            # Hitung progres global (keseluruhan batch)
                            global_percent = int(((file_idx + file_fraction) / total_files) * 100)
                            self.progress_signal.emit(global_percent)
                        except:
                            pass
                    else:
                        self.output_signal.emit(text)

                for line in process.stderr:
                    self.output_signal.emit(f"Error: {line.strip()}")

                process.stdout.close()
                process.stderr.close()
                return_code = process.wait()
                
                if return_code != 0:
                    self.output_signal.emit(f"Proses terhenti karena error pada file: {filename}")
                    self.finished_signal.emit(False)
                    return

            self.finished_signal.emit(True)
        
        except Exception as e:
            self.output_signal.emit(f"Error: {str(e)}")
            self.finished_signal.emit(False)

# --- WIDGET VIEWER 3D CUSTOM ---
class PointCloudViewer(QWidget):
    """Widget kustom untuk menampung View 3D utama, sumbu di pojok, dan tombol reset."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor('k')
        
        self.scatter = gl.GLScatterPlotItem()
        self.scatter.setGLOptions('opaque') 
        self.view.addItem(self.scatter)
        self.layout.addWidget(self.view)
        
        self.axis_view = gl.GLViewWidget(self.view)
        self.axis_view.setBackgroundColor((0, 0, 0, 0))
        
        self.axis_view.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.axis_view.setAttribute(Qt.WidgetAttribute.WA_AlwaysStackOnTop, True)
        
        self.axis_view.mousePressEvent = lambda ev: None
        self.axis_view.mouseMoveEvent = lambda ev: None
        self.axis_view.wheelEvent = lambda ev: None
        
        axis_size = 15
        x_axis = gl.GLLinePlotItem(pos=np.array([[0,0,0], [axis_size,0,0]]), color=(1,0.2,0.2,1), width=4, antialias=True) 
        y_axis = gl.GLLinePlotItem(pos=np.array([[0,0,0], [0,axis_size,0]]), color=(0.2,1,0.2,1), width=4, antialias=True) 
        z_axis = gl.GLLinePlotItem(pos=np.array([[0,0,0], [0,0,axis_size]]), color=(0.3,0.3,1,1), width=4, antialias=True) 
        
        self.axis_view.addItem(x_axis)
        self.axis_view.addItem(y_axis)
        self.axis_view.addItem(z_axis)
        
        self.original_mouseMove = self.view.mouseMoveEvent
        def custom_mouseMove(ev):
            self.original_mouseMove(ev)
            self.axis_view.setCameraPosition(
                elevation=self.view.opts['elevation'],
                azimuth=self.view.opts['azimuth']
            )
        self.view.mouseMoveEvent = custom_mouseMove
        
        self.reset_btn = QPushButton("Reset View", self.view)
        self.reset_btn.setObjectName("resetBtn")
        self.reset_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.reset_btn.clicked.connect(self.reset_camera)
        
        self.default_distance = 100

    def reset_camera(self):
        self.view.setCameraPosition(distance=self.default_distance, elevation=30, azimuth=45)
        self.axis_view.setCameraPosition(distance=40, elevation=30, azimuth=45)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self.axis_view.setGeometry(10, 10, 120, 120)
        btn_width, btn_height = 100, 30
        self.reset_btn.setGeometry(self.width() - btn_width - 20, 20, btn_width, btn_height)

    def load_data(self, points, colors):
        self.scatter.setData(pos=points, color=colors, size=3, pxMode=True)
        
        bbox_max = np.max(points, axis=0)
        bbox_min = np.min(points, axis=0)
        diagonal = np.linalg.norm(bbox_max - bbox_min)
        
        self.default_distance = float(diagonal) * 1.5
        if self.default_distance < 10: 
            self.default_distance = 100
            
        self.reset_camera()
        
    def clear_data(self):
        self.scatter.setData(pos=np.empty((0,3)))


# --- MAIN APPLICATION GUI ---
class PointCloudClassificationGUI(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Class Point 3D')
        self.setGeometry(100, 100, 1200, 700)
        self.setWindowIcon(QIcon("ui/logo.png")) 

        main_layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # ==================== LEFT PANEL ====================
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 15, 0)

        # Config Group
        config_group = QGroupBox("Project Configuration")
        config_layout = QVBoxLayout()

        # -------------------------------------------------------------
        # BARU: Dropdown Pemilihan Algoritma
        # -------------------------------------------------------------
        self.algo_label = QLabel('Classification Algorithm:')
        self.algo_combo = QComboBox(self)
        self.algo_combo.addItems(["Deep Learning (DGCNN)", "Machine Learning (XGBoost)"])
        config_layout.addWidget(self.algo_label)
        config_layout.addWidget(self.algo_combo)
        # -------------------------------------------------------------

        # Point Cloud Input
        self.pointcloud_label = QLabel('Point Cloud Input (.las/.laz file or Folder):')
        self.pointcloud_path = QLineEdit(self)
        self.pointcloud_path.setReadOnly(True)
        self.pointcloud_path.setPlaceholderText("Select single las or laz file OR folder...")
        
        self.pointcloud_file_btn = QPushButton('Select File')
        self.pointcloud_file_btn.clicked.connect(self.select_pointcloud_file)
        
        self.pointcloud_folder_btn = QPushButton('Select Folder')
        self.pointcloud_folder_btn.clicked.connect(self.select_pointcloud_folder)
        
        pc_layout = QHBoxLayout()
        pc_layout.addWidget(self.pointcloud_path)
        pc_layout.addWidget(self.pointcloud_file_btn)
        pc_layout.addWidget(self.pointcloud_folder_btn)  
        config_layout.addWidget(self.pointcloud_label)
        config_layout.addLayout(pc_layout)
        
        # Model Input
        self.model_label = QLabel('Trained Model File (.t7 / .pkl):')
        self.model_path = QLineEdit(self)
        self.model_path.setReadOnly(True)
        self.model_path.setPlaceholderText("Select model file...")
        self.model_btn = QPushButton('Browse...')
        self.model_btn.clicked.connect(self.select_model)
        model_layout = QHBoxLayout()
        model_layout.addWidget(self.model_path)
        model_layout.addWidget(self.model_btn)
        config_layout.addWidget(self.model_label)
        config_layout.addLayout(model_layout)

        # Output Dir
        self.output_label = QLabel('Output Directory:')
        self.output_path = QLineEdit(self)
        self.output_path.setReadOnly(True)
        self.output_path.setPlaceholderText("Select output folder...")
        self.output_btn = QPushButton('Browse...')
        self.output_btn.clicked.connect(self.select_output_file)
        out_layout = QHBoxLayout()
        out_layout.addWidget(self.output_path)
        out_layout.addWidget(self.output_btn)
        config_layout.addWidget(self.output_label)
        config_layout.addLayout(out_layout)

        config_group.setLayout(config_layout)
        left_layout.addWidget(config_group)

        # Advanced Options Group
        adv_group = QGroupBox("Advanced Options")
        adv_layout = QVBoxLayout()
        self.advanced_options_btn = QPushButton('Show Advanced Options ▼')
        self.advanced_options_btn.setObjectName("advBtn")
        self.advanced_options_btn.setCheckable(True)
        self.advanced_options_btn.clicked.connect(self.toggle_advanced_options)
        
        self.batch_size_label = QLabel('Batch Size (For Deep Learning):')
        self.batch_size = QSpinBox(self)
        self.batch_size.setRange(1, 1024)
        self.batch_size.setValue(16)
        self.batch_size.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.adv_content = QWidget()
        adv_content_layout = QHBoxLayout(self.adv_content)
        adv_content_layout.setContentsMargins(0, 10, 0, 0)
        adv_content_layout.addWidget(self.batch_size_label)
        adv_content_layout.addWidget(self.batch_size)
        adv_content_layout.addStretch()
        self.adv_content.setVisible(False)

        adv_layout.addWidget(self.advanced_options_btn)
        adv_layout.addWidget(self.adv_content)
        adv_group.setLayout(adv_layout)
        left_layout.addWidget(adv_group)

        # Execution Group
        exec_layout = QHBoxLayout()
        self.start_btn = QPushButton('Start Classification', self)
        self.start_btn.setObjectName("startBtn")
        self.start_btn.setMinimumHeight(45)
        self.start_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.start_btn.clicked.connect(self.start_process)

        self.replay_btn = QPushButton(' Clear')
        self.replay_btn.setObjectName("clearBtn")
        self.replay_btn.setIcon(QIcon("ui/replay.png")) 
        self.replay_btn.setMinimumHeight(45)
        self.replay_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.replay_btn.clicked.connect(self.clear_inputs)

        exec_layout.addWidget(self.start_btn, stretch=3)
        exec_layout.addWidget(self.replay_btn, stretch=1)
        left_layout.addLayout(exec_layout)

        # Status Group
        status_group = QGroupBox("Processing Status")
        status_layout = QVBoxLayout()
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Process Classification... %p%")
        self.log_console = QTextEdit(self)
        self.log_console.setReadOnly(True)
        self.log_console.setStyleSheet("background-color: #f8f9fa; border: 1px solid #ced4da; font-family: Consolas;")
        status_layout.addWidget(self.progress_bar)
        status_layout.addWidget(self.log_console)
        status_group.setLayout(status_layout)
        left_layout.addWidget(status_group)

        # Watermark
        watermark_layout = QHBoxLayout()
        watermark_logo = QLabel(self)
        watermark_logo.setPixmap(QPixmap("ui/ugm.png").scaled(40, 40, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        watermark_layout.addWidget(watermark_logo)
        watermark_text = QLabel("Department of Geodetic Engineering\nFaculty of Engineering Universitas Gadjah Mada")
        watermark_text.setStyleSheet("color: gray; font-size: 10px;")
        watermark_layout.addWidget(watermark_text)
        watermark_layout.addStretch(1)
        left_layout.addLayout(watermark_layout)

        # ==================== RIGHT PANEL ====================
        right_panel = QTabWidget()
        
        self.viewer_rgb = PointCloudViewer()
        self.viewer_cls = PointCloudViewer()

        right_panel.addTab(self.viewer_rgb, "Input Point Cloud (RGB)")
        right_panel.addTab(self.viewer_cls, "Classification Result")

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 800])
        main_layout.addWidget(splitter)

    def update_progress_bar(self, value):
        if self.progress_bar.maximum() == 0:
            self.progress_bar.setMaximum(100)
            
        self.progress_bar.setValue(value)

    def toggle_advanced_options(self):
        is_checked = self.advanced_options_btn.isChecked()
        self.adv_content.setVisible(is_checked)
        self.advanced_options_btn.setText('Show Advanced Options ▲' if is_checked else 'Show Advanced Options ▼')

    def select_model(self):
        # PERUBAHAN: Memperbolehkan ekstensi .t7 (DGCNN) atau .pkl (XGBoost)
        model, _ = QFileDialog.getOpenFileName(self, "Select Model", "", "Model Files (*.t7 *.pkl)")
        if model:
            self.model_path.setText(model)
            self.log_console.append(f"Selected Model: {model}")

    def select_pointcloud_file(self):
        pointcloud, _ = QFileDialog.getOpenFileName(self, "Select Point Cloud (*.las)", "", "LAS Files (*.las *.laz)")
        if pointcloud:
            self.pointcloud_path.setText(pointcloud)
            self.log_console.append(f"Selected Point Cloud File: {pointcloud}")
            self.process_and_render_las(pointcloud, self.viewer_rgb, is_classification=False)

    def select_pointcloud_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder Containing LAS/LAZ Files")
        if folder:
            self.pointcloud_path.setText(folder)
            self.log_console.append(f"Selected Folder for Batch Processing: {folder}")
            
            las_files = [f for f in os.listdir(folder) if f.lower().endswith(('.las', '.laz'))]
            if las_files:
                first_file = os.path.join(folder, las_files[0])
                self.log_console.append(f"Found {len(las_files)} point cloud files. Previewing first file...")
                self.process_and_render_las(first_file, self.viewer_rgb, is_classification=False)
            else:
                self.log_console.append("Warning: No .las or .laz file found in the folder.")

    def select_output_file(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.output_path.setText(directory)
            self.log_console.append(f"Selected Output Directory: {directory}")

    def process_and_render_las(self, filepath, viewer, is_classification=False):
        if laspy is None:
            self.log_console.append("Warning: Modul 'laspy' tidak ditemukan. Preview 3D dilewati.")
            return

        try:
            las = laspy.read(filepath)
            points = np.vstack((las.x, las.y, las.z)).transpose()
            
            has_rgb = hasattr(las, 'red') and np.max(las.red) > 0
            has_cls = hasattr(las, 'classification')

            max_points = 100000
            if len(points) > max_points:
                indices = np.random.choice(len(points), max_points, replace=False)
                points = points[indices]
                if is_classification and has_cls:
                    class_vals = np.array(las.classification)[indices]
                elif has_rgb and not is_classification:
                    colors_raw = np.vstack((las.red[indices], las.green[indices], las.blue[indices])).transpose()
            else:
                if is_classification and has_cls:
                    class_vals = np.array(las.classification)
                elif has_rgb and not is_classification:
                    colors_raw = np.vstack((las.red, las.green, las.blue)).transpose()

            colors_matrix = None

            if is_classification and has_cls:
                unique_classes = np.unique(class_vals)
                colors_matrix = np.zeros((len(class_vals), 4))
                
                cmap = {
                    2: (0.6, 0.4, 0.2, 1.0), # Ground
                    3: (0.0, 0.9, 0.0, 1.0), # Low Veg
                    4: (0.0, 0.7, 0.0, 1.0), # Med Veg
                    5: (0.0, 0.4, 0.0, 1.0), # High Veg
                    6: (0.9, 0.1, 0.1, 1.0), # Building
                    9: (0.0, 0.0, 0.9, 1.0), # Water
                }
                for cls in unique_classes:
                    if cls in cmap:
                        colors_matrix[class_vals == cls] = cmap[cls]
                    else:
                        np.random.seed(cls)
                        colors_matrix[class_vals == cls] = (np.random.rand(), np.random.rand(), np.random.rand(), 1.0)

            elif has_rgb and not is_classification:
                if colors_raw.max() > 255: colors_raw = colors_raw / 65535.0
                else: colors_raw = colors_raw / 255.0
                alpha = np.ones((colors_raw.shape[0], 1))
                colors_matrix = np.hstack((colors_raw, alpha))

            if colors_matrix is None:
                colors_matrix = np.ones((len(points), 4)) * np.array([0.4, 0.4, 0.4, 1.0])

            centroid = np.mean(points, axis=0)
            points -= centroid

            viewer.load_data(points, colors_matrix)
            self.log_console.append(f"Successfully rendering 3D preview: {os.path.basename(filepath)}")

        except Exception as e:
            self.log_console.append(f"Fail to load 3D preview: {str(e)}")

    def start_process(self):
        model = self.model_path.text()
        input_path = self.pointcloud_path.text()
        output_dir = self.output_path.text()

        if not model or not input_path or not output_dir:
            self.log_console.append("Error: Please fill all required fields")
            return
            
        target_files = []
        if os.path.isfile(input_path):
            target_files.append(input_path)
        elif os.path.isdir(input_path):
            target_files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.lower().endswith(('.las', '.laz'))]
            if not target_files:
                self.log_console.append("Error: Folder kosong, tidak ada file .las/.laz untuk diproses.")
                return

        # -------------------------------------------------------------
        # BARU: Tentukan script mana yang berjalan berdasarkan Dropdown
        # -------------------------------------------------------------
        selected_algo = self.algo_combo.currentText()
        if "Deep Learning" in selected_algo:
            script_to_run = "scripts/predict_rgb.py"
        else:
            script_to_run = "scripts/classify_xgb.py"

        self.start_btn.setEnabled(False)
        self.progress_bar.setMaximum(0)  
        self.log_console.append(f"== Batch Classification Process Start! ({len(target_files)} Files) ==")
        self.log_console.append(f"Using Algorithm: {selected_algo}")

        batch_size = self.batch_size.value()
        
        # Kirim list of files (target_files) dan script_to_run ke dalam Thread
        self.process_thread = ProcessThread(target_files, model, output_dir, batch_size, script_to_run)
        self.process_thread.output_signal.connect(self.update_console_log)
        self.process_thread.finished_signal.connect(self.process_finished)
        self.process_thread.progress_signal.connect(self.update_progress_bar)
        self.process_thread.start()

    def process_finished(self, success):
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(100 if success else 0)
        self.start_btn.setEnabled(True)

        if success:
            QMessageBox.information(self, "Process Complete", "Point cloud data has been classified successfully.")
            
            input_path = self.pointcloud_path.text()
            output_dir = self.output_path.text()
            
            first_file_to_render = None
            
            if os.path.isfile(input_path):
                first_file_to_render = input_path
            elif os.path.isdir(input_path):
                las_files = [f for f in os.listdir(input_path) if f.lower().endswith(('.las', '.laz'))]
                if las_files:
                    first_file_to_render = os.path.join(input_path, las_files[0])
            
            if first_file_to_render:
                filename = os.path.basename(first_file_to_render)
                name, ext = os.path.splitext(filename)
                classified_filename = f"{name}_classified{ext}"
                possible_output = os.path.join(output_dir, classified_filename)
                
                if os.path.exists(possible_output):
                    self.log_console.append(f"Rendering first classification result: {classified_filename}")
                    self.process_and_render_las(possible_output, self.viewer_cls, is_classification=True)
                else:
                    self.log_console.append(f"Warning: classification file result {classified_filename} is not found to render.")
            else:
                self.log_console.append("Warning: No valid file to rendered in classification tab.")
                
        else:
            QMessageBox.warning(self, "Process Failed", "Point cloud data failed to classify.")
    
    def update_console_log(self, message):
        self.log_console.append(message)
        scrollbar = self.log_console.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def clear_inputs(self):
        self.model_path.clear()
        self.pointcloud_path.clear()
        self.output_path.clear()
        self.log_console.clear()
        self.progress_bar.setValue(0)
        
        self.viewer_rgb.clear_data()
        self.viewer_cls.clear_data()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    app.setStyleSheet("""
        QWidget {
            font-family: 'Segoe UI', Arial, sans-serif;
            font-size: 13px;
        }
        QGroupBox {
            font-weight: bold;
            border: 1px solid #ced4da;
            border-radius: 6px;
            margin-top: 15px;
            padding-top: 15px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 10px;
            padding: 0 5px;
            color: #495057;
        }
        QLineEdit, QComboBox { /* <-- Tambahan QComboBox di dalam rule border */
            background-color: #ffffff;
            padding: 6px;
            border: 1px solid #ced4da;
            border-radius: 4px;
            color: #495057;
        }
        QSpinBox {
            padding: 5px;
            border: 1px solid #ced4da;
            border-radius: 4px;
            background-color: #ffffff;
            width: 70px;
        }
        QPushButton {
            background-color: #f8f9fa;
            border: 1px solid #ced4da;
            border-radius: 4px;
            padding: 6px 12px;
            color: #495057;
            font-weight: 500;
        }
        QPushButton:hover {
            background-color: #e2e6ea;
            border: 1px solid #dae0e5;
        }
        QPushButton:pressed {
            background-color: #dae0e5;
        }
        QPushButton#startBtn {
            background-color: #28a745;
            color: white;
            border: 1px solid #28a745;
            font-weight: bold;
            font-size: 14px;
        }
        QPushButton#startBtn:hover {
            background-color: #218838;
            border: 1px solid #1e7e34;
        }
        QPushButton#startBtn:disabled {
            background-color: #94d3a2;
            border: 1px solid #94d3a2;
        }
        QPushButton#clearBtn {
            background-color: #ffffff;
            color: #dc3545;
            border: 1px solid #dc3545;
            font-weight: bold;
        }
        QPushButton#clearBtn:hover {
            background-color: #dc3545;
            color: white;
        }
        QPushButton#advBtn {
            background-color: transparent;
            border: none;
            color: #007bff;
            text-align: left;
            padding: 0px;
        }
        QPushButton#advBtn:hover {
            color: #0056b3;
            text-decoration: underline;
        }
        QPushButton#resetBtn {
            background-color: rgba(255, 255, 255, 220);
            border: 1px solid #adb5bd;
            border-radius: 12px;
            color: #343a40;
            font-size: 11px;
            font-weight: bold;
        }
        QPushButton#resetBtn:hover {
            background-color: rgba(230, 230, 255, 255);
        }
        QProgressBar {
            border: 1px solid #ced4da;
            border-radius: 4px;
            text-align: center;     
            height: 22px;            
            color: #212529;          
            font-weight: bold;       
        }
        QProgressBar::chunk {
            background-color: #007bff;
            border-radius: 3px;
        }
    """)

    ex = PointCloudClassificationGUI()
    ex.show()
    sys.exit(app.exec())