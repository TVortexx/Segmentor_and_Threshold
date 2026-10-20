import cv2
from PySide6.QtCore import Qt, QPointF, Signal
from PySide6.QtGui import QPainter, QPen, QColor, QPixmap, QImage
import sys
import numpy as np
from PySide6.QtWidgets import (QMainWindow, QApplication, QWidget,
                               QVBoxLayout, QHBoxLayout, QLabel,
                               QPushButton, QComboBox, QFileDialog)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt


def kmeans_clustering(image, seeds, k):
    """
    Perform K-means clustering segmentation

    Args:
        image: NumPy array of image data
        seeds: List of (x, y, cluster_id) tuples
        k: Number of clusters

    Returns:
        NumPy array of cluster labels for each pixel
    """
    height, width = image.shape[:2]

    # Extract image features (you can use more features)
    if len(image.shape) == 3:  # RGB
        features = image.reshape((-1, 3))
    else:  # Grayscale
        features = image.reshape((-1, 1))

    # Initialize centroids from seeds
    centroids = np.zeros((k, features.shape[1]))
    for x, y, cluster_id in seeds:
        if 1 <= cluster_id <= k:
            idx = y * width + x
            centroids[cluster_id - 1] = features[idx]

    # Check if we have at least one seed per cluster
    initialized_clusters = [False] * k
    for _, _, cluster_id in seeds:
        if 1 <= cluster_id <= k:
            initialized_clusters[cluster_id - 1] = True

    # For clusters without seeds, initialize randomly
    for i in range(k):
        if not initialized_clusters[i]:
            random_idx = np.random.randint(0, features.shape[0])
            centroids[i] = features[random_idx]

    # Run K-means algorithm
    max_iter = 100
    labels = np.zeros(features.shape[0], dtype=np.int32)

    for _ in range(max_iter):
        # Assign pixels to clusters
        for i in range(features.shape[0]):
            distances = np.linalg.norm(features[i] - centroids, axis=1)
            labels[i] = np.argmin(distances)

        # Update centroids
        new_centroids = np.zeros_like(centroids)
        for i in range(k):
            if np.sum(labels == i) > 0:
                new_centroids[i] = np.mean(features[labels == i], axis=0)
            else:
                new_centroids[i] = centroids[i]  # Keep old centroid if cluster is empty

        # Check convergence
        if np.allclose(centroids, new_centroids, rtol=1e-5):
            break

        centroids = new_centroids

    # Reshape labels back to image dimensions
    return labels.reshape(height, width)


def region_growing(image, seeds, threshold=10):
    """
    Perform region growing segmentation

    Args:
        image: NumPy array of image data
        seeds: List of (x, y) tuples as starting points
        threshold: Intensity threshold for region inclusion

    Returns:
        NumPy array of binary segmentation mask
    """
    if len(image.shape) == 3:
        # Convert to grayscale for simplicity
        gray_image = np.mean(image, axis=2).astype(np.uint8)
    else:
        gray_image = image.copy()

    height, width = gray_image.shape
    # create segmentation to draw over the image for all seeds
    overall_segmentation = np.zeros((height, width), dtype=np.uint8)
    # initialize segment for each seed
    seed_segment = np.zeros((height, width), dtype=np.uint8)

    # Process each seed point separately (start enumerate from 1 to multiply by)
    for seed_idx, (seed_x, seed_y) in enumerate(seeds, 1):
        # Skip if out of bounds
        if 0 <= seed_x < width and 0 <= seed_y < height:
            # to address if it is visited or not
            visited = np.zeros((height, width), dtype=bool)
            # reset segment for each seed
            seed_segment = np.zeros((height, width), dtype=np.uint8)

            # Get seed pixel value
            seed_value = gray_image[seed_y, seed_x]

            # Initialize queue with seed ( the only (x,y) and the rest are (y,x)
            processing_queue = [(seed_x, seed_y)]
            visited[seed_y, seed_x] = True
            seed_segment[seed_y, seed_x] = 1

            # # Define 4-connected neighbors
            # neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

            # test 8 neighbors
            neighbors = [
                (-1, -1), (-1, 0), (-1, 1),
                (0, -1),         (0, 1),
                (1, -1), (1, 0), (1, 1)
            ]

            # Process queue
            while processing_queue:
                x, y = processing_queue.pop(0)

                for dx, dy in neighbors:
                    nx, ny = x + dx, y + dy

                    # Check bounds
                    if 0 <= nx < width and 0 <= ny < height:
                        # Skip if already visited
                        if not visited[ny, nx]:
                            visited[ny, nx] = True
                            # Check if pixel is within threshold
                            pixel_value = gray_image[ny, nx]
                            if abs(int(pixel_value) - int(seed_value)) <= threshold:
                                seed_segment[ny, nx] = 1
                                processing_queue.append((nx, ny))

        # Add this segment to the final segmentation
        overall_segmentation = np.maximum(overall_segmentation, seed_segment * seed_idx)

    return overall_segmentation

class SeedPixelWidget(QWidget):
    # Signal to notify when seeds have changed
    seedsChanged = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(500, 500)
        self.setMouseTracking(True)

        # Store image and transformation properties
        self.image = None
        self.display_image = None
        self.scale_factor = 1.0
        self.min_scale = 0.1
        self.max_scale = 5.0

        # Store seeds as (x, y, cluster_id) tuples
        self.seeds = []
        self.active_cluster = 1
        self.num_clusters = 3  # Default K value
        self.colors = [QColor(255, 0, 0), QColor(0, 255, 0), QColor(0, 0, 255),
                       QColor(255, 255, 0), QColor(255, 0, 255), QColor(0, 255, 255)]

        # State variables
        self.hovering_point = None
        self.dragging_seed = None
        self.current_algorithm = "kmeans"  # or "region_growing"

        # Set up the UI
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Top controls
        controls = QHBoxLayout()

        # Algorithm selection
        self.algo_combo = QComboBox()
        self.algo_combo.addItems(["K-means Clustering", "Region Growing"])
        self.algo_combo.currentIndexChanged.connect(self.algorithm_changed)

        # Number of clusters (for K-means)
        self.cluster_label = QLabel("Number of clusters (K):")
        self.cluster_combo = QComboBox()
        self.cluster_combo.addItems([str(i) for i in range(2, 11)])
        self.cluster_combo.setCurrentIndex(1)  # Default to 3 clusters
        self.cluster_combo.currentIndexChanged.connect(self.update_num_clusters)

        # Active cluster selector
        self.active_cluster_label = QLabel("Active cluster:")
        self.active_cluster_combo = QComboBox()
        self.update_active_cluster_combo()
        self.active_cluster_combo.currentIndexChanged.connect(self.update_active_cluster)

        # Clear buttons
        self.clear_button = QPushButton("Clear Seeds")
        self.clear_button.clicked.connect(self.clear_seeds)

        # Run segmentation button
        self.run_button = QPushButton("Run Segmentation")
        self.run_button.clicked.connect(self.run_segmentation)

        # Add widgets to controls
        controls.addWidget(self.algo_combo)
        controls.addWidget(self.cluster_label)
        controls.addWidget(self.cluster_combo)
        controls.addWidget(self.active_cluster_label)
        controls.addWidget(self.active_cluster_combo)
        controls.addWidget(self.clear_button)
        controls.addWidget(self.run_button)

        # Status bar for coordinates
        self.status_label = QLabel("Position: ")

        # Add layouts to main layout
        layout.addLayout(controls)
        layout.addStretch()
        layout.addWidget(self.status_label)

    def set_image(self, image):
        """Set the image for segmentation (NumPy array or file path)"""
        if isinstance(image, str):
            # Load image from file path
            self.image = QImage(image)
            if self.image.isNull():
                raise ValueError(f"Failed to load image from {image}")
        elif isinstance(image, np.ndarray):
            # Convert numpy array to QImage
            height, width = image.shape[:2]
            if len(image.shape) == 2:  # Grayscale
                qimg = QImage(image.data, width, height, width, QImage.Format_Grayscale8)
            else:  # RGB/RGBA
                bytes_per_line = 3 * width
                qimg = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.image = qimg
        else:
            raise TypeError("Image must be a file path or NumPy array")

        # Create a display copy
        self.display_image = QPixmap.fromImage(self.image)
        self.update()

    def algorithm_changed(self, index):
        """Handle algorithm selection change"""
        if index == 0:
            self.current_algorithm = "kmeans"
            self.cluster_label.setVisible(True)
            self.cluster_combo.setVisible(True)
            self.active_cluster_label.setVisible(True)
            self.active_cluster_combo.setVisible(True)
        else:
            self.current_algorithm = "region_growing"
            self.cluster_label.setVisible(False)
            self.cluster_combo.setVisible(False)
            # For region growing, we still allow multiple seeds
            # but we don't need to assign them to clusters
            self.active_cluster_label.setVisible(False)
            self.active_cluster_combo.setVisible(False)

        # Clear seeds when changing algorithms
        self.clear_seeds()

    def update_num_clusters(self, index):
        """Update number of clusters"""
        self.num_clusters = index + 2  # Index 0 = 2 clusters
        # Update active cluster combo box
        self.update_active_cluster_combo()

        # Filter out seeds with cluster IDs greater than num_clusters
        self.seeds = [seed for seed in self.seeds if seed[2] <= self.num_clusters]
        self.seedsChanged.emit(self.seeds)
        self.update()

    def update_active_cluster_combo(self):
        """Update the items in the active cluster combo box"""
        current = self.active_cluster_combo.currentIndex()
        self.active_cluster_combo.clear()
        for i in range(1, self.num_clusters + 1):
            self.active_cluster_combo.addItem(f"Cluster {i}")

        # Restore selection or set to first item
        if current >= 0 and current < self.num_clusters:
            self.active_cluster_combo.setCurrentIndex(current)
        else:
            self.active_cluster_combo.setCurrentIndex(0)

    def update_active_cluster(self, index):
        """Update the active cluster ID"""
        self.active_cluster = index + 1  # 1-based cluster IDs

    def clear_seeds(self):
        """Remove all seed points"""
        self.seeds = []
        self.seedsChanged.emit(self.seeds)
        self.update()

    def paintEvent(self, event):
        if not self.display_image:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw the image
        scaled_width = int(self.display_image.width() * self.scale_factor)
        scaled_height = int(self.display_image.height() * self.scale_factor)

        # Center the image in the widget
        x = (self.width() - scaled_width) // 2
        y = (self.height() - scaled_height) // 2

        painter.drawPixmap(x, y, scaled_width, scaled_height, self.display_image)

        # Draw seed points
        self.draw_seeds(painter, x, y)

        # Draw hover indication
        if self.hovering_point:
            painter.setPen(QPen(Qt.white, 2, Qt.DashLine))
            painter.drawEllipse(self.hovering_point, 10, 10)

    def draw_seeds(self, painter, offset_x, offset_y):
        """Draw all seed points with their cluster colors"""
        for seed_x, seed_y, cluster_id in self.seeds:
            # Convert image coordinates to widget coordinates
            x = offset_x + seed_x * self.scale_factor
            y = offset_y + seed_y * self.scale_factor

            # Select color based on cluster ID (modulo number of colors)
            color_index = (cluster_id - 1) % len(self.colors)
            color = self.colors[color_index]

            # Draw seed marker
            painter.setPen(QPen(Qt.black, 2))
            painter.setBrush(color)
            painter.drawEllipse(QPointF(x, y), 8, 8)

            # Draw label for K-means clusters
            if self.current_algorithm == "kmeans":
                painter.setPen(QPen(Qt.white, 1))
                painter.drawText(int(x - 4), int(y + 4), str(cluster_id))

    def mousePressEvent(self, event):
        if not self.display_image:
            return

        # Check if clicking on an existing seed
        for i, (seed_x, seed_y, cluster_id) in enumerate(self.seeds):
            # Convert to widget coordinates
            x = (self.width() - self.display_image.width() * self.scale_factor) // 2 + seed_x * self.scale_factor
            y = (self.height() - self.display_image.height() * self.scale_factor) // 2 + seed_y * self.scale_factor

            # Check if click is within seed marker
            if (QPointF(x, y) - event.position()).manhattanLength() < 10:
                if event.button() == Qt.LeftButton:
                    self.dragging_seed = i
                elif event.button() == Qt.RightButton:
                    # Remove seed on right click
                    self.seeds.pop(i)
                    self.seedsChanged.emit(self.seeds)
                    self.update()
                return

        # If not clicking on existing seed, add new seed
        if event.button() == Qt.LeftButton:
            # Convert widget coordinates to image coordinates
            offset_x = (self.width() - self.display_image.width() * self.scale_factor) // 2
            offset_y = (self.height() - self.display_image.height() * self.scale_factor) // 2

            image_x = (event.position().x() - offset_x) / self.scale_factor
            image_y = (event.position().y() - offset_y) / self.scale_factor

            # Check if coordinates are within image bounds
            if (0 <= image_x < self.display_image.width() and
                    0 <= image_y < self.display_image.height()):

                # Add new seed with active cluster
                if self.current_algorithm == "kmeans":
                    cluster_id = self.active_cluster
                else:
                    # For region growing, use cluster_id=1 for all seeds
                    cluster_id = 1

                self.seeds.append((int(image_x), int(image_y), cluster_id))
                self.seedsChanged.emit(self.seeds)
                self.update()

    def mouseMoveEvent(self, event):
        if not self.display_image:
            return

        # Update status bar with position
        offset_x = (self.width() - self.display_image.width() * self.scale_factor) // 2
        offset_y = (self.height() - self.display_image.height() * self.scale_factor) // 2

        image_x = (event.position().x() - offset_x) / self.scale_factor
        image_y = (event.position().y() - offset_y) / self.scale_factor

        if (0 <= image_x < self.display_image.width() and
                0 <= image_y < self.display_image.height()):
            # Get pixel value at current position (if needed)
            self.status_label.setText(f"Position: ({int(image_x)}, {int(image_y)})")
            self.hovering_point = event.position()
        else:
            self.status_label.setText("Position: Outside image")
            self.hovering_point = None

        # Update dragging seed if any
        if self.dragging_seed is not None:
            if (0 <= image_x < self.display_image.width() and
                    0 <= image_y < self.display_image.height()):
                seed = self.seeds[self.dragging_seed]
                self.seeds[self.dragging_seed] = (int(image_x), int(image_y), seed[2])
                self.seedsChanged.emit(self.seeds)

        self.update()

    def mouseReleaseEvent(self, event):
        self.dragging_seed = None

    def wheelEvent(self, event):
        """Handle mouse wheel for zooming"""
        zoom_factor = 1.1

        # Zoom in/out based on wheel direction
        if event.angleDelta().y() > 0:
            self.scale_factor = min(self.scale_factor * zoom_factor, self.max_scale)
        else:
            self.scale_factor = max(self.scale_factor / zoom_factor, self.min_scale)

        self.update()

    def get_seed_pixels(self):
        """Return seed pixels in a format suitable for algorithms"""
        if self.current_algorithm == "kmeans":
            # For k-means: list of (x, y, cluster_id) tuples
            return self.seeds
        else:
            # For region growing: just list of (x, y) points
            return [(x, y) for x, y, _ in self.seeds]

    def run_segmentation(self):
        """Run the selected segmentation algorithm"""
        if not self.seeds:
            return

        if self.current_algorithm == "kmeans":
            self.run_kmeans()
        else:
            self.run_region_growing()

    def run_kmeans(self):
        """Run K-means clustering with the selected seeds"""
        # Implement your K-means algorithm here
        # You would call something like:
        # result = kmeans_clustering(np.array(self.image), self.seeds, self.num_clusters)
        print(f"Running K-means with {len(self.seeds)} seeds and k={self.num_clusters}")

    def run_region_growing(self):
        """Run Region Growing with the selected seeds"""
        # Implement your Region Growing algorithm here
        # You would call something like:
        # result = region_growing(np.array(self.image), [(x, y) for x, y, _ in self.seeds])
        print(f"Running Region Growing with {len(self.seeds)} seeds")


class SegmentationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Segmentation Tool")
        self.resize(1000, 700)

        self.image = None
        self.segmentation_result = None

        self.setup_ui()

    def setup_ui(self):
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left panel - seed pixel widget
        self.seed_widget = SeedPixelWidget()
        self.seed_widget.seedsChanged.connect(self.handle_seeds_changed)
        self.seed_widget.run_button.clicked.connect(self.run_segmentation)

        # Right panel - result display and controls
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Result image
        self.result_label = QLabel("Segmentation Result")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setMinimumSize(400, 400)
        self.result_label.setStyleSheet("border: 1px solid #999")

        # Controls
        controls_layout = QHBoxLayout()

        # Load image button
        self.load_button = QPushButton("Load Image")
        self.load_button.clicked.connect(self.load_image)

        # Threshold slider (for region growing)
        threshold_layout = QVBoxLayout()
        threshold_layout.addWidget(QLabel("Region Growing Threshold:"))
        self.threshold_combo = QComboBox()
        self.threshold_combo.addItems([str(i) for i in range(5, 51, 5)])
        self.threshold_combo.setCurrentIndex(1)  # Default to 10
        threshold_layout.addWidget(self.threshold_combo)

        # Save result button
        self.save_button = QPushButton("Save Result")
        self.save_button.clicked.connect(self.save_result)
        self.save_button.setEnabled(False)

        # Add widgets to controls
        controls_layout.addWidget(self.load_button)
        controls_layout.addLayout(threshold_layout)
        controls_layout.addWidget(self.save_button)

        # Add to right panel
        right_layout.addWidget(self.result_label)
        right_layout.addLayout(controls_layout)

        # Add panels to main layout
        main_layout.addWidget(self.seed_widget, 3)
        main_layout.addWidget(right_panel, 2)

    def load_image(self):
        """Load an image from file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )

        if file_path:
            # Load the image
            self.image = QImage(file_path)
            if not self.image.isNull():
                # Set image to seed widget
                self.seed_widget.set_image(file_path)

                # Clear result
                self.segmentation_result = None
                self.save_button.setEnabled(False)
                self.result_label.clear()
                self.result_label.setText("Segmentation Result")

    def handle_seeds_changed(self, seeds):
        """Handle changes in seed points"""
        if self.segmentation_result:
            # Clear result when seeds change
            self.segmentation_result = None
            self.save_button.setEnabled(False)
            self.result_label.setText("Segmentation Result")

    def run_segmentation(self):
        """Run the selected segmentation algorithm"""
        if not self.image or self.image.isNull():
            return

        # Get algorithm and seeds
        algorithm = self.seed_widget.current_algorithm
        seeds = self.seed_widget.get_seed_pixels()

        if not seeds:
            return

        # Convert QImage to NumPy array
        width = self.image.width()
        height = self.image.height()
        if self.image.format() == QImage.Format_RGB888:
            ptr = self.image.constBits()
            image_array = np.array(ptr).reshape(height, width, 3)
        else:
            # Convert to RGB format if needed
            rgb_image = self.image.convertToFormat(QImage.Format_RGB888)
            ptr = rgb_image.constBits()
            image_array = np.array(ptr).reshape(height, width, 3)

        # Run the appropriate algorithm
        if algorithm == "kmeans":
            k = self.seed_widget.num_clusters
            result = kmeans_clustering(image_array, seeds, k)
        else:
            # Get threshold for region growing
            threshold = int(self.threshold_combo.currentText())
            result = region_growing(image_array, seeds, threshold)

        # Store and display the result
        self.segmentation_result = result
        self.display_segmentation_result()
        self.save_button.setEnabled(True)

    def display_segmentation_result(self):
        """Display the segmentation result on top of the input image"""
        if self.segmentation_result is None or self.image is None:
            return

        # Get dimensions from segmentation result
        height, width = self.segmentation_result.shape

        # Convert QImage image to numpy array
        image_array = self.qimage_to_numpy(self.image)

        # Resize if necessary to match segmentation dimensions
        if image_array.shape[:2] != (height, width):
            image_array = cv2.resize(image_array, (width, height))

        # Create result image using input image as background
        result = image_array.copy()

        # Color each segment's boundary based on its label
        unique_labels = np.unique(self.segmentation_result)

        for label in unique_labels:
            if label == 0:  # Skip background
                continue

            # Get color for this label
            color_idx = (label - 1) % len(self.seed_widget.colors)
            color = self.seed_widget.colors[color_idx]
            rgb_color = (color.red(), color.green(), color.blue())

            # Get mask for this label
            label_mask = (self.segmentation_result == label).astype(np.uint8)

            # Find contours of the segment
            contours, _ = cv2.findContours(label_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Draw contours on the result image
            cv2.drawContours(result, contours, -1, rgb_color, 2)

        # Convert numpy array back to QImage
        qimg = QImage(result.data, width, height, width * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        # Scale to fit in result label
        label_size = self.result_label.size()
        pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        self.result_label.setPixmap(pixmap)

    def qimage_to_numpy(self, qimage):
        """Convert QImage to numpy array"""
        width = qimage.width()
        height = qimage.height()

        # Convert QImage to a format we can work with
        if qimage.format() != QImage.Format_RGB888:
            qimage = qimage.convertToFormat(QImage.Format_RGB888)

        # Use numpy to create an array from the underlying bytes
        bits = qimage.constBits()
        # Create a numpy array using the buffer protocol
        arr = np.asarray(bits).reshape(height, width, 3)
        return arr.copy()

    def save_result(self):
        """Save the segmentation result to a file"""
        if self.segmentation_result is None:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Segmentation Result", "", "PNG Files (*.png)"
        )

        if file_path:
            # Create colored visualization
            height, width = self.segmentation_result.shape
            visualization = np.zeros((height, width, 3), dtype=np.uint8)

            # Color each segment
            unique_labels = np.unique(self.segmentation_result)
            for i, label in enumerate(unique_labels):
                if label == 0:  # Background
                    continue

                color_idx = (label - 1) % len(self.seed_widget.colors)
                color = self.seed_widget.colors[color_idx]
                mask = (self.segmentation_result == label)
                visualization[mask] = [color.red(), color.green(), color.blue()]

            # Save as image
            qimg = QImage(visualization.data, width, height, width * 3, QImage.Format_RGB888)
            qimg.save(file_path)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SegmentationApp()
    window.show()
    sys.exit(app.exec())