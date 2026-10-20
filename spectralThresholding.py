import numpy as np
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog
from PyQt5.QtCore import Qt, pyqtSignal, QObject
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.uic import loadUi

class ThresholdingProcessor(QObject):
    update_output_signal = pyqtSignal(QImage)
    
    def __init__(self):
        super().__init__()
        self.current_image = None
    
    def set_image(self, image):
        self.current_image = image
    
    def spectral_first_mode(self, image):
        """
        Spectral thresholding using first mode (binary)
        This is essentially Otsu's method but named differently for the UI
        """
        # Calculate histogram
        hist, bins = np.histogram(image.flatten(), bins=256, range=[0,256])
        hist = hist.astype(float) / hist.sum()  # Normalize
        
        # Initialize variables
        best_thresh = 0
        best_var = 0
        
        # Iterate through all possible thresholds
        for threshold in range(1, 256):
            # Class probabilities
            w0 = np.sum(hist[:threshold])
            w1 = np.sum(hist[threshold:])
            
            if w0 == 0 or w1 == 0:
                continue
            
            # Class means
            mean0 = np.sum(np.arange(threshold) * hist[:threshold]) / w0
            mean1 = np.sum(np.arange(threshold, 256) * hist[threshold:]) / w1
            
            # Between-class variance
            var = w0 * w1 * (mean0 - mean1) ** 2
            
            if var > best_var:
                best_var = var
                best_thresh = threshold
        
        # Apply threshold
        binary = np.zeros_like(image, dtype=np.uint8)
        binary[image > best_thresh] = 255
        return binary
    
    def spectral_second_mode(self, image):
        """
        Spectral thresholding using second mode (multi-level)
        This implements multi-level Otsu for 3 classes
        """
        # Calculate histogram
        hist, bins = np.histogram(image.flatten(), bins=256, range=[0,256])
        hist = hist.astype(float) / hist.sum()  # Normalize
        
        # Cambio: el original probaba los ~32.000 pares (t1, t2) con dos bucles
        # Python y recalculaba sumas del histograma en cada par (~0,6 s).
        # Con sumas acumuladas, cada clase sale en O(1) y todos los pares se
        # evalúan a la vez como una matriz 256x256 (mismos umbrales, ~ms).
        P = np.concatenate([[0.0], np.cumsum(hist)])                  # P[k] = sum(hist[:k])
        S = np.concatenate([[0.0], np.cumsum(hist * np.arange(256))])  # S[k] = sum(i*hist[:k])
        t1 = np.arange(256)[:, None]
        t2 = np.arange(256)[None, :]
        w0, w1, w2 = P[t1], P[t2] - P[t1], P[256] - P[t2]
        s0, s1, s2 = S[t1], S[t2] - S[t1], S[256] - S[t2]
        valido = (t1 >= 1) & (t1 <= 253) & (t2 > t1) & (t2 <= 254) & (w0 > 0) & (w1 > 0) & (w2 > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            m0, m1, m2 = s0 / w0, s1 / w1, s2 / w2
            mean_total = m0 * w0 + m1 * w1 + m2 * w2
            var = (w0 * (m0 - mean_total) ** 2 + w1 * (m1 - mean_total) ** 2
                   + w2 * (m2 - mean_total) ** 2)
        var = np.where(valido, np.nan_to_num(var), -1.0)
        best_thresholds = []
        if var.max() > 0:
            i, j = np.unravel_index(np.argmax(var), var.shape)   # primer máximo, como el original
            best_thresholds = [int(i), int(j)]

        # Apply thresholds
        segmented = np.zeros_like(image, dtype=np.uint8)
        if len(best_thresholds) == 2:
            segmented[image <= best_thresholds[0]] = 0
            segmented[(image > best_thresholds[0]) & (image <= best_thresholds[1])] = 128
            segmented[image > best_thresholds[1]] = 255
        else:
            # Fallback to binary if we didn't find good thresholds
            threshold = 128
            segmented[image > threshold] = 255
        
        return segmented
    

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi('task04.ui', self)
        
        # Initialize threshold processor
        self.threshold_processor = ThresholdingProcessor()
        
        self.threshold_processor.update_output_signal.connect(self.update_output_image)

        
    
    
    
    def update_output_image(self, qimage):
        """Update the output image display"""
        pixmap = QPixmap.fromImage(qimage)
        self.OutputImage.setPixmap(pixmap.scaled(
            self.OutputImage.width(), 
            self.OutputImage.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ))
    

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())