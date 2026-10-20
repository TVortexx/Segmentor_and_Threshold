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
        
        # Initialize best thresholds and variance
        best_thresholds = []
        best_var = 0
        
        # Try all possible threshold combinations
        for t1 in range(1, 254):
            for t2 in range(t1+1, 255):
                # Class probabilities
                w0 = np.sum(hist[:t1])
                w1 = np.sum(hist[t1:t2])
                w2 = np.sum(hist[t2:])
                
                if w0 == 0 or w1 == 0 or w2 == 0:
                    continue
                
                # Class means
                mean0 = np.sum(np.arange(t1) * hist[:t1]) / w0
                mean1 = np.sum(np.arange(t1, t2) * hist[t1:t2]) / w1
                mean2 = np.sum(np.arange(t2, 256) * hist[t2:]) / w2
                
                # Total mean
                mean_total = mean0 * w0 + mean1 * w1 + mean2 * w2
                
                # Between-class variance
                var = (w0 * (mean0 - mean_total)**2 + 
                    w1 * (mean1 - mean_total)**2 + 
                    w2 * (mean2 - mean_total)**2)
                
                if var > best_var:
                    best_var = var
                    best_thresholds = [t1, t2]
        
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