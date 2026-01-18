// API Client for Plate Detection System
class PlateDetectionAPI {
    constructor(baseUrl = '') {
        this.baseUrl = baseUrl;
    }

    // Check system health
    async checkHealth() {
        const response = await fetch(`${this.baseUrl}/api/health`);
        return await response.json();
    }

    // Upload single image
    async uploadImage(file) {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${this.baseUrl}/api/upload/single`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Upload failed');
        }

        return await response.json();
    }

    // Process single image
    async processImage(file, options = {}) {
        const formData = new FormData();
        formData.append('image', file);
        
        // Default options
        const defaultOptions = {
            confidence_threshold: 0.3,
            use_yolo: true,
            use_transformer: false,
            multi_scale: false,
            ocr: {
                easyocr: true,
                tesseract: true,
                google_vision: true,
                deep_ocr: true
            },
            save_output: true
        };

        formData.append('options', JSON.stringify({ ...defaultOptions, ...options }));

        const response = await fetch(`${this.baseUrl}/api/process/single`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Processing failed');
        }

        return await response.json();
    }

    // Batch upload
    async uploadMultiple(files) {
        const formData = new FormData();
        files.forEach(file => {
            formData.append('files', file);
        });

        const response = await fetch(`${this.baseUrl}/api/upload/multiple`, {
            method: 'POST',
            body: formData
        });

        return await response.json();
    }

    // Get results
    async getResults(page = 1, perPage = 20, filters = {}) {
        const params = new URLSearchParams({
            page: page.toString(),
            per_page: perPage.toString(),
            ...filters
        });

        const response = await fetch(`${this.baseUrl}/api/results?${params}`);
        return await response.json();
    }

    // Get statistics
    async getStatistics() {
        const response = await fetch(`${this.baseUrl}/api/results/stats`);
        return await response.json();
    }

    // Export results
    async exportResults(format = 'json') {
        const response = await fetch(`${this.baseUrl}/api/results/export?format=${format}`);
        
        if (format === 'json') {
            return await response.json();
        } else {
            return await response.blob();
        }
    }
}

// Global API client instance
const apiClient = new PlateDetectionAPI();

// Utility functions
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function formatTime(seconds) {
    if (seconds < 1) return `${(seconds * 1000).toFixed(0)}ms`;
    return `${seconds.toFixed(2)}s`;
}

// Drawing utilities
class DetectionVisualizer {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.image = null;
    }

    setImage(image) {
        this.image = image;
        this.canvas.width = image.width;
        this.canvas.height = image.height;
        this.ctx.drawImage(image, 0, 0);
    }

    drawDetections(detections, plateResults = []) {
        if (!this.image) return;

        // Clear and redraw image
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.drawImage(this.image, 0, 0);

        // Draw each detection
        detections.forEach((detection, index) => {
            this.drawDetection(detection, plateResults[index]);
        });
    }

    drawDetection(detection, plateResult = {}) {
        const [x1, y1, x2, y2] = detection.bbox;
        const confidence = detection.confidence || 0;
        
        // Choose color based on confidence
        let color;
        if (confidence >= 0.8) color = '#10b981'; // Green
        else if (confidence >= 0.6) color = '#f59e0b'; // Yellow
        else color = '#ef4444'; // Red

        // Draw bounding box
        this.ctx.strokeStyle = color;
        this.ctx.lineWidth = 3;
        this.ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        // Draw label background
        const label = `Plate: ${(confidence * 100).toFixed(1)}%`;
        this.ctx.fillStyle = color;
        this.ctx.font = 'bold 14px Arial';
        const textWidth = this.ctx.measureText(label).width;
        
        this.ctx.fillRect(x1, y1 - 25, textWidth + 10, 25);

        // Draw label text
        this.ctx.fillStyle = 'white';
        this.ctx.fillText(label, x1 + 5, y1 - 7);

        // Draw OCR text if available
        const plateText = plateResult.text || '';
        if (plateText) {
            this.ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
            this.ctx.font = 'bold 16px Arial';
            const plateTextWidth = this.ctx.measureText(plateText).width;
            
            this.ctx.fillRect(x1, y2 + 5, plateTextWidth + 10, 25);
            
            this.ctx.fillStyle = 'white';
            this.ctx.fillText(plateText, x1 + 5, y2 + 22);
        }
    }

    clear() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.image = null;
    }
}