import { useState, useRef } from 'react';
import './ImageSegmenter.css';

const API_URL = 'http://127.0.0.1:5000';

interface ImageSegmenterProps {
  title?: string;
}

const ImageSegmenter = ({ title = 'Wall Segmentation Tool' }: ImageSegmenterProps) => {
  const [originalImage, setOriginalImage] = useState<string | null>(null);
  const [segmentedImage, setSegmentedImage] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [statusText, setStatusText] = useState('');
  const [processingTime, setProcessingTime] = useState<number | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Function to simulate progress while waiting for response
  const simulateProgress = () => {
    let currentProgress = 0;
    const maxProgress = 90; // Only go to 90%, the last 10% when response arrives
    
    // Return the setInterval ID so we can clear it later
    return setInterval(() => {
      if (currentProgress < 30) {
        currentProgress += 5;
      } else if (currentProgress < 60) {
        currentProgress += 2;
      } else if (currentProgress < maxProgress) {
        currentProgress += 0.5;
      } else {
        return;
      }
      
      setProgress(currentProgress);
      
      // Update status text based on progress
      if (currentProgress < 25) {
        setStatusText("Loading image...");
      } else if (currentProgress < 50) {
        setStatusText("Preprocessing...");
      } else if (currentProgress < 75) {
        setStatusText("Analyzing walls...");
      } else {
        setStatusText("Generating segmentation mask...");
      }
    }, 200);
  };

  const handleImageUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    
    // Display original image
    const reader = new FileReader();
    reader.onload = (event) => {
      setOriginalImage(event.target?.result as string);
    };
    reader.readAsDataURL(file);
    
    // Reset and prepare for processing
    setSegmentedImage(null);
    setIsLoading(true);
    setProgress(0);
    setStatusText("Starting...");
    
    // Start progress simulation
    const progressInterval = simulateProgress();
    
    // Create form data
    const formData = new FormData();
    formData.append('image', file);
    
    try {
      const startTime = Date.now();
      
      // Send request to Flask backend
      const response = await fetch(`${API_URL}/segment`, {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error(`Error: ${response.status} ${response.statusText}`);
      }
      
      // Get response as blob
      const blob = await response.blob();
      const imageUrl = URL.createObjectURL(blob);
      
      // Complete progress
      clearInterval(progressInterval);
      setProgress(100);
      
      // Calculate processing time
      const timeElapsed = (Date.now() - startTime) / 1000;
      setProcessingTime(timeElapsed);
      setStatusText(`Processing completed in ${timeElapsed.toFixed(1)}s`);
      
      // Display segmented image
      setSegmentedImage(imageUrl);
    } catch (error) {
      clearInterval(progressInterval);
      setStatusText(`Error: ${error instanceof Error ? error.message : 'Unknown error'}`);
      console.error(error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setOriginalImage(null);
    setSegmentedImage(null);
    setProcessingTime(null);
    setStatusText('');
    setProgress(0);
    
    // Reset file input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="image-segmenter">
      <h2>{title}</h2>
      
      <div className="info-box">
        <p>Upload an image containing walls, and the AI will identify and highlight them.</p>
        <p>
          <strong>Note:</strong> Large images will be automatically resized for faster processing.
          Processing time depends on image complexity and server load.
        </p>
      </div>
      
      <div className="upload-container">
        <input 
          type="file" 
          accept="image/*" 
          onChange={handleImageUpload} 
          disabled={isLoading}
          ref={fileInputRef}
          className="file-input"
        />
        <button 
          onClick={handleReset} 
          className="reset-button"
          disabled={isLoading || (!originalImage && !segmentedImage)}
        >
          Reset
        </button>
      </div>

      {isLoading && (
        <div className="loading-container">
          <div className="progress-bar-container">
            <div className="progress-bar" style={{ width: `${progress}%` }}></div>
          </div>
          <p className="status-text">{statusText}</p>
        </div>
      )}
      
      {(originalImage || segmentedImage) && (
        <div className="results-container">
          <h3>Results:</h3>
          {processingTime !== null && (
            <p className="processing-time">
              Processing time: {processingTime.toFixed(2)}s
            </p>
          )}
          
          <div className="images-container">
            {originalImage && (
              <div className="image-box">
                <h4>Original Image</h4>
                <img src={originalImage} alt="Original" className="result-image" />
              </div>
            )}
            
            {segmentedImage && (
              <div className="image-box">
                <h4>Wall Segmentation</h4>
                <img src={segmentedImage} alt="Segmented" className="result-image" />
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default ImageSegmenter;
