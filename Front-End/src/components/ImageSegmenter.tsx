import { useState, useRef } from 'react';

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
  };  return (
    <div className="max-w-4xl mx-auto p-8 bg-white rounded-xl shadow-xl border border-gray-100">
      <h2 className="text-3xl font-bold text-center text-gray-800 mb-6 tracking-tight">{title}</h2>
      
      <div className="bg-blue-50 border-l-4 border-[#3498db] p-5 rounded-lg mb-8">
        <div className="flex items-start">
          <svg className="w-6 h-6 text-[#3498db] mr-3 mt-1 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
          </svg>
          <div>
            <p className="mb-2 text-gray-700">Upload an image containing walls, and the AI will identify and highlight them.</p>
            <p className="text-gray-600">
              <strong className="font-medium">Note:</strong> Large images will be automatically resized for faster processing.
              Processing time depends on image complexity and server load.
            </p>
          </div>
        </div>
      </div>
      
      <div className="bg-gray-50 p-6 rounded-lg border border-gray-200 mb-8">
        <p className="text-lg font-medium mb-4 text-gray-700">Upload an image to analyze</p>
        <div className="flex flex-col sm:flex-row items-center gap-4">
          <label className="flex-1 w-full">
            <input 
              type="file" 
              accept="image/*" 
              onChange={handleImageUpload} 
              disabled={isLoading}
              ref={fileInputRef}
              className="block w-full text-gray-700 bg-white border-2 border-gray-300 rounded-lg py-3 px-4 focus:outline-none focus:ring-2 focus:ring-[#3498db] focus:border-[#3498db] transition-all duration-200 file:mr-4 file:py-2 file:px-4 file:border-0 file:rounded-md file:text-sm file:font-medium file:bg-[#3498db] file:text-white hover:file:bg-[#2980b9]"
            />
          </label>
          <button 
            onClick={handleReset} 
            className="w-full sm:w-auto bg-gradient-to-r from-[#e74c3c] to-[#c0392b] hover:from-[#c0392b] hover:to-[#e74c3c] text-white py-3 px-8 rounded-lg shadow-md transition-all duration-300 font-medium disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:from-[#e74c3c] disabled:hover:to-[#c0392b]"
            disabled={isLoading || (!originalImage && !segmentedImage)}
          >
            Reset
          </button>
        </div>
      </div>      {isLoading && (
        <div className="mb-6">
          <div className="relative pt-6 pb-8">
            <div className="flex mb-2 items-center justify-between">
              <div>
                <span className="text-sm font-semibold inline-block py-1 px-2 uppercase rounded-full text-[#3498db] bg-blue-50">
                  Processing
                </span>
              </div>
              <div className="text-right">
                <span className="text-sm font-semibold inline-block text-[#3498db]">
                  {Math.round(progress)}%
                </span>
              </div>
            </div>
            <div className="h-2 bg-gray-200 rounded-full overflow-hidden shadow-inner">
              <div 
                className="h-full bg-gradient-to-r from-[#3498db] to-[#2ecc71] transition-all duration-300 rounded-full" 
                style={{ width: `${progress}%` }}
              ></div>
            </div>
          </div>
          <div className="flex items-center justify-center">
            {progress < 100 && (
              <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-[#3498db]" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
            )}
            <p className="text-gray-700 font-medium">{statusText}</p>
          </div>
        </div>
      )}
      
      {(originalImage || segmentedImage) && (
        <div className="mt-8 bg-white rounded-lg border border-gray-200 shadow-md p-6">
          <div className="flex justify-between items-center border-b pb-4 mb-4">
            <h3 className="text-xl font-bold text-gray-800">Results</h3>
            {processingTime !== null && (
              <span className="px-3 py-1 rounded-full bg-blue-50 text-[#3498db] text-sm font-medium">
                Processed in {processingTime.toFixed(2)}s
              </span>
            )}
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {originalImage && (
              <div className="flex flex-col">
                <div className="bg-gray-50 rounded-t-lg p-3 border border-gray-200 border-b-0">
                  <h4 className="text-lg font-medium text-center text-gray-700">Original Image</h4>
                </div>
                <div className="border border-gray-200 rounded-b-lg p-2 bg-gray-800 shadow-inner overflow-hidden flex items-center justify-center">
                  <img 
                    src={originalImage} 
                    alt="Original" 
                    className="max-w-full max-h-[400px] object-contain rounded" 
                  />
                </div>
              </div>
            )}
            
            {segmentedImage && (
              <div className="flex flex-col">
                <div className="bg-gray-50 rounded-t-lg p-3 border border-gray-200 border-b-0">
                  <h4 className="text-lg font-medium text-center text-gray-700">Wall Segmentation</h4>
                </div>
                <div className="border border-gray-200 rounded-b-lg p-2 bg-gray-800 shadow-inner overflow-hidden flex items-center justify-center">
                  <img 
                    src={segmentedImage} 
                    alt="Segmented" 
                    className="max-w-full max-h-[400px] object-contain rounded" 
                  />
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default ImageSegmenter;
