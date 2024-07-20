import React, { useState } from 'react';

const ReportGeneration = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [result, setResult] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [previewUrl, setPreviewUrl] = useState(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreviewUrl(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const uploadImage = async () => {
    if (!selectedFile) {
      alert('Please select an image first.');
      return;
    }

    setIsLoading(true);
    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
      const response = await fetch('http://127.0.0.1:8000/upload_image', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setResult(data.text);
    } catch (error) {
      console.error('Error uploading image:', error);
      setResult('Error: ' + error.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="report-generation">
      <h1>Image Analysis App</h1>
      <div className="input-section">
        <input
          type="file"
          accept="image/*"
          onChange={handleFileChange}
        />
        <button 
          onClick={uploadImage} 
          disabled={isLoading}
        >
          {isLoading ? 'Analyzing...' : 'Upload and Analyze'}
        </button>
      </div>
      <div className="content-section">
        {previewUrl && (
          <div className="image-preview">
            <h2>Selected Image:</h2>
            <img src={previewUrl} alt="Selected" />
          </div>
        )}
        {result && (
          <div className="analysis-result">
            <h2>Analysis Result:</h2>
            <p>{result}</p>
          </div>
        )}
      </div>
      <style jsx>{`
        .report-generation {
          font-family: Arial, sans-serif;
          max-width: 800px;
          margin: 0 auto;
          padding: 20px;
        }
        h1 {
          color: #333;
          font-size: 24px;
          margin-bottom: 20px;
        }
        h2 {
          color: #444;
          font-size: 18px;
          margin-bottom: 10px;
        }
        .input-section {
          margin-bottom: 20px;
        }
        input[type="file"] {
          margin-right: 10px;
        }
        button {
          background-color: #4CAF50;
          border: none;
          color: white;
          padding: 10px 20px;
          text-align: center;
          text-decoration: none;
          display: inline-block;
          font-size: 16px;
          margin: 4px 2px;
          cursor: pointer;
          transition-duration: 0.4s;
        }
        button:hover {
          background-color: #45a049;
        }
        button:disabled {
          background-color: #cccccc;
          cursor: not-allowed;
        }
        .content-section {
          display: flex;
          flex-wrap: wrap;
          gap: 20px;
        }
        .image-preview, .analysis-result {
          flex: 1 1 300px;
          border: 1px solid #ddd;
          padding: 10px;
          border-radius: 4px;
        }
        .image-preview img {
          max-width: 100%;
          height: auto;
        }
        .analysis-result p {
          white-space: pre-wrap;
          word-break: break-word;
        }
      `}</style>
    </div>
  );
};

export default ReportGeneration;