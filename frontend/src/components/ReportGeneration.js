import React, { useState } from 'react';

const ReportGeneration = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [result, setResult] = useState('');
  const [isLoading, setIsLoading] = useState(false);

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
    <div className="p-4">
      <h1 className="text-2xl font-bold mb-4">Image Analysis App</h1>
      <input
        type="file"
        accept="image/*"
        onChange={(e) => setSelectedFile(e.target.files[0])}
        className="mb-4"
      />
      <button 
        onClick={uploadImage} 
        disabled={isLoading}
        className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
      >
        {isLoading ? 'Analyzing...' : 'Upload and Analyze'}
      </button>
      {result && (
        <div className="mt-4 p-4 border rounded">
          <h2 className="text-xl font-semibold mb-2">Analysis Result:</h2>
          <p>{result}</p>
        </div>
      )}
    </div>
  );
};

export default ReportGeneration;