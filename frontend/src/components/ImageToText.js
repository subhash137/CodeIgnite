import React, { useState } from 'react';
import './ImageToText.css'

const ImageToText= () => {
  const [file, setFile] = useState(null);
  const [description, setDescription] = useState("");

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleUpload = () => {
    const formData = new FormData();
    formData.append('file', file);

    fetch('http://localhost:8000/upload_image', {
      method: 'POST',
      body: formData
    })
    .then(response => response.json())
    .then(data => {
      setDescription(data.description);
    })
    .catch(error => {
      console.error("There was an error uploading the file!", error);
    });
  };

  return (
    <div className="App">
      <h1>Image Description</h1>
      <input type="file" onChange={handleFileChange} />
      <button onClick={handleUpload}>Upload</button>
      <p>{description}</p>
    </div>
  );
}

export default ImageToText;
