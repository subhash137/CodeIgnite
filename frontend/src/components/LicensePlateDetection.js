import React, { useState } from 'react';
import axios from 'axios'; // Make sure to install axios: `npm install axios`
import './LicensePlateDetection.css'; // Import the CSS file

const LicensePlateDetection = () => {
    const [file, setFile] = useState(null);
    const [message, setMessage] = useState('');
    const [plates, setPlates] = useState([]);

    // Handle file input change
    const handleFileChange = (e) => {
        setFile(e.target.files[0]);
    };

    // Handle form submission
    const handleUpload = async () => {
        if (!file) {
            setMessage('Please select a file to upload.');
            return;
        }
    
        const formData = new FormData();
        formData.append('file', file);
    
        try {
            const response = await axios.post('http://127.0.0.1:8000/upload', formData);
            const data = response.data;
            setPlates(data.plates);
            setMessage('File uploaded successfully.');
        } catch (error) {
            setMessage('Error uploading file.');
            console.error('Error uploading file:', error);
        }
    };

    return (
        <div className="license-plate-detection">
            <h1>License Plate Detection</h1>
            <input type="file" onChange={handleFileChange} />
            <button onClick={handleUpload}>Upload</button>
            {message && <p className="message">{message}</p>}
            {plates.length > 0 && (
                <div className="plates-list">
                    <h2>Detected Plates</h2>
                    {plates.map((plate, index) => (
                        <div key={index} className="plate-item">
                            <img src={`data:image/png;base64,${plate.image}`} alt="License Plate" />
                            <p>{plate.text}</p>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
};

export default LicensePlateDetection;

