import React, { useState, useEffect } from 'react';
import io from 'socket.io-client';
import './AccidentDetection.css'; // Optional: For custom styling

const AccidentDetection = () => {
    const [message, setMessage] = useState('');
    const [snapshot, setSnapshot] = useState(null);

    useEffect(() => {
        const socket = io('http://localhost:8000');

        socket.on('accident_detected', (data) => {
            setMessage(data.message || 'Accident detected!');
            fetchSnapshot();
        });

        return () => {
            socket.off('accident_detected');
        };
    }, []);

    const fetchSnapshot = () => {
        fetch('http://127.0.0.1:8000/get_accidents')
            .then(response => response.json())
            .then(data => {
                if (data.length > 0) {
                    setSnapshot(data[0]); // Display the first snapshot
                }
            })
            .catch(error => console.error('Error fetching accidents:', error));
    };

    return (
        <div className="accident-detection">
            <h1>Accident Detection</h1>
            <p>Accident Detection activated. Monitoring for accidents...</p>
            <div className="aciddent">
            <div className="video-container">
                <img src="http://localhost:8000/video_feed" alt="Video Stream" />
            </div>
            <div className="inner-container">
            
            {snapshot && (
                <div className="snapshot-gallery">
                    <div className="snapshot-item">
                        <img
                            src={`http://127.0.0.1:8000/snapshots/${snapshot}`}
                            alt="Accident snapshot"
                            onError={(e) => {
                                e.target.src = 'path/to/default-image.jpg'; // Provide a default image path
                            }}
                        />
                    </div>
                </div>
            )}

            {message && <p className="notification-message">{message}</p>}
                </div>
            </div>
        </div>
    );
};

export default AccidentDetection;