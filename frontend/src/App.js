// // src/App.js
// import React from 'react';
// import VideoStream from './components/VideoStream';
// import MobileSidebar from './components/MobileSidebar';

// function App() {
//   return (
//     <div className="App">
//       <img src="http://127.0.0.1:8000/video_feed" alt="Video Stream" />
//       {/* <MobileSidebar/> */}
//       {/* <VideoStream /> */}
//     </div>
//   );
// }

// export default App;

// src/App.js
// import React from 'react';
// import VideoStream from './components/VideoStream';
// import MobileSidebar from './components/MobileSidebar';
// import AccidentSnapshots from './components/AccidentSnapshots';
// import AccidentAlert from './components/AccidentAlert';

// function App() {
//   return (
//     <div className="App">
//       <img src="http://127.0.0.1:8000/video_feed" alt="Video Stream" autoPlay />
//       {/* <MobileSidebar/> */}
//       {/* <VideoStream /> */}
//       <AccidentAlert />
//       <AccidentSnapshots />
//     </div>
//   );
// }

// export default App;


// import React, { useEffect, useState, useRef } from 'react';
// import io from 'socket.io-client';

// const socket = io('http://127.0.0.1:5000');

// function App() {
//   const [accidentDetected, setAccidentDetected] = useState(false);
//   const [snapshot, setSnapshot] = useState(null);
//   const [videoFrames, setVideoFrames] = useState([]);
//   const videoRef = useRef();

//   useEffect(() => {
//     socket.on('accident_detected', (data) => {
//       setAccidentDetected(true);
//       setSnapshot(data.snapshot);
//     });

//     socket.on('video_frame', (data) => {
//       setVideoFrames(prevFrames => [...prevFrames, `data:image/jpeg;base64,${data.frame}`]);
//     });

//     socket.on('video_processed', (data) => {
//       console.log('Video processing completed:', data);
//     });

//     return () => {
//       socket.off('accident_detected');
//       socket.off('video_frame');
//       socket.off('video_processed');
//     };
//   }, []);

//   const startVideoProcessing = () => {
//     socket.emit('start_video', { video_path: 'video3.mp4' });
//   };

//   useEffect(() => {
//     if (videoRef.current && videoFrames.length > 0) {
//       const context = videoRef.current.getContext('2d');
//       const image = new Image();
//       image.src = videoFrames[videoFrames.length - 1];
//       image.onload = () => {
//         context.drawImage(image, 0, 0, videoRef.current.width, videoRef.current.height);
//       };
//     }
//   }, [videoFrames]);

//   return (
//     <div className="App">
//       <h1>Accident Detection System</h1>
//       <img src="http://127.0.0.1:6000/video_feed" alt="Video Stream" />
//       <button onClick={startVideoProcessing}>Start Video Processing</button>
//       <canvas ref={videoRef} width="640" height="480"></canvas>
//       {accidentDetected && (
//         <div>
//           <h2>Accident Detected!</h2>
//           {snapshot && <img src={`/${snapshot}`} alt="Accident Snapshot" />}
//         </div>
//       )}
//     </div>
//   );
// }

// export default App;


// import React from 'react';
// import AccidentSnapshots from './components/AccidentSnapshots';
// import { BrowserRouter, Routes, Route } from "react-router-dom";
// import LicensePlate from './components/LicensePlate';
// import ImageToText from './components/ImageToText';

// function App() {
//   return (
//     <div className="App">
//       <h1>Accident Detection System</h1>
//       <div className="video-container">
//         <img src="http://localhost:8000/video_feed" alt="Video Stream" />
//       </div>
//       <AccidentSnapshots />
//       <BrowserRouter>
//       <Routes>
//         <Route path="/accident-detection" element={<AccidentSnapshots />} />
//         <Route path="/license-plate" element={<LicensePlate />} />
//         <Route path="/image-to-text" element={<ImageToText />} />
//       </Routes>
//     </BrowserRouter>
//     </div>
//   );
// }

// export default App;



import React from 'react';
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Navbar from './components/Navbar';
import AccidentDetection from './components/AccidentDetection';
import LicensePlateDetection from './components/LicensePlateDetection';
import ReportGeneration from './components/ReportGeneration';
import './App.css'; // Optional: For custom styling

function App() {
    return (
        <BrowserRouter>
            <div className="App">
                <Navbar />
                <div id="messageContainer">
                    <Routes>
                        <Route path="/accident-detection" element={<AccidentDetection />} />
                        <Route path="/license-plate-detection" element={<LicensePlateDetection />} />
                        <Route path="/report-generation" element={<ReportGeneration />} />
                        <Route path="/" element={
                            <div>
                                <h1>Welcome!</h1>
                                <p>Please select an option from the navbar.</p>
                            </div>
                        } />
                    </Routes>
                </div>
            </div>
        </BrowserRouter>
    );
}

export default App;
