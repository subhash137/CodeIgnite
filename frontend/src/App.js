
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
