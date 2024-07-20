import React from 'react';
import { NavLink } from 'react-router-dom';
import './Navbar.css'; // Optional: For custom styling

const Navbar = () => {
    return (
        <nav>
            <ul>
                <li><NavLink to="/accident-detection" activeClassName="active">Accident Detection</NavLink></li>
                <li><NavLink to="/license-plate-detection" activeClassName="active">License Plate Detection</NavLink></li>
                <li><NavLink to="/report-generation" activeClassName="active">Report Generation</NavLink></li>
            </ul>
        </nav>
    );
};

export default Navbar;
