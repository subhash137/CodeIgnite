// import React, { useState } from 'react';
// import axios from 'axios';

// function LicensePlate() {
//   const [file, setFile] = useState(null);
//   const [plates, setPlates] = useState([]);

//   const handleFileChange = (event) => {
//     setFile(event.target.files[0]);
//   };

//   const handleUpload = async () => {
//     if (!file) return;

//     const formData = new FormData();
//     formData.append('file', file);

//     try {
//       const response = await axios.post('http://127.0.0.1:8000/upload', formData, {
//         headers: {
//           'Content-Type': 'multipart/form-data'
//         }
//       });

//       setPlates(response.data.plates);
//     } catch (error) {
//       console.error('Error uploading file:', error);
//     }
//   };

//   return (
//     <div className="min-h-screen bg-gray-100 flex flex-col items-center justify-center">
//       <h1 className="text-3xl font-bold mb-6">License Plate Detection</h1>
//       <input type="file" onChange={handleFileChange} className="mb-4" />
//       <button onClick={handleUpload} className="px-4 py-2 bg-blue-500 text-white rounded">Upload</button>

//       {plates.length > 0 && (
//         <div className="mt-6">
//           <h2 className="text-2xl mb-4">Detected Plates</h2>
//           {plates.map((plate, index) => (
//             <div key={index} className="mb-4 p-4 bg-white shadow rounded">
//               <img src={`data:image/png;base64,${btoa(String.fromCharCode(...new Uint8Array(plate.image)))}`} alt="License Plate" className="mb-2" />
//               <p>{plate.text}</p>
//             </div>
//           ))}
//         </div>
//       )}
//     </div>
//   );
// }

// export default LicensePlate;

import React, { useState } from 'react';

function LicensePlate() {
  const [file, setFile] = useState(null);
  const [plates, setPlates] = useState([]);

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleUpload = async () => {
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://127.0.0.1:8000/upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const data = await response.json();
      setPlates(data.plates);
    } catch (error) {
      console.error('Error uploading file:', error);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col items-center justify-center">
      <h1 className="text-3xl font-bold mb-6">License Plate Detection</h1>
      <input type="file" onChange={handleFileChange} className="mb-4" />
      <button onClick={handleUpload} className="px-4 py-2 bg-blue-500 text-white rounded">Upload</button>

      {plates.length > 0 && (
        <div className="mt-6">
          <h2 className="text-2xl mb-4">Detected Plates</h2>
          {plates.map((plate, index) => (
            <div key={index} className="mb-4 p-4 bg-white shadow rounded">
              <img src={`data:image/png;base64,${plate.image}`} alt="License Plate" className="mb-2" />
              <p>{plate.text}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default LicensePlate;
