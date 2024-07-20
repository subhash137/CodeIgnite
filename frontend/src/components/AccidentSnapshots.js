// import React, { useState, useEffect } from 'react';
// import io from 'socket.io-client';

// const AccidentSnapshots = () => {
//   const [snapshot, setSnapshot] = useState(null);
//   const [alert, setAlert] = useState(null);

//   useEffect(() => {
//     // Set up Socket.IO connection
//     const socket = io('http://localhost:8000');
    
//     socket.on('accident_detected', (data) => {
//       setAlert(data.message);
//       // Fetch snapshot when an accident is detected
//       fetchSnapshots();
//     });

//     return () => socket.disconnect();
//   }, []);

//   const fetchSnapshots = () => {
//     fetch('http://127.0.0.1:8000/get_accidents')
//       .then(response => response.json())
//       .then(data => {
//         // Update state with the latest snapshot
//         if (data.length > 0) {
//           setSnapshot(data[0]);  // Display the first snapshot
//         }
//       })
//       .catch(error => console.error('Error fetching accidents:', error));
//   };

//   return (
//     <div>
//       {alert && <div className="alert">{alert}</div>}
//       <h2>Accident Snapshots</h2>
//       <div className="snapshot-gallery">
//         {snapshot ? (
//           <div className="snapshot-item">
//             <img
//               src={`http://127.0.0.1:8000/snapshots/${snapshot}`}
//               alt="Accident snapshot"
//               onError={(e) => {
//                 e.target.src = 'path/to/default-image.jpg';  // Fallback image
//               }}
//             />
//           </div>
//         ) : (
//           <p>No accidents detected</p>  // Display this when no accident has been detected
//         )}
//       </div>
//     </div>
//   );
// };

// export default AccidentSnapshots;


import React, { useState, useEffect } from 'react';
import io from 'socket.io-client';

const AccidentSnapshots = () => {
  const [snapshot, setSnapshot] = useState(null);
  const [alert, setAlert] = useState(null);

  useEffect(() => {
    // Set up Socket.IO connection
    const socket = io('http://localhost:8000');
    
    socket.on('accident_detected', (data) => {
      setAlert(data.message);
  
      fetchSnapshots();
    });

    return () => socket.disconnect();
  }, []);

  const fetchSnapshots = () => {
    fetch('http://127.0.0.1:8000/get_accidents')
      .then(response => response.json())
      .then(data => {
        // Update state with the latest snapshot
        if (data.length > 0) {
          setSnapshot(data[0]);  // Display the first snapshot
        }
      })
      .catch(error => console.error('Error fetching accidents:', error));
  };

  return (
    <div>
      {alert && <div className="alert">{alert}</div>}
      <h2>Accident Snapshots</h2>
      <div className="snapshot-gallery">
        {snapshot ? (
          <div className="snapshot-item">
            <img
              src={`http://127.0.0.1:8000/snapshots/${snapshot}`}
              alt="Accident snapshot"
              onError={(e) => {
                e.target.src = 'path/to/default-image.jpg';  // Fallback image
              }}
            />
          </div>
        ) : (
          <p> no accident detected</p>  // Display this when no accident has been detected
        )}
      </div>
    </div>
  );
};

export default AccidentSnapshots;
