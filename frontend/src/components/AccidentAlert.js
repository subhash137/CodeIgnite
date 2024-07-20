// import React, { useEffect, useState } from 'react';
// import io from 'socket.io-client';

// const AccidentAlert = () => {
//   const [alert, setAlert] = useState(null);

//   useEffect(() => {
//     const socket = io('http://127.0.0.1:8000');
//     socket.on('accident_detected', (data) => {
//       setAlert(data.message);
//     });
//     return () => socket.disconnect();
//   }, []);

//   return alert ? <div className="alert">{alert}</div> : null;
// };

// export default AccidentAlert;

// // Add this component to your App.js