// // src/components/VideoStream.js
// import React from 'react';
// import VCD from '../assets/tenor.gif';

// const VideoStream = () => {
//   return (
//     <section className='bg-gray-900 mb-12 text-white'>
//       <div>
//         <h1>Real-time Object Detection</h1>
//         {/* <img src="http://127.0.0.1:9000/video_feed" alt="Video Stream" /> */}
//         <img
//           src={VCD}
//           className="w-full rounded-md border-2 border-gray-100"
//           width={500}
//           height={500}
//           alt="VCD"
//         />
//       </div>
//     </section>
//   );
// };

// export default VideoStream;


// src/App.js

import React from 'react';

import { FaCheckCircle } from 'react-icons/fa';

const VideoStream=()=> {
  return (
    <div className="flex h-screen">

      <div className="flex-1 bg-gray-200 p-4">
        {/* <video
          className="w-full h-full object-cover"
          src="path_to_your_video.mp4"
          controls
        /> */}
        <img src="http://127.0.0.1:9000/video_feed" alt="Video Stream"/>
      </div>


      <div className="flex-1 bg-white p-4 flex flex-col items-center justify-center">
        <div className="bg-green-100 p-4 rounded-md shadow-md w-3/4 flex items-center">
          <FaCheckCircle className="text-green-500 mr-2" />
          <p className="text-green-700">Message Sent Successfully!</p>
        </div>
        <div className="mt-4 text-gray-700 text-center">
          <p>Your message has been delivered to the recipient.</p>
        </div>
      </div>
    </div>
  );
}

export default VideoStream;
