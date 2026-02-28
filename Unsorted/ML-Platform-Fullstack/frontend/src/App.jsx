import React, { useState } from 'react';
import { register, startTraining } from './api';
import Upload from './Upload';

export default function App() {
  const [userId, setUserId] = useState(null);

  const handleRegister = async () => {
    const res = await register('user@example.com', 'password');
    setUserId(1); // Simplified for demo
  };

  return (
    <div>
      {!userId ? (
        <button onClick={handleRegister}>Register</button>
      ) : (
        <>
          <Upload userId={userId} />
          <button onClick={() => startTraining('random_forest', '/data/file.csv', userId)}>
            Train Model
          </button>
        </>
      )}
    </div>
  );
}