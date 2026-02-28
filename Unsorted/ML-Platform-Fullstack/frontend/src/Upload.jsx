import React, { useState } from 'react';
import { uploadFile } from './api';

export default function Upload({ userId }) {
  const [file, setFile] = useState(null);

  const handleUpload = async () => {
    if (!file) return;
    await uploadFile(file, userId);
    alert('File uploaded!');
  };

  return (
    <div>
      <input type="file" onChange={(e) => setFile(e.target.files[0])} />
      <button onClick={handleUpload}>Upload</button>
    </div>
  );
}