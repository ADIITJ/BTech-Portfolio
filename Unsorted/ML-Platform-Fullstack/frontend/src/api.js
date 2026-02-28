import axios from 'axios';

const api = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:8000',
});

export const register = (email, password) => 
  api.post('/register', { email, password });

export const uploadFile = (file, userId) => {
  const formData = new FormData();
  formData.append('file', file);
  return api.post(`/upload?user_id=${userId}`, formData);
};

export const startTraining = (modelType, datasetPath, userId) =>
  api.post('/train', { model_type: modelType, dataset_path: datasetPath, user_id: userId });