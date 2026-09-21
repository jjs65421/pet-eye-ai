import { File } from 'expo-file-system';
import { fetch } from 'expo/fetch';

// Expo Go's modern networking stack accepts File objects in FormData, rather
// than the deprecated { uri, type, name } React Native upload object.
export const uploadPhotoForAnalysis = (url, uri) => {
  const form = new FormData();
  form.append('image', new File(uri));
  return fetch(url, { method: 'POST', body: form });
};
