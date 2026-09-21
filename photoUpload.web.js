export const uploadPhotoForAnalysis = async (url, uri) => {
  const form = new FormData();
  form.append('image', await (await fetch(uri)).blob(), 'pet-eye.jpg');
  return fetch(url, { method: 'POST', body: form });
};
