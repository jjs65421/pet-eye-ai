// Browsers do not expose the device photo library used by the native app.
// App.js already skips these calls on web; this module keeps the web bundle safe.
export const requestPermissionsAsync = async () => ({ granted: false });
export const createAssetAsync = async () => {
  throw new Error('Photo-library saving is unavailable on web.');
};
