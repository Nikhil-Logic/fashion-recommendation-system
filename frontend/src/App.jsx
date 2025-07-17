import { useState } from 'react';
import axios from 'axios';
import { Upload, Sparkles } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);

  const backendUrl = 'https://fashion-recommendation-system-ibub.onrender.com'; // Your deployed FastAPI backend
  const HUGGINGFACE_IMAGE_BASE_URL = "https://huggingface.co/datasets/TheNikhil/fashion-product-dataset/resolve/main/";

  const handleFileChange = (e) => {
    const uploaded = e.target.files[0];
    setFile(uploaded);
    setPreview(URL.createObjectURL(uploaded));
    setRecommendations([]);
  };

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await axios.post(`${backendUrl}/recommend`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });

      // Fix relative paths by prefixing Hugging Face base URL
      const fixedUrls = res.data.recommendations.map(path =>
        path.startsWith("http") ? path : `${HUGGINGFACE_IMAGE_BASE_URL}${path}`
      );

      console.log("Recommended URLs:", fixedUrls);
      setRecommendations(fixedUrls);
    } catch (err) {
      console.error(err);
      alert('Error fetching recommendations');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-tr from-indigo-100 to-white flex flex-col items-center p-6">
      <motion.div
        className="text-center mb-10"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.7 }}
      >
        <h1 className="text-4xl font-bold text-indigo-700 mb-2 flex items-center justify-center gap-2">
          <Sparkles className="h-7 w-7 text-yellow-500 animate-pulse" />
          Fashion Recommender
        </h1>
        <p className="text-gray-600">Upload a fashion image and get 5 similar styles recommended!</p>
      </motion.div>

      <motion.label
        className="cursor-pointer bg-white border-2 border-dashed border-indigo-300 rounded-lg p-8 flex flex-col items-center hover:shadow-lg transition duration-300"
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
      >
        <Upload className="h-10 w-10 text-indigo-500 mb-2" />
        <span className="text-indigo-700 font-medium">Click to upload an image</span>
        <input
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          className="hidden"
        />
      </motion.label>

      <AnimatePresence>
        {preview && (
          <motion.div
            className="mt-6"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
          >
            <h3 className="text-lg text-gray-700 mb-2">Preview:</h3>
            <img
              src={preview}
              alt="Preview"
              className="h-60 w-60 object-cover rounded-xl shadow-md border"
            />
          </motion.div>
        )}
      </AnimatePresence>

      <motion.button
        onClick={handleUpload}
        disabled={!file || loading}
        className="mt-6 bg-indigo-600 text-white px-6 py-3 rounded-xl hover:bg-indigo-700 transition duration-300 disabled:opacity-50"
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
      >
        {loading ? 'Recommending...' : 'Get Recommendations'}
      </motion.button>

      <AnimatePresence>
        {recommendations.length > 0 && (
          <motion.div
            className="mt-10 w-full max-w-5xl"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <h3 className="text-xl font-semibold text-gray-800 mb-4">Recommended Styles:</h3>
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-4">
              {recommendations.map((url, idx) => (
                <motion.img
                  key={idx}
                  src={url}
                  alt={`Recommended ${idx}`}
                  className="h-48 w-full object-cover rounded-xl shadow-md border"
                  whileHover={{ scale: 1.02 }}
                />
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default App;
