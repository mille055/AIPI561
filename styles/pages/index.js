import { useState } from 'react';
import axios from 'axios';

export default function Home() {
  const [query, setQuery] = useState('');
  const [response, setResponse] = useState('');
  const [sources, setSources] = useState([]);
  const [pdfImage, setPdfImage] = useState(null);

  const handleQuery = async () => {
    const res = await axios.post('http://localhost:5000/generate_response', { query });
    setResponse(res.data.response);
    setSources(res.data.sources);
  };

  const handleViewSource = async (source) => {
    if (source.includes('http://') || source.includes('https://')) {
      window.open(source, '_blank');
    } else if (source.includes('pdf')) {
      const pagenumber = source.split('_').pop();
      const filename = `data/docs/${source.split('_page')[0]}`;
      const res = await axios.get('http://localhost:5000/get_page_image', {
        params: { pdf_file_path: filename, page_num: pagenumber - 1 },
        responseType: 'arraybuffer',
      });
      const image = Buffer.from(res.data, 'binary').toString('base64');
      setPdfImage(`data:image/png;base64,${image}`);
    }
  };

  return (
    <div>
      <h1>Duke Radiology Resident Chatbot</h1>
      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="What questions can I help you with?"
      />
      <button onClick={handleQuery}>Submit</button>
      <div>
        <h2>Response</h2>
        <p>{response}</p>
        {sources.length > 0 && (
          <button onClick={() => handleViewSource(sources[0])}>
            View Source
          </button>
        )}
        {pdfImage && <img src={pdfImage} alt="PDF Page" />}
      </div>
    </div>
  );
}
