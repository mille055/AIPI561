"use client"; // This marks the component as a client component

import { useState } from 'react';
import axios from 'axios';
import Image from 'next/image';

export default function Home() {
  const [message, setMessage] = useState('');
  const [response, setResponse] = useState('');
  const [sources, setSources] = useState([]);
  const [imageSrc, setImageSrc] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const res = await axios.post('http://127.0.0.1:5000/api/chat', { message });
      setResponse(res.data.response);
      setSources(res.data.sources);
    } catch (error) {
      console.error(error);
    }
  };

  const handleViewSource = async (source) => {
    if (source.includes('http')) {
      window.open(source, '_blank');
    } else if (source.includes('pdf')) {
      try {
        const res = await axios.post('http://127.0.0.1:5000/api/get_pdf_page', { source });
        const imageBlob = new Blob([res.data], { type: 'image/png' });
        const imageUrl = URL.createObjectURL(imageBlob);
        setImageSrc(imageUrl);
      } catch (error) {
        console.error(error);
      }
    }
  };

  return (
    <div>
      <h1>Duke Radiology Resident Chatbot</h1>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder="What questions can I help you with?"
        />
        <button type="submit">Send</button>
      </form>
      {response && (
        <div>
          <p>{response}</p>
          {sources.length > 0 && (
            <div>
              {sources.map((source, index) => (
                <button
                  key={index}
                  onClick={() => handleViewSource(source)}
                  style={{
                    backgroundColor: '#3F7D7B',
                    color: 'white',
                    padding: '10px 24px',
                    margin: '10px',
                    border: 'none',
                    borderRadius: '12px',
                    cursor: 'pointer'
                  }}
                >
                  View Source {index + 1}
                </button>
              ))}
            </div>
          )}
          {imageSrc && <img src={imageSrc} alt="PDF Page" />}
        </div>
      )}
    </div>
  );
}
