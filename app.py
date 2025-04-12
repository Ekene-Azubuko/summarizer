import yt_dlp 
import assemblyai as aai 
import numpy as np
import os
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify
from audiologic import save_audio
from dotenv import load_dotenv

app = Flask(__name__)

load_dotenv()
client = OpenAI(
    api_key = os.getenv("OPENAI_API_KEY")
)

aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY") 
transcriber = aai.Transcriber()

def chunk_transcript(transcript, chunk_size=500):
    words = transcript.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=model).data[0].embedding

def retrieve_relevant_chunks(query, embeddings, chunks, top_k=3):
    query_embedding = get_embedding(query)
    similarities = [cosine_similarity([query_embedding], [emb])[0][0] for emb in embeddings]
    top_indices = np.argsort(similarities)[-top_k:]
    # Retrieve top_k chunks (you might sort them for context continuity)
    retrieved_chunks = [chunks[i] for i in top_indices]
    return "\n".join(retrieved_chunks)

def answer_question(query, embeddings, chunks):
    context = retrieve_relevant_chunks(query, embeddings, chunks)
    prompt = f"""
    You are an assistant that answers questions exclusively using the provided context from a YouTube video transcript.
    Context:
    {context}

    If the transcript does not provide the requested information, please respond: "The transcript does not provide that detail."

    Question: {query}
    """
    response = client.responses.create(
        model="gpt-4",
        input=[{"role": "user", "content": prompt}]
    )
    answer = response.output_text
    return answer

# --- Flask API Route ---
@app.route('/', methods=['GET'])
def hello():
    return "Hello, this the summarizer API"

@app.route('/api/file-url', methods=['POST'])
def get_transcript():
    """
    This endpoint gets the transcript and posts it ot the databse
    """
    
    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({"error": "Missing 'url' parameter in JSON request"}), 400
    URLS = [data['url']]
    try:
        res, status = save_audio(URLS, "team-3ws")
        if status == 200:
            res_data = res.get_json()
            key = res_data.get('key')
            url = f"https://team-3ws.s3.amazonaws.com/{key}"
            transcript = transcriber.transcribe(url) 
            print(transcript)
            return jsonify({"transcript": transcript.text}), 200
        else:
            return res
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/api/answer', methods = ['POST'])
def api_answer():
    """
    This endpoint expects a JSON payload with keys 'query', 'transcript' containing the question.
    Example JSON payload:
    {
       "query": "What's the battery capacity?",
       "transcript: Long text...."
    }
    """
    data = request.get_json()
    if not data or 'query' not in data or 'transcript' not in data:
        return jsonify({"error": "Missing parameters in JSON request"}), 400
    
    transcript = data['transcript']
    query = data['query']
    try:
        chunks = chunk_transcript(transcript)
        embeddings = [get_embedding(chunk) for chunk in chunks]
        answer = answer_question(query, embeddings, chunks)
        return jsonify({"answer": answer}), 200 
    except Exception as e:
        return jsonify({"error": str(e)}), 500
        
    
    
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host="0.0.0.0", port=port)
    