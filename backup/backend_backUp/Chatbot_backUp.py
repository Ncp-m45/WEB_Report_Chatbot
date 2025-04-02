import pymupdf, ollama 
from sentence_transformers import SentenceTransformer
from database import get_db_connection


conn = get_db_connection()
cur = conn.cursor()

embedder = SentenceTransformer("BAAI/bge-m3")  # 1024-D vector

def extract_text_from_pdf(pdf_path, chunk_size=500):
    doc = pymupdf.open(pdf_path)
    full_text = "\n".join(page.get_text("text") for page in doc)
    
    # แบ่งข้อความเป็น chunks
    chunks = [full_text[i:i + chunk_size] for i in range(0, len(full_text), chunk_size)]
    return chunks


def insert_pdf_data(report_title, pdf_path):
    emb_project_title = embedder.encode(report_title).tolist()
    
    cur.execute("SELECT report_id FROM report WHERE path = %s", (pdf_path,))
    report_id = cur.fetchone()[0]
    
    cur.execute(
        "INSERT INTO projects (report_id, report_title, emb_project_title, pdf_path) VALUES (%s, %s, %s, %s)",
        (report_id, report_title, emb_project_title, pdf_path)
    ) 
    conn.commit()

    chunks = extract_text_from_pdf(pdf_path)
    for chunk_id, chunk_text in enumerate(chunks):
        embedding = embedder.encode(chunk_text).tolist()
        cur.execute(
            "INSERT INTO embeddings (report_id, chunk_id, content, vector) VALUES (%s, %s, %s, %s::vector)",
            (report_id, chunk_id, chunk_text, embedding)
        )
    conn.commit()
    
def query_postgresql(query_text, k=5):
    query_embedding = embedder.encode(query_text).tolist()
    
    sql_query = """
        SELECT projects.report_title, projects.pdf_path, embeddings.content, embeddings.vector <=> %s::vector AS similarity_score
        FROM embeddings
        JOIN projects ON projects.report_id = embeddings.report_id
        ORDER BY similarity_score ASC
        LIMIT %s;
    """
    cur.execute(sql_query, (query_embedding, k))
    result = cur.fetchall()
    return result
    
conversation_history = []  
def generate_response(query_text):
    global conversation_history
    try:
        retrieved_docs = query_postgresql(query_text)
        context = "\n".join([f"Title: {doc[0]}, File: {doc[1]}, Content: {doc[2]}" for doc in retrieved_docs])
        
        prompt = f"Answer based on this context:\n{context}\n\nQuestion: {query_text}"
        
        conversation_history.append({"role": "user", "content": query_text})
        
        response = ollama.chat(
            model="llama3.2",  
            messages=[
                {"role": "system", "content": "Answer based on the context"},
                *conversation_history,
                {"role": "user", "content": prompt}
            ]
        )
        
        conversation_history.append({"role": "assistant", "content": response["message"]["content"]})
        
        return response['message']['content']
    except Exception as e:
        return f"Error: {str(e)}"



#insert_pdf_data("Workshop Risk Register", "data/Workshop Risk Register.pdf")  

