from database import get_db_connection
from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer("BAAI/bge-m3")  # 1024-D vector
conn = get_db_connection()
cur = conn.cursor()

def query_postgresql(query_text):
    
    query_embedding = embedder.encode(query_text).tolist()
    sql_query = """
            WITH scored_docs AS (
                SELECT 
                    projects.report_title, 
                    embeddings.line_content, 
                    1 - (embeddings.line_content_vector <=> %s::vector) AS similarity_score
                FROM embeddings
                JOIN projects ON projects.report_id = embeddings.report_id
            )
            SELECT report_title, line_content, similarity_score
            FROM scored_docs
            WHERE similarity_score > 0.45
            ORDER BY similarity_score DESC;
    """ 
    
    cur.execute(sql_query, (query_embedding,))
    result = cur.fetchall()
      # กรองข้อมูลซ้ำ
    unique_results = []
    seen = set()
    for row in result:
        if row[1] not in seen:  # ตรวจสอบว่า line_content ซ้ำหรือไม่
            unique_results.append(row)
            seen.add(row[1])
    
    print("Filtered Query Result:", unique_results)  # Debug: ดูผลลัพธ์ที่กรองแล้ว
    return unique_results
    

def generate_response(query_text):
    retrieved_docs = query_postgresql(query_text)
    context = "\n".join([f"Title: {doc[0]}, Content: {doc[1]}" for doc in retrieved_docs])
    prompt = f"Answer based on this context:\n{context}\n\nQuestion: {query_text}"
    return prompt
  
query_postgresql("ฉันอยากทราบว่า รายชื่อสมาชิกกลุ่ม ของโปรเจคเรื่อง Risk Register มีใครบ้าง")