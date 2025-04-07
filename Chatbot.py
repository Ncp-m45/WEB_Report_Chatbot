import pymupdf, ollama 
from sentence_transformers import SentenceTransformer
from pythainlp.tokenize import word_tokenize
from pythainlp.util import Trie
from database import get_db_connection

conn = get_db_connection()
cur = conn.cursor()

def preprocess_thai_text(text):
    conn = get_db_connection()
    cur = conn.cursor()
    
    sql_query = """
            SELECT au_name
            FROM author
        """
    cur.execute(sql_query)
    db_person_names = [row[0] for row in cur.fetchall()]  
    cur.close()
    conn.close()
    
    for i in text:
        if i == "\n":
            text = text.replace(i, "")
    
    for i in range(len(db_person_names)):
        for unwanted in ["นาย", "นางสาว"]:
            db_person_names[i] = db_person_names[i].replace(unwanted, "")
    
    dict_word = [word for word in word_tokenize(text, engine="newmm")]
    dict_word.extend(db_person_names)
    
    custom_dict = Trie(dict_word)
    tokens = word_tokenize(text, custom_dict=custom_dict, engine="newmm")
    num_word = len(tokens)//2
    list_word_Tokens = ["".join(tokens[i:i + num_word]) for i in range(0, len(tokens), num_word+1)]
    
    return list_word_Tokens

def extract_text_from_pdf(pdf_path):
    doc = pymupdf.open(pdf_path)
    all_pages_tokens = []
    for page_num in range(len(doc)):
        page_text = doc[page_num].get_text("text")
        page_tokens = preprocess_thai_text(page_text)
        all_pages_tokens.append(page_tokens)
        
    return all_pages_tokens


embedder = SentenceTransformer("BAAI/bge-m3")  # 1024-D vector
def insert_pdf_data(report_title, pdf_path):
    
    cur.execute("SELECT report_id FROM report WHERE path = %s", (pdf_path,))
    # report_id = cur.fetchone()[0]
    report_id = 66

    chunks = extract_text_from_pdf(pdf_path)
    # print(chunks)
    # print(chunks[0])
    # print(len(chunks[0][0]))
    page_id = 0
    for chunk_text in chunks:
        paragraph_id = 0
        for paragraph_content in chunk_text:
            paragraph_embedding = embedder.encode(paragraph_content).tolist()
            cur.execute(
                "INSERT INTO embeddings (report_id, page_id ,  line_id, line_content, line_content_vector) VALUES (%s, %s,%s, %s, %s::vector)",
                (report_id, page_id,  paragraph_id, paragraph_content, paragraph_embedding)
            )
            print(f"++++++++++++++{paragraph_id}+++++++++++++\n",paragraph_content)
            paragraph_id += 1
        page_id += 1    
    conn.commit()
    
    emb_project_title = embedder.encode(report_title).tolist()
    cur.execute(
        "INSERT INTO projects (report_id, report_title, emb_project_title, pdf_path) VALUES (%s, %s, %s, %s)",
        (report_id, report_title, emb_project_title, pdf_path)
    ) 
    conn.commit()

# insert_pdf_data("Workshop: Risk Register", "uploads/file_9f8460da-8673-47ba-a8a4-24d1c3d9b44b.pdf")
    
    
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
            WHERE similarity_score > 0.4
            ORDER BY similarity_score DESC;
    """ 
    
    cur.execute(sql_query, (query_embedding,))
    result = cur.fetchall()
      # กรองข้อมูลซ้ำ
    unique_results = []
    seen = set()
    for row in result:
        if row[1] not in seen:  
            unique_results.append(row)
            seen.add(row[1])
    
    return unique_results
    
conversation_history = []  
def generate_response(query_text):
    global conversation_history
    try:
        retrieved_docs = query_postgresql(query_text)
        context = "\n".join([f"Title: {doc[0]}, Content: {doc[1]}" for doc in retrieved_docs])
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

# print(generate_response("ฉันอยากทราบว่ารายชื่อสมาชิกกลุ่มของโปรเจคเรื่อง Risk Register มีใครบ้าง"))