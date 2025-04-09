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
    report_id = cur.fetchone()[0]
    
    chunks = extract_text_from_pdf(pdf_path)
    page_id = 0
    for chunk_text in chunks:
        paragraph_id = 0
        for paragraph_content in chunk_text:
            paragraph_embedding = embedder.encode(paragraph_content).tolist()
            cur.execute(
                "INSERT INTO embeddings (report_id, page_id ,  line_id, line_content, line_content_vector) VALUES (%s, %s,%s, %s, %s::vector)",
                (report_id, page_id,  paragraph_id, paragraph_content, paragraph_embedding)
            )
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
    
    
# def query_postgresql(query_text):
#     query_embedding = embedder.encode(query_text).tolist()
#     sql_query = """
#             WITH scored_docs AS (
#                 SELECT 
#                     projects.report_title, 
#                     embeddings.line_content, 
#                     1 - (embeddings.line_content_vector <=> %s::vector) AS similarity_score
#                 FROM embeddings
#                 JOIN projects ON projects.report_id = embeddings.report_id
#             )
#             SELECT report_title, line_content, similarity_score
#             FROM scored_docs
#             WHERE similarity_score > 0.4
#             ORDER BY similarity_score DESC;
#     """ 
    
#     cur.execute(sql_query, (query_embedding,))
#     result = cur.fetchall()
#       # กรองข้อมูลซ้ำ
#     unique_results = []
#     seen = set()
#     for row in result:
#         if row[1] not in seen:  
#             unique_results.append(row)
#             seen.add(row[1])
    
#     return unique_results
    


def cosine_similarity(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm_a = sum(a ** 2 for a in vec1) ** 0.5
    norm_b = sum(b ** 2 for b in vec2) ** 0.5
    return dot_product / (norm_a * norm_b) if norm_a and norm_b else 0.0

import json

def weighted_rerank(query_text, title_weight=0.3, content_weight=0.7):
    query_embedding = embedder.encode(query_text).tolist()
    sql_query = """
            SELECT report_title, line_content, line_content_vector
            FROM embeddings
            JOIN projects ON projects.report_id = embeddings.report_id
    """ 

    cur.execute(sql_query)
    result = cur.fetchall()
   
    reranked_results = []
    for row in result:
        try:
            vector = json.loads(row[2]) if isinstance(row[2], str) else row[2]
            content_similarity = cosine_similarity(query_embedding, vector)
            
            # เพิ่มคะแนนความเกี่ยวข้องของชื่อโปรเจค
            title_similarity = cosine_similarity(query_embedding, embedder.encode(row[0]).tolist())
            
            # คำนวณคะแนนรวมแบบถ่วงน้ำหนัก
            weighted_score = (title_weight * title_similarity) + (content_weight * content_similarity)
            if weighted_score > 0.5:
                reranked_results.append((row[0], row[1], weighted_score))
        except Exception as e:
            print(f"Error processing row: {row}, Error: {e}")
    unique_reranked_results = []
    seen = set()
    for row in reranked_results:
        if row[1] not in seen:  
            unique_reranked_results.append(row)
            seen.add(row[1])
    return sorted(unique_reranked_results, key=lambda x: x[2], reverse=True)

# data = weighted_rerank("RFM Model คืออะไร")
# for row in data:
#     print(f"Title: {row[0]}\nContent: {row[1]}\nSimilarity_score: {row[2]}\n\n")


conversation_history = []  
def generate_response(query_text):
    global conversation_history
    try:
        retrieved_docs = weighted_rerank(query_text)
        context = "\n".join([f"Title: {row[0]}\nContent: {row[1]}\n\n" for row in retrieved_docs])
        prompt = f"Answer based on this context:\n{context}\n\nQuestion: {query_text}"
        # return prompt
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

# def ROUGE_score(reference, candidate):
#     reference = reference.split()
#     candidate = candidate.split()
    
#     reference_set = set(reference)
#     candidate_set = set(candidate)
    
#     overlap = len(reference_set.intersection(candidate_set))
    
#     if len(reference_set) == 0:
#         return 0.0
    
#     return overlap / len(reference_set)



# def ROUGE_score_full(reference, candidate):
#     reference = reference.split()
#     candidate = candidate.split()
    
#     reference_set = set(reference)
#     candidate_set = set(candidate)
    
#     overlap = len(reference_set.intersection(candidate_set))
    
#     recall = overlap / len(reference_set) if reference_set else 0.0
#     precision = overlap / len(candidate_set) if candidate_set else 0.0
#     f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    
#     return {
#         'ROUGE-1 Precision': precision,
#         'ROUGE-1 Recall': recall,
#         'ROUGE-1 F1': f1
#     }

# question = "สมาชิกกลุ่มของโปรเจคเรื่อง Risk Register มีใครบ้าง"
# generate = generate_response(question)
# reference = """รายชื่อสมาชิกกลุ่ม 
# 1. กัณฐภรณ์ พิมพ์ภาค  65050054 
# 2. ชัญญา สิงห์ทองคำ 65050195 
# 3. ณัชพล หมื่นศรีชัย  65050253 
# 4. ธิรดา ดาราฉาย 65050409 
# 5. ปุณยวัจน์ แสงรัตนายนต์ 65050565 
# 6.ศีลกุล สุขศีล 65050878 
# 7. สุกันยาภรณ์  ฤดีชุติพร 65050916 
# 8. สุวภัชร ชุ่มชวย 65050954 
# 9. ฐณะวัฒน์ ศรีอัครโชตน์ 64050402"""
# print(f"คำตอบจริง (reference) : \n{reference}\n\nคำตอบที่โมเดลสร้างขึ้น (candidate) : \n{generate}\n\n {ROUGE_score_full(reference, generate)}")  


