import pymupdf
from database import get_db_connection
from sentence_transformers import SentenceTransformer
from pythainlp.tokenize import word_tokenize
from pythainlp.util import Trie
embedder = SentenceTransformer("BAAI/bge-m3")  # 1024-D vector

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


def insert_pdf_data(report_title, pdf_path):
    
    
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
    
insert_pdf_data("Workshop: Risk Register", "uploads/file_9f8460da-8673-47ba-a8a4-24d1c3d9b44b.pdf")
#print(extract_text_from_pdf("uploads/file_9f8460da-8673-47ba-a8a4-24d1c3d9b44b.pdf"))