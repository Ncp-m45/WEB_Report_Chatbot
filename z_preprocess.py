from pythainlp.tokenize import word_tokenize
from pythainlp.util import Trie
from database import get_db_connection
import pymupdf
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
    num_word = len(tokens)//4
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



# doc = pymupdf.open("uploads/file_9f8460da-8673-47ba-a8a4-24d1c3d9b44b.pdf")

# # วนลูปผ่านทุกหน้าของ PDF และเก็บผลลัพธ์ในลิสต์
# all_pages_tokens = []
# for page_num in range(len(doc)):
#     page_text = doc[page_num].get_text("text")
#     page_tokens = preprocess_thai_text(page_text)
#     all_pages_tokens.append(page_tokens)
    

# แสดงผลลัพธ์
print(extract_text_from_pdf("uploads/file_9f8460da-8673-47ba-a8a4-24d1c3d9b44b.pdf"))
# print(test)
# print(len(test))
# print(len(test[0]))