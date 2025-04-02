from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_from_directory
import os, uuid
from database import get_db_connection  
from models import User, Student, Teacher, Report, CoOpReport
from flask_cors import CORS 
from Chatbot import generate_response, insert_pdf_data

app = Flask(__name__)
CORS(app)  

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER  # upload to 'uploads' folder
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit
app.secret_key = 'your_secret_key'

''' Main page '''
@app.route('/', methods=['GET', 'POST'])
def login():
    error = "0"
    if request.method == 'POST':
        # เรียกใช้ authenticate() ของ User class
        user = User.authenticate(
            request.form['email'],
            request.form['password'],
            request.form['role']
        )
        
        if user:
            session['email'] = user.email
            session['role'] = user.role

            # ตรวจสอบว่า user เป็น instance ของคลาส Student หรือเป็นของคลาสลูกของ Student ไหม
            if isinstance(user, Student):
                session['student_id'] = user.student_id
                
                # Get student name
                conn = get_db_connection()
                cur = conn.cursor()

                cur.execute('SELECT au_name FROM author WHERE au_id = %s', (user.student_id,))
                session['studentName'] = cur.fetchone()[0]

                cur.close()
                conn.close()

                # test print data                
                print(f'Email : {session['email']}')
                print(f'ID : {session['student_id']}')
                print(f'Student Name : {session['studentName']}')
                print(f'Role : {session['role']}')
                # end test print data

                return redirect('/hpStudent')
            return redirect('/hpTeacher')
        else:
            error = "1"
    return render_template('Login_page.html',error = error)


''' Role is student '''
@app.route('/hpStudent')
def hpStudent():
    if 'email' not in session:
        return redirect('/')
    
    student = Student(session['email'], session['role'])
    projects = student.get_projects()
    
    conn = get_db_connection()
    cur = conn.cursor()

    # Check user added project yet
    cur.execute('SELECT report_id FROM author WHERE au_id = %s AND report_id IS NOT NULL', (session['student_id'],))
    check = "already have a project" if cur.fetchone() else "not have a project yet"
    
    query_getReport = 'SELECT * FROM report ORDER BY year DESC'
    cur.execute(query_getReport)
    result_getReport = cur.fetchall()
    
    cur.close()
    conn.close()
    
    return render_template('Homepage_student.html',
                         email=session['email'],
                         check=check,
                         projects = result_getReport)
    

''' Role is teacher '''
@app.route('/hpTeacher',methods=['GET'])
def hpTeacher():
    if 'email' not in session:
        return redirect('/')
    
    teacher = Teacher(session['email'], session['role'])
    projects = teacher.get_all_projects()

    return render_template('Homepage_teacher.html',
                         email=session['email'],
                         projects=projects)

''' Add Report '''
@app.route('/add_project', methods=['GET'])
def add_report_form():
    return render_template('AddProject_page.html', studentId=session.get('student_id', ''))


@app.route('/add_report', methods=['POST'])
def add_report():
    try:
        # Handle file upload
        pdf_file = request.files['path']
        filename = f"file_{uuid.uuid4()}.pdf"
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        pdf_file.save(pdf_path)

        # Create report type
        category = request.form['category']
        author_ids = [request.form[f'name_author{i}'] for i in range(1, int(request.form['numCreators'])+1)]
        
        if category == "co-op":
            report = CoOpReport(
                title = request.form['title'],
                user_email = session['email'],
                author_ids = author_ids
            )
        else:
            report = Report(
                title = request.form['title'],
                category = category,
                user_email = session['email'],
                author_ids = author_ids
            )

        # Save report
        if report.save(request.form, pdf_path):
           
            insert_pdf_data(request.form['title'], pdf_path)
            
            return jsonify({
                "status": "success",
                "message": "Report added successfully",
                "redirect": "/hpStudent"
            })
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400
    

''' Read detail for ReadProject_page '''   
@app.route('/read_project/<int:report_id>')
def read_project(report_id):
    conn = get_db_connection()
    cur = conn.cursor()

    query_reportDetail = '''
    SELECT * 
    FROM report
    NATURAL JOIN author
    NATURAL JOIN advisor
    NATURAL JOIN report_type
    WHERE report_id = %s
    '''
    cur.execute(query_reportDetail, (report_id,))
    report = cur.fetchall()
    formatted_typeData = ", ".join(item.strip("'") for item in report[0][15])
    cur.close()
    conn.close()

    # Test print
    print(report[0][9])
        
    if not report:
        return "Report not found", 404
    return render_template('ReadProject_page.html', report=report, formatted_typeData=formatted_typeData)

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.template_filter('basename')
def basename_filter(path):
    return os.path.basename(path)  # Extracts only the filename


''' Chatbot '''
@app.route('/chat', methods=['POST'])
def chat():
    try:
        question = request.json.get("message")
        if not question:
            return jsonify({"error": "No question provided"}), 400
        
        # เรียกใช้ generate_response จาก Chatbot.py
        answer = generate_response(question)
        
        return jsonify({"response": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/logout')
def logout():
    session.pop('email', None)
    session.pop('role', None)
    return redirect(url_for('login'))


if __name__ == '__main__':
    app.run(debug=True)
    
    


