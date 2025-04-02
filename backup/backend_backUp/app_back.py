from flask import Flask, render_template, request, redirect, url_for, session,jsonify,send_from_directory
import psycopg2, os, uuid

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER  # upload to 'uploads' folder
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit
app.secret_key = 'your_secret_key'

def get_db_connection():
    return psycopg2.connect(database='DB', user='postgres', password='1234',  port='5432')

''' Main page '''
@app.route('/', methods=['GET', 'POST'])
def login():
    error = "0"
    if request.method == 'POST':
        role = request.form['role']
        email = request.form['email']
        password = request.form['password']
        
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('SELECT user_email, role FROM "user" WHERE user_email = %s AND password = %s AND role = %s', (email, password, role))
        user = cur.fetchone()
        
        if user:
            session['email'] = user[0]
            session['role'] = user[1]
            if (session['role'] == "student") :
                idStudent = user[0].split("@")[0]
                session['studentId'] = idStudent
                cur.execute('SELECT au_name FROM author WHERE au_id = %s',(idStudent,))
                studentName = cur.fetchone()
                cur.close()
                conn.close()
                session['studentName'] = studentName[0]

                # test print data                
                print(f'Email : {session['email']}')
                print(f'ID : {session['studentId']}')
                print(f'Student Name : {session['studentName']}')
                print(f'Role : {session['role']}')
                # end test print data

                return redirect('/hpStudent')
            else :
                return redirect('/hpTeacher')
        else:
            error = "1"
    return render_template('Login_page.html',error = error)

''' Role is student '''
@app.route('/hpStudent',methods=['GET'])
def hpStudent():
    if 'email' in session:
        check = "not have a project yet"

        conn = get_db_connection()
        cur = conn.cursor()

        query_checkAuthor = 'SELECT report_id FROM author WHERE au_id = %s'
        cur.execute(query_checkAuthor,(session['studentId'],))
        result_checkAuthor = cur.fetchone()[0]

        query_getReport = 'SELECT * FROM report'
        cur.execute(query_getReport)
        result_getReport = cur.fetchall()

        if result_checkAuthor :
            check = "already have a project"

        # test print
        print(result_getReport)

        return render_template('Homepage_student.html', email=session['email'], check = check, projects = result_getReport)
    return redirect('/')

''' Role is teacher '''
@app.route('/hpTeacher',methods=['GET'])
def hpTeacher():
    if 'email' in session:
        conn = get_db_connection()
        cur = conn.cursor()

        query_getReport = 'SELECT * FROM report'
        cur.execute(query_getReport)
        result_getReport = cur.fetchall()

        return render_template('Homepage_teacher.html', email=session['email'], projects = result_getReport)
    return redirect('/')

@app.route('/logout')
def logout():
    session.pop('email', None)
    session.pop('role', None)
    return redirect(url_for('login'))

''' Add Report '''
@app.route('/add_report', methods=['GET','POST'])
def add_report():
    if request.method == 'POST' :
        title = request.form['title']  

        pdf_file = request.files['path']

        filename = f"file_{uuid.uuid4()}.pdf"
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        pdf_file.save(pdf_path)

        category=request.form['category']
        position=request.form.get('position')
        org=request.form.get('org')
        type_org=request.form.get('type_org')

        selected_reports_types = request.form.getlist('report_types')
        other_input = request.form['other_input']

        if other_input: # Check if Other is selected
            index = selected_reports_types.index('Other')
            selected_reports_types[index] = other_input
    
        list_reports_types = '{' + ', '.join(f"'{list}'" for list in selected_reports_types) + '}'

        intro=request.form['intro']

        num_creators = int(request.form['numCreators'])
        author_id = [request.form[f'name_author{i}']
                         for i in range(1, num_creators+1)]
        authors_array_insert = '{' + ', '.join(f"'{a_Id}'" for a_Id in author_id) + '}'
        
        name_advisor=request.form['name_advisor']
        year=request.form['year']
        
        # print retrive data
        print("Title " + title) 
        print(pdf_file)
        print(pdf_path)
        print("Category " + category) 
        print("Position of your work : " + position)
        print("Oraganization : " + org)
        print("Type of Oraganization : " + type_org)
        print(selected_reports_types)
        print("Other input : " + other_input)
        print("intro : " + intro)
        print("number of creator : " + str(num_creators))
        print(authors_array_insert)
        print("Create Year : " + year)

        conn = get_db_connection()
        cur = conn.cursor()
        query = f"SELECT au_id FROM author WHERE au_id IN ({','.join(['%s'] * len(author_id))})" # Ex. IN ('65050315', '65050113', '65050107')
        cur.execute(query, author_id)
        result_existing_au_ids = cur.fetchall()
        existing_ids = {row[0] for row in result_existing_au_ids}

        # Check author_id were db yet
        if set(author_id).issubset(existing_ids): 
            print("All IDs exist in the database.")

            # Check data of advisor
            cur.execute('SELECT advisor_email, advisor_name FROM advisor WHERE advisor_email = %s ', (name_advisor,))
            advisor = cur.fetchone()        

            if advisor :
                advisor_email = advisor[0]
                SQL_insertReport = ''' 
                INSERT INTO report (title, intro, year, category, org, type_org, position, path, user_email, advisor_email, author)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                RETURNING report_id
                '''
                
                # Insert report
                cur.execute(SQL_insertReport, (title,intro,year,category,org,type_org,position,pdf_path,
                                            session['email'],advisor_email,authors_array_insert))
                report_id = cur.fetchone()[0]
                conn.commit()           
                print(report_id)
                
                # Insert report type if user input 
                SQL_insertReportType = 'INSERT INTO report_type VALUES (%s,%s)'
                cur.execute(SQL_insertReportType,( report_id, list_reports_types ))
                conn.commit()

                SQL_updateAuthor = f'UPDATE author SET report_id = %s WHERE au_id IN ({','.join(['%s'] * len(author_id))})'
                cur.execute(SQL_updateAuthor,[report_id] + author_id)
                conn.commit()

                cur.close()
                conn.close()

                print('Insert successful')
                return jsonify({"status": "success", "message": "Report added successfully", "redirect": "/hpTeacher"})
            else :
                cur.close()
                conn.close()
                return jsonify({"status": "error", "message": "Advisor not found"}), 400

        else:
            cur.close()
            conn.close()
            id_missing = f"Some IDs are missing: {set(author_id) - existing_ids}"
            print(id_missing)
            return jsonify({"status": "error", "message": id_missing}), 400
        


    return render_template('AddProject_page.html', email=session['email'], studentId=session['studentId'],)

''' Read detail for ReadProject_page '''   
@app.route('/read_project/<int:report_id>')
def read_project(report_id):
    conn = get_db_connection()
    cur = conn.cursor()

    qury_reportDetail = '''
    SELECT * 
    FROM report
    NATURAL JOIN author
    NATURAL JOIN advisor
    NATURAL JOIN report_type
    WHERE report_id = %s
    '''
    cur.execute(qury_reportDetail, (report_id,))
    report = cur.fetchall()
    formatted_typeData = ", ".join(item.strip("'") for item in report[0][15])
    cur.close()
    conn.close()

    print(report[0][9])
    # projects = [
    # (1, "Project 1", "Summary 1", "Data A1", "Data B1", "special-project", 'Data Analyst', 'Seagate', 'Tech', "Data G1", "Data H1", "Data I1", "Data J1", "Data K1", "Data L1"),
    # (2, "Project 2", "Summary 2", "Data A2", "Data B2", "Data C2", "Data D2", "Data E2", "Data F2", "Data G2", "Data H2", "Data I2", "Data J2", "Data K2", "Data L2"),
    # (3, "Project 3", "Summary 3", "Data A3", "Data B3", "Data C3", "Data D3", "Data E3", "Data F3", "Data G3", "Data H3", "Data I3", "Data J3", "Data K3", "Data L3"),
    # ]
        
    if not report:
        return "Report not found", 404
    return render_template('ReadProject_page.html', report=report, formatted_typeData=formatted_typeData)

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.template_filter('basename')
def basename_filter(path):
    return os.path.basename(path)  # Extracts only the filename

if __name__ == '__main__':
    app.run(debug=True)
