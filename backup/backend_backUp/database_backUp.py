import psycopg2

def get_db_connection():
    return psycopg2.connect(
        database='DB',
        user='postgres',
        password='1234',
        port='5432'
    )