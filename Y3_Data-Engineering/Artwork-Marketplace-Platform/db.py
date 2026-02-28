# db.py
import mysql.connector
from mysql.connector import Error

def create_connection():
    try:
        connection = mysql.connector.connect(
            host='localhost',
            database='ArtmarketPlace',
            user='root',
            password='what is mysql'
        )
        if connection.is_connected():
            print("Connection successful")
        return connection
    except Error as e:
        print(f"Error: '{e}'")
        return None

def execute_query(query, data=None):
    connection = create_connection()
    cursor = connection.cursor()
    try:
        cursor.execute(query, data)
        connection.commit()
        print("Query executed successfully")
    except Error as e:
        print(f"Error: '{e}'")
    finally:
        cursor.close()
        connection.close()

def fetch_query(query, data=None):
    connection = create_connection()
    cursor = connection.cursor(dictionary=True)
    try:
        cursor.execute(query, data)
        results = cursor.fetchall()
        return results
    except Error as e:
        print(f"Error: '{e}'")
        return []
    finally:
        cursor.close()
        connection.close()
