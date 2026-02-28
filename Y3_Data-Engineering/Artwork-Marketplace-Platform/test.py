import mysql.connector
import hashlib
import time
from db import execute_query, create_connection

# Function to execute a query
def execute_query(query, data=None):
    connection = create_connection()
    cursor = connection.cursor()
    try:
        cursor.execute(query, data)
        connection.commit()
        print("Query executed successfully")
    except mysql.connector.Error as e:
        print(f"Error: '{e}'")
    finally:
        cursor.close()
        connection.close()

# Function to fetch query results
def fetch_query(query, data=None):
    connection = create_connection()
    cursor = connection.cursor(dictionary=True)
    try:
        cursor.execute(query, data)
        results = cursor.fetchall()
        return results
    except mysql.connector.Error as e:
        print(f"Error: '{e}'")
        return []
    finally:
        cursor.close()
        connection.close()

# Hash function using common alphabets from roll numbers
def custom_hash(value, roll_number="B22AI045"):
    common_chars = ''.join(filter(str.isalpha, roll_number))
    hasher = hashlib.sha256()
    hasher.update((common_chars + value).encode('utf-8'))
    return hasher.hexdigest()

# Create clustering index
def create_clustering_index():
    query = """
    CREATE TABLE IF NOT EXISTS Artwork (
        artwork_id INT PRIMARY KEY,
        artist_id INT,
        title VARCHAR(255),
        description TEXT,
        image_url VARCHAR(255),
        price DECIMAL(10, 2),
        availability BOOLEAN,
        created_date DATE,
        INDEX(price)
    ) ENGINE=InnoDB;
    """
    execute_query(query)

# Create secondary index
def create_secondary_index():
    query = """
    CREATE INDEX idx_title ON Artwork(title);
    """
    execute_query(query)

# Compare storage and execution time
def test_indexing_performance():
    connection = create_connection()
    cursor = connection.cursor()
    
    # Measure time for clustering index query
    start_time = time.time()
    cursor.execute("SELECT * FROM Artwork WHERE price < 1000;")
    results = cursor.fetchall()
    clustering_time = time.time() - start_time

    print(f"Clustering Index Query Time: {clustering_time:.6f} seconds")

    # Measure time for secondary index query
    start_time = time.time()
    cursor.execute("SELECT * FROM Artwork WHERE title LIKE '%Sunset%';")
    results = cursor.fetchall()
    secondary_time = time.time() - start_time

    print(f"Secondary Index Query Time: {secondary_time:.6f} seconds")

    cursor.close()
    connection.close()

# Example usage
if __name__ == "__main__":
    # Design hash function
    artwork_id = "A001"
    hashed_id = custom_hash(artwork_id)
    print(f"Hashed ID for artwork {artwork_id}: {hashed_id}")

    # Create clustering and secondary indexes
    create_clustering_index()
    create_secondary_index()

    # Test and compare indexing performance
    test_indexing_performance()
