import mysql.connector
from faker import Faker
import random
from datetime import datetime, timedelta

# Set up Faker and database connection
fake = Faker()

from db import create_connection

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

# Generate and insert sample data
def insert_sample_data(num_records):
    connection = create_connection()
    cursor = connection.cursor()
    
    for _ in range(num_records):
        artwork_id = fake.uuid4()
        artist_id = random.randint(1, 100)  # Assuming you have 100 artists
        title = fake.sentence(nb_words=4)
        description = fake.text(max_nb_chars=200)
        image_url = fake.image_url()
        price = round(random.uniform(50, 5000), 2)
        availability = random.choice([True, False])
        created_date = (datetime.now() - timedelta(days=random.randint(0, 365))).strftime('%Y-%m-%d')

        query = """
        INSERT INTO Artwork (artwork_id, artist_id, title, description, image_url, price, availability, created_date)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        data = (artwork_id, artist_id, title, description, image_url, price, availability, created_date)
        try:
            cursor.execute(query, data)
        except mysql.connector.Error as e:
            print(f"Error: '{e}'")
    
    connection.commit()
    cursor.close()
    connection.close()
    print(f"{num_records} records inserted successfully.")

# Example usage
if __name__ == "__main__":
    num_records = 10000  # Adjust the number as needed
    insert_sample_data(num_records)
