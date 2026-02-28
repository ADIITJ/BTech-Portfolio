import mysql.connector
from faker import Faker
import random
from datetime import date, timedelta

fake = Faker()

# Connect to MySQL
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="what is mysql",
    database="ArtmarketPlace"
)
cursor = conn.cursor()

# Insert fake data for Buyer
def insert_buyers(n):
    for _ in range(n):
        name = fake.name()
        email = fake.email()
        password = fake.password()
        cursor.execute(f"INSERT INTO Buyer (name, email, password) VALUES ('{name}', '{email}', '{password}')")

# Insert fake data for ArtistProfile
def insert_artists(n):
    for _ in range(n):
        name = fake.name()
        email = fake.email()
        password = fake.password()
        biography = fake.text()
        portfolio_link = fake.url()
        cursor.execute(f"INSERT INTO ArtistProfile (name, email, password, biography, portfolio_link) "
                       f"VALUES ('{name}', '{email}', '{password}', '{biography}', '{portfolio_link}')")

# Insert fake data for Artwork (covering multiple years and artwork types)
def insert_artwork(artist_ids):
    artwork_types = ["oil painting", "sculpture", "watercolor", "digital art", "acrylic painting", "charcoal drawing", "mixed media"]
    
    # Insert artwork from May 2014 to September 2024
    for artist_id in artist_ids:
        for year in range(2014, 2025):
            start_month = 5 if year == 2014 else 1
            end_month = 9 if year == 2024 else 12
            
            for month in range(start_month, end_month + 1):
                title = fake.word() + " Artwork"
                description = random.choice(artwork_types)
                image_url = fake.image_url()
                price = round(random.uniform(100, 5000), 2)
                availability = random.choice([True, False])
                created_date = date(year, month, random.choice(range(1, 28)))
                cursor.execute(f"INSERT INTO Artwork (artist_id, title, description, image_url, price, availability, created_date) "
                               f"VALUES ({artist_id}, '{title}', '{description}', '{image_url}', {price}, {availability}, '{created_date}')")

# Insert fake data for Orders and OrderItems (ensuring valid buyer and artwork ids)
def insert_orders(buyer_ids, artwork_ids):
    for buyer_id in buyer_ids:
        for _ in range(random.randint(1, 5)):  # Some buyers made multiple purchases
            total_amount = round(random.uniform(100, 5000), 2)
            order_date = fake.date_between(start_date=date(2014, 5, 1), end_date=date(2024, 9, 15))
            order_time = fake.time()
            status = random.choice(["Delivered", "Processing", "Cancelled"])
            cursor.execute(f"INSERT INTO `Order` (buyer_id, total_amount, order_date, order_time, status) "
                           f"VALUES ({buyer_id}, {total_amount}, '{order_date}', '{order_time}', '{status}')")
            order_id = cursor.lastrowid

            # Insert order items for existing artworks
            for artwork_id in random.sample(artwork_ids, random.randint(1, 3)):
                quantity = random.randint(1, 3)
                price = round(random.uniform(100, 5000), 2)
                cursor.execute(f"INSERT INTO OrderItems (order_id, artwork_id, quantity, price) "
                               f"VALUES ({order_id}, {artwork_id}, {quantity}, {price})")

# Insert fake data for Reviews (ensuring valid buyer and artwork ids)
def insert_reviews(buyer_ids, artwork_ids):
    for _ in range(random.randint(20, 50)):  # Insert multiple reviews
        artwork_id = random.choice(artwork_ids)
        buyer_id = random.choice(buyer_ids)
        rating = random.randint(1, 5)
        comment = fake.sentence()
        review_date = fake.date_between(start_date=date(2014, 5, 1), end_date=date(2024, 9, 15))
        review_time = fake.time()
        cursor.execute(f"INSERT INTO Review (artwork_id, user_id, rating, comment, review_date, review_time) "
                       f"VALUES ({artwork_id}, {buyer_id}, {rating}, '{comment}', '{review_date}', '{review_time}')")

# Function to get artist_ids, buyer_ids, and artwork_ids for data insertion
def get_ids(table_name, id_column):
    cursor.execute(f"SELECT {id_column} FROM {table_name}")
    return [row[0] for row in cursor.fetchall()]

# Insert Buyers, Artists, Artworks, Orders, OrderItems, and Reviews
def populate_database():
    insert_buyers(20)  # Insert 20 buyers
    insert_artists(10)  # Insert 10 artists

    # Get artist_ids and buyer_ids
    artist_ids = get_ids('ArtistProfile', 'artist_id')
    buyer_ids = get_ids('Buyer', 'buyer_id')

    insert_artwork(artist_ids)  # Insert artworks for each artist across multiple years
    artwork_ids = get_ids('Artwork', 'artwork_id')  # Get artwork_ids

    insert_orders(buyer_ids, artwork_ids)  # Insert orders and order items
    insert_reviews(buyer_ids, artwork_ids)  # Insert reviews for artworks

# Insert fake data
populate_database()

# Commit changes and close connection
conn.commit()
cursor.close()
conn.close()

print("Data inserted successfully!")
