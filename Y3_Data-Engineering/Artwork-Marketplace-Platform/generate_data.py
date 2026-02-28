import mysql.connector

# Connect to MySQL
conn = mysql.connector.connect(
    host="localhost",
    user="root", 
    password="what is mysql",  
    database="ArtmarketPlace" 
)
cursor = conn.cursor()

# Function to execute SQL queries
def execute_query(query):
    try:
        cursor.execute(query)
        conn.commit()
        print(f"Query executed successfully: {query}")
    except Exception as e:
        print(f"Error: {e}")
        conn.rollback()

# SQL query to create the Buyer table
create_buyer_table = """
CREATE TABLE IF NOT EXISTS Buyer (
    buyer_id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(100),
    password VARCHAR(100)
);
"""

# SQL query to create the ArtistProfile table
create_artist_table = """
CREATE TABLE IF NOT EXISTS ArtistProfile (
    artist_id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(100),
    password VARCHAR(100),
    biography TEXT,
    portfolio_link VARCHAR(255)
);
"""

# SQL query to create the Artwork table
create_artwork_table = """
CREATE TABLE IF NOT EXISTS Artwork (
    artwork_id INT AUTO_INCREMENT PRIMARY KEY,
    artist_id INT,
    title VARCHAR(255),
    description TEXT,
    image_url VARCHAR(255),
    price DECIMAL(10, 2),
    availability BOOLEAN,
    created_date DATE,
    FOREIGN KEY (artist_id) REFERENCES ArtistProfile(artist_id)
);
"""

# SQL query to create the Order table
create_order_table = """
CREATE TABLE IF NOT EXISTS `Order` (
    order_id INT AUTO_INCREMENT PRIMARY KEY,
    buyer_id INT,
    total_amount DECIMAL(10, 2),
    order_date DATE,
    order_time TIME,
    status VARCHAR(50),
    FOREIGN KEY (buyer_id) REFERENCES Buyer(buyer_id)
);
"""

# SQL query to create the OrderItems table
create_order_items_table = """
CREATE TABLE IF NOT EXISTS OrderItems (
    order_item_id INT AUTO_INCREMENT PRIMARY KEY,
    order_id INT,
    artwork_id INT,
    quantity INT,
    price DECIMAL(10, 2),
    FOREIGN KEY (order_id) REFERENCES `Order`(order_id),
    FOREIGN KEY (artwork_id) REFERENCES Artwork(artwork_id)
);
"""

# SQL query to create the Review table
create_review_table = """
CREATE TABLE IF NOT EXISTS Review (
    review_id INT AUTO_INCREMENT PRIMARY KEY,
    artwork_id INT,
    user_id INT,
    rating INT,
    comment TEXT,
    review_date DATE,
    review_time TIME,
    FOREIGN KEY (artwork_id) REFERENCES Artwork(artwork_id),
    FOREIGN KEY (user_id) REFERENCES Buyer(buyer_id)
);
"""

# Executing all queries to create tables
execute_query(create_buyer_table)
execute_query(create_artist_table)
execute_query(create_artwork_table)
execute_query(create_order_table)
execute_query(create_order_items_table)
execute_query(create_review_table)

# Closing the connection
cursor.close()
conn.close()

print("All tables created successfully!")
