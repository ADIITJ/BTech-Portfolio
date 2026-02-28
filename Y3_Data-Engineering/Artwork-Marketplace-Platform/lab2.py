import mysql.connector
import time

def execute_query(cursor, query):
    start_time = time.time()  # Start the clock
    
    cursor.execute(query) 
    
    end_time = time.time()  # Stop the clock
    execution_time = end_time - start_time
    print(f"Query executed successfully. Execution time: {execution_time} seconds\n")
    return execution_time


def main():

    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="password",
        database="ArtmarketPlace"
    )
    cursor = conn.cursor(buffered=True)
    
    queries = [
        # Query 1
        """
        SELECT a.name AS artist_name, aw.title AS artwork
        FROM ArtistProfile a
        JOIN Artwork aw ON a.artist_id = aw.artist_id
        WHERE YEAR(aw.created_date) = 2023
        GROUP BY a.name, aw.title
        HAVING COUNT(DISTINCT MONTH(aw.created_date)) = 12;
        """,
        
        # Query 2
        """
        SELECT DISTINCT a.name
        FROM ArtistProfile a
        JOIN Artwork aw ON a.artist_id = aw.artist_id
        WHERE aw.description LIKE '%sculpture%';
        """,
        
        # Query 3
        """
        SELECT a.*
        FROM ArtistProfile a
        LEFT JOIN Artwork aw ON a.artist_id = aw.artist_id
        WHERE aw.artwork_id IS NULL;
        """,
        
        # Query 4
        """
        SELECT DISTINCT b.name
        FROM Buyer b
        JOIN `Order` o ON b.buyer_id = o.buyer_id
        JOIN OrderItems oi ON o.order_id = oi.order_id
        JOIN Artwork aw ON oi.artwork_id = aw.artwork_id
        WHERE aw.description LIKE '%oil painting%'
        AND YEAR(o.order_date) = 2022;
        """,
        
        # Query 5
        """
        SELECT DISTINCT ap.*
        FROM ArtistProfile ap
        JOIN Artwork aw ON ap.artist_id = aw.artist_id
        JOIN OrderItems oi ON aw.artwork_id = oi.artwork_id
        JOIN `Order` o ON oi.order_id = o.order_id
        JOIN Buyer b ON o.buyer_id = b.buyer_id
        WHERE aw.description LIKE '%oil painting%'
        AND YEAR(o.order_date) = 2022;
        """,
        
        # Query 6
        """
        SELECT b.*
        FROM Buyer b
        LEFT JOIN `Order` o ON b.buyer_id = o.buyer_id
        WHERE o.order_id IS NULL;
        """
    ]
    
    total_time = 0
    for i, query in enumerate(queries, 1):
        print(f"Running Query {i}:")
        execution_time = execute_query(cursor, query)
        total_time += execution_time
    
    print(f"Total execution time for all queries: {total_time} seconds")
    
    cursor.close()
    conn.close()

if __name__ == "__main__":
    main()
