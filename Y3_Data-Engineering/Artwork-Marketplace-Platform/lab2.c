#include <mysql/mysql.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


// Function to execute a query and handle its results
void execute_query(MYSQL *conn, const char *query) {
    MYSQL_RES *res;
    MYSQL_ROW row;
    clock_t start, end;
    double cpu_time_used;

    start = clock();  // Start the clock

    if (mysql_query(conn, query)) {
        fprintf(stderr, "Query Failed: %s\n", mysql_error(conn));
        return;
    }

    // If the query is a SELECT query, process the result set
    if (mysql_field_count(conn) > 0) {
        res = mysql_store_result(conn);
        if (res == NULL) {
            fprintf(stderr, "mysql_store_result() failed\n");
            return;
        }
        mysql_free_result(res);
    } else {
        // For non-SELECT queries
        printf("Query Succeeded\n");
    }

    end = clock();  // Stop the clock

    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Execution time: %f seconds\n\n", cpu_time_used);
}

int main() {
    MYSQL *conn;
    conn = mysql_init(NULL);

    if (conn == NULL) {
        fprintf(stderr, "mysql_init() failed\n");
        return EXIT_FAILURE;
    }

    if (mysql_real_connect(conn, "localhost", "root", "what is mysql", "ArtmarketPlace", 0, NULL, 0) == NULL) {
        fprintf(stderr, "mysql_real_connect() failed\n");
        mysql_close(conn);
        return EXIT_FAILURE;
    }

    const char *queries[] = {
        // Query 1
        "SELECT a.name AS artist_name, aw.title AS artwork "
        "FROM ArtistProfile a "
        "JOIN Artwork aw ON a.artist_id = aw.artist_id "
        "WHERE YEAR(aw.created_date) = 2023 "
        "GROUP BY a.name, aw.title "
        "HAVING COUNT(DISTINCT MONTH(aw.created_date)) = 12;",
        
        // Query 2
        "SELECT DISTINCT a.name "
        "FROM ArtistProfile a "
        "JOIN Artwork aw ON a.artist_id = aw.artist_id "
        "WHERE aw.description LIKE '%sculpture%';",
        
        // Query 3
        "SELECT a.* "
        "FROM ArtistProfile a "
        "LEFT JOIN Artwork aw ON a.artist_id = aw.artist_id "
        "WHERE aw.artwork_id IS NULL;",
        
        // Query 4
        "SELECT DISTINCT b.name "
        "FROM Buyer b "
        "JOIN `Order` o ON b.buyer_id = o.buyer_id "
        "JOIN OrderItems oi ON o.order_id = oi.order_id "
        "JOIN Artwork aw ON oi.artwork_id = aw.artwork_id "
        "WHERE aw.description LIKE '%oil painting%' "
        "AND YEAR(o.order_date) = 2022;",
        
        // Query 5
        "SELECT DISTINCT ap.* "
        "FROM ArtistProfile ap "
        "JOIN Artwork aw ON ap.artist_id = aw.artist_id "
        "JOIN OrderItems oi ON aw.artwork_id = oi.artwork_id "
        "JOIN `Order` o ON oi.order_id = o.order_id "
        "JOIN Buyer b ON o.buyer_id = b.buyer_id "
        "WHERE aw.description LIKE '%oil painting%' "
        "AND YEAR(o.order_date) = 2022;",
        
        // Query 6
        "SELECT b.* "
        "FROM Buyer b "
        "LEFT JOIN `Order` o ON b.buyer_id = o.buyer_id "
        "WHERE o.order_id IS NULL;"
    };

    int query_count = sizeof(queries) / sizeof(queries[0]);
    double total_time = 0.0;
    
    for (int i = 0; i < query_count; i++) {
        printf("Running Query %d:\n", i + 1);
        clock_t start = clock();
        execute_query(conn, queries[i]);
        clock_t end = clock();
        total_time += ((double)(end - start)) / CLOCKS_PER_SEC;
    }

    printf("Total execution time for all queries: %f seconds\n", total_time);

    mysql_close(conn);
    return EXIT_SUCCESS;
}
