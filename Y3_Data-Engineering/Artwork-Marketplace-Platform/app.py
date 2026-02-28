# import streamlit as st
# from db import execute_query, fetch_query
# import bcrypt

# # Set up the Streamlit application
# st.set_page_config(page_title="Art Marketplace", layout="wide")
# st.title("Art Marketplace")
# st.sidebar.title("Navigation")

# menu = ["Home", "Artist Profiles", "Browse Artworks", "Cart", "Login/Sign Up", "Manage Artworks", "Reviews"]
# choice = st.sidebar.selectbox("Menu", menu)

# # Home Page
# if choice == "Home":
#     st.subheader("Welcome to the Art Marketplace")
#     st.text("Browse and buy artworks from talented artists.")
#     # Add featured artworks or news

# # Artist Profiles Page
# elif choice == "Artist Profiles":
#     st.subheader("Artist Profiles")
#     artists = fetch_query("SELECT * FROM ArtistProfile")
#     for artist in artists:
#         st.write(f"**Name:** {artist['name']}")
#         st.write(f"**Biography:** {artist['biography']}")
#         st.write(f"**Portfolio:** [Link]({artist['portfolio_link']})")
#         st.write("---")

# # Browse Artworks Page
# elif choice == "Browse Artworks":
#     st.subheader("Browse Artworks")
#     search_term = st.text_input("Search for artworks")
#     min_price, max_price = st.slider("Price range", 0, 1000, (0, 1000))
#     query = """
#     SELECT * FROM Artwork WHERE title LIKE %s AND price BETWEEN %s AND %s AND availability = 1
#     """
#     artworks = fetch_query(query, ('%' + search_term + '%', min_price, max_price))
#     for artwork in artworks:
#         st.image(artwork['image_url'], width=300)
#         st.write(f"**Title:** {artwork['title']}")
#         st.write(f"**Description:** {artwork['description']}")
#         st.write(f"**Price:** ${artwork['price']}")
#         if st.button(f"Add to Cart {artwork['artwork_id']}"):
#             if 'cart' not in st.session_state:
#                 st.session_state.cart = []
#             st.session_state.cart.append(artwork)
#             st.success("Artwork added to cart!")

# # Cart Page
# elif choice == "Cart":
#     st.subheader("Your Cart")
#     if 'cart' not in st.session_state:
#         st.session_state.cart = []
#     total_amount = 0
#     for item in st.session_state.cart:
#         st.write(f"{item['title']} - ${item['price']}")
#         total_amount += item['price']
#     st.write(f"Total Amount: ${total_amount}")
#     if st.button("Checkout"):
#         st.success("Checkout successful!")

# # Login/Sign Up Page
# elif choice == "Login/Sign Up":
#     st.subheader("Login or Sign Up")
#     user_role = st.radio("Role", ("Artist", "Buyer"))
#     email = st.text_input("Email")
#     password = st.text_input("Password", type="password")

#     if st.button("Login"):
#         query = f"SELECT * FROM {user_role} WHERE email = {email} AND password = {bcrypt.hashpw(password.decode('utf-8'))}"
#         result = fetch_query(query, (email, password))
#         if result:
#             st.session_state.user_id = result[0]['id']
#             st.session_state.user_role = user_role
#             st.success(f"Welcome, {result[0]['name']}!")
#         else:
#             st.error("Invalid credentials")

#     if st.button("Sign Up"):
#         name = st.text_input("Name")
#         if user_role == "Artist":
#             biography = st.text_area("Biography")
#             portfolio_link = st.text_input("Portfolio Link")
#             query = "INSERT INTO ArtistProfile (name, email, password, biography, portfolio_link) VALUES (%s, %s, %s, %s, %s)"
#             data = (name, email, bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()), biography, portfolio_link)
#         else:
#             query = "INSERT INTO Buyer (name, email, password) VALUES (%s, %s, %s)"
#             data = (name, email, bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()))

#         execute_query(query, data)
#         st.success(f"Account created successfully as {user_role}!")

# # Manage Artworks Page
# elif choice == "Manage Artworks":
#     if 'user_role' in st.session_state and st.session_state.user_role == "Artist":
#         st.subheader("Manage Your Artworks")
#         action = st.radio("Action", ("Add Artwork", "Delete Artwork"))

#         if action == "Add Artwork":
#             title = st.text_input("Title")
#             description = st.text_area("Description")
#             price = st.number_input("Price", min_value=0.0)
#             availability = st.selectbox("Availability", [1, 0])
#             image_url = st.text_input("Image URL")

#             if st.button("Add Artwork"):
#                 artist_id = st.session_state.user_id
#                 query = """
#                 INSERT INTO Artwork (artist_id, title, description, image_url, price, availability, created_date)
#                 VALUES (%s, %s, %s, %s, %s, %s, CURDATE())
#                 """
#                 execute_query(query, (artist_id, title, description, image_url, price, availability))
#                 st.success("Artwork added successfully!")

#         elif action == "Delete Artwork":
#             artwork_id = st.number_input("Artwork ID to Delete", min_value=1)
#             if st.button("Delete Artwork"):
#                 query = "DELETE FROM Artwork WHERE artwork_id = %s AND artist_id = %s"
#                 execute_query(query, (artwork_id, st.session_state.user_id))
#                 st.success("Artwork deleted successfully!")

# # Reviews Page
# elif choice == "Reviews":
#     st.subheader("Reviews")
#     if 'user_role' in st.session_state and st.session_state.user_role == "Buyer":
#         artwork_id = st.number_input("Artwork ID", min_value=1)
#         rating = st.slider("Rating", 1, 5)
#         comment = st.text_area("Comment")

#         if st.button("Submit Review"):
#             query = "INSERT INTO Review (artwork_id, user_id, rating, comment, review_date) VALUES (%s, %s, %s, %s, CURDATE())"
#             execute_query(query, (artwork_id, st.session_state.user_id, rating, comment))
#             st.success("Review submitted!")

#         review_id = st.number_input("Review ID to Delete", min_value=1)
#         if st.button("Delete Review"):
#             query = "DELETE FROM Review WHERE review_id = %s AND user_id = %s"
#             execute_query(query, (review_id, st.session_state.user_id))
#             st.success("Review deleted successfully!")

import streamlit as st
from db import execute_query, fetch_query
import bcrypt

# Set up the Streamlit application
st.set_page_config(page_title="Art Marketplace", layout="wide")
st.title("Art Marketplace")
st.sidebar.title("Navigation")

menu = ["Home", "Artist Profiles", "Browse Artworks", "Cart", "Login/Sign Up", "Manage Artworks", "Reviews"]
choice = st.sidebar.selectbox("Menu", menu)

# Home Page
if choice == "Home":
    st.subheader("Welcome to the Art Marketplace")
    st.text("Browse and buy artworks from talented artists.")
    # Add featured artworks or news

# Artist Profiles Page
elif choice == "Artist Profiles":
    st.subheader("Artist Profiles")
    artists = fetch_query("SELECT * FROM ArtistProfile")
    for artist in artists:
        st.write(f"**Name:** {artist['name']}")
        st.write(f"**Biography:** {artist['biography']}")
        st.write(f"**Portfolio:** [Link]({artist['portfolio_link']})")
        st.write("---")

# Browse Artworks Page
elif choice == "Browse Artworks":
    st.subheader("Browse Artworks")
    search_term = st.text_input("Search for artworks")
    min_price, max_price = st.slider("Price range", 0, 1000, (0, 1000))
    query = """
    SELECT * FROM Artwork WHERE title LIKE %s AND price BETWEEN %s AND %s AND availability = 1
    """
    artworks = fetch_query(query, ('%' + search_term + '%', min_price, max_price))
    for artwork in artworks:
        st.image(artwork['image_url'], width=300)
        st.write(f"**Title:** {artwork['title']}")
        st.write(f"**Description:** {artwork['description']}")
        st.write(f"**Price:** ${artwork['price']}")
        if st.button(f"Add to Cart {artwork['artwork_id']}"):
            if 'cart' not in st.session_state:
                st.session_state.cart = []
            st.session_state.cart.append(artwork)
            st.success("Artwork added to cart!")

# Cart Page
elif choice == "Cart":
    st.subheader("Your Cart")
    if 'cart' not in st.session_state:
        st.session_state.cart = []
    total_amount = 0
    for item in st.session_state.cart:
        st.write(f"{item['title']} - ${item['price']}")
        total_amount += item['price']
    st.write(f"Total Amount: ${total_amount}")
    if st.button("Checkout"):
        st.success("Checkout successful!")

# Login/Sign Up Page
elif choice == "Login/Sign Up":
    st.subheader("Login or Sign Up")
    user_role = st.radio("Role", ("Artist", "Buyer"))
    email = st.text_input("Email")
    password = st.text_input("Password") #, type="password")

    if st.button("Login"):
        query = f"SELECT * FROM {user_role} WHERE email = {email}"
        result = fetch_query(query)
        if result and password==result[0]['password']:
            st.session_state.user_id = result[0]['id']
            st.session_state.user_role = user_role
            st.success(f"Welcome, {result[0]['name']}!")
        else:
            st.error("Invalid credentials")

    if st.button("Sign Up"):
        name = st.text_input("Name")
        if user_role == "Artist":
            biography = st.text_area("Biography")
            portfolio_link = st.text_input("Portfolio Link")
            query = "INSERT INTO ArtistProfile (name, email, password, biography, portfolio_link) VALUES (%s, %s, %s, %s, %s)"
            data = (name, email, password, biography, portfolio_link)
        else:
            query = "INSERT INTO Buyer (name, email, password) VALUES (%s, %s, %s)"
            data = (name, email, password)

        execute_query(query, data)
        st.success(f"Account created successfully as {user_role}!")

# Manage Artworks Page
elif choice == "Manage Artworks":
    if 'user_role' in st.session_state and st.session_state.user_role == "Artist":
        st.subheader("Manage Your Artworks")
        action = st.radio("Action", ("Add Artwork", "Delete Artwork"))

        if action == "Add Artwork":
            title = st.text_input("Title")
            description = st.text_area("Description")
            price = st.number_input("Price", min_value=0.0)
            availability = st.selectbox("Availability", [1, 0])
            image_url = st.text_input("Image URL")

            if st.button("Add Artwork"):
                artist_id = st.session_state.user_id
                query = """
                INSERT INTO Artwork (artist_id, title, description, image_url, price, availability, created_date)
                VALUES (%s, %s, %s, %s, %s, %s, CURDATE())
                """
                execute_query(query, (artist_id, title, description, image_url, price, availability))
                st.success("Artwork added successfully!")

        elif action == "Delete Artwork":
            artwork_id = st.number_input("Artwork ID to Delete", min_value=1)
            if st.button("Delete Artwork"):
                query = "DELETE FROM Artwork WHERE artwork_id = %s AND artist_id = %s"
                execute_query(query, (artwork_id, st.session_state.user_id))
                st.success("Artwork deleted successfully!")

# Reviews Page
elif choice == "Reviews":
    st.subheader("Reviews")
    if 'user_role' in st.session_state and st.session_state.user_role == "Buyer":
        artwork_id = st.number_input("Artwork ID", min_value=1)
        rating = st.slider("Rating", 1, 5)
        comment = st.text_area("Comment")

        if st.button("Submit Review"):
            query = "INSERT INTO Review (artwork_id, user_id, rating, comment, review_date) VALUES (%s, %s, %s, %s, CURDATE())"
            execute_query(query, (artwork_id, st.session_state.user_id, rating, comment))
            st.success("Review submitted!")

        review_id = st.number_input("Review ID to Delete", min_value=1)
        if st.button("Delete Review"):
            query = "DELETE FROM Review WHERE review_id = %s AND user_id = %s"
            execute_query(query, (review_id, st.session_state.user_id))
            st.success("Review deleted successfully!")
