import streamlit as st
import pandas as pd
import pickle
from huggingface_hub import hf_hub_download

# Load Popular Books (local file or Hugging Face)
popular_df = pickle.load(open('popular_books.pkl', 'rb'))
pt = pickle.load(open('books_users_pivot.pkl', 'rb'))
similarity_scores = pickle.load(open('cosine_similarity.pkl', 'rb'))

@st.cache_resource(show_spinner=False)
def load_books():
    repo_id = "MShoaib123/book-recommender-data"
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename="Books.pkl",
        repo_type="dataset"
        # Remove token if public dataset, else uncomment below
        # token=st.secrets["HF_TOKEN"]  
    )
    with open(local_path, "rb") as f:
        books = pickle.load(f)
    return books

books = load_books()

def recommend(book_name):
    try:
        index = pt.index.get_loc(book_name)
    except KeyError:
        return []

    similar_items = sorted(list(enumerate(similarity_scores[index])), 
                            key=lambda x: x[1], reverse=True)[1:6]
    data = []
    for i in similar_items:
        temp_df = books[books['Book-Title'] == pt.index[i[0]]]
        data.append({
            "title": pt.index[i[0]],
            "author": temp_df['Book-Author'].values[0],
            "image": temp_df['Image-URL-M'].values[0] if pd.notna(temp_df['Image-URL-M'].values[0]) else ""
        })
    return data

# Streamlit UI
st.set_page_config(page_title="📚 Book Recommender", layout="wide")

st.title("📚 Book Recommender System")
st.markdown("A collaborative filtering and popularity-based book recommender system built with Streamlit.")

# Tabs for Navigation
tab1, tab2 = st.tabs(["🏆 Top 50 Books", "🔍 Recommend Books"])

# Tab 1 - Popular Books
with tab1:
    st.subheader("Top 50 Most Rated Books")
    for i in range(0, len(popular_df), 5):  # 5 books per row
        cols = st.columns(5)
        for idx, col in enumerate(cols):
            if i + idx < len(popular_df):
                book = popular_df.iloc[i + idx]
                
                # Show image only if URL is present
                if pd.notna(book.get("Image-URL-M", None)) and book["Image-URL-M"].strip() != "":
                    col.image(book["Image-URL-M"], width=120)
                else:
                    col.write("📚 No Image Available")
                
                # Always show book info
                col.markdown(f"**{book['Book-Title']}**")
                col.caption(f"Author: {book['Book-Author']}")
                col.write(f"⭐ {book['Avg_rating']:.2f}  |  📈 {book['Num_Ratings']} ratings")

# Tab 2 - Recommend Books
with tab2:
    st.subheader("Find Your Next Book ❤️")
    user_input = st.text_input("Enter a book title:", placeholder="Try: Harry Potter, The Hobbit, 1984...")
    if st.button("Get Recommendations"):
        if user_input.strip():
            results = recommend(user_input.strip())
            if results:
                rec_cols = st.columns(5)
                for idx, book in enumerate(results):
                    col = rec_cols[idx % 5]
                    if book['image']:
                        col.image(book['image'], width=120)
                    else:
                        col.write("📚 No Image Available")
                    col.write(f"**{book['title']}**")
                    col.caption(f"✍️ {book['author']}")
            else:
                st.warning("No similar books found. Please check the spelling or try another title.")
        else:
            st.error("Please enter a book title.")
