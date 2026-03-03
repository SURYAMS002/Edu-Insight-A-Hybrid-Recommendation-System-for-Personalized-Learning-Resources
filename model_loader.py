import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

print("Loading model... Please wait ⏳")
model = SentenceTransformer('all-MiniLM-L6-v2')


# =========================================================
# LOAD DATASETS
# =========================================================
def load_datasets():
    print("Loading datasets...")
    courses = pd.read_csv("data/udemy_course_data.csv", encoding="ISO-8859-1")
    youtube = pd.read_csv("data/you_tube.csv", encoding="ISO-8859-1")
    books = pd.read_excel("data/book_recommendation.xlsx")
    skills = pd.read_csv("data/skill_to_career_mapping.csv", encoding="ISO-8859-1")
    print("Datasets loaded successfully ✅")
    return courses, youtube, books, skills


# =========================================================
# RECOMMENDER FUNCTIONS
# =========================================================
def recommend_courses_full(query, courses, top_n=5, alpha=0.7):
    course_col = "course_title" if "course_title" in courses.columns else courses.columns[0]
    numeric_cols_courses = [col for col in ["num_subscribers", "num_reviews", "num_lectures", "price"] if col in courses.columns]

    query_emb = model.encode([query])
    course_embeddings = model.encode(courses[course_col].fillna("").astype(str))
    sims = cosine_similarity(query_emb, course_embeddings)[0]

    if numeric_cols_courses:
        numeric_score = courses[numeric_cols_courses].apply(lambda x: (x - x.min()) / (x.max() - x.min())).mean(axis=1).values
    else:
        numeric_score = np.zeros(len(courses))

    final_score = alpha * sims + (1 - alpha) * numeric_score
    courses_copy = courses.copy()
    courses_copy["final_score"] = final_score
    top_courses = courses_copy.sort_values(by="final_score", ascending=False).head(top_n)
    return top_courses


def recommend_youtube_full(query, youtube, top_n=5, alpha=0.7):
    title_col = next((col for col in youtube.columns if "title" in col.lower()), youtube.columns[0])
    numeric_cols_youtube = [col for col in ["Likes", "views", "subscribers", "share"] if col in youtube.columns]

    query_emb = model.encode([query])
    youtube_embeddings = model.encode(youtube[title_col].fillna("").astype(str))
    sims = cosine_similarity(query_emb, youtube_embeddings)[0]

    if numeric_cols_youtube:
        engagement_score = youtube[numeric_cols_youtube].apply(lambda x: (x - x.min()) / (x.max() - x.min())).mean(axis=1).values
    else:
        engagement_score = np.zeros(len(youtube))

    final_score = alpha * sims + (1 - alpha) * engagement_score
    youtube_copy = youtube.copy()
    youtube_copy["final_score"] = final_score
    top_youtube = youtube_copy.sort_values(by="final_score", ascending=False).head(top_n)

    keep_cols = [col for col in ["course_title1111", "youtube_video_link", "Channel Name", title_col] if col in youtube.columns]
    return top_youtube[keep_cols].reset_index(drop=True)


def recommend_books(query, books, top_n=5, alpha=0.7):
    book_col = next((col for col in books.columns if "title" in col.lower()), books.columns[0])
    numeric_cols_books = [col for col in ["average_book_rating", "num_review", "num_rating", "publication_year"] if col in books.columns]

    query_emb = model.encode([query])
    book_embeddings = model.encode(books[book_col].fillna("").astype(str))
    sims = cosine_similarity(query_emb, book_embeddings)[0]

    if numeric_cols_books:
        numeric_score = books[numeric_cols_books].apply(lambda x: (x - x.min()) / (x.max() - x.min())).mean(axis=1).values
    else:
        numeric_score = np.zeros(len(books))

    final_score = alpha * sims + (1 - alpha) * numeric_score
    books_copy = books.copy()
    books_copy["final_score"] = final_score
    top_books = books_copy.sort_values(by="final_score", ascending=False).head(top_n)

    keep_cols = [col for col in [book_col, "author", "image_url"] if col in books.columns]
    return top_books[keep_cols].reset_index(drop=True)


def recommend_skills_and_careers(query, skills, top_n=5):
    """Return skill–career mapping"""
    text_col = "Skill" if "Skill" in skills.columns else skills.columns[0]
    query_emb = model.encode([query])
    skill_embeddings = model.encode(skills[text_col].fillna("").astype(str))
    sims = cosine_similarity(query_emb, skill_embeddings)[0]

    top_indices = sims.argsort()[-top_n:][::-1]
    keep_cols = [col for col in ["Skill", "Skill Category", "Career(s)", "Description / Use in Career"] if col in skills.columns]
    return skills.iloc[top_indices][keep_cols].reset_index(drop=True)


# =========================================================
# COMBINED FUNCTION FOR FLASK
# =========================================================
def recommend_items(skill, courses, youtube, books, skills, model):

    top_courses = recommend_courses_full(skill, courses)
    top_youtube = recommend_youtube_full(skill, youtube)
    top_books = recommend_books(skill, books)
    top_skills = recommend_skills_and_careers(skill, skills)

    return {
        "courses": top_courses.to_dict(orient="records"),
        "youtube": top_youtube.to_dict(orient="records"),
        "books": top_books.to_dict(orient="records"),
        "skills": top_skills.to_dict(orient="records"),
    }
