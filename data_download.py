import os

# Create data directory if it doesn't exist
if not os.path.exists("data"):
    os.mkdir("data")

# Download reviews data
reviews_url = "https://drive.google.com/uc?id=1pQnXa7DWLdeUpvUFsKusYzwbA5CAAZx7"
reviews_file = "data/reviews_data.json.gz"
os.system(f"curl -o {reviews_file} {reviews_url}")

# Download books data
books_url = "https://drive.google.com/uc?id=1LXpK1UfqtP89H1tYy0pBGHjYk8IhigUK"
books_file = "data/books_data.json.gz"
os.system(f"curl -o {books_file} {books_url}")
