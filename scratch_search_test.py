import os
from app import fetch_trusted_news

query = "The image shows Raghav Chadha wearing a BJP scarf and greeting Amit Shah. They are standing in front of a BJP logo."
news = fetch_trusted_news(query)
print("News for long query:", news)

query2 = "Raghav Chadha BJP"
news2 = fetch_trusted_news(query2)
print("News for short query:", news2)
