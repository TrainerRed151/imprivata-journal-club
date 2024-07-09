import sys
import numpy as np

# dirichlet distribution paramteres
topics = {
    "politics": 0.3,
    "economics": 0.3,
    "science": 0.2,
    "entertainment": 0.7,
    "sports": 0.4,
    "kids": 0.1,
}

words = {
    "vote": [0.1, 0.4, 0.6, 0.6, 0.8, 0.8],
    "congress": [0.1, 0.4, 0.6, 0.6, 0.8, 0.8],
    "stock": [0.3, 0.1, 0.6, 0.6, 0.8, 0.8],
    "money": [0.2, 0.1, 0.5, 0.5, 0.5, 0.8],
    "director": [0.3, 0.3, 0.2, 0.2, 0.8, 0.8],
    "discovery": [0.6, 0.5, 0.1, 0.6, 0.8, 0.8],
    "technology": [0.6, 0.5, 0.1, 0.6, 0.8, 0.8],
    "play": [0.8, 0.8, 0.9, 0.3, 0.1, 0.1],
    "movie": [0.9, 0.9, 0.9, 0.1, 0.3, 0.3],
    "game": [0.9, 0.7, 0.9, 0.5, 0.1, 0.2],
}


# article generation
def gen_article(num_words=5):
    topics_distribution = np.random.dirichlet([alpha for alpha in topics.values()])
    words_distribution_dict = {}
    for i, topic in enumerate(topics):
        words_distribution_dict[topic] = np.random.dirichlet([betas[i] for betas in words.values()])

    doc = []
    for _ in range(num_words):
        topic = np.random.choice(list(topics.keys()), p=topics_distribution)
        word = np.random.choice(list(words.keys()), p=words_distribution_dict[topic])

        doc.append(word)

    return ' '.join(doc)


if __name__ == '__main__':
    num_articles = int(sys.argv[1])

    for i in range(num_articles):
        print(f'Article {i+1}: {gen_article()}')

    print('developers developers developers developers')
