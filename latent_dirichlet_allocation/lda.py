import sys
import numpy as np
from collections import Counter
import nltk
from nltk.corpus import reuters


# dirichlet distribution paramteres
topics = {
    "politics": 0.3,
    "economics": 0.3,
    "science": 0.4,
    "entertainment": 0.2,
    "sports": 0.5,
    "kids": 0.7,
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
    "win": [0.8, 0.3, 0.1, 0.4, 0.8, 0.7],
}


# article generation
def gen_article(num_words=15):
    topics_distribution = np.random.dirichlet([alpha for alpha in topics.values()])
    words_distribution_dict = {}
    for i, topic in enumerate(topics):
        words_distribution_dict[topic] = np.random.dirichlet([betas[i] for betas in words.values()])

    doc = []
    for _ in range(num_words):
        topic = np.random.choice(list(topics.keys()), p=topics_distribution)
        word = np.random.choice(list(words.keys()), p=words_distribution_dict[topic])

        doc.append(word)

    return doc


# topic assignment
def fit_articles(articles, n_topics=6, n_runs=100_000, epsilon=0.1):
    # randomly assign topics
    topic_mapping = []
    for article in articles:
        topic_mapping.append(np.random.randint(n_topics, size=len(article)))

    for _ in range(n_runs):
        # select a random word in a random article
        article_i = np.random.randint(len(articles))
        article = articles[article_i]

        word_i = np.random.randint(len(article))
        word = article[word_i]

        # get topic distribution of article
        article_topic_counts = dict(Counter(topic_mapping[article_i]))

        # get topic distribution of word in all articles
        word_topic_counts = {}
        for r in range(len(articles)):
            for c in range(len(articles[r])):
                if articles[r][c] == word:
                    if r == article_i and c == word_i:
                        continue

                    topic = topic_mapping[r][c]
                    if topic not in word_topic_counts:
                        word_topic_counts[topic] = 1
                    else:
                        word_topic_counts[topic] += 1

        # gibbs sampling
        topic_weights = []
        for i in range(n_topics):
            topic_weights.append(
                #(article_topic_counts.get(i, 0) + list(topics.values())[i]) * (word_topic_counts.get(i, 0) + words[word][i])
                (article_topic_counts.get(i, 0) + epsilon) * (word_topic_counts.get(i, 0) + epsilon)
            )

        #topic_mapping[article_i][word_i] = np.random.choice(range(n_topics), p=topic_weights/np.sum(topic_weights))
        topic_mapping[article_i][word_i] = np.random.choice(range(n_topics), p=np.array(topic_weights)/np.sum(topic_weights))

    return topic_mapping


if __name__ == '__main__':
    np.random.seed(1)
    num_articles = int(sys.argv[1])

    articles = [gen_article() for _ in range(num_articles)]
    print('Articles:\n-------')
    for i, article in enumerate(articles):
        print(f'Article {i+1}: {" ".join(article)}')

    print('developers developers developers developers')



    # nltk documents
    nltk.download('reuters')
    file_ids = reuters.fileids()

    articles = [reuters.raw(file_id) for file_id in file_ids[:10]]
    for i, article in enumerate(articles):
        print(f'Article {i+1}:\n{article[:200]}...\n')  # Print the first 200 characters of each article



    topics = fit_articles(articles)
    print('\nTopics:\n-------')
    for i, word_topics in enumerate(topics):
        print(f'Article {i+1}: {word_topics}')
