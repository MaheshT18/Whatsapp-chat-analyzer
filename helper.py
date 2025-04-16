# --------------------- Helper Imports ---------------------
from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import emoji
from textblob import TextBlob
import networkx as nx
from gensim import corpora, models
from nltk.corpus import stopwords
import nltk
import numpy as np

# Ensure NLTK stopwords are downloaded.
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

# Initialize URL extractor
extract = URLExtract()

# --------------------- 1. Basic Statistics Functions ---------------------
def fetch_stats(selected_user, df):
    """
    Returns overall statistics for the chat:
    - Number of messages, total words, media messages, and links shared.
    
    If a specific user is selected (not "Overall"), the stats are computed only for that user.
    """
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    num_messages = df.shape[0]
    
    # Count the number of words across all messages
    words = []
    for message in df['message']:
        words.extend(message.split())
    
    # Count media messages (assumes '<Media omitted>\n' indicates a media message)
    num_media_messages = df[df['message'] == '<Media omitted>\n'].shape[0]
    
    # Count the number of links shared using URLExtract
    num_links = []
    for message in df['message']:
        num_links.extend(extract.find_urls(message))
    
    return num_messages, len(words), num_media_messages, len(num_links)

def most_busy_users(df):
    """
    Returns two values:
    - A Series of the top users by message count.
    - A DataFrame with usernames and their percentage of total messages.
    """
    x = df['user'].value_counts().head()
    df_perc = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'Username', 'user': 'Percentage'})
    return x, df_perc

# --------------------- 2. WordCloud & Common Words ---------------------
def create_wordcloud(selected_user, df):
    """
    Generates and returns a WordCloud image based on chat messages,
    filtering out common stopwords (loaded from a file).
    
    This function excludes group notifications and media omitted messages.
    """
    # Read the stopwords from the file
    with open('stopwords.txt', 'r') as f:
        stop_words = f.read()
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    temp = df[(df['user'] != 'group_notification') & (df['message'] != '<Media omitted>\n')]
    
    # Remove stopwords from each message
    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return " ".join(y)
    
    temp['message'] = temp['message'].apply(remove_stop_words)
    
    # Generate WordCloud
    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
    df_wc = wc.generate(temp['message'].str.cat(sep=" "))
    return df_wc

def most_common_words(selected_user, df):
    """
    Returns a DataFrame of the 20 most common words in the chat (excluding stopwords),
    along with their frequencies. Works for a specific user or overall.
    """
    with open('stopwords.txt', 'r') as f:
        stop_words = f.read()
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    temp = df[(df['user'] != 'group_notification') & (df['message'] != '<Media omitted>\n')]
    
    words = []
    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)
    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df

# --------------------- 3. Emoji Analysis ---------------------
def extract_emojis(s):
    """
    Extracts all emojis present in the given string 's'.
    """
    return [c for c in s if c in emoji.EMOJI_DATA]

def emoji_helper(selected_user, df):
    """
    Analyzes emoji usage in chat messages.
    Returns a DataFrame with emojis and their counts, sorted by frequency.
    """
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    
    emojis = []
    for message in df['message'].dropna():
        emojis.extend(extract_emojis(message))
    emoji_df = pd.DataFrame(Counter(emojis).most_common(), columns=["emoji", "count"])
    return emoji_df

# --------------------- 4. Timeline & Activity Analysis ---------------------
def monthly_timeline(selected_user, df):
    """
    Generates a timeline DataFrame grouped by month showing the count of messages.
    The timeline is formatted with month names and years.
    """
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()
    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))
    timeline['time'] = time
    return timeline

def daily_timeline(selected_user, df):
    """
    Generates a daily timeline DataFrame that shows the number of messages per day.
    """
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    daily_timeline = df.groupby('only_date').count()['message'].reset_index()
    return daily_timeline

def week_activity_map(selected_user, df):
    """
    Returns a Series with the message counts per day of the week.
    """
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df['day_name'].value_counts()

def month_activity_map(selected_user, df):
    """
    Returns a Series with the message counts per month.
    """
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df['month'].value_counts()

def activity_heatmap(selected_user, df):
    """
    Creates and returns a pivot table (DataFrame) for the activity heatmap,
    where rows represent days of the week and columns represent time periods.
    """
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)
    return user_heatmap

# --------------------- 5. Sentiment Analysis ---------------------
def sentiment_analysis(selected_user, df):
    """
    Performs sentiment analysis on the chat messages using TextBlob.
    Returns a DataFrame with sentiment counts and their percentages.
    Excludes group notifications and media omitted messages.
    """
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    df = df[(df['user'] != 'group_notification') & (df['message'] != '<Media omitted>\n')]
    if df.empty:
        return pd.DataFrame({
            'Sentiment': ['Positive', 'Negative', 'Neutral'],
            'Count': [0, 0, 0],
            'Percentage': [0.0, 0.0, 0.0]
        })
    polarity_scores = []
    for message in df['message']:
        analysis = TextBlob(message)
        polarity_scores.append(analysis.sentiment.polarity)
    positive_count = sum(score > 0 for score in polarity_scores)
    negative_count = sum(score < 0 for score in polarity_scores)
    neutral_count = sum(score == 0 for score in polarity_scores)
    total = positive_count + negative_count + neutral_count
    if total == 0:
        return pd.DataFrame({
            'Sentiment': ['Positive', 'Negative', 'Neutral'],
            'Count': [0, 0, 0],
            'Percentage': [0.0, 0.0, 0.0]
        })
    positive_percent = (positive_count / total) * 100
    negative_percent = (negative_count / total) * 100
    neutral_percent = (neutral_count / total) * 100
    sentiment_df = pd.DataFrame({
        'Sentiment': ['Positive', 'Negative', 'Neutral'],
        'Count': [positive_count, negative_count, neutral_count],
        'Percentage': [positive_percent, negative_percent, neutral_percent]
    })
    return sentiment_df

# --------------------- 6. Conversation Network Graph ---------------------
def conversation_network_graph(selected_user, df):
    """
    Builds a conversation network graph using NetworkX.
    For Overall analysis, it filters to the top 10 active users to reduce clutter.
    Connects consecutive messages between different users.
    """
    # For Overall analysis: consider only top 10 active users
    if selected_user == 'Overall':
        msg_counts = df[df['user'] != 'group_notification']['user'].value_counts().nlargest(10).index.tolist()
        df = df[df['user'].isin(msg_counts)]
    else:
        df = df[df['user'] == selected_user]
    graph = nx.Graph()
    users = df['user'].tolist()
    for i in range(1, len(users)):
        if users[i] != users[i-1]:
            if graph.has_edge(users[i-1], users[i]):
                graph[users[i-1]][users[i]]['weight'] += 1
            else:
                graph.add_edge(users[i-1], users[i], weight=1)
    return graph

# --------------------- 7. Topic Modeling & Keyword Extraction ---------------------
def topic_modeling(selected_user, df, num_topics=3, num_words=5):
    """
    Performs topic modeling using LDA (Latent Dirichlet Allocation) from gensim.
    Filters out group notifications and media omitted messages.
    Returns topics as a list of formatted strings showing the top words and their weights.
    """
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    temp = df[(df['user'] != 'group_notification') & (df['message'] != '<Media omitted>\n')]
    documents = temp['message'].tolist()
    if not documents:
        return "Not enough data for topic modeling."
    
    # Basic tokenization and stopword removal using NLTK stopwords
    stops = set(stopwords.words('english'))
    texts = []
    for doc in documents:
        tokens = doc.lower().split()
        filtered_tokens = [token for token in tokens if token not in stops]
        texts.append(filtered_tokens)
    
    # Create dictionary and corpus for LDA
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    if not corpus:
        return "Not enough data for topic modeling."
    
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)
    topics = lda_model.show_topics(num_topics=num_topics, num_words=num_words, formatted=False)
    
    # Format topics in a user-friendly way
    formatted_topics = []
    for topic_no, word_probs in topics:
        topic_words = [f"{word} ({weight:.3f})" for word, weight in word_probs]
        topic_str = f"Topic {topic_no + 1}: " + ", ".join(topic_words)
        formatted_topics.append(topic_str)
    
    return formatted_topics

# --------------------- 8. Engagement & Response Time Analysis ---------------------
def response_time_analysis(selected_user, df):
    """
    Computes the average response time (in hours) for each user based on the 
    time difference between consecutive messages. Group notifications are excluded.
    """
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    df = df.sort_values('date')
    df = df[df['user'] != 'group_notification']
    # Calculate time difference in hours between consecutive messages
    df['response_time'] = df['date'].diff().dt.total_seconds() / 3600
    response_df = df.groupby('user')['response_time'].mean().reset_index()
    response_df['response_time'] = response_df['response_time'].fillna(0).round(2)
    return response_df

# --------------------- 9. Silent Observers List ---------------------
def silent_observers(df):
    """
    Identifies the 3 users with the lowest message counts (excluding group notifications),
    which can be seen as "silent observers".
    """
    temp = df[df['user'] != 'group_notification']
    msg_counts = temp['user'].value_counts().reset_index().rename(columns={'index': 'User', 'user': 'Message Count'})
    silent_list = msg_counts.sort_values('Message Count').head(3)
    return silent_list
