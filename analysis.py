"""
Customer Experience & Sentiment Analysis - Home Improvement Products
=====================================================================
Analysis of Home Depot product reviews to uncover what drives customer
satisfaction and dissatisfaction across home improvement categories.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections import Counter
import re
import os
import warnings
warnings.filterwarnings('ignore')

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'home_depot_reviews.csv')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Plot styling
plt.rcParams.update({
    'figure.figsize': (12, 6),
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.3,
})
COLORS = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#9b59b6',
          '#1abc9c', '#e67e22', '#34495e', '#16a085', '#c0392b']

# Common stopwords for word frequency analysis
STOPWORDS = set([
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'is', 'it', 'was', 'are', 'were', 'be',
    'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
    'would', 'could', 'should', 'may', 'might', 'shall', 'can', 'need',
    'dare', 'ought', 'used', 'this', 'that', 'these', 'those', 'i', 'me',
    'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
    'she', 'her', 'hers', 'herself', 'its', 'itself', 'they', 'them',
    'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
    'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few',
    'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
    'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'just', 'don',
    'now', 'also', 'get', 'got', 'one', 'two', 'go', 'going', 'went',
    'really', 'even', 'much', 'many', 'well', 'back', 'still', 'thing',
    'things', 'way', 'us', 'am', 'up', 'out', 'about', 'as', 'into',
    'if', 'then', 'there', 'here', 'after', 'before', 'above', 'below',
    'between', 'through', 'during', 'while', 'over', 'under', 'again',
    'further', 'once', 'review', 'collected', 'part', 'promotion',
    'product', 'bought', 'buy', 'purchased', 'came', 'come', 'use',
    'using', 'make', 'made', 'put', 'take', 'took', 'say', 'said',
    'like', 'know', 'see', 'look', 'looking', 'want', 'wanted',
    'lot', 'would', 'first', 'time', 'day', 'days', 'year', 'years',
    'month', 'months', 'week', 'weeks', 'new', 'old', 'good', 'great',
    'nice', 'love', 'sure', 'right', 'left', 'little', 'big', 'long',
    'high', 'low', 'small', 'large', 'think', 'thought', 'keep', 'find',
    'give', 'tell', 'try', 'tried', 'let', 'seem', 'help', 'show',
    'turn', 'move', 'set', 'run', 'need', 'work', 'call', 'asked',
    'home', 'depot', 'ordered', 'order', 'received', 'item', 'items',
    'however', 'though', 'although', 'yet', 'since', 'because',
    'another', 'around', 'away', 'down', 'off', 'far', 'near',
    've', 'll', 're', 'don', 'didn', 'doesn', 'won', 'isn', 'wasn',
    'weren', 'hasn', 'haven', 'hadn', 'couldn', 'shouldn', 'wouldn',
    'aren', 'ain', 'ma',
])


def print_section(title):
    """Print a clear section header."""
    print('\n' + '=' * 70)
    print(f'  {title}')
    print('=' * 70)


def clean_text(text):
    """Clean review text for analysis."""
    if pd.isna(text) or not isinstance(text, str):
        return ''
    # Remove the promotional disclaimer that appears in many reviews
    text = re.sub(r'\[This review was collected as part of a promotion\.\]\s*', '', text)
    return text.strip()


def get_word_frequencies(texts, top_n=30):
    """Extract word frequencies from a list of texts, excluding stopwords."""
    words = []
    for text in texts:
        if not text:
            continue
        cleaned = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        words.extend([w for w in cleaned.split() if w not in STOPWORDS and len(w) > 2])
    return Counter(words).most_common(top_n)


# =========================================================================
# SECTION 1: Data Loading and Cleaning
# =========================================================================
print_section('1. DATA LOADING AND CLEANING')

df = pd.read_csv(DATA_PATH)
print(f'Loaded {len(df):,} reviews across {df["category"].nunique()} product categories')
print(f'Columns: {", ".join(df.columns)}')

# Check for nulls
null_counts = df.isnull().sum()
print(f'\nMissing values:')
for col in df.columns:
    if null_counts[col] > 0:
        print(f'  - {col}: {null_counts[col]} ({null_counts[col]/len(df)*100:.1f}%)')

# Remove duplicates
dupes = df.duplicated().sum()
if dupes > 0:
    df = df.drop_duplicates().reset_index(drop=True)
    print(f'\nRemoved {dupes} duplicate rows. Dataset now has {len(df):,} reviews.')

# Clean review text
df['review_text_clean'] = df['review_text'].apply(clean_text)

# Drop rows with no review text (cannot analyze sentiment without text)
no_text_count = (df['review_text_clean'] == '').sum()
print(f'Reviews with no text: {no_text_count} (will be excluded from text-based analysis)')
df_text = df[df['review_text_clean'] != ''].copy()

print(f'\nDataset ready: {len(df):,} total reviews, {len(df_text):,} with review text')


# =========================================================================
# SECTION 2: Rating Distribution Analysis
# =========================================================================
print_section('2. RATING DISTRIBUTION')

rating_counts = df['review_rating'].value_counts().sort_index()
rating_pct = (rating_counts / len(df) * 100).round(1)

print('Review Rating Distribution:')
for rating in sorted(rating_counts.index):
    bar = '#' * int(rating_pct[rating])
    print(f'  {rating} star: {rating_counts[rating]:>5,} reviews ({rating_pct[rating]:>5.1f}%)  {bar}')

avg_rating = df['review_rating'].mean()
median_rating = df['review_rating'].median()
print(f'\nAverage rating: {avg_rating:.2f}')
print(f'Median rating: {median_rating:.1f}')

positive = (df['review_rating'] >= 4).sum()
negative = (df['review_rating'] <= 2).sum()
neutral = (df['review_rating'] == 3).sum()
print(f'Positive (4-5 stars): {positive:,} ({positive/len(df)*100:.1f}%)')
print(f'Neutral (3 stars): {neutral:,} ({neutral/len(df)*100:.1f}%)')
print(f'Negative (1-2 stars): {negative:,} ({negative/len(df)*100:.1f}%)')

# Chart: Rating distribution
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(rating_counts.index, rating_counts.values, color=COLORS[:5], edgecolor='white', linewidth=0.5)
for bar, count in zip(bars, rating_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
            f'{count:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_xlabel('Star Rating')
ax.set_ylabel('Number of Reviews')
ax.set_title('Distribution of Customer Review Ratings')
ax.set_xticks([1, 2, 3, 4, 5])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig(os.path.join(OUTPUT_DIR, 'rating_distribution.png'))
plt.close()
print('\nSaved: outputs/rating_distribution.png')


# =========================================================================
# SECTION 3: Sentiment Analysis (VADER + TextBlob)
# =========================================================================
print_section('3. SENTIMENT ANALYSIS')

analyzer = SentimentIntensityAnalyzer()

print('Running VADER sentiment analysis on review texts...')
df_text['vader_compound'] = df_text['review_text_clean'].apply(
    lambda x: analyzer.polarity_scores(x)['compound']
)

print('Running TextBlob sentiment analysis on review texts...')
df_text['textblob_polarity'] = df_text['review_text_clean'].apply(
    lambda x: TextBlob(x).sentiment.polarity
)

# Classify VADER sentiment
df_text['vader_label'] = df_text['vader_compound'].apply(
    lambda x: 'Positive' if x >= 0.05 else ('Negative' if x <= -0.05 else 'Neutral')
)

vader_dist = df_text['vader_label'].value_counts()
print('\nVADER Sentiment Distribution:')
for label in ['Positive', 'Neutral', 'Negative']:
    if label in vader_dist.index:
        count = vader_dist[label]
        print(f'  {label}: {count:,} ({count/len(df_text)*100:.1f}%)')

# Average sentiment by star rating
print('\nAverage Sentiment Score by Star Rating:')
print(f'  {"Rating":<8} {"VADER":<12} {"TextBlob":<12} {"Count":<8}')
print(f'  {"-"*40}')
for rating in sorted(df_text['review_rating'].unique()):
    subset = df_text[df_text['review_rating'] == rating]
    vader_avg = subset['vader_compound'].mean()
    tb_avg = subset['textblob_polarity'].mean()
    print(f'  {rating:<8} {vader_avg:<12.3f} {tb_avg:<12.3f} {len(subset):<8,}')

# Chart: Sentiment by star rating
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# VADER
grouped = df_text.groupby('review_rating')['vader_compound'].mean()
axes[0].bar(grouped.index, grouped.values, color=['#e74c3c', '#e67e22', '#f39c12', '#2ecc71', '#27ae60'])
axes[0].set_xlabel('Star Rating')
axes[0].set_ylabel('Average VADER Compound Score')
axes[0].set_title('VADER Sentiment Score by Star Rating')
axes[0].set_xticks([1, 2, 3, 4, 5])
axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

# TextBlob
grouped_tb = df_text.groupby('review_rating')['textblob_polarity'].mean()
axes[1].bar(grouped_tb.index, grouped_tb.values, color=['#e74c3c', '#e67e22', '#f39c12', '#2ecc71', '#27ae60'])
axes[1].set_xlabel('Star Rating')
axes[1].set_ylabel('Average TextBlob Polarity')
axes[1].set_title('TextBlob Polarity by Star Rating')
axes[1].set_xticks([1, 2, 3, 4, 5])
axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'sentiment_by_rating.png'))
plt.close()
print('\nSaved: outputs/sentiment_by_rating.png')


# =========================================================================
# SECTION 4: Sentiment vs Star Rating Alignment
# =========================================================================
print_section('4. SENTIMENT vs STAR RATING ALIGNMENT')

# Check for mismatches - reviews where sentiment and rating disagree
df_text['sentiment_rating_match'] = 'Aligned'
# Positive sentiment but low rating
mask_mismatch_pos = (df_text['vader_compound'] >= 0.3) & (df_text['review_rating'] <= 2)
df_text.loc[mask_mismatch_pos, 'sentiment_rating_match'] = 'Positive text, Low rating'
# Negative sentiment but high rating
mask_mismatch_neg = (df_text['vader_compound'] <= -0.1) & (df_text['review_rating'] >= 4)
df_text.loc[mask_mismatch_neg, 'sentiment_rating_match'] = 'Negative text, High rating'

match_counts = df_text['sentiment_rating_match'].value_counts()
print('Sentiment-Rating Alignment:')
for label, count in match_counts.items():
    print(f'  {label}: {count:,} ({count/len(df_text)*100:.1f}%)')

# Correlation between sentiment and rating
corr_vader = df_text['review_rating'].corr(df_text['vader_compound'])
corr_tb = df_text['review_rating'].corr(df_text['textblob_polarity'])
print(f'\nCorrelation between star rating and VADER sentiment: {corr_vader:.3f}')
print(f'Correlation between star rating and TextBlob polarity: {corr_tb:.3f}')

# Show some mismatched examples
mismatched_pos = df_text[mask_mismatch_pos].head(3)
if len(mismatched_pos) > 0:
    print('\nExamples - Positive language but low star rating:')
    for _, row in mismatched_pos.iterrows():
        snippet = row['review_text_clean'][:120].replace('\n', ' ')
        print(f'  [{row["review_rating"]} stars, VADER: {row["vader_compound"]:.2f}] "{snippet}..."')

mismatched_neg = df_text[mask_mismatch_neg].head(3)
if len(mismatched_neg) > 0:
    print('\nExamples - Negative language but high star rating:')
    for _, row in mismatched_neg.iterrows():
        snippet = row['review_text_clean'][:120].replace('\n', ' ')
        print(f'  [{row["review_rating"]} stars, VADER: {row["vader_compound"]:.2f}] "{snippet}..."')

# Chart: Scatter plot of sentiment vs rating
fig, ax = plt.subplots(figsize=(10, 6))
jitter = np.random.normal(0, 0.12, size=len(df_text))
ax.scatter(df_text['review_rating'] + jitter, df_text['vader_compound'],
           alpha=0.15, s=8, c=df_text['review_rating'],
           cmap='RdYlGn', edgecolors='none')
# Add mean line
means = df_text.groupby('review_rating')['vader_compound'].mean()
ax.plot(means.index, means.values, 'ko-', markersize=8, linewidth=2, label='Mean sentiment')
ax.set_xlabel('Star Rating')
ax.set_ylabel('VADER Compound Sentiment Score')
ax.set_title('Sentiment Score vs Star Rating (with jitter)')
ax.set_xticks([1, 2, 3, 4, 5])
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.4)
ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig(os.path.join(OUTPUT_DIR, 'sentiment_vs_rating_scatter.png'))
plt.close()
print('\nSaved: outputs/sentiment_vs_rating_scatter.png')


# =========================================================================
# SECTION 5: Word Frequency Analysis - Positive vs Negative Reviews
# =========================================================================
print_section('5. WORD FREQUENCY ANALYSIS')

positive_reviews = df_text[df_text['review_rating'] >= 4]['review_text_clean'].tolist()
negative_reviews = df_text[df_text['review_rating'] <= 2]['review_text_clean'].tolist()

print(f'Positive reviews (4-5 stars): {len(positive_reviews):,}')
print(f'Negative reviews (1-2 stars): {len(negative_reviews):,}')

pos_words = get_word_frequencies(positive_reviews, top_n=25)
neg_words = get_word_frequencies(negative_reviews, top_n=25)

print('\nTop 25 Words in POSITIVE Reviews (4-5 stars):')
for i, (word, count) in enumerate(pos_words, 1):
    print(f'  {i:>2}. {word:<20} {count:>5,} mentions')

print('\nTop 25 Words in NEGATIVE Reviews (1-2 stars):')
for i, (word, count) in enumerate(neg_words, 1):
    print(f'  {i:>2}. {word:<20} {count:>5,} mentions')

# Chart: Side-by-side word frequencies
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Positive words
pos_df = pd.DataFrame(pos_words[:20], columns=['word', 'count'])
axes[0].barh(pos_df['word'][::-1], pos_df['count'][::-1], color='#2ecc71', edgecolor='white')
axes[0].set_title('Top 20 Words in Positive Reviews (4-5 stars)')
axes[0].set_xlabel('Frequency')
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

# Negative words
neg_df = pd.DataFrame(neg_words[:20], columns=['word', 'count'])
axes[1].barh(neg_df['word'][::-1], neg_df['count'][::-1], color='#e74c3c', edgecolor='white')
axes[1].set_title('Top 20 Words in Negative Reviews (1-2 stars)')
axes[1].set_xlabel('Frequency')
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'word_frequency_comparison.png'))
plt.close()
print('\nSaved: outputs/word_frequency_comparison.png')


# =========================================================================
# SECTION 6: Common Complaint Themes (Negative Reviews)
# =========================================================================
print_section('6. COMMON COMPLAINT THEMES (1-2 Star Reviews)')

# Define theme keywords - what patterns signal specific complaints
complaint_themes = {
    'Quality and Durability': ['broke', 'broken', 'cheap', 'flimsy', 'fell', 'apart', 'crack',
                               'cracked', 'defective', 'poor', 'quality', 'rust', 'rusted',
                               'weak', 'fragile', 'junk', 'garbage', 'trash', 'wore', 'worn'],
    'Shipping and Delivery': ['shipping', 'shipped', 'delivery', 'delivered', 'package',
                              'packaging', 'damaged', 'broken', 'arrived', 'missing',
                              'lost', 'late', 'delayed', 'ups', 'fedex', 'box'],
    'Not as Described': ['description', 'described', 'picture', 'pictured', 'photo',
                         'different', 'expected', 'misleading', 'false', 'wrong',
                         'color', 'size', 'smaller', 'bigger', 'advertised'],
    'Difficult Assembly': ['assembly', 'assemble', 'instructions', 'install', 'installation',
                           'difficult', 'hard', 'confusing', 'complicated', 'pieces',
                           'parts', 'missing', 'holes', 'align', 'fit'],
    'Customer Service': ['customer', 'service', 'support', 'return', 'returned', 'refund',
                         'replacement', 'warranty', 'response', 'contact', 'phone',
                         'email', 'manager', 'store', 'exchange'],
    'Value for Money': ['price', 'expensive', 'overpriced', 'money', 'waste', 'worth',
                        'cost', 'paid', 'cheap', 'rip', 'ripoff', 'dollar', 'dollars'],
    'Malfunction': ['stopped', 'working', 'broken', 'motor', 'leak', 'leaking', 'leaks',
                    'malfunction', 'defect', 'noise', 'noisy', 'loud', 'failed', 'failure',
                    'dead', 'replacement'],
}

neg_texts = df_text[df_text['review_rating'] <= 2]['review_text_clean'].str.lower()

theme_counts = {}
for theme, keywords in complaint_themes.items():
    pattern = '|'.join(keywords)
    count = neg_texts.str.contains(pattern, na=False).sum()
    theme_counts[theme] = count

theme_counts = dict(sorted(theme_counts.items(), key=lambda x: x[1], reverse=True))

print(f'Analysis of {len(neg_texts):,} negative reviews (1-2 stars):')
print(f'(Reviews can match multiple themes)\n')
for theme, count in theme_counts.items():
    pct = count / len(neg_texts) * 100
    bar = '#' * int(pct / 2)
    print(f'  {theme:<25} {count:>4} reviews ({pct:>5.1f}%)  {bar}')

# Chart: Complaint themes
fig, ax = plt.subplots(figsize=(10, 6))
themes = list(theme_counts.keys())
counts = list(theme_counts.values())
bars = ax.barh(themes[::-1], counts[::-1], color='#e74c3c', edgecolor='white', alpha=0.85)
ax.set_xlabel('Number of Negative Reviews Mentioning Theme')
ax.set_title('Common Complaint Themes in 1-2 Star Reviews')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for bar, count in zip(bars, counts[::-1]):
    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
            f'{count}', va='center', fontsize=10)
plt.savefig(os.path.join(OUTPUT_DIR, 'complaint_themes.png'))
plt.close()
print('\nSaved: outputs/complaint_themes.png')


# =========================================================================
# SECTION 7: Common Praise Themes (Positive Reviews)
# =========================================================================
print_section('7. COMMON PRAISE THEMES (4-5 Star Reviews)')

praise_themes = {
    'Easy Setup': ['easy', 'simple', 'quick', 'minutes', 'straightforward',
                   'effortless', 'snap', 'breeze', 'assemble', 'install'],
    'Appearance and Design': ['beautiful', 'gorgeous', 'pretty', 'stunning', 'attractive',
                              'elegant', 'stylish', 'design', 'color', 'colors', 'looks',
                              'aesthetic', 'decor', 'decorative', 'complement'],
    'Durability and Quality': ['sturdy', 'solid', 'durable', 'strong', 'heavy', 'duty',
                               'quality', 'built', 'robust', 'tough', 'reliable',
                               'well', 'constructed', 'thick', 'substantial'],
    'Value for Money': ['price', 'value', 'affordable', 'worth', 'deal', 'reasonable',
                        'inexpensive', 'bargain', 'money', 'budget', 'savings'],
    'Perfect Fit': ['perfect', 'fit', 'fits', 'exactly', 'precise', 'measurement',
                    'size', 'dimensions', 'space', 'room', 'kitchen', 'bathroom'],
    'Performance': ['works', 'working', 'performs', 'performance', 'efficient',
                    'effective', 'functional', 'quiet', 'powerful', 'fast',
                    'smooth', 'consistent', 'reliable', 'excellent'],
    'Recommendation': ['recommend', 'recommended', 'recommend', 'love', 'loved',
                       'amazing', 'fantastic', 'awesome', 'wonderful', 'outstanding',
                       'exceeded', 'expectations', 'happy', 'pleased', 'satisfied'],
}

pos_texts = df_text[df_text['review_rating'] >= 4]['review_text_clean'].str.lower()

praise_counts = {}
for theme, keywords in praise_themes.items():
    pattern = '|'.join(keywords)
    count = pos_texts.str.contains(pattern, na=False).sum()
    praise_counts[theme] = count

praise_counts = dict(sorted(praise_counts.items(), key=lambda x: x[1], reverse=True))

print(f'Analysis of {len(pos_texts):,} positive reviews (4-5 stars):')
print(f'(Reviews can match multiple themes)\n')
for theme, count in praise_counts.items():
    pct = count / len(pos_texts) * 100
    bar = '#' * int(pct / 2)
    print(f'  {theme:<25} {count:>5} reviews ({pct:>5.1f}%)  {bar}')

# Chart: Praise themes
fig, ax = plt.subplots(figsize=(10, 6))
themes_p = list(praise_counts.keys())
counts_p = list(praise_counts.values())
bars = ax.barh(themes_p[::-1], counts_p[::-1], color='#2ecc71', edgecolor='white', alpha=0.85)
ax.set_xlabel('Number of Positive Reviews Mentioning Theme')
ax.set_title('Common Praise Themes in 4-5 Star Reviews')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for bar, count in zip(bars, counts_p[::-1]):
    ax.text(bar.get_width() + 15, bar.get_y() + bar.get_height()/2,
            f'{count:,}', va='center', fontsize=10)
plt.savefig(os.path.join(OUTPUT_DIR, 'praise_themes.png'))
plt.close()
print('\nSaved: outputs/praise_themes.png')


# =========================================================================
# SECTION 8: Category-Level Satisfaction Comparison
# =========================================================================
print_section('8. CATEGORY-LEVEL SATISFACTION')

cat_stats = df.groupby('root_category').agg(
    avg_rating=('review_rating', 'mean'),
    review_count=('review_rating', 'count'),
    pct_5_star=('review_rating', lambda x: (x == 5).sum() / len(x) * 100),
    pct_1_star=('review_rating', lambda x: (x == 1).sum() / len(x) * 100),
).round(2)
cat_stats = cat_stats.sort_values('avg_rating', ascending=False)

print(f'{"Category":<25} {"Avg Rating":<12} {"Reviews":<10} {"5-Star %":<10} {"1-Star %":<10}')
print(f'{"-"*67}')
for cat, row in cat_stats.iterrows():
    print(f'{cat:<25} {row["avg_rating"]:<12.2f} {int(row["review_count"]):<10,} {row["pct_5_star"]:<10.1f} {row["pct_1_star"]:<10.1f}')

# Sentiment by category (for reviews with text)
if 'vader_compound' in df_text.columns:
    cat_sentiment = df_text.groupby('root_category')['vader_compound'].mean().sort_values(ascending=False)
    print('\nAverage VADER Sentiment by Category:')
    for cat, score in cat_sentiment.items():
        print(f'  {cat:<25} {score:.3f}')

# Chart: Category satisfaction
fig, ax = plt.subplots(figsize=(12, 6))
x_pos = range(len(cat_stats))
ax.bar(x_pos, cat_stats['avg_rating'], color=COLORS[:len(cat_stats)], edgecolor='white')
ax.set_xticks(x_pos)
ax.set_xticklabels(cat_stats.index, rotation=30, ha='right')
ax.set_ylabel('Average Review Rating')
ax.set_title('Average Customer Rating by Product Category')
ax.set_ylim(0, 5.5)
for i, (_, row) in enumerate(cat_stats.iterrows()):
    ax.text(i, row['avg_rating'] + 0.1, f'{row["avg_rating"]:.2f}',
            ha='center', fontsize=10, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig(os.path.join(OUTPUT_DIR, 'category_satisfaction.png'))
plt.close()
print('\nSaved: outputs/category_satisfaction.png')


# =========================================================================
# SECTION 9: Price vs Satisfaction Correlation
# =========================================================================
print_section('9. PRICE vs SATISFACTION')

df_price = df.dropna(subset=['product_price']).copy()

# Price bins
bins = [0, 25, 50, 100, 200, 500, 3000]
labels = ['Under $25', '$25-50', '$50-100', '$100-200', '$200-500', '$500+']
df_price['price_bin'] = pd.cut(df_price['product_price'], bins=bins, labels=labels)

price_rating = df_price.groupby('price_bin', observed=True).agg(
    avg_rating=('review_rating', 'mean'),
    review_count=('review_rating', 'count'),
    pct_5_star=('review_rating', lambda x: (x == 5).sum() / len(x) * 100),
).round(2)

print(f'{"Price Range":<15} {"Avg Rating":<12} {"Reviews":<10} {"5-Star %":<10}')
print(f'{"-"*47}')
for price_range, row in price_rating.iterrows():
    print(f'{str(price_range):<15} {row["avg_rating"]:<12.2f} {int(row["review_count"]):<10,} {row["pct_5_star"]:<10.1f}')

corr_price_rating = df_price['product_price'].corr(df_price['review_rating'])
print(f'\nCorrelation between price and rating: {corr_price_rating:.3f}')
if abs(corr_price_rating) < 0.1:
    print('Interpretation: Very weak correlation - price does not strongly predict satisfaction')
elif corr_price_rating > 0:
    print('Interpretation: Slight positive correlation - pricier products tend to rate slightly higher')
else:
    print('Interpretation: Slight negative correlation - pricier products tend to rate slightly lower')

# Chart: Price vs Rating
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(range(len(price_rating)), price_rating['avg_rating'],
       color=['#3498db', '#2ecc71', '#f39c12', '#e67e22', '#e74c3c', '#9b59b6'],
       edgecolor='white')
ax.set_xticks(range(len(price_rating)))
ax.set_xticklabels(price_rating.index, rotation=15)
ax.set_ylabel('Average Review Rating')
ax.set_title('Average Rating by Price Range')
ax.set_ylim(0, 5.5)
for i, (_, row) in enumerate(price_rating.iterrows()):
    ax.text(i, row['avg_rating'] + 0.1, f'{row["avg_rating"]:.2f}\n({int(row["review_count"])})',
            ha='center', fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig(os.path.join(OUTPUT_DIR, 'price_vs_satisfaction.png'))
plt.close()
print('\nSaved: outputs/price_vs_satisfaction.png')


# =========================================================================
# SECTION 10: Review Length vs Rating
# =========================================================================
print_section('10. REVIEW LENGTH vs RATING')

df_text['review_length'] = df_text['review_text_clean'].str.len()
df_text['word_count'] = df_text['review_text_clean'].str.split().str.len()

length_by_rating = df_text.groupby('review_rating').agg(
    avg_chars=('review_length', 'mean'),
    avg_words=('word_count', 'mean'),
    median_words=('word_count', 'median'),
).round(1)

print(f'{"Rating":<8} {"Avg Chars":<12} {"Avg Words":<12} {"Median Words":<14}')
print(f'{"-"*46}')
for rating, row in length_by_rating.iterrows():
    print(f'{rating:<8} {row["avg_chars"]:<12.0f} {row["avg_words"]:<12.0f} {row["median_words"]:<14.0f}')

print('\nInsight: ', end='')
longest_rating = length_by_rating['avg_words'].idxmax()
shortest_rating = length_by_rating['avg_words'].idxmin()
print(f'{longest_rating}-star reviews are longest (avg {length_by_rating.loc[longest_rating, "avg_words"]:.0f} words), '
      f'while {shortest_rating}-star reviews are shortest (avg {length_by_rating.loc[shortest_rating, "avg_words"]:.0f} words)')

# Chart: Review length by rating
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(length_by_rating.index, length_by_rating['avg_words'],
       color=['#e74c3c', '#e67e22', '#f39c12', '#2ecc71', '#27ae60'], edgecolor='white')
ax.set_xlabel('Star Rating')
ax.set_ylabel('Average Word Count')
ax.set_title('Average Review Length by Star Rating')
ax.set_xticks([1, 2, 3, 4, 5])
for i, (rating, row) in enumerate(length_by_rating.iterrows()):
    ax.text(rating, row['avg_words'] + 1, f'{row["avg_words"]:.0f}',
            ha='center', fontsize=10, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig(os.path.join(OUTPUT_DIR, 'review_length_by_rating.png'))
plt.close()
print('\nSaved: outputs/review_length_by_rating.png')


# =========================================================================
# SECTION 11: Top Manufacturers by Customer Satisfaction
# =========================================================================
print_section('11. TOP MANUFACTURERS BY SATISFACTION')

# Only include manufacturers with a meaningful number of reviews
min_reviews = 15
mfr_stats = df.groupby('manufacturer').agg(
    avg_rating=('review_rating', 'mean'),
    review_count=('review_rating', 'count'),
    pct_5_star=('review_rating', lambda x: (x == 5).sum() / len(x) * 100),
    pct_negative=('review_rating', lambda x: (x <= 2).sum() / len(x) * 100),
    avg_price=('product_price', 'mean'),
).round(2)
mfr_stats = mfr_stats[mfr_stats['review_count'] >= min_reviews]
mfr_stats = mfr_stats.sort_values('avg_rating', ascending=False)

print(f'Manufacturers with {min_reviews}+ reviews, ranked by average rating:\n')
print(f'{"Manufacturer":<35} {"Avg Rating":<11} {"Reviews":<9} {"5-Star %":<10} {"Neg %":<8} {"Avg Price":<10}')
print(f'{"-"*83}')
for mfr, row in mfr_stats.iterrows():
    avg_price_str = f'${row["avg_price"]:.0f}' if not pd.isna(row['avg_price']) else 'N/A'
    print(f'{mfr:<35} {row["avg_rating"]:<11.2f} {int(row["review_count"]):<9,} '
          f'{row["pct_5_star"]:<10.1f} {row["pct_negative"]:<8.1f} {avg_price_str:<10}')

# Chart: Manufacturer ratings (top 15)
top_n = min(15, len(mfr_stats))
mfr_top = mfr_stats.head(top_n)
fig, ax = plt.subplots(figsize=(14, 7))
bars = ax.barh(range(top_n), mfr_top['avg_rating'].values,
               color=COLORS[:top_n] if top_n <= len(COLORS) else COLORS * 2,
               edgecolor='white')
ax.set_yticks(range(top_n))
ax.set_yticklabels(mfr_top.index)
ax.set_xlabel('Average Review Rating')
ax.set_title(f'Top {top_n} Manufacturers by Average Customer Rating (min {min_reviews} reviews)')
ax.set_xlim(0, 5.5)
ax.invert_yaxis()
for i, (_, row) in enumerate(mfr_top.iterrows()):
    ax.text(row['avg_rating'] + 0.05, i,
            f'{row["avg_rating"]:.2f} ({int(row["review_count"])} reviews)',
            va='center', fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig(os.path.join(OUTPUT_DIR, 'manufacturer_satisfaction.png'))
plt.close()
print('\nSaved: outputs/manufacturer_satisfaction.png')


# =========================================================================
# SECTION 12: Summary Statistics and Key Findings
# =========================================================================
print_section('12. SUMMARY STATISTICS AND KEY FINDINGS')

print(f'Dataset Overview:')
print(f'  Total reviews analyzed: {len(df):,}')
print(f'  Reviews with text: {len(df_text):,}')
print(f'  Unique products: {df["product_name"].nunique():,}')
print(f'  Unique manufacturers: {df["manufacturer"].nunique():,}')
print(f'  Product categories: {df["root_category"].nunique()} root categories, {df["category"].nunique()} subcategories')
print(f'  Price range: ${df["product_price"].min():.2f} to ${df["product_price"].max():,.2f}')

print(f'\nRating Summary:')
print(f'  Overall average rating: {avg_rating:.2f} out of 5')
print(f'  Positive reviews (4-5 stars): {positive:,} ({positive/len(df)*100:.1f}%)')
print(f'  Negative reviews (1-2 stars): {negative:,} ({negative/len(df)*100:.1f}%)')

print(f'\nSentiment Summary:')
vader_pos = (df_text['vader_label'] == 'Positive').sum()
print(f'  VADER detected positive sentiment: {vader_pos:,} ({vader_pos/len(df_text)*100:.1f}%)')
print(f'  VADER-to-Rating correlation: {corr_vader:.3f}')
print(f'  TextBlob-to-Rating correlation: {corr_tb:.3f}')

print(f'\nKey Findings:')
best_cat = cat_stats.index[0]
worst_cat = cat_stats.index[-1]
print(f'  1. {best_cat} has the highest satisfaction ({cat_stats.loc[best_cat, "avg_rating"]:.2f} avg rating)')
print(f'  2. {worst_cat} has the lowest satisfaction ({cat_stats.loc[worst_cat, "avg_rating"]:.2f} avg rating)')
top_complaint = list(theme_counts.keys())[0]
top_complaint_pct = list(theme_counts.values())[0] / len(neg_texts) * 100
print(f'  3. "{top_complaint}" is the most common complaint theme ({top_complaint_pct:.0f}% of negative reviews)')
top_praise = list(praise_counts.keys())[0]
top_praise_pct = list(praise_counts.values())[0] / len(pos_texts) * 100
print(f'  4. "{top_praise}" is the most common praise theme ({top_praise_pct:.0f}% of positive reviews)')
print(f'  5. Price has a {abs(corr_price_rating):.3f} correlation with satisfaction ({"weak" if abs(corr_price_rating) < 0.15 else "moderate"})')
print(f'  6. {longest_rating}-star reviews tend to be the longest, suggesting stronger opinions drive detailed writing')
best_mfr = mfr_stats.index[0]
print(f'  7. {best_mfr} leads in customer satisfaction ({mfr_stats.loc[best_mfr, "avg_rating"]:.2f} avg, {int(mfr_stats.loc[best_mfr, "review_count"])} reviews)')
mismatch_total = mask_mismatch_pos.sum() + mask_mismatch_neg.sum()
print(f'  8. {mismatch_total:,} reviews ({mismatch_total/len(df_text)*100:.1f}%) show sentiment-rating mismatches')

# Save summary stats to CSV
summary_data = {
    'Metric': [
        'Total Reviews', 'Reviews With Text', 'Unique Products', 'Unique Manufacturers',
        'Average Rating', 'Median Rating', 'Positive Reviews (4-5)', 'Negative Reviews (1-2)',
        'VADER Positive Sentiment %', 'VADER-Rating Correlation', 'Price-Rating Correlation',
    ],
    'Value': [
        len(df), len(df_text), df['product_name'].nunique(), df['manufacturer'].nunique(),
        round(avg_rating, 2), median_rating, positive, negative,
        round(vader_pos/len(df_text)*100, 1), round(corr_vader, 3), round(corr_price_rating, 3),
    ]
}
pd.DataFrame(summary_data).to_csv(os.path.join(OUTPUT_DIR, 'summary_statistics.csv'), index=False)
print('\nSaved: outputs/summary_statistics.csv')

# Save category stats
cat_stats.to_csv(os.path.join(OUTPUT_DIR, 'category_statistics.csv'))
print('Saved: outputs/category_statistics.csv')

# Save manufacturer stats
mfr_stats.to_csv(os.path.join(OUTPUT_DIR, 'manufacturer_statistics.csv'))
print('Saved: outputs/manufacturer_statistics.csv')

print('\n' + '=' * 70)
print('  ANALYSIS COMPLETE')
print('  All charts and data files saved to the outputs/ folder.')
print('=' * 70)
