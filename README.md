# Customer Experience & Sentiment Analysis - Home Improvement Products

[View Interactive Report](https://htmlpreview.github.io/?https://github.com/Tommy-Nguyen-Stonera/python-customer-sentiment-analysis/blob/main/report.html)

## Problem Statement

I wanted to understand what makes customers happy or unhappy when buying home improvement products. With thousands of real reviews from Home Depot, I built a sentiment analysis pipeline to go beyond simple star ratings and dig into the language customers actually use. The goal was to uncover patterns that a business could act on, such as which product categories struggle the most, what complaints come up repeatedly, and whether the text of a review actually matches the star rating given.

## Dataset

The analysis uses 8,199 cleaned reviews (after removing 92 duplicates) from Home Depot, covering:

- 944 unique products across 8 root categories and 59 subcategories
- 20 manufacturers
- Price range from $3.98 to $2,608
- Product types including outdoor products, window treatments, holiday decorations, plumbing, lighting, appliances, and more

Columns: product_name, manufacturer, root_category, category, product_rating, product_price, review_rating, review_title, review_text, reviewer_name

## Methodology

1. **Data cleaning** - Removed 92 duplicate rows, handled missing values (12.6% of reviews had no text), and stripped promotional disclaimer text that appeared at the start of many reviews.

2. **Rating distribution analysis** - Examined how customers distribute their star ratings and calculated positive, neutral, and negative splits.

3. **Dual sentiment analysis** - Ran both VADER (rule-based, tuned for social media text) and TextBlob (pattern-based) on every review to get sentiment scores independent of star ratings.

4. **Sentiment-rating alignment** - Compared what customers wrote against the stars they gave. Flagged mismatches where positive language came with low ratings, or negative language came with high ratings.

5. **Word frequency analysis** - Extracted the most common words in positive (4-5 star) and negative (1-2 star) reviews to see what language each group gravitates toward.

6. **Theme extraction** - Used keyword-based pattern matching to categorize negative reviews into complaint themes and positive reviews into praise themes.

7. **Segmented analysis** - Broke down satisfaction by category, price range, manufacturer, and review length to identify where problems concentrate.

## Key Findings

- **85.5% of reviews are positive** (4-5 stars), with an overall average of 4.41 out of 5. The dataset skews heavily satisfied.

- **VADER sentiment correlates 0.681 with star ratings**, confirming that what people write generally matches what they rate. TextBlob was weaker at 0.532.

- **Quality and durability is the number one complaint**, appearing in 41% of all negative reviews. Words like "cheap," "broke," "flimsy," and "cracked" drive this theme.

- **Appearance and design is the top praise theme** at 53% of positive reviews. Customers care deeply about how products look, especially in outdoor and decorative categories.

- **Lighting products have the highest satisfaction** (4.62 avg rating), while Appliances rank lowest (4.00 avg). Appliances also had the highest 1-star rate at 14%.

- **Price has almost no correlation with satisfaction** (r = -0.015). Expensive products are not rated meaningfully higher or lower than cheap ones.

- **3.9% of reviews show sentiment-rating mismatches** - 192 reviews had positive language but low stars, and 91 had negative language but high stars. This suggests some customers write balanced text even when dissatisfied.

- **Home Decorators Collection and iFit have the highest dissatisfaction rates** among manufacturers with enough reviews, with 27% and 43% negative review rates respectively.

## Generated Outputs

All charts and data files are saved in the `outputs/` folder:

- `rating_distribution.png` - Bar chart of star rating distribution
- `sentiment_by_rating.png` - VADER and TextBlob scores by star rating
- `sentiment_vs_rating_scatter.png` - Scatter plot showing sentiment-rating alignment
- `word_frequency_comparison.png` - Top words in positive vs negative reviews
- `complaint_themes.png` - Most common complaint categories
- `praise_themes.png` - Most common praise categories
- `category_satisfaction.png` - Average rating by product category
- `price_vs_satisfaction.png` - Rating by price range
- `review_length_by_rating.png` - Average review length by star rating
- `manufacturer_satisfaction.png` - Manufacturer ranking by satisfaction
- `summary_statistics.csv` - Key metrics in tabular form
- `category_statistics.csv` - Category-level breakdown
- `manufacturer_statistics.csv` - Manufacturer-level breakdown

## How to Run

```bash
# Install dependencies
pip install pandas matplotlib textblob vaderSentiment wordcloud

# Run the analysis
python analysis.py
```

The script expects the dataset at `data/home_depot_reviews.csv` relative to the script location. All output files are written to the `outputs/` folder automatically.

## Tools Used

- Python 3.12
- pandas - data manipulation and aggregation
- matplotlib - chart generation
- VADER (vaderSentiment) - rule-based sentiment scoring tuned for product reviews
- TextBlob - secondary sentiment scoring for cross-validation
- NumPy - numerical operations
