# Customer Experience & Sentiment Analysis (Python)

[View Interactive Report](https://htmlpreview.github.io/?https://raw.githubusercontent.com/Tommy-Nguyen-Stonera/python-customer-sentiment-analysis/main/report.html)

## Overview

This project analyses 8,199 Home Depot customer reviews using VADER and TextBlob to understand what drives satisfaction and complaints across product categories. The goal was to find out whether star ratings and written sentiment actually align, which categories have real satisfaction problems, and whether price influences how customers feel.

## Dataset

- Source: Home Depot product reviews (home_depot_reviews.csv)
- Record count: 8,199 rows after deduplication
- Coverage: 944 products, 8 root categories, 20 manufacturers
- Key columns: product_name, manufacturer, root_category, category, product_rating, product_price, review_rating, review_title, review_text, reviewer_name
- Single flat table, no joins required

## Research Questions

1. Does what customers write in their review actually match the star rating they give?
2. Which categories have the highest and lowest satisfaction scores?
3. What complaint themes dominate negative reviews, and how concentrated are they?
4. What language do satisfied customers use vs dissatisfied customers?
5. Does price correlate with satisfaction, or do customers judge quality independently of what they paid?
6. Which manufacturers have the highest dissatisfaction rates?
7. Do longer reviews tend to come from less satisfied customers?

## Data Model

Single flat table: home_depot_reviews. Each row is one customer review with product attributes and text fields. VADER and TextBlob scores are computed at runtime and added as derived columns. No external joins needed.

## What Was Analysed

- VADER compound score vs star rating correlation across all 8,199 reviews
- Category-level average satisfaction ranking (highest to lowest)
- Keyword frequency analysis on negative reviews to surface complaint themes
- Comparative word frequency: positive reviews vs negative reviews
- Pearson correlation between product price and review rating
- Manufacturer dissatisfaction rate (share of 1-2 star reviews by manufacturer)
- Review length (word count) by star rating to test the longer-means-angrier hypothesis

## Key Insights

1. 85.5% of reviews are positive (4 to 5 stars), with an average rating of 4.41 out of 5. The dataset skews heavily positive.
2. VADER sentiment correlates 0.681 with star ratings, which is solid but not perfect. 3.9% of reviews show a clear mismatch where sentiment and stars point in opposite directions.
3. Quality and durability is the top complaint theme, appearing in 41% of negative reviews. Appearance and design is the top praise, appearing in 53% of positive reviews.
4. Lighting has the highest category satisfaction at 4.62 out of 5. Appliances has the lowest at 4.00, a gap large enough to reflect real structural differences in product quality or customer expectations.
5. Price has almost no correlation with satisfaction (r = -0.015). Customers do not rate more expensive products higher, and cheaper products do not get penalised.
6. Longer reviews do tend to come from less satisfied customers, though the effect is moderate. It signals that effort to write is often effort to complain.

## Recommendations

1. Focus product improvement investment on quality and durability, not aesthetics. Complaints concentrate there (41% of negative reviews), but praise goes to appearance (53% of positive reviews). The gap between what fails and what delights is a clear product brief.
2. Investigate the Appliances category satisfaction gap (4.00 vs 4.62 for Lighting). At this scale, a 0.62 gap in average rating represents a large volume of unhappy customers and real churn risk.
3. Do not use price tier as a proxy for customer satisfaction when planning the product range. The near-zero price-satisfaction correlation means premium pricing does not buy goodwill on its own.
4. Flag the 3.9% of sentiment-rating mismatches for manual review. These reviews often contain nuanced feedback that both star ratings and automated sentiment scoring miss.

## Tools

Python 3.12, Pandas, VADER, TextBlob, Matplotlib

## Files

- `analysis.py` - Full analysis script
- `report.html` - Interactive report with 10 embedded charts
- `data/home_depot_reviews.csv` - 8,199 reviews
- `outputs/` - 10 chart PNGs + 3 summary CSVs
