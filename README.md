# Customer Experience & Sentiment Analysis (Python)

[View Interactive Report](https://htmlpreview.github.io/?https://raw.githubusercontent.com/Tommy-Nguyen-Stonera/python-customer-sentiment-analysis/main/report.html)

Sentiment analysis of 8,199 Home Depot customer reviews using VADER and TextBlob. Covers complaint themes, praise themes, category satisfaction, price correlation, and manufacturer rankings.

## Key Findings

- 85.5% of reviews are positive (4-5 stars), average rating 4.41 out of 5
- VADER sentiment correlates 0.681 with star ratings
- Quality and durability is the top complaint (41% of negative reviews)
- Appearance and design is the top praise (53% of positive reviews)
- Lighting has the highest satisfaction (4.62), Appliances the lowest (4.00)
- Price has almost no correlation with satisfaction (r = -0.015)
- 3.9% of reviews show sentiment-rating mismatches

## Tools

Python 3.12, Pandas, VADER, TextBlob, Matplotlib

## Files

- `analysis.py` - Full analysis script
- `report.html` - Interactive report with 10 embedded charts
- `data/home_depot_reviews.csv` - 8,199 reviews
- `outputs/` - 10 chart PNGs + 3 summary CSVs
