

# Issue: [Word Distribution Trends](https://github.com/humlab/welfare_state_analytics/issues/10)

To consider:

- How to normalize the distributions? Can we normalize (som of) each words distribution to 1.
- What clustering to use? Ward? Try K-means to start with.
- UX two select different kinds of clusters

## Compute goddness-of-fit to uniform distribution
1. Compute using chi-square test
1. Visualisera distribution of "godness value" ("how many word deviates how much")
1. List words that for which the null-hypothesis is true 0.05

## Compute goddness-of-fit to uniform distribution
1. Select n (=500) words that deviatesclusters that covers 9+% all words.
1. Select n clusters that covers 9+% all words.
1. Visualize single cluster
    - Mean / scatter / box-plot
1. Visualize all clusters
    - Each represented as a mean curve
1. Export individual words for each cluster
