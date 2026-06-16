<p align="center">
  <img src="app/assets/images/brand_logo_slogan.png" alt="Beyond Fragrancy Logo" width="800" style="display: block; border-radius: 12px 12px 0 0;"/>
  <img src="app/assets/images/hero.jpg" alt="Luxury fragrance" width="800" style="display: block; border-radius: 0 0 12px 12px;"/>
</p>

## The Idea

You know that feeling when someone walks past you and the whole room shifts? That is a perfume doing its job. But finding *your* version of that moment has always been weirdly hard. You either rely on a sales associate pushing whatever is on commission, spend money on something that smells completely different on your skin than it did on the tester strip, or just keep repurchasing the one safe option you found years ago because you do not know where to start.

Beyond Fragrancy exists to fix that. Notes? Budget? Vibe? Say less.

We are building an intelligent perfume recommendation engine that learns what you love, understands the DNA of a scent, and suggests fragrances that actually match your taste, your budget, and where you can go buy them today. Not just internationally. Locally too.

---

## The Problem

The global fragrance market is worth over $50 billion and growing. Yet the experience of discovering a new perfume in 2025 is still somehow stuck in the past. You walk into a store overwhelmed, you spray things on paper, your nose gives up after three, and you leave with nothing or the wrong thing.

Online is not much better. Reviews are scattered across forums, fragrance databases, and YouTube rabbit holes. Price comparisons are a separate exercise. And if you are in Nairobi, Lagos, or Accra, almost none of the recommendation tools that exist were built with you in mind. They do not know where you can buy locally. They do not factor in that you want something that performs in a warm, humid climate. They do not speak to your market at all.

The gap is real and it is big.

---

## Who This Is For

Beyond Fragrancy is built for two people simultaneously.

The first is the fragrance curious. Someone who knows they like Dior Sauvage or Chanel Chance but has no idea what to try next or why they liked those in the first place. They want guidance without gatekeeping.

The second is the budget conscious shopper. Someone who has seen Baccarat Rouge 540 all over their feed, wants to understand what makes it special, and needs to know whether there is a version of that experience that does not cost $400 or whether there is a local option that comes close.

Both of these people deserve a tool that takes them seriously. This is that tool.

---

## What It Does

At its core, Beyond Fragrancy is a scent recommendation engine with three layers working together:

**Scent matching.** You tell us perfumes you have loved, or you describe the notes and vibes you are drawn to, and we find fragrances with overlapping DNA. Bergamot forward, woody base, something that lasts. Got it.

**Budget filtering.** Every recommendation comes with a real price tier. Entry level. Mid range. Premium. Luxury. You set your range and we stay inside it. No suggestions you cannot act on.

**Buy links that make sense for where you are.** If you are in Nairobi, we tell you which local stockists carry it and link you to online retailers that actually ship to you. If you are in London or New York, you get the right international options with affiliate links so you can purchase directly.

**Dupe engine.** Cannot afford the original? We find you budget alternatives that share 80-99% of the scent DNA. Smell rich, not broke.

---

## Business Understanding

This project sits at the intersection of machine learning, fragrance culture, and a gap in the African retail market that nobody has seriously addressed yet.

The recommendation engine uses a hybrid content-based filtering approach at its foundation. Three separate TF-IDF vectorizers capture different aspects of a perfume's identity: one for raw ingredient notes, one for weighted accord profiles, and one for contextual signals like gender and season. These are combined into a single weighted matrix at 60% notes, 30% accords, and 10% context. Cosine similarity then measures how closely any two perfumes match in this multi-dimensional scent space.

The system layers five additional signals on top of similarity: a Bayesian popularity score that weights ratings by vote confidence, a perfumer affinity boost for fragrances by the same nose, a collaborative filtering proxy using Fragrantica's pre-computed similar perfume lists, olfactive family diversity enforcement, and smart flanker deduplication so the same fragrance does not appear in multiple versions.

On the business side, the revenue model is affiliate commissions. Every time a user clicks through to purchase a perfume via a partner retailer, Beyond Fragrancy earns a percentage. The primary affiliate partners are Notino (which ships to Kenya and much of Africa), FragranceNet (global discount shipping), and Sephora via Impact.com for the international market. For Kenyan users specifically, Jumia Kenya is included as a local e-commerce option, and the app surfaces physical store locations for retailers like Essenza, Rayan Perfumes, Carrefour, and Naivas where relevant stock is known.

The Kenyan and broader East African angle is not an afterthought. It is a deliberate strategic focus. No perfume discovery app currently exists that is built around local availability, warm climate performance preferences, or African price sensitivities. That is the gap we are walking into.

---

## Data Sources

We did the sniffing so you don't have to. Beyond Fragrancy is built on a master dataset of 150,288 unique perfumes assembled from three public Kaggle datasets and enriched through our own data pipeline. The primary dataset was scraped from Fragrantica as of June 2026, making this one of the most current publicly available perfume datasets in existence, including 2025 and 2026 releases.

### Primary Dataset
**Fragrantica.com Fragrance Dataset** by olgagmiufana1
https://www.kaggle.com/datasets/olgagmiufana1/fragrantica-com-fragrance-dataset

131,930 perfumes with name, brand, launch year, gender, top/middle/base notes, scent accords, ratings, longevity, sillage, price perception votes, seasonal suitability, perfumer credits, and cover image URLs.

### Reference Tables
**FragDB Fragrance Database** by eriklindqvist
https://www.kaggle.com/datasets/eriklindqvist/fragdb-fragrance-database

Four structured lookup tables covering notes, accords, brands, and perfumers. Critical for standardising note names across sources and solving the problem of variant spellings like "Bergamotte" and "Bergamot" being treated as different ingredients.

### Supplementary Dataset
**Fragrantica Perfumes** by ledecanteur
https://www.kaggle.com/datasets/ledecanteur/fragrantica-perfumes

70,100 additional perfume records with strong accord tagging. After deduplication against the primary dataset, this contributed 18,358 genuinely new entries and filled empty fields in existing records.

### The Master Dataset

All three sources are merged into a single master_dataset.csv through our data pipeline. The result after deduplication, enrichment, and validation:

- 150,288 unique perfumes across 4 sources
- 141,975 with full notes data
- 147,108 with accord classifications
- 11,857 flanker relationships mapped across fragrance families
- 7,366 dupe relationships identified between budget and luxury perfumes
- 49,923 perfumer credits linked
- 43,293 similar perfume relationships from Fragrantica community data
- 9 derived features including olfactive family, occasion tags, confidence scoring, and popularity weighting
- 95.51% average data completeness

To reproduce the master dataset, download the three Kaggle datasets above, place them in the data/ folder, and run the data pipeline notebook.

---

## Data Pipeline

The pipeline handles several non-trivial cleaning challenges:

**Encoding repair.** Perfume names from French, Portuguese, Arabic, and German sources contained corrupted characters from latin-1/utf-8 mismatches. A layered encoding fix resolves these before any other processing.

**Note standardisation.** 23,000+ unique note terms were extracted from the dataset. A frequency-based autocorrection system detected rare terms that were likely misspellings of common ones and mapped them to canonical forms, eliminating 830 variant terms. A protected terms list prevents false corrections between distinct ingredients that share string similarity.

**Accord parsing.** The accords column stored data as accord_name:percentage pairs. We parsed these properly, extracting both clean accord names for TF-IDF and strength scores for weighted feature engineering. This eliminated artefacts that were being generated by naive string processing.

**Flanker detection.** An algorithm strips concentration markers and variant suffixes from perfume names, then uses fuzzy matching within brand groups to identify which perfumes are variations of a common original. 11,857 flanker relationships were detected, grouping perfumes into families for better recommendation deduplication.

---

## Model Architecture

```
User Input
  [perfume names they love] or [notes description]
         |
Query Vector Construction
  Seed perfume TF-IDF vectors averaged
  + 20% soft signal from similar_perfumes (collaborative proxy)
         |
Cosine Similarity Search
  Against 150,288 perfume vectors
         |
Filter Layer
  Budget tier, gender, occasion, season
         |
Scoring
  60% TF-IDF similarity
  + 40% Bayesian popularity score
  + perfumer affinity boost
         |
Post-processing
  Flanker deduplication
  Olfactive family diversity enforcement
  Multi-level tie-breaking by rating
         |
Results
  Top N recommendations with buy links
```

---

## Key Results

Sanity check similarity scores after model tuning:

| Pair | Similarity | Expected |
|------|-----------|---------|
| Aventus vs Al Dur Al Maknoon Silver (known dupe) | 0.4179 | High |
| Aventus vs Aventus for Her (flanker) | 0.3245 | High |
| Aventus vs Amarige (different family) | 0.1004 | Low |
| Aventus vs Angel (different family) | 0.1109 | Low |

Recommendation test results:

- Aventus input correctly returns Club de Nuit Intense Man by Armaf as top result (the most widely recognised Aventus clone globally)
- Baccarat Rouge 540 dupe search correctly returns Club de Nuit Untold by Armaf and Amber Rouge by Orientica as top results (both well known in the fragrance community as BR540 alternatives)
- Vanilla oud amber budget query correctly surfaces Arabic budget houses Rasasi, Afnan, and Al Haramain

---

## Repository Structure

```
Beyond-Fragrancy/
  data/
    README.md              # Download instructions for datasets
    .gitkeep
  notebooks/
    beyond_fragrancy_data_pipeline.ipynb    # Data collection and merging
    beyond_fragrancy_recommender.ipynb      # Recommendation engine
    beyond_fragrancy_eda.ipynb              # Exploratory data analysis
  models/
    README.md              # Instructions to regenerate model files
    .gitkeep
  app/
    app.py                 # Streamlit application (in progress)
    assets/
      images/              # Brand imagery
      css/
        style.css          # Custom brand styling
  README.md
  .gitignore
```

---


