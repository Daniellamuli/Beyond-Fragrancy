# Beyond Fragrancy
### *say less. we know your scent.*

---

## The Idea

You know that feeling when someone walks past you and the whole room shifts? That's a perfume doing its job. But finding *your* version of that moment has always been weirdly hard. You either rely on a sales associate who's pushing whatever's on commission, spend money on something that smells completely different on your skin than it did on the tester strip, or just keep repurchasing the one safe option you found years ago because you don't know where to start with anything else.

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

---

## Business Understanding

This project sits at the intersection of machine learning, fragrance culture, and a gap in the African retail market that nobody has seriously addressed yet.

The recommendation engine uses content based filtering at its foundation, measuring cosine similarity between perfume note vectors to find scents with the closest DNA to what a user already loves. Over time, as users interact with the app and leave feedback, a collaborative filtering layer kicks in and the model gets sharper. The more people use it, the better it gets for everyone.

On the business side, the revenue model is affiliate commissions. Every time a user clicks through to purchase a perfume via a partner retailer, Beyond Fragrancy earns a percentage. The primary affiliate partners are Notino (which ships to Kenya and much of Africa), FragranceNet (global discount shipping), and Sephora via Impact.com for the international market. For Kenyan users specifically, Jumia Kenya is included as a local e-commerce option, and the app will surface physical store locations for retailers like Essenza, Rayan Perfumes, Carrefour, and Naivas where relevant stock is known.

The Kenyan and broader East African angle is not an afterthought. It is a deliberate strategic focus. No perfume discovery app currently exists that is built around local availability, warm climate performance preferences, or African price sensitivities. That is the gap we are walking into.

---

## Data Sources

We did the sniffing so you don't have to. Beyond Fragrancy is built 
on a master dataset of 150,288 unique perfumes assembled from three 
public Kaggle datasets and enriched through our own data pipeline.

### Primary Dataset
**Fragrantica.com Fragrance Dataset** by olgagmiufana1
https://www.kaggle.com/datasets/olgagmiufana1/fragrantica-com-fragrance-dataset

131,930 perfumes scraped from Fragrantica as of June 2026, making it 
one of the most current publicly available perfume datasets in existence. 
Each record includes name, brand, launch year, gender classification, 
top/middle/base notes, scent accords, ratings, longevity, sillage, 
price perception votes, seasonal suitability, and perfumer credits.

### Reference Tables
**FragDB Fragrance Database** by eriklindqvist
https://www.kaggle.com/datasets/eriklindqvist/fragdb-fragrance-database

Four structured lookup tables covering notes, accords, brands, and 
perfumers. Critical for standardizing note names across sources — 
solving the problem of "Bergamotte" and "Bergamot" being the same 
ingredient stored differently.

### Supplementary Dataset
**Fragrantica Perfumes** by ledecanteur
https://www.kaggle.com/datasets/ledecanteur/fragrantica-perfumes

70,100 additional perfume records with strong accord tagging. After 
deduplication against the primary dataset, this contributed 18,358 
genuinely new entries and filled 371 empty fields in existing records.

### The Master Dataset
All three sources are merged into a single master_dataset.csv through 
our data pipeline (see notebooks/). The result after deduplication, 
enrichment, and validation:

- 150,288 unique perfumes
- 141,975 with full notes data
- 147,108 with accord classifications  
- 11,857 flanker relationships mapped
- 9 derived features including olfactive family, occasion tags, 
  confidence scoring, and popularity weighting
- 95.51% average data completeness

To reproduce the master dataset, download the three Kaggle datasets 
above, place them in the data/ folder, and run the data pipeline notebook.

---


