# app/app.py — Beyond Fragrancy
# ─────────────────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import scipy.sparse as sp
import re
import json
import os
from datetime import date
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import hstack
from rapidfuzz import fuzz, process

# ── Page configuration — must be first Streamlit call ─────────────────
st.set_page_config(
    page_title        = "Beyond Fragrancy",
    page_icon         = "🖤",
    layout            = "wide",
    initial_sidebar_state = "collapsed"
)

# ── Load custom CSS ────────────────────────────────────────────────────
def load_css():
    css_path = os.path.join(
        os.path.dirname(__file__), 'assets', 'css', 'style.css'
    )
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f'<style>{f.read()}</style>',
                        unsafe_allow_html=True)

load_css()

# ── Load brand assets ──────────────────────────────────────────────────
ASSETS = os.path.join(os.path.dirname(__file__), 'assets', 'images')

def get_image_path(filename):
    path = os.path.join(ASSETS, filename)
    return path if os.path.exists(path) else None

# ── Model loading ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    """Load all model checkpoints. Cached so only runs once."""
    models_dir = os.path.join(
        os.path.dirname(__file__), '..', 'models'
    )

    with st.spinner("Loading Beyond Fragrancy..."):
        df = pd.read_pickle(
            os.path.join(models_dir, 'df_with_features_checkpoint.pkl')
        )
        tfidf = sp.load_npz(
            os.path.join(models_dir, 'tfidf_matrix_checkpoint.npz')
        )
        with open(os.path.join(models_dir, 'checkpoints.pkl'), 'rb') as f:
            checkpoints = pickle.load(f)

    return df, tfidf, checkpoints

df, tfidf_matrix, checkpoints = load_models()

notes_vectorizer   = checkpoints['notes_vectorizer']
accords_vectorizer = checkpoints['accords_vectorizer']
context_vectorizer = checkpoints['context_vectorizer']

MIN_FEATURE_LENGTH = 50

# ── Translation ────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def translate_to_english(text):
    """
    Detects non-English input and translates to English.
    Falls back gracefully if translation fails.
    """
    try:
        from deep_translator import GoogleTranslator
        from langdetect import detect
        lang = detect(text)
        if lang != 'en':
            translated = GoogleTranslator(
                source=lang, target='en'
            ).translate(text)
            return translated, lang
        return text, 'en'
    except Exception:
        return text, 'en'

# ── Buy link builder ───────────────────────────────────────────────────
def build_buy_links(perfume_name, brand, price_tier, user_location='KE'):
    """
    Builds affiliate buy links for a given perfume.
    Notino and FragranceNet ship to Kenya.
    Sephora for international users.
    Jumia for Kenyan users.
    """
    q = perfume_name.replace(' ', '+')
    links = []

    links.append({
        "retailer":  "Notino",
        "url":       f"https://www.notino.co.uk/search/?q={q}",
        "emoji":     "🌍",
        "note":      "Ships to Kenya",
        "ships_ke":  True
    })
    links.append({
        "retailer": "FragranceNet",
        "url":      f"https://www.fragrancenet.com/search?q={q}",
        "emoji":    "💰",
        "note":     "Discounted prices",
        "ships_ke": True
    })
    if user_location == 'KE':
        links.append({
            "retailer": "Jumia Kenya",
            "url":      f"https://www.jumia.co.ke/catalog/?q={q}",
            "emoji":    "🇰🇪",
            "note":     "Local delivery",
            "ships_ke": True
        })
    else:
        links.append({
            "retailer": "Sephora",
            "url":      f"https://www.sephora.com/search?keyword={q}",
            "emoji":    "✨",
            "note":     "USA/EU/UK",
            "ships_ke": False
        })
    return links

# ── Core recommender functions ─────────────────────────────────────────
def average_sparse_rows(matrix, indices):
    stacked  = sp.vstack([matrix[i] for i in indices])
    averaged = stacked.mean(axis=0)
    return sp.csr_matrix(np.asarray(averaged))

def find_perfume_index(name, df_source):
    name_lower = name.lower().strip()
    exact = df_source[
        df_source['name'].str.lower().str.strip() == name_lower
    ]
    if len(exact) > 0:
        return exact.index[0], df_source.loc[exact.index[0], 'name'], 100
    match = process.extractOne(
        name, df_source['name'].tolist(), scorer=fuzz.ratio
    )
    if match and match[1] >= 70:
        idx = df_source[df_source['name'] == match[0]].index[0]
        return idx, match[0], match[1]
    return None, None, 0

def get_current_season():
    month = date.today().month
    if month in [12, 1, 2]:  return 'Winter'
    elif month in [3, 4, 5]: return 'Spring'
    elif month in [6, 7, 8]: return 'Summer'
    else:                     return 'Autumn'

def recommend(
    perfume_names=None, notes_input=None,
    budget_tier=None, gender=None,
    occasion=None, n_results=10,
    show_dupes=False
):
    if not perfume_names and not notes_input:
        return None, []

    found_names = []
    seed_indices = []
    similar_pool = []

    if perfume_names:
        for name in perfume_names:
            idx, matched, score = find_perfume_index(name, df)
            if idx is None:
                continue
            found_names.append(
                f"{matched} by {df.loc[idx, 'brand']}"
            )
            seed_indices.append(idx)
            sim = df.loc[idx, 'similar_perfumes'] \
                  if 'similar_perfumes' in df.columns else None
            if pd.notna(sim) if sim is not None else False:
                similar_pool.extend(str(sim).split(', ')[:3])

        if not seed_indices:
            return None, []

        query_vector = average_sparse_rows(tfidf_matrix, seed_indices)

        similar_indices = []
        for sim_name in similar_pool:
            idx_s, _, sc = find_perfume_index(sim_name.strip(), df)
            if idx_s is not None and sc >= 80:
                similar_indices.append(idx_s)
        if similar_indices:
            sim_vec      = average_sparse_rows(
                tfidf_matrix, similar_indices
            )
            query_vector = (
                query_vector.multiply(0.80) + sim_vec.multiply(0.20)
            )
    else:
        cleaned    = re.sub(r'[^\w\s]', ' ', notes_input.lower())
        notes_part = notes_vectorizer.transform([cleaned]) * 0.60
        empty_acc  = sp.csr_matrix((1, accords_vectorizer.vocabulary_.__len__()
                                    if hasattr(accords_vectorizer, 'vocabulary_')
                                    else tfidf_matrix.shape[1] - notes_part.shape[1] - 3))
        empty_ctx  = sp.csr_matrix((1, 3))
        try:
            query_vector = hstack([notes_part, empty_acc, empty_ctx])
        except Exception:
            query_vector = notes_part

    similarity_scores         = cosine_similarity(
        query_vector, tfidf_matrix
    ).flatten()
    results                   = df.copy()
    results['similarity_score'] = similarity_scores

    if perfume_names:
        owned = [n.lower().strip() for n in perfume_names]
        results = results[
            ~results['name'].str.lower().str.strip().isin(owned)
        ]

    results = results[
        results['feature_string'].notna() &
        (results['feature_string'].str.len() >= MIN_FEATURE_LENGTH)
    ] if 'feature_string' in results.columns else results

    if budget_tier:
        f = results[results['price_tier'] == budget_tier]
        if len(f) >= 5:
            results = f

    if gender and gender != 'Any':
        g = gender.lower()
        gf = results[results['gender'].isin([g, 'unisex'])]
        if len(gf) >= 5:
            results = gf

    if occasion and occasion != 'Any':
        if 'occasion_tags' in results.columns:
            of = results[
                results['occasion_tags'].str.contains(
                    occasion.lower(), case=False, na=False
                )
            ]
            if len(of) >= 5:
                results = of

    if show_dupes:
        results = results[
            results['price_tier'].isin(['budget', 'mid'])
        ]

    scaler = MinMaxScaler()
    if len(results) > 1:
        results = results.copy()
        results['sim_norm'] = scaler.fit_transform(
            results[['similarity_score']]
        )
        if 'popularity_score' in results.columns:
            results['pop_norm'] = scaler.fit_transform(
                results[['popularity_score']].fillna(0)
            )
        else:
            results['pop_norm'] = 0
    else:
        results['sim_norm'] = results.get('similarity_score', 0)
        results['pop_norm'] = 0

    results['final_score'] = (
        0.60 * results['sim_norm'] +
        0.40 * results['pop_norm']
    )

    results['r_fill'] = results['rating_avg'].fillna(0)
    results['rc_fill'] = results['rating_count'].fillna(0)
    results = results.sort_values(
        ['final_score', 'r_fill', 'rc_fill'],
        ascending=[False, False, False]
    )

    if 'flanker_group' in results.columns:
        results = results.drop_duplicates(
            subset=['flanker_group'], keep='first'
        )

    return results.head(n_results), found_names

# ── UI Components ──────────────────────────────────────────────────────

def render_header():
    """Renders the hero header with logo and tagline."""
    hero_path = get_image_path('hero.jpg')

    if hero_path:
        import base64
        with open(hero_path, 'rb') as f:
            hero_b64 = base64.b64encode(f.read()).decode()
        hero_css = f"""
        <style>
        .hero-section {{
            background-image: linear-gradient(
                to bottom,
                rgba(10,10,10,0.55) 0%,
                rgba(10,10,10,0.80) 60%,
                rgba(10,10,10,1.00) 100%
            ),
            url('data:image/jpeg;base64,{hero_b64}');
            background-size: cover;
            background-position: center 40%;
            padding: 4rem 2rem 3rem;
            text-align: center;
            border-bottom: 0.5px solid #2a2a2a;
        }}
        </style>
        """
        st.markdown(hero_css, unsafe_allow_html=True)

    logo_path = get_image_path('brand_logo_slogan.png')
    if logo_path:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(logo_path, use_container_width=True)
    else:
        st.markdown("""
        <div style='text-align:center; padding: 2rem 0;'>
            <h1 style='font-family: Playfair Display SC, serif;
                       color: #C9A84C; font-size: 2.5rem;
                       letter-spacing: 0.15em;'>
                Beyond Fragrancy
            </h1>
            <p style='color: #888; letter-spacing: 0.2em;
                      font-size: 0.85rem; text-transform: uppercase;'>
                say less. we know your scent.
            </p>
        </div>
        """, unsafe_allow_html=True)

def render_recommendation_card(row, idx, user_location='KE'):
    """Renders a single perfume recommendation card."""
    name       = str(row.get('name', 'Unknown'))
    brand      = str(row.get('brand', ''))
    price_tier = str(row.get('price_tier', 'mid'))
    accords    = str(row.get('accords', ''))
    rating     = row.get('rating_avg')
    r_count    = row.get('rating_count')
    sim_score  = row.get('similarity_score', 0)
    dupe_of    = row.get('dupe_of')
    image_url  = row.get('image_url')
    gender_val = str(row.get('gender', ''))
    season_val = str(row.get('best_season', ''))

    tier_colors = {
        'budget':  '#7A8C7E',
        'mid':     '#C9A84C',
        'premium': '#E8C87A',
        'luxury':  '#FFD700'
    }
    tier_color = tier_colors.get(price_tier, '#C9A84C')

    match_pct  = int(sim_score * 100)
    accord_list = [a.strip() for a in accords.split(',')
                   if a.strip()][:5]
    accord_tags = ' · '.join(accord_list) if accord_list else ''

    buy_links  = build_buy_links(name, brand, price_tier, user_location)

    with st.container():
        st.markdown(f"""
        <div style='
            background: #111111;
            border: 0.5px solid #2a2a2a;
            border-radius: 12px;
            padding: 1.4rem;
            margin-bottom: 1rem;
            transition: border-color 0.2s;
        '>
            <div style='display:flex; justify-content:space-between;
                        align-items:flex-start; margin-bottom: 0.6rem;'>
                <div>
                    <div style='font-family: Playfair Display SC, serif;
                                color: #F5F0E8; font-size: 1.05rem;
                                margin-bottom: 2px;'>
                        {name}
                    </div>
                    <div style='color: #666; font-size: 0.82rem;
                                letter-spacing: 0.06em;'>
                        {brand}
                    </div>
                </div>
                <div style='text-align:right;'>
                    <div style='color: {tier_color};
                                font-size: 0.75rem;
                                letter-spacing: 0.12em;
                                text-transform: uppercase;
                                border: 0.5px solid {tier_color}44;
                                padding: 3px 10px; border-radius: 3px;
                                margin-bottom: 4px;'>
                        {price_tier}
                    </div>
                    <div style='color: #C9A84C; font-size: 0.82rem;'>
                        {match_pct}% match
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        if image_url and str(image_url) not in ['nan', 'None', '']:
            try:
                st.image(str(image_url), width=80)
            except Exception:
                pass

        if accord_tags:
            st.markdown(f"""
            <div style='color: #555; font-size: 0.78rem;
                        margin: 0.5rem 0; line-height: 1.6;'>
                {accord_tags}
            </div>
            """, unsafe_allow_html=True)

        meta_parts = []
        if rating and not pd.isna(rating):
            meta_parts.append(f"⭐ {rating:.2f}")
        if r_count and not pd.isna(r_count):
            meta_parts.append(f"{int(r_count):,} votes")
        if gender_val and gender_val != 'nan':
            meta_parts.append(gender_val)
        if season_val and season_val != 'nan':
            meta_parts.append(season_val)

        if meta_parts:
            st.markdown(f"""
            <div style='color: #444; font-size: 0.75rem;
                        margin-bottom: 0.8rem;'>
                {' · '.join(meta_parts)}
            </div>
            """, unsafe_allow_html=True)

        if dupe_of and str(dupe_of) not in ['nan', 'None']:
            st.markdown(f"""
            <div style='color: #8B3A52; font-size: 0.75rem;
                        border-top: 0.5px solid #1e1e1e;
                        padding-top: 0.5rem; margin-bottom: 0.6rem;'>
                💡 Smells like {dupe_of}
            </div>
            """, unsafe_allow_html=True)

        cols = st.columns(len(buy_links))
        for col, link in zip(cols, buy_links):
            with col:
                st.markdown(
                    f"[{link['emoji']} {link['retailer']}]({link['url']})",
                    unsafe_allow_html=False
                )

        st.markdown("</div>", unsafe_allow_html=True)

# ── Main App ───────────────────────────────────────────────────────────

def main():
    render_header()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Input mode selector ────────────────────────────────────────────
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        mode = st.radio(
            "How would you like to find your scent?",
            ["By perfume I love", "By notes & vibe", "Find a dupe"],
            horizontal=True,
            label_visibility="collapsed"
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Input fields based on mode ─────────────────────────────────
        perfume_names = None
        notes_input   = None
        show_dupes    = False

        if mode == "By perfume I love":
            raw_input = st.text_input(
                "Enter perfumes you love (comma separated)",
                placeholder="e.g. Dior Sauvage, Chanel Chance, Aventus",
                label_visibility="visible"
            )
            if raw_input.strip():
                raw_input, detected_lang = translate_to_english(raw_input)
                if detected_lang != 'en':
                    st.info(f"Input translated from {detected_lang}")
                perfume_names = [
                    p.strip() for p in raw_input.split(',')
                    if p.strip()
                ]

        elif mode == "By notes & vibe":
            notes_raw = st.text_input(
                "Describe your ideal scent",
                placeholder="e.g. vanilla oud amber warm spicy, "
                            "or: something fresh for the office",
                label_visibility="visible"
            )
            if notes_raw.strip():
                notes_input, detected_lang = translate_to_english(
                    notes_raw
                )
                if detected_lang != 'en':
                    st.info(f"Input translated from {detected_lang}")

        else:  # Find a dupe
            dupe_input = st.text_input(
                "Which perfume are you looking for a dupe of?",
                placeholder="e.g. Baccarat Rouge 540, Creed Aventus",
                label_visibility="visible"
            )
            if dupe_input.strip():
                perfume_names = [dupe_input.strip()]
                show_dupes    = True

        # ── Filters ───────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("Filters (optional)", expanded=False):
            fcol1, fcol2, fcol3, fcol4 = st.columns(4)
            with fcol1:
                budget = st.selectbox(
                    "Budget",
                    ["Any", "budget", "mid", "premium", "luxury"],
                    index=0
                )
            with fcol2:
                gender = st.selectbox(
                    "Gender",
                    ["Any", "female", "male", "unisex"],
                    index=0
                )
            with fcol3:
                occasion = st.selectbox(
                    "Occasion",
                    ["Any", "office", "date", "casual",
                     "evening", "wedding"],
                    index=0
                )
            with fcol4:
                location = st.selectbox(
                    "Your location",
                    ["Kenya (KE)", "International"],
                    index=0
                )
                user_loc = 'KE' if 'Kenya' in location else 'INT'

            n_results = st.slider(
                "Number of recommendations", 5, 20, 10
            )

        budget_filter   = None if budget   == 'Any' else budget
        gender_filter   = None if gender   == 'Any' else gender
        occasion_filter = None if occasion == 'Any' else occasion

        # ── Run recommender ───────────────────────────────────────────
        run = st.button(
            "Find my scent →",
            type="primary",
            use_container_width=True
        )

        if run and (perfume_names or notes_input):
            with st.spinner("We did the sniffing so you don't have to..."):
                results, found = recommend(
                    perfume_names = perfume_names,
                    notes_input   = notes_input,
                    budget_tier   = budget_filter,
                    gender        = gender_filter,
                    occasion      = occasion_filter,
                    n_results     = n_results,
                    show_dupes    = show_dupes
                )

            if results is None or len(results) == 0:
                st.warning(
                    "No results found. Try adjusting your filters "
                    "or entering a different perfume name."
                )
            else:
                if found:
                    st.markdown(f"""
                    <div style='color: #666; font-size: 0.8rem;
                                margin-bottom: 1rem;'>
                        Based on: {', '.join(found)}
                    </div>
                    """, unsafe_allow_html=True)

                season_label = get_current_season()
                st.markdown(f"""
                <div style='
                    font-family: Playfair Display SC, serif;
                    color: #C9A84C;
                    font-size: 0.8rem;
                    letter-spacing: 0.2em;
                    text-transform: uppercase;
                    margin-bottom: 1.2rem;
                    padding-bottom: 0.6rem;
                    border-bottom: 0.5px solid #2a2a2a;
                '>
                    {"Budget alternatives" if show_dupes else "Your recommendations"}
                    &nbsp;·&nbsp; {season_label}
                </div>
                """, unsafe_allow_html=True)

                for idx, row in results.iterrows():
                    render_recommendation_card(row, idx, user_loc)

        elif run:
            st.warning("Please enter a perfume name or describe your scent.")

    # ── Footer ─────────────────────────────────────────────────────────
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='
        text-align: center;
        padding: 2rem;
        border-top: 0.5px solid #1a1a1a;
        color: #333;
        font-size: 0.75rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
    '>
        No gatekeeping. Just good smells.
        &nbsp;·&nbsp;
        Beyond Fragrancy
        &nbsp;·&nbsp;
        Data sourced from Fragrantica
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()