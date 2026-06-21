# app/app.py — Beyond Fragrancy
import streamlit as st
import pandas as pd
import numpy as np
import json, re, os, base64
import scipy.sparse as sp
from datetime import date
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import hstack
from rapidfuzz import fuzz, process

st.set_page_config(
    page_title="Beyond Fragrancy",
    page_icon="🖤",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE = os.path.dirname(os.path.abspath(__file__))
IMGS = os.path.join(BASE, "assets", "images")
MODS = os.path.join(BASE, "..", "models")

SEASON_MAP = {
    'Summer': 'Hot & Humid',
    'Winter': 'Cool & Dry',
    'Spring': 'Fresh & Mild',
    'Autumn': 'Warm & Transitional'
}
REVERSE_SEASON = {v: k for k, v in SEASON_MAP.items()}
SEASON_EMOJI   = {
    'Hot & Humid': '🔥', 'Cool & Dry': '🌬️',
    'Fresh & Mild': '🌤️', 'Warm & Transitional': '🍂'
}

PRICE_LABELS = {
    'budget': 'Under $30',
    'mid':    '$30 – $80',
    'premium':'$80 – $150',
    'luxury': '$150+'
}

BRAND_FIX = {
    "Victoria s Secret": "Victoria's Secret",
    "O Boticario":       "O Boticário",
    "Bath Body Works":   "Bath & Body Works",
}

NOTE_EMOJIS = {
    'rose':'🌹', 'jasmine':'🌸', 'vanilla':'🍦', 'musk':'🌫️',
    'cedar':'🌲', 'sandalwood':'🪵', 'oud':'🪔', 'bergamot':'🍋',
    'amber':'🟡', 'patchouli':'🍂', 'vetiver':'🌿', 'leather':'🧥',
    'citrus':'🍊', 'lavender':'💜', 'iris':'💐', 'pineapple':'🍍',
    'peach':'🍑', 'apple':'🍎', 'tobacco':'🪴', 'coffee':'☕',
    'chocolate':'🍫', 'pepper':'🌶️', 'grapefruit':'🍈',
    'lemon':'🍋', 'lime':'🍈', 'frankincense':'✨', 'saffron':'🔶',
}

MIN_SIM = 0.20

DUPE_BOOSTS = {
    'Club de Nuit Untold': 0.92,
    'Amber Rouge': 0.88,
    'Club de Nuit Intense Man': 0.90,
    'Al Dur Al Maknoon Silver': 0.85,
    'Armaf Club de Nuit': 0.88,
    'Orientica Amber Rouge': 0.86,
    'Afnan Supremacy Silver': 0.85,
    'Afnan Supremacy Not Only Intense': 0.83,
    'Lattafa Al Dur Al Maknoon': 0.82,
    'Al Haramain Amber Oud': 0.80,
    'Rasasi Al Wisam': 0.78,
    "Lattafa Qaa'ed": 0.76,
    'Haan Amber': 0.75,
    'Ruby Whispers': 0.72,
    'Aquila': 0.70,
    'Baccarat Rouge 540 Scented Hair Mist': 0.75,
    'Flora by Gucci Gorgeous Gardenia': 0.85,
    'Flora Gorgeous Gardenia Limited Edition 2020': 0.85,
    'Flora Gorgeous Gardenia Limited Edition 2018': 0.85,
    'Flora Gorgeous Gardenia Eau de Parfum': 0.85,
    'Gucci Flora Gorgeous Gardenia': 0.85,
}

KNOWN_BRANDS = [
    'gucci', 'dior', 'chanel', 'armani', 'tom ford', 'ysl',
    'mugler', 'versace', 'paco rabanne', 'calvin klein',
    'hermes', 'jean paul gaultier', 'baccarat', 'creed',
    'kilian', 'mancera', 'montale', 'nishane', 'parfums de marly',
    'lattafa', 'armaf', 'afnan', 'al haramain', 'rasasi',
    'burberry', 'carolina herrera', 'givenchy', 'guerlain',
    'valentino', 'viktor rolf', 'prada', 'boss', 'zara',
    'kayali', 'hugo boss', 'lancome', 'estee lauder',
    # Multi-word brands
    'parfums de marly', 'carolina herrera', 'calvin klein',
    'tom ford', 'jean paul gaultier', 'paco rabanne',
    'yves saint laurent', 'victoria secret', 'bath body works',
]

VIBE_CATEGORIES = {
    "🍬 Sweet & Gourmand": {
        "keywords": ["vanilla", "caramel", "honey", "chocolate", "praline",
                     "toffee", "marshmallow", "sugar", "whiskey", "rum",
                     "malt", "coffee", "cacao", "candy", "gourmand",
                     "brown sugar", "syrup", "butter", "cream"],
        "image": "scent_gourmand.png",
        "description": "Delicious, comforting, and irresistibly sweet"
    },
    "🔥 Warm & Sensual": {
        "keywords": ["amber", "musk", "incense", "cardamom", "cinnamon",
                     "clove", "saffron", "oud", "labdanum", "styrax",
                     "resin", "balsamic", "opoponax"],
        "image": "scent_oriental.png",
        "description": "Rich, seductive, and deeply inviting"
    },
    "🌲 Woody & Bold": {
        "keywords": ["cedar", "sandalwood", "vetiver", "oakmoss", "pine",
                     "mahogany", "guaiac", "cypress", "teak", "ebony",
                     "earthy", "smoky"],
        "image": "scent_woody.png",
        "description": "Strong, grounded, and confidently masculine"
    },
    "🌹 Floral & Soft": {
        "keywords": ["rose", "jasmine", "peony", "tuberose", "magnolia",
                     "lily", "violet", "freesia", "gardenia", "ylang-ylang",
                     "mimosa", "orange blossom", "neroli", "heliotrope"],
        "image": "scent_floral.png",
        "description": "Delicate, feminine, and beautifully soft"
    },
    "🌊 Fresh & Clean": {
        "keywords": ["bergamot", "lemon", "lime", "grapefruit", "aquatic",
                     "marine", "mint", "basil", "neroli", "citron",
                     "ozonic", "aldehydic", "green"],
        "image": "scent_fresh.png",
        "description": "Crisp, energetic, and effortlessly clean"
    }
}

def get_note_emoji(note):
    nl = note.lower()
    for key, emoji in NOTE_EMOJIS.items():
        if key in nl:
            return emoji
    return '·'

def img_b64(filename):
    for ext in ['png', 'jpg', 'jfif', 'jpeg']:
        base = filename.rsplit('.', 1)[0] if '.' in filename else filename
        path = os.path.join(IMGS, f"{base}.{ext}")
        if os.path.exists(path):
            mime = 'jpeg' if ext in ['jpg', 'jfif', 'jpeg'] else 'png'
            with open(path, 'rb') as f:
                return f"data:image/{mime};base64,{base64.b64encode(f.read()).decode()}"
    return None

def inject_css():
    css_path = os.path.join(BASE, 'assets', 'css', 'style.css')
    if os.path.exists(css_path):
        with open(css_path, 'r') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    st.markdown("""
    <style>
    div[data-testid="stButton"] > button[kind="primary"] {
        display: block !important;
        width: 100% !important;
        height: 2.5rem !important;
        min-height: 2.5rem !important;
        font-size: 0.95rem !important;
        margin-top: 0 !important;
        background-color: #C9A84C !important;
        color: #0A0A0A !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
    }
    div[data-testid="stTextInput"] input {
        height: 2.5rem !important;
    }
    @media (max-width: 768px) {
        div[data-testid="stButton"] > button[kind="primary"] {
            font-size: 1rem !important;
            height: 2.75rem !important;
            min-height: 2.75rem !important;
        }
        div[data-testid="stTextInput"] input {
            height: 2.75rem !important;
        }
    }
    .stTabs [data-baseweb="tab-list"] {
        display: flex !important;
        justify-content: center !important;
        gap: 0.5rem !important;
        flex-wrap: wrap !important;
    }
    .category-card {
        background: #111;
        border: 0.5px solid #2a2a2a;
        border-radius: 12px;
        padding: 0.8rem;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s ease;
        height: 100%;
    }
    .category-card:hover {
        border-color: #C9A84C;
        transform: translateY(-3px);
        box-shadow: 0 4px 20px rgba(201,168,76,0.1);
    }
    .suggestion-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        background: #111;
        border: 0.5px solid #2a2a2a;
        border-radius: 10px;
        padding: 0.5rem;
        transition: all 0.3s ease;
        height: 100%;
        min-height: 100px;
    }
    .suggestion-container:hover {
        border-color: #C9A84C;
        transform: translateY(-2px);
    }
    .suggestion-image {
        width: 60px;
        height: 60px;
        object-fit: contain;
        border-radius: 6px;
        background: #161616;
        padding: 4px;
    }
    .suggestion-name {
        color: #F5F0E8;
        font-size: 0.65rem;
        text-align: center;
        margin-top: 0.3rem;
        line-height: 1.2;
        overflow: hidden;
        font-family: 'DM Sans', sans-serif;
    }
    .suggestion-brand {
        color: #555;
        font-size: 0.55rem;
        text-align: center;
        font-family: 'DM Sans', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

inject_css()

@st.cache_resource(show_spinner="Loading Beyond Fragrancy...")
def load_models():
    if not hasattr(np, '_core'):
        np._core = np.core

    df_path = os.path.join(MODS, 'df_app.csv')
    if not os.path.exists(df_path):
        raise FileNotFoundError(f"Data file not found at {df_path}")

    essential_columns = [
        'name', 'brand', 'price_tier', 'all_notes', 'accords',
        'rating_avg', 'rating_count', 'popularity_score', 'image_url',
        'url', 'gender', 'olfactive_family', 'best_season',
        'flanker_group', 'flanker_type', 'is_flanker', 'dupe_of',
        'perfumers', 'similar_perfumes', 'feature_string'
    ]
    try:
        df = pd.read_csv(df_path, low_memory=False, usecols=essential_columns)
    except ValueError:
        df = pd.read_csv(df_path, low_memory=False)
    df = df.reset_index(drop=True)

    for col in ['brand', 'gender', 'price_tier', 'olfactive_family', 'best_season']:
        if col in df.columns:
            df[col] = df[col].astype('category')

    tfidf_path = os.path.join(MODS, 'tfidf_matrix_checkpoint.npz')
    tfidf = sp.load_npz(tfidf_path) if os.path.exists(tfidf_path) else None

    def rebuild_vec(path):
        with open(path) as f:
            d = json.load(f)
        p   = d['params']
        vec = TfidfVectorizer(
            max_features=p['max_features'],
            ngram_range=tuple(p['ngram_range']),
            min_df=p['min_df'],
            max_df=p['max_df'],
            sublinear_tf=p['sublinear_tf']
        )
        vec.vocabulary_ = d['vocabulary']
        vec.idf_         = np.array(d['idf'])
        vec._tfidf._idf_diag = sp.diags(
            vec.idf_, offsets=0,
            shape=(len(vec.idf_), len(vec.idf_)),
            format='csr', dtype=np.float64
        )
        return vec

    nv = rebuild_vec(os.path.join(MODS, 'notes_vocab.json'))
    av = rebuild_vec(os.path.join(MODS, 'accords_vocab.json'))
    cv = rebuild_vec(os.path.join(MODS, 'context_vocab.json'))

    df['brand'] = df['brand'].apply(
        lambda b: BRAND_FIX.get(str(b).strip(), str(b).strip())
    )
    df['weather_label'] = (
        df['best_season'].map(SEASON_MAP).fillna('')
        if 'best_season' in df.columns else ''
    )

    return df, tfidf, nv, av, cv

try:
    df, tfidf_matrix, notes_vec, accords_vec, context_vec = load_models()
    MODEL_OK  = True
    all_names = df['name'].dropna().tolist()
except Exception as e:
    MODEL_OK  = False
    MODEL_ERR = str(e)
    all_names = []

# ── Precomputed lookup structures (built once, not per-keystroke) ──────────
# These used to be recomputed with .str.contains()/.str.lower() scans
# inside find_idx() and get_flanker_suggestions_cached() on every call,
# which is what made name search slow. Building them once at startup
# turns those repeated O(n) string scans into O(1) dict lookups.
if MODEL_OK:
    df['_name_lower']  = df['name'].astype(str).str.lower().str.strip()
    df['_brand_lower'] = df['brand'].astype(str).str.lower().str.strip()

    # exact name -> first matching index
    NAME_TO_IDX = {}
    for _idx, _nl in zip(df.index, df['_name_lower']):
        if _nl not in NAME_TO_IDX:
            NAME_TO_IDX[_nl] = _idx

    # brand (lowercase) -> list of row indices, for fast brand scoping
    BRAND_TO_INDICES = {}
    for _idx, _bl in zip(df.index, df['_brand_lower']):
        BRAND_TO_INDICES.setdefault(_bl, []).append(_idx)

    # brand substring -> list of row indices (replaces .str.contains scans)
    # only needed for partial/substring brand matches
    def _brand_contains_indices(substr):
        return [idx for bl, idxs in BRAND_TO_INDICES.items()
                if substr in bl for idx in idxs]
else:
    NAME_TO_IDX = {}
    BRAND_TO_INDICES = {}
    def _brand_contains_indices(substr):
        return []

def get_weather_now():
    m   = date.today().month
    raw = ('Summer' if m in [6,7,8] else
           'Winter' if m in [12,1,2] else
           'Spring' if m in [3,4,5] else 'Autumn')
    return SEASON_MAP.get(raw, raw)

def rescale(sim):
    return int(max(0.0, min(1.0, float(sim))) * 100)

def match_color(pct):
    if pct >= 75: return '#4CAF50'
    if pct >= 55: return '#C9A84C'
    if pct >= 40: return '#E8C87A'
    return '#888'

def avg_sparse(mat, idxs):
    stacked = sp.vstack([mat[i] for i in idxs])
    return sp.csr_matrix(np.asarray(stacked.mean(axis=0)))

def find_idx(name):
    nl = name.lower().strip()

    # Exact match first — O(1) dict lookup instead of a full-column scan
    direct = NAME_TO_IDX.get(nl)
    if direct is not None:
        return direct, df.loc[direct, 'name'], 100

    # Try to detect brand + perfume name pattern
    words = nl.split()
    brand_match = None
    perfume_term = nl

    # Check for multi-word brands first (e.g., "Parfums de Marly")
    brand_candidates = []
    for i in range(len(words)):
        for j in range(i + 1, min(i + 4, len(words) + 1)):
            candidate = ' '.join(words[i:j])
            if candidate in KNOWN_BRANDS:
                brand_candidates.append((candidate, i, j))

    # Sort by length (longest brand match first)
    brand_candidates.sort(key=lambda x: len(x[0]), reverse=True)

    if brand_candidates:
        brand_match = brand_candidates[0][0]
        # Remove brand from search term
        perfume_term = nl.replace(brand_match, '').strip()
        # Clean up extra spaces
        perfume_term = re.sub(r'\s+', ' ', perfume_term).strip()

    # If brand detected, search within that brand first
    if brand_match:
        brand_idxs = _brand_contains_indices(brand_match)
        if brand_idxs and perfume_term:
            # Try exact match within brand
            for idx in brand_idxs:
                if df.at[idx, '_name_lower'] == perfume_term:
                    return idx, df.at[idx, 'name'], 100

            # Fuzzy match within brand
            brand_names = df.loc[brand_idxs, 'name'].tolist()
            m = process.extractOne(
                perfume_term, brand_names, scorer=fuzz.ratio
            )
            if m and m[1] >= 70:
                matched_name = m[0]
                idx = next(i for i in brand_idxs
                           if df.at[i, 'name'] == matched_name)
                return idx, matched_name, m[1]
        elif brand_idxs:
            # If only brand was provided, return the most popular from that brand
            brand_sub = df.loc[brand_idxs]
            top_brand = brand_sub.nlargest(1, 'popularity_score')
            if len(top_brand) > 0:
                idx = top_brand.index[0]
                return idx, top_brand.iloc[0]['name'], 80

    # Fallback to general fuzzy search
    m = process.extractOne(name, all_names, scorer=fuzz.ratio)
    if m and m[1] >= 70:
        idx = NAME_TO_IDX.get(m[0].lower().strip())
        if idx is None:
            idx = df[df['name'] == m[0]].index[0]
        return idx, m[0], m[1]

    return None, None, 0

BRAND_INDEX = {}

def build_brand_index():
    """Kept for compatibility — BRAND_INDEX is now just a thin view over
    the BRAND_TO_INDICES dict built once at startup, so this is O(1)
    instead of re-scanning the whole dataframe."""
    global BRAND_INDEX
    if not BRAND_INDEX and MODEL_OK:
        for bl, idxs in BRAND_TO_INDICES.items():
            if bl and bl not in ('nan', 'none', ''):
                BRAND_INDEX[bl] = df.loc[idxs, 'name'].tolist()

@st.cache_data(ttl=3600, show_spinner=False)
def get_flanker_suggestions_cached(query, n=6):
    build_brand_index()
    last = query.split(',')[-1].strip()
    if len(last) < 3:
        return []

    last_lower = last.lower()
    brand_match = None
    for word in last_lower.split():
        if word in KNOWN_BRANDS:
            brand_match = word
            break

    if brand_match and brand_match in BRAND_INDEX:
        pool = BRAND_INDEX[brand_match][:100]
    else:
        pool = (
            df.nlargest(800, 'popularity_score')['name'].tolist()
            if 'popularity_score' in df.columns else all_names[:800]
        )

    # token_sort_ratio is much cheaper than partial_ratio and works well
    # for "did you mean" style prefix/word matching on perfume names.
    matches = process.extract(
        last, pool, scorer=fuzz.token_sort_ratio, limit=n + 5
    )
    seen, results = set(), []
    for name, score, _ in matches:
        if score < 65 or len(name.strip()) < 3:
            continue
        if name.lower().strip() == last_lower.strip():
            continue
        if name not in seen:
            seen.add(name)
            results.append(name)
        if len(results) >= n:
            break

    if not results and brand_match:
        brand_idxs = BRAND_TO_INDICES.get(brand_match, [])
        results = df.loc[brand_idxs, 'name'].tolist()[:4]

    return results

def get_perfume_image_by_name(name):
    idx = NAME_TO_IDX.get(name.lower().strip())
    if idx is None:
        return None
    img = df.at[idx, 'image_url']
    return str(img) if pd.notna(img) and str(img) not in ['nan','None',''] else None

def get_perfume_brand_by_name(name):
    idx = NAME_TO_IDX.get(name.lower().strip())
    if idx is None:
        return ''
    return df.at[idx, 'brand']

@st.cache_data(show_spinner=False)
def get_category_perfumes(category_keywords, min_matches=2, limit=12):
    scores = []
    for idx, row in df.iterrows():
        notes = str(row.get('all_notes', '')).lower()
        if not notes:
            continue
        n_matches = sum(1 for kw in category_keywords if kw in notes)
        if n_matches >= min_matches:
            scores.append({
                'idx':        idx,
                'row':        row,
                'matches':    n_matches,
                'popularity': row.get('popularity_score', 0)
            })
    scores.sort(key=lambda x: (x['matches'], x['popularity']), reverse=True)
    result_rows = [s['row'] for s in scores[:limit]]
    return pd.DataFrame(result_rows) if result_rows else pd.DataFrame()

def recommend(perfume_names=None, notes_input=None, n=12, dupes_only=False):
    if not perfume_names and not notes_input:
        return None, []

    found, seed_idx, sim_pool = [], [], []

    if perfume_names:
        for name in perfume_names:
            idx, matched, score = find_idx(name)
            if idx is None:
                continue
            found.append(f"{matched} by {df.loc[idx,'brand']}")
            seed_idx.append(idx)
            if 'similar_perfumes' in df.columns:
                sp_val = df.loc[idx, 'similar_perfumes']
                if pd.notna(sp_val):
                    sim_pool.extend(str(sp_val).split(', ')[:3])

        if not seed_idx:
            return None, []

        qvec = avg_sparse(tfidf_matrix, seed_idx)
        sim_idxs = []
        for sn in sim_pool:
            si, _, sc = find_idx(sn.strip())
            if si is not None and sc >= 80:
                sim_idxs.append(si)
        if sim_idxs:
            sv   = avg_sparse(tfidf_matrix, sim_idxs)
            qvec = qvec.multiply(0.8) + sv.multiply(0.2)
    else:
        clean = re.sub(r'[^\w\s]', ' ', notes_input.lower())
        np_   = notes_vec.transform([clean]) * 0.60
        n_acc = max(tfidf_matrix.shape[1] - np_.shape[1] - 3, 1)
        ea    = sp.csr_matrix((1, n_acc))
        ec    = sp.csr_matrix((1, 3))
        try:
            qvec = hstack([np_, ea, ec])
        except Exception:
            qvec = np_

    sims        = cosine_similarity(qvec, tfidf_matrix).flatten()
    res         = df.copy()
    res['_sim'] = sims

    for boost_name, boost_score in DUPE_BOOSTS.items():
        mask = res['name'].str.lower() == boost_name.lower()
        if mask.any():
            cur = res.loc[mask, '_sim'].values[0]
            if cur < boost_score:
                res.loc[mask, '_sim'] = min(cur + 0.25, 0.95)

    if perfume_names:
        owned = [n.lower().strip() for n in perfume_names]
        res   = res[~res['name'].str.lower().str.strip().isin(owned)]

    if 'feature_string' in res.columns:
        res = res[res['feature_string'].str.len().fillna(0) >= 50]

    res = res[res['_sim'] >= MIN_SIM]

    if dupes_only:
        res = res[res['price_tier'].isin(['budget', 'mid'])]

    if len(res) == 0:
        return None, found

    sc  = MinMaxScaler()
    res = res.copy()
    res['_pn'] = (
        sc.fit_transform(res[['popularity_score']].fillna(0))
        if 'popularity_score' in res.columns else 0
    )
    # Ranking is driven by true match quality (_sim, raw cosine similarity —
    # the same number shown on the card as "X% match"). Popularity only
    # nudges the order slightly so that among very close matches, the
    # better-known perfume can edge ahead; it can no longer pull a
    # weaker match above a meaningfully stronger one.
    res['_score'] = 0.90 * res['_sim'] + 0.10 * res['_pn']
    res['_rf']    = res['rating_avg'].fillna(0)
    res['_rc']    = res['rating_count'].fillna(0)
    res = res.sort_values(['_score','_rf','_rc'], ascending=False)

    if 'flanker_group' in res.columns:
        res = res.drop_duplicates(subset=['flanker_group'], keep='first')

    res['_dd'] = (res['name'].str.lower().str.strip() + '|' +
                  res['brand'].str.lower().str.strip())
    res = res.drop_duplicates(subset=['_dd'], keep='first')

    return res.head(n), found

def build_retailer_links(name, url, loc='KE'):
    q     = name.replace(' ', '+')
    links = []

    if url and str(url) not in ['nan','None','']:
        frag_url = url if url.startswith('http') \
                   else f"https://www.fragrantica.com{url}"
        links.append(('📖', 'Fragrantica', frag_url, 'Perfume Info'))

    if loc == 'KE':
        links.append(('🛍️', 'Perfume Plug KE',
                       f'https://perfumeplugkenya.com/?s={q}', 'Kenya · Local'))
        links.append(('💄', 'Lintons Beauty',
                       f'https://www.lintonsbeauty.com/?s={q}', 'Kenya · Local'))
        links.append(('🌍', 'Notino',
                       f'https://www.notino.co.uk/search/?q={q}', "Int'l · Ships KE"))
        links.append(('💰', 'FragranceNet',
                       f'https://www.fragrancenet.com/fragrances?q={q}', "Int'l · Discounted"))
    else:
        links.append(('🌍', 'Notino',
                       f'https://www.notino.co.uk/search/?q={q}', 'UK / Europe'))
        links.append(('💰', 'FragranceNet',
                       f'https://www.fragrancenet.com/fragrances?q={q}', 'USA · Discounted'))

    return links

def render_card_html(row, loc='KE', show_match=True):
    name      = str(row.get('name', ''))
    brand     = str(row.get('brand', ''))
    tier      = str(row.get('price_tier', 'mid'))
    accords   = str(row.get('accords', '') or '')
    rating    = row.get('rating_avg')
    rcount    = row.get('rating_count')
    sim       = row.get('_sim', 0)
    dupe_of   = row.get('dupe_of')
    img_url   = row.get('image_url')
    url       = row.get('url', '')
    gender_v  = str(row.get('gender', '') or '').title()
    weather_v = str(row.get('weather_label', '') or '')
    notes_raw = str(row.get('all_notes', '') or '')

    tier_clr  = {'budget':'#7A8C7E','mid':'#C9A84C',
                 'premium':'#E8C87A','luxury':'#FFD700'}.get(tier,'#C9A84C')
    price_lbl = PRICE_LABELS.get(tier, tier)
    match_pct = rescale(sim)
    match_clr = match_color(match_pct)

    img_tag = ''
    if img_url and str(img_url) not in ['nan','None','']:
        img_tag = (
            f'<img src="{img_url}" style="width:100%;height:140px;'
            f'object-fit:contain;background:#161616;border-radius:8px;'
            f'padding:8px;margin-bottom:0.7rem;" />'
        )

    note_items = []
    for part in re.split(r'[|,]', notes_raw):
        p = part.strip()
        if p and p.lower() not in ['nan','none',''] and len(p) > 1:
            note_items.append(p)
        if len(note_items) >= 5:
            break

    pills_html = ''.join(
        f'<span style="display:inline-block;background:#1a1a1a;'
        f'border:0.5px solid #2a2a2a;color:#666;border-radius:10px;'
        f'padding:2px 8px;font-size:0.68rem;margin:2px 2px 2px 0;">'
        f'{get_note_emoji(n)} {n}</span>'
        for n in note_items
    )

    accord_items = [a.strip() for a in accords.split(',')
                    if a.strip() and a.lower() not in ['nan','']][:4]
    accord_line  = ' · '.join(accord_items)

    meta_parts = []
    if rating and not pd.isna(rating):
        meta_parts.append(f'⭐ {float(rating):.1f}')
    if rcount and not pd.isna(rcount):
        meta_parts.append(f'{int(rcount):,} reviews')
    if gender_v and gender_v not in ['Nan','None','']:
        meta_parts.append(gender_v)
    we = SEASON_EMOJI.get(weather_v,'')
    if weather_v and weather_v not in ['nan','None','']:
        meta_parts.append(f'{we} {weather_v}')
    meta_line = '  ·  '.join(meta_parts)

    dupe_str  = str(dupe_of) if dupe_of else ''
    dupe_html = (
        f'<div style="color:#8B3A52;font-size:0.72rem;'
        f'margin:0.4rem 0;padding-top:0.4rem;'
        f'border-top:0.5px solid #1e1e1e;">💡 Smells like {dupe_str}</div>'
    ) if dupe_str and dupe_str not in ['nan','None',''] else ''

    retailers  = build_retailer_links(name, url, loc)
    links_html = ''.join(
        f'<a href="{u}" target="_blank" style="display:inline-block;'
        f'margin:3px 4px 3px 0;text-decoration:none;color:#C9A84C;'
        f'font-size:0.72rem;border:0.5px solid #2a2a2a;border-radius:5px;'
        f'padding:3px 8px;">{e} {lbl}</a>'
        for e, lbl, u, _ in retailers
    )

    match_html = (
        f'<div style="text-align:right;flex-shrink:0;margin-left:0.6rem;">'
        f'<div style="color:{tier_clr};font-size:0.65rem;letter-spacing:0.08em;'
        f'border:0.5px solid {tier_clr}44;padding:2px 6px;border-radius:3px;'
        f'white-space:nowrap;">{price_lbl}</div>'
        f'<div style="color:{match_clr};font-size:1.1rem;font-weight:700;'
        f'margin-top:3px;">{match_pct}%</div>'
        f'<div style="color:#333;font-size:0.6rem;">match</div>'
        f'</div>'
    ) if show_match else (
        f'<div style="text-align:right;flex-shrink:0;margin-left:0.6rem;">'
        f'<div style="color:{tier_clr};font-size:0.65rem;letter-spacing:0.08em;'
        f'border:0.5px solid {tier_clr}44;padding:2px 6px;border-radius:3px;">'
        f'{price_lbl}</div></div>'
    )

    return f'''
    <div style="background:#111;border:0.5px solid #1e1e1e;border-radius:14px;
                padding:1.1rem;display:flex;flex-direction:column;">
      {img_tag}
      <div style="display:flex;justify-content:space-between;
                  align-items:flex-start;margin-bottom:0.4rem;">
        <div style="flex:1;min-width:0;">
          <div style="font-family:Playfair Display SC,serif;color:#F5F0E8;
                      font-size:0.95rem;white-space:nowrap;overflow:hidden;
                      text-overflow:ellipsis;">{name}</div>
          <div style="color:#666;font-size:0.78rem;margin-top:1px;
                      white-space:nowrap;overflow:hidden;
                      text-overflow:ellipsis;">{brand}</div>
        </div>
        {match_html}
      </div>
      {f'<div style="color:#555;font-size:0.74rem;margin-bottom:0.4rem;">{accord_line}</div>' if accord_line else ''}
      {f'<div style="margin-bottom:0.4rem;">{pills_html}</div>' if pills_html else ''}
      {f'<div style="color:#2e2e2e;font-size:0.7rem;margin-bottom:0.3rem;">{meta_line}</div>' if meta_line else ''}
      {dupe_html}
      <div style="margin-top:auto;padding-top:0.6rem;border-top:0.5px solid #1a1a1a;">
        {links_html}
      </div>
    </div>'''

def render_header():
    official = img_b64('official_logo')
    hero     = img_b64('hero')
    hero_css = ''
    if hero:
        hero_css = (
            f"background-image:linear-gradient("
            f"to bottom,rgba(10,10,10,0.2) 0%,"
            f"rgba(10,10,10,0.85) 70%,"
            f"rgba(10,10,10,1) 100%),"
            f"url('{hero}');"
            f"background-size:cover;"
            f"background-position:center 35%;"
        )
    img_part = (
        f"<img src='{official}' style='max-width:800px;width:95%;"
        f"display:block;margin:0 auto;' />"
    ) if official else (
        "<h1 style='font-family:Playfair Display SC,serif;"
        "color:#C9A84C;letter-spacing:0.15em;font-size:2rem;margin:0;'>"
        "Beyond Fragrancy</h1>"
        "<p style='color:#888;letter-spacing:0.2em;font-size:0.78rem;"
        "text-transform:uppercase;margin:0.3rem 0 0;'>"
        "say less. we know your scent.</p>"
    )
    st.markdown(
        f"<div style='{hero_css}padding:2.5rem 1rem 1.5rem;"
        f"text-align:center;border-bottom:0.5px solid #1a1a1a;'>"
        f"{img_part}</div>",
        unsafe_allow_html=True
    )

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="font-family:'Playfair Display SC',serif;color:#C9A84C;
                    font-size:0.8rem;letter-spacing:0.2em;text-transform:uppercase;
                    margin-bottom:1rem;padding-bottom:0.5rem;
                    border-bottom:0.5px solid #1a1a1a;">Refine Results</div>
        """, unsafe_allow_html=True)
        location = st.radio("loc", ["🇰🇪 Kenya","🌍 International"],
                            label_visibility="collapsed")
        loc = 'KE' if 'Kenya' in location else 'INT'
        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
        n_res = st.slider("n", 6, 24, 12, label_visibility="collapsed")
        st.markdown("""
        <div style="font-size:0.72rem;color:#333;line-height:1.6;
                    padding:0.8rem;background:#0d0d0d;border-radius:8px;
                    border:0.5px solid #1a1a1a;">
          <strong style="color:#555;">How matching works</strong><br>
          We compute scent DNA similarity across 150,000+ perfumes.
          Results above 70% are strong matches.
          Budget tiers are shown on each card.
        </div>
        """, unsafe_allow_html=True)
    return loc, n_res

def render_scent_explorer():
    st.markdown("""
    <div style="text-align:center;padding:1rem 0 0.2rem;">
      <h2 style="font-family:'Playfair Display SC',serif;color:#C9A84C;
                 font-size:1.1rem;letter-spacing:0.15em;margin-bottom:0;">
        🌿 Explore by Scent Family</h2>
      <p style="color:#555;font-size:0.8rem;">Browse perfumes by mood and style</p>
    </div>
    """, unsafe_allow_html=True)

    categories = [
        ("🍬 Sweet & Gourmand", "scent_gourmand.png",
         VIBE_CATEGORIES["🍬 Sweet & Gourmand"]["description"]),
        ("🔥 Warm & Sensual",  "scent_oriental.png",
         VIBE_CATEGORIES["🔥 Warm & Sensual"]["description"]),
        ("🌹 Floral & Soft",   "scent_floral.png",
         VIBE_CATEGORIES["🌹 Floral & Soft"]["description"]),
        ("🌲 Woody & Bold",    "scent_woody.png",
         VIBE_CATEGORIES["🌲 Woody & Bold"]["description"]),
        ("🌊 Fresh & Clean",   "scent_fresh.png",
         VIBE_CATEGORIES["🌊 Fresh & Clean"]["description"]),
    ]

    cols = st.columns(5)
    for i, (name, img, desc) in enumerate(categories):
        with cols[i]:
            b64 = img_b64(img)
            if b64:
                st.markdown(
                    f'<div class="category-card">'
                    f'<img src="{b64}" style="width:100%;border-radius:8px;'
                    f'max-height:70px;object-fit:cover;"/>'
                    f'<div style="color:#F5F0E8;font-size:0.7rem;font-weight:600;'
                    f'margin:0.3rem 0 0.1rem;font-family:Playfair Display SC,serif;">'
                    f'{name}</div>'
                    f'<div style="color:#555;font-size:0.55rem;">'
                    f'{desc[:28]}...</div></div>',
                    unsafe_allow_html=True
                )
            if st.button("Explore", key=f"cat_{i}", use_container_width=True):
                st.session_state['selected_category'] = name
                st.rerun()

def render_category_page(category_name):
    cat = VIBE_CATEGORIES.get(category_name)
    if not cat:
        return

    top_back_col, _ = st.columns([1, 4])
    with top_back_col:
        if st.button("← Back", use_container_width=True, key="back_top"):
            st.session_state['selected_category'] = None
            st.session_state['category_limit']    = 12
            st.rerun()

    st.markdown(
        f"<div style='text-align:center;padding:0.5rem 0;'>"
        f"<h2 style='font-family:Playfair Display SC,serif;color:#C9A84C;"
        f"font-size:1.3rem;'>{category_name}</h2>"
        f"<p style='color:#555;font-size:0.85rem;'>{cat['description']}</p>"
        f"</div>",
        unsafe_allow_html=True
    )

    b64 = img_b64(cat['image'])
    if b64:
        st.markdown(
            f'<img src="{b64}" style="max-height:120px;width:100%;'
            f'object-fit:cover;border-radius:12px;"/>',
            unsafe_allow_html=True
        )

    if 'category_limit' not in st.session_state:
        st.session_state['category_limit'] = 12

    with st.spinner("Finding perfumes..."):
        cat_df = get_category_perfumes(
            cat['keywords'], min_matches=2,
            limit=st.session_state['category_limit']
        )

    if len(cat_df) == 0:
        st.info("No perfumes found in this category.")
    else:
        st.markdown(
            f"<div style='margin:0.5rem 0;padding-bottom:0.3rem;"
            f"border-bottom:0.5px solid #1e1e1e;'>"
            f"<span style='font-family:Playfair Display SC,serif;"
            f"color:#C9A84C;font-size:0.7rem;letter-spacing:0.15em;'>"
            f"{len(cat_df)} PERFUMES IN THIS FAMILY</span></div>",
            unsafe_allow_html=True
        )
        cols = st.columns(3)
        for idx, (_, row) in enumerate(cat_df.iterrows()):
            with cols[idx % 3]:
                st.markdown(
                    render_card_html(row, 'KE', show_match=False),
                    unsafe_allow_html=True
                )
        if len(cat_df) >= st.session_state['category_limit']:
            _, mid, _ = st.columns([1,2,1])
            with mid:
                if st.button("📥 Load More", use_container_width=True):
                    st.session_state['category_limit'] += 12
                    st.rerun()

    if st.button("← Back to Categories", use_container_width=True, key="back_bottom"):
        st.session_state['selected_category'] = None
        st.session_state['category_limit']    = 12
        st.rerun()

def run_search(perfume_names, notes_input, n_res, dupes_only, loc):
    """Shared search logic called from both the button and suggestion clicks."""
    with st.spinner("We did the sniffing so you don't have to..."):
        results, found = recommend(
            perfume_names=perfume_names,
            notes_input=notes_input,
            n=n_res,
            dupes_only=dupes_only
        )

    if results is None or len(results) == 0:
        st.markdown("""
        <div style="text-align:center;padding:3rem 1rem;color:#333;">
          <div style="font-size:2.5rem;margin-bottom:0.5rem;">🤷</div>
          <p style="color:#555;">No strong matches found.</p>
          <p style="font-size:0.82rem;">
            Try a different name, check spelling,
            or describe the notes instead.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    if found:
        st.markdown(
            f"<p style='color:#333;font-size:0.76rem;margin:0.5rem 0 0.3rem;'>"
            f"Based on: {', '.join(found)}</p>",
            unsafe_allow_html=True
        )

    weather_now = get_weather_now()
    we          = SEASON_EMOJI.get(weather_now,'')
    label       = "Affordable alternatives" if dupes_only else "Recommended for you"

    st.markdown(
        f"<div style='display:flex;justify-content:space-between;"
        f"align-items:center;margin:0.8rem 0 1rem;"
        f"padding-bottom:0.5rem;border-bottom:0.5px solid #1e1e1e;'>"
        f"<span style='font-family:Playfair Display SC,serif;"
        f"color:#C9A84C;font-size:0.76rem;"
        f"letter-spacing:0.2em;text-transform:uppercase;'>{label}</span>"
        f"<span style='color:#2a2a2a;font-size:0.72rem;'>"
        f"{we} {weather_now} &nbsp;·&nbsp; {len(results)} results</span>"
        f"</div>",
        unsafe_allow_html=True
    )

    cols = st.columns(3)
    for idx, (_, row) in enumerate(results.iterrows()):
        with cols[idx % 3]:
            st.markdown(render_card_html(row, loc, show_match=True),
                        unsafe_allow_html=True)

    st.markdown(
        "<div style='text-align:center;padding:1rem 0;color:#222;"
        "font-size:0.76rem;'>Budget shown on each card.</div>",
        unsafe_allow_html=True
    )

def main():
    if not MODEL_OK:
        st.error(f"Model loading failed: {MODEL_ERR}")
        return

    render_header()
    loc, n_res = render_sidebar()

    # category page takes over the whole body
    if st.session_state.get('selected_category'):
        render_category_page(st.session_state['selected_category'])
        return

    st.markdown("""
    <div style="text-align:center;padding:0.5rem 1rem 0.1rem;
                max-width:600px;margin:0 auto;">
      <p style="color:#555;font-size:0.85rem;line-height:1.75;">
        Tell us a perfume you love, describe notes you are drawn to,
        or find an affordable dupe of your dream scent.
      </p>
    </div>
    """, unsafe_allow_html=True)

    _, c2, _ = st.columns([0.2, 5, 0.2])
    with c2:
        tab1, tab2, tab3 = st.tabs(
            ["🔍  Perfume", "🎵  Notes & Vibe", "💰  Find a Dupe"]
        )

        # shared state
        perfume_names  = None
        notes_input    = None
        dupes_only     = False
        any_search_ran = False  # tracks whether ANY tab rendered results this run

        # ── TAB 1: By perfume name ─────────────────────────────────────
        with tab1:
            st.markdown(
                "<p style='color:#444;font-size:0.82rem;margin:0 0 0.5rem;'>"
                "Enter perfumes you already love. "
                "Separate multiple with commas.</p>",
                unsafe_allow_html=True
            )

            # If a suggestion was just clicked, seed the widget's own
            # session-state key BEFORE the widget is created. Streamlit
            # widgets own their value via their `key` once instantiated,
            # so writing to session_state[key] (not just a separate
            # tracking variable) is the only reliable way to override
            # what's displayed/returned on the next run.
            if st.session_state.get('_apply_pf_value') is not None:
                st.session_state['pf_input'] = st.session_state.pop('_apply_pf_value')

            in_col, btn_col = st.columns([4, 1], vertical_alignment="bottom")
            with in_col:
                raw = st.text_input(
                    "pf",
                    placeholder="e.g. Dior Sauvage, Chanel Chance",
                    label_visibility="collapsed",
                    key="pf_input"
                )
            with btn_col:
                pf_go = st.button("Find my scent", type="primary",
                                   use_container_width=True, key="go_btn_pf")

            # suggestions — only show when user is actively typing
            # and has not yet clicked search
            if raw and len(raw.split(',')[-1].strip()) >= 3:
                with st.spinner("Looking for matches..."):
                    sugs = get_flanker_suggestions_cached(raw, 6)
                if sugs:
                    st.markdown(
                        "<p style='color:#333;font-size:0.72rem;"
                        "margin:0.4rem 0 0.3rem;'>Did you mean:</p>",
                        unsafe_allow_html=True
                    )
                    sug_cols = st.columns(min(len(sugs), 6))
                    for i, sug in enumerate(sugs):
                        with sug_cols[i]:
                            img_url = get_perfume_image_by_name(sug)
                            img_html = (
                                f'<img src="{img_url}" class="suggestion-image"/>'
                                if img_url else
                                '<div style="width:60px;height:60px;background:#1a1a1a;'
                                'border-radius:6px;display:flex;align-items:center;'
                                'justify-content:center;color:#333;margin:0 auto;">📷</div>'
                            )
                            sug_brand = get_perfume_brand_by_name(sug)

                            st.markdown(
                                f'<div class="suggestion-container">'
                                f'{img_html}'
                                f'<div class="suggestion-name">{sug}</div>'
                                f'<div class="suggestion-brand">{sug_brand}</div>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                            # Clicking the button stores the chosen name
                            # to be applied to the widget on rerun, and
                            # fires the search directly off that name
                            # (not off whatever's in session_state['pf_value'],
                            # which could otherwise be stale).
                            if st.button("Select", key=f"sug_{i}",
                                         use_container_width=True):
                                st.session_state['_apply_pf_value']  = sug
                                st.session_state['trigger_search']   = True
                                st.session_state['trigger_source']   = 'perfume'
                                st.session_state['trigger_value']    = sug
                                st.rerun()

            if raw and raw.strip():
                perfume_names = [p.strip() for p in raw.split(',')
                                 if p.strip()]

            tab1_trigger = pf_go or (
                st.session_state.get('trigger_search', False)
                and st.session_state.get('trigger_source') == 'perfume'
            )

            if tab1_trigger:
                if st.session_state.get('trigger_search', False):
                    perfume_names = [st.session_state.get('trigger_value', '')]
                    st.session_state['trigger_search'] = False
                    st.session_state['trigger_source']  = None
                    st.session_state['trigger_value']   = None

                if perfume_names:
                    run_search(perfume_names, None, n_res, False, loc)
                    any_search_ran = True
                else:
                    st.warning("Please enter a perfume name.")

        # ── TAB 2: By notes — no vibe buttons ─────────────────────────
        with tab2:
            st.markdown(
                "<p style='color:#444;font-size:0.82rem;margin:0 0 0.5rem;'>"
                "Describe what you want to smell like — ingredients, "
                "mood, or feeling.</p>",
                unsafe_allow_html=True
            )
            in_col, btn_col = st.columns([4, 1], vertical_alignment="bottom")
            with in_col:
                raw_notes = st.text_input(
                    "nt",
                    placeholder="e.g. warm vanilla oud, fresh citrus office",
                    label_visibility="collapsed",
                    key="nt_input"
                )
            with btn_col:
                nt_go = st.button("Find my scent", type="primary",
                                   use_container_width=True, key="go_btn_nt")

            if raw_notes and raw_notes.strip():
                notes_input = raw_notes.strip()

            if nt_go:
                if notes_input:
                    run_search(None, notes_input, n_res, False, loc)
                    any_search_ran = True
                else:
                    st.warning("Please describe the scent you're looking for.")

        # ── TAB 3: Find a dupe ─────────────────────────────────────────
        with tab3:
            st.markdown(
                "<p style='color:#444;font-size:0.82rem;margin:0 0 0.5rem;'>"
                "Enter a luxury perfume and we find budget-friendly "
                "alternatives that smell 80-99% similar.</p>",
                unsafe_allow_html=True
            )
            in_col, btn_col = st.columns([4, 1], vertical_alignment="bottom")
            with in_col:
                raw_dupe = st.text_input(
                    "dp",
                    placeholder="e.g. Baccarat Rouge 540, Creed Aventus",
                    label_visibility="collapsed",
                    key="dp_input"
                )
            with btn_col:
                dp_go = st.button("Find my scent", type="primary",
                                   use_container_width=True, key="go_btn_dp")

            if raw_dupe and raw_dupe.strip():
                perfume_names = [raw_dupe.strip()]
                dupes_only    = True

            if dp_go:
                if perfume_names:
                    run_search(perfume_names, None, n_res, dupes_only, loc)
                    any_search_ran = True
                else:
                    st.warning("Please enter a perfume name.")

    # scent explorer shown below when no tab just rendered search results
    if not any_search_ran:
        st.markdown("<div style='height:1rem;'></div>",
                    unsafe_allow_html=True)
        render_scent_explorer()

    st.markdown(
        "<div style='text-align:center;padding:2rem 1rem;"
        "border-top:0.5px solid #111;margin-top:2rem;"
        "color:#1a1a1a;font-size:0.68rem;"
        "letter-spacing:0.14em;text-transform:uppercase;'>"
        "No gatekeeping. Just good smells."
        " &nbsp;·&nbsp; Beyond Fragrancy"
        " &nbsp;·&nbsp; Data: Fragrantica"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()