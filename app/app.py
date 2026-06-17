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

# Known dupe and flanker boosting scores
DUPE_BOOSTS = {
    # Known dupes
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
    'Lattafa Qaa\'ed': 0.76,
    'Haan Amber': 0.75,
    'Ruby Whispers': 0.72,
    'Aquila': 0.70,
    'Baccarat Rouge 540 Scented Hair Mist': 0.75,
    
    # Flankers and alternate versions
    'Flora by Gucci Gorgeous Gardenia': 0.85,
    'Flora Gorgeous Gardenia Limited Edition 2020': 0.85,
    'Flora Gorgeous Gardenia Limited Edition 2018': 0.85,
    'Flora Gorgeous Gardenia Eau de Parfum': 0.85,
    'Gucci Flora Gorgeous Gardenia': 0.85,
}

# Known brands for suggestion filtering
KNOWN_BRANDS = [
    'gucci', 'dior', 'chanel', 'armani', 'tom ford', 'ysl', 
    'mugler', 'versace', 'paco rabanne', 'calvin klein', 
    'hermes', 'jean paul gaultier', 'baccarat', 'creed',
    'kilian', 'mancera', 'montale', 'nishane', 'parfums de marly',
    'lattafa', 'armaf', 'afnan', 'al haramain', 'rasasi',
    'burberry', 'carolina herrera', 'givenchy', 'guerlain',
    'valentino', 'viktor rolf', 'prada', 'boss', 'zara',
    'elie saab', 'estee lauder', 'clinique', 'lancome'
]

def get_note_emoji(note):
    nl = note.lower()
    for key, emoji in NOTE_EMOJIS.items():
        if key in nl:
            return emoji
    return '·'

def img_b64(filename):
    for ext in ['png','jpg','jfif','jpeg']:
        base = filename.rsplit('.', 1)[0] if '.' in filename else filename
        path = os.path.join(IMGS, f"{base}.{ext}")
        if os.path.exists(path):
            mime = 'jpeg' if ext in ['jpg','jfif','jpeg'] else 'png'
            with open(path, 'rb') as f:
                return f"data:image/{mime};base64,{base64.b64encode(f.read()).decode()}"
    return None

def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display+SC:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif !important;
        background-color: #0A0A0A !important;
        color: #F5F0E8;
    }
    .main { background-color: #0A0A0A !important; }
    .main .block-container {
        padding-top: 0 !important;
        padding-bottom: 3rem;
        max-width: 1100px;
    }
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }

    .stTabs [data-baseweb="tab-list"] {
        background: transparent !important;
        gap: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        background: #111 !important;
        color: #666 !important;
        border: 0.5px solid #2a2a2a !important;
        border-radius: 8px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.08em !important;
        padding: 0.4rem 1.4rem !important;
    }
    .stTabs [aria-selected="true"] {
        background: #C9A84C !important;
        color: #0A0A0A !important;
        border-color: #C9A84C !important;
    }
    .stTabs [data-baseweb="tab-border"] { display: none !important; }
    .stTabs [data-baseweb="tab-panel"] { padding-top: 0.8rem !important; }

    .stTextInput > div > div > input {
        background: #111 !important;
        border: 0.5px solid #2a2a2a !important;
        color: #F5F0E8 !important;
        border-radius: 8px;
        font-size: 1rem;
        padding: 0.7rem 1rem;
    }
    .stTextInput > div > div > input:focus {
        border-color: #C9A84C !important;
        box-shadow: 0 0 0 1px #C9A84C22 !important;
    }
    .stTextInput > div > div > input::placeholder { color: #444 !important; }
    .stTextInput label { display: none; }

    div[data-testid="stButton"] > button {
        background: #111;
        color: #666;
        border: 0.5px solid #2a2a2a;
        border-radius: 8px;
        font-family: 'DM Sans', sans-serif;
        font-size: 0.82rem;
        padding: 0.45rem 0.8rem;
        transition: all 0.2s;
        text-transform: none;
        letter-spacing: 0.04em;
        font-weight: 400;
        width: 100%;
    }
    div[data-testid="stButton"] > button:hover {
        border-color: #C9A84C !important;
        color: #C9A84C !important;
        background: #141414 !important;
    }
    div[data-testid="stButton"] > button[kind="primary"] {
        background: #C9A84C !important;
        color: #0A0A0A !important;
        border: none !important;
        font-size: 0.9rem !important;
        letter-spacing: 0.12em !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        padding: 0.7rem 1.5rem !important;
    }
    div[data-testid="stButton"] > button[kind="primary"]:hover {
        background: #E8C87A !important;
        color: #0A0A0A !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 20px #C9A84C22 !important;
    }

    [data-testid="stSidebar"] {
        background: #0d0d0d !important;
        border-right: 0.5px solid #1a1a1a;
    }
    [data-testid="stSidebar"] label { color: #666 !important; font-size: 0.8rem; }
    [data-testid="stSidebar"] .stSelectbox > div > div {
        background: #111 !important;
        border: 0.5px solid #2a2a2a !important;
        color: #F5F0E8 !important;
        border-radius: 6px;
    }
    [data-testid="stSidebar"] .stRadio label { color: #888 !important; }

    ::-webkit-scrollbar { width: 3px; }
    ::-webkit-scrollbar-track { background: #0A0A0A; }
    ::-webkit-scrollbar-thumb { background: #2a2a2a; border-radius: 2px; }
    </style>
    """, unsafe_allow_html=True)

inject_css()

@st.cache_resource(show_spinner="Loading Beyond Fragrancy...")
def load_models():
    df  = pd.read_csv(os.path.join(MODS, "df_app.csv"), low_memory=False)
    df  = df.reset_index(drop=True)
    mat = sp.load_npz(
        os.path.join(MODS, "tfidf_matrix_checkpoint.npz")
    ).tocsr()

    def rebuild_vec(path):
        with open(path) as f:
            d = json.load(f)
        p   = d['params']
        vec = TfidfVectorizer(
            max_features=p['max_features'],
            ngram_range=tuple(p['ngram_range']),
            min_df=p['min_df'], max_df=p['max_df'],
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

    nv = rebuild_vec(os.path.join(MODS, "notes_vocab.json"))
    av = rebuild_vec(os.path.join(MODS, "accords_vocab.json"))
    cv = rebuild_vec(os.path.join(MODS, "context_vocab.json"))

    df['brand'] = df['brand'].apply(
        lambda b: BRAND_FIX.get(str(b).strip(), str(b).strip())
    )
    if 'best_season' in df.columns:
        df['weather_label'] = df['best_season'].map(SEASON_MAP).fillna('')
    else:
        df['weather_label'] = ''

    return df, mat, nv, av, cv

try:
    df, tfidf_matrix, notes_vec, accords_vec, context_vec = load_models()
    MODEL_OK  = True
    all_names = df['name'].dropna().tolist()
except Exception as e:
    MODEL_OK  = False
    MODEL_ERR = str(e)
    all_names = []

def get_weather_now():
    m   = date.today().month
    raw = ('Summer' if m in [6,7,8] else
           'Winter' if m in [12,1,2] else
           'Spring' if m in [3,4,5] else 'Autumn')
    return SEASON_MAP.get(raw, raw)

def rescale(sim):
    c = max(0.0, min(1.0, float(sim)))
    return int(c * 100)

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
    ex = df[df['name'].str.lower().str.strip() == nl]
    if len(ex):
        return ex.index[0], df.loc[ex.index[0], 'name'], 100
    m = process.extractOne(name, all_names, scorer=fuzz.ratio)
    if m and m[1] >= 70:
        idx = df[df['name'] == m[0]].index[0]
        return idx, m[0], m[1]
    return None, None, 0

def get_flanker_suggestions(query, n=4):
    """
    Shows only relevant perfume suggestions based on the typed query.
    Filters out nonsense matches by checking if the suggestion is a real perfume.
    """
    last = query.split(',')[-1].strip()
    if len(last) < 3:
        return []

    words = last.split()
    brand_match = None
    
    for word in words:
        if word.lower() in KNOWN_BRANDS:
            brand_match = word.lower()
            break
    
    matches = process.extract(
        last, all_names, scorer=fuzz.partial_ratio, limit=30
    )
    
    seen, results = set(), []
    for name, score, _ in matches:
        if score < 60:
            continue
        if len(name.strip()) < 3:
            continue
        if name.lower().strip() == last.lower().strip():
            continue
        
        if brand_match:
            row = df[df['name'] == name]
            if len(row) > 0:
                brand = str(row.iloc[0].get('brand', '')).lower()
                if brand_match not in brand and brand_match not in name.lower():
                    continue
        
        query_words = set(last.lower().split())
        name_words = set(name.lower().split())
        overlap = query_words & name_words
        if not overlap and brand_match is None:
            if score < 80:
                continue
        
        if name not in seen:
            seen.add(name)
            results.append(name)
        if len(results) >= n:
            break
    
    if not results and brand_match:
        brand_perfumes = df[df['brand'].str.lower().str.contains(brand_match, na=False)]
        brand_names = brand_perfumes['name'].tolist()[:4]
        results = brand_names
    
    return results

def get_perfume_image(name):
    row = df[df['name'].str.lower() == name.lower()]
    if len(row) == 0:
        return None
    img = row.iloc[0].get('image_url')
    if pd.notna(img) and str(img) not in ['nan','None','']:
        return str(img)
    return None

def recommend(perfume_names=None, notes_input=None,
              n=12, dupes_only=False):
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

    # Dupe and flanker boosting
    for name, boost_score in DUPE_BOOSTS.items():
        mask = res['name'].str.lower() == name.lower()
        if mask.any():
            current_sim = res.loc[mask, '_sim'].values[0]
            if current_sim < boost_score:
                res.loc[mask, '_sim'] = min(current_sim + 0.25, 0.95)

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

    sc = MinMaxScaler()
    res = res.copy()
    res['_sn'] = sc.fit_transform(res[['_sim']])
    if 'popularity_score' in res.columns:
        res['_pn'] = sc.fit_transform(res[['popularity_score']].fillna(0))
    else:
        res['_pn'] = 0

    res['_score'] = 0.65 * res['_sn'] + 0.35 * res['_pn']
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
    """
    Builds retailer links for a perfume.
    Includes: Fragrantica (reliable info), FragranceNet, Notino (search links)
    """
    q = name.replace(' ', '+')
    links = []
    
    # Always add Fragrantica link (most reliable)
    if url and str(url) not in ['nan','None','']:
        fragrantica_url = url if url.startswith('http') else f"https://www.fragrantica.com{url}"
        links.append(('📖', 'Fragrantica', fragrantica_url, 'Perfume Info'))
    
    # International search links
    if loc == 'KE':
        links.append(('💰', 'Search: FragranceNet', 
                     f'https://www.fragrancenet.com/fragrances?q={q}', 
                     'Search manually'))
        links.append(('🌍', 'Search: Notino', 
                     f'https://www.notino.co.uk/search/?q={q}', 
                     'Search manually'))
    else:
        links.append(('💰', 'Search: FragranceNet', 
                     f'https://www.fragrancenet.com/fragrances?q={q}', 
                     'Search manually'))
        links.append(('🌍', 'Search: Notino', 
                     f'https://www.notino.co.uk/search/?q={q}', 
                     'Search manually'))
    
    return links

def render_card_html(row, loc='KE'):
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

    tier_clr   = {'budget':'#7A8C7E','mid':'#C9A84C',
                  'premium':'#E8C87A','luxury':'#FFD700'}.get(tier,'#C9A84C')
    price_lbl  = PRICE_LABELS.get(tier, tier)
    match_pct  = rescale(sim)
    match_clr  = match_color(match_pct)

    img_tag = ''
    if img_url and str(img_url) not in ['nan','None','']:
        img_tag = (
            f'<img src="{img_url}" '
            f'style="width:100%;height:140px;object-fit:contain;'
            f'background:#161616;border-radius:8px;padding:8px;'
            f'margin-bottom:0.7rem;" '
            f'onerror="this.style.display=\'none\'" />'
        )

    note_items = []
    for part in re.split(r'[|,]', notes_raw):
        p = part.strip()
        if p and p.lower() not in ['nan','none',''] and len(p) > 1:
            note_items.append(p)
        if len(note_items) >= 5:
            break

    pills_html = ''
    for note in note_items:
        emoji = get_note_emoji(note)
        pills_html += (
            f'<span style="display:inline-block;background:#1a1a1a;'
            f'border:0.5px solid #2a2a2a;color:#666;border-radius:10px;'
            f'padding:2px 8px;font-size:0.68rem;margin:2px 2px 2px 0;">'
            f'{emoji} {note}</span>'
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

    dupe_html = ''
    dupe_str  = str(dupe_of) if dupe_of else ''
    if dupe_str and dupe_str not in ['nan','None','']:
        dupe_html = (
            f'<div style="color:#8B3A52;font-size:0.72rem;'
            f'margin:0.4rem 0;padding-top:0.4rem;'
            f'border-top:0.5px solid #1e1e1e;">'
            f'💡 Smells like {dupe_str}</div>'
        )

    retailers   = build_retailer_links(name, url, loc)
    links_html  = ''
    for emoji, label, link_url, note in retailers:
        links_html += (
            f'<a href="{link_url}" target="_blank" '
            f'style="display:inline-block;margin:3px 4px 3px 0;'
            f'text-decoration:none;color:#C9A84C;font-size:0.72rem;'
            f'border:0.5px solid #2a2a2a;border-radius:5px;'
            f'padding:3px 8px;">'
            f'{emoji} {label}</a>'
        )

    card = f"""
    <div style="background:#111;border:0.5px solid #1e1e1e;border-radius:14px;padding:1.1rem;display:flex;flex-direction:column;transition:border-color 0.25s,box-shadow 0.25s,transform 0.2s;">
      {img_tag}
      <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:0.4rem;">
        <div style="flex:1;min-width:0;">
          <div style="font-family:'Playfair Display SC',serif;color:#F5F0E8;font-size:0.95rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{name}</div>
          <div style="color:#444;font-size:0.78rem;margin-top:1px;">{brand}</div>
        </div>
        <div style="text-align:right;flex-shrink:0;margin-left:0.6rem;">
          <div style="color:{tier_clr};font-size:0.65rem;letter-spacing:0.08em;border:0.5px solid {tier_clr}44;padding:2px 6px;border-radius:3px;white-space:nowrap;">{price_lbl}</div>
          <div style="color:{match_clr};font-size:1.1rem;font-weight:700;margin-top:3px;">{match_pct}%</div>
          <div style="color:#333;font-size:0.6rem;">match</div>
        </div>
      </div>
      {f'<div style="color:#555;font-size:0.74rem;margin-bottom:0.4rem;">{accord_line}</div>' if accord_line else ''}
      {f'<div style="margin-bottom:0.4rem;">{pills_html}</div>' if pills_html else ''}
      {f'<div style="color:#2e2e2e;font-size:0.7rem;margin-bottom:0.3rem;">{meta_line}</div>' if meta_line else ''}
      {dupe_html}
      <div style="margin-top:auto;padding-top:0.6rem;border-top:0.5px solid #1a1a1a;">
        {links_html}
      </div>
    </div>
    """
    return card

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

    img_part = ''
    if official:
        img_part = (
            f"<img src='{official}' "
            f"style='max-width:460px;width:80%;"
            f"display:block;margin:0 auto;' />"
        )
    else:
        img_part = (
            "<h1 style='font-family:Playfair Display SC,serif;"
            "color:#C9A84C;letter-spacing:0.15em;"
            "font-size:2rem;margin:0;'>Beyond Fragrancy</h1>"
            "<p style='color:#888;letter-spacing:0.2em;"
            "font-size:0.78rem;text-transform:uppercase;"
            "margin:0.3rem 0 0;'>say less. we know your scent.</p>"
        )

    st.markdown(
        f"<div style='{hero_css}padding:2.5rem 1rem 1.5rem;"
        f"text-align:center;border-bottom:0.5px solid #1a1a1a;'>"
        f"{img_part}</div>",
        unsafe_allow_html=True
    )

def render_sidebar(loc_default='KE'):
    with st.sidebar:
        st.markdown("""
        <div style="font-family:'Playfair Display SC',serif;
                    color:#C9A84C;font-size:0.8rem;
                    letter-spacing:0.2em;text-transform:uppercase;
                    margin-bottom:1rem;padding-bottom:0.5rem;
                    border-bottom:0.5px solid #1a1a1a;">
          Refine Results
        </div>
        """, unsafe_allow_html=True)

        st.markdown(
            "<p style='color:#555;font-size:0.75rem;"
            "margin-bottom:0.3rem;'>Location</p>",
            unsafe_allow_html=True
        )
        location = st.radio(
            "loc", ["🇰🇪 Kenya", "🌍 International"],
            label_visibility="collapsed"
        )
        loc = 'KE' if 'Kenya' in location else 'INT'

        st.markdown("<div style='height:0.8rem'></div>",
                    unsafe_allow_html=True)

        st.markdown(
            "<p style='color:#555;font-size:0.75rem;"
            "margin-bottom:0.3rem;'>Number of results</p>",
            unsafe_allow_html=True
        )
        n_res = st.slider("n", 6, 24, 12,
                          label_visibility="collapsed")

        st.markdown("<div style='height:0.8rem'></div>",
                    unsafe_allow_html=True)

        st.markdown("""
        <div style="font-size:0.72rem;color:#333;
                    line-height:1.6;padding:0.8rem;
                    background:#0d0d0d;border-radius:8px;
                    border:0.5px solid #1a1a1a;">
          <strong style="color:#555;">How matching works</strong><br>
          We compute scent DNA similarity across 150,000+ perfumes.
          Results above 70% are strong matches.
          Budget tiers are shown on each card — filter visually.
        </div>
        """, unsafe_allow_html=True)

    return loc, n_res

def main():
    if not MODEL_OK:
        st.error(f"Model loading failed: {MODEL_ERR}")
        st.info(
            "Ensure df_app.csv, tfidf_matrix_checkpoint.npz, "
            "notes_vocab.json, accords_vocab.json, "
            "context_vocab.json are in the models/ folder."
        )
        return

    render_header()
    loc, n_res = render_sidebar()

    st.markdown("""
    <div style="text-align:center;padding:1.2rem 1rem 0.3rem;
                max-width:600px;margin:0 auto;">
      <p style="color:#555;font-size:0.88rem;line-height:1.75;">
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

        perfume_names = None
        notes_input   = None
        dupes_only    = False
        raw           = ''

        with tab1:
            st.markdown(
                "<p style='color:#444;font-size:0.82rem;"
                "margin:0 0 0.5rem;'>Enter perfumes you already love. "
                "Separate multiple with commas.</p>",
                unsafe_allow_html=True
            )
            raw = st.text_input(
                "pf", placeholder="e.g. Dior Sauvage, Chanel Chance",
                label_visibility="collapsed", key="pf_input"
            )

            if raw and len(raw.split(',')[-1].strip()) >= 3:
                sugs = get_flanker_suggestions(raw, 4)
                if sugs:
                    st.markdown(
                        "<p style='color:#333;font-size:0.72rem;"
                        "margin:0.4rem 0 0.3rem;'>"
                        "Did you mean:</p>",
                        unsafe_allow_html=True
                    )
                    sug_cols = st.columns(min(len(sugs), 4))
                    for i, sug in enumerate(sugs):
                        img_preview = get_perfume_image(sug)
                        with sug_cols[i]:
                            if img_preview:
                                st.image(img_preview, width=55)
                            if st.button(
                                sug, key=f"sg_{i}",
                                use_container_width=True
                            ):
                                parts     = [p.strip()
                                             for p in raw.split(',')]
                                parts[-1] = sug
                                raw       = ', '.join(parts)

            st.markdown(
                "<p style='color:#222;font-size:0.7rem;"
                "letter-spacing:0.1em;margin:0.7rem 0 0.3rem;'>"
                "POPULAR</p>",
                unsafe_allow_html=True
            )
            chips     = ["Aventus","Baccarat Rouge 540",
                         "Black Opium","Dior Sauvage","Good Girl"]
            chip_cols = st.columns(5)
            for i, chip in enumerate(chips):
                with chip_cols[i]:
                    if st.button(chip, key=f"chip_{i}",
                                 use_container_width=True):
                        raw = chip

            if raw and raw.strip():
                try:
                    from deep_translator import GoogleTranslator
                    from langdetect import detect
                    lang = detect(raw)
                    if lang != 'en':
                        raw = GoogleTranslator(
                            source=lang, target='en'
                        ).translate(raw)
                        st.caption(f"Translated from {lang}")
                except Exception:
                    pass
                perfume_names = [
                    p.strip() for p in raw.split(',') if p.strip()
                ]

        with tab2:
            st.markdown(
                "<p style='color:#444;font-size:0.82rem;"
                "margin:0 0 0.5rem;'>Describe what you want to smell "
                "like — ingredients, mood, or feeling.</p>",
                unsafe_allow_html=True
            )
            raw_notes = st.text_input(
                "nt",
                placeholder="e.g. warm vanilla oud, or fresh citrus office",
                label_visibility="collapsed", key="nt_input"
            )
            vibes = [
                ("🔥 Warm & Sensual","vanilla oud amber warm spicy musk"),
                ("🌊 Fresh & Clean", "bergamot citrus aquatic green fresh"),
                ("🌹 Floral & Soft", "rose jasmine peony white floral"),
                ("🌲 Woody & Bold",  "cedar sandalwood vetiver leather")
            ]
            vc = st.columns(4)
            for i, (label, query) in enumerate(vibes):
                with vc[i]:
                    if st.button(label, key=f"vb_{i}",
                                 use_container_width=True):
                        raw_notes = query
            if raw_notes and raw_notes.strip():
                try:
                    from deep_translator import GoogleTranslator
                    from langdetect import detect
                    lang = detect(raw_notes)
                    if lang != 'en':
                        raw_notes = GoogleTranslator(
                            source=lang, target='en'
                        ).translate(raw_notes)
                        st.caption(f"Translated from {lang}")
                except Exception:
                    pass
                notes_input = raw_notes.strip()

        with tab3:
            st.markdown(
                "<p style='color:#444;font-size:0.82rem;"
                "margin:0 0 0.5rem;'>Enter a luxury perfume and we "
                "find budget-friendly alternatives that smell "
                "80-99% similar.</p>",
                unsafe_allow_html=True
            )
            raw_dupe = st.text_input(
                "dp",
                placeholder="e.g. Baccarat Rouge 540, Creed Aventus",
                label_visibility="collapsed", key="dp_input"
            )
            if raw_dupe and raw_dupe.strip():
                perfume_names = [raw_dupe.strip()]
                dupes_only    = True

        st.markdown("<div style='height:0.5rem'></div>",
                    unsafe_allow_html=True)

        go = st.button(
            "Find my scent →", type="primary",
            use_container_width=True, key="go_btn"
        )

        if go and (perfume_names or notes_input):
            with st.spinner(
                "We did the sniffing so you don't have to..."
            ):
                results, found = recommend(
                    perfume_names=perfume_names,
                    notes_input=notes_input,
                    n=n_res,
                    dupes_only=dupes_only
                )

            if results is None or len(results) == 0:
                st.markdown("""
                <div style="text-align:center;padding:3rem 1rem;
                            color:#333;">
                  <div style="font-size:2.5rem;margin-bottom:0.5rem;">🤷</div>
                  <p style="color:#555;font-size:0.95rem;">
                    No strong matches found.</p>
                  <p style="font-size:0.82rem;color:#333;">
                    Try a different perfume name, check your spelling,
                    or describe the notes instead.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                if found:
                    st.markdown(
                        f"<p style='color:#333;font-size:0.76rem;"
                        f"margin:0.5rem 0 0.3rem;'>Based on: "
                        f"{', '.join(found)}</p>",
                        unsafe_allow_html=True
                    )

                weather_now = get_weather_now()
                we          = SEASON_EMOJI.get(weather_now,'')
                label       = ("Affordable alternatives"
                               if dupes_only
                               else "Recommended for you")

                st.markdown(
                    f"<div style='display:flex;"
                    f"justify-content:space-between;"
                    f"align-items:center;"
                    f"margin:0.8rem 0 1rem;"
                    f"padding-bottom:0.5rem;"
                    f"border-bottom:0.5px solid #1e1e1e;'>"
                    f"<span style='font-family:Playfair Display SC,serif;"
                    f"color:#C9A84C;font-size:0.76rem;"
                    f"letter-spacing:0.2em;text-transform:uppercase;'>"
                    f"{label}</span>"
                    f"<span style='color:#2a2a2a;font-size:0.72rem;'>"
                    f"{we} {weather_now} &nbsp;·&nbsp; "
                    f"{len(results)} results</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )

                cols = st.columns(3)
                for idx, (_, row) in enumerate(results.iterrows()):
                    with cols[idx % 3]:
                        st.markdown(render_card_html(row, loc), unsafe_allow_html=True)

                st.markdown(
                    "<div style='text-align:center;padding:1rem 0;"
                    "color:#222;font-size:0.76rem;margin-top:0.5rem;'>"
                    "Budget shown on each card. "
                    "Scroll to scan all options."
                    "</div>",
                    unsafe_allow_html=True
                )

        elif go:
            st.warning(
                "Please enter a perfume name or describe your scent."
            )

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