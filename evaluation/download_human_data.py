# DEPENDENCIES
import os
import re
from tqdm import tqdm
from pathlib import Path
from langdetect import detect
from datasets import load_dataset


DATA_DIR = Path("evaluation/human")

DOMAINS  = ["legal",
            "ai_ml", 
            "science", 
            "general", 
            "tutorial",
            "business",
            "academic", 
            "creative",
            "medical", 
            "marketing",
            "journalism", 
            "engineering", 
            "social_media",
            "software_dev",
            "technical_doc", 
            "blog_personal", 
           ]


def save_text(domain: str, text_id: str, text: str, min_words=50, min_chars=100):
    words = text.split()

    if ((len(words) < min_words) or (len(words) > 10000)):
        return

    try:
        if (detect(text) != 'en'):
            return

    except:
        return

    clean = re.sub(r'\s+', ' ', text).strip()

    if (len(clean) < min_chars):
        return

    (DATA_DIR / domain).mkdir(parents = True, exist_ok = True)

    with open(DATA_DIR / domain / f"{text_id}.txt", "w", encoding = "utf-8") as f:
        f.write(clean)


def fetch_general():
    ds = load_dataset("wikipedia", "20220301.en", split="train")

    for i, ex in enumerate(ds.shuffle(seed = 42).select(range(50))):
        save_text("general", f"wiki_{i}", ex["text"][:2000])



def fetch_academic():
    ds    = load_dataset("scientific_papers", "arxiv", split = "validation", streaming = True)
    count = 0

    for i, ex in enumerate(ds):
        if (count >= 50): 
            break

        abstract = ex.get("abstract", "").strip()

        if (abstract and (80 <= len(abstract.split()) <= 600)):
            save_text("academic", f"arxiv_{i}", abstract)

            count += 1


def fetch_creative():
    try:
        ds      = load_dataset("sedthh/gutenberg_english", split = "train")
        samples = ds.shuffle(seed = 42).select(range(min(100, len(ds))))

        for i, ex in enumerate(samples):
            save_text("creative", f"gutenberg_{i}", ex["TEXT"])

    except Exception as e:
        print(f"⚠️ Gutenberg fallback: {e}")
        _fetch_from_c4("creative", ["story", "once upon", "narrative"])


def fetch_ai_ml():
    # Use arXiv subset with ML keywords
    ds          = load_dataset("scientific_papers", "arxiv", split = "validation", streaming = True)
    count       = 0
    ml_keywords = ["machine learning", "neural network", "transformer", "llm", "deep learning", "generative AI", "GenAI", "ML", "Artifical Intelligence", "AI"]

    for i, ex in enumerate(ds):
        if (count >= 50): 
            break

        text = ex.get("abstract", "") + " " + ex.get("article", "")

        if any(kw in text.lower() for kw in (ml_keywords) and (500 <= len(text.split()) <= 2000)):
            save_text("ai_ml", f"arxiv_ml_{i}", text)
            count += 1


def fetch_software_dev():
    try:
        ds    = load_dataset("codeparrot/github-readme", split = "train", streaming = True)
        count = 0

        for i, ex in enumerate(ds):
            if (count >= 50): 
                break

            readme = ex.get("readme", "")

            if (("##" in readme) and (100 <= len(readme.split()) <= 800)):
                save_text("software_dev", f"github_readme_{i}", readme)
                count += 1

    except Exception as e:
        print(f"⚠️ GitHub README fallback: {e}")
        _fetch_from_c4("software_dev", ["code", "def ", "import ", "function", "api"])


def fetch_technical_doc():
    _fetch_from_c4("technical_doc", ["documentation", "user guide", "api reference", "manual"])


def fetch_engineering():
    ds    = load_dataset("scientific_papers", "arxiv", split = "validation", streaming = True)
    count = 0

    for i, ex in enumerate(ds):
        if (count >= 50): 
            break

        if (("engineering" in ex.get("article", "").lower()) and (500 <= len(ex["article"].split()) <= 2000)):
            save_text("engineering", f"eng_{i}", ex["article"])
            count += 1


def fetch_science():
    try:
        ds      = load_dataset("allenai/scitext", "abstracts", split = "train")
        samples = ds.shuffle(seed=42).select(range(50))

        for i, ex in enumerate(samples):
            save_text("science", f"scitext_{i}", ex["text"])

    except Exception as e:
        print(f"⚠️ SciText fallback: {e}")
        _fetch_from_c4("science", ["experiment", "hypothesis", "scientific", "study"])


def fetch_business():
    _fetch_from_c4("business", ["company", "revenue", "quarterly", "market analysis", "profit"])


def fetch_legal():
    try:
        ds      = load_dataset("lex_glue", "case_hold", split = "train")
        samples = ds.shuffle(seed = 42).select(range(50))

        for i, ex in enumerate(samples):
            save_text("legal", f"lexglue_{i}", ex["context"])

    except Exception as e:
        print(f"⚠️ LexGLUE fallback: {e}")
        _fetch_from_c4("legal", ["court", "agreement", "contract", "jurisdiction", "plaintiff"])


def fetch_medical():
    ds    = load_dataset("scientific_papers", "pubmed", split="validation", streaming=True)
    count = 0

    for i, ex in enumerate(ds):
        if (count >= 50): 
            break

        abstract = ex.get("abstract", "")

        if (abstract and (100 <= len(abstract.split()) <= 2000)):
            save_text("medical", f"pubmed_{i}", abstract)

            count += 1


def fetch_journalism():
    try:
        ds      = load_dataset("cnn_dailymail", "3.0.0", split = "validation")
        samples = ds.shuffle(seed=42).select(range(50))

        for i, ex in enumerate(samples):
            save_text("journalism", f"cnn_{i}", ex["article"])

    except Exception as e:
        print(f"⚠️ CNN/DailyMail fallback: {e}")
        _fetch_from_c4("journalism", ["report", "news", "according to", "announced"])


def fetch_marketing():
    _fetch_from_c4("marketing", ["best solution", "premium", "offer", "buy now", "exclusive"])


def fetch_social_media():
    try:
        ds      = load_dataset("tweet_eval", "sentiment", split = "train")
        samples = ds.shuffle(seed = 42).select(range(100))  # get more to compensate
        count   = 0

        for i, ex in enumerate(samples):
            if (count >= 50):
                break

            # Allow short texts for social media
            save_text("social_media", f"tweet_{i}", ex["text"], min_words = 10, min_chars = 30)
            count += 1

    except Exception as e:
        print(f"⚠️ TweetEval fallback: {e}")
        _fetch_from_c4("social_media", ["lol", "im ", "i think", "check this out"])


def fetch_blog_personal():
    _fetch_from_c4("blog_personal", ["i believe", "my experience", "in my opinion", "personally"])


def fetch_tutorial():
    _fetch_from_c4("tutorial", ["step by step", "how to", "guide for", "beginner's guide", "tutorial"])


def _fetch_from_c4(domain: str, keywords: list):
    ds    = load_dataset("c4", "en", split = "validation", streaming = True)
    count = 0

    for i, ex in enumerate(ds):
        if (count >= 50): 
            break

        text = ex["text"]

        if any(kw in text.lower() for kw in keywords) and 100 <= len(text.split()) <= 800:
            save_text(domain, f"c4_{domain}_{i}", text)
            count += 1


# Execution
if __name__ == "__main__":
    print("📥 Downloading human texts for all 16 domains...")
    
    for domain in DOMAINS:
        print(f" → Fetching {domain}...")
        locals()[f"fetch_{domain}"]()

    print("✅ Human data collection complete.")