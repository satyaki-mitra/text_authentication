# DEPENDENCIES
import os
import re
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from langdetect import detect
from datasets import load_dataset


# Configuration
DATA_DIR                   = Path("evaluation/human")
TARGET_SAMPLES_PER_DOMAIN  = 50
MAX_STREAMING_ITERATIONS   = 500  
MAX_C4_ITERATIONS          = 1000  

DOMAINS                    = ["legal",
                              "ai_ml", 
                              "medical", 
                              "science", 
                              "general", 
                              "tutorial",
                              "business",
                              "academic", 
                              "creative",
                              "marketing",
                              "journalism", 
                              "engineering", 
                              "social_media",
                              "software_dev",
                              "technical_doc",
                              "blog_personal",
                             ]


def save_text(domain: str, text_id: str, text: str, min_words: int = 50, min_chars: int = 100):
    """
    Save text sample with validation
    
    Arguments:
    ----------
        domain     { str } : Domain name
        
        text_id    { str } : Unique identifier
        
        text       { str } : Text content
        
        min_words  { int } : Minimum word count
        
        min_chars  { int } : Minimum character count
    """
    words = text.split()

    if ((len(words) < min_words) or (len(words) > 10000)):
        return False

    try:
        if (detect(text) != 'en'):
            return False
    except:
        return False

    clean = re.sub(r'\s+', ' ', text).strip()

    if (len(clean) < min_chars):
        return False

    (DATA_DIR / domain).mkdir(parents = True, exist_ok = True)

    with open(DATA_DIR / domain / f"{text_id}.txt", "w", encoding = "utf-8") as f:
        f.write(clean)
    
    return True


def fetch_general():
    """
    Fetch general knowledge texts from Wikipedia
    """
    logger.info(f"→ Fetching general (target: {TARGET_SAMPLES_PER_DOMAIN})...")
    
    ds    = load_dataset("wikipedia", "20220301.en", split = "train")
    count = 0

    for i in tqdm(range(min(1000, len(ds))), desc = "  general"):
        if (count >= TARGET_SAMPLES_PER_DOMAIN):
            break
        
        ex = ds.shuffle(seed = 42).select([i])[0]
        
        if save_text("general", f"wiki_{count}", ex["text"]):
            count += 1
    
    logger.info(f"  Collected {count}/{TARGET_SAMPLES_PER_DOMAIN} samples")


def fetch_academic():
    """
    Fetch academic papers from arXiv
    """
    logger.info(f"→ Fetching academic (target: {TARGET_SAMPLES_PER_DOMAIN})...")
    
    ds         = load_dataset("scientific_papers", "arxiv", split = "validation", streaming = True)
    count      = 0
    iterations = 0
    pbar       = tqdm(total = TARGET_SAMPLES_PER_DOMAIN, desc = "  academic")

    for i, ex in enumerate(ds):
        if (count >= TARGET_SAMPLES_PER_DOMAIN) or (iterations >= MAX_STREAMING_ITERATIONS):
            break

        iterations += 1
        abstract    = ex.get("abstract", "").strip()

        if (abstract and (80 <= len(abstract.split()) <= 600)):
            if save_text("academic", f"arxiv_{count}", abstract):
                count += 1
                pbar.update(1)
    
    pbar.close()
    logger.info(f"  Collected {count}/{TARGET_SAMPLES_PER_DOMAIN} samples")


def fetch_creative():
    """
    Fetch creative writing from Project Gutenberg
    """
    logger.info(f"→ Fetching creative (target: {TARGET_SAMPLES_PER_DOMAIN})...")
    
    try:
        ds      = load_dataset("sedthh/gutenberg_english", split = "train")
        samples = ds.shuffle(seed = 42).select(range(min(500, len(ds))))
        count   = 0

        for i, ex in enumerate(tqdm(samples, desc = "  creative")):
            if (count >= TARGET_SAMPLES_PER_DOMAIN):
                break
            
            if save_text("creative", f"gutenberg_{count}", ex["TEXT"]):
                count += 1
        
        if (count < TARGET_SAMPLES_PER_DOMAIN):
            logger.warning(f"  Gutenberg gave {count}, supplementing with C4...")
            _fetch_from_c4("creative", ["story", "narrative", "fiction", "character"], 
                          target = TARGET_SAMPLES_PER_DOMAIN - count, 
                          offset = count)
        
        logger.info(f"  Collected {count}/{TARGET_SAMPLES_PER_DOMAIN} samples")

    except Exception as e:
        logger.warning(f"  Gutenberg failed: {e}, using C4 fallback")
        _fetch_from_c4("creative", ["story", "narrative", "fiction"])


def fetch_ai_ml():
    """
    Fetch AI/ML texts from arXiv
    """
    logger.info(f"→ Fetching ai_ml (target: {TARGET_SAMPLES_PER_DOMAIN})...")
    
    ds          = load_dataset("scientific_papers", "arxiv", split = "validation", streaming = True)
    count       = 0
    iterations  = 0
    
    # More comprehensive keywords
    ml_keywords = ["machine learning", "neural network", "deep learning", "artificial intelligence", "transformer", "neural", "learning", "model training", "dataset", "classification", "generative AI", "GenAI", "LLM", "Natural Language Processing", "NLP"]
    
    pbar        = tqdm(total = TARGET_SAMPLES_PER_DOMAIN, desc = "  ai_ml")

    for i, ex in enumerate(ds):
        if (count >= TARGET_SAMPLES_PER_DOMAIN) or (iterations >= MAX_STREAMING_ITERATIONS):
            break

        iterations += 1
        text        = (ex.get("abstract", "") + " " + ex.get("article", "")[:2000]).lower()

        # More lenient matching - just one keyword
        if any(kw in text for kw in ml_keywords):
            full_text = ex.get("abstract", "") or ex.get("article", "")[:1000]
            
            if (100 <= len(full_text.split()) <= 2000):
                if save_text("ai_ml", f"arxiv_ml_{count}", full_text):
                    count += 1
                    pbar.update(1)
    
    pbar.close()
    
    # If still not enough, supplement with C4
    if (count < TARGET_SAMPLES_PER_DOMAIN):
        logger.warning(f"  arXiv gave {count}, supplementing with C4...")
        _fetch_from_c4("ai_ml", ["machine learning", "neural network", "AI", "deep learning"], 
                      target = TARGET_SAMPLES_PER_DOMAIN - count, 
                      offset = count)
    
    logger.info(f"  Collected {count}/{TARGET_SAMPLES_PER_DOMAIN} samples")


def fetch_software_dev():
    """
    Fetch software development texts
    """
    logger.info(f"→ Fetching software_dev (target: {TARGET_SAMPLES_PER_DOMAIN})...")
    
    # Go straight to C4 with good keywords
    _fetch_from_c4("software_dev", ["function", "API", "code", "programming", "developer", "software"])


def fetch_technical_doc():
    """
    Fetch technical documentation
    """
    logger.info(f"→ Fetching technical_doc (target: {TARGET_SAMPLES_PER_DOMAIN})...")
    
    # More lenient keywords
    _fetch_from_c4("technical_doc", ["documentation", "manual", "guide", "instructions", "tutorial", "how to use"])


def fetch_engineering():
    """
    Fetch engineering texts
    """
    logger.info(f"→ Fetching engineering (target: {TARGET_SAMPLES_PER_DOMAIN})...")
    
    ds         = load_dataset("scientific_papers", "arxiv", split = "validation", streaming = True)
    count      = 0
    iterations = 0
    
    # Better keywords
    eng_keywords = ["engineering", "design", "system", "implementation", "performance", "optimization"]
    
    pbar       = tqdm(total = TARGET_SAMPLES_PER_DOMAIN, desc = "  engineering")

    for i, ex in enumerate(ds):
        if (count >= TARGET_SAMPLES_PER_DOMAIN) or (iterations >= MAX_STREAMING_ITERATIONS):
            break

        iterations += 1
        text        = (ex.get("abstract", "") + " " + ex.get("article", "")[:2000]).lower()
        
        if any(kw in text for kw in eng_keywords):
            full_text = ex.get("abstract", "") or ex.get("article", "")[:1000]
            
            if (200 <= len(full_text.split()) <= 2000):
                if save_text("engineering", f"eng_{count}", full_text):
                    count += 1
                    pbar.update(1)
    
    pbar.close()
    
    # Supplement with C4 if needed
    if (count < TARGET_SAMPLES_PER_DOMAIN):
        logger.warning(f"  arXiv gave {count}, supplementing with C4...")
        _fetch_from_c4("engineering", ["engineering", "system design", "technical"], 
                      target = TARGET_SAMPLES_PER_DOMAIN - count, 
                      offset = count)
    
    logger.info(f"  Collected {count}/{TARGET_SAMPLES_PER_DOMAIN} samples")


def fetch_science():
    """
    Fetch science texts
    """
    logger.info(f"→ Fetching science (target: {TARGET_SAMPLES_PER_DOMAIN})...")
    
    # Use C4 directly with good keywords
    _fetch_from_c4("science", ["scientific", "research", "study", "experiment", "analysis"])


def fetch_business():
    """
    Fetch business texts
    """
    logger.info(f"→ Fetching business (target: {TARGET_SAMPLES_PER_DOMAIN})...")
    _fetch_from_c4("business", ["business", "company", "market", "financial", "strategy"])


def fetch_legal():
    """
    Fetch legal texts
    """
    logger.info(f"→ Fetching legal (target: {TARGET_SAMPLES_PER_DOMAIN})...")
    
    # Use C4 directly with legal keywords
    _fetch_from_c4("legal", ["legal", "court", "law", "contract", "agreement", "jurisdiction"])


def fetch_medical():
    """
    Fetch medical texts from PubMed
    """
    logger.info(f"→ Fetching medical (target: {TARGET_SAMPLES_PER_DOMAIN})...")
    
    ds         = load_dataset("scientific_papers", "pubmed", split = "validation", streaming = True)
    count      = 0
    iterations = 0
    pbar       = tqdm(total = TARGET_SAMPLES_PER_DOMAIN, desc = "  medical")

    for i, ex in enumerate(ds):
        if (count >= TARGET_SAMPLES_PER_DOMAIN) or (iterations >= MAX_STREAMING_ITERATIONS):
            break

        iterations += 1
        abstract    = ex.get("abstract", "")

        if (abstract and (100 <= len(abstract.split()) <= 2000)):
            if save_text("medical", f"pubmed_{count}", abstract):
                count += 1
                pbar.update(1)
    
    pbar.close()
    logger.info(f"  Collected {count}/{TARGET_SAMPLES_PER_DOMAIN} samples")


def fetch_journalism():
    """
    Fetch journalism texts
    """
    logger.info(f"→ Fetching journalism (target: {TARGET_SAMPLES_PER_DOMAIN})...")
    
    # Use C4 with journalism keywords
    _fetch_from_c4("journalism", ["news", "reported", "according to", "announced", "said"])


def fetch_marketing():
    """
    Fetch marketing texts
    """
    logger.info(f"→ Fetching marketing (target: {TARGET_SAMPLES_PER_DOMAIN})...")
    _fetch_from_c4("marketing", ["product", "service", "customer", "offer", "solution"])


def fetch_social_media():
    """
    Fetch social media texts
    """
    logger.info(f"→ Fetching social_media (target: {TARGET_SAMPLES_PER_DOMAIN})...")
    
    # Use C4 with social media style keywords
    _fetch_from_c4("social_media", ["post", "share", "comment", "like", "follow", "tweet"], min_words = 20)


def fetch_blog_personal():
    """
    Fetch personal blog texts
    """
    logger.info(f"→ Fetching blog_personal (target: {TARGET_SAMPLES_PER_DOMAIN})...")
    
    # More inclusive keywords
    _fetch_from_c4("blog_personal", ["I think", "my", "personally", "in my opinion", "blog", "experience"])


def fetch_tutorial():
    """
    Fetch tutorial texts
    """
    logger.info(f"→ Fetching tutorial (target: {TARGET_SAMPLES_PER_DOMAIN})...")
    _fetch_from_c4("tutorial", ["step", "how to", "guide", "tutorial", "learn"])


def _fetch_from_c4(domain: str, keywords: list, min_words: int = 100, target: int = None, offset: int = 0):
    """
    Fallback fetch from C4 dataset - IMPROVED with better iteration limits
    
    Arguments:
    ----------
        domain    { str }  : Domain name
        
        keywords  { list } : Keywords to filter
        
        min_words { int }  : Minimum word count
        
        target    { int }  : Target samples (default: TARGET_SAMPLES_PER_DOMAIN)
        
        offset    { int }  : Starting count offset
    """
    if target is None:
        target = TARGET_SAMPLES_PER_DOMAIN
    
    ds         = load_dataset("allenai/c4", "en", split = "validation", streaming = True)
    count      = offset
    iterations = 0
    pbar       = tqdm(total = target, initial = offset, desc = f"  {domain}")

    for i, ex in enumerate(ds):
        if (count >= target) or (iterations >= MAX_C4_ITERATIONS):
            break

        iterations += 1
        text        = ex["text"]
        
        # More lenient matching - any keyword (case insensitive)
        if any(kw.lower() in text.lower() for kw in keywords):
            word_count = len(text.split())
            
            if (min_words <= word_count <= 800):
                if save_text(domain, f"c4_{domain}_{count}", text, min_words = min_words):
                    count += 1
                    pbar.update(1)
    
    pbar.close()
    actual_collected = count - offset
    logger.info(f"  Collected {actual_collected}/{target} samples")


# Execution
if __name__ == "__main__":
    print("=" * 70)
    print("TEXT-AUTH: Downloading Human Data")
    print("=" * 70)
    print(f"\nTarget: {TARGET_SAMPLES_PER_DOMAIN} samples per domain")
    print(f"Total domains: {len(DOMAINS)}")
    print(f"Expected total: {TARGET_SAMPLES_PER_DOMAIN * len(DOMAINS)} samples\n")
    
    for domain in DOMAINS:
        try:
            locals()[f"fetch_{domain}"]()
        
        except Exception as e:
            logger.error(f"Failed to fetch {domain}: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("Collection Summary")
    print("=" * 70)
    
    total_collected = 0

    for domain in DOMAINS:
        domain_path = DATA_DIR / domain

        if domain_path.exists():
            count            = len(list(domain_path.glob("*.txt")))
            total_collected += count
            status           = "✓" if count >= 45 else "⚠"

            print(f"  {status} {domain:20s}: {count:3d} samples")
    
    print("=" * 70)
    print(f"Total collected: {total_collected} samples")
    print(f"Target was: {TARGET_SAMPLES_PER_DOMAIN * len(DOMAINS)} samples")
    print("=" * 70)
    
    # Show warnings
    insufficient = [d for d in DOMAINS if len(list((DATA_DIR / d).glob("*.txt"))) < 45]
    
    if insufficient:
        print("\n⚠ Domains with < 45 samples:")
        for d in insufficient:
            print(f"  - {d}")
        print("\nTip: You can rerun the script to collect more samples")