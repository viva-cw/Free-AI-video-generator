import os
import spacy
import time
import math
import warnings
import cv2
import random
from dataclasses import dataclass, asdict 
import openai
import google.generativeai as genai
from google.cloud import texttospeech
from google.api_core.client_options import ClientOptions
import mss
import re
import requests
import PyPDF2
from typing import Iterator
import subprocess
import shlex
from collections import Counter
import pyautogui
import nltk
import pyperclip
from datetime import datetime
from gtts import gTTS
from transformers import pipeline, AutoTokenizer, BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
from sentence_transformers import SentenceTransformer, util
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from IPython.display import Audio, display
nlp = spacy.load("en_core_web_sm")
api_key = "API_key"
# Suppress slow processor warning
warnings.filterwarnings("ignore", message="Using a slow image processor.*")

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
device = torch.device("cpu")
model = model.to(device)
nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize

def text_generator(text, max_sentence_length=500):
    """Yield sentences from input text for streaming API."""
    sentences = sent_tokenize(text)
    buffer = ""
    for s in sentences:
        if len(buffer) + len(s) > max_sentence_length:
            yield buffer.strip()
            buffer = s
        else:
            buffer += " " + s
    if buffer.strip():
        yield buffer.strip()

def synthesize_streaming(text_iterator, voice, client, speaking_rate=1.0):
    """Yields audio_content for each chunk from the streaming API."""
    config_request = texttospeech.StreamingSynthesizeRequest(
    streaming_config=texttospeech.StreamingSynthesizeConfig(
        voice=voice  # Only pass the voice; no audio_config!
        )
    )
    def request_generator() -> Iterator[texttospeech.StreamingSynthesizeRequest]:
        yield config_request
        for text in text_iterator:
            yield texttospeech.StreamingSynthesizeRequest(
                input=texttospeech.StreamingSynthesisInput(text=text)
            )

    streaming_responses: Iterator[texttospeech.StreamingSynthesizeResponse] = (
        client.streaming_synthesize(request_generator())
    )

    for response in streaming_responses:
        yield response.audio_content

def process_streaming_audio(
    text, voice, client,
    out_file="output.wav",
    speaking_rate=1.0,
    display_individual_chunks=False,
    show_audio=False,
    sample_rate=24000
):
    """Stream TTS and merge audio to one file."""
    import numpy as np
    sentences = list(text_generator(text))
    audio_iterator = synthesize_streaming(iter(sentences), voice, client, speaking_rate=speaking_rate)
    final_audio_data = np.array([], dtype=np.int16)
    chunk_files = []
    for idx, audio_content in enumerate(audio_iterator):
        audio_chunk = np.frombuffer(audio_content, dtype=np.int16)
        final_audio_data = np.concatenate((final_audio_data, audio_chunk))
        chunk_fn = f"{os.path.splitext(out_file)[0]}_chunk{idx+1}.wav"
        with open(chunk_fn, "wb") as f:
            f.write(audio_content)
        chunk_files.append(chunk_fn)
        if display_individual_chunks and len(audio_chunk) > 0:
            print(f"Processed chunk # {idx+1}")
            if show_audio:
                display(Audio(audio_chunk, rate=sample_rate))
    # Optionally merge with ffmpeg if >1 chunk
    if len(chunk_files) > 1:
        concat_file = f"{os.path.splitext(out_file)[0]}_concat.txt"
        with open(concat_file, "w") as f:
            for fn in chunk_files:
                f.write(f"file '{fn}'\n")
        os.system(f'ffmpeg -y -f concat -safe 0 -i "{concat_file}" -c copy "{out_file}"')
        for fn in chunk_files:
            os.remove(fn)
        os.remove(concat_file)
    else:
        os.rename(chunk_files[0], out_file)
    print(f"🎉 Streaming Orus/Chirp3 voiceover ready: {out_file}")
    return out_file
def inject_symbols(text: str) -> str:
    """
    Infer symbolic visual elements from context (e.g. religious, emotional, metaphysical, cultural).
    Returns a short comma-separated list of symbolic objects or motifs.
    """
    doc = nlp(text.lower())
    keywords = {
        "death": ["skull", "withered tree", "dark river"],
        "memory": ["faded photograph", "broken clock", "mirror"],
        "ritual": ["candles", "circle of stones", "dagger", "scroll"],
        "sacrifice": ["bloodstained altar", "bound hands", "flame"],
        "rebirth": ["phoenix", "sprouting seed", "sunrise"],
        "war": ["helmet", "broken sword", "banner"],
        "wisdom": ["owl", "open book", "lantern"],
        "faith": ["cross", "beads", "sunlight breaking clouds"],
        "technology": ["circuitry", "neon wires", "floating screen"],
        "dream": ["veil", "floating objects", "eyeless mask"],
    }

    found = set()
    for word in text.split():
        for theme, symbols in keywords.items():
            if theme in word:
                found.update(symbols)

    # Additional: entity label triggers
    if any(ent.label_ == "RELIGION" for ent in doc.ents):
        found.update(["sacred text", "halo", "light beam"])

    if not found:
        return "no symbolic objects"
    return ", ".join(list(found)[:4])
def get_camera(mood: str) -> str:
    """
    Returns camera setup (lens, depth of field, angle, lighting) based on mood.
    """
    mood = mood.lower()
    settings = {
        "serene":      "50mm lens, wide shot, soft lighting, shallow depth of field",
        "joy":         "35mm lens, eye-level, bright lighting, medium DOF",
        "fear":        "24mm lens, low angle, harsh shadows, deep focus",
        "anger":       "85mm lens, close-up, dramatic contrast lighting",
        "neutral":     "35mm lens, eye-level, natural light, medium DOF",
        "sadness":     "50mm lens, high angle, soft contrast, muted tones",
        "surprise":    "28mm lens, dynamic tilt, directional lighting",
        "hopeful":     "35mm lens, golden hour light, soft DOF",
        "tense":       "40mm lens, deep shadows, tight framing, medium DOF",
        "nostalgic":   "35mm lens, faded lighting, film grain effect",
        "clinical":    "80mm lens, cold light, centered framing, flat DOF",
        "curious":     "24mm lens, wide angle, exploratory panning perspective"
    }

    return settings.get(mood, "35mm lens, eye-level, natural lighting, shallow depth of field")

# Captioning function
def caption_image(path: str) -> str:
    image = Image.open(path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=50)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


device = "cuda" if torch.cuda.is_available() else "cpu"
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
# 1) load tokenizer so we know the model’s max_length
_MOOD_MODEL = "j-hartmann/emotion-english-distilroberta-base"
_tokenizer = AutoTokenizer.from_pretrained(_MOOD_MODEL)
tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base", use_fast=True)

emotion_classifier = pipeline(
    "text-classification",
    model=_MOOD_MODEL,
    tokenizer=_tokenizer,
    top_k=1            # replaces return_all_scores=False
)
def ban_not_but(text: str) -> str:
    """
    1) Rewrite “if not X but Y” → “if Y”
    2) Rewrite any “not X but Y” → “Y”
    """
    # 1) if not X but Y → if Y
    text = re.sub(
        r'\bif\s+not\s+([^,\.!?]+?)\s+but\s+([^,\.!?]+)',
        r'if \2',
        text,
        flags=re.IGNORECASE
    )
    # 2) general not X but Y → Y
    text = re.sub(
        r'\bnot\s+([^,\.!?]+?)\s+but\s+([^,\.!?]+)',
        r'\2',
        text,
        flags=re.IGNORECASE
    )
    return text
def extract_focal_point(text: str) -> str:
    """
    Extract the most semantically central noun or concept from the input text.
    Uses a combination of named entity recognition, noun salience, and action alignment.
    """
    doc = nlp(text)

    # Priority 1: High-value named entities (PERSON, ORG, WORK_OF_ART, EVENT, etc.)
    priority_ents = {"PERSON", "ORG", "WORK_OF_ART", "EVENT", "PRODUCT", "NORP", "FAC"}
    entities = [ent.text for ent in doc.ents if ent.label_ in priority_ents]
    if entities:
        return entities[0]

    # Priority 2: Noun chunk scoring based on frequency and token importance
    noun_chunks = list(doc.noun_chunks)
    if noun_chunks:
        scores = {}
        for chunk in noun_chunks:
            score = sum(
                (1.5 if tok.pos_ == "NOUN" else 1.0)
                for tok in chunk
            ) * len(chunk.text.split())
            scores[chunk.text] = scores.get(chunk.text, 0) + score
        best_chunk = max(scores.items(), key=lambda x: x[1])[0]
        return best_chunk.strip()

    # Priority 3: Fallback to most frequent standalone noun
    nouns = [token.text for token in doc if token.pos_ == "NOUN"]
    if nouns:
        return max(set(nouns), key=nouns.count)

    # Priority 4: Try to extract main verb to build a conceptual object
    verbs = [token.lemma_ for token in doc if token.pos_ == "VERB"]
    if verbs:
        return f"act of {verbs[0]}"

    # Final fallback
    return "central subject"
def generate_dynamics(text: str) -> str:
    """
    Analyze the input text and return dynamic elements suitable for image prompts,
    such as movement, conflict, tension, transformation, or energy.
    """
    doc = nlp(text)

    # Extract action verbs
    verbs = [token.lemma_ for token in doc if token.pos_ == "VERB" and not token.is_stop]
    if verbs:
        top_verb = verbs[0]
        return f"{top_verb} in motion"  # e.g. "marching in motion"

    # Check for contrast/tension markers
    contrast_words = {"but", "however", "yet", "although", "despite", "versus"}
    if any(token.text.lower() in contrast_words for token in doc):
        return "conflict or tension between opposing forces"

    # Check for emotional transitions
    emotions = {"anger", "fear", "joy", "sorrow", "surprise", "despair", "hope"}
    emo_found = [tok.text for tok in doc if tok.lemma_.lower() in emotions]
    if emo_found:
        return f"emotional shift: {emo_found[0]}"

    # Check for setting changes
    transitions = {"enter", "leave", "arrive", "emerge", "descend", "rise", "transform"}
    for tok in doc:
        if tok.lemma_ in transitions:
            return f"transitional movement: {tok.lemma_}"

    # Default fallback
    return "calm but expectant atmosphere"

def detect_mood(text: str) -> str:
    """
    1) First check for high-priority keywords.
    2) Else truncate to model max length and classify.
    """
    mood_map = {
        "discover": "curious",
        "war":      "tense",
        "peace":    "serene",
        "future":   "hopeful",
        "history":  "nostalgic",
        "science":  "clinical"
    }
    lower = text.lower()
    for kw, mood in mood_map.items():
        if kw in lower:
            return mood

    # If no keyword, run the classifier on a truncated slice
    try:
        # tokenizer.model_max_length is usually 512
        result = emotion_classifier(
            text,
            truncation=True,
            max_length=_tokenizer.model_max_length
        )[0]
        return result["label"].lower()
    except Exception:
        # on any failure, default to neutral
        return "neutral"
try:
    _NLP = spacy.load("en_core_web_sm")
except Exception:
    _NLP = None
# Configurable defaults
IMAGE_DURATION = 30
FRAME_DURATION = IMAGE_DURATION               # default seconds per image frame
PROMPT_WAIT_TIME = 150            # seconds to wait for each image
KENBURNS_FPS     = 30
INTRO_OUTRO_DURATION  = 4 
FADE_DURATION        = 1
DEEPSEEK_API_KEY = "YOUR API KEY"
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
INTRO_FONTFILE      = "C:/Windows/Fonts/Arial.ttf"  # adjust if needed
CROSSFADE_DURATION  = 0.5
ner_pipeline = pipeline(
    "ner",
    model="dbmdz/bert-large-cased-finetuned-conll03-english",
    revision="4c53496",
    aggregation_strategy="SIMPLE"
)
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=1
)

def deepseek_generate(
    prompt: str,
    knowledge_base: str = "",
    temperature: float = 0.7,
    max_tokens: int = 32000  # Now supports up to 64K tokens (deepseek-reasoner)
) -> str:
    """
    Send a prompt (plus optional knowledge base) to DeepSeek.
    Uses `deepseek-reasoner` for long outputs (up to 64K tokens).
    """
    if knowledge_base:
        prompt = (
            "Use only the information below to answer the prompt. Do not invent details.\n\n"
            f"SOURCE MATERIAL:\n\"\"\"\n{knowledge_base}\n\"\"\"\n\n"
            f"{prompt}"
        )

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "deepseek-reasoner",  # Changed from "deepseek-chat" to "deepseek-reasoner"
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens  # Now allows up to 64K tokens
    }
    
    try:
        resp = requests.post(DEEPSEEK_URL, json=payload, headers=headers)
        resp.raise_for_status()  # Raises an error for HTTP issues
        return resp.json()["choices"][0]["message"]["content"].strip()
    except requests.exceptions.RequestException as e:
        print(f"API Request Failed: {e}")
        return ""
    except KeyError as e:
        print(f"Failed to parse API response: {e}")
        return ""
import language_tool_python

# 2) Readability
import textstat

# 3) Embedding coherence
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.corpus import wordnet as wn
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

def extract_texture(text: str, domain: str = "default") -> str:
    """
    Ultra-advanced texture extractor.
    Uses adjective/noun patterning, synonym nets, domain cues, and fallback generation.
    """
    doc = nlp(text)

    # Domain-specific texture themes
    domain_textures = {
        "cyberpunk": ["neon-lit", "chrome", "wet pavement", "holographic", "grimy", "flickering"],
        "biblical": ["worn parchment", "sun-scorched", "stone tablets", "dusty robes", "weathered scrolls"],
        "sci-fi": ["sterile metal", "glossy alloy", "plasma sheen", "carbon fiber", "polished void"],
        "ancient": ["eroded stone", "sun-baked clay", "crumbling pillars", "aged bronze", "faded paint"]
    }

    texture_adjs = {
        "gritty", "smooth", "rough", "slick", "wet", "dusty", "foggy", "cracked", 
        "misty", "porous", "frosted", "coarse", "polished", "glossy", "shiny", 
        "smudged", "wrinkled", "grainy", "fuzzy", "velvety", "sticky"
    }

    material_nouns = {
        "stone", "glass", "metal", "ash", "mud", "bark", "smoke", "cloth", 
        "iron", "leather", "fur", "rust", "paper", "wood", "ice", "bone", "sand"
    }

    # WordNet synonym expansion for texture adjectives
    def expand_texture_synonyms(word):
        synonyms = set()
        for syn in wn.synsets(word, pos=wn.ADJ):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name().replace('_', ' '))
        return synonyms

    expanded_adjs = set(texture_adjs)
    for adj in texture_adjs:
        expanded_adjs.update(expand_texture_synonyms(adj))

    adjectives = {token.text.lower() for token in doc if token.pos_ == "ADJ"}
    nouns = {token.text.lower() for token in doc if token.pos_ == "NOUN"}

    adj_matches = sorted(adjectives & expanded_adjs)
    noun_matches = sorted(nouns & material_nouns)

    # Detect joined noun phrases
    phrases = []
    for chunk in doc.noun_chunks:
        words = {token.text.lower() for token in chunk}
        if words & expanded_adjs and words & material_nouns:
            phrases.append(chunk.text)

    # Domain-driven enrichment
    domain_terms = domain_textures.get(domain.lower(), [])

    # Symbolic fallback generation
    symbolic = []
    if "fog" in text or "mist" in text:
        symbolic.append("hazy presence")
    if "ritual" in text or "altar" in text:
        symbolic.append("etched sacred surfaces")
    if "dream" in text or "memory" in text:
        symbolic.append("blurred or decaying textures")

    # Assemble
    result = phrases + adj_matches + noun_matches + domain_terms + symbolic
    if not result:
        return "abstract symbolic textures"
    
    return ", ".join(dict.fromkeys(result))  # Deduplicated, ordered

# ── instantiate tools ───────────────────────────────────────────────────────
_LT       = language_tool_python.LanguageTool('en-US')
_MODEL    = SentenceTransformer('all-MiniLM-L6-v2')

BANNED_CLICHES = [
    "from time immemorial", "needless to say", "throughout history",
    "in today's society", "ever-changing world", "since the dawn of time"
]
_CLICHE_RE = re.compile("|".join(re.escape(x) for x in BANNED_CLICHES), flags=re.I)

def grammar_polish(text: str) -> str:
    matches = _LT.check(text)
    return language_tool_python.correct(text, matches)

def adjust_readability(text: str, target_grade: float = 10.0) -> str:
    grade = textstat.flesch_kincaid_grade(text)
    if abs(grade - target_grade) > 0.5:
        prompt = f"Rewrite to reach a Flesch-Kincaid grade of ~{target_grade:.1f}:\n\n{text}"
        return deepseek_generate(prompt, temperature=0.3)
    return text

def self_critique_and_fix(text: str) -> str:
    critique = deepseek_generate(
        "As an editor, list up to 3 weaknesses in this text:\n\n" + text,
        temperature=1.0, max_tokens=1000
    )
    fix = deepseek_generate(
        "Revise to address these weaknesses:\n\n" + critique + "\n\n" + text,
        temperature=1.0
    )
    return fix

def coherence_score(bullet: str, section: str) -> float:
    b_emb = _MODEL.encode(bullet)
    s_emb = _MODEL.encode(section)
    return util.cos_sim(b_emb, s_emb).item()
def load_knowledge_sources(file_paths: list[str], max_chars: int = 30000) -> str:
    combined = ""
    for path in file_paths:
        if not os.path.isfile(path) or not path.endswith(".txt"):
            continue
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
            combined += f"\n--- FILE: {os.path.basename(path)} ---\n{text.strip()}"
            if len(combined) > max_chars:
                break
    return combined[:max_chars].strip()
def identify_action(text: str) -> str:
    """
    Return a short verb phrase that describes the main action
    in *text*. Falls back to a simple regex if spaCy isn't available.
    """
    # ── Preferred: spaCy dependency parse ──
    if _NLP:
        doc = _NLP(text)
        # 1) use sentence root verb if it’s a real verb
        root = [t for t in doc if t.dep_ == "ROOT" and t.pos_ == "VERB"]
        if root:
            verb = root[0]
            # capture its particle/aux/object for a compact phrase
            children = [c.text for c in verb.children
                        if c.dep_ in {"prt", "aux", "dobj", "prep", "pobj", "advmod"}]
            return " ".join([verb.lemma_] + children)  # e.g. "study the heavens"
        # 2) fallback: first verb in doc
        for t in doc:
            if t.pos_ == "VERB":
                return t.lemma_
    
    # ── Fallback: regex – first present-participle or base form ending in -s/-es ──
    m = re.search(r"\b(\w+?ing|\w+(?:ed|s))\b", text, flags=re.I)
    return m.group(1) if m else "describe"

def load_style_sample(path: str, max_chars: int = 20000) -> str:
    text = ""
    if path.lower().endswith(".pdf"):
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
    else:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

    # If the file is shorter than max_chars, just return the whole thing
    total_len = len(text)
    if total_len <= max_chars:
        return text.strip()

    # Otherwise, pick a random starting index so that we can grab max_chars characters
    start = random.randint(0, total_len - max_chars)
    return text[start : start + max_chars].strip()

BANNED_CLICHES = [
    "from time immemorial", "needless to say", "throughout history",
    "in today's society", "ever-changing world", "since the dawn of time"
]
_CLICHE_RE = re.compile("|".join(re.escape(x) for x in BANNED_CLICHES), flags=re.I)

@dataclass
class ScriptConfig:
    min_words_per_sec: int = 800          # speech speed floor
    max_words_per_sec: int = 1000          # speech speed ceiling
    outline_min: int      = 10
    outline_max: int      = 20
    bullets_max_tokens: int = 800
    section_temperature: float = 1.2
    outline_temperature: float = 1.1
    allow_rhet_qs: bool   = False
    tag_uncertain: bool   = True
    banned_clichés: tuple = (
        "from time immemorial", "needless to say",
        "throughout history",   "ever-changing world",
        "in today's society",   "since the dawn of time","this is not", "this is not only", "this is not just","this is not merely", "this is not simply",
        "will not just", "will not merely", "will not simply","will not only", "will not just be", "will not merely be", "will not simply be","will not only be", "—"
    )

# ── 2. Generate script with config injection ───────────────────────────────
def generate_script(topic: str,
                    tone: str,
                    sample: str | None,
                    min_w: int,
                    max_w: int,
                    vid_len: int,
                    cfg: ScriptConfig = ScriptConfig()) -> str:

    # ---------- derive numbers ----------
    wps = (cfg.min_words_per_sec + cfg.max_words_per_sec) / 2 / 60
    est_total = math.ceil(vid_len * wps)
    avg_section = (min_w + max_w) / 2
    num_sections = max(cfg.outline_min,
                       min(cfg.outline_max,
                           math.ceil(est_total / avg_section)))

    # ---------- build outline ----------
    outline_prompt = f"""
You are a meticulous documentary writer. Draft a numbered outline
({cfg.outline_min}–{cfg.outline_max} bullets) for a {tone.lower()} film titled:
“{topic}”

• chronological or causal order
• each bullet ≤ 12 words, concrete
• no clichés: {', '.join(cfg.banned_clichés)}
• {'avoid rhetorical questions' if not cfg.allow_rhet_qs else 'rhetorical Qs permitted'}
Return ONLY the outline."""
    if sample:
        outline_prompt = (
            f"Emulate this style and obey the rules:\n\"\"\"\n{sample}\n\"\"\"\n\n"
            + outline_prompt
        )

    outline = deepseek_generate(
        outline_prompt,
        temperature=cfg.outline_temperature,
        max_tokens=cfg.bullets_max_tokens
    )
    bullets = [b.lstrip("•").strip(" -.") for b in outline.splitlines() if b.strip()]
    attempt = 0
    while len(bullets) < cfg.outline_min and attempt < 3:
        print("⚠️ Malformed outline, retrying...")
        outline = deepseek_generate(
            outline_prompt,
            temperature=cfg.outline_temperature,
            max_tokens=cfg.bullets_max_tokens
        )
        bullets = [b.lstrip("•").strip(" -.") for b in outline.splitlines() if b.strip()]
        attempt += 1
    if len(bullets) < cfg.outline_min:
        print("❌ Outline failed after 3 attempts. Check your prompt or the LLM.")
        print("Raw outline output:", outline)
        return ""

    # ---------- compile regex for cliché detection ----------
    ban_re = re.compile("|".join(re.escape(c) for c in cfg.banned_clichés), re.I)

    # ---------- expand bullets ----------
    script_path = f"script_{datetime.now():%Y%m%d_%H%M%S}.txt"
    full_script = "OUTLINE\n" + "\n".join(bullets) + "\n\n"

    for i, bullet in enumerate(bullets, 1):
        # 1) Build the rule list for this section
        rules = [
            f"STRICTLY BETWEEN {min_w}–{max_w} words",
            "2–4 sentence paragraphs, blank line between",
            "vary sentence length; no two consecutive sentences start alike",
            ("" if cfg.allow_rhet_qs else "avoid rhetorical questions unless answered"),
            f"avoid clichés: {', '.join(cfg.banned_clichés)}",
        ]
        if cfg.tag_uncertain:
            rules.append("tag any uncertain claim with “[uncertain]”")

        # 2) Build the section prompt
        section_prompt = f"""
Expand outline bullet {i}:

{bullet}

RULES
• {chr(10) + '• '.join(rules)}
• Active, declarative voice only
• STRICTLY BETWEEN {min_w}–{max_w} words; 2–4 sentence paragraphs
• Vary vocabulary: no key word >2 repeats
• Sensory detail: 1 per paragraph
• ≤1 original metaphor or simile
• ≤1 rhetorical question, immediately answered
• Every 4th sentence may be a 1–3 word fragment
• Rotate transitions: ‘however’, ‘yet’, ‘nevertheless’, etc.
• No fillers: really, very, actually, basically
• Show, don’t tell: depict actions/emotions
• After 3–4 facts, include a concrete anecdote
• Tag any uncertain claim “[uncertain]”
• Maintain a {tone} tone; do NOT repeat earlier sections
"""
        if sample:
            section_prompt = (
                f"Match this style:\n\"\"\"\n{sample[:2000]}\n\"\"\"\n\n"
                + section_prompt
            )

        # 3) Generate the raw section
        section = deepseek_generate(
            section_prompt,
            temperature=cfg.section_temperature
        ).strip()

        # 4) Clean “if not X but Y” → “if Y” and “not X but Y” → “Y”
        section = ban_not_but(section)

        # 5) Rewrite clichés if any remain
        if ban_re.search(section):
            fix_prompt = (
                f"Replace each cliché with a fresh phrase.\n"
                f"BANNED: {', '.join(cfg.banned_clichés)}\n\n"
                f"TEXT\n\"\"\"{section}\"\"\""
                f"Return only the corrected text, not a list or explanation.\n\n"
            )
            section = deepseek_generate(fix_prompt, temperature=0.2).strip()

        # 6) Append and report
        full_script += f"Section {i}:\n{section}\n\n"
        print(f"✅ Section {i} done ({len(section.split())} words)")
    full_script = full_script.replace("*", "")
    full_script = re.sub(r"[\*\-\•]", "", full_script)
    # ---------- save ----------
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(full_script)
    print("📄 Saved", script_path)

    return full_script

def validate_image_against_prompt(image_path: str, prompt: str, threshold: float = 0.4) -> tuple[bool, str, float]:
    caption = caption_image(image_path)
    score = caption_similarity_score(prompt, caption)
    return score >= threshold, caption, score

def semantic_chunk_script(script_text: str, total_images: int) -> list[str]:
    doc = nlp(script_text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    avg = math.ceil(len(sentences) / total_images)
    chunks = [" ".join(sentences[i:i+avg]) for i in range(0, len(sentences), avg)]
    return chunks[:total_images]

# --- Entity extraction ---

# --- Action extraction ---

# --- Setting identification ---
def identify_setting(text: str) -> str:
    entities = ner_pipeline(text)
    for ent in entities:
        if ent["entity_group"] in {"LOC", "GPE", "FAC"}:
            return f"{ent['word'].strip()} environment"
    keywords = {
        "forest", "jungle", "desert", "city", "temple", "church", "ocean",
        "space", "room", "lab", "mountain", "hall", "castle", "basement", "ruin"
    }
    for word in re.findall(r'\w+', text.lower()):
        if word in keywords:
            return f"{word} environment"
    return "unspecified environment"

def ensure_visual(text):
    if not text or not text.strip():
        return ""
    # Normalize and lower
    val = text.strip().lower()
    # Generic non-visuals, by hand
    nonvisual = {
        "central subject", "central idea", "main subject", "the main subject",
        "topic", "idea", "subject", "concept", "theme", "entity", "phenomenon",
        "question", "problem", "process", "aspect", "quality", "event", "situation", "experience",
        "feeling", "notion", "truth", "nature", "being", "existence", "essence", "change",
        "meaning", "thought", "belief", "attitude", "emotion", "outcome", "tendency",
        "", "none"
    }
    # Ban single-word abstractions or any word that is clearly not visual
    abstract_keywords = {
        "idea", "concept", "process", "quality", "aspect", "theme", "subject", "notion",
        "value", "belief", "emotion", "feeling", "principle", "purpose", "topic",
        "change", "meaning", "state", "existence", "reality", "thought"
    }
    # Direct check
    if val in nonvisual:
        return ""
    if val in abstract_keywords:
        return ""
    # Heuristic: If only 1-2 words and the noun is abstract, drop it
    doc = nlp(val)
    tokens = [t for t in doc if not t.is_stop]
    if len(tokens) <= 2:
        # Check for high abstractness
        for t in tokens:
            if t.lemma_ in abstract_keywords or t.pos_ in {"PRON", "DET", "SCONJ"}:
                return ""
            # Heuristic: too short and not depictable
            if t.pos_ == "NOUN" and t.lemma_ in abstract_keywords:
                return ""
    # Final filter: Must have at least one depictable noun/adjective/propn
    if not any(t.pos_ in {"NOUN", "PROPN", "ADJ"} and t.is_alpha and t.lemma_ not in abstract_keywords for t in doc):
        return ""
    # Remove any trailing 'idea', 'concept', etc. from end (e.g. "freedom concept" -> "freedom")
    for bad in abstract_keywords:
        if val.endswith(" " + bad):
            val = val[: -len(bad) - 1]
    # Remove any leading articles ("the ", "a ", "an ")
    val = re.sub(r"^(the|a|an)\s+", "", val)
    # Return stripped, cleaned text (title-case if needed)
    return val.strip()

# --- Style-based detail injection ---
def style_specific_details(style: str) -> str:
    injects = {
        "dark fantasy": "antlers, veils, firelight, arcane runes",
        "sci-fi": "neon mist, biotech cables, chrome textures",
        "mythological": "aureoles, animals, divine symbols"
    }
    return injects.get(style.lower(), "")
def extract_visual_template(text: str) -> str:
    """
    Analyze the input text and try to extract a visual anchor (scene, character, setting, object).
    Fall back to a metaphorical template if abstract.
    """
    # Use your existing extract/identify functions or simple keyword search
    setting = identify_setting(text)
    subject = extract_focal_point(text)
    objects = extract_key_entities(text)
    # Filter out nonvisuals as before
    subject = ensure_visual(subject)
    objects = ensure_visual(objects)
    
    # Combine elements found (ignore if generic fallback)
    visual_elements = [e for e in [setting, subject, objects] if e and e not in {"central subject", "central idea, main subject"}]
    
    if visual_elements:
        scene = ", ".join(set(visual_elements))
        return f"Depict this as a cinematic scene: {scene}. Use symbolic objects if necessary, but do not show any words or text."
    else:
        # Fallback for fully abstract content
        return (
            "Depict the underlying concept as a metaphorical scene in a grand library, "
            "with dramatic light and symbolic objects, but no words or letters."
        )

def build_image_prompt(context: str, style_description: str = "Dark Fantasy") -> str:
    mood = detect_mood(context)
    setting = identify_setting(context)
    subject = extract_key_entities(context)
    action = identify_action(context)
    focal = extract_focal_point(context)
    dynamics = generate_dynamics(context)
    texture = extract_texture(context)
    symbols = inject_symbols(context)
    camera = get_camera(mood)

    # Optional: style-specific artist reference
    style_modifier = {
        "dark fantasy": "in the style of Zdzisław Beksiński",
        "mythological": "in the style of Bosch",
        "sci-fi": "in the style of H.R. Giger"
    }.get(style_description.lower(), "")

    details = ", ".join(filter(None, [symbols, texture]))
    visual_template = extract_visual_template(context)
    prompt = (
        f"[SUBJECT] {subject}\n"
        f"[ACTION] {action}\n"
        f"[SETTING] {setting}\n"
        f"[MOOD] {mood}, cinematic lighting\n"
        f"[FOCAL POINT] {focal}\n"
        f"[CAMERA] {camera}, 35mm lens, shallow DOF\n"
        f"[STYLE] {style_description}\n"
        f"[TEXTURE] {texture}\n"
        f"[DYNAMICS] {dynamics}\n"
        f"[DETAILS] {details}\n"
        f"[VISUAL TEMPLATE] {visual_template}\n"
        f"[CONTEXT] Use this scene description to inspire the image: {context.strip()}\n"
        f"[RULES] 1. No text in the image. 2. Match the tone and style. 3. Make it vivid, symbolic, and cinematic."
    )
    if api_key:
        prompt = review_image_prompt_with_openrouter(prompt, api_key)
    return prompt
def review_image_prompt_with_openrouter(prompt: str, api_key: str) -> str:
    # You can pick the latest Qwen model from https://openrouter.ai/models
    model = "qwen/qwen3-30b-a3b:free"  # Or whichever Qwen model you prefer

    system_prompt = (
    "You are an elite expert in prompt engineering for cutting-edge AI image generation. "
    "Your task is to take the following prompt and transform it into a visually spectacular, cinematic masterpiece—"
    "epic, captivating, and awe-inspiring in composition, lighting, and detail. "
    "Rewrite any prompt that is abstract, dry, or could result in text-only output, so that it depicts a grand, concrete scene—"
    "featuring people, objects, dynamic actions, or remarkable settings—never text, letters, or typographic elements. "
    "Enhance with vivid atmosphere, strong mood, and powerful symbolism. "
    "Push for scale, drama, and visual storytelling worthy of a blockbuster film or a fine art painting. "
    "Add missing visual details as needed to make the image stunning and unforgettable. "
    "Return only the improved image prompt, ready for immediate use with advanced generative AI.")
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    }
    try:
        resp = requests.post(url, headers=headers, json=data, timeout=40)
        resp.raise_for_status()
        result = resp.json()
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"OpenRouter API ERROR: {e}")
        return prompt  # fallback: just return the original
# --- Prompt generator ---
def script_to_prompts(script_text: str, style_description: str, total_images: int) -> list[str]:
    chunks = semantic_chunk_script(script_text, total_images)
    prompts = []
    for i, chunk in enumerate(chunks):
        print(f"→ Processing prompt {i+1}/{len(chunks)}")
        prompt = build_image_prompt(chunk, style_description)
        prompts.append(prompt)
    return prompts



# Helper functions
def truncate_text(text: str, max_chars: int) -> str:
    return text[:max_chars] + "..." if len(text) > max_chars else text

def extract_key_entities(text: str) -> str:
    """
    Extracts 2–4 high-value thematic or visual entities from text.
    Prioritizes named entities and compound nouns. Filters junk.
    """
    doc = nlp(text)
    entities = []
    
    # 1. Collect named entities (ORG, PERSON, EVENT, NORP, etc.)
    for ent in doc.ents:
        if ent.label_ in {"PERSON", "ORG", "EVENT", "LAW", "NORP", "FAC", "WORK_OF_ART"}:
            cleaned = ent.text.strip()
            if cleaned.lower() not in STOP_WORDS and cleaned.isalpha():
                entities.append((cleaned, 3))  # weighted higher

    # 2. Compound nouns (e.g., "power struggle", "constitutional order")
    for chunk in doc.noun_chunks:
        if len(chunk) <= 5:
            text_chunk = chunk.text.strip()
            if all(w.lower() not in STOP_WORDS for w in text_chunk.split()):
                entities.append((text_chunk, 2))

    # 3. High-value nouns (excluding stopwords, function words)
    for token in doc:
        if token.pos_ == "NOUN" and token.text.lower() not in STOP_WORDS and token.is_alpha and len(token.text) > 3:
            entities.append((token.text, 1))

    # 4. Score and sort
    scored = {}
    for term, weight in entities:
        if term not in scored:
            scored[term] = 0
        scored[term] += weight

    if not scored:
        return "central idea, main subject"

    # 5. Return top 3–4 unique entries
    top_entities = sorted(scored.items(), key=lambda x: x[1], reverse=True)
    return ", ".join([ent for ent, _ in top_entities[:4]])

ner_pipeline = pipeline("ner", grouped_entities=True)

def identify_setting(text: str) -> str:
    """Identify location/environment context using NER, fallback to keyword lookup."""
    # 1) Try NER for any location‐type entities
    entities = ner_pipeline(text)
    for ent in entities:
        if ent["entity_group"] in {"LOC", "GPE", "FAC"}:
            return f"{ent['word']} environment"

    # 2) Fallback: check for simple keywords
    setting_keywords = {"forest", "city", "lab", "ocean", "space", "room", "mountain", "desert"}
    for word in text.split():
        if word.lower() in setting_keywords:
            return f"{word} environment"
    return ""

emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=False
)


_sim_model = SentenceTransformer("all-MiniLM-L6-v2")

def caption_similarity_score(prompt: str, caption: str) -> float:
    """
    Return semantic similarity between prompt and BLIP caption.
    """
    emb1 = _sim_model.encode(prompt, convert_to_tensor=True)
    emb2 = _sim_model.encode(caption, convert_to_tensor=True)
    return util.cos_sim(emb1, emb2).item()

def open_browser_to_sora():
    chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    url = "https://sora.chatgpt.com/explore"
    os.system(f'start "" "{chrome_path}" --start-maximized {url}')
    print("🌐 Opening Sora...")
    time.sleep(10)
def record_screen_region(filename="section_0.mp4", duration=10, region=(1033, 592, 1778, 1188), fps=30):
    left, top, width, height = region
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    sct = mss.mss()
    import time
    start = time.time()
    while time.time() - start < duration:
        img = np.array(sct.grab({'left': left, 'top': top, 'width': width, 'height': height}))
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        out.write(frame)
        time.sleep(1 / fps)
    out.release()
    print(f"✅ Saved screen recording to {filename}")

def generate_images_with_mouse(prompts: list[str], wait_sec: int):
    print("⚠️ Make sure browser is in focus and Sora is ready!")
    time.sleep(5)

    for i, p in enumerate(prompts):
        print(f"🖼 Generating image {i+1}/{len(prompts)}...")

        # Focus prompt box (coordinates may need adjustment)
        pyautogui.moveTo(1466, 1890)
        pyautogui.click()
        time.sleep(1)

        # Enter prompt
        pyperclip.copy(p)
        pyautogui.hotkey('ctrl', 'v')
        time.sleep(1.5)
        pyautogui.press('enter')

        print(f"⏳ Waiting {wait_sec} seconds...")
        time.sleep(wait_sec)

        # Download image (coordinates may need adjustment)
        pyautogui.moveTo(3613, 375)
        pyautogui.click()
        time.sleep(3)
        pyautogui.moveTo(3478, 690)
        pyautogui.click()
        time.sleep(2)
        pyautogui.moveTo(2093, 1223)
        pyautogui.click()
        time.sleep(15)

        # Save screenshot
        name = f"section_{i}.mp4"
        record_screen_region(filename=name, duration=10, region=(1033, 592, 1778, 1188))
        print(f"✅ Saved {name}")

        # Validate against prompt using BLIP-2
        try:
            valid, caption, score = validate_image_against_prompt(name, p)
            if valid:
                print(f"✅ Verified: caption matches prompt (score = {score:.2f})")
            else:
                print(f"⚠️ Mismatch: caption may not match prompt (score = {score:.2f})")
            print(f"📝 Caption: {caption}")
        except Exception as e:
            print(f"❌ BLIP-2 validation failed: {e}")

        # Clear for next prompt
        pyautogui.moveTo(75, 386)
        pyautogui.click()
        time.sleep(2)



def compile_with_ffmpeg(video_count: int, voice_file: str = "voice.mp3", output: str = "final_video.mp4"):
    # Create a list of section_*.mp4 files
    with open("filelist.txt", "w") as f:
        for i in range(video_count):
            f.write(f"file 'section_{i}.mp4'\n")
    # Concatenate video clips
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", "filelist.txt",
        "-i", voice_file,
        "-c:v", "libx264",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        output
    ], check=True)
    print(f"✅ {output} ready")
    # Cleanup
    os.remove("filelist.txt")
    for i in range(video_count):
        p = f"section_{i}.mp4"
        if os.path.exists(p):
            os.remove(p)

def get_all_image_files():
    """
    Return a sorted list of all 'image_<n>.png' files in the current directory.
    Sorting is numeric on the <n> portion.
    """
    imgs = [f for f in os.listdir() if re.fullmatch(r"image_(\d+)\.png", f)]
    return sorted(imgs, key=lambda fn: int(re.search(r"image_(\d+)\.png", fn).group(1)))

def generate_kenburns_clips(image_files):
    """
    For each image_<n>.png in image_files, produce clip_<n>.mp4 with a slow Ken Burns zoom
    from zoom=1.0 to zoom=1.1 over IMAGE_DURATION seconds at KENBURNS_FPS.
    Each resulting clip_<n>.mp4 will then have a fade‐in during its first FADE_DURATION seconds,
    and a fade‐out during its last FADE_DURATION seconds.
    """
    for fname in image_files:
        idx = re.search(r"image_(\d+)\.png", fname).group(1)
        clip_out = f"clip_{idx}.mp4"

        # 1) Ken Burns zoompan filter
        total_frames = IMAGE_DURATION * KENBURNS_FPS
        zoom_inc = (1.1 - 1.0) / total_frames
        zoompan_filter = (
            f"zoompan="
            f"z='if(lte(zoom,1.1),zoom+{zoom_inc:.6f},1.1)':"
            f"d={total_frames}:"
            f"x='iw/2-(iw/zoom/2)':"
            f"y='ih/2-(ih/zoom/2)'"
        )

        # 2) Fade‐in at start, Fade‐out at end, all in one filtergraph:
        #    - fade=in:  t=“in”, st=0, d=FADE_DURATION
        #    - fade=out: t=“out”, st=(IMAGE_DURATION - FADE_DURATION), d=FADE_DURATION
        fade_filter = (
            f"{zoompan_filter},"
            f"fade=t=in:st=0:d={FADE_DURATION},"
            f"fade=t=out:st={IMAGE_DURATION - FADE_DURATION}:d={FADE_DURATION}"
        )

        # 3) Run FFmpeg to create clip_<n>.mp4
        cmd = [
            "ffmpeg", "-y",
            "-loop", "1",
            "-i", fname,
            "-vf", fade_filter,
            "-c:v", "libx264",
            "-t", str(IMAGE_DURATION),
            "-r", str(KENBURNS_FPS),
            "-pix_fmt", "yuv420p",
            clip_out
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def create_title_card(text: str, output_path: str):
    """
    Create a fade‐in/fade‐out title card on a black background for INTRO_OUTRO_DURATION seconds.
    The title text appears centered in white, fades in over the first second, stays steady,
    then fades out during the last second.
    """
    # Escape any single quotes/backslashes for drawtext:
    safe_text = text.replace("\\", "\\\\").replace("'", "\\'")
    # Build the alpha expression: fade in 0→1 over 1 sec, hold, fade out 1→0 over last 1 sec
    alpha_expr = (
        "if(lt(t,1),"
            "t,"
        "if(lt(t,{D}-1),"
            "1,"
            "1-(t-({D}-1))"
        "))"
    ).replace("{D}", str(INTRO_OUTRO_DURATION))

    drawtext_filter = (
        f"drawtext=fontfile='{INTRO_FONTFILE}':"
        f"text='{safe_text}':"
        "fontsize=64:fontcolor=white:"
        "x=(w-text_w)/2:y=(h-text_h)/2:"
        f"alpha='{alpha_expr}'"
    )

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"color=size=1280x720:duration={INTRO_OUTRO_DURATION}:rate=30:color=black",
        "-vf", drawtext_filter,
        "-c:v", "libx264",
        "-r", "30",
        "-pix_fmt", "yuv420p",
        output_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def get_video_duration(path: str) -> float:
    """
    Return the duration (in seconds) of a video file using ffprobe.
    If anything fails, fall back to IMAGE_DURATION.
    """
    try:
        result = subprocess.run([
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path
        ], capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except Exception:
        return IMAGE_DURATION

def concat_with_black_transitions(clip_list, output_path):
    with open("final_list.txt", "w", encoding="utf-8") as f:
        for clip in clip_list:
            f.write(f"file '{clip}'\n")

    # 2) Run ffmpeg with the concat demuxer
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", "final_list.txt",
        "-c:v", "libx264",
        "-c:a", "aac", "-b:a", "192k",
        output_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # 3) Remove the temporary list file
    os.remove("final_list.txt")

def compile_existing_images():
    """
    1) Gather all image_<n>.png → produce fade‐in/out KenBurns clips (clip_<n>.mp4).
    2) Prompt for optional intro/outro text → produce intro.mp4 / outro.mp4.
    3) Concatenate [intro.mp4, clip_0.mp4, clip_1.mp4, …, outro.mp4] → final_master.mp4.
    4) Delete only the generated clip_*.mp4, intro.mp4, outro.mp4, and image_<n>.png files.
    """
    print("\n🎞 Option 3: Building a video from existing image_<n>.png files…")
    images = get_all_image_files()
    if not images:
        print("❌ No 'image_<n>.png' files found in the folder.")
        return

    print(f"📸 Found {len(images)} images. Generating Ken Burns clips with fade…")
    generate_kenburns_clips(images)

    # Ask for intro/outro text (skip if blank)
    intro_text = input("🎬 Enter intro title (or leave blank to skip): ").strip()
    outro_text = input("🎬 Enter outro title (or leave blank to skip): ").strip()

    clip_sequence = []

    if intro_text:
        print("▶︎ Creating intro.mp4…")
        create_title_card(intro_text, "intro.mp4")
        clip_sequence.append("intro.mp4")

    # Add all generated clip_<n>.mp4, in ascending order
    for img_fn in images:
        idx = re.search(r"image_(\d+)\.png", img_fn).group(1)
        clip_sequence.append(f"clip_{idx}.mp4")

    if outro_text:
        print("▶︎ Creating outro.mp4…")
        create_title_card(outro_text, "outro.mp4")
        clip_sequence.append("outro.mp4")

    # Concatenate everything into final_master.mp4
    print(f"▶︎ Concatenating {len(clip_sequence)} clips → 'final_master.mp4' …")
    concat_with_black_transitions(clip_sequence, "final_master.mp4")

    # ─── Cleanup: remove the generated clips & original images ───
    for fn in os.listdir():
        if re.fullmatch(r"clip_\d+\.mp4", fn) or fn in ("intro.mp4", "outro.mp4"):
            try:
                os.remove(fn)
            except OSError:
                pass
        if re.fullmatch(r"image_\d+\.png", fn):
            try:
                os.remove(fn)
            except OSError:
                pass

    print("✅ Done!  Final video saved as 'final_master.mp4'.")
    print("   (All clip_*.mp4 and image_<n>.png files have been deleted.)\n")

def duck_background_music(voice_path: str,
                          music_path: str,
                          output_path: str,
                          threshold: float = 0.1,
                          ratio: float = 5,
                          attack: float = 5,
                          release: float = 200):
    """
    Mixes music_path under voice_path, ducking music by ratio whenever voice is present.
    """
    # Normalize voice
    subprocess.run([
        "ffmpeg", "-y",
        "-i", voice_path,
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=7",
        "voice_norm.mp3"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Normalize music
    subprocess.run([
        "ffmpeg", "-y",
        "-i", music_path,
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=7",
        "music_norm.mp3"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Duck music under voice
    subprocess.run([
        "ffmpeg", "-y",
        "-i", "music_norm.mp3",
        "-i", "voice_norm.mp3",
        "-filter_complex",
        (
            "[0:a]volume=1,"
            f"sidechaincompress=threshold={threshold}:ratio={ratio}:"
            f"attack={attack}:release={release}:makeup=1[ducked];"
            "[ducked][1:a]amix=inputs=2:duration=shortest"
        ),
        "-c:a", "aac", "-b:a", "192k",
        output_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Cleanup
    for f in ["voice_norm.mp3", "music_norm.mp3"]:
        if os.path.exists(f):
            os.remove(f)

def apply_color_grading(input_video: str,
                        lut_file: str,
                        output_video: str):
    """
    Applies a 3D LUT to input_video.
    """
    subprocess.run([
        "ffmpeg", "-y",
        "-i", input_video,
        "-vf", f"lut3d={lut_file}",
        "-c:a", "copy",
        output_video
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def mux_audio_video(input_video: str,
                    input_audio: str,
                    output_path: str):
    """
    Muxes video and audio.
    """
    subprocess.run([
        "ffmpeg", "-y",
        "-i", input_video,
        "-i", input_audio,
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        output_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
def make_stock_intro_min_duration(script_path, min_total_seconds=180, max_chunks=8):
    """
    Downloads a sequence of semantic stock clips so that their *total* duration >= min_total_seconds.
    Each clip matches a script chunk. Quality and native duration are preserved.
    """
    with open(script_path, "r", encoding="utf-8") as f:
        script_text = f.read()
    # Start with N chunks, increase if not enough total duration
    n_chunks = 4
    total_duration = 0
    video_files = []
    used_chunks = []
    while total_duration < min_total_seconds and n_chunks <= max_chunks:
        chunks = semantic_chunk_script(script_text, n_chunks)
        # For each chunk, if we don't already have a vid for it:
        for i, chunk in enumerate(chunks):
            if i < len(used_chunks):
                continue  # already did this one
            print(f"Downloading stock video for chunk {i+1}/{n_chunks} ...")
            vid = get_best_nostalgic_stock_video(chunk)
            if vid and os.path.exists(vid):
                # Get clip duration via ffprobe
                try:
                    result = subprocess.run(
                        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                         "-of", "default=noprint_wrappers=1:nokey=1", vid],
                        capture_output=True, text=True, check=True
                    )
                    dur = float(result.stdout.strip())
                except Exception:
                    dur = 0
                total_duration += dur
                video_files.append(vid)
                used_chunks.append(chunk)
                print(f"Added {vid} ({dur:.1f} sec). Total: {total_duration:.1f} sec.")
                if total_duration >= min_total_seconds:
                    break
        if total_duration < min_total_seconds:
            n_chunks += 1  # try with more chunks if not enough total

    if total_duration < min_total_seconds:
        print("⚠️ Warning: Could not reach desired total duration with available stock clips.")

    # Concatenate all video files
    concat_filelist = "concat_stock_intro.txt"
    with open(concat_filelist, "w", encoding="utf-8") as f:
        for v in video_files:
            f.write(f"file '{v}'\n")
    output_name = "stock_intro_3min.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", concat_filelist,
        "-c", "copy", output_name
    ])
    print(f"✅ Stock intro video saved as {output_name} ({total_duration:.1f} sec)")
    os.remove(concat_filelist)
    return output_name
def produce_documentary_with_enhancements(title: str,
                                          image_count: int,
                                          frame_duration: int,
                                          voice_path: str,
                                          music_path: str | None,
                                          lut_file: str | None,
                                          output_path: str):
    """
    Enhanced documentary production pipeline.
    """
    # 1) Ken Burns effect clips
    print("🎬 Creating Ken Burns effect clips...")
    generate_kenburns_clips(image_count, frame_duration)
    
    # 2) Intro/Outro
    print("🎬 Creating intro/outro...")
    create_intro_outro(title, "intro.mp4")
    create_intro_outro("End Credits", "outro.mp4")
    
    # 3) Concatenate with crossfade
    print("🎬 Combining clips with transitions...")
    clip_paths = ["intro.mp4"] + [f"clip_{i}.mp4" for i in range(image_count)] + ["outro.mp4"]
    concat_with_crossfade(clip_paths, "raw_video.mp4", cross_duration=0.5)
    
    # 4) Audio processing
    print("🎬 Mixing audio...")
    if music_path and os.path.exists(music_path):
        duck_background_music(voice_path, music_path, "mixed_audio.mp3")
        audio_to_use = "mixed_audio.mp3"
    else:
        audio_to_use = voice_path
    
    # 5) Mux video + audio
    print("🎬 Combining video and audio...")
    mux_audio_video("raw_video.mp4", audio_to_use, "prelim_video.mp4")
    
    # 6) Apply LUT if provided
    if lut_file and os.path.exists(lut_file):
        print("🎨 Applying color grading...")
        apply_color_grading("prelim_video.mp4", lut_file, output_path)
    else:
        os.rename("prelim_video.mp4", output_path)
    
    # Cleanup
    print("🧹 Cleaning up temporary files...")
    for f in ["raw_video.mp4", "prelim_video.mp4", "mixed_audio.mp3"] + \
             [f"clip_{i}.mp4" for i in range(image_count)] + \
             ["intro.mp4", "outro.mp4"]:
        if os.path.exists(f):
            os.remove(f)
    
    print(f"✅ Final video saved to {output_path}")
def prepend_stock_intro(clip_list):
    if os.path.exists("stock_intro_3min.mp4"):
        print("▶︎ Adding 3-minute stock intro at start…")
        return ["stock_intro_3min.mp4"] + clip_list
    return clip_list
if __name__ == "__main__":
    while True:
        print("="*50)
        print("Documentary Creator v2.0")
        print("="*50)
        use_limited_knowledge = input("📚 Restrict agent to specific text files? (y/n) ").lower().startswith("y")
        knowledge_text = ""

        if use_limited_knowledge:
            paths = input("📄 Enter paths to .txt files (comma-separated): ").split(",")
            paths = [p.strip() for p in paths if p.strip()]
            knowledge_text = load_knowledge_sources(paths)
            print(f"✅ Loaded {len(knowledge_text)} characters of source knowledge.\n")

        print("\n🛠 Mode:")
        print(" (1) Full generation")
        print(" (2) Images-only")
        print(" (3) Compile existing images")
        print(" (4) Stock footage mode")
        print(" (5) Script only")
        print(" (0) Exit")
        mode = input("Select a mode (0-6): ").strip()

        if mode == "0":
            print("Exiting Documentary Creator.")
            break
        elif mode == "6":
            script_path = input("Enter path to your script text file: ").strip()
            if not os.path.exists(script_path):
                print("❌ File not found.")
            else:
                with open(script_path, "r", encoding="utf-8") as f:
                    script_text = f.read()
                # Let user select Chirp3 voice and language
                language_code = input("Language code? [default=en-US]: ").strip() or "en-US"
                voice_short = input("Chirp3-HD Voice (Aoede, Puck, Charon, Kore, Fenrir, Leda, Orus, Zephyr) [default=Orus]: ").strip() or "Orus"
                out_file = input("Output audio file name (e.g., voiceover.wav) [default=voiceover.wav]: ").strip() or "voiceover.wav"
                speaking_rate = float(input("Speaking rate (0.75 = wise, 1.0 = normal, 1.25 = fast) [default=0.90]: ").strip() or "0.90")
                voice_name = f"{language_code}-Chirp3-HD-{voice_short}"
                # Initialize client
                client = texttospeech.TextToSpeechClient()
                voice = texttospeech.VoiceSelectionParams(
                    language_code=language_code,
                    name=voice_name,
                )
                print(f"🔹 Streaming TTS: {voice_name} at rate={speaking_rate}")
                process_streaming_audio(
                    script_text,
                    voice,
                    client,
                    out_file=out_file,
                    speaking_rate=speaking_rate,
                    display_individual_chunks=False
                )
        elif mode == "5":
            topic  = input("🎬 Topic? ").strip()
            tone   = input("🎭 Tone? ").strip()
            min_w  = int(input("➖ Min words/section? ").strip())
            max_w  = int(input("➕ Max words/section? ").strip())
            sample = None
            use_sample = input("🎨 Use a writing sample? (y/n): ").lower().startswith("y")
            if use_sample:
                path = input("📄 Path to your sample (.txt or .pdf): ").strip()
                if os.path.exists(path):
                    sample = load_style_sample(path)
                    print(f"✅ Loaded {len(sample)} chars of style sample\n")
                else:
                    print("❌ File not found—continuing without sample.\n")

            vid_len = int(input("⏳ Approx. video length in seconds? ").strip() or "1200")
            generate_script(topic, tone, sample, min_w, max_w, vid_len)
        
        # ─── Option 3: Compile existing images into a video ───
        if mode == "4":
            script_path = input("Enter path to your script file: ").strip()
            make_stock_intro_min_duration(script_path, min_total_seconds=180)

        if mode == "3":
            compile_existing_images()

        # ─── Option 2: Images‑only (generate images then auto‑edit) ───
        if mode == "2":
            sp    = input("📄 Script file path: ").strip()
            style = input("🎨 Image style description: ").strip()
            total = int(input("🔢 How many images? ").strip())
            wait  = int(input("⏱ Wait time per image (sec)? ").strip())

            # 1) Read the existing script and generate prompts → images
            with open(sp, "r", encoding="utf-8") as f:
                script = f.read()
            prompts = script_to_prompts(script, style, total)

            input("🟢 Press Enter to start image generation (make sure browser is open!)")
            open_browser_to_sora()
            generate_images_with_mouse(prompts, wait)

            # --- Now we have section_0.mp4, section_1.mp4, ... ---

            print("🎬 Compiling and editing the video clips into a video…")

            # 1. Create intro/outro if desired
            create_intro_outro("Documentary", "intro.mp4")
            create_intro_outro("End Credits", "outro.mp4")

            # 2. List all the video segments
            video_clips = [f"section_{i}.mp4" for i in range(total)]
            clip_paths = ["intro.mp4"] + video_clips + ["outro.mp4"]

            # 3. Concatenate all the clips
            concat_with_black_transitions(clip_paths, "raw_video.mp4")

            # 4. Music/voice muxing as before
            if os.path.exists("voice.mp3") and os.path.exists("background_music.mp3"):
                duck_background_music("voice.mp3", "background_music.mp3", "mixed_audio.mp3")
                audio_to_mux = "mixed_audio.mp3"
            elif os.path.exists("voice.mp3"):
                audio_to_mux = "voice.mp3"
            else:
                audio_to_mux = None

            if audio_to_mux:
                mux_audio_video("raw_video.mp4", audio_to_mux, "prelim_video.mp4")
                os.remove("raw_video.mp4")
            else:
                os.rename("raw_video.mp4", "prelim_video.mp4")

            if os.path.exists("FilmEmulation.cube"):
                apply_color_grading("prelim_video.mp4", "FilmEmulation.cube", "final_master.mp4")
                os.remove("prelim_video.mp4")
            else:
                os.rename("prelim_video.mp4", "final_master.mp4")

            # Clean up section videos and other temp files
            for fn in ["raw_video.mp4", "prelim_video.mp4", "mixed_audio.mp3"] + video_clips + ["intro.mp4", "outro.mp4"]:
                if os.path.exists(fn):
                    os.remove(fn)

            print("✅ Video clips have been compiled and auto‑edited into final_master.mp4")


        # ─── Option 1: Full generation (script → images → voice → final video) ───
        if mode == "1":
            # 0) Maybe load a writing sample
            use_sample = input("🎨 Do you have a writing sample to mimic? (y/n) ").lower().startswith("y")
            sample = None
            if use_sample:
                path = input("📄 Path to your sample (.txt or .pdf): ").strip()
                if os.path.exists(path):
                    sample = load_style_sample(path)
                    print(f"✅ Loaded {len(sample)} chars of style sample\n")
                else:
                    print("❌ File not found—continuing without sample.\n")

            # 1) Ask for script/image parameters
            topic  = input("🎬 Topic? ").strip()
            tone   = input("🎭 Tone? ").strip()
            min_w  = int(input("➖ Min words/section? ").strip())
            max_w  = int(input("➕ Max words/section? ").strip())
            style  = input("🎨 Image style? ").strip()
            vid_len = int(input("⏳ Total video length (sec)? ").strip() or "1200")
            frame   = int(input(f"🖼 Frame duration (sec)? ({IMAGE_DURATION}) ") or IMAGE_DURATION)
            wait    = int(input(f"⏱ Wait per image (sec)? ({PROMPT_WAIT_TIME}) ") or PROMPT_WAIT_TIME)

            total_imgs = vid_len // frame

            # 2) Generate the full script (automatically splits into sections)
            script = generate_script(topic, tone, sample, min_w, max_w, vid_len)

            # 3) Possibly skip image generation
            skip_images = input("🟢 Skip image generation and use existing image_<n>.png files? (y/n) ").lower().startswith("y")
            if not skip_images:
                prompts = script_to_prompts(script, style, total_imgs)
                input("🟢 Press Enter to generate images (make sure browser is open!)")
                open_browser_to_sora()
                generate_images_with_mouse(prompts, wait)
            else:
                print("⚠️ Skipping image generation; assuming image_<n>.png files already exist.")

            # 4) Possibly skip voice generation
            skip_voice = input("🟢 Skip voice‑over generation (use existing voice.mp3 or leave silent)? (y/n) ").lower().startswith("y")
            if not skip_voice:
                print("🎤 Generating voice‑over from script…")
            else:
                print("⚠️ Skipping voice generation; assuming 'voice.mp3' is already present (or leaving audio silent).")


            # 5) Build the video from (image_<n>.png + optional voice.mp3)
            print("🎬 Compiling images (and optional voice) into a first draft…")

            # 5a) Create Ken Burns clips for each image_<n>.png
            #     Use the same generate_kenburns_clips signature as in your “Full gen” code:
            generate_kenburns_clips(total_imgs, frame_duration=frame)

            # 5b) Create an intro/outro title card
            create_intro_outro(topic,       "intro.mp4")
            create_intro_outro("End Credits","outro.mp4")

            # 5c) Cross‑fade concatenate [intro.mp4] + [all clip_<n>.mp4] + [outro.mp4] → raw_video.mp4
            clip_paths = ["intro.mp4"] + [f"clip_{i}.mp4" for i in range(total_imgs)] + ["outro.mp4"]
            concat_with_crossfade(clip_paths, "raw_video.mp4")

            # 5d) Mux with voice.mp3 if it exists
            if os.path.exists("voice.mp3"):
                mux_audio_video("raw_video.mp4", "voice.mp3", "prelim_video.mp4")
                os.remove("raw_video.mp4")
            else:
                # No voice track found; rename raw_video → prelim_video
                os.rename("raw_video.mp4", "prelim_video.mp4")

            # 5e) Apply LUT if available, else just rename preliminary
            if os.path.exists("FilmEmulation.cube"):
                apply_color_grading("prelim_video.mp4", "FilmEmulation.cube", "final_master.mp4")
                os.remove("prelim_video.mp4")
            else:
                os.rename("prelim_video.mp4", "final_master.mp4")

            # 5f) Cleanup all intermediate files (clip_*.mp4, intro.mp4, outro.mp4)
            for fn in os.listdir():
                if re.fullmatch(r"clip_\d+\.mp4", fn) or fn in ("intro.mp4", "outro.mp4"):
                    try:
                        os.remove(fn)
                    except OSError:
                        pass

            print("✅ Full pipeline (Option 1) complete!  ‘final_master.mp4’ has been created.")

        # If the user typed anything other than 1, 2, or 3:
        else:
            print("❌ Invalid selection. Please enter a number from 0 to 5.")
        input("\n✅ Action completed. Press Enter to return to the main menu...\n")
