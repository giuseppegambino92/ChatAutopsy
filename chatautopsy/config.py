"""
ChatAutopsy — Configurazione centrale.

Contiene: endpoint LLM locali, modelli di embedding, soglie algoritmiche,
stopword italiane/inglesi e tutti i dizionari euristici (conflitto, accusa,
scuse, affetto, pronomi, passivo-aggressività, guilt-tripping, ecc.).

Nessun valore qui punta a servizi cloud: tutti gli endpoint sono localhost.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Percorsi
# ---------------------------------------------------------------------------
PACKAGE_DIR = Path(__file__).resolve().parent
REPO_DIR = PACKAGE_DIR.parent
# Checkout dei sorgenti (pyproject accanto al pacchetto) -> dati in <repo>/data (ignorata da git);
# installazione da pip -> ~/.chatautopsy. Sovrascrivibile con CHATAUTOPSY_HOME / CHATAUTOPSY_DATA_DIR.
_DEFAULT_HOME = REPO_DIR if (REPO_DIR / "pyproject.toml").exists() else Path.home() / ".chatautopsy"
HOME_DIR = Path(os.environ.get("CHATAUTOPSY_HOME", str(_DEFAULT_HOME))).expanduser()
DATA_DIR = Path(os.environ.get("CHATAUTOPSY_DATA_DIR", str(HOME_DIR / "data"))).expanduser()
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = DATA_DIR / "logs"
SETTINGS_FILE = DATA_DIR / "settings.json"
BASE_DIR = PACKAGE_DIR  # compatibilità
# Nessuna cartella di upload: i file raw caricati vengono processati solo in memoria e mai scritti su disco.

# ---------------------------------------------------------------------------
# LLM locale (API OpenAI-compatible)
# ---------------------------------------------------------------------------
LLM_ENDPOINT_PRESETS = {
    "MLX Server (localhost:8080)": "http://localhost:8080/v1",
    "Ollama (localhost:11434)": "http://localhost:11434/v1",
    "LM Studio (localhost:1234)": "http://localhost:1234/v1",
}
MLX_PORT = int(os.environ.get("CHATAUTOPSY_MLX_PORT", "8080"))
OLLAMA_URL = os.environ.get("CHATAUTOPSY_OLLAMA_URL", "http://localhost:11434")

# Catalogo modelli gestiti automaticamente (download + avvio del server locale). Aggiornato a novembre 2026.
# backend "mlx": Apple Silicon, pesi da Hugging Face (mlx-community, 4-bit), server mlx_lm OpenAI-compatible.
# backend "ollama": qualsiasi piattaforma con Ollama installato.
# size_gb = spazio su disco (misurato), min_ram_gb = RAM consigliata; "recommend" = candidato al ⭐ per la sua fascia
# di RAM (vince quello con min_ram_gb più alto che sta nella RAM della macchina).
# Tutti i Qwen 3.x sono modelli "ibridi" che possono emettere <think>…</think>: l'agente li rimuove dalla risposta.
LLM_CATALOG = [
    {"id": "mlx-community/Qwen3.5-0.8B-4bit", "backend": "mlx", "label": "Qwen3.5 0.8B · solo test",
     "size_gb": 0.7, "min_ram_gb": 4, "tier": "test"},
    {"id": "mlx-community/Qwen3.5-4B-4bit", "backend": "mlx", "label": "Qwen3.5 4B · leggero",
     "size_gb": 3.1, "min_ram_gb": 8, "tier": "small", "recommend": True},
    {"id": "mlx-community/Qwen3.5-9B-4bit", "backend": "mlx", "label": "Qwen3.5 9B · bilanciato",
     "size_gb": 6.0, "min_ram_gb": 16, "tier": "medium", "recommend": True},
    {"id": "mlx-community/gemma-4-26b-a4b-it-4bit", "backend": "mlx", "label": "Gemma 4 26B-A4B · alternativa veloce (MoE)",
     "size_gb": 15.4, "min_ram_gb": 24, "tier": "large"},
    {"id": "mlx-community/Qwen3.6-35B-A3B-4bit", "backend": "mlx", "label": "Qwen3.6 35B-A3B · veloce (MoE)",
     "size_gb": 20.4, "min_ram_gb": 32, "tier": "large"},
    {"id": "mlx-community/Qwen3.8-27B-4bit", "backend": "mlx", "label": "Qwen3.8 27B · consigliato, il più recente",
     "size_gb": 16.1, "min_ram_gb": 32, "tier": "large", "recommend": True},
    {"id": "mlx-community/Qwen3.5-122B-A10B-4bit", "backend": "mlx", "label": "Qwen3.5 122B-A10B · massima qualità",
     "size_gb": 69.6, "min_ram_gb": 96, "tier": "xl", "recommend": True},
    {"id": "qwen3.5:4b", "backend": "ollama", "label": "Qwen3.5 4B (Ollama)", "size_gb": 3.4, "min_ram_gb": 8,
     "tier": "small", "recommend": True},
    {"id": "qwen3.5:9b", "backend": "ollama", "label": "Qwen3.5 9B (Ollama)", "size_gb": 6.6, "min_ram_gb": 16,
     "tier": "medium", "recommend": True},
    {"id": "gemma4:26b-a4b", "backend": "ollama", "label": "Gemma 4 26B-A4B (Ollama)", "size_gb": 17.0, "min_ram_gb": 24,
     "tier": "large"},
    {"id": "qwen3.6:35b-a3b", "backend": "ollama", "label": "Qwen3.6 35B-A3B (Ollama)", "size_gb": 23.0, "min_ram_gb": 32,
     "tier": "large"},
    {"id": "qwen3.8:27b", "backend": "ollama", "label": "Qwen3.8 27B (Ollama)", "size_gb": 17.0, "min_ram_gb": 32,
     "tier": "large", "recommend": True},
    {"id": "qwen3.5:122b-a10b", "backend": "ollama", "label": "Qwen3.5 122B-A10B (Ollama)", "size_gb": 81.0, "min_ram_gb": 96,
     "tier": "xl", "recommend": True},
]
DEFAULT_LLM_BASE_URL = os.environ.get("CHATAUTOPSY_LLM_URL", f"http://localhost:{MLX_PORT}/v1")
DEFAULT_LLM_MODEL = os.environ.get("CHATAUTOPSY_LLM_MODEL", "mlx-community/Qwen3.8-27B-4bit")
DEFAULT_LLM_API_KEY = os.environ.get("CHATAUTOPSY_LLM_KEY", "local-no-key")
DEFAULT_TEMPERATURE = 0.3
LLM_CONTEXT_TOKENS = int(os.environ.get("CHATAUTOPSY_CTX", "32768"))
LLM_MAX_OUTPUT_TOKENS = 4096
LLM_REQUEST_TIMEOUT_SECONDS = 900
AGENT_MAX_ITERATIONS = 8
AGENT_HISTORY_TURNS = 8
TOOL_RESULT_MAX_CHARS = 14000
DOSSIER_RESULT_MAX_CHARS = 36000
CHARS_PER_TOKEN_ESTIMATE = 3.2
SQL_TOOL_MAX_ROWS = 200
SQL_TOOL_TIMEOUT_SECONDS = 20

# ---------------------------------------------------------------------------
# Embedding locali
# ---------------------------------------------------------------------------
# Default "veloce": MiniLM multilingue (~470 MB, 384 dim, ~10 min per 40k sessioni su M-series).
# "Alta qualità": bge-m3 (~2,2 GB, 1024 dim, più lento). Entrambi scaricati una sola volta.
EMBEDDING_LIGHT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_QUALITY_MODEL = "BAAI/bge-m3"
EMBEDDING_MODEL = os.environ.get("CHATAUTOPSY_EMBED_MODEL", EMBEDDING_LIGHT_MODEL)
EMBEDDING_ALTERNATIVE_MODEL = "intfloat/multilingual-e5-large"
EMBEDDING_PRESETS = {
    "Veloce (MiniLM, 470 MB)": EMBEDDING_LIGHT_MODEL,
    "Alta qualità (bge-m3, 2,2 GB)": EMBEDDING_QUALITY_MODEL,
    "Alternativo (multilingual-e5-large, 2,2 GB)": EMBEDDING_ALTERNATIVE_MODEL,
}
EMBEDDING_BATCH_SIZE = 32
EMBEDDING_MAX_SEQ_LENGTH = 1024        # bge-m3 supporta 8192 token: 1024 basta per 45 messaggi ed evita l'attenzione quadratica su MPS
EMBEDDING_COMMIT_EVERY = 128
EMBEDDING_CACHE_CLEAR_EVERY = 500      # ogni N sessioni: torch.mps.empty_cache() + gc.collect()

# ---------------------------------------------------------------------------
# Smart Session Chunking
# ---------------------------------------------------------------------------
SESSION_GAP_SECONDS = 2700           # 45 minuti di silenzio => nuova sessione
SESSION_MAX_MESSAGES = 45            # dimensione massima blocco
SESSION_OVERLAP_MESSAGES = 5         # overlap con la sessione precedente
TRANSCRIPT_MAX_CHARS_PER_MESSAGE = 1200

# ---------------------------------------------------------------------------
# Soglie euristiche e statistiche
# ---------------------------------------------------------------------------
NIGHT_START_MINUTES = 60             # 01:00
NIGHT_END_MINUTES = 330              # 05:30
LONG_MESSAGE_CHARS = 350             # "muro di testo"
COLD_REPLY_TRIGGER_CHARS = 200       # messaggio lungo che rende "fredda" una risposta breve (score sessione)
COLD_REPLY_MAX_CHARS = 15            # risposta monosillabica
SHOUTING_UPPERCASE_RATIO = 0.6
SHOUTING_MIN_LETTERS = 5
# Smorzamento del conflict score nelle sessioni di risate/emoji positive
LAUGHTER_DAMPING_THRESHOLD = 0.30      # quota di messaggi con risate/emoji positive oltre cui si smorza
LAUGHTER_DAMPING_FACTOR = 0.4          # moltiplicatore applicato al conflict score
POSITIVE_LAUGH_EMOJIS = {"😂", "🤣", "😆", "😅", "😹", "😁", "😄", "😃", "😀", "😊", "☺️", "☺", "😜", "😝", "😛", "🥳", "😍", "🥰", "😘", "❤️", "❤", "💕"}
LAUGHTER_PATTERN = r"(?<!\w)(?:a?h?a?(?:ha|ah){2,}h?|(?:he|eh){2,}h?|(?:ih|hi){2,}|lol+|xd+|ahah\w*|ahaha\w*|hahah\w*|sjsj\w*|asdf\w*)(?!\w)"
STONEWALL_SILENCE_SECONDS = 4 * 3600
STONEWALL_GROUP_WINDOW_MESSAGES = 6     # gruppi: finestra di messaggi prima del msg lungo per verificare il dialogo 1-a-1
INITIATION_GAP_SECONDS = 6 * 3600
DOUBLE_TEXT_MIN_GAP_SECONDS = 180
DOUBLE_TEXT_MAX_GAP_SECONDS = 24 * 3600
DOUBLE_TEXT_MIN_ATTEMPTS = 3
LATENCY_MAX_SECONDS = 8 * 3600
LATENCY_MIN_REPLIES_PER_MONTH = 5
FADING_RECENT_MONTHS = 3
FADING_MIN_BASELINE_MONTHS = 4
FADING_RATIO_THRESHOLD = 1.5
FADING_Z_THRESHOLD = 2.0
LOVE_BOMBING_WINDOW_DAYS = 60
HIGH_CONFLICT_THRESHOLD = 65
# Soglia adattiva: "sessione tesa" = top 2% delle sessioni di QUESTA chat (calcolato sull'intero DB, non sul
# filtro date, così i periodi restano confrontabili), limitata tra un minimo e la soglia assoluta.
CONFLICT_RELATIVE_PERCENTILE = 98
CONFLICT_RELATIVE_MIN = 20
REPAIR_SILENCE_SECONDS = 4 * 3600
MARKOV_WINDOW_SECONDS = 180
KILLER_SILENCE_SECONDS = 4 * 3600
KILLER_EXCLUDE_NIGHT = True
KILLER_NIGHT_START_HOUR = 23
KILLER_NIGHT_END_HOUR = 7
CATALYST_MIN_STARTS = 3
TFIDF_TOP_N = 30
TFIDF_MIN_TERM_COUNT = 3
NICKNAME_MIN_COUNT = 15
NICKNAME_MIN_LIFT = 2.5
NICKNAME_MAX_DETECTED = 8
# Parole frequenti accanto ai cuori ma che non sono soprannomi (saluti, cortesie, routine)
NICKNAME_EXCLUDE = {
    "buongiorno", "buonanotte", "buonasera", "buon", "buona", "buone", "buoni", "grazie", "prego", "notte",
    "giorno", "sera", "cena", "pranzo", "colazione", "ohi", "ehi", "ciao", "salve", "dormi", "sogni", "dolci",
    "domani", "stasera", "oggi", "anche", "tanto", "tantissimo", "molto", "sempre", "mio", "mia", "tuo", "tua",
}
# Fasce di lunghezza (caratteri) e di latenza (secondi) per le distribuzioni
LENGTH_BUCKETS = [(1, 5, "1-5"), (6, 15, "6-15"), (16, 40, "16-40"), (41, 100, "41-100"), (101, 350, "101-350"),
                  (351, 10**9, ">350")]
LATENCY_BUCKETS = [(0, 30, "< 30 s"), (30, 120, "30 s-2 min"), (120, 600, "2-10 min"), (600, 1800, "10-30 min"),
                   (1800, 7200, "30 min-2 h"), (7200, 8 * 3600, "2-8 h")]
# Verdetti di equilibrio (rapporto max/min tra i partecipanti)
BALANCE_MILD_RATIO = 1.25
BALANCE_STRONG_RATIO = 1.75
EMOJI_TOP_N = 15
RRF_K = 60
RAG_CANDIDATE_POOL = 60

WEEKDAY_NAMES_IT = ["Lunedì", "Martedì", "Mercoledì", "Giovedì", "Venerdì", "Sabato", "Domenica"]

# ---------------------------------------------------------------------------
# Messaggi di sistema WhatsApp (da filtrare) — regex case-insensitive
# ---------------------------------------------------------------------------
# "Strong": filtrati sempre (notifiche che non possono essere testo umano)
SYSTEM_MESSAGE_PATTERNS_STRONG = [
    r"i messaggi e le chiamate sono crittografati end-to-end",
    r"messages and calls are end-to-end encrypted",
    r"messages to this (group|chat) are now secured",
    r"i messaggi inviati a questo gruppo sono ora protetti",
    r"il tuo codice di sicurezza .* è cambiato",
    r"il codice di sicurezza .* è cambiato",
    r"security code .* changed",
    r"^hai bloccato questo contatto",
    r"^you blocked this contact",
    r"^hai sbloccato questo contatto",
    r"^you unblocked this contact",
    r"ha attivato i messaggi effimeri|ha disattivato i messaggi effimeri",
    r"turned (on|off) disappearing messages",
    r"questa chat è con un account aziendale",
    r"this chat is with a business account",
]
# "Weak": eventi di gruppo; filtrati solo se marcati da WhatsApp col carattere invisibile U+200E
# (formato iOS "[data] Gruppo: \u200eMario ha aggiunto Luigi") per non scartare messaggi umani.
SYSTEM_MESSAGE_PATTERNS = [
    r"ha creato il gruppo",
    r"created (the )?group",
    r"ti ha aggiunto",
    r"added you",
    r"\bha aggiunto\b",
    r"\badded\b",
    r"\bha rimosso\b",
    r"\bremoved\b",
    r"\bè uscit[oa]\b",
    r"\bleft$",
    r"ha cambiato (l'oggetto|il nome|l'immagine|la descrizione|le impostazioni)",
    r"changed (the subject|this group's icon|the group description|the group name|the settings)",
    r"ha eliminato l'immagine del gruppo",
    r"deleted this group's icon",
    r"si è unit[oa] tramite il link",
    r"joined using this group's invite link",
    r"ora è un amministratore",
    r"now an admin",
    r"ha cambiato (il suo )?numero",
    r"changed (their|his|her) phone number",
    r"ha fissato un messaggio",
    r"pinned a message",
]

# ---------------------------------------------------------------------------
# Media omessi / eventi non testuali: (regex, media_type)
# ---------------------------------------------------------------------------
MEDIA_PATTERNS = [
    (r"^<?\s*(media omess[oai]|media omitted|media non inclus[oi])\s*>?$", "media"),
    (r"^(immagine omessa|image omitted|<immagine omessa>)$", "image"),
    (r"^(foto omessa|photo omitted)$", "image"),
    (r"^(video omesso|video omitted)$", "video"),
    (r"^(audio omesso|audio omitted|messaggio vocale omesso|voice message omitted)$", "audio"),
    (r"^(sticker omesso|sticker omitted|adesivo omesso)$", "sticker"),
    (r"^(gif omessa|gif omesso|gif omitted)$", "gif"),
    (r"^(documento omesso|document omitted)$", "document"),
    (r"^.*\b(documento omesso|document omitted)$", "document"),
    (r"^(scheda contatto omessa|contact card omitted)$", "contact"),
    (r"^<?\s*(allegato|attached)\s*:\s*.+\.(jpe?g|png|webp|heic)\s*>?$", "image"),
    (r"^<?\s*(allegato|attached)\s*:\s*.+\.(mp4|mov|3gp)\s*>?$", "video"),
    (r"^<?\s*(allegato|attached)\s*:\s*.+\.(opus|m4a|mp3|aac|ogg)\s*>?$", "audio"),
    (r"^<?\s*(allegato|attached)\s*:\s*.+\.(webp|was)\s*>?$", "sticker"),
    (r"^<?\s*(allegato|attached)\s*:\s*.+>?$", "document"),
    (r"^.+\.(jpe?g|png|webp|heic) \((file allegato|file attached)\)$", "image"),
    (r"^.+\.(opus|m4a|mp3) \((file allegato|file attached)\)$", "audio"),
    (r"^.+\.(mp4|mov) \((file allegato|file attached)\)$", "video"),
    (r"^.+ \((file allegato|file attached)\)$", "document"),
    (r"^(posizione|location|posizione in tempo reale condivisa|live location shared)\b.*", "location"),
    (r"^(questo messaggio è stato eliminato|hai eliminato questo messaggio|this message was deleted|you deleted this message|messaggio eliminato)\.?$", "deleted"),
    (r"^(chiamata vocale persa|chiamata vocale|videochiamata persa|videochiamata|missed voice call|missed video call|voice call|video call)\b.*", "call"),
    (r"^(messaggio visualizzabile una volta omesso|view once message omitted)$", "view_once"),
    (r"^(sondaggio|poll)\s*:.*", "poll"),
    (r"^null$", "media"),
]
EDITED_SUFFIX_PATTERN = r"\s*<\s*(questo messaggio è stato modificato|this message was edited|messaggio modificato)\s*>\s*$"
# Media che, in risposta a un messaggio lungo, contano come risposta "glaciale"
COLD_MEDIA_TYPES = {"sticker", "gif", "deleted"}

# ---------------------------------------------------------------------------
# Stopword (italiano + inglese) — solo parole funzionali, NON risate/intercalari
# ---------------------------------------------------------------------------
ITALIAN_STOPWORDS = """
a ad agli ai al alla alle allo agl all anche ancora altro altri altra altre avere aveva avevano avevo avevi avete abbiamo
anzi assai avrà avrai avrei avresti avrebbe avremmo avreste avrebbero avuto avuta avuti avute ben bene buono c ce ci
cio ciò che chi cioè come comunque con contro cosa cosi così cui da dal dall dalla dalle dallo dagli dai de degli dei del
dell della delle dello dentro deve devi devo devono dove dunque di do due e è ed egli ella ecco era erano eravamo
eri ero essa esse essi esso essere fa fai faccio facciamo fanno fare fatto fra fu fui fummo furono gli già gia giu giù
ha hai hanno ho i il in invece io l la le lei li lo loro lui ma me mentre mi mia mie miei mio molto molta molti molte
ne nei nel nell nella nelle nello negli né nemmeno neppure noi non nostra nostre nostri nostro o od ogni oppure
ora per perché perche perciò però pero più piu poco poca pochi poche poi possa posso può puo puoi qua quale quali
qualche qualcosa qualcuno quando quanto quanta quanti quante quasi quel quella quelle quelli quello questa queste
questi questo qui quindi se sei sembra senza si sia siamo siano siate siete sono sta stai stanno stato stata stati state
sto su sua sue sui sul sull sulla sulle sullo suo suoi ti tra tu tua tue tuo tuoi tutta tutte tutti tutto un una uno
vi voi vostra vostre vostri vostro è sì si no c' l' un' dell' all' nell' sull' dall' po' fosse fossi fossero sarà sarai
sarò saranno sarei saresti sarebbe saremmo sareste sarebbero stavo stava stavamo stavano stesso stessa stessi stesse
tanto tanta tanti tante troppo tale tali essendo avendo facendo detto dire dici dico dice vuoi voglio vuole vogliamo
vado vai va andiamo vanno sto stai sta stiamo stanno ho hai ha abbiamo avete hanno nn x xk xke cmq ke ki
"""
ENGLISH_STOPWORDS = """
a about above after again against all am an and any are aren't as at be because been before being below between both
but by can can't cannot could couldn't did didn't do does doesn't doing don't down during each few for from further had
hadn't has hasn't have haven't having he he'd he'll he's her here here's hers herself him himself his how how's i i'd
i'll i'm i've if in into is isn't it it's its itself let's me more most mustn't my myself no nor not of off on once only
or other ought our ours ourselves out over own same shan't she she'd she'll she's should shouldn't so some such than
that that's the their theirs them themselves then there there's these they they'd they'll they're they've this those
through to too under until up very was wasn't we we'd we'll we're we've were weren't what what's when when's where
where's which while who who's whom why why's with won't would wouldn't you you'd you'll you're you've your yours
yourself yourselves just get got also will im dont
"""
# Intercalari di chat italiani privi di valore semantico (esclusi dal TF-IDF)
CHAT_FILLER_STOPWORDS = """
eh ah oh mah boh beh bah mh mmh mmm uh uff ehm ahaha ahah ahahah ahahaha ahahahah hahaha haha hahah hahahah ahha
ok okok okk okay oki va bene sì si no già gia dai poi ora lì li là qui qua cosa fatto detto tipo comunque cmq vabbè vabbe
vabbene vbb ecco allora quindi niente nulla vero sisi nono
"""
STOPWORDS = sorted({w.strip().lower() for w in (ITALIAN_STOPWORDS + " " + ENGLISH_STOPWORDS + " " + CHAT_FILLER_STOPWORDS).split() if w.strip()})
# Parole-placeholder dei media da escludere sempre dall'analisi lessicale
MEDIA_NOISE_WORDS = {"omesso", "omessa", "omitted", "media", "immagine", "allegato", "attached", "sticker", "gif", "null"}

# ---------------------------------------------------------------------------
# Dizionari euristici (lemmi e frasi; minuscolo, match a confine di parola)
# ---------------------------------------------------------------------------
CONFLICT_LEXICON = [
    "basta", "smettila", "lasciami stare", "lasciami in pace", "non ce la faccio più", "non ne posso più", "sono stanco",
    "sono stanca", "mi hai stufato", "mi hai rotto", "mi hai stancato", "che palle", "vaffanculo", "fanculo", "cazzo",
    "stronzo", "stronza", "idiota", "cretino", "cretina", "deficiente", "coglione", "cogliona", "egoista", "bugiardo",
    "bugiarda", "falso", "falsa", "ipocrita", "litigare", "litigio", "litighiamo", "discussione", "arrabbiato",
    "arrabbiata", "incazzato", "incazzata", "nervoso", "nervosa", "delusione", "deluso", "delusa", "mi deludi",
    "offeso", "offesa", "ferito", "ferita", "mi hai ferito", "mi hai ferita", "non mi rispetti", "rispetto",
    "non mi ascolti", "non capisci", "non hai capito niente", "sei incredibile", "ridicolo", "ridicola", "assurdo",
    "assurda", "vergogna", "vergognati", "odio", "ti odio", "chiudiamola qui", "finita", "è finita", "lasciamoci",
    "ci lasciamo", "addio", "non mi cercare", "non scrivermi", "sparisci", "problema", "problemi", "colpa",
    "perché fai così", "che cosa vuoi", "cosa vuoi", "ma che dici", "ma smettila", "per l'ennesima volta",
    "ogni volta", "sempre la stessa storia", "sempre così", "sei sempre", "non sei mai", "stupido", "stupida",
    "angry", "fight", "shut up", "leave me alone", "hate", "liar", "selfish", "stop it", "whatever",
]
ACCUSATION_LEXICON = [
    "colpa tua", "è colpa tua", "per colpa tua", "tu non", "tu sempre", "tu mai", "sei tu", "sei sempre",
    "non sei mai", "fai sempre", "fai sempre così", "non fai mai", "mi hai mentito", "hai mentito", "mi menti",
    "mi prendi in giro", "mi hai preso in giro", "mi hai presa in giro", "non ti importa", "non ti interessa",
    "te ne freghi", "te ne sei fregato", "te ne sei fregata", "pensi solo a te", "pensi solo a te stesso",
    "pensi solo a te stessa", "esageri", "stai esagerando", "inventi", "ti inventi", "te lo sei sognato",
    "te lo sei sognata", "sei pazzo", "sei pazza", "sei matto", "sei matta", "paranoie", "paranoico", "paranoica",
    "it's your fault", "you always", "you never",
]
NEGATION_LEXICON = [
    "no", "non", "mai", "niente", "nulla", "nessuno", "nessuna", "neanche", "nemmeno", "neppure", "macché",
    "per niente", "assolutamente no", "never", "nothing", "nope",
]
AFFECTION_LEXICON = [
    "amore", "amo", "amoruccio", "amorino", "amoreee", "tesoro", "tesorino", "cuore", "cuoricino", "vita mia",
    "cucciolo", "cucciola", "patato", "patata", "bimba", "bimbo", "principessa",
    "baby", "honey", "ti amo", "ti voglio bene", "tvb", "tvtb", "ti adoro", "mi manchi", "mi sei mancato",
    "mi sei mancata", "un bacio", "baci", "bacio", "bacione", "bacioni", "abbraccio", "abbracci", "coccole",
    "sei bellissima", "sei bellissimo", "sei stupenda", "sei stupendo", "love you", "miss you", "sweetheart", "darling",
]
# Termini monitorati esplicitamente nella Affection & Pet-Name Timeline
PET_NAME_TERMS = [
    "amore", "amo", "tesoro", "cuore", "vita", "piccola", "piccolo", "cucciolo", "cucciola", "bimba", "bimbo",
    "principessa", "patata", "ti amo", "ti voglio bene", "tvb", "mi manchi", "ti adoro", "baci", "bacio",
]
HEART_EMOJIS = {"❤️", "❤", "😍", "🥰", "😘", "💕", "💖", "💗", "💓", "💞", "💘", "💝", "♥️", "♥", "😻", "🫶", "💜", "💙", "💚", "🧡", "🤍", "🖤", "🤎", "💛", "❣️"}
NEGATIVE_EMOJIS = {"🙄", "😒", "😤", "😡", "🤬", "😠", "💔", "😑", "😐", "🖕", "😾", "🤦", "🤦‍♀️", "🤦‍♂️"}
TRACKED_EMOJI_GROUPS = {
    "❤️ Cuori & affetto": HEART_EMOJIS,
    "😂 Risate": {"😂", "🤣", "😹", "😆", "😅"},
    "🙄 Fastidio & freddezza": NEGATIVE_EMOJIS,
    "😢 Tristezza": {"😢", "😭", "🥺", "😞", "😔", "☹️", "🙁", "😿"},
}

# Pronomi (Indice di Gottman "Noi" vs "Io/Tu")
WE_PRONOUNS = ["noi", "ci", "nostro", "nostra", "nostri", "nostre", "insieme", "we", "us", "our", "together"]
I_PRONOUNS = ["io", "mio", "mia", "miei", "mie", "me", "my", "mine"]  # "i" escluso: collide con l'articolo italiano
YOU_PRONOUNS = ["tu", "tuo", "tua", "tuoi", "tue", "te", "voi", "vostro", "vostra", "you", "your"]
ABSOLUTE_TERMS = ["sempre", "mai", "always", "never"]

APOLOGY_LEXICON = [
    "scusa", "scusami", "scusate", "mi scuso", "chiedo scusa", "perdonami", "perdono", "colpa mia", "è colpa mia",
    "mi dispiace", "mi spiace", "mi dispiace tanto", "ho sbagliato", "hai ragione", "avevi ragione", "ho esagerato",
    "non volevo", "non succederà più", "sorry", "my bad", "i'm sorry", "forgive me", "my fault",
]
BLAME_LEXICON = [
    "colpa tua", "è colpa tua", "per colpa tua", "tu non", "esageri", "stai esagerando", "inventi", "ti inventi",
    "te lo sei sognato", "te lo sei sognata", "sei pazzo", "sei pazza", "sei matto", "sei matta", "paranoie",
    "paranoico", "paranoica", "fai sempre così", "fai sempre", "non fai mai", "è sempre colpa tua", "è un problema tuo",
    "problema tuo", "sei tu il problema", "sei tu che", "te la prendi per niente", "sei troppo sensibile",
    "non è mai successo", "non l'ho mai detto", "non ho mai detto", "ti ricordi male", "ricordi male", "drammatico",
    "drammatica", "melodrammatico", "melodrammatica", "your fault", "you're crazy", "you're overreacting",
]
GASLIGHTING_LEXICON = [
    "te lo sei sognato", "te lo sei sognata", "non è mai successo", "non l'ho mai detto", "non ho mai detto",
    "ti ricordi male", "ricordi male", "inventi", "ti inventi", "ti stai inventando", "sei pazzo", "sei pazza",
    "sei matto", "sei matta", "paranoie", "paranoico", "paranoica", "esageri", "stai esagerando",
    "sei troppo sensibile", "te la prendi per niente", "è tutto nella tua testa", "ti fai i film", "ti fai film",
    "ti fai i viaggi", "ti fai le paranoie", "non hai capito niente", "stai sclerando", "sei fuori", "sei fuori di testa",
    "you're crazy", "that never happened", "you're imagining",
]
PASSIVE_AGGRESSIVE_LEXICON = [
    "fai come vuoi", "fai come ti pare", "come ti pare", "come vuoi", "contento tu", "contenta tu", "va bene così",
    "certo certo", "sì sì certo", "si si certo", "ok.", "va bene.", "bene.", "perfetto.", "ottimo.", "figurati",
    "nessun problema", "tranquillo", "tranquilla", "non fa niente", "lascia stare", "lascia perdere", "lasciamo perdere",
    "non importa", "ah ok", "ah ecco", "ah bene", "buon per te", "complimenti", "grazie eh", "grazie mille eh",
    "simpatico", "simpatica", "divertiti", "fine.", "whatever", "k.", "sure.", "fine",
]
GUILT_TRIPPING_LEXICON = [
    "dopo tutto quello che ho fatto", "dopo tutto quello che faccio", "con tutto quello che ho fatto",
    "non ti importa di me", "non ti importa niente", "se mi amassi", "se mi volessi bene", "se ci tenessi",
    "mi fai sentire", "mi fai stare male", "per colpa tua sto male", "sto male per te", "non pensi mai a me",
    "sempre io", "faccio tutto io", "sono sempre io", "io ci sono sempre", "io per te ci sono sempre", "ti ho dato tutto",
    "ho rinunciato", "mi sacrifico", "mi hai deluso", "mi hai delusa", "non meriti", "pensavo fossi diverso",
    "pensavo fossi diversa", "chissà cosa direbbe", "alla fine sono io quello", "alla fine sono io quella",
    "if you loved me", "after everything i did",
]
# ---------------------------------------------------------------------------
# Inglese — chat internazionali. Solo espressioni che non collidono con l'italiano
# (niente "male", "come", "so", "i"…). Il pronome "i" è in I_PRONOUNS_EN_ONLY: attivo solo su chat in inglese.
# ---------------------------------------------------------------------------
CONFLICT_LEXICON += [
    "fuck", "fuck you", "fuck off", "wtf", "bullshit", "stupid", "idiot", "asshole", "jerk", "pathetic", "ridiculous",
    "unbelievable", "i'm done", "we're done", "it's over", "i hate you", "annoying", "annoyed", "pissed", "pissed off",
    "furious", "mad at you", "disappointed", "you're impossible", "every single time", "don't text me", "screw you",
    "seriously?", "not again", "i can't do this anymore", "so tired of this", "get lost",
]
ACCUSATION_LEXICON += [
    "your fault", "you lied", "you lie", "you don't care", "you only think about yourself", "you're overreacting",
    "you're exaggerating", "you made me", "you never listen", "you don't listen", "it's always you", "you're making things up",
]
NEGATION_LEXICON += ["not", "don't", "didn't", "can't", "won't", "no way", "nobody", "nowhere"]
AFFECTION_LEXICON += [
    "i love you", "love u", "luv you", "love ya", "babe", "my love", "sweetie", "cutie", "handsome", "beautiful",
    "gorgeous", "kisses", "xoxo", "hugs", "cuddles", "i miss you", "missing you", "can't wait to see you",
    "my everything", "sweet dreams",
]
PET_NAME_TERMS += ["babe", "baby", "honey", "sweetheart", "darling", "sweetie", "my love", "love you", "miss you", "kisses"]
WE_PRONOUNS += ["we're", "we've", "ours", "let's"]
I_PRONOUNS += ["i'm", "i've", "i'll", "i'd", "myself"]
I_PRONOUNS_EN_ONLY = ["i"]
YOU_PRONOUNS += ["you're", "yours", "yourself", "u"]
ABSOLUTE_TERMS += ["every time", "all the time"]
APOLOGY_LEXICON += [
    "i apologize", "i apologise", "so sorry", "i was wrong", "you're right", "you were right", "i messed up",
    "my mistake", "please forgive me",
]
BLAME_LEXICON += [
    "because of you", "you made me do it", "it's on you", "you're too sensitive", "you're being dramatic",
    "your problem", "that's your problem", "you're the problem", "you always do this",
]
GASLIGHTING_LEXICON += [
    "you're overreacting", "you're too sensitive", "i never said that", "you're remembering it wrong",
    "it's all in your head", "you're making it up", "you're being paranoid", "stop being dramatic", "you're insane",
]
PASSIVE_AGGRESSIVE_LEXICON += [
    "whatever you want", "do what you want", "suit yourself", "if you say so", "good for you", "thanks a lot",
    "it's fine", "i'm fine", "nevermind", "never mind", "forget it", "ok then", "cool.", "great.", "sure, whatever",
]
GUILT_TRIPPING_LEXICON += [
    "after all i've done", "after everything i've done for you", "you don't care about me", "if you really cared",
    "you make me feel", "i always have to", "i do everything", "i gave you everything", "i sacrificed",
    "you disappointed me", "i thought you were different",
]
NICKNAME_EXCLUDE |= {"good", "morning", "night", "goodnight", "thanks", "thank", "dinner", "lunch", "tonight", "today",
                     "tomorrow", "sleep", "dreams", "sweet", "hey", "hello", "you", "too", "much", "very", "really"}

COLD_REPLY_EXAMPLES = ["ok", "k", "fine", "sure", "cool", "yeah", "va bene", "fai come vuoi", "capito", "👍", "ok.", "sì", "si", "mh", "mmh", "boh", "vabbè", "vabbe", "ah", "bene"]

# Query semantiche usate dalle pipeline di red flag (rag_engine + agent)
RED_FLAG_SEMANTIC_QUERIES = {
    "gaslighting": "negare ciò che è successo, dire all'altro che si sta inventando le cose, che è pazzo, che esagera, che ricorda male, ribaltare la realtà e la colpa",
    "stonewalling": "uno dei due scrive un lungo messaggio emotivo e l'altro risponde freddo, ok, fai come vuoi, oppure sparisce senza rispondere per ore",
    "love_bombing_devaluation": "dichiarazioni d'amore intense, complimenti esagerati, promesse e attenzioni continue, poi freddezza, critiche e distacco",
    "passive_aggressive": "risposte sarcastiche e ironiche, fai come ti pare, contento tu, certo certo, ok punto, frecciatine e tono freddo",
    "guilt_tripping": "far sentire in colpa l'altro, dopo tutto quello che ho fatto per te, se mi amassi, mi fai stare male, sono sempre io a fare tutto",
}

# Pulsanti one-click del tab AI Detective
DEEP_DIVE_PROMPTS = {
    "🧭 Radiografia Relazionale & Asimmetrie di Potere": (
        "Esegui una radiografia relazionale completa: chi investe di più, chi insegue e chi si distanzia (initiation rate, "
        "double/triple texting, effort ratio, latenze direzionali e trend di fading). Per ogni asimmetria fornisci Prova "
        "Matematica e Prova Forense con citazioni reali."
    ),
    "🚩 Trova tutte le Red Flag (Gaslighting, Stonewalling, Manipolazione) con prove": (
        "Analizza in modo sistematico gaslighting, stonewalling, love bombing/svalutazione, passivo-aggressività e "
        "guilt-tripping usando analyze_specific_red_flag per ciascuno. Classifica le red flag per gravità e per ciascuna "
        "riporta Prova Matematica e almeno 2-3 citazioni reali."
    ),
    "⚔️ Analizza come litighiamo, per quali motivi e chi fa pace": (
        "Analizza i conflitti: quando avvengono (giorno/ora), i temi ricorrenti dei litigi leggendo le sessioni con "
        "conflict_score più alto, lo stile di ciascuno durante il litigio (scuse vs accuse, pronomi, muri di testo) e chi "
        "fa il primo passo per fare pace (conflict recovery). Cita estratti reali."
    ),
    "📈 Come è cambiato il nostro rapporto dal primo anno ad oggi?": (
        "Confronta il primo anno di chat con l'ultimo anno: volumi, latenze, affetto e soprannomi, indice Noi vs Io/Tu, "
        "emoji, conflittualità, love bombing vs svalutazione. Individua i punti di svolta mese per mese e citane le "
        "conversazioni reali."
    ),
    "🎭 Estrai la Lore: inside joke, momenti epici e tormentoni della chat": (
        "Estrai la lore della chat: parole identitarie TF-IDF di ciascuno, tormentoni, soprannomi e quando sono nati, "
        "inside joke ricorrenti, momenti epici e le sessioni con più intensità emotiva positiva. Cita le conversazioni reali."
    ),
}

DEEP_DIVE_PROMPTS_EN = {
    "🧭 Relationship X-ray & power asymmetries": (
        "Run a complete relationship X-ray: who invests more, who chases and who withdraws (initiation rate, "
        "double/triple texting, effort ratio, directional latencies and fading trends). For every asymmetry give a "
        "Mathematical Proof and a Forensic Proof with real quotes."
    ),
    "🚩 Find every red flag (gaslighting, stonewalling, manipulation) with evidence": (
        "Systematically analyse gaslighting, stonewalling, love bombing/devaluation, passive aggression and guilt-tripping "
        "using analyze_specific_red_flag for each. Rank the red flags by severity and for each one report the "
        "Mathematical Proof and at least 2-3 real quotes."
    ),
    "⚔️ How do we fight, about what, and who makes peace?": (
        "Analyse the conflicts: when they happen (day/hour), recurring topics (read the sessions with the highest "
        "conflict_score), each person's style during a fight (apologies vs blame, pronouns, walls of text) and who takes "
        "the first step to make peace (conflict recovery). Quote real excerpts."
    ),
    "📈 How has our relationship changed since the first year?": (
        "Compare the first year of the chat with the last one: volumes, latencies, affection and pet names, We vs I/You "
        "index, emoji, conflict, love bombing vs devaluation. Find the turning points month by month and quote the real "
        "conversations."
    ),
    "🎭 Extract the lore: inside jokes, epic moments, catchphrases": (
        "Extract the chat's lore: each person's TF-IDF signature words, catchphrases, pet names and when they were born, "
        "recurring inside jokes, epic moments and the sessions with the highest positive emotional intensity. Quote the "
        "real conversations."
    ),
}

GROUP_SIMULATION_LABEL = "👥 Intero Gruppo/Coppia"
