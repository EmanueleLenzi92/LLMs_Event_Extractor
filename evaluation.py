import json
import re
import nltk

nltk.download("punkt")

DEBUG_PRINT_BROKEN = False     # metti True per vedere le broken sentences
DEBUG_ONLY_MODEL = None        # es: "gemma2:9b-instruct-q8_0" oppure None per tutti
DEBUG_ONLY_TITLE = None        # es: "LONGAustrianAlps" oppure None per tutte
DEBUG_MAX_BROKEN_TO_PRINT = 50 # limite totale righe di debug
DEBUG_PRINT_BROKEN = True
DEBUG_ONLY_MODEL = "phi3.5:latest"
DEBUG_ONLY_TITLE = "NorthernApenninesItaly"   # o il titolo esatto nel tuo JSON
DEBUG_MAX_BROKEN_TO_PRINT = 200

# ======================
# CONFIGURAZIONE
# ======================


SUFFIX= "shortNarratives"
SUFFIX= "5shortNarrativesParagraphs"
SUFFIX = "allNarrativesParagraphs"
SUFFIX= "longFiveNarratives"

NARRATIVES_DICT_FILE = f"narratives_dict_output-{SUFFIX}.json"
METADATA_FILE = f"metadata-{SUFFIX}.json"

# Output: vuoi entrambi
PRINT_FULL_WITH_DETAILS = True
PRINT_MODEL_TABLE_ONLY = True

# Default regole split (fallback)
DEFAULT_EVENT_RULES = ["event_label", "numbered", "bullet", "blankline", "newline"]

# Quanti output campionare per narrazione per inferire le regole per modello
INFER_SAMPLE_PER_TITLE = 2

# Se vuoi forzare manualmente regole per un modello:
MANUAL_MODEL_EVENT_RULES = {
    # "llama3:8b-instruct-q8_0": ["numbered", "blankline", "newline"],
}

# Punteggiatura di fine frase (come richiesto)
END_PUNCT = {".", "!", "?"}

# ======================
# FUNZIONI BASE
# ======================

def jaccard_similarity(str1, str2):
    str1 = strip_think_blocks(str1)
    str2 = strip_think_blocks(str2)

    set1 = set((str1 or "").split())
    set2 = set((str2 or "").split())
    inter = len(set1 & set2)
    union = len(set1 | set2)
    return inter / union if union else 0.0


def strip_think_blocks(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def count_sentences(text: str) -> int:
    return len(nltk.sent_tokenize(text or ""))


def strip_event_prefix(seg: str) -> str:
    """
    Rimuove prefissi tipo '1.' / '2)' / '- ' / 'Event 3:' ecc
    per evitare tokenizzazioni errate.
    """
    if not seg:
        return ""
    s = seg.strip()

    # normalizza markdown bold tipo **Event 1:** o __Event 1:__
    s = re.sub(r"^\s*(\*\*|__)\s*", "", s)
    s = re.sub(r"\s*(\*\*|__)\s*", "", s)

    # Event X:
    s = re.sub(r"^\s*Event\s*\d+\s*:\s*", "", s, flags=re.IGNORECASE)

    # Numerazione: 1. / 1) / 1:
    s = re.sub(r"^\s*\d+\s*[\.\)\:]\s*", "", s)

    # Bullet: -, *, •
    s = re.sub(r"^\s*[\*\-\u2022]\s*", "", s)

    return s.strip()


# ======================
# SPLIT EVENTI SMART PER MODELLO
# ======================

def infer_model_event_rules(narratives_dict_for_model: dict, sample_per_title: int = 2):
    """
    Inferisce l'ordine delle regole di split più probabile per un modello
    guardando un campione di output.
    """
    counts = {"event_label": 0, "numbered": 0, "bullet": 0, "blankline": 0, "newline": 0}

    for _title, prompts_dict in (narratives_dict_for_model or {}).items():
        iters = (prompts_dict.get("prompt_0") or {})
        if not isinstance(iters, dict) or not iters:
            continue

        for k in list(iters.keys())[:sample_per_title]:
            t = strip_think_blocks(iters.get(k) or "")

            if re.search(r"^\s*Event\s*\d+\s*:", t, flags=re.MULTILINE | re.IGNORECASE):
                counts["event_label"] += 1
            if re.search(r"^\s*\d+[\.\)]\s+", t, flags=re.MULTILINE):
                counts["numbered"] += 1
            if re.search(r"^\s*[\*\-\u2022]\s+", t, flags=re.MULTILINE):
                counts["bullet"] += 1
            if re.search(r"\n\s*\n", t):
                counts["blankline"] += 1
            if "\n" in t:
                counts["newline"] += 1

    ranked = sorted(
        ["event_label", "numbered", "bullet", "blankline"],
        key=lambda r: counts[r],
        reverse=True,
    )

    rules = [r for r in ranked if counts[r] > 0] + ["newline"]
    return rules if rules else DEFAULT_EVENT_RULES[:]


def split_events_smart(text: str, model_name: str, model_rules: dict):
    """
    Ritorna lista di segmenti-evento usando regole smart per modello (con fallback).
    """
    if not text:
        return []

    t = strip_think_blocks(text)
    if not t:
        return []

    rules = model_rules.get(model_name, DEFAULT_EVENT_RULES)

    def split_on_anchors(pattern):
        matches = list(re.finditer(pattern, t, flags=re.MULTILINE | re.IGNORECASE))
        if not matches:
            return []
        starts = [m.start() for m in matches] + [len(t)]
        chunks = []
        for i in range(len(starts) - 1):
            chunk = t[starts[i]:starts[i + 1]].strip()
            if chunk:
                chunks.append(chunk)
        return chunks

    for rule in rules:
        if rule == "event_label":
            chunks = split_on_anchors(r"^\s*Event\s*\d+\s*:")
            if chunks:
                return chunks

        elif rule == "numbered":
            chunks = split_on_anchors(r"^\s*\d+[\.\)]\s+")
            if chunks:
                return chunks

        elif rule == "bullet":
            chunks = split_on_anchors(r"^\s*[\*\-\u2022]\s+")
            if chunks:
                return chunks

        elif rule == "blankline":
            if re.search(r"\n\s*\n", t):
                chunks = [c.strip() for c in re.split(r"\n\s*\n", t) if c.strip()]
                if chunks:
                    return chunks

        elif rule == "newline":
            t2 = re.sub(r"\n+", "\n", t)
            chunks = [c.strip() for c in t2.split("\n") if c.strip()]
            if chunks:
                return chunks

    return []

def strip_non_sentence_lines_Vecchia(text: str) -> str:
    """
    Rimuove righe che sono chiaramente metadati/titoli e non frasi contenutistiche:
    - headings markdown (#, ##, ...)
    - righe tipo 'Here are the events ...:' (prefazioni che finiscono con ':')
    - righe titolo brevi (es. 'Revival Efforts') che stanno da sole e non terminano con .!?
    """
    if not text:
        return ""
    s = text.replace("\r\n", "\n").replace("\r", "\n")

    out = []
    for line in s.split("\n"):
        l = line.strip()
        if not l:
            continue

        # 1) markdown headings
        if re.match(r"^\s*#{1,6}\s+", l):
            continue

        # 2) prefazioni/label brevi che finiscono con ':'
        if l.endswith(":") and len(l.split()) <= 20:
            # es: "Here are the events ...:" oppure "Events:"
            continue

        # 3) titoletti da soli: riga breve che NON finisce con .!?,
        #    e che non sembra una frase (euristica: poche parole e nessuna virgola/punto e virgola)
        if (l[-1] not in ".!?"
            and len(l.split()) <= 8
            and "," not in l
            and ";" not in l
            and ":" not in l):
            # es: "Revival Efforts" / "Decline of Sheep Farming"
            continue

        out.append(line)

    return "\n".join(out).strip()

def strip_non_sentence_lines(text: str) -> str:
    """
    Rimuove righe che sono chiaramente metadati/titoli e non frasi contenutistiche:
    - headings markdown (#, ##, ...)
    - righe tipo 'Here are the events ...:' (prefazioni che finiscono con ':')
    - titoli di sezione tipo 'Section 1: ...' (anche dentro ** **)
    - righe titolo brevi (es. 'Revival Efforts') che stanno da sole e non terminano con .!?
    """
    if not text:
        return ""
    s = text.replace("\r\n", "\n").replace("\r", "\n")

    out = []
    for line in s.split("\n"):
        l = line.strip()
        if not l:
            continue

        # 1) markdown headings
        if re.match(r"^\s*#{1,6}\s+", l):
            continue

        # Normalizza eventuale bold markdown che racchiude tutta la riga
        l2 = re.sub(r"^\s*(\*\*|__)\s*", "", l)
        l2 = re.sub(r"\s*(\*\*|__)\s*$", "", l2).strip()

        # 2) prefazioni/label brevi che finiscono con ':'
        if l2.endswith(":") and len(l2.split()) <= 20:
            continue

        # 3) titoli di sezione tipo "Section 1: Foo" / "Part 2: Bar" / "Chapter 3: Baz"
        if re.match(r"^(Section|Part|Chapter)\s+\d+\s*:\s+.+$", l2, flags=re.IGNORECASE):
            continue

        # 4) titoletti da soli (senza punteggiatura finale e senza segni tipici di frase)
        if (l2[-1] not in ".!?"
            and len(l2.split()) <= 8
            and "," not in l2
            and ";" not in l2
            and ":" not in l2):
            continue

        out.append(line)

    return "\n".join(out).strip()

def strip_event_headers(text: str) -> str:
    """
    Rimuove righe di intestazione tipo Markdown headings (es. '## ...')
    e label brevi che finiscono con ':'.
    Serve a evitare falsi broken tipo '## Events ...:'.
    """
    if not text:
        return ""
    s = text.replace("\r\n", "\n").replace("\r", "\n")

    keep = []
    for line in s.split("\n"):
        l = line.strip()
        if not l:
            continue

        # headings markdown: #, ##, ### ...
        if re.match(r"^\s*#{1,6}\s+", l):
            continue

        # label brevi che finiscono con ":" (es. "Events:", "Event list:")
        # (limite parole per non eliminare frasi normali con ':')
        if l.endswith(":") and len(l.split()) <= 8:
            continue

        keep.append(line)

    return "\n".join(keep).strip()

def count_events_smart(text: str, model_name: str, model_rules: dict) -> int:
    return len(split_events_smart(text, model_name, model_rules))


def mean_sentences_per_event_smart(text: str, model_name: str, model_rules: dict) -> float:
    segs = split_events_smart(text, model_name, model_rules)
    if not segs:
        return 0.0

    cleaned = [strip_non_sentence_lines(strip_event_prefix(seg)) for seg in segs]
    cleaned = [c for c in cleaned if c]
    if not cleaned:
        return 0.0

    sents = [count_sentences(seg) for seg in cleaned]
    return (sum(sents) / len(sents)) if sents else 0.0


# ======================
# BROKEN SENTENCES
# ======================

def is_broken_sentence(sentence: str) -> bool:
    if not sentence:
        return False
    s = sentence.strip()
    if not s:
        return False

    # se sembra un titolo/label, non lo considerare frase broken
    if is_heading_like_line(s):
        return False

    s = re.sub(r'[\s"\')\]\}»]+$', "", s)
    if not s:
        return False

    return s[-1] not in END_PUNCT

def is_noise_sentence(s: str) -> bool:
    """
    True se la 'frase' è solo rumore (bullet, simboli, spazi).
    Esempi: '*', '-', '•', '—', '...' ecc.
    """
    if not s:
        return True
    t = s.strip()
    if not t:
        return True

    # Se contiene almeno una lettera o un numero, NON è rumore
    if re.search(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]", t):
        return False

    # Se è composto solo da simboli/punteggiatura
    if re.fullmatch(r"[\W_]+", t):
        return True

    return False
    
def count_broken_sentences_smart(
    text: str,
    model_name: str,
    model_rules: dict,
    debug: bool = False,
    title: str | None = None,
    max_to_print: int = 50,
):
    """
    Conta quante frasi (dentro gli eventi) sono "broken".
    Se debug=True, stampa le frasi broken con contesto.
    Ritorna: (broken_count, total_sentences, broken_ratio)
    """
    segs = split_events_smart(text, model_name, model_rules)
    if not segs:
        return 0, 0, 0.0

    cleaned_events = [strip_event_prefix(seg) for seg in segs]
    cleaned_events = [c for c in cleaned_events if c]
    if not cleaned_events:
        return 0, 0, 0.0

    total = 0
    broken = 0
    printed = 0

    for ev_i, ev in enumerate(cleaned_events, start=1):
        ev2 = strip_non_sentence_lines(ev)
        if not ev2:
            continue
        
        # >>> aggiungi questa riga
        ev2 = normalize_for_sentence_split(ev2)

        # Se dopo la normalizzazione rimangono righe heading, eliminale
        lines = []
        for line in ev2.split("\n"):
            if is_heading_like_line(line):
                continue
            lines.append(line)
        ev2 = "\n".join(lines).strip()

        if not ev2:
            continue        
        
        sentences = nltk.sent_tokenize(ev2)
        for sent_i, sent in enumerate(sentences, start=1):
            s = sent.strip()
            if not s:
                continue
            
            if is_noise_sentence(s):
                continue

            total += 1
            is_broken = is_broken_sentence(s)
            if is_broken:
                broken += 1

                if debug and printed < max_to_print:
                    # stampa contesto + la frase
                    tlabel = title if title is not None else ""
                    print(
                        f"[BROKEN] model={model_name} title={tlabel} "
                        f"event={ev_i} sent={sent_i} :: {s}"
                    )
                    printed += 1

    ratio = (broken / total) if total else 0.0
    return broken, total, ratio

def normalize_for_sentence_split(text: str) -> str:
    """
    Normalizza casi frequenti che causano falsi 'broken':
    - 'Title:In...' -> 'Title:\nIn...'
    - '... .Title:' -> '...\nTitle:'
    - spazio dopo ':' quando manca e segue una maiuscola
    """
    if not text:
        return ""

    s = text.replace("\r\n", "\n").replace("\r", "\n")

    # 1) Se hai ':In' / ':The' / ':Questo' ecc. metti separazione netta
    #    (newline è meglio di spazio perché ti aiuta anche a rimuovere i titoli)
    s = re.sub(r":(?=[A-ZÀ-ÖØ-Þ])", ":\n", s)

    # 2) Se dopo un punto parte un "titolo:" attaccato, spezza riga
    #    es: 'cattle.Cooperative Structure & Challenges:' -> 'cattle.\nCooperative...:'
    s = re.sub(r"(?<=[\.\!\?])(?=[A-ZÀ-ÖØ-Þ][^:\n]{0,80}:)", "\n", s)

    # 3) Normalizza un po' gli spazi
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def is_heading_like_line(line: str) -> bool:
    """
    Riconosce righe che sembrano intestazioni/label, incluse quelle tipo:
    'Revival Efforts:' / 'Cooperative Structure & Challenges:'.
    """
    if not line:
        return False
    l = line.strip()

    # heading markdown
    if re.match(r"^\s*#{1,6}\s+", l):
        return True

    # rimuovi **bold** wrapper se presente
    l2 = re.sub(r"^\s*(\*\*|__)\s*", "", l)
    l2 = re.sub(r"\s*(\*\*|__)\s*$", "", l2).strip()

    # label breve che finisce con ':'
    if l2.endswith(":") and len(l2.split()) <= 12:
        return True

    return False

# ======================
# CARICAMENTO DATI
# ======================

with open(NARRATIVES_DICT_FILE, "r", encoding="utf-8") as f:
    narratives_dict = json.load(f)

with open(METADATA_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)

title2orig = dict(zip(metadata.get("narrative_titles", []), metadata.get("narratives", [])))

# ======================
# INFERENZA REGOLE PER-MODELLO
# ======================

MODEL_EVENT_RULES = {}
for model in narratives_dict.keys():
    if model in MANUAL_MODEL_EVENT_RULES:
        MODEL_EVENT_RULES[model] = MANUAL_MODEL_EVENT_RULES[model]
    else:
        MODEL_EVENT_RULES[model] = infer_model_event_rules(
            narratives_dict.get(model, {}),
            sample_per_title=INFER_SAMPLE_PER_TITLE,
        )

# ======================
# CALCOLO METRICHE (best-of su prompt_0)
# ======================

avg_jaccard_per_model = {}
avg_events_per_model = {}
avg_sents_per_event_per_model = {}
avg_broken_sents_per_model = {}
avg_num_responses_prompt0_per_model = {}

# Dettaglio per titolo (per stampa FULL)
avg_jaccard_per_model_title = {}
avg_events_per_model_title = {}
avg_sents_per_event_per_model_title = {}
avg_broken_sents_per_model_title = {}
avg_broken_ratio_per_model_title = {}
avg_failed_per_model_title = {}

for model, titles_dict in narratives_dict.items():
    best_jaccards = []
    best_events = []
    best_sents_per_event = []
    best_broken_sents = []
    best_failed = []
    num_responses = []

    avg_jaccard_per_model_title[model] = {}
    avg_events_per_model_title[model] = {}
    avg_sents_per_event_per_model_title[model] = {}
    avg_broken_sents_per_model_title[model] = {}
    avg_broken_ratio_per_model_title[model] = {}
    avg_failed_per_model_title[model] = {}

    for title, prompts_dict in (titles_dict or {}).items():
        original = title2orig.get(title, "")
        if not original:
            continue

        iters_dict = prompts_dict.get("prompt_0", {})
        if not isinstance(iters_dict, dict) or not iters_dict:
            continue

        num_responses.append(len(iters_dict))

        # ---- al posto del best-of: valuta TUTTE le iterazioni ----
        for iter_key, gen in iters_dict.items():
            # Jaccard su questa iterazione
            jac = jaccard_similarity(gen, original)
        
            # metriche su questa iterazione
            n_events = count_events_smart(gen, model, MODEL_EVENT_RULES)
            mse = mean_sentences_per_event_smart(gen, model, MODEL_EVENT_RULES)
        
            do_debug = (
                DEBUG_PRINT_BROKEN
                and (DEBUG_ONLY_MODEL is None or model == DEBUG_ONLY_MODEL)
                and (DEBUG_ONLY_TITLE is None or title == DEBUG_ONLY_TITLE)
            )
        
            broken_cnt, total_sents, broken_ratio = count_broken_sentences_smart(
                gen,
                model,
                MODEL_EVENT_RULES,
                debug=do_debug,
                title=title,
                max_to_print=DEBUG_MAX_BROKEN_TO_PRINT,
            )
        
            failed = 1 if ("\n" not in (gen or "")) else 0
        
            # accumula per modello (ora è su TUTTI gli output)
            best_jaccards.append(jac)
            best_events.append(n_events)
            best_sents_per_event.append(mse)
            best_broken_sents.append(broken_cnt)
            best_failed.append(failed)
        
            # se vuoi definire "coherence" come broken_ratio medio o altro,
            # qui puoi accumularlo in una lista separata

    if not best_jaccards:
        continue

    avg_jaccard_per_model[model] = sum(best_jaccards) / len(best_jaccards)
    avg_events_per_model[model] = sum(best_events) / len(best_events) if best_events else float("nan")
    avg_sents_per_event_per_model[model] = (
        sum(best_sents_per_event) / len(best_sents_per_event)
        if best_sents_per_event else float("nan")
    )
    avg_broken_sents_per_model[model] = (
        sum(best_broken_sents) / len(best_broken_sents)
        if best_broken_sents else float("nan")
    )
    avg_num_responses_prompt0_per_model[model] = (
        sum(num_responses) / len(num_responses)
        if num_responses else float("nan")
    )

# ======================
# STAMPA FULL (con dettaglio per narrazione)
# ======================

if PRINT_FULL_WITH_DETAILS:
    print("\n=== RISULTATI PER MODELLO (FULL) ===")

    for model in sorted(avg_jaccard_per_model.keys()):
        j_model = avg_jaccard_per_model.get(model, float("nan"))
        ev_model = avg_events_per_model.get(model, float("nan"))
        sp_model = avg_sents_per_event_per_model.get(model, float("nan"))
        br_model = avg_broken_sents_per_model.get(model, float("nan"))
        coh_model = avg_num_responses_prompt0_per_model.get(model, float("nan"))

        print(f"\nModello: {model}")
        print(f"  Jaccard (avg) globale [best-of]: {j_model:.4f}")
        print(f"  # eventi (avg) globale [best-of]: {ev_model:.2f}")
        print(f"  Frasi per evento (avg) globale [best-of]: {sp_model:.2f}")
        print(f"  Broken sentences (avg) globale [best-of]: {br_model:.2f}")
        print(f"  Coerenza (avg #risposte in prompt_0): {coh_model:.2f}")

        print("  Dettaglio per narrazione:")
        titles = sorted(avg_jaccard_per_model_title[model].keys())
        for title in titles:
            j_t = avg_jaccard_per_model_title[model].get(title, float("nan"))
            e_t = avg_events_per_model_title[model].get(title, float("nan"))
            sp_t = avg_sents_per_event_per_model_title[model].get(title, float("nan"))
            br_t = avg_broken_sents_per_model_title[model].get(title, float("nan"))
            br_r = avg_broken_ratio_per_model_title[model].get(title, float("nan"))
            f_t = avg_failed_per_model_title[model].get(title, float("nan"))
            f_print = int(f_t) if f_t == f_t else f_t

            print(
                f"    - {title}: "
                f"Jaccard(best) = {j_t:.4f}, "
                f"#eventi(best) = {e_t:.2f}, "
                f"frasi/evento(best) = {sp_t:.2f}, "
                f"broken_sents(best) = {int(br_t) if br_t==br_t else br_t}, "
                f"broken_ratio(best) = {br_r:.2f}, "
                f"failed(best) = {f_print}"
            )

# ======================
# STAMPA SOLO TABELLA MODELLO (medie globali)
# ======================

if PRINT_MODEL_TABLE_ONLY:
    rows = []
    for model in avg_jaccard_per_model.keys():
        rows.append((
            model,
            avg_jaccard_per_model.get(model, float("nan")),
            avg_events_per_model.get(model, float("nan")),
            avg_sents_per_event_per_model.get(model, float("nan")),
            avg_broken_sents_per_model.get(model, float("nan")),
            avg_num_responses_prompt0_per_model.get(model, float("nan")),
        ))

    # ordina per Jaccard desc (cambia qui se vuoi)
    rows.sort(key=lambda x: x[1], reverse=True)

    header = (
        f"{'Model':<35} | "
        f"{'Jaccard_avg':>10} | "
        f"{'Events_avg':>10} | "
        f"{'Sents/Event':>11} | "
        f"{'Broken_avg':>10} | "
        f"{'Coherence':>10}"
    )
    sep = "-" * len(header)


    print("\n=== TABELLA RIASSUNTIVA PER MODELLO ===")
    print(sep)
    print(header)
    print(sep)

    for model, j, ev, sp, br, coh in rows:
        print(
            f"{model:<35} | "
            f"{j:10.4f} | "
            f"{ev:10.2f} | "
            f"{sp:11.2f} | "
            f"{br:10.2f} | "
            f"{coh:10.2f}"
        )

    print(sep)