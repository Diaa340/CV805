"""Nonce-word CLEVR (paper/preregistration_nonce.md): every task renames the CLEVR attribute words with its own made-up
words (disjoint across tasks). Answers, tables and exception labels are unchanged; only question text changes."""
import re
import numpy as np
ENGLISH = {w.strip().lower() for f in ("/usr/share/dict/american-english", "/usr/share/dict/british-english") for w in open(f, errors="ignore")}
CONCEPTS = {"gray": ["gray", "grey"], "red": ["red"], "blue": ["blue"], "green": ["green"], "brown": ["brown"], "purple": ["purple"], "cyan": ["cyan"],
            "yellow": ["yellow"], "cube": ["cube", "block"], "sphere": ["sphere", "ball"], "cylinder": ["cylinder"], "metal": ["metal", "metallic", "shiny"],
            "rubber": ["rubber", "matte"], "large": ["large", "big"], "small": ["small", "tiny"]}
NOUNS = {"cube", "sphere", "cylinder"}
BLOCK = {"fuk", "fak", "fuc", "kok", "nig", "neg", "nazi", "sek", "sekz"}   # look-alikes of offensive words


def vocab(n_tasks=6, seed=70):
    """Pronounceable CVCV / CVC words, not in the system English word lists (nor their plural), disjoint across tasks."""
    rng = np.random.default_rng(seed); C, V = list("bdfgklmnprstvz"), list("aeiou"); used, out = set(), []
    for _ in range(n_tasks):
        m = {}
        for c in CONCEPTS:
            while True:
                w = "".join(rng.choice(C)) + "".join(rng.choice(V)) + "".join(rng.choice(C)) + ("".join(rng.choice(V)) if rng.random() < 0.5 else "")
                if w not in used and w not in BLOCK and w not in ENGLISH and w + "s" not in ENGLISH and len(w) >= 3:
                    used.add(w); m[c] = w; break
        out.append(m)
    return out


def rename(text, m):
    """Replace every attribute word (and plural nouns) with the task's nonce word; keeps capitalisation of the first letter."""
    pairs = sorted([(s, m[c]) for c, syn in CONCEPTS.items() for s in syn], key=lambda p: -len(p[0]))
    def sub(mo):
        w = mo.group(0); low = w.lower(); plural = False
        base = low[:-1] if low.endswith("s") and low[:-1] in {s for c in NOUNS for s in CONCEPTS[c]} else low
        plural = base != low; new = dict(pairs)[base] + ("s" if plural else "")
        return new.capitalize() if w[0].isupper() else new
    words = "|".join(re.escape(s) + ("s?" if any(s in CONCEPTS[c] for c in NOUNS) else "") for s, _ in pairs)
    return re.sub(rf"\b({words})\b", sub, text, flags=re.IGNORECASE)
