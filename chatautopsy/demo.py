"""
ChatAutopsy — chat sintetiche per demo e test.

Genera export WhatsApp realistici ma interamente inventati (nessun dato reale):
una coppia con ~500 giorni di storia (affetto, litigi, raffreddamento finale) e un piccolo gruppo.
"""
from __future__ import annotations

import datetime as dt
import random
from typing import List, Tuple

COUPLE = ("Giulia", "Marco")
# Parametri della chat mostrata nel README e aperta da `chatautopsy demo` (stessi numeri ovunque)
SHOWCASE = {"days": 900, "seed": 7}

_POOL_A = ["amore come stai?", "ti amo ❤️", "ahahah sei scemo 😂", "stasera pizza?", "mi manchi tanto 🥰", "ok",
           "fai come vuoi", "scusami, ho sbagliato", "sei sempre il solito, non mi ascolti mai!!", "dove sei??",
           "Ci vediamo dopo al bar Sport", "buongiorno tesoro ☀️", "com'è andata al lavoro?", "che film guardiamo?",
           "cucciolo mio 🥺", "hai mangiato?", "ti penso 💕", "domenica dai miei?", "che freddo oggi 🥶",
           "guarda questo 😂", "sono stanchissima", "buonanotte amore mio 😘", "mi fai ridere troppo 🤣"]
_POOL_B = ["tutto bene tesoro", "sì dai", "ahah", "ok", "va bene", "sto arrivando", "colpa tua, esageri come sempre",
           "te lo sei sognata", "boh", "Noi due insieme siamo forti", "anche tu mi manchi", "domani ti chiamo",
           "giornata pesante oggi", "facciamo sushi?", "piccola mia ❤️", "esco ora dall'ufficio", "partita stasera ⚽",
           "ti va un weekend al mare?", "ahahah 😂", "buonanotte piccola 😴", "sono in riunione, dopo ti scrivo"]
_HONEYMOON = ["non vedo l'ora di vederti 😍", "sei la cosa più bella che mi sia capitata ❤️", "mi manchi già 🥰",
              "amore mio 💕", "ti amo da impazzire ❤️❤️", "sogni d'oro piccola 😘"]
_FIGHT_A = ["non mi ascolti mai!!", "sei sempre il solito", "è colpa tua, non mia", "basta, sono stufa?!",
            "fai come vuoi", "ma ti rendi conto??", "lascia stare"]
_FIGHT_B = ["te lo sei sognata", "esageri come sempre", "ok", "non è vero", "colpa tua", "va bene, come vuoi", "👍"]
_PEACE = ["scusami amore, ho esagerato", "facciamo pace? 🥺", "mi dispiace per prima", "ti chiedo scusa ❤️"]
_LONG = ("Lunghissimo messaggio emotivo perché sento che le cose non vanno come prima e vorrei che tu "
         "mi ascoltassi davvero, ") * 6
# (testo iOS, tipo) — su Android WhatsApp scrive sempre "<Media omessi>"
_MEDIA_IOS = ["\u200eimmagine omessa", "\u200eaudio omesso", "\u200esticker omesso", "\u200evideo omesso", "\u200eGIF omessa"]

# --- Versione inglese (README internazionale, `chatautopsy demo --lang en`) ---
COUPLE_EN = ("Emma", "Jake")
SHOWCASE_EN = {"days": 900, "seed": 7, "lang": "en"}
_EN = {
    "pool_a": ["babe how are you?", "i love you ❤️", "hahaha you're so dumb 😂", "pizza tonight?", "i miss you so much 🥰", "ok",
               "do what you want", "sorry, my bad", "you always do this, you never listen!!", "where are you??",
               "see you later at the coffee shop", "good morning honey ☀️", "how was work?", "which movie tonight?",
               "my sweetie 🥺", "did you eat?", "thinking of you 💕", "dinner at my parents sunday?", "so cold today 🥶",
               "look at this 😂", "i'm exhausted", "good night my love 😘", "you make me laugh so much 🤣"],
    "pool_b": ["all good babe", "sure", "haha", "ok", "fine", "on my way", "your fault, you always overreact",
               "that never happened", "dunno", "we're a great team", "miss you too", "i'll call you tomorrow",
               "rough day today", "sushi?", "my love ❤️", "leaving the office now", "game tonight ⚽",
               "beach weekend?", "hahaha 😂", "night sweetheart 😴", "in a meeting, text you later"],
    "honeymoon": ["can't wait to see you 😍", "you're the best thing that ever happened to me ❤️", "missing you already 🥰",
                  "my love 💕", "i love you like crazy ❤️❤️", "sweet dreams babe 😘"],
    "fight_a": ["you never listen!!", "you always do this", "it's your fault, not mine", "i'm done, seriously?!",
                "do what you want", "are you kidding me??", "forget it"],
    "fight_b": ["that never happened", "you're overreacting again", "ok", "not true", "your fault", "fine, whatever", "👍"],
    "peace": ["i'm sorry babe, i overreacted", "can we make up? 🥺", "sorry about earlier", "i apologize ❤️"],
    "peace_reply": "i'm sorry too ❤️",
    "long": ("This is a really long emotional message because i feel like things aren't the way they used to be and "
             "i really need you to listen to me, ") * 5,
    "media": ["\u200eimage omitted", "\u200eaudio omitted", "\u200esticker omitted", "\u200evideo omitted", "\u200eGIF omitted"],
    "android_media": "<Media omitted>",
    "second_line": "second line of a multi-line message",
    "encrypted": "\u200eMessages and calls are end-to-end encrypted. No one outside of this chat can read them.",
}
_IT = {"pool_a": _POOL_A, "pool_b": _POOL_B, "honeymoon": _HONEYMOON, "fight_a": _FIGHT_A, "fight_b": _FIGHT_B,
       "peace": _PEACE, "peace_reply": "anche io scusa ❤️", "long": _LONG, "media": _MEDIA_IOS, "android_media": "<Media omessi>", "second_line": "seconda riga del messaggio multilinea",
       "encrypted": "\u200eI messaggi e le chiamate sono crittografati end-to-end. Nessuno al di fuori di questa chat può leggerli."}


def couple_chat(days: int = 500, seed: int = 1, android: bool = False, lang: str = "it") -> str:
    """Export WhatsApp (iOS di default, Android con android=True) di una coppia inventata.

    La storia ha fasi: luna di miele (primi ~3 mesi), quotidianità, litigi periodici seguiti da silenzio e
    riappacificazione, raffreddamento nell'ultimo quarto (Marco risponde a monosillabi, Giulia più lenta).
    Tutte le estrazioni casuali sono indipendenti dalla piattaforma: iOS e Android sono la stessa conversazione.
    """
    rnd = random.Random(seed)
    a, b = COUPLE_EN if lang == "en" else COUPLE
    P = _EN if lang == "en" else _IT
    lines: List[str] = []
    if not android:
        lines.append(f"[10/01/22, 08:59:00] {a}: " + P["encrypted"])

    def emit(t: dt.datetime, who: str, txt: str, media_idx: int = -1, multiline: bool = False) -> None:
        if android:
            lines.append(t.strftime("%d/%m/%y, %H:%M") + f" - {who}: " + (P["android_media"] if media_idx >= 0 else txt))
        else:
            stamp = t.strftime("%d/%m/%y, %H:%M:%S")
            lines.append(f"[{stamp}] {who}: " + (P["media"][media_idx] if media_idx >= 0 else txt))
        if multiline and media_idx < 0:
            lines.append(P["second_line"])

    for day in range(days):
        honeymoon = day < 90
        cooling = day > days * 0.76
        fight_day = day > 100 and day % 47 == 0
        t = dt.datetime(2022, 1, 10) + dt.timedelta(days=day, hours=rnd.choice([8, 12, 19, 22, 1]), minutes=rnd.randint(0, 59))
        who = rnd.choice([a, b])
        n = rnd.randint(8, 32) if honeymoon else rnd.randint(3, 25)
        for _ in range(n):
            if rnd.random() < 0.6:
                who = b if who == a else a
            txt = rnd.choice(P["pool_a"] if who == a else P["pool_b"])
            r_phase = rnd.random()
            if honeymoon and r_phase < 0.35:
                txt = rnd.choice(P["honeymoon"])
            if fight_day and r_phase < 0.7:
                txt = rnd.choice(P["fight_a"] if who == a else P["fight_b"])
            if cooling and who == b and r_phase < 0.5:
                txt = "ok"
            if rnd.random() < 0.04:
                txt = P["long"]
            media = rnd.random() < (0.06 if who == a else 0.04)
            media_idx = rnd.randrange(len(P["media"])) if media else -1
            multiline = rnd.random() < 0.05
            emit(t, who, txt, media_idx, multiline)
            late = cooling and who == a
            t += dt.timedelta(seconds=rnd.choice([600, 4000, 16000]) if late else
                              rnd.choice([10, 20, 45] if honeymoon else [20, 60, 200, 600, 1500]))
        if fight_day:
            # silenzio di qualche ora, poi qualcuno fa pace (più spesso Giulia)
            t += dt.timedelta(hours=rnd.choice([5, 7, 10]))
            peacemaker = a if rnd.random() < 0.65 else b
            emit(t, peacemaker, rnd.choice(P["peace"]))
            emit(t + dt.timedelta(minutes=2), b if peacemaker == a else a, P["peace_reply"])
    return "\n".join(lines)


def whatsapp_lines(lines: List[Tuple[float, str, str]], start: dt.datetime = dt.datetime(2024, 3, 1, 18, 0)) -> str:
    """[(minuti dall'inizio, mittente, testo)] -> export WhatsApp iOS."""
    return "\n".join(f"[{(start + dt.timedelta(minutes=m)).strftime('%d/%m/%y, %H:%M:%S')}] {s}: {x}" for m, s, x in lines)


def group_chat() -> str:
    long_msg = ("Ti devo dire una cosa importante perché mi sento davvero ferita da come sono andate le cose "
                "ultimamente e vorrei che tu capissi ") * 4
    lines = [(0, "Anna", "ciao a tutti"), (1, "Bea", "ciao!"), (2, "Carlo", "ehi"),
             (3, "Carlo", "chi viene stasera?"), (4, "Anna", long_msg), (5, "Bea", "ok"),
             (130, "Anna", "senti"), (131, "Bea", "dimmi"), (132, "Anna", long_msg), (133, "Bea", "ok"),
             (300, "Dario", "buongiorno"), (301, "Anna", "ciao"), (302, "Bea", "ciao"), (303, "Anna", "ciao"),
             (304, "Bea", "ciao"), (305, "Anna", "ciao"), (306, "Bea", "ciao"), (307, "Anna", "ciao"),
             (308, "Anna", long_msg), (309, "Bea", "k")]
    for day in range(1, 60):
        for i, who in enumerate(["Anna", "Bea", "Carlo", "Dario", "Anna", "Carlo"]):
            lines.append((day * 1440 + i * 3, who, random.Random(day * 7 + i).choice(
                ["ahah", "ci sto!", "domani?", "io porto le birre 🍺", "che noia", "foto bellissime ❤️", "ok"])))
    return whatsapp_lines(lines)


def telegram_html_page(messages: List[Tuple[str, str, str]], title: str = "Demo") -> str:
    """Pagina messages.html stile Telegram Desktop da [(data 'dd.mm.yyyy HH:MM:SS', mittente, testo)]."""
    import html as _h

    out = [f'<div class="page_header"><div class="content"><div class="name bold">{_h.escape(title)}</div></div></div>',
           '<div class="history">']
    prev = None
    for i, (stamp, who, text) in enumerate(messages, start=1):
        joined = " joined" if who == prev else ""
        out.append(f'<div class="message default clearfix{joined}" id="message{i}"><div class="body">'
                   f'<div class="pull_right date details" title="{stamp}">x</div>'
                   + ("" if joined else f'<div class="from_name">{_h.escape(who)}</div>')
                   + f'<div class="text">{_h.escape(text)}</div></div></div>')
        prev = who
    out.append("</div>")
    return "\n".join(out)
