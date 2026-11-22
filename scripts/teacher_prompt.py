"""Per-input ("privileged") teacher prompts for klein: describe ONLY what is in this photo, never negations.

Why: generic prompts lightened dark skin (median +35 L*) and erased age; adding rules like "glasses only if present"
injected glasses/grey hair/beards into everyone (results/prompt_ab.json, grids/prompt_ab_skin.jpg).
For synthetic inputs we know the generation attributes (manifest prompt) and we MEASURE skin lightness (L*), so the
teacher prompt names this person's real gender/age/hair/accessories/skin tone. The student never sees text.
"""
import re

TONES = [(38, "deep dark brown"), (48, "dark brown"), (57, "medium brown"), (65, "light brown olive"), (73, "light"), (101, "very fair")]
AGE_WORDS = {"twenties": "young adult", "thirties": "adult in their thirties", "forties": "middle-aged adult in their forties",
             "fifties": "older adult in their fifties", "sixties": "older adult in their sixties", "seventies": "elderly adult in their seventies"}


def tone_word(L):
    if L is None:
        return None
    return next(w for t, w in TONES if L < t)


def parse(manifest_prompt):
    """Extract gender, age decade, hair, extra from a gen_data/gen_pairs face prompt."""
    m = re.search(r"\b(woman|man|person)\b (in their [a-z ]+?) with (.+?), (?:(.+?), )?(facing the camera|head turned|three-quarter|head tilted)",
                  manifest_prompt)
    if not m:
        return None
    gender, age, hair, extra = m.group(1), m.group(2), m.group(3), m.group(4)
    if extra and any(k in extra for k in ("light", "background", "portrait", "photo")):
        extra = None
    dec = next((k for k in AGE_WORDS if k in age), None)
    return {"gender": gender, "age": age.replace("in their ", ""), "age_decade": dec, "hair": hair, "extra": extra}


def person_desc(attrs, L):
    if attrs is None:  # real photo (no generation attributes): only what we can measure, i.e. skin tone (product-feasible)
        tw = tone_word(L)
        return f"a person with {tw} skin" if tw else None
    g = {"woman": "woman", "man": "man", "person": "person"}[attrs["gender"]]
    d = f"a {g} in their {attrs['age']} with {attrs['hair']}"
    if attrs["extra"]:
        d += f", {attrs['extra']}"
    tw = tone_word(L)
    if tw:
        d += f", with {tw} skin"
    return d


def avatar_prompt(desc_style, pdesc):
    if pdesc is None:
        return (f"Redraw the person in this photo as an avatar portrait, {desc_style}. Keep the same face shape, skin tone, age, "
                f"gender, hairstyle and expression. Head and shoulders, centered, plain background.")
    return (f"Redraw the person in this photo as an avatar portrait, {desc_style}. The person is {pdesc}; the avatar shows "
            f"exactly this person with the same skin tone, age, gender and hairstyle. Head and shoulders, centered, plain background.")


def sticker_prompt(desc_style, pdesc):
    if pdesc is None:
        return (f"Redraw the subject in this photo as a die-cut chat sticker, {desc_style}. Keep the identity and key features "
                f"(species or object type, colors, markings). Full subject, thick white outline, isolated on a plain flat white background.")
    return (f"Redraw the person in this photo as a die-cut chat sticker, {desc_style}. The person is {pdesc}; the sticker shows "
            f"exactly this person with the same skin tone, age, gender and hairstyle. Full subject, thick white outline, "
            f"isolated on a plain flat white background.")


ANIMALS = {"cat", "kitten", "dog", "puppy", "retriever", "pug", "corgi", "inu", "husky", "dachshund", "poodle", "rabbit", "hamster",
           "pig", "parrot", "budgie", "goldfish", "turtle", "hedgehog", "ferret", "duckling", "chick", "bird", "fish", "animal"}


def parse_subject(manifest_prompt):
    """'Realistic smartphone photo of a fluffy corgi, ...' -> 'fluffy corgi' (privileged: we generated the input)."""
    m = re.search(r"photo of (?:an?|the) (.+?),", manifest_prompt)
    return m.group(1) if m else None


def is_animal(subj):
    return subj is not None and any(w in ANIMALS for w in re.findall(r"[a-z]+", subj.lower()))


def subject_sticker_prompt(subj_desc, subj, variant="named"):
    """Non-person sticker. `subj_desc` must not contain face/eye/hair/skin words (style.yaml subject_description).
    variant 'named': names the subject and its kind; animals keep animal anatomy, objects stay inanimate objects."""
    if subj is None:
        return (f"Redraw the subject in this photo as a die-cut chat sticker, {subj_desc}. Keep the species or object type, colors, "
                f"shape and markings. Full subject, thick white outline, isolated on a plain flat white background.")
    kind = (f"a cartoon {subj} with natural {subj} anatomy, the same fur or feather colors and markings" if is_animal(subj)
            else f"a cartoon still-life of the {subj} as an inanimate object, with its real shape, colors and details")
    return (f"Redraw the {subj} in this photo as a die-cut chat sticker, {subj_desc}. The sticker shows {kind}, exactly as in the "
            f"photo. Full subject, thick white outline, isolated on a plain flat white background.")


def expr_sticker_prompt(desc_style, pdesc, phrase, age=None):
    """Sticker-set expression/action sticker (configs/sticker_expressions.yaml). Head and upper body with adult proportions:
    the pilot (results/expr_pilot.json) showed full-body framings shrink the person into a child-like figure and lose identity."""
    who = f"The person is {pdesc}; the sticker shows exactly this person with the same skin tone, age, gender and hairstyle" if pdesc \
        else "The sticker shows exactly this person with the same skin tone, age, gender and hairstyle"
    return (f"Redraw the person in this photo as a die-cut chat sticker, {desc_style}. {who}, with {phrase}. Head and upper body, "
            f"adult proportions, expressive, thick white outline, isolated on a plain flat white background.")


def expr_sticker_prompt_adult(desc_style, pdesc, phrase, age):
    """Round 4 (user request 2026-11-17, grids/age_prompt_pilot.jpg variant B): klein drew older adults as children in gesture
    stickers. Stating the age decade + adult body/hands + 'keep visible signs of age, never a child' doubled SFace cos
    (0.099 -> 0.195 on 16 older-face gesture stickers) and restored grey hair/wrinkles."""
    who = f"The person is {pdesc}; the sticker shows exactly this person with the same skin tone, age, gender and hairstyle" if pdesc \
        else "The sticker shows exactly this person with the same skin tone, age, gender and hairstyle"
    adult = (f"This is an adult in their {age}: realistic adult head-to-body ratio, adult shoulders and adult hands, "
             f"keep the visible signs of age from the photo (wrinkles, grey hair, facial hair); never draw a child or a baby")
    return (f"Redraw the person in this photo as a die-cut chat sticker, {desc_style}. {who}, with {phrase}. {adult}. "
            f"Head and upper body, expressive, thick white outline, isolated on a plain flat white background.")

# ---------------------------------------------------------------------------------------------------------------------
# Data v2 prompts (2026-11-18): avatar v4, sticker/expression v5. Developed with scripts/teacher_consistency.py (README
# "Teacher prompt v2 -> v4" and "v4 sticker / expression prompts"). `cap` = local Qwen3-VL attributes (caption_photos.py),
# `attrs` = generator attributes from parse() (synthetic photos only), `tone` = tone_word(L*) (used only with attrs: the
# L*-based word mislabels real-photo lighting, the VLM skin word is used for real photos).
NONE_WORDS = ("", "none", "no", "not visible", "n/a", "unknown")
OLDER = ("forties", "fifties", "sixties", "seventies")


def v4_desc(cap, tone, attrs):
    """-> (person description, ', wearing a <clothing>' or '', age phrase or ''). Never names an absent item."""
    cap = cap or {}
    g = cap.get("gender") if cap.get("gender") in ("man", "woman") else (attrs or {}).get("gender", "person")
    if attrs:
        age = attrs["age"]
        d = f"a {g} in their {age} with {attrs['hair']}" + (f", {attrs['extra']}" if attrs.get("extra") else "")
    else:
        age = cap.get("age") or ""
        d = f"a {g}" + (f" in their {age}" if age else "")
        hair = "a bald or shaved head" if (cap.get("bald") or "").lower().startswith("y") else (cap.get("hair") or "")
        if hair and hair.lower() not in NONE_WORDS:
            d += f" with {hair}"
    for k in ("glasses", "facial_hair"):
        v = (cap.get(k) or "").strip()
        if v.lower() not in NONE_WORDS:
            d += f", with {v}"
    if not attrs:
        hw = (cap.get("headwear") or "").strip()
        if hw.lower() not in NONE_WORDS:
            d += f", wearing a {hw} exactly where it is in the photo"
    sk = tone if attrs else (cap.get("skin_tone") or "").strip()
    if sk and sk.lower() not in NONE_WORDS:
        d += f", with {sk} skin"
    cl = (cap.get("clothing") or "").strip()
    return d, (f", wearing a {cl}" if cl.lower() not in NONE_WORDS else ""), age


def avatar_prompt_v4(desc_style, cap, tone, attrs):
    d, wear, _ = v4_desc(cap, tone, attrs)
    return (f"Redraw the person in this photo as an avatar portrait, {desc_style}. The person is {d}{wear}. "
            f"Keep the face, skin tone, age, gender, expression and head pose exactly as in the photo, the same hairstyle, "
            f"hair length and hair colour as in the photo, and the same clothing in the same colors. Keep the photo's framing, "
            f"head size and head position. Head and shoulders on a plain light grey background, nothing else in the background.")


def sticker_prompt_v5(desc_style, cap, tone, attrs, phrase=None):
    """Plain sticker (phrase None) or expression sticker (menu phrase)."""
    d, wear, age = v4_desc(cap, tone, attrs)
    if phrase:  # synthetic attributes carry the photo's expression ("smiling broadly"): it would fight e.g. "crying"
        d = ", ".join(x for x in d.split(", ") if not re.search(r"smil|laugh|grin|frown|neutral|serious", x))
    keep = ("The sticker shows exactly this person: keep the face, skin tone, age and gender, the same hairstyle, hair length and "
            "hair colour, and the same clothing in the same colors as in the photo.")
    act = f" Expression and pose: {phrase}." if phrase else " Keep the expression and head pose from the photo."
    cues = " (wrinkles, grey hair)" if any(o in age for o in OLDER) else ""
    adult = (f" This is an adult in their {age}: realistic adult head-to-body ratio, adult shoulders and adult hands, keep the "
             f"visible signs of age from the photo{cues}; never draw a child or a baby.") if age else ""
    return (f"Redraw the person in this photo as a die-cut chat sticker, {desc_style}. The person is {d}{wear}. {keep}{act}{adult} "
            f"Head and upper body in a close crop with a large head, {'expressive, ' if phrase else ''}thick white outline, isolated "
            f"on a plain flat white background.")


# ---------------------------------------------------------------------------------------------------------------------
# v6 (2026-11-20, new chat styles chibi / comic_line; cartoon3d data keeps v4/v5). Preview bug: stickers of some synthetic
# faces got grey hair while the avatar kept dark hair. Cause: the age came from the generator label ("forties") while the
# photo looks younger (VLM caption "late twenties"), and v5 names "(wrinkles, grey hair)" for 40+ -> klein draws them.
# v6: age from the VLM caption when present; never names grey hair / wrinkles ("keep the hair colour and the signs of age
# exactly as in the photo"); the adult-body sentence is optional per style (off for chibi: it fights big-head proportions).
def _caption_age(cap, attrs):
    if attrs and cap and (cap.get("age") or "").strip().lower() not in NONE_WORDS:
        return dict(attrs, age=cap["age"].strip())
    return attrs


def avatar_prompt_v5(desc_style, cap, tone, attrs):
    return avatar_prompt_v4(desc_style, cap, tone, _caption_age(cap, attrs))


def sticker_prompt_v6(desc_style, cap, tone, attrs, phrase=None, adult_body=True):
    d, wear, age = v4_desc(cap, tone, _caption_age(cap, attrs))
    if phrase:
        d = ", ".join(x for x in d.split(", ") if not re.search(r"smil|laugh|grin|frown|neutral|serious", x))
    keep = ("The sticker shows exactly this person: keep the face, skin tone, age and gender, the same hairstyle, hair length and "
            "hair colour, and the same clothing in the same colors as in the photo.")
    act = f" Expression and pose: {phrase}." if phrase else " Keep the expression and head pose from the photo."
    body = "realistic adult head-to-body ratio, adult shoulders and adult hands, " if adult_body else ""
    adult = (f" This is an adult in their {age}: {body}keep the hair colour and the signs of age exactly as in the photo; "
             f"never draw a child or a baby.") if age else ""
    return (f"Redraw the person in this photo as a die-cut chat sticker, {desc_style}. The person is {d}{wear}. {keep}{act}{adult} "
            f"Head and upper body in a close crop with a large head, {'expressive, ' if phrase else ''}thick white outline, isolated "
            f"on a plain flat white background.")
