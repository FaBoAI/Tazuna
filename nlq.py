"""Rule-based reading of simple pick-and-place requests (Japanese / English).

Purpose: routine requests such as 「青いキューブを箱に入れて」 or "put the white cap into the box"
do not need an LLM at all. This module turns them into the query dictionaries of `fd pick`, and
take-out requests such as 「箱の中の白いキャップを取り出して」 or "take the white cap out of the
box" into those of `fd take` (result["action"] says which: "put_in" or "take_out").
Anything it cannot read with confidence is reported as such, so the caller (the LLM operator or
a person) can decide; it never guesses.

Only a PLAIN POSITIVE COMMAND is read. The rules find which object and which container a sentence
names; they do not understand the sentence. A prohibition (「取り出さないで」, "don't take ..."), an
exclusion (「白いキャップ以外」), a condition (「〜たら」), a question, a statement (「取り出した」) and a
request with a second part (「取り出してから箱に戻して」) name the same object and container as the
plain command does. So such sentences are refused before anything is read (result["not_plain"] says
what was found), and a sentence is accepted only when everything outside the object description is
one of the fixed command frames (「〜を〜に入れて」, 「〜から〜を取り出して(机に置いて)」, "put ... into
the ...", "take ... out of the ... (and put it on the table)"). A refused sentence goes to the
operator (`fd do` returns "need_llm"); nothing moves.

Run this file to execute its self-test.
"""

from __future__ import annotations

import re
import unicodedata

COLORS = {
    "red": ("赤", "あか", "レッド", "red"),
    "blue": ("青", "あお", "ブルー", "blue"),
    "green": ("緑", "みどり", "グリーン", "green"),
    "yellow": ("黄", "きいろ", "イエロー", "yellow"),
    "white": ("白", "しろ", "ホワイト", "white"),
    "black": ("黒", "くろ", "ブラック", "black"),
    "ivory": ("アイボリー", "クリーム", "ベージュ", "象牙", "生成り", "ivory", "cream", "beige"),
    "orange": ("オレンジ", "橙", "orange"),
    "purple": ("紫", "むらさき", "パープル", "purple", "violet"),
    "pink": ("ピンク", "桃色", "pink"),
    "gray": ("グレー", "灰色", "ねずみ色", "gray", "grey"),
}
KINDS = {
    "box": ("キューブ", "ブロック", "立方体", "直方体", "積み木", "つみき", "レゴ", "サイコロ", "cube", "block", "brick"),
    "cylinder": ("円柱", "シリンダー", "キャップ", "ふた", "蓋", "フタ", "筒", "コイン", "コップ", "カップ", "cylinder", "cap", "lid",
                 "cup"),
}
# words for things that are hollow and normally lie open side up: when such a thing is taken out of a
# container its wall may be pinched (take-out requests report "hollow": True)
HOLLOW_WORDS = ("キャップ", "ふた", "蓋", "フタ", "コップ", "カップ", "cap", "lid", "cup")
# as whole words where they are English ("solid" is not a lid, a "capsule" is not a cap); and the counting
# word 「ふたつ」 (two), 「ふたり」, 「ふたたび」 are not a 「ふた」 (lid)
_HOLLOW_RE = re.compile("|".join(rf"(?<![a-z]){w}s?(?![a-z])" if w.isascii() else
                                 re.escape(w) + ("(?![つりた])" if w == "ふた" else "") for w in HOLLOW_WORDS),
                        re.IGNORECASE)
CONTAINERS = ("箱", "はこ", "ボックス", "容器", "トレー", "トレイ", "ケース", "入れ物", "かご", "カゴ",
              "box", "bin", "tray", "container", "case", "basket")
ALL_WORDS = ("全部", "ぜんぶ", "すべて", "全て", "みんな", "残り", "all ", "every", "everything")
PUT_WORDS = ("入れ", "いれ", "置", "移", "運", "しまっ", "しまう", "片付", "かたづ", "put", "place", "move", "drop",
             "into", "in the")
_CONT_RE = "|".join(re.escape(w) for w in CONTAINERS)
# ---- taking something OUT of a container
OUT_WORDS = ("取り出", "とりだ", "取りだ", "とり出", "出し", "だし", "出す", "だす", "出せ")  # ... 出して
TAKE_WORDS = ("取っ", "とっ", "取る", "取れ")  # 「箱から〜を取って」: only together with a container to take it from
_JA_RE = re.compile(r"[ぁ-んァ-ヶ一-龥]")
# <container>(の中)から ...
_FROM_RE = re.compile(rf"(?:{_CONT_RE})\s*(?:の\s*(?:中|なか)|内)?\s*から", re.IGNORECASE)
# <container>の中の / の中にある / の中に入っている <object>
_INSIDE_RE = re.compile(rf"(?:{_CONT_RE})\s*(?:の\s*(?:中|なか)|内)\s*(?:の|にある|に入っている|に入ってる|にはいっている|"
                        rf"にはいってる|に入った)", re.IGNORECASE)
# ... <another container> に / へ: the object is to go INTO something again
_INTO_RE = re.compile(rf"(?:{_CONT_RE})\s*(?:の\s*(?:中|なか)|内)?\s*(?:に|へ)", re.IGNORECASE)
_TAKE_EN = re.compile(r"\b(?:take|get|pull|pick|lift|remove|bring)\s+(?:out\s+)?(?:all\s+)?(?:the\s+|a\s+|an\s+)?"
                      r"(?P<t>.+?)\s+(?:out\s+of|out\s+from|from\s+inside|from)\s+(?:the\s+)?(?P<c>[^,.;!]+)")
_INTO_EN = re.compile(rf"\b(?:into|in|inside)\s+(?:the\s+|another\s+|a\s+)?(?:\w+\s+)?(?:{_CONT_RE})\b", re.IGNORECASE)


# ============================================================================== only plain positive commands
# 1. Words that make a sentence something else than a plain positive command. Found anywhere, the sentence is
#    refused: (kind, what it means, pattern). The Japanese patterns are tried on the sentence without its
#    closing punctuation, the English ones on its lower-case form.
_JA_MARKS = (
    ("prohibition", "it forbids or stops something", re.compile(
        r"ないで|ないよう|ないこと|[うくぐすつぬぶむる]な(?:よ|ね)?$|ては?いけ|ちゃいけ|じゃいけ|ては?なら|ちゃだめ|ちゃダメ|じゃだめ|じゃダメ|"
        r"だめ|ダメ|駄目|禁止|厳禁|(?<!は)やめ|止め|中止|中断|ストップ|キャンセル|取り消|取消|待って|待て|不要|放置|そのまま|残し|のこし|無視|"
        r"(?<!ま)ずに")),
    ("exclusion", "it excludes something", re.compile(
        r"以外|除[いくきけ]|のぞ[いくきけ]|抜き|ではなく|じゃなく|でなく|代わり|かわりに|ほかの|他の|違[うっい]")),
    ("negation", "it holds a negation", re.compile(r"ない|なく|なかっ|ません|無い|無く|無かっ")),
    ("comparison", "it compares or orders things", re.compile(r"より|先に|順に|順番|の次に")),
    ("condition", "it holds a condition or an order in time", re.compile(
        r"たら|だら|れば|なら|もし|場合|とき(?!どき)|時[にはも、]|際[にはも]|てから|でから|(?<!手)前に|後[でに]|あと[でに]|うちに|間に|まで|次第|"
        r"ながら|限り|かぎり|たびに")),
    ("choice", "it leaves a choice open", re.compile(
        r"または|もしくは|あるいは|どちらか|どれか|いずれか|(?<=[一-龥ァ-ヶー])か(?=[一-龥ァ-ヶ])")),
    ("question", "it asks something", re.compile(
        r"\?|かな$|かしら$|でしょうか|だろうか|ですか|ますか|のか$|誰|だれ|どれ|(?<![ほな])どの|どこ|どっち|どちら|いつ|なぜ|どうして|なんで|"
        r"どうや|どうす|いくつ|何個|何色|何[がをでにの]|なに[がをでにの]")),
)
_EN_MARKS = (
    ("prohibition", "it forbids or stops something", re.compile(
        r"(?<![a-z])(?:do\s*n['’`´]?t|do\s+not|never|not|no|[a-z]+n['’`´]t|cannot|stop|cancel|abort|halt|wait|avoid|leave|"
        r"keep|forbidden|prohibited|ng)(?![a-z])")),
    ("exclusion", "it excludes something", re.compile(
        r"(?<![a-z])(?:except|excluding|exclude|but|other\s+than|apart\s+from|besides|instead|rather\s+than|without|nor|"
        r"neither)(?![a-z])")),
    ("condition", "it holds a condition or an order in time", re.compile(
        r"(?<![a-z])(?:if|unless|when|whenever|once|after|before|until|till|while|as\s+soon\s+as|in\s+case)(?![a-z])")),
    ("negation", "it holds a negation", re.compile(r"(?<![a-z])(?:non|none|nothing|nobody|nowhere|zero)(?![a-z])")),
    ("choice", "it leaves a choice open", re.compile(r"(?<![a-z])(?:or|either)(?![a-z])")),
    ("comparison", "it compares or orders things", re.compile(r"(?<![a-z])than(?![a-z])")),
    ("question", "it asks something", re.compile(
        r"\?|(?<![a-z])(?:who|whom|whose|what|which|where|why|how|whether)(?![a-z])")),
)

# 2. The fixed frames of a plain command. A sentence is read only when everything outside the object
#    description is one of them: an unknown verb, a second instruction, a place the rules do not know
#    (「机の右に」) make it something for the operator.
# what may follow the て form of the verb: 〜てください, 〜ておいて, 〜てしまって, 〜てみて, 〜てね
_AUX = r"(?:おいて|しまって|みて|ください(?:ませ)?|下さい(?:ませ)?|くれ|ちょうだい|頂戴|(?:ほ|欲)しい(?:です)?|ね|よ)"
_PLEASE = r"(?:[、,\s]*お願い(?:します|いたします|致します|ね)?)"
_END = rf"{_AUX}*{_PLEASE}?"
_PUT_TE = r"(?:入れて|いれて|置いて|おいて|移して|うつして|運んで|はこんで|しまって|片付けて|片づけて|かたづけて)"
_PUT_IMP = (r"(?:入れろ|いれろ|入れなさい|いれなさい|置け|置きなさい|移せ|移しなさい|運べ|運びなさい|しまえ|しまいなさい|片付けろ|片づけろ|"
            r"かたづけろ|片付けなさい|片づけなさい|かたづけなさい)")
_OUT_TE = r"(?:取り出して|取りだして|とり出して|とりだして|出して|だして)"
_OUT_IMP = r"(?:取り出せ|取りだせ|とり出せ|とりだせ|出せ|だせ|取り出しなさい|とりだしなさい|出しなさい|だしなさい)"
_TAKE_TE = r"(?:取って|とって)"
_TAKE_IMP = r"(?:取れ|とれ|取りなさい|とりなさい)"
_PICK_TE = r"(?:取って|とって|つかんで|掴んで|持って|もって|拾って|ひろって|持ち上げて|もちあげて|つまんで|摘んで)"  # 〜を取って箱に入れて
_ADV = (r"(?:そっと|ゆっくり|優しく|やさしく|丁寧に|ていねいに|静かに|しずかに|すぐに?|早く|はやく|急いで|いそいで|ちゃんと|しっかり|きちんと|"
        r"一旦|いったん|とりあえず|今すぐ|今|いま|また|もう一度|もう一回|もう1回|もう1度|[0-9一二三四五六七八九十]+\s*(?:個|つ|コ|こ)|"
        r"ひとつ|ふたつ|みっつ|よっつ)")
_QUANT = r"(?:全部|ぜんぶ|すべて|全て|みんな)"
_SEP = r"[、,\s]"
_C = rf"(?:{_CONT_RE})"
_MOD = r"(?:(?![てで])[^をにへ、,。\s])*?"  # what describes a container (黒い / その大きな / 右の): no verb in its て form
_CON_JA = rf"{_MOD}{_C}(?:\s*の\s*(?:中|なか)|内)?"
# <object> を | [pick-up verb] <container> に/へ [put verb]            (「青いキューブを箱へ」 has no verb)
_FRAME_PUT = re.compile(rf"(?:{_ADV}|{_QUANT}|{_PICK_TE}|{_SEP})*{_CON_JA}\s*(?:に|へ)(?:{_ADV}|{_QUANT}|{_SEP})*"
                        rf"(?:{_PUT_TE}{_END}|{_PUT_IMP}よ?)?", re.IGNORECASE)
# <container> | に <object> を | <put verb>
_FRAME_PUT_CON = re.compile(rf"(?:{_ADV}|{_QUANT}|{_SEP})*{_CON_JA}", re.IGNORECASE)
_FRAME_PUT_VERB = re.compile(rf"(?:{_ADV}|{_QUANT}|{_SEP})*(?:{_PUT_TE}{_END}|{_PUT_IMP}よ?)")
# what may stand before the container of a take-out, and what may follow its object: the take verb, with
# "to the outside / onto the table" before it or "... and put it on the table / beside the box" after it
_FRAME_TAKE_PRE = re.compile(rf"(?:{_ADV}|{_SEP})*{_MOD}")
_TABLE = r"(?:机|つくえ|テーブル)(?:\s*の\s*上)?"
_DEST_OUT = rf"(?:(?:{_C}\s*の\s*)?(?:外|そと)|{_TABLE})\s*(?:に|へ)"
_DEST_PUT = rf"(?:(?:{_C}\s*の\s*)?(?:外|そと|横|よこ|隣|となり|そば|わき|脇|近く)|{_TABLE})\s*(?:に|へ)"
_FRAME_TAKE = re.compile(
    rf"[をはも]?(?:{_ADV}|{_SEP})*(?:{_DEST_OUT})?(?:{_ADV}|{_SEP})*"
    rf"(?:(?:{_OUT_TE}|{_TAKE_TE})(?:{_SEP}*{_DEST_PUT}\s*(?:置いて|おいて|出して|だして))?{_END}|(?:{_OUT_IMP}|{_TAKE_IMP})よ?)",
    re.IGNORECASE)
# a request ends in a て form (+ ください ...), an imperative, or - without a verb - in に / へ
_JA_REQ_END = re.compile(rf"(?:[てで]{_END}|(?:{_PUT_IMP}|{_OUT_IMP}|{_TAKE_IMP})よ?|[にへ])$")
# the object description: a container word in it is the object itself (「赤いBox」, at its end) or a landmark
# (「箱の横の」); anything else is a second container. A て form in it is another clause, unless it describes
# the object (「置いてある」)
_CONT_ANY = re.compile(_C, re.IGNORECASE)
_OBJ_CONT_OK = re.compile(rf"{_C}(?:\s*の\s*(?:横|よこ|隣|となり|そば|側|前|まえ|手前|後ろ|うしろ|奥|右|左|近く|ちかく|脇|わき|外|そと)"
                          rf"(?:の|に|で)|\s*$)", re.IGNORECASE)
_OBJ_TE = re.compile(r"[てで](?!ある|あった|あり|いる|いた|る|た|おいた)")
_OBJ_TE_WORDS = re.compile(r"すべて|全て|はじめて|初めて|とても|でかい|での|ので")  # (「邪魔なので〜」 is a reason, not a clause to act on)
_OBJ_STATE = re.compile(r"\s*(?:ある|あった|入って|はいって|入った|はいった|いる|置いてある|残って)")  # 「箱の中に|ある〜」
# 「青いキューブの隣のやつ」: a colour or a shape named BEFORE such a word belongs to a landmark, not to the object
_OBJ_REL = re.compile(r"の\s*(?:隣|となり|横|よこ|そば|側|上|下|前|まえ|手前|後ろ|うしろ|奥|右|左|近く|ちかく|間|あいだ|向こう|むこう|中|なか|"
                      r"内側|外側|脇|わき)")
# English
_EN_LEAD = (r"(?:(?:please|pls|kindly|ok|okay|now|then|next|first|so|hey|hi|hello|robot|arm|fable|lapika|just)(?![a-z])"
            r"[\s,.:!]*)*")
_EN_START = re.compile(rf"{_EN_LEAD}(?:put|place|move|drop|take|get|pull|pick|lift|remove|bring)(?![a-z])")
_EN_ASK = re.compile(rf"{_EN_LEAD}(?:can|could|would|will|shall|should|may|might|do|does|did|is|are|am|was|were|has|have|"
                     rf"had)(?![a-z])")
_EN_FILL = r"(?:please|pls|now|right\s+now|quickly|carefully|gently|slowly|for\s+me|thanks|thank\s+you)"
_EN_TAIL = rf"(?:[\s,]*{_EN_FILL})*[\s.!]*"
_EN_COLORS = "|".join(sorted({w for ws in COLORS.values() for w in ws if w.isascii()}, key=len, reverse=True))
_EN_MOD = (rf"(?:(?:{_EN_COLORS}|big|small|large|little|dark|light|open|empty|same|cardboard|plastic|wooden|paper|storage|"
           r"toy)\s+){0,3}")
_EN_C = r"(?:box|bin|tray|container|case|basket)(?:es|s)?"
_EN_CON = rf"(?:(?:the|a|an|that|this|my|our)\s+)?{_EN_MOD}{_EN_C}"
_EN_FRAME_PUT = re.compile(rf"{_EN_CON}{_EN_TAIL}")
_EN_FRAME_TAKE = re.compile(rf"{_EN_CON}(?:\s+and\s+(?:then\s+)?(?:put|place|set|lay|drop)\s+(?:it|them)\s+(?:down\s+)?"
                            rf"(?:on|onto)\s+(?:the\s+)?(?:table|desk))?{_EN_TAIL}")
_EN_AFTER = re.compile(rf"[\s,.;!]*(?:{_EN_FILL}[\s,.!]*)*")
_EN_OBJ_CONT = re.compile(rf"(?<![a-z]){_EN_C}(?![a-z])")
_EN_OBJ_STOP = re.compile(r"(?<![a-z])(?:next|close|near|adjacent|beside|behind|front|top|on|upon|atop|onto|at|by|with|under|"
                          r"above|below|over|off|out|from|away|against|between|outside|to|into|in|inside|toward|towards|of|"
                          r"me|us|then|put|place|move|drop|take|get|pull|pick|lift|remove|bring|open|shut|push|press|turn|"
                          r"go|park|grab|throw|hold|look|picture|photo|photograph|image|video|care|note|rid)(?![a-z])")
# "the block on the table", "put the cap back": no place and no second instruction
_EN_OBJ_PLAIN = re.compile(r"\s+(?:on\s+the\s+(?:table|desk|floor)|back)(?![a-z])")
# words in front of a request that change nothing: 「すみません、」「まず」「ロボット、」
_LEAD_JA = re.compile(
    r"^(?:(?:(?:すみません|すいません|ごめん(?:なさい)?|悪いん?だ?けど|お手数ですが|恐れ入りますが)(?:が|けど)?|まず(?:は)?|最初に|次に|つぎに|"
    r"今度は|続いて|最後に|じゃあ|それでは|それじゃあ?|えっと|えーと)[、,\s!]*"
    r"|(?:はい|では|じゃ|さあ|さて|よし|ねえ|ねぇ|あのー?|あと|それと|それから|ok|okay|オーケー|オッケー|ロボット(?:くん|さん|君)?|アーム|"
    r"ラピカ)[、,\s!]+)+", re.IGNORECASE)
_QUOTES = "「」『』\"'“”‘’"
# ... and behind it: 「〜してください。お願いします。」, "... . Thanks!" (not 「ありがとう」: 「〜して、ありがとう」 may be
# thanks for something done, not a request)
_TRAIL = re.compile(r"(?:[\s。.!、,]*(?:お願い(?:します|いたします|致します|ね)?|よろしく(?:お願いします)?|頼みます|頼む|"
                    r"(?<![a-z])(?:please|pls|thanks|thank\s+you|thx)))+[\s。.!]*$", re.IGNORECASE)
# spoken short forms of the request, written out: 入れといて = 入れておいて, 入れちゃって = 入れてしまって
_SHORT = ((re.compile(r"といて(?=(?:ください|下さい|ね|よ|くれ)*$)"), "ておいて"),
          (re.compile(r"どいて(?=(?:ください|下さい|ね|よ|くれ)*$)"), "でおいて"),
          (re.compile(r"ちゃって(?=(?:ください|下さい|ね|よ|くれ)*$)"), "てしまって"),
          (re.compile(r"じゃって(?=(?:ください|下さい|ね|よ|くれ)*$)"), "でしまって"))


def _clean(text: str) -> str:
    """The sentence as the rules read it: full-width letters, digits and punctuation as ASCII (？ -> ?),
    without quotation marks around it, without a greeting in front of it or a "please" / "thanks" behind
    it, and with the spoken short forms of a request written out."""
    t = unicodedata.normalize("NFKC", text)
    t = re.sub("[\u200b-\u200d\u2060\ufeff]", "", t).strip().strip(_QUOTES).strip()  # (and no zero-width characters)
    t = _LEAD_JA.sub("", t)
    cut = _TRAIL.search(t)
    if cut is not None and cut.start() > 0:
        t = t[:cut.start()]
    core = _strip_end(t)
    for rx, full in _SHORT:
        core = rx.sub(full, core)
    return core + t[len(_strip_end(t)):]


def _strip_end(s: str) -> str:
    return re.sub(r"[\s。.!…~〜、,]+$", "", s)


def _not_plain(text: str):
    """(kind, reason) when the sentence is something else than a plain positive command (see _JA_MARKS),
    None when it may be one. Nothing is read from a sentence this refuses."""
    low = text.lower()
    core = _strip_end(low)
    japanese = _JA_RE.search(text) is not None
    for marks, where in ((_EN_MARKS, low), (_JA_MARKS if japanese else (), core)):
        for kind, what, rx in marks:
            m = rx.search(where)
            if m:
                return kind, f"not a plain command: {what} (「{m.group(0)}」). Read the request yourself"
    if re.search(r"[\r\n\u2028\u2029]", core):
        return "compound", "not a plain command: more than one line. Read the request yourself"
    if not japanese or _EN_START.match(low):  # an English sentence (or one that starts as an English command)
        if _EN_ASK.match(low):
            return "question", "not a plain command: it starts like a question. Read the request yourself"
        if not _EN_START.match(low):
            return "statement", ("not a plain command: it does not start with a command verb (put / take ...): a "
                                 "statement or something else. Read the request yourself")
        if re.search(r"[.!?;]\s+\S", core):
            return "compound", "not a plain command: more than one sentence. Read the request yourself"
        return None
    if core.count("を") > 1:
        return "compound", "not a plain command: it names two things to act on (を twice). Read the request yourself"
    if re.search(r"[。!?]\s*\S", core):
        return "compound", "not a plain command: more than one sentence. Read the request yourself"
    if not _JA_REQ_END.search(core):
        return "statement", ("not a plain command: it does not end as a request (〜て / 〜てください / 〜を箱へ): a statement, "
                             "a question or a wish. Read the request yourself")
    return None


def _frame_problem(frame, piece: str, what: str) -> str:
    """"" when `piece` (a part of the sentence outside the object description) is nothing but the fixed
    frame of a plain command, else why the sentence is not read. A frame names no object: in 「赤いブロックの
    ある箱に」 the colour would be read as the container's."""
    core = _strip_end(piece.strip())
    if frame.fullmatch(core) and not _find(re.sub("ふた(?=[つりた])", "", core), KINDS):  # (「ふたつ」 is no lid)
        return ""
    return f"the sentence says more than {what}: 「{core[:40]}」 is not something these rules read. Read the request yourself"


def _ja_obj_problem(obj: str) -> str:
    """"" when a Japanese object description is only that, else why the sentence is not read."""
    for m in _CONT_ANY.finditer(obj):
        if not _OBJ_CONT_OK.match(obj, m.start()):
            return (f"the object description names a container (「{obj.strip()[:40]}」): two containers in one "
                    "request, or a place these rules do not read. Read the request yourself")
    if _OBJ_TE.search(_OBJ_TE_WORDS.sub("", obj)):
        return (f"the object description holds another clause (「{obj.strip()[:40]}」): more than one instruction? "
                "Read the request yourself")
    for m in _OBJ_REL.finditer(obj):
        if _find(obj[:m.start()], COLORS) or _find(obj[:m.start()], KINDS):
            return (f"the object is described by where it lies relative to another object (「{obj.strip()[:40]}」): "
                    "these rules cannot tell the two apart. Read the request yourself")
    return ""


def _en_obj_problem(obj: str) -> str:
    """The same for an English object description (lower case)."""
    core = _EN_OBJ_PLAIN.sub("", re.sub(r"[\s,]+$", "", obj))
    for m in _EN_OBJ_CONT.finditer(core):
        if m.end() != len(core):
            return (f"the object description names a container ('{core[:40]}'): two containers in one request, or "
                    "a place these rules do not read. Read the request yourself")
    m = _EN_OBJ_STOP.search(core)
    if m:
        return (f"the object description holds more than an object ('{m.group(0)}' in '{core[:40]}'): a place or a "
                "second instruction. Read the request yourself")
    return ""


def _before_frame(tail: str) -> str:
    """A take-out sentence without を (「箱の中の白いキャップは取り出して」): the object description is `tail`
    up to where the request frame begins (nothing when the frame is all there is, all of it when there is
    no such frame)."""
    end = _strip_end(tail)
    for i in range(len(end)):
        if _FRAME_TAKE.fullmatch(end[i:]):
            return tail[:i]
    return tail


def _find(text: str, table: dict) -> list:
    """Keys of `table` whose words occur in the text, in order of first appearance."""
    low = text.lower()
    hits = []
    for key, words in table.items():
        pos = [low.find(w.lower()) for w in words if w.lower() in low]
        if pos:
            hits.append((min(pos), key))
    return [k for _, k in sorted(hits)]


def _has_container(text: str) -> bool:
    return any(w.lower() in text.lower() for w in CONTAINERS)


def _split(text: str):
    """(target phrase, container phrase) or None."""
    t = text.strip()
    if "を" in t:
        left, right = t.split("を", 1)
        if _has_container(right):
            return left, right  # <object> を <container> に ...
        # <container> (の中) に <object> を ...
        m = re.search(rf"^(.*?(?:{_CONT_RE})(?:の中)?)\s*(?:に|へ)(.+)$", left, flags=re.IGNORECASE)
        if m:
            return m.group(2), m.group(1)
        return left, right
    low = t.lower()
    m = re.search(r"(?:put|place|move|drop)\s+(?:the\s+|all\s+the\s+|all\s+)?(.+?)\s+(?:into|inside|in|to)\s+(?:the\s+)?(.+)",
                  low)
    if m:
        return m.group(1), m.group(2)
    return None


def _put_problem(text: str, tgt_txt: str, con_txt: str) -> str:
    """"" when a put-in sentence (already split by _split into object and container phrase) is nothing but
    an object description inside the frame of a plain command, else why it is not read."""
    what = "'put <object> into <container>'"
    t = text.strip()
    if "を" not in t:  # English: put <object> into <container>
        return _en_obj_problem(tgt_txt) or _frame_problem(_EN_FRAME_PUT, con_txt, what)
    left, right = t.split("を", 1)
    if _has_container(right):  # <object> を <container> に ...
        return _ja_obj_problem(left) or _frame_problem(_FRAME_PUT, right, what)
    # <container> (の中) に <object> を <verb>
    if _OBJ_STATE.match(tgt_txt):
        return ("the object is described as lying in the container already (「〜にある」「〜に入っている」): that is not "
                "a request to put it in. Read the request yourself")
    return (_frame_problem(_FRAME_PUT_CON, con_txt, what) or _ja_obj_problem(tgt_txt)
            or _frame_problem(_FRAME_PUT_VERB, right, what))


def _targets(tgt_txt: str, want_all: bool):
    """Target queries named in a phrase, or a reason (str) why they cannot be read."""
    colors = _find(tgt_txt, COLORS)
    kinds = _find(tgt_txt, KINDS)
    if len(kinds) > 1:
        return f"several shapes named ({kinds}); one request per shape"
    if len(colors) > 3:
        return f"too many colours named ({colors})"
    base = {"kind": kinds[0]} if kinds else {}
    targets = [{"color": c, **base} for c in colors] or ([base] if (base or want_all) else [])
    return targets or "no colour or shape recognised in the object description"


def _container(con_txt: str):
    """Container query (dict, possibly empty) named in a phrase, or a reason (str)."""
    ccolors = _find(con_txt, COLORS)
    if len(ccolors) > 1:
        return f"ambiguous container colour {ccolors}"
    return {"color": ccolors[0]} if ccolors else {}


def _parse_take(text: str):
    """A request to take something OUT of a container: the result of parse() with action "take_out",
    a refusal ({"ok": False, ...}) when it is one but cannot be read safely, None when it is not one."""
    low = text.lower()
    what = "'take <object> out of <container>' (and put it on the table)"
    two = "moving something from one container into another is not a rule-readable request"

    def no(reason, kind=None):
        r = {"ok": False, "action": "take_out", "reason": reason}
        if kind:
            r["not_plain"] = kind
        return r

    if not _JA_RE.search(text):
        m = _TAKE_EN.search(low)
        if m is None:
            return None
        tgt_txt, con_txt = m.group("t"), m.group("c")
        if not _has_container(con_txt):
            return None  # "take the cap from the table": not out of a container
        rest = low[m.end("c"):] + " " + con_txt
        if _INTO_EN.search(rest):
            return no(two)
        # nothing but the object, the container and at most "and put it on the table": "... and then put it
        # back", "... , then put the blue cube in the box" are requests with a second part
        problem = (_en_obj_problem(tgt_txt) or _frame_problem(_EN_FRAME_TAKE, con_txt, what)
                   or _frame_problem(_EN_AFTER, low[m.end("c"):], what))
    else:
        out_verb = any(w in text for w in OUT_WORDS)
        take_verb = any(w in text for w in TAKE_WORDS)
        hits = [m for m in (_FROM_RE.search(text), _INSIDE_RE.search(text)) if m is not None]
        if not hits:
            if out_verb and not any(w.lower() in low for w in PUT_WORDS):
                return no("a take-out request needs the container it is taken out of (「箱から〜を出して」)")
            return None
        if not (out_verb or take_verb):
            return None
        m = min(hits, key=lambda h: h.start())
        cut = text.find("を")
        if 0 <= cut < m.start():  # <object> を <container> から ...
            tgt_txt, con_txt, rest = text[:cut], text[cut + 1:m.end()], text[m.end():]
            before = text[cut + 1:m.start()]
        else:  # <container> から <object> を ... / <container> の中の <object> を ...
            con_txt, tail = text[:m.end()], text[m.end():]
            tgt_txt = tail.split("を", 1)[0] if "を" in tail else _before_frame(tail)
            rest = tail[len(tgt_txt):]
            before = text[:m.start()]
        # 「赤い箱から白い箱に白いキャップを出して」: the second container must not be dropped silently
        if _INTO_RE.search(rest) or _INTO_RE.search(tgt_txt):
            return no(two)
        problem = (_frame_problem(_FRAME_TAKE_PRE, before, what)
                   or ("two containers are named before the object. Read the request yourself"
                       if _CONT_ANY.search(before) else "")
                   or _ja_obj_problem(tgt_txt) or _frame_problem(_FRAME_TAKE, rest, what))
    if any(w.lower() in low for w in ALL_WORDS):
        return no("taking everything out is not supported: one object (or one colour) per request")
    if problem:
        return no(problem, "extra")
    targets = _targets(tgt_txt, False)
    if isinstance(targets, str):
        return no(targets)
    container = _container(con_txt)
    if isinstance(container, str):
        return no(container)
    return {"ok": True, "action": "take_out", "targets": targets, "target": targets[0], "container": container or None,
            "all": False, "hollow": _HOLLOW_RE.search(tgt_txt) is not None}


def parse(text: str) -> dict:
    """Returns {"ok": True, "action": "put_in" | "take_out", "targets": [query, ...], "target": first query,
    "container": query | None, "all": bool} or {"ok": False, "reason": ...}. Query keys are those of
    `fd pick` / `fd take`: color, kind. Several colours ("赤と青のブロック") give one query per colour, to
    be handled in that order. For "take_out" the container is the one the object is taken out of, and
    "hollow" says whether the request calls the object a cap, a cup or a lid.

    Only a plain positive command comes back ok. A refusal that carries "not_plain" ("prohibition",
    "exclusion", "negation", "comparison", "condition", "choice", "question", "statement", "compound",
    "extra") means: the sentence is, or holds, something these rules do not understand - nothing must be
    done on the strength of the object and container names in it."""
    if not text or not text.strip():
        return {"ok": False, "reason": "empty request"}
    text = _clean(text)
    if not text:
        return {"ok": False, "reason": "empty request"}
    refused = _not_plain(text)
    if refused is not None:
        return {"ok": False, "reason": refused[1], "not_plain": refused[0]}
    low = text.lower()
    take = _parse_take(text)
    if take is not None:
        return take
    verbless = "を" in text and re.search(r"(?:に|へ)\s*[。．.!！]*\s*$", text) is not None
    if not any(w.lower() in low for w in PUT_WORDS) and not verbless:
        return {"ok": False, "reason": "not a put-into request (no verb such as 入れて / put)"}
    if any(w in text for w in ("出し", "取り出", "だし")) or re.search(r"\b(?:out of|take out|remove)\b", low):
        return {"ok": False, "action": "take_out",
                "reason": "looks like a take-out request, but which object comes out of which container cannot be read"}
    parts = _split(text)
    if parts is None:
        return {"ok": False, "reason": "cannot tell the object from the destination (expected 「〜を〜に入れて」 "
                                       "or 'put the ... into the ...')"}
    tgt_txt, con_txt = parts
    if not _has_container(con_txt):
        return {"ok": False, "reason": "the destination is not a container word (箱 / ボックス / box ...)"}
    problem = _put_problem(text, tgt_txt, con_txt)
    if problem:
        return {"ok": False, "reason": problem, "not_plain": "extra"}
    want_all = any(w.lower() in low for w in ALL_WORDS)
    targets = _targets(tgt_txt, want_all)
    if isinstance(targets, str):
        return {"ok": False, "reason": targets}
    container = _container(con_txt)
    if isinstance(container, str):
        return {"ok": False, "reason": container}
    return {"ok": True, "action": "put_in", "targets": targets, "target": targets[0], "container": container or None,
            "all": want_all}


def _selftest() -> int:
    blue_box = [{"color": "blue", "kind": "box"}]
    white_cap = [{"color": "white", "kind": "cylinder"}]
    cases = [
        ("青いキューブをボックスに入れて", blue_box, None, False),
        ("白いキャップをBoxに入れて", [{"color": "white", "kind": "cylinder"}], None, False),
        ("赤いBoxを黒い箱の中に入れてください", [{"color": "red"}], {"color": "black"}, False),
        ("クリーム色の円柱を箱に入れて", [{"color": "ivory", "kind": "cylinder"}], None, False),
        ("put the red block into the black box", [{"color": "red", "kind": "box"}], {"color": "black"}, False),
        ("Put the blue cube in the box.", blue_box, None, False),
        ("箱に青いキューブを入れて", blue_box, None, False),
        ("黒い箱の中に赤いブロックを入れて", [{"color": "red", "kind": "box"}], {"color": "black"}, False),
        ("青いキューブを箱へ", blue_box, None, False),
        ("テーブルの上のものを全部箱に入れて", [{}], None, True),
        ("ブロックを全部箱に片付けて", [{"kind": "box"}], None, True),
        ("赤と青のブロックを箱に入れて", [{"color": "red", "kind": "box"}, {"color": "blue", "kind": "box"}], None, False),
        ("青いブロックと赤いブロックを箱に入れて", [{"color": "blue", "kind": "box"}, {"color": "red", "kind": "box"}],
         None, False),
        ("move all blocks to the bin", [{"kind": "box"}], None, True),
        ("アームを止めて", "fail"),
        ("全部箱に入れて", "fail"),  # no 「を」: object and destination cannot be told apart
        ("青いキューブを右に動かして", "fail"),
        ("キャップとブロックを箱に入れて", "fail"),
        ("何かを箱に入れて", "fail"),
        ("", "fail"),
        # ---- taking something out of a container (a 5th element names the action)
        ("箱の中の白いキャップを取り出して", white_cap, None, False, "take_out"),
        ("白いキャップを箱から出して", white_cap, None, False, "take_out"),
        ("箱から赤いブロックを取って机に置いて", [{"color": "red", "kind": "box"}], None, False, "take_out"),
        ("箱から赤いブロックを出して", [{"color": "red", "kind": "box"}], None, False, "take_out"),
        ("黒い箱の中にある青いキューブを外に出してください", blue_box, {"color": "black"}, False, "take_out"),
        ("白いキャップを黒いトレーから取り出して", white_cap, {"color": "black"}, False, "take_out"),
        ("箱のなかの赤と青のブロックをとりだして", [{"color": "red", "kind": "box"}, {"color": "blue", "kind": "box"}],
         None, False, "take_out"),
        ("take the white cap out of the box", white_cap, None, False, "take_out"),
        ("Remove the red block from the black box.", [{"color": "red", "kind": "box"}], {"color": "black"}, False,
         "take_out"),
        ("please take out the blue cube from the tray and put it on the table", blue_box, None, False, "take_out"),
        ("箱から青いコップを取り出して", [{"color": "blue", "kind": "cylinder"}], None, False, "take_out"),
        ("箱から白いキャップを出して別の箱に入れて", "fail"),  # from one container into another
        ("take the white cap out of the box and put it into the tray", "fail"),
        ("箱の中のものを全部出して", "fail"),  # everything: not supported
        ("白いキャップを出して", "fail"),  # out of what?
        ("箱から何かを出して", "fail"),
        ("白いキャップを取って", "fail"),  # from the table? not a take-out, not a put-in
        ("take the white cap from the table", "fail"),
        # ---- the requests of 2026-11-11 as they were typed, and a few more plain commands (a greeting, a
        # "please", "pick it up and ...", an adverb change nothing)
        ("Boxの中のペットボトルのキャップ(白)を取り出して", white_cap, None, False, "take_out"),
        ("ペットボトルキャップをBoxから取り出して", [{"kind": "cylinder"}], None, False, "take_out"),
        ("すみません、青いキューブを箱に入れてください。", blue_box, None, False),
        ("青いキューブを取って箱に入れて", blue_box, None, False),
        ("箱の横の青いキューブをそっと箱に入れておいて", blue_box, None, False),
        ("箱から白いキャップをふたつ出して", white_cap, None, False, "take_out"),  # (how many is the skill's business)
        ("「箱の中の白いキャップを取り出して！」", white_cap, None, False, "take_out"),
        ("箱から白いキャップを取り出してテーブルの上に置いてください", white_cap, None, False, "take_out"),
        ("箱の中の白いキャップは取り出して", white_cap, None, False, "take_out"),
        ("Please put the blue cube in the box, thanks", blue_box, None, False),
        ("箱から白いキャップを取り出してください。お願いします。", white_cap, None, False, "take_out"),
        ("青いキューブを箱に入れといて", blue_box, None, False),  # (= 入れておいて)
        ("take the white cap out of the black box quickly, please", white_cap, {"color": "black"}, False, "take_out"),
    ]
    # ---- not a plain positive command: never ok, whatever object and container it names (key: what is found)
    not_plain = {
        "prohibition": (
            "箱の中の白いキャップを取り出さないで", "箱から白いキャップを出さないでください", "箱の中の白いキャップは取り出すな",
            "箱から白いキャップを出してはいけない", "箱から白いキャップを出しちゃだめ", "箱の中の白いキャップを取り出すのは禁止",
            "箱の中の白いキャップを取り出すのをやめて", "箱の中の赤いキャップは残して白いキャップは取り出して",
            "青いキューブを箱に入れないで", "青いキューブは箱に入れるな", "青いキューブを箱に入れてはいけません",
            "青いキューブを箱に入れるのをやめて", "青いキューブを箱に入れずに待機して",
            "白いキャップに触らずに赤いブロックを箱に入れて",
            "don't take the white cap out of the box", "do not remove the red block from the box",
            "never take the white cap out of the box", "Please don't put the blue cube in the box",
            "put no blocks into the box", "take the red block out of the box and leave the blue one"),
        "exclusion": (
            "箱から白いキャップ以外を出して", "箱の中の白いキャップ以外を取り出して", "白いキャップではなく赤いブロックを箱から出して",
            "青いキューブ以外を箱に入れて", "白いキャップを除いて箱に入れて", "白いキャップの代わりに赤いブロックを箱に入れて",
            "take everything except the white cap out of the box", "take all but the white cap out of the box",
            "put everything except the blue cube into the box", "put the blocks into the box without the red one"),
        "negation": ("箱の中の白くないキャップを取り出して", "青くないキューブを箱に入れて", "take the non-white things out of the box"),
        "comparison": ("赤いブロックより先に青いキューブを箱に入れて", "put the cube bigger than the red one into the box"),
        "condition": (
            "箱から白いキャップを出したら教えて", "もし赤いブロックがあれば箱に入れて", "箱が空なら白いキャップを入れて",
            "青いキューブを箱に入れてから白いキャップを出して", "白いキャップを出す前に青いキューブを箱に入れて",
            "あとで青いキューブを箱に入れて", "put the blue cube into the box if it is empty",
            "take the white cap out of the box when I say so", "put the red block into the box after the blue one"),
        "choice": ("赤か青のブロックを箱に入れて", "put the red or the blue block into the box"),
        "question": (
            "箱の中の白いキャップを取り出せる?", "箱の中の白いキャップを取り出したのは誰?", "箱の中の白いキャップを取り出せますか",
            "青いキューブを箱に入れましたか？", "どの箱に青いキューブを入れて", "can you see the white cap in the box?",
            "did you take the white cap out of the box", "could you put the blue cube into the box"),
        "statement": (
            "箱の中の白いキャップを取り出した", "箱の中の白いキャップを取り出しました。", "青いキューブを箱に入れた",
            "青いキューブを箱に入れる", "青いキューブを箱に入れたい", "箱の中の白いキャップを取り出せる",
            "I took the white cap out of the box", "the white cap was taken out of the box",
            "you should put the blue cube into the box"),
        "compound": (
            "箱からコップを出して水を入れて", "青いキューブを箱に入れて、ふたを閉めて", "箱の中の白いキャップを取り出して。それから箱に戻して",
            "Put the blue cube in the box. Then park.", "put the blue cube in the box\nthen take it out"),
        # the rest of the sentence is not the frame of a plain command: a second instruction, a place the
        # rules do not know, an object that is said to lie somewhere else or next to another one
        "extra": (
            "take the white cap out of the box and then put it back", "take the white cap out of the box and put it back in",
            "take the white cap out of the box, then put the blue cube away",
            "take the lid off and take the white cap out of the box", "put the blue cube in the box and then take it out",
            "put the blue cube next to the box", "put the blue cube in front of the box",
            "move the white cap from the red box to the white box", "take the thing next to the blue cube out of the box",
            "箱の中の白いキャップを取り出して机の右に置いて", "箱の中の白いキャップを取り出して赤いブロックの隣に置いて",
            "箱の中の白いキャップを取り出して赤いキャップも取り出して", "箱の中の白いキャップをこっちに出して",
            "箱の中の白いキャップを箱の外に置いて", "箱の中にある白いキャップを別の場所に移して", "赤い箱の白いキャップを黒い箱に入れて",
            "青いキューブを箱の横に置いて", "青いキューブを投げて箱に入れて", "青いキューブを箱に入れて待機して",
            "赤いブロックは机に置いて青いキューブを箱から出して", "箱の中の青いキューブの隣のやつを取り出して"),
    }
    # ---- two containers in one sentence: neither is dropped silently
    cases += [(t, "fail") for t in ("赤い箱から白い箱に白いキャップを出して", "箱の中の白いキャップを取り出して赤い箱に入れて",
                                    "箱の中の白いキャップを取り出してからまた箱に戻して",
                                    "take the white cap out of the red box into the white box")]
    ok = True
    for case in cases:
        text = case[0]
        r = parse(text)
        if case[1] == "fail":
            good = not r["ok"]
        else:
            action = case[4] if len(case) > 4 else "put_in"
            good = (r["ok"] and r["targets"] == case[1] and r["container"] == case[2] and r["all"] == case[3]
                    and r["action"] == action)
        ok &= good
        print("PASS" if good else "FAIL", repr(text), "->", r)
    for kind, texts in not_plain.items():
        for text in texts:
            r = parse(text)
            good = not r["ok"] and r.get("not_plain") == kind and "targets" not in r
            ok &= good
            print("PASS" if good else "FAIL", repr(text), "-> not a plain command:", r.get("not_plain"), "|", r.get("reason"))
    # a cap, a cup or a lid is hollow (its wall can be pinched); a block or a plain cylinder is not said to be
    for text, hollow in (("箱の中の白いキャップを取り出して", True), ("白いふたを箱から出して", True),
                         ("take the white cup out of the box", True), ("箱から青いコップを取り出して", True),
                         ("Take the white caps out of the box", True), ("箱から白いcapを出して", True),
                         ("箱から赤いブロックを出して", False), ("箱の中の白い円柱を取り出して", False),
                         ("take the solid white cylinder out of the box", False),
                         ("箱の中のふたつの白い円柱を取り出して", False), ("箱の中のふたつの白いふたを取り出して", True),
                         ("Boxの中のペットボトルのキャップ(白)を取り出して", True), ("ペットボトルキャップをBoxから取り出して", True)):
        r = parse(text)
        good = bool(r["ok"]) and r.get("action") == "take_out" and r.get("hollow") is hollow
        ok &= good
        print("PASS" if good else "FAIL", repr(text), "-> hollow", r.get("hollow"))
    print("nlq self-test:", "all passed" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
