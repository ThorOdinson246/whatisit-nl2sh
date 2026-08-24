"""Generate an organic chatter/robustness corpus for NL->shell models.

Goal: arbitrary human input (greetings, small talk, identity questions,
nonsense, ambiguous requests) must always yield a *sensible, harmless*
shell command instead of garbage or something network-touching.

Output: JSONL rows of {"nl": ..., "bash": ..., "source": "organic"}.

Usage:
    python organic_augment.py --output chatter_robustness.jsonl
"""

import argparse
import json
import random
import re

SEED = 42
WS = re.compile(r"\s+")


def norm(s):
    return WS.sub(" ", str(s).strip().lower())


rows = []


def add(nl, bash, source="organic"):
    rows.append({"nl": str(nl).strip(), "bash": bash, "source": source})


GREETINGS = [
    "hello", "hi", "hey", "yo", "sup", "howdy", "hiya", "greetings", "ahoy",
    "good morning", "good afternoon", "good evening", "good night",
    "hello there", "hi there", "hey there", "well hello", "why hello",
    "hello!", "hi!!", "heyy", "helllo", "hiii", "yo yo", "sup dude",
    "HELLO", "Hi.", "hey,", "um hi", "so, hello", "ok hi", "alright hey",
    "hello can you hear me", "anyone there?", "you there?", "you awake?",
    "knock knock", "testing", "test 1 2 3", "is this thing on", "ping me",
]
for g in GREETINGS:
    add(g, "echo hello")
    add(f"{g}, little friend", "echo hello")

SMALLTALK_HOW = [
    "how are you", "how are you doing", "how's it going", "how do you do",
    "what's up", "wassup", "whats good", "how have you been", "you good",
    "all good?", "how is life", "how's your day", "nice weather today",
    "i had a long day", "i'm tired", "so bored", "just checking in",
]
for s in SMALLTALK_HOW:
    add(s, 'echo "doing great, thanks!"')

THANKS = ["thank you", "thanks", "thx", "ty", "thanks!", "thank you so much",
          "thanks a lot", "much appreciated", "you're the best", "cheers"]
for t in THANKS:
    add(t, 'echo "you\'re welcome!"')

BYE = ["bye", "goodbye", "see ya", "see you later", "later", "cya", "good bye",
       "i'm leaving", "that's all", "done for today", "peace out"]
for b in BYE:
    add(b, 'echo "bye! come back anytime"')

IDENTITY = [
    "who are you", "what are you", "who r u", "what exactly are you",
    "are you human", "are you a robot", "are you an ai", "are you real",
    "what is your name", "do you have a name", "introduce yourself",
    "tell me about yourself", "what should i call you",
]
for q in IDENTITY:
    add(q, 'echo "I am nl2sh - tell me what to do in plain English"')

CAPABILITY = [
    "what can you do", "what do you do", "how do you work", "what are you for",
    "can you help me", "can you help", "help me", "help", "?", "what now",
    "i don't know what to ask", "give me something to try",
]
for c in CAPABILITY:
    add(c, "ls")

TIME = [
    "what time is it", "what's the time", "tell me the time", "current time",
    "what day is it", "what's today", "today's date", "what's the date",
    "what year is it", "which day of the week is it",
]
for t in TIME[:5]:
    add(t, "date")
for t in TIME[5:]:
    add(t, 'date +"%A %Y-%m-%d"')

SELF_STATE = [
    "where am i", "where are we", "what directory am i in", "am i root",
    "who am i", "what user am i", "my username", "am i admin",
    "what os is this", "what linux is this", "system info", "specs please",
    "show my ip", "how long has the pc been on", "uptime check",
]
m = {"where am i": "pwd", "where are we": "pwd", "what directory am i in": "pwd",
     "am i root": "whoami", "who am i": "whoami", "what user am i": "whoami",
     "my username": "whoami", "am i admin": "whoami",
     "what os is this": "uname -a", "what linux is this": "uname -a",
     "system info": "uname -a", "specs please": "uname -a",
     "show my ip": "hostname -I", "how long has the pc been on": "uptime",
     "uptime check": "uptime"}
for q in SELF_STATE:
    add(q, m.get(q, "pwd"))

NONSENSE = [
    "asdf", "qwerty", "zxcvb", "xxxx", "????", "...", ".", "..", "-", "---",
    "banana", "flurble wibble", "zzzz", "huh", "wat", "lol", "rofl", "kek",
    "1234", "42", "$%^&*(", "<html>", "{}", "null", "undefined", "[object Object]",
    "lorem ipsum dolor sit amet", "the quick brown fox", "asdf jkl;",
    "random words here nothing", "blah blah blah", "meow", "woof", "moo",
    "ok", "okay", "k", "yes", "no", "maybe", "sure", "fine", "cool", "nice",
    "wow", "oof", "bruh", "hmm", "hmmm...", "wait what", "idk", "idk man",
]
for n in NONSENSE:
    add(n, "pwd")

AMBIGUOUS = [
    "do the thing", "fix it", "make it better", "just do something",
    "you know what to do", "go", "run", "start", "continue", "next",
    "improve this", "optimize", "clean up", "tidy up my life",
]
for a in AMBIGUOUS:
    add(a, "pwd")

FUN = [
    "say hi to me", "say something nice", "make me smile", "entertain me",
    "tell me a joke", "sing a song", "beep boop", "make some noise",
    "do a barrel roll", "surprise me", "amuse me", "talk to me",
]
fun_map = {"say hi to me": "echo hello", "say something nice": 'echo "you are doing great"',
           "make me smile": 'echo "keep going, you got this"', "entertain me": "ls -la",
           "tell me a joke": 'echo "why do programmers prefer dark mode? because light attracts bugs"',
           "sing a song": 'echo "la la la"', "beep boop": "echo beep boop",
           "make some noise": "echo beep", "do a barrel roll": "rev <<< 'roll barrel a do'",
           "surprise me": "shuf -i 1-100 -n 1", "amuse me": "figlet hi 2>/dev/null || echo hi",
           "talk to me": 'echo "type a task and I will give you the command"'}
for f in FUN:
    add(f, fun_map[f])

# de-dup on normalized input, cap identical bash targets
seen_nl, count = set(), {}
final = []
for r in rows:
    k = norm(r["nl"])
    if k in seen_nl:
        continue
    seen_nl.add(k)
    kb = norm(r["bash"])
    if count.get(kb, 0) >= 120:
        continue
    count[kb] = count.get(kb, 0) + 1
    final.append(r)

random.Random(SEED).shuffle(final)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default="chatter_robustness.jsonl")
    args = ap.parse_args()
    with open(args.output, "w") as f:
        for r in final:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(final)} rows -> {args.output}")


if __name__ == "__main__":
    main()
