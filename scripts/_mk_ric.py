import re

src = open("src/data/anytop_dataset.py").read()

start = src.index("def _recover_world_positions")
end = src.index("\ndef ", start + 10)
body = src[start:end]

# pull in every private helper the body calls, transitively
need, seen, helpers = list(set(re.findall(r"\b(_[a-z_][a-z_0-9]*)\(", body))), set(), []
while need:
    name = need.pop()
    if name in seen or name == "_recover_world_positions":
        continue
    seen.add(name)
    m = re.search(rf"^def {re.escape(name)}\(.*?(?=\n(?:def |class |@|\Z))", src, re.S | re.M)
    if not m:
        continue
    fn = m.group(0)
    helpers.append(fn)
    need += [n for n in re.findall(r"\b(_[a-z_][a-z_0-9]*)\(", fn) if n not in seen]

parts = [
    '"""Route B: decode the RIC position channels (0:3) of the AnyTop-13ch\n'
    'representation into world joint positions.\n\n'
    'Extracted verbatim from the training data loader so this pack runs standalone.\n'
    '"""',
    "import numpy as np",
]
parts += helpers[::-1]
parts += [body.replace("def _recover_world_positions", "def recover_world_positions")]

open("scratch/animal_minipack/ric.py", "w").write("\n\n".join(parts) + "\n")
print("ric.py written; helpers pulled:", [h.split("(")[0].replace("def ", "") for h in helpers])
