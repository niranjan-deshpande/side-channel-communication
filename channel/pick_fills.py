"""From a fill-class map (mon_capture.py --map), pick fill sizes W so that training covers every class and testing
uses different W values in the same classes. Prints two lines: TRAIN W... and TEST W...

usage: python pick_fills.py results/fmap_on/map.json
"""
import json
import sys

rows = json.load(open(sys.argv[1]))
classes = {}
for r in rows:
    classes.setdefault(r["on"], []).append(r["W"])
train, test = [], []
for ws in classes.values():
    ws = sorted(ws)
    if len(ws) >= 3:
        train += [ws[0], ws[-1]]
        test.append(ws[len(ws) // 2])
    elif len(ws) == 2:
        train.append(ws[0]); test.append(ws[1])
    else:
        train.append(ws[0]); test.append(ws[0])
print("TRAIN", *sorted(train))
print("TEST", *sorted(test))
print(f"# {len(classes)} classes, {len(rows)} fills", file=sys.stderr)
