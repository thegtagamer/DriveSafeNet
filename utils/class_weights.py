import json, collections, argparse, torch, pathlib, sys

weather_map = {
    'clear': 0,
    'partly cloudy': 1,
    'overcast': 2,
    'rainy': 3,
    'snowy': 4,
    'foggy': 5,
}

parser = argparse.ArgumentParser()
parser.add_argument("--label_json", required=True,
                    help="BDD100K detection train/val JSON")
parser.add_argument("--out", default="weather_class_weights.pt",
                    help="Save *.pt* file here")
args = parser.parse_args()

ctr = collections.Counter()
skipped = 0

for ann in json.load(open(args.label_json)):
    w = ann.get("attributes", {}).get("weather", None)
    if w in weather_map:
        ctr[weather_map[w]] += 1
    else:
        skipped += 1

if skipped:
    print(f"Skipped {skipped} records with undefined weather", file=sys.stderr)

# ensure all 6 classes are present (0 counts for missing)
counts = torch.tensor([ctr[i] for i in range(6)], dtype=torch.float32)
weights = (1. / counts).clone()
weights[counts == 0] = 0        # avoid inf if a class truly absent
weights /= weights.sum() / len(counts)  # rescale so mean(weight)=1

print("counts :", counts.tolist())
print("weights:", [round(float(w), 4) for w in weights])

torch.save(weights, args.out)
print(f"saved {args.out}")
