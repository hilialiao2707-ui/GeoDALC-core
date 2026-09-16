# Data layout

The repository does not include the geoscience dataset. Put the source file at:

```text
data/geogpt_qa.jsonl
```

Each line must be a JSON object. The preferred schema is:

```json
{"index": 0, "question": "...", "answer": "..."}
```

Records containing a single `text` field are also accepted by the compression loader.

GeoDALC deterministically shuffles the file with seed 42 and uses an 80/10/10 calibration, validation, and test split. Keep the same source file and order when reproducing the reported experiments.

WikiText-2 is downloaded through `datasets` by default. For an offline run, optionally provide:

```text
data/wikitext2-train.json
data/wikitext2-test.json
```

Each offline WikiText file should be a JSON list of strings or objects with a `text` field.

You may override the default locations with:

```bash
export GEODALC_DATA_DIR=/path/to/data
export GEODALC_GEOGPT_JSONL=/path/to/geogpt_qa.jsonl
export GEODALC_CACHE_DIR=/path/to/cache
```
