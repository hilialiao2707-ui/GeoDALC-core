import os
import random
import torch
import sys
from pathlib import Path
from datasets import load_dataset
from torch.utils.data.dataset import Dataset

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_path)

REPO_ROOT = Path(parent_path)
DATA_DIR = Path(os.environ.get("GEODALC_DATA_DIR", REPO_ROOT / "data"))
GEOGPT_JSONL = Path(os.environ.get("GEODALC_GEOGPT_JSONL", DATA_DIR / "geogpt_qa.jsonl"))
WIKITEXT_TRAIN_JSON = DATA_DIR / "wikitext2-train.json"
WIKITEXT_TEST_JSON = DATA_DIR / "wikitext2-test.json"
CACHE_DIR = Path(os.environ.get("GEODALC_CACHE_DIR", REPO_ROOT / ".cache"))


def _load_local_json_text_list(path):
    import json
    from pathlib import Path
    path = Path(path)
    if not path.exists():
        return None
    if path.suffix != '.json':
        return None
    raw = path.read_text(encoding='utf-8')
    texts = []
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    txt = item.get('text', '')
                else:
                    txt = str(item)
                if txt is not None:
                    texts.append(txt)
            return texts
    except json.JSONDecodeError:
        pass
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        if isinstance(item, dict):
            txt = item.get('text', '')
        else:
            txt = str(item)
        if txt is not None:
            texts.append(txt)
    return texts


def _record_to_text(record):
    if isinstance(record, str):
        return record
    if not isinstance(record, dict):
        return str(record)
    question = record.get("question", "")
    answer = record.get("answer", "")
    question = "" if question is None else str(question).strip()
    answer = "" if answer is None else str(answer).strip()
    if question and answer:
        return f"Question: {question}\nAnswer: {answer}"
    text = record.get("text", "")
    return text if isinstance(text, str) else ""


def _load_geogpt_splits(seed=42):
    import json
    path = GEOGPT_JSONL
    if not path.exists():
        raise FileNotFoundError(f"Missing GeoGPT-QA source file: {path}")
    texts = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            text = _record_to_text(item)
            if text:
                texts.append(text)
    rng = random.Random(seed)
    rng.shuffle(texts)
    n = len(texts)
    n_calib = int(n * 0.8)
    n_val = int(n * 0.1)
    calib = texts[:n_calib]
    val = texts[n_calib:n_calib + n_val]
    test = texts[n_calib + n_val:]
    return calib, val, test


def _load_wikitext_texts(split, dataset_cache_dir=None):
    local_path = WIKITEXT_TRAIN_JSON if split == "train" else WIKITEXT_TEST_JSON
    local_texts = _load_local_json_text_list(local_path)
    if local_texts is not None:
        return local_texts
    dataset = load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        split=split,
        cache_dir=dataset_cache_dir,
    )
    return list(dataset["text"])


def _build_mixed_texts(wiki_texts, geo_texts, geo_ratio, seed=42):
    rng = random.Random(seed)
    wiki = list(wiki_texts)
    geo = list(geo_texts)
    if not wiki:
        raise ValueError("WikiText pool is empty")
    if not geo:
        raise ValueError("GeoGPT pool is empty")
    n_geo = max(1, int(len(wiki) * geo_ratio))
    if n_geo <= len(geo):
        geo_part = rng.sample(geo, n_geo)
    else:
        geo_part = [geo[rng.randrange(len(geo))] for _ in range(n_geo)]
    mixed = wiki + geo_part
    rng.shuffle(mixed)
    return mixed


def get_calib_train_data(name, tokenizer, nsamples, seqlen=2048, seed=3, batch_size=1, dataset_cache_dir=None):
    import random
    random.seed(seed)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{name}_{nsamples}_{seqlen}_{seed}_{batch_size}.pt"
    nsamples += 1 #############################
    if cache_file.exists():
        traindataset = torch.load(cache_file)
        return traindataset
    if name == "c4":
        traindata = load_dataset("json", data_files="utils/c4-train.json")['train']
        tot_text = "\n\n".join(traindata["text"])
    elif name == "ptb":
        traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train', cache_dir=dataset_cache_dir)
        tot_text = "\n\n".join(traindata["sentence"])
    elif name == "wikitext2":
        tot_text = "\n\n".join(_load_wikitext_texts("train", dataset_cache_dir))
    elif name == "geogpt":
        calib_texts, _, _ = _load_geogpt_splits(seed=42)
        tot_text = "\n\n".join(calib_texts)
    elif name == "wikitext2_geogpt5":
        local_train = _load_wikitext_texts("train", dataset_cache_dir)
        calib_texts, _, _ = _load_geogpt_splits(seed=42)
        tot_text = "\n\n".join(_build_mixed_texts(local_train, calib_texts, 0.05, seed=seed))
    elif name == "wikitext2_geogpt10":
        local_train = _load_wikitext_texts("train", dataset_cache_dir)
        calib_texts, _, _ = _load_geogpt_splits(seed=42)
        tot_text = "\n\n".join(_build_mixed_texts(local_train, calib_texts, 0.10, seed=seed))
    elif name == "wikitext2_geogpt15":
        local_train = _load_wikitext_texts("train", dataset_cache_dir)
        calib_texts, _, _ = _load_geogpt_splits(seed=42)
        tot_text = "\n\n".join(_build_mixed_texts(local_train, calib_texts, 0.15, seed=seed))
    elif name == "wikitext2_geogpt20":
        local_train = _load_wikitext_texts("train", dataset_cache_dir)
        calib_texts, _, _ = _load_geogpt_splits(seed=42)
        tot_text = "\n\n".join(_build_mixed_texts(local_train, calib_texts, 0.20, seed=seed))
    elif name == "wikitext2_geogpt30":
        local_train = _load_wikitext_texts("train", dataset_cache_dir)
        calib_texts, _, _ = _load_geogpt_splits(seed=42)
        tot_text = "\n\n".join(_build_mixed_texts(local_train, calib_texts, 0.30, seed=seed))
    else:
        raise NotImplementedError
    traindataset = []
    for s in range(nsamples):
        i = random.randint(0, len(tot_text) - seqlen - 1)
        j = i + seqlen * 10
        trainenc = tokenizer(tot_text[i:j], return_tensors="pt")
        if trainenc.input_ids.shape[1] < seqlen:
            s = s - 1
            continue
        if s % batch_size == 0:
            if s != 0:
                attention_mask = torch.ones_like(inp)
                traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
            inp = trainenc.input_ids[:, :seqlen]
        else:
            inp = torch.cat((inp, trainenc.input_ids[:, :seqlen]), dim=0)
    torch.save(traindataset, cache_file)
    return traindataset



def get_wikitext2(nsamples, seed, seqlen, tokenizer, dataset_cache_dir=None):
    local_train = _load_wikitext_texts("train", dataset_cache_dir)
    local_test = _load_wikitext_texts("test", dataset_cache_dir)
    trainenc = tokenizer("\n\n".join(local_train), return_tensors="pt")
    testenc = tokenizer("\n\n".join(local_test), return_tensors="pt")

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def _tokenized_train_test_from_texts(train_texts, test_texts, tokenizer):
    trainenc = tokenizer("\n\n".join(train_texts), return_tensors="pt")
    testenc = tokenizer("\n\n".join(test_texts), return_tensors="pt")
    return trainenc, testenc


def get_geogpt(nsamples, seed, seqlen, tokenizer):
    calib_texts, _, test_texts = _load_geogpt_splits(seed=42)
    trainenc, testenc = _tokenized_train_test_from_texts(calib_texts, test_texts, tokenizer)
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_wikitext2_geogpt_mix(nsamples, seed, seqlen, tokenizer, geo_ratio):
    wiki_train = _load_wikitext_texts("train")
    wiki_test = _load_wikitext_texts("test")
    calib_texts, _, geo_test = _load_geogpt_splits(seed=42)
    mixed_train = _build_mixed_texts(wiki_train, calib_texts, geo_ratio, seed=seed)
    mixed_test = _build_mixed_texts(wiki_test, geo_test, geo_ratio, seed=seed + 1)
    trainenc, testenc = _tokenized_train_test_from_texts(mixed_train, mixed_test, tokenizer)
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_ptb(nsamples, seed, seqlen, tokenizer, dataset_cache_dir=None):
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train', cache_dir=dataset_cache_dir)
    valdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation', cache_dir=dataset_cache_dir)

    trainenc = tokenizer("\n\n".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(valdata['sentence']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_c4(nsamples, seed, seqlen, tokenizer):
    traindata = load_dataset("json", data_files="utils/c4-train.json")['train']
    valdata = load_dataset("json", data_files="utils/c4-validation.json")['train']

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    import random
    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
            if tmp.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)
    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc



def get_ptb_new(nsamples, seed, seqlen, tokenizer, dataset_cache_dir=None):
    from datasets import load_dataset
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train', cache_dir=dataset_cache_dir)
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test', cache_dir=dataset_cache_dir)

    trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_c4_new(nsamples, seed, seqlen, tokenizer):
    traindata = load_dataset("json", data_files="utils/c4-train.json")['train']
    valdata = load_dataset("json", data_files="utils/c4-validation.json")['train']

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc
def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    if name == 'geogpt':
        return get_geogpt(nsamples, seed, seqlen, tokenizer)
    if name == 'wikitext2_geogpt5':
        return get_wikitext2_geogpt_mix(nsamples, seed, seqlen, tokenizer, 0.05)
    if name == 'wikitext2_geogpt10':
        return get_wikitext2_geogpt_mix(nsamples, seed, seqlen, tokenizer, 0.10)
    if name == 'wikitext2_geogpt15':
        return get_wikitext2_geogpt_mix(nsamples, seed, seqlen, tokenizer, 0.15)
    if name == 'wikitext2_geogpt20':
        return get_wikitext2_geogpt_mix(nsamples, seed, seqlen, tokenizer, 0.20)
    if name == 'wikitext2_geogpt30':
        return get_wikitext2_geogpt_mix(nsamples, seed, seqlen, tokenizer, 0.30)
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if 'ptb' in name:
        if 'new' in name:
            return get_ptb_new(nsamples, seed, seqlen, tokenizer)
        return get_ptb(nsamples, seed, seqlen, tokenizer)
    if 'c4' in name:
        if 'new' in name:
            return get_c4_new(nsamples, seed, seqlen, tokenizer)
        return get_c4(nsamples, seed, seqlen, tokenizer)



def get_test_data(name, tokenizer, seq_len=2048, batch_size = 4):
    class IndexDataset(Dataset):
        def __init__(self, tensors):
            self.tensors = tensors

        def __getitem__(self, index):
            return self.tensors[index]

        def __len__(self):
            return len(self.tensors)
    ####
    def process_data(samples, tokenizer, seq_len, field_name):
        vals = samples[field_name]
        normalized = []
        for item in vals:
            if isinstance(item, str):
                normalized.append(item)
            elif isinstance(item, dict):
                if field_name in item and isinstance(item[field_name], str):
                    normalized.append(item[field_name])
                elif 'text' in item and isinstance(item['text'], str):
                    normalized.append(item['text'])
                else:
                    normalized.append(str(item))
            else:
                normalized.append(str(item))

        test_ids = tokenizer("\n\n".join(normalized), return_tensors='pt').input_ids[0]
        test_ids_batch = []
        nsamples = test_ids.numel() // seq_len

        for i in range(nsamples):
            batch = test_ids[(i * seq_len):((i + 1) * seq_len)]
            test_ids_batch.append(batch)
        test_ids_batch = torch.stack(test_ids_batch)
        return IndexDataset(tensors=test_ids_batch)
    ####
    if name == 'geogpt':
        _, _, test_texts = _load_geogpt_splits(seed=42)
        test_data = {'text': test_texts}
        test_dataset = process_data(test_data, tokenizer, seq_len, 'text')
    elif name == 'wikitext2_geogpt5':
        wiki_test = _load_wikitext_texts("test")
        _, _, geo_test = _load_geogpt_splits(seed=42)
        test_data = {'text': _build_mixed_texts(wiki_test, geo_test, 0.05, seed=43)}
        test_dataset = process_data(test_data, tokenizer, seq_len, 'text')
    elif name == 'wikitext2_geogpt10':
        wiki_test = _load_wikitext_texts("test")
        _, _, geo_test = _load_geogpt_splits(seed=42)
        test_data = {'text': _build_mixed_texts(wiki_test, geo_test, 0.10, seed=43)}
        test_dataset = process_data(test_data, tokenizer, seq_len, 'text')
    elif name == 'wikitext2_geogpt15':
        wiki_test = _load_wikitext_texts("test")
        _, _, geo_test = _load_geogpt_splits(seed=42)
        test_data = {'text': _build_mixed_texts(wiki_test, geo_test, 0.15, seed=43)}
        test_dataset = process_data(test_data, tokenizer, seq_len, 'text')
    elif name == 'wikitext2_geogpt20':
        wiki_test = _load_wikitext_texts("test")
        _, _, geo_test = _load_geogpt_splits(seed=42)
        test_data = {'text': _build_mixed_texts(wiki_test, geo_test, 0.20, seed=43)}
        test_dataset = process_data(test_data, tokenizer, seq_len, 'text')
    elif name == 'wikitext2_geogpt30':
        wiki_test = _load_wikitext_texts("test")
        _, _, geo_test = _load_geogpt_splits(seed=42)
        test_data = {'text': _build_mixed_texts(wiki_test, geo_test, 0.30, seed=43)}
        test_dataset = process_data(test_data, tokenizer, seq_len, 'text')
    elif 'wikitext2' in name:
        test_data = {'text': _load_wikitext_texts("test")}
        test_dataset = process_data(test_data, tokenizer, seq_len, 'text')
    if 'ptb' in name:
        test_data = load_dataset('ptb_text_only', 'penn_treebank', split='test')
        test_dataset = process_data(test_data, tokenizer, seq_len, 'sentence')
    elif 'c4' in name:
        test_data = load_dataset("json", data_files="utils/c4-validation.json")['train']
        test_dataset = process_data(test_data[0:2000], tokenizer, seq_len, 'text')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader
