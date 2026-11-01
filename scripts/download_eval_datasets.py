"""Download public, already-annotated retrieval benchmarks into data/eval_datasets.

Groups are fetched on demand. Archives are kept, extracted once, and re-runs
resume partial downloads (curl -C -) and skip finished extractions.

    .venv/bin/python scripts/download_eval_datasets.py base annotations
    .venv/bin/python scripts/download_eval_datasets.py photobench      # ~11 GB
    .venv/bin/python scripts/download_eval_datasets.py disbench        # ~14.4 GB
    .venv/bin/python scripts/download_eval_datasets.py docci           # ~7.6 GB
    .venv/bin/python scripts/download_eval_datasets.py coco_unlabeled  # ~20 GB
"""
import argparse
import subprocess
import sys
import zipfile
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data" / "eval_datasets"
HF = "https://huggingface.co/datasets/"
GH = "https://raw.githubusercontent.com/"

GROUPS = {
    # Small capability suites (~1.6 GB together).
    "base": [
        ("coco/val2017.zip", "http://images.cocodataset.org/zips/val2017.zip", True),
        ("coco/annotations_trainval2017.zip", "http://images.cocodataset.org/annotations/annotations_trainval2017.zip", True),
        ("visdial/VisualDialog_val2018.zip", "https://www.dropbox.com/s/twmtutniktom7tu/VisualDialog_val2018.zip?dl=1", True),
        ("visdial/visdial_1.0_val.zip", "https://www.dropbox.com/s/ibs3a0zhw74zisc/visdial_1.0_val.zip?dl=1", True),
        ("flickr1k/images_flickr_1k_test.zip", HF + "nlphuji/flickr_1k_test_image_text_retrieval/resolve/main/images_flickr_1k_test.zip", True),
        ("flickr1k/test_1k_flickr.csv", HF + "nlphuji/flickr_1k_test_image_text_retrieval/resolve/main/test_1k_flickr.csv", False),
        ("urban1k/Urban1k.zip", HF + "BeichenZhang/Urban1k/resolve/main/Urban1k.zip", True),
        ("countbench/countbench.parquet", HF + "nielsr/countbench/resolve/main/data/train-00000-of-00001-cf54c241ba947306.parquet", False),
    ],
    # Queries / ground truth / protocols only (a few MB, no large image sets).
    "annotations": [
        ("photobench/validation/album1_validation.json", HF + "SorrowTea/PhotoBench/resolve/main/data/validation/album1_validation.json", False),
        ("photobench/validation/album2_validation.json", HF + "SorrowTea/PhotoBench/resolve/main/data/validation/album2_validation.json", False),
        ("photobench/validation/album3_validation.json", HF + "SorrowTea/PhotoBench/resolve/main/data/validation/album3_validation.json", False),
        ("photobench/test/album1_test.json", HF + "SorrowTea/PhotoBench/resolve/main/data/test/album1_test.json", False),
        ("photobench/test/album2_test.json", HF + "SorrowTea/PhotoBench/resolve/main/data/test/album2_test.json", False),
        ("photobench/test/album3_test.json", HF + "SorrowTea/PhotoBench/resolve/main/data/test/album3_test.json", False),
        ("photobench/README.md", HF + "SorrowTea/PhotoBench/resolve/main/README.md", False),
        ("disbench/queries.jsonl", HF + "RUC-NLPIR/DISBench/resolve/main/queries.jsonl", False),
        ("disbench/README.md", HF + "RUC-NLPIR/DISBench/resolve/main/README.md", False),
        ("disbench/download_images.py", HF + "RUC-NLPIR/DISBench/resolve/main/download_images.py", False),
        ("chatir/dialogues/VisDial_v1.0_queries_val.json", GH + "levymsn/ChatIR/main/dialogues/VisDial_v1.0_queries_val.json", False),
        ("chatir/dialogues/ChatGPT_BLIP2.json", GH + "levymsn/ChatIR/main/dialogues/ChatGPT_BLIP2.json", False),
        ("chatir/dialogues/Human_BLIP2.json", GH + "levymsn/ChatIR/main/dialogues/Human_BLIP2.json", False),
        ("chatir/dialogues/FLAN_ALPACA_XXL_BLIP2.json", GH + "levymsn/ChatIR/main/dialogues/FLAN_ALPACA_XXL_BLIP2.json", False),
        ("chatir/dialogues/Unanswered_ChatGPT_BLIP2.json", GH + "levymsn/ChatIR/main/dialogues/Unanswered_ChatGPT_BLIP2.json", False),
        ("chatir/Search_Space_val_50k.json", GH + "levymsn/ChatIR/main/ChatIR_Protocol/Search_Space_val_50k.json", False),
        ("plugir/Search_Space_coco.json", GH + "Saehyung-Lee/PlugIR/main/Protocol/Search_Space_coco.json", False),
        ("plugir/coco_captions.json", GH + "Saehyung-Lee/PlugIR/main/Protocol/coco_captions.json", False),
        ("plugir/flickr_captions.json", GH + "Saehyung-Lee/PlugIR/main/Protocol/flickr_captions.json", False),
        ("plugir/visdial_captions.json", GH + "Saehyung-Lee/PlugIR/main/Protocol/visdial_captions.json", False),
        ("circo/annotations/val.json", GH + "miccunifi/CIRCO/main/annotations/val.json", False),
        ("circo/annotations/test.json", GH + "miccunifi/CIRCO/main/annotations/test.json", False),
        ("docci/docci_descriptions.jsonlines", "https://storage.googleapis.com/docci/data/docci_descriptions.jsonlines", False),
    ],
    "disbench": [
        ("disbench/images.zip", HF + "RUC-NLPIR/DISBench/resolve/main/images.zip", True),
    ],
    "docci": [
        ("docci/docci_images.tar.gz", "https://storage.googleapis.com/docci/data/docci_images.tar.gz", True),
    ],
    "coco_unlabeled": [
        ("coco/unlabeled2017.zip", "http://images.cocodataset.org/zips/unlabeled2017.zip", True),
    ],
}
# Google Drive folders need gdown (pip install gdown). The optional filter keeps
# only matching files (the NegBench folder also holds large model checkpoints).
# PhotoBench: the HF card's Drive ID drops a character; this one is from GitHub.
DRIVE = {
    "annotations": [("negbench", "https://drive.google.com/drive/folders/1kSEq0mkV1t1T8GuOAM65iz_iAA7e5gxB",
                     ("COCO_val_retrieval.csv", "COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv"))],
    "photobench": [("photobench/raw_albums", "https://drive.google.com/drive/folders/1IyAlRskgnXG6pJ7fYL5Ie0SFDKZ95-cI", None)],
}


def fetch(rel, url):
    target = ROOT / rel
    done = target.with_name(target.name + ".done")
    if done.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    print("下载", rel, flush=True)
    subprocess.run(["curl", "-L", "--fail", "--retry", "5", "--retry-delay", "5", "-C", "-",
                    "-o", str(target), url], check=True)
    done.touch()
    return target


def extract(archive):
    marker = archive.with_name(archive.name + ".extracted")
    if marker.exists():
        return
    print("解压", archive.relative_to(ROOT), flush=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(archive.parent)
    else:
        with tarfile.open(archive) as t:
            t.extractall(archive.parent, filter="data")
    marker.touch()


def drive(rel, url, names=None):
    target = ROOT / rel
    done = target / ".done"
    if done.exists():
        return
    try:
        import gdown
    except ImportError:
        sys.exit("需要 gdown：.venv/bin/python -m pip install gdown")
    target.mkdir(parents=True, exist_ok=True)
    print("下载 Google Drive 文件夹", rel, flush=True)
    listing = gdown.download_folder(url, skip_download=True, quiet=True)
    if not listing:
        raise RuntimeError(rel + "：无法读取 Google Drive 文件夹，可手动打开链接下载到 " + str(target))
    for item in listing:
        path = target / Path(item.path).name
        if names and path.name not in names:
            continue
        if not path.exists():
            # gdown resumes a partial file left in the output folder.
            gdown.download(id=item.id, output=str(path), quiet=False, resume=True)
        if path.suffix == ".zip":
            extract(path)
    done.touch()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("groups", nargs="+", choices=sorted(set(GROUPS) | set(DRIVE)))
    args = parser.parse_args()
    failed = []
    for group in args.groups:
        for rel, url, unpack in GROUPS.get(group, []):
            try:
                path = fetch(rel, url)
                if unpack:
                    extract(path)
            except Exception as exc:
                failed.append(f"{rel}: {exc}")
        for rel, url, names in DRIVE.get(group, []):
            try:
                drive(rel, url, names)
            except Exception as exc:
                failed.append(f"{rel}: {exc}")
    if failed:
        print("\n未完成：\n" + "\n".join(failed))
        sys.exit(1)
    print("完成：", ", ".join(args.groups))


if __name__ == "__main__":
    main()
