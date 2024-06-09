'''
Reference: https://huggingface.co/datasets/pierresi/cord/blob/main/cord.py
'''

import json
import os
from pathlib import Path

import datasets
from PIL import Image

logger = datasets.logging.get_logger(__name__)

_CITATION = """\
@article{park2019cord,
  title={CORD: A Consolidated Receipt Dataset for Post-OCR Parsing},
  author={Park, Seunghyun and Shin, Seung and Lee, Bado and Lee, Junyeop and Surh, Jaeheung and Seo, Minjoon and Lee, Hwalsuk}
  booktitle={Document Intelligence Workshop at Neural Information Processing Systems}
  year={2019}
}
"""

_DESCRIPTION = "https://github.com/clovaai/cord/"

_URLS = [
    "https://hcsun.net/assets/files/CORD-1k-001.zip",
    "https://hcsun.net/assets/files/CORD-1k-002.zip"
]

class Cord(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="cord", 
            version=datasets.Version("1.0.0"), 
            description="CORD dataset"
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "words": datasets.Sequence(datasets.Value("string")),
                    "bboxes": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=["O","B-MENU.NM","B-MENU.NUM","B-MENU.UNITPRICE","B-MENU.CNT","B-MENU.DISCOUNTPRICE","B-MENU.PRICE","B-MENU.ITEMSUBTOTAL","B-MENU.VATYN","B-MENU.ETC","B-MENU.SUB_NM","B-MENU.SUB_UNITPRICE","B-MENU.SUB_CNT","B-MENU.SUB_PRICE","B-MENU.SUB_ETC","B-VOID_MENU.NM","B-VOID_MENU.PRICE","B-SUB_TOTAL.SUBTOTAL_PRICE","B-SUB_TOTAL.DISCOUNT_PRICE","B-SUB_TOTAL.SERVICE_PRICE","B-SUB_TOTAL.OTHERSVC_PRICE","B-SUB_TOTAL.TAX_PRICE","B-SUB_TOTAL.ETC","B-TOTAL.TOTAL_PRICE","B-TOTAL.TOTAL_ETC","B-TOTAL.CASHPRICE","B-TOTAL.CHANGEPRICE","B-TOTAL.CREDITCARDPRICE","B-TOTAL.EMONEYPRICE","B-TOTAL.MENUTYPE_CNT","B-TOTAL.MENUQTY_CNT","I-MENU.NM","I-MENU.NUM","I-MENU.UNITPRICE","I-MENU.CNT","I-MENU.DISCOUNTPRICE","I-MENU.PRICE","I-MENU.ITEMSUBTOTAL","I-MENU.VATYN","I-MENU.ETC","I-MENU.SUB_NM","I-MENU.SUB_UNITPRICE","I-MENU.SUB_CNT","I-MENU.SUB_PRICE","I-MENU.SUB_ETC","I-VOID_MENU.NM","I-VOID_MENU.PRICE","I-SUB_TOTAL.SUBTOTAL_PRICE","I-SUB_TOTAL.DISCOUNT_PRICE","I-SUB_TOTAL.SERVICE_PRICE","I-SUB_TOTAL.OTHERSVC_PRICE","I-SUB_TOTAL.TAX_PRICE","I-SUB_TOTAL.ETC","I-TOTAL.TOTAL_PRICE","I-TOTAL.TOTAL_ETC","I-TOTAL.CASHPRICE","I-TOTAL.CHANGEPRICE","I-TOTAL.CREDITCARDPRICE","I-TOTAL.EMONEYPRICE","I-TOTAL.MENUTYPE_CNT","I-TOTAL.MENUQTY_CNT"]
                        )
                    ),
                    "image": datasets.features.Image(),
                }
            ),
            supervised_keys=None,
            citation=_CITATION,
            homepage="https://github.com/clovaai/cord/",
        )

    def _split_generators(self, dl_manager):
        downloaded_file = dl_manager.download_and_extract(_URLS)
        dest = Path(downloaded_file[0])/"CORD"
        self._move_files_to_dest(downloaded_file, dest)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"filepath": dest/"train"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, gen_kwargs={"filepath": dest/"dev"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"filepath": dest/"test"}
            ),
        ]

    def _move_files_to_dest(self, downloaded_file, dest):
        for split in ["train", "dev", "test"]:
            for file_type in ["image", "json"]:
                if split == "test" and file_type == "json":
                    continue
                files = (Path(downloaded_file[1])/"CORD"/split/file_type).iterdir()
                for f in files:
                    os.rename(f, dest/split/file_type/f.name)

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        ann_dir = os.path.join(filepath, "json")
        img_dir = os.path.join(filepath, "image")
        for guid, file in enumerate(sorted(os.listdir(ann_dir))):
            file_path = os.path.join(ann_dir, file)
            with open(file_path, "r", encoding="utf8") as f:
                data = json.load(f)
            image_path = os.path.join(img_dir, file).replace("json", "png")
            image, size = self._load_image(image_path)
            words, bboxes, ner_tags = self._process_data(data, size)
            yield guid, {"id": str(guid), "words": words, "bboxes": bboxes, "ner_tags": ner_tags, "image": image}

    def _load_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        return image, (w, h)

    def _process_data(self, data, size):
        words = []
        bboxes = []
        ner_tags = []
        for item in data["valid_line"]:
            line_words, label = item["words"], item["category"]
            line_words = [w for w in line_words if w["text"].strip() != ""]
            if len(line_words) == 0:
                continue
            cur_line_bboxes = self._process_line_words(line_words, label, size, words, ner_tags)
            bboxes.extend(cur_line_bboxes)
        return words, bboxes, ner_tags

    def _process_line_words(self, line_words, label, size, words, ner_tags):
        cur_line_bboxes = []
        if label == "other":
            for w in line_words:
                words.append(w["text"])
                ner_tags.append("O")
                cur_line_bboxes.append(self._normalize_bbox(self._quad_to_box(w["quad"]), size))
        else:
            words.append(line_words[0]["text"])
            ner_tags.append("B-" + label.upper())
            cur_line_bboxes.append(self._normalize_bbox(self._quad_to_box(line_words[0]["quad"]), size))
            for w in line_words[1:]:
                words.append(w["text"])
                ner_tags.append("I-" + label.upper())
                cur_line_bboxes.append(self._normalize_bbox(self._quad_to_box(w["quad"]), size))
        cur_line_bboxes = self._get_line_bbox(cur_line_bboxes)
        return cur_line_bboxes

    def _normalize_bbox(self, bbox, size):
        return [
            int(1000 * bbox[0] / size[0]),
            int(1000 * bbox[1] / size[1]),
            int(1000 * bbox[2] / size[0]),
            int(1000 * bbox[3] / size[1]),
        ]

    def _quad_to_box(self, quad):
        box = (
            max(0, quad["x1"]),
            max(0, quad["y1"]),
            quad["x3"],
            quad["y3"]
        )
        if box[3] < box[1] or box[2] < box[0]:
            box = self._fix_box(box)
        return box

    def _fix_box(self, box):
        bbox = list(box)
        if box[3] < box[1]:
            bbox[1], bbox[3] = bbox[3], bbox[1]
        if box[2] < box[0]:
            bbox[0], bbox[2] = bbox[2], bbox[0]
        return tuple(bbox)

    def _get_line_bbox(self, bboxs):
        x = [bboxs[i][j] for i in range(len(bboxs)) for j in range(0, len(bboxs[i]), 2)]
        y = [bboxs[i][j] for i in range(len(bboxs)) for j in range(1, len(bboxs[i]), 2)]
        x0, y0, x1, y1 = min(x), min(y), max(x), max(y)
        assert x1 >= x0 and y1 >= y0
        return [[x0, y0, x1, y1] for _ in range(len(bboxs))]