from typing import List
from typing import Tuple

import tempfile
from pathlib import Path

import torch
from IPython.display import display
from IPython.display import Markdown

from health_multimodal.common.visualization import plot_phrase_grounding_similarity_map
from health_multimodal.text import get_bert_inference
from health_multimodal.text.utils import BertEncoderType
from health_multimodal.image import get_image_inference
from health_multimodal.image.utils import ImageModelType
from health_multimodal.vlp import ImageTextInferenceEngine

text_inference = get_bert_inference(BertEncoderType.BIOVIL_T_BERT)
image_inference = get_image_inference(ImageModelType.BIOVIL_T)

image_text_inference = ImageTextInferenceEngine(
    image_inference_engine=image_inference,
    text_inference_engine=text_inference,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_text_inference.to(device)

TypeBox = Tuple[float, float, float, float]

def plot_phrase_grounding(image_path: Path, text_prompt: str, bboxes: List[TypeBox]) -> None:
    similarity_map = image_text_inference.get_similarity_map_from_raw_data(
        image_path=image_path,
        query_text=text_prompt,
        interpolation="bilinear",
    )
    plot_phrase_grounding_similarity_map(
        image_path=image_path,
        similarity_map=similarity_map,
        bboxes=bboxes
    )

def plot_phrase_grounding_from_url(image_url: str, text_prompt: str, bboxes: List[TypeBox]) -> None:
    image_path = Path('tmpfile', "downloaded_chest_xray.jpg")
    # !curl -s -L -o {image_path} {image_url}
    plot_phrase_grounding(image_path, text_prompt, bboxes)

image_url = "https://openi.nlm.nih.gov/imgs/512/242/1445/CXR1445_IM-0287-4004.png"
text_prompt = "Left basilar consolidation seen"
# Ground-truth bounding box annotation(s) for the input text prompt
bboxes = [
    (306, 168, 124, 101),
]

text = (
    'The ground-truth bounding box annotation for the phrase'
    f' *{text_prompt}* is shown in the middle figure (in black).'
)

display(Markdown(text))
plot_phrase_grounding_from_url(image_url, text_prompt, bboxes)