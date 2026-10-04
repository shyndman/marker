from unittest.mock import Mock

from PIL import Image
from surya.common.progress import BatchProgressHandler
from surya.inference import SuryaInferenceManager
from surya.inference.schema import BatchInputItem, BatchOutputItem
from surya.layout import LayoutPredictor
from surya.recognition import RecognitionPredictor

from marker.builders.layout import LayoutBuilder
from marker.builders.ocr import OcrBuilder
from marker.progress import LLMProgressEvent, SuryaProgressEvent
from marker.schema import BlockTypes
from marker.schema.document import Document
from marker.schema.groups.page import PageGroup
from marker.schema.polygon import PolygonBox


def test_clean_html(recognition_model):
    builder = OcrBuilder(recognition_model)

    # Debug attributes are stripped, truncated tags are balanced
    html = '<p data-bbox="1 2 3 4" data-label="Text">Hello <b>world'
    cleaned = builder.clean_html(html)
    assert "data-bbox" not in cleaned
    assert "data-label" not in cleaned
    assert "</b>" in cleaned
    assert "Hello" in cleaned

    # Repetition loops are dropped
    looping = "<p>" + "same phrase " * 400
    assert builder.clean_html(looping) == ""

    assert builder.clean_html("") == ""


def test_inference_progress_precedes_document_updates():
    image = Image.new("RGB", (100, 100), "black")
    pages = [
        PageGroup(
            page_id=index,
            polygon=PolygonBox.from_bbox([0, 0, 100, 100]),
            lowres_image=image,
            highres_image=image,
            text_extraction_method="surya",
        )
        for index in range(2)
    ]
    document = Document(filepath="batch.pdf", pages=pages)
    events: list[SuryaProgressEvent | LLMProgressEvent] = []
    manager = Mock(spec=SuryaInferenceManager)

    def generate(
        batch: list[BatchInputItem],
        *,
        on_progress: BatchProgressHandler | None = None,
    ) -> list[BatchOutputItem]:
        operation = "layout" if batch[0].prompt_type == "layout" else "ocr"
        assert events[-1] == SuryaProgressEvent(operation, 0, len(batch))
        outputs = []
        for index in range(len(batch)):
            if operation == "layout":
                raw = '[{"label":"Equation","bbox":[0,0,1000,1000],"count":50}]'
                assert all(not page.children for page in pages)
            else:
                raw = f"<math>equation-{index}</math>"
                assert all(
                    not block.html
                    for block in document.contained_blocks((BlockTypes.Equation,))
                )
            outputs.append(
                BatchOutputItem(
                    raw=raw, token_count=5, error=False, metadata=batch[index].metadata
                )
            )
            assert on_progress is not None
            on_progress(index + 1, len(batch))
        return outputs

    manager.generate.side_effect = generate
    layout_builder = LayoutBuilder(LayoutPredictor(manager), Mock())
    layout_builder(document, Mock(), on_progress=events.append)
    OcrBuilder(RecognitionPredictor(manager), {"ocr_full_page": False})(
        document, Mock(), on_progress=events.append
    )
    assert [(event.operation, event.completed, event.total) for event in events] == [
        ("layout", 0, 2),
        ("layout", 1, 2),
        ("layout", 2, 2),
        ("ocr", 0, 2),
        ("ocr", 1, 2),
        ("ocr", 2, 2),
    ]
    assert [
        block.html for block in document.contained_blocks((BlockTypes.Equation,))
    ] == ["<math>equation-0</math>", "<math>equation-1</math>"]
