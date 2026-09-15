"""Teacher-only PNG attachments; never resize policy arrays or source images."""
from io import BytesIO
from pathlib import Path

from PIL import Image


def image_max_edge(value):
    if not str(value).isdigit():
        raise ValueError('CODEX_IMAGE_MAX_EDGE must be a non-negative integer (0 disables resizing)')
    return int(value)


def prepare_image(item, max_edge):
    """Return a copied descriptor and exact attachment bytes, saving any preview.

    ``path`` remains the full-resolution source for native view_image access.
    A separate preview is stored next to the immutable observation, so the
    existing artifact inventory and RPC log preserve what Codex actually saw.
    """
    max_edge = image_max_edge(max_edge)
    source = Path(item['path'])
    original = source.read_bytes()
    with Image.open(BytesIO(original)) as picture:
        if max_edge == 0 or max(picture.size) <= max_edge:
            return dict(item), original
        original_size = list(picture.size)
        picture.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
        buffer = BytesIO()
        picture.save(buffer, format='PNG')
        payload = buffer.getvalue()
        size = list(picture.size)
    preview = source.parent/'codex_previews'/f'{source.stem}.max{max_edge}.png'
    preview.parent.mkdir(exist_ok=True)
    try:
        with preview.open('xb') as stream:
            stream.write(payload)
    except FileExistsError:
        if preview.read_bytes() != payload:
            raise ValueError(f'Existing Codex preview differs; refusing to overwrite {preview}')
    return dict(item, codex_preview=dict(path=str(preview), size=size,
        original_size=original_size, resampling='LANCZOS')), payload
