import json
from pathlib import Path

from raqw.provenance import sha256_file, write_manifest


def test_writes_sha256_manifest(tmp_path: Path) -> None:
    source = tmp_path / "input.txt"
    source.write_text("abc", encoding="utf-8")
    output = tmp_path / "manifest.json"

    write_manifest([source], output, root=tmp_path)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert sha256_file(source) == (
        "ba7816bf8f01cfea414140de5dae2223"
        "b00361a396177a9cb410ff61f20015ad"
    )
    assert payload["files"][0]["path"] == "input.txt"

