#!/usr/bin/env python3
"""Replace selected objects in a NInfer v2 artifact using exact encoded payloads.

This utility intentionally uses only Python's standard library.

It does not import tools.artifact because that package imports torch. It also
lives outside tools/artifact so the repository's inspect.py cannot shadow the
standard-library inspect module.

Output layout strategy
----------------------
The primary artifact's existing object order and inter-object padding are
preserved. Every output offset equals:

    primary_offset + cumulative_size_delta_of_prior_replacements

This preserves the primary artifact's proven alignment pattern as long as each
cumulative shift preserves the original alignment. The tool requires every
replacement size delta to be divisible by PAYLOAD_ALIGNMENT (4096 in the
current NInfer v2 format), a deliberately conservative condition for this
artifact-composition workflow.

No quantization or dequantization occurs.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import struct
from pathlib import Path


PREFIX = struct.Struct("<8sQ")
PREFIX_BYTES = PREFIX.size
MAGIC = b"NINFER\x00\x02"


def align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def read_payload_alignment(repo: Path) -> int:
    source = repo / "tools/artifact/container.py"
    tree = ast.parse(source.read_text())

    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "PAYLOAD_ALIGNMENT"
            and isinstance(node.value, ast.Constant)
            and type(node.value.value) is int
        ):
            value = node.value.value
            if value <= 0:
                raise RuntimeError("PAYLOAD_ALIGNMENT is not positive")
            return value

    raise RuntimeError(
        "could not read PAYLOAD_ALIGNMENT from tools/artifact/container.py"
    )


def sha256_file(path: Path, chunk: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_region(
    path: Path,
    absolute_offset: int,
    length: int,
    chunk: int = 16 * 1024 * 1024,
) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        f.seek(absolute_offset)
        remaining = length

        while remaining:
            b = f.read(min(chunk, remaining))
            if not b:
                raise EOFError(f"short read from {path}")
            h.update(b)
            remaining -= len(b)

    return h.hexdigest()


class Artifact:
    def __init__(self, path: Path, payload_alignment: int):
        self.path = path
        self.file_bytes = path.stat().st_size

        with path.open("rb") as f:
            raw_prefix = f.read(PREFIX_BYTES)

            if len(raw_prefix) != PREFIX_BYTES:
                raise RuntimeError(f"{path}: truncated prefix")

            magic, json_bytes = PREFIX.unpack(raw_prefix)

            if magic != MAGIC:
                raise RuntimeError(
                    f"{path}: unexpected artifact magic {magic!r}"
                )

            if json_bytes <= 0:
                raise RuntimeError(f"{path}: invalid JSON length")

            encoded_directory = f.read(json_bytes)

            if len(encoded_directory) != json_bytes:
                raise RuntimeError(f"{path}: truncated JSON directory")

        self.magic = magic
        self.json_bytes = json_bytes
        self.directory = json.loads(encoded_directory)

        if set(self.directory) != {"identity", "objects"}:
            raise RuntimeError(
                f"{path}: unexpected root members "
                f"{sorted(self.directory)}"
            )

        self.objects = self.directory["objects"]

        if not isinstance(self.objects, list) or not self.objects:
            raise RuntimeError(f"{path}: invalid objects array")

        self.payload_start = align_up(
            PREFIX_BYTES + json_bytes,
            payload_alignment,
        )

        self.by_name: dict[str, dict] = {}
        previous_end = 0

        for obj in self.objects:
            name = obj["name"]

            if name in self.by_name:
                raise RuntimeError(f"{path}: duplicate object {name}")

            offset = int(obj["offset"])
            size = int(obj["bytes"])

            if offset < previous_end:
                raise RuntimeError(f"{path}: object overlap at {name}")

            if self.payload_start + offset + size > self.file_bytes:
                raise RuntimeError(
                    f"{path}: payload range beyond EOF for {name}"
                )

            self.by_name[name] = obj
            previous_end = offset + size

    def obj(self, name: str) -> dict:
        try:
            return self.by_name[name]
        except KeyError:
            raise RuntimeError(
                f"{self.path}: missing object {name}"
            ) from None

    def absolute_offset(self, obj: dict) -> int:
        return self.payload_start + int(obj["offset"])


def descriptor_without_offset(obj: dict) -> dict:
    return {
        key: value
        for key, value in obj.items()
        if key != "offset"
    }


def replacement_compatible(primary: dict, donor: dict) -> None:
    if primary["name"] != donor["name"]:
        raise RuntimeError("replacement names differ")

    if primary["kind"] != donor["kind"]:
        raise RuntimeError(
            f"{primary['name']}: primary/donor kinds differ"
        )

    if primary["kind"] == "tensor":
        for field in ("shape", "layout"):
            if primary[field] != donor[field]:
                raise RuntimeError(
                    f"{primary['name']}: {field} differs "
                    f"primary={primary[field]!r} donor={donor[field]!r}"
                )

    elif primary["kind"] == "resource":
        if primary["encoding"] != donor["encoding"]:
            raise RuntimeError(
                f"{primary['name']}: resource encoding differs"
            )

    else:
        raise RuntimeError(
            f"{primary['name']}: unsupported kind {primary['kind']!r}"
        )


def copy_region(
    src,
    dst,
    absolute_offset: int,
    length: int,
    chunk: int = 16 * 1024 * 1024,
) -> None:
    src.seek(absolute_offset)
    remaining = length

    while remaining:
        data = src.read(min(chunk, remaining))
        if not data:
            raise EOFError("short payload read")
        dst.write(data)
        remaining -= len(data)


def encode_directory(identity: dict, objects: list[dict]) -> bytes:
    return json.dumps(
        {
            "identity": identity,
            "objects": objects,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--primary", type=Path, required=True)
    ap.add_argument("--donor", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--replace", action="append", default=[])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    replacements = set(args.replace)

    if not replacements:
        raise SystemExit("at least one --replace is required")

    payload_alignment = read_payload_alignment(args.repo)

    primary = Artifact(args.primary, payload_alignment)
    donor = Artifact(args.donor, payload_alignment)

    if primary.directory["identity"] != donor.directory["identity"]:
        raise RuntimeError("primary/donor identity differs")

    primary_names = [obj["name"] for obj in primary.objects]

    missing_primary = sorted(
        replacements - set(primary_names)
    )
    missing_donor = sorted(
        replacements - set(donor.by_name)
    )

    if missing_primary:
        raise RuntimeError(
            f"replacement objects missing from primary: {missing_primary}"
        )

    if missing_donor:
        raise RuntimeError(
            f"replacement objects missing from donor: {missing_donor}"
        )

    output_objects: list[dict] = []
    source_rows = []

    cumulative_delta = 0

    print(f"payload_alignment={payload_alignment}")
    print(f"objects={len(primary.objects)}")
    print(f"replace_count={len(replacements)}")

    for primary_obj in primary.objects:
        name = primary_obj["name"]

        if name in replacements:
            donor_obj = donor.obj(name)
            replacement_compatible(primary_obj, donor_obj)

            delta = int(donor_obj["bytes"]) - int(primary_obj["bytes"])

            if delta % payload_alignment != 0:
                raise RuntimeError(
                    f"{name}: replacement size delta {delta} is not "
                    f"a multiple of PAYLOAD_ALIGNMENT={payload_alignment}"
                )

            chosen = dict(donor_obj)
            source = donor
            source_label = "donor"

            print(
                "REPLACE "
                f"name={name} "
                f"primary_format={primary_obj.get('format')} "
                f"primary_bytes={primary_obj['bytes']} "
                f"donor_format={donor_obj.get('format')} "
                f"donor_bytes={donor_obj['bytes']} "
                f"delta={delta}"
            )

        else:
            chosen = dict(primary_obj)
            source = primary
            source_label = "primary"
            delta = 0

        target_offset = int(primary_obj["offset"]) + cumulative_delta

        if target_offset % payload_alignment != int(primary_obj["offset"]) % payload_alignment:
            raise RuntimeError(
                f"{name}: payload-alignment residue changed unexpectedly"
            )

        chosen["offset"] = target_offset
        output_objects.append(chosen)
        source_rows.append((source_label, source, chosen["name"]))

        cumulative_delta += delta

    # Validate the shifted primary gap structure and ensure no overlap.
    previous_primary_end = 0
    previous_output_end = 0

    for pobj, oobj in zip(primary.objects, output_objects):
        primary_gap = int(pobj["offset"]) - previous_primary_end
        output_gap = int(oobj["offset"]) - previous_output_end

        if primary_gap != output_gap:
            raise RuntimeError(
                f"{oobj['name']}: inter-object gap changed "
                f"{primary_gap} -> {output_gap}"
            )

        previous_primary_end = int(pobj["offset"]) + int(pobj["bytes"])
        previous_output_end = int(oobj["offset"]) + int(oobj["bytes"])

    print(f"final_cumulative_delta={cumulative_delta}")

    if args.dry_run:
        print("DRY_RUN=PASS")
        return 0

    encoded_directory = encode_directory(
        primary.directory["identity"],
        output_objects,
    )

    output_payload_start = align_up(
        PREFIX_BYTES + len(encoded_directory),
        payload_alignment,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    temp = args.output.with_name(args.output.name + ".tmp")
    temp.unlink(missing_ok=True)

    try:
        with (
            args.primary.open("rb") as primary_file,
            args.donor.open("rb") as donor_file,
            temp.open("wb") as out_file,
        ):
            out_file.write(
                PREFIX.pack(
                    MAGIC,
                    len(encoded_directory),
                )
            )
            out_file.write(encoded_directory)

            directory_pad = (
                output_payload_start
                - PREFIX_BYTES
                - len(encoded_directory)
            )

            out_file.write(b"\x00" * directory_pad)

            cursor = 0

            for index, (oobj, source_row) in enumerate(
                zip(output_objects, source_rows),
                start=1,
            ):
                source_label, source_artifact, name = source_row
                source_obj = source_artifact.obj(name)

                offset = int(oobj["offset"])
                size = int(oobj["bytes"])

                if offset < cursor:
                    raise RuntimeError(
                        f"{name}: output overlap during write"
                    )

                if offset > cursor:
                    out_file.write(b"\x00" * (offset - cursor))

                source_file = (
                    primary_file
                    if source_artifact is primary
                    else donor_file
                )

                copy_region(
                    source_file,
                    out_file,
                    source_artifact.absolute_offset(source_obj),
                    int(source_obj["bytes"]),
                )

                cursor = offset + size

                if (
                    index == 1
                    or index == len(output_objects)
                    or index % 100 == 0
                    or name in replacements
                ):
                    print(
                        f"[{index}/{len(output_objects)}] "
                        f"{name} "
                        f"source={source_label} "
                        f"bytes={size}",
                        flush=True,
                    )

        temp.replace(args.output)

    except BaseException:
        temp.unlink(missing_ok=True)
        raise

    output = Artifact(args.output, payload_alignment)

    if output.directory["identity"] != primary.directory["identity"]:
        raise RuntimeError("output identity mismatch")

    if [obj["name"] for obj in output.objects] != primary_names:
        raise RuntimeError("output object order mismatch")

    verified_replacements = []

    for out_obj in output.objects:
        name = out_obj["name"]

        expected_artifact = (
            donor
            if name in replacements
            else primary
        )

        expected_obj = expected_artifact.obj(name)

        if descriptor_without_offset(out_obj) != descriptor_without_offset(expected_obj):
            raise RuntimeError(
                f"{name}: descriptor mismatch\n"
                f"output={descriptor_without_offset(out_obj)}\n"
                f"expected={descriptor_without_offset(expected_obj)}"
            )

        out_hash = sha256_region(
            output.path,
            output.absolute_offset(out_obj),
            int(out_obj["bytes"]),
        )

        expected_hash = sha256_region(
            expected_artifact.path,
            expected_artifact.absolute_offset(expected_obj),
            int(expected_obj["bytes"]),
        )

        if out_hash != expected_hash:
            raise RuntimeError(
                f"{name}: payload SHA mismatch"
            )

        if name in replacements:
            verified_replacements.append(
                {
                    "name": name,
                    "format": out_obj.get("format"),
                    "bytes": out_obj["bytes"],
                    "payload_sha256": out_hash,
                }
            )

    output_sha = sha256_file(args.output)

    manifest = {
        "tool": "tools/ninfer_artifact_replace.py",
        "method": "shift-primary-layout-preserve-gaps",
        "requantization": False,
        "payload_alignment": payload_alignment,
        "primary": {
            "path": str(args.primary),
            "sha256": sha256_file(args.primary),
        },
        "donor": {
            "path": str(args.donor),
            "sha256": sha256_file(args.donor),
        },
        "output": {
            "path": str(args.output),
            "sha256": output_sha,
            "bytes": args.output.stat().st_size,
        },
        "replace_names": sorted(replacements),
        "final_cumulative_payload_delta": cumulative_delta,
        "verified_replacements": verified_replacements,
        "verification": {
            "identity": True,
            "object_order": True,
            "primary_gap_pattern_preserved": True,
            "all_descriptors": True,
            "all_payload_sha256": True,
        },
    }

    if args.manifest is not None:
        args.manifest.write_text(
            json.dumps(
                manifest,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

    print(f"output={args.output}")
    print(f"output_sha256={output_sha}")
    print(f"output_bytes={args.output.stat().st_size}")
    print("VERIFY=PASS")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
