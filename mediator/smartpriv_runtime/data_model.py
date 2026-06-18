from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np


Caps = Dict[str, Any]
Annotation = Dict[str, Any]
Metadata = Dict[str, Any]


def now_ms() -> int:
    return int(time.time() * 1000)


def cap_type(caps: Optional[Caps]) -> str:
    if not caps:
        return ""
    return str(caps.get("semantic_type") or caps.get("media_type") or "")


def cap_schema(caps: Optional[Caps]) -> str:
    if not caps:
        return ""
    return str(caps.get("schema") or "")


def is_media(caps: Optional[Caps]) -> bool:
    t = cap_type(caps)
    return t.startswith("image/") or t.startswith("video/") or t.startswith("audio/")


@dataclasses.dataclass
class DataItem:
    """Runtime item passed between preprocessing operators.

    The shape mirrors the contract vocabulary:
      - caps: typed media/semantic description
      - data: actual payload, e.g. numpy image/audio array or dict event
      - annotations: structured observations produced by inference operators
      - metadata: timestamp, source id, processing history, etc.

    Operators mutate by returning a new DataItem rather than editing in place.
    """

    caps: Caps
    data: Any = None
    annotations: List[Annotation] = dataclasses.field(default_factory=list)
    metadata: Metadata = dataclasses.field(default_factory=dict)

    def clone(
        self,
        *,
        caps: Optional[Caps] = None,
        data: Any = None,
        annotations: Optional[List[Annotation]] = None,
        metadata: Optional[Metadata] = None,
    ) -> "DataItem":
        next_meta = dict(self.metadata)
        if metadata:
            next_meta.update(metadata)
        next_annotations = list(self.annotations if annotations is None else annotations)
        return DataItem(
            caps=dict(self.caps if caps is None else caps),
            data=self.data if data is None else data,
            annotations=next_annotations,
            metadata=next_meta,
        )

    def add_history(self, operator_id: str, params: Optional[Dict[str, Any]] = None) -> "DataItem":
        hist = list(self.metadata.get("process", []))
        hist.append({"operator": operator_id, "params": params or {}, "at_ms": now_ms()})
        return self.clone(metadata={"process": hist})

    def to_jsonable(self, include_payload: bool = False) -> Dict[str, Any]:
        payload: Any
        if include_payload:
            payload = _jsonable(self.data)
        else:
            payload = summarize_payload(self.data)
        return {
            "caps": _jsonable(self.caps),
            "data": payload,
            "annotations": _jsonable(self.annotations),
            "metadata": _jsonable(self.metadata),
        }

    @classmethod
    def from_jsonable(cls, obj: Dict[str, Any]) -> "DataItem":
        return cls(
            caps=dict(obj.get("caps") or {}),
            data=obj.get("data"),
            annotations=list(obj.get("annotations") or []),
            metadata=dict(obj.get("metadata") or {}),
        )


def _jsonable(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return {"ndarray": True, "shape": list(x.shape), "dtype": str(x.dtype)}
    if isinstance(x, (np.generic,)):
        return x.item()
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    return x


def summarize_payload(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, np.ndarray):
        return {"kind": "ndarray", "shape": list(payload.shape), "dtype": str(payload.dtype)}
    if isinstance(payload, dict):
        return {"kind": "dict", "keys": sorted(map(str, payload.keys()))[:25]}
    if isinstance(payload, list):
        return {"kind": "list", "len": len(payload)}
    if isinstance(payload, (str, bytes)):
        return {"kind": type(payload).__name__, "len": len(payload)}
    return {"kind": type(payload).__name__}


def load_json_item(path: Union[str, Path]) -> DataItem:
    with open(path, "r", encoding="utf-8") as f:
        return DataItem.from_jsonable(json.load(f))


def dump_json_item(item: Optional[DataItem], path: Union[str, Path], include_payload: bool = False) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(None if item is None else item.to_jsonable(include_payload=include_payload), f, indent=2)


def ensure_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]
