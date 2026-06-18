from __future__ import annotations

import abc
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from .data_model import DataItem, cap_type


class OperatorError(RuntimeError):
    pass


class Operator(abc.ABC):
    operator_id: str = "op.unknown"

    def __init__(self, **params: Any) -> None:
        self.params = dict(params)

    def __call__(self, item: Optional[DataItem]) -> Optional[DataItem]:
        if item is None:
            return None
        out = self.apply(item)
        if out is not None:
            out = out.add_history(self.operator_id, self.params)
        return out

    @abc.abstractmethod
    def apply(self, item: DataItem) -> Optional[DataItem]:
        raise NotImplementedError

    def matches_media(self, item: DataItem, prefix: str) -> bool:
        return cap_type(item.caps).startswith(prefix)


def merge_caps(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(old)
    out.update(new)
    props = dict(old.get("properties") or {})
    props.update(new.get("properties") or {})
    if props:
        out["properties"] = props
    return out


def semantic_item(schema: str, semantic_type: str, data: Any, source: DataItem, annotations: Optional[List[Dict[str, Any]]] = None) -> DataItem:
    return DataItem(
        caps={"semantic_type": semantic_type, "schema": schema},
        data=data,
        annotations=list(annotations or []),
        metadata=dict(source.metadata),
    )
