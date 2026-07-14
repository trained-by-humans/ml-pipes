from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

PayloadT = TypeVar("PayloadT")


class SideEffectOp(ABC, Generic[PayloadT]):
    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "__call__" in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} must not override __call__; implement effect() instead"
            )

    @abstractmethod
    def effect(self, payload: PayloadT) -> None:
        raise NotImplementedError

    def __call__(self, payload: PayloadT) -> PayloadT:
        self.effect(payload)
        return payload

    def resolve_contract(
        self,
        upstream_annotation: Any | None,
        stored_annotations: dict[str, Any],
        validation_error_type: type[Exception],
    ) -> tuple[tuple[Any, ...], Any]:
        del stored_annotations, validation_error_type
        return (Any,), upstream_annotation
