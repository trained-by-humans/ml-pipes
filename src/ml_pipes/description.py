from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal

from .operator import OperatorArgument, get_operator_argument_entries, get_operator_name, render_operator

_FALLBACK_OPERATOR = object()


@dataclass(frozen=True)
class OperatorDescription:
    name: str
    arguments: tuple[OperatorArgument, ...] = field(default_factory=tuple)
    _operator: Any | None = field(default=None, repr=False, compare=False)

    @classmethod
    def from_operator(cls, operator: Any) -> "OperatorDescription":
        return cls(
            name=get_operator_name(operator),
            arguments=get_operator_argument_entries(operator),
            _operator=operator,
        )

    @property
    def passed_args(self) -> dict[str, Any]:
        return OperatorArgument.to_dict(argument for argument in self.arguments if argument.is_passed)

    @property
    def default_args(self) -> dict[str, Any]:
        return OperatorArgument.to_dict(argument for argument in self.arguments if argument.is_default)

    @property
    def all_args(self) -> dict[str, Any]:
        return OperatorArgument.to_dict(self.arguments)

    def render(
        self,
        *,
        show_defaults: bool = False,
        mode: Literal["repr", "describe"] = "repr",
    ) -> str:
        return render_operator(
            self._operator if self._operator is not None else _FALLBACK_OPERATOR,
            name=self.name,
            arguments=self.arguments,
            show_defaults=show_defaults,
            mode=mode,
        )

    def describe(self, *, show_defaults: bool = False) -> str:
        return self.render(show_defaults=show_defaults, mode="describe")

    def __repr__(self) -> str:
        return self.render()

    __str__ = __repr__


@dataclass(frozen=True)
class PipelineDescription:
    operators: list[OperatorDescription] = field(default_factory=list)

    def render(
        self,
        *,
        show_defaults: bool = False,
        mode: Literal["repr", "describe"] = "repr",
    ) -> str:
        if not self.operators:
            return "Pipeline([])"

        lines = ["Pipeline(["]
        for operator in self.operators:
            rendered = operator.render(show_defaults=show_defaults, mode=mode)
            operator_lines = rendered.splitlines() or [""]
            for line in operator_lines[:-1]:
                lines.append(f"  {line}")
            lines.append(f"  {operator_lines[-1]},")
        lines.append("])")
        return "\n".join(lines)

    def describe(self, *, show_defaults: bool = False) -> str:
        return self.render(show_defaults=show_defaults, mode="describe")

    def __repr__(self) -> str:
        return self.render()

    __str__ = __repr__


def _build_pipeline_description(
    operators: Iterable[Any],
) -> PipelineDescription:
    return PipelineDescription(
        operators=[OperatorDescription.from_operator(operator) for operator in operators]
    )
