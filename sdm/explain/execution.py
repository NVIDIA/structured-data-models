from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Literal

from torch import Tensor

from sdm import RelatedTables, TableTensor
from sdm.explain.base import ExplanationReplacements
from sdm.explain.result import InputSite


@dataclass(frozen=True)
class PreparedICLInputs:
    r"""Processed feature inputs bound to one ICL execution."""

    context: TableTensor
    query: TableTensor
    related_context: RelatedTables | None = None
    related_query: RelatedTables | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.context, TableTensor):
            raise TypeError("'context' needs to be a 'TableTensor'")
        if not isinstance(self.query, TableTensor):
            raise TypeError("'query' needs to be a 'TableTensor'")
        for name, tables in (
            ("related_context", self.related_context),
            ("related_query", self.related_query),
        ):
            if tables is not None and not isinstance(tables, RelatedTables):
                raise TypeError(
                    f"'{name}' needs to be 'RelatedTables' or None"
                )

    def table(self, site: InputSite) -> TableTensor:
        r"""Return the processed table at ``site``."""
        if not isinstance(site, InputSite):
            raise TypeError("'site' needs to be an 'InputSite'")
        if site.table is None:
            return self.context if site.split == "context" else self.query

        related = (
            self.related_context
            if site.split == "context"
            else self.related_query
        )
        if related is None or site.table not in related.tables:
            raise KeyError(site)
        return related.tables[site.table]

    @property
    def sites(self) -> Mapping[InputSite, TableTensor]:
        r"""Processed numerical input sites in stable execution order."""
        sites: dict[InputSite, TableTensor] = {}
        for split, table, related in (
            ("context", self.context, self.related_context),
            ("query", self.query, self.related_query),
        ):
            if table.numerical.size(-1) > 0:
                sites[InputSite(split=split)] = table
            if related is not None:
                sites.update(
                    {
                        InputSite(split=split, table=name): value
                        for name, value in related.tables.items()
                        if value.numerical.size(-1) > 0
                    }
                )
        return MappingProxyType(sites)

    def replace_numerical(
        self,
        replacements: ExplanationReplacements | None,
    ) -> "PreparedICLInputs":
        r"""Return inputs with validated processed numerical replacements."""
        replacements = _validate_replacements(self.sites, replacements)
        if len(replacements) == 0:
            return self

        def replace_table(table: TableTensor, site: InputSite) -> TableTensor:
            numerical = replacements.get(site)
            if numerical is None:
                return table
            return table.replace_blocks(numerical=numerical)

        def replace_related(
            related: RelatedTables | None,
            *,
            split: Literal["context", "query"],
        ) -> RelatedTables | None:
            if related is None:
                return None
            tables = {
                name: replace_table(
                    table,
                    InputSite(split=split, table=name),
                )
                for name, table in related.tables.items()
            }
            if all(
                table is related.tables[name] for name, table in tables.items()
            ):
                return related
            return replace(related, tables=tables)

        return PreparedICLInputs(
            context=replace_table(
                self.context,
                InputSite(split="context"),
            ),
            query=replace_table(
                self.query,
                InputSite(split="query"),
            ),
            related_context=replace_related(
                self.related_context,
                split="context",
            ),
            related_query=replace_related(
                self.related_query,
                split="query",
            ),
        )


@dataclass(frozen=True)
class PreparedFittedInputs:
    r"""Processed query inputs bound to one fitted execution."""

    query: TableTensor
    related_query: RelatedTables | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.query, TableTensor):
            raise TypeError("'query' needs to be a 'TableTensor'")
        if self.related_query is not None and not isinstance(
            self.related_query, RelatedTables
        ):
            raise TypeError(
                "'related_query' needs to be 'RelatedTables' or None"
            )

    @property
    def sites(self) -> Mapping[InputSite, TableTensor]:
        r"""Processed query sites in stable execution order."""
        sites: dict[InputSite, TableTensor] = {}
        if self.query.numerical.size(-1) > 0:
            sites[InputSite(split="query")] = self.query
        if self.related_query is not None:
            sites.update(
                {
                    InputSite(split="query", table=name): table
                    for name, table in self.related_query.tables.items()
                    if table.numerical.size(-1) > 0
                }
            )
        return MappingProxyType(sites)

    def replace_numerical(
        self,
        replacements: ExplanationReplacements | None,
    ) -> "PreparedFittedInputs":
        r"""Return query inputs with processed numerical replacements."""
        replacements = _validate_replacements(self.sites, replacements)
        if len(replacements) == 0:
            return self

        query = self.query
        numerical = replacements.get(InputSite(split="query"))
        if numerical is not None:
            query = query.replace_blocks(numerical=numerical)

        related_query = self.related_query
        if related_query is not None:
            tables: dict[str, TableTensor] = {}
            for name, table in related_query.tables.items():
                replacement = replacements.get(
                    InputSite(split="query", table=name)
                )
                tables[name] = (
                    table.replace_blocks(numerical=replacement)
                    if replacement is not None
                    else table
                )
            if any(
                table is not related_query.tables[name]
                for name, table in tables.items()
            ):
                related_query = replace(related_query, tables=tables)

        return PreparedFittedInputs(
            query=query,
            related_query=related_query,
        )


def _validate_replacements(
    inputs: Mapping[InputSite, TableTensor],
    replacements: ExplanationReplacements | None,
) -> dict[InputSite, Tensor]:
    replacements = {} if replacements is None else dict(replacements)
    validated: dict[InputSite, Tensor] = {}
    for site, replacement in replacements.items():
        if not isinstance(site, InputSite):
            raise TypeError("Replacement keys need to be 'InputSite' values")
        if site not in inputs:
            raise KeyError(f"Unknown input site: {site!r}")
        if not isinstance(replacement, Tensor) or isinstance(
            replacement, TableTensor
        ):
            raise TypeError(
                "Processed replacement values need to be plain tensors"
            )
        if replacement.is_inference():
            raise ValueError(
                f"Replacement at {site!r} needs to be a normal tensor, "
                "not an inference tensor"
            )

        expected = inputs[site].numerical
        if replacement.shape != expected.shape:
            raise ValueError(
                f"Replacement at {site!r} needs shape "
                f"{tuple(expected.shape)} (got {tuple(replacement.shape)})"
            )
        if replacement.device != expected.device:
            raise ValueError(
                f"Replacement at {site!r} needs device "
                f"'{expected.device}' (got '{replacement.device}')"
            )
        if replacement.dtype != expected.dtype:
            raise ValueError(
                f"Replacement at {site!r} needs dtype "
                f"'{expected.dtype}' (got '{replacement.dtype}')"
            )
        validated[site] = replacement
    return validated
