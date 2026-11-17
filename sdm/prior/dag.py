# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from sdm.prior.hp import HP, UniformInt
from sdm.prior.mixture import Mixture


@dataclass(frozen=True)
class DAG:
    """Directed acyclic graph in CSR layout.

    Args:
        parent_ptr: Parent offsets with shape ``[N + 1]`` for ``N`` nodes.
        parent_idx: Parent indices with shape ``[E]`` for ``E`` edges.
            Each parent precedes its child.
        input_nodes: Optional feature node indices with shape ``[F]`` for
            ``F`` features.
        target_node: Optional target node index.
    """

    parent_ptr: Tensor  # [num_nodes + 1]: CSR offsets
    parent_idx: Tensor  # [num_edges]: Parent indices
    input_nodes: Tensor | None = None
    target_node: int | None = None

    @property
    def num_nodes(self) -> int:
        """Return the node count."""
        return len(self.parent_ptr) - 1

    @property
    def num_edges(self) -> int:
        """Return the edge count."""
        return len(self.parent_idx)

    def parents(self, i: int) -> Tensor:
        """Return a node's parent indices.

        Args:
            i: Node index.
        """
        return self.parent_idx[self.parent_ptr[i] : self.parent_ptr[i + 1]]

    @property
    def topological_order(self) -> Tensor:
        """Return all nodes in topological order."""
        return torch.arange(self.num_nodes, device=self.parent_ptr.device)

    @property
    def pruned_topological_order(self) -> Tensor:
        """Return assigned nodes and their ancestors in topological order.

        Return all nodes if either assignment is missing.
        """
        if self.input_nodes is None or self.target_node is None:
            return self.topological_order

        mask = torch.zeros(
            self.num_nodes,
            dtype=torch.bool,
            device=self.input_nodes.device,
        )
        mask[self.input_nodes] = True
        mask[self.target_node] = True
        for i in range(self.num_nodes - 1, -1, -1):
            if bool(mask[i]):
                mask[self.parents(i)] = True
        return mask.nonzero().view(-1)

    def with_columns(
        self,
        num_columns: int,
        generator: torch.Generator | None = None,
    ) -> "DAG":
        """Return a graph with sampled feature and target assignments.

        Args:
            num_columns: Feature count.
            generator: Pseudorandom generator.
        """
        target_nodes = _sample_columns(
            num_columns=1,
            num_nodes=self.num_nodes,
            generator=generator,
        )
        input_nodes = _sample_columns(
            num_columns=num_columns,
            num_nodes=self.num_nodes,
            generator=generator,
        )
        return dataclasses.replace(
            self,
            target_node=int(target_nodes),
            input_nodes=input_nodes,
        )

    def is_valid(self) -> bool:
        """Check for an undirected path between target and feature nodes.

        Unassigned graphs are accepted.
        """
        if self.input_nodes is None or self.target_node is None:
            return True

        candidates = set(self.input_nodes.tolist())
        if self.target_node in candidates:
            return True

        children: list[list[int]] = [[] for _ in range(self.num_nodes)]
        for node in range(self.num_nodes):
            for parent in self.parents(node).tolist():
                children[parent].append(node)

        visited = {self.target_node}
        stack = [self.target_node]
        while stack:
            node = stack.pop()
            for neighbor in self.parents(node).tolist() + children[node]:
                if neighbor in candidates:
                    return True
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)

        return False


class DAGSampler(ABC):
    """Base class for sampling directed acyclic graphs."""

    @abstractmethod
    def sample(
        self,
        generator: torch.Generator | None = None,
    ) -> DAG:
        """Sample a graph without feature or target assignments.

        Args:
            generator: Pseudorandom generator.
        """
        ...


class RandomCauchyDAGSampler(DAGSampler):
    """Sample DAGs with Cauchy-distributed connectivity parameters.

    For node indices i < j, the edge i -> j is placed with probability

        p_ij = sigmoid(A + B_i + C_j)

    where A, B_i, C_j are independent standard Cauchy random variables.

    A controls the overall connectivity level; B_i controls node i's
    outgoing tendency; C_j controls node j's incoming tendency.

    Args:
        num_nodes: Node count.
    """

    def __init__(self, num_nodes: HP[int] | int) -> None:
        self.num_nodes = HP.from_value(num_nodes)

    def sample(  # noqa: D102
        self,
        generator: torch.Generator | None = None,
    ) -> DAG:
        n = self.num_nodes.sample(generator)

        # Inverse-CDF sampling for standard Cauchy:
        # x ~ Uniform(0, 1) => tan(pi * (x - 0.5)) ~ Cauchy(0, 1).
        x = torch.rand(2 * n + 1, device="cpu", generator=generator)
        x = (torch.pi * (x - 0.5)).tan()

        # Build edge probability matrix. Enforce i < j:
        prob = x[0] + x[1 : n + 1].unsqueeze(1) + x[n + 1 :].unsqueeze(0)
        prob = prob.sigmoid()

        mask = torch.rand((n, n), device="cpu", generator=generator) < prob
        triu = torch.triu(
            torch.ones((n, n), device="cpu", dtype=torch.bool),
            diagonal=1,
        )
        adj = mask & triu

        return DAG(
            parent_ptr=torch.cat(
                [
                    torch.zeros(1, dtype=torch.long, device="cpu"),
                    adj.sum(dim=0).cumsum(0),
                ]
            ),
            parent_idx=adj.t().nonzero()[:, 1].contiguous(),
        )


class MixtureDAGSampler(DAGSampler, Mixture[DAGSampler]):
    """Pick one DAG sampler per call from a weighted pool.

    Args:
        samplers: Samplers or ``(sampler, weight)`` pairs. Bare samplers have
            weight one.
    """

    def __init__(
        self,
        samplers: Sequence[DAGSampler | tuple[DAGSampler, float]],
    ) -> None:
        Mixture.__init__(self, samplers, correlated=False)

    def sample(  # noqa: D102
        self,
        generator: torch.Generator | None = None,
    ) -> DAG:
        return self.pick(generator=generator).sample(generator=generator)


def _sample_columns(
    num_columns: int,
    num_nodes: int,
    generator: torch.Generator | None = None,
) -> Tensor:
    num_eligible = UniformInt(1, num_nodes).sample(generator=generator)
    eligible = torch.randperm(
        num_nodes,
        device="cpu",
        generator=generator,
    )[:num_eligible]
    index = torch.randint(
        low=0,
        high=num_eligible,
        size=(num_columns,),
        device="cpu",
        generator=generator,
    )
    return eligible[index]
