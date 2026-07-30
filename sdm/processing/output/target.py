from sdm.processing.base import Processor
from sdm.stype import Stype
from sdm.tensor import TableTensor


class TargetDecode(Processor):
    r"""Map member outputs back to their common fitted target space.

    In :class:`~sdm.processing.Recipe`, regression values are inverse-
    transformed and classification logits are aligned before any estimator
    reduction. Outside a fitted Recipe this processor is an identity.
    """

    supported_stypes = frozenset({Stype.numerical})
    supports_leading_variants = True
    requires_fit = False

    def _transform(self, table: TableTensor) -> TableTensor:
        return table
