"""Post-processing - ground-truth-free self-consistency checks."""
from __future__ import annotations


# ===========================================================================
# checks.py
# ===========================================================================

from edp.config import ValidateConfig
from edp.types import Net, Symbol, ValidationResult


def validate(symbols: list[Symbol], nets: list[Net], cfg: ValidateConfig) -> ValidationResult:
    result = ValidationResult()

    connected_symbol_ids = {sym_id for net in nets for sym_id, _ in net.terminals}
    for symbol in symbols:
        if symbol.terminals and symbol.id not in connected_symbol_ids:
            result.isolated_symbols.append(symbol.id)
        if symbol.confidence < cfg.min_confidence:
            result.low_confidence_symbols.append((symbol.id, symbol.confidence))
        for terminal in symbol.terminals:
            if terminal.net_id is None:
                result.unattached_terminals.append((symbol.id, terminal.index))

    for net in nets:
        if len(net.terminals) < 2:
            result.single_terminal_nets.append(net.id)

    return result
