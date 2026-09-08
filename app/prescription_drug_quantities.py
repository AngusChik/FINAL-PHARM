"""Conservative quantity parsing and display for ordering-sheet history."""

from decimal import Decimal
import re


_UNIT_ALIASES = {
    'tabs': 'tablets', 'tablet': 'tablets', 'tablets': 'tablets',
    'caps': 'capsules', 'capsule': 'capsules', 'capsules': 'capsules',
    'pack': 'packs', 'packs': 'packs',
    'box': 'boxes', 'boxes': 'boxes',
    'bottle': 'bottles', 'bottles': 'bottles',
    'vial': 'vials', 'vials': 'vials',
    'ampoule': 'ampoules', 'ampoules': 'ampoules',
    'unit': 'units', 'units': 'units',
    'ml': 'mL', 'mls': 'mL',
    'g': 'g', 'gram': 'g', 'grams': 'g',
    'tube': 'tubes', 'tubes': 'tubes',
    'inhaler': 'inhalers', 'inhalers': 'inhalers',
}
_QUANTITY = re.compile(
    r'(?P<number>(?:[0-9]+(?:\.[0-9]{1,3})?|\.[0-9]{1,3}))'
    r'[ \t]*(?P<unit>[a-z]+)?', re.I | re.ASCII,
)
_MAX_QUANTITY = Decimal('999999999')


def parse_requested_quantity(raw):
    """Return an exact quantity and explicit unit, or ``(None, '')``.

    A bare number retains an unspecified unit. Package size is never inferred,
    and annotations, ranges, arithmetic, and unsupported units stay unparsed.
    Three fractional digits are accepted without rounding.
    """
    if not isinstance(raw, str) or len(raw) > 128:
        return None, ''
    if any(ord(character) < 32 and character != '\t' for character in raw):
        return None, ''
    match = _QUANTITY.fullmatch(raw.strip())
    if match is None:
        return None, ''
    raw_unit = match['unit']
    unit = _UNIT_ALIASES.get(raw_unit.lower()) if raw_unit else 'unspecified'
    if unit is None:
        return None, ''
    quantity = Decimal(match['number'])
    if quantity > _MAX_QUANTITY:
        return None, ''
    return quantity.normalize(), unit


def format_quantity_totals(totals):
    """Display stored decimal totals separately for each canonical unit."""
    parts = []
    for unit in sorted(totals, key=str.casefold):
        number = format(Decimal(totals[unit]), 'f')
        if '.' in number:
            number = number.rstrip('0').rstrip('.')
        label = '(unit unspecified)' if unit == 'unspecified' else unit
        parts.append(f'{number} {label}')
    return '; '.join(parts)
