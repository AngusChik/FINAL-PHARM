"""Conservative extraction of explicitly written ordering-sheet drug details.

This is a text parser, not a drug dictionary: it does not correct names, expand
trade names, infer units, or identify a manufacturer that was not typed. Labels
outside the small supported grammar are left for staff to complete themselves.
"""

from decimal import Decimal
import re


_BRANDS = ('APO', 'PMS', 'JAMP', 'TEVA', 'SANDOZ', 'MYLAN', 'TARO')
_BRAND_PATTERN = '|'.join(_BRANDS)
_BRAND_TOKEN = re.compile(rf'(?<![A-Z0-9])(?:{_BRAND_PATTERN})(?![A-Z0-9])', re.I)
_PREFIX_BRAND = re.compile(
    rf'^(?P<brand>{_BRAND_PATTERN})(?:\s*-\s*|\s+)(?P<name>.+)$', re.I,
)
_SUFFIX_BRAND = re.compile(
    rf'^(?P<name>.+?)(?:\s*-\s*|\s+)(?P<brand>{_BRAND_PATTERN})$', re.I,
)
_PAREN_BRAND = re.compile(
    rf'^(?P<name>.+?)\s+\((?P<brand>{_BRAND_PATTERN})\)$', re.I,
)

_NUMBER = r'(?:\d+(?:\.\d+)?|\.\d+)'
_DOSE_UNIT = r'(?:mcg|mg|g|iu|units?|%)'
_VOLUME_UNIT = r'(?:ml|l)'
_STRENGTH_PART = rf'{_NUMBER}\s*{_DOSE_UNIT}'
_DENOMINATOR = rf'(?:{_NUMBER}\s*(?:{_DOSE_UNIT}|{_VOLUME_UNIT})|{_VOLUME_UNIT})'
_STRENGTH_TEXT = rf'{_STRENGTH_PART}(?:\s*/\s*{_DENOMINATOR})*'
_TERMINAL_STRENGTH = re.compile(
    rf'(?<![\d./])(?P<strength>{_STRENGTH_TEXT})$', re.I,
)
_ANY_STRENGTH = re.compile(rf'{_STRENGTH_PART}(?![A-Z])', re.I)
_COMPONENT = re.compile(
    rf'(?P<number>{_NUMBER})?\s*(?P<unit>{_DOSE_UNIT}|{_VOLUME_UNIT})', re.I,
)
_UNIT_DISPLAY = {
    'mg': 'mg', 'mcg': 'mcg', 'g': 'g', 'ml': 'mL', 'l': 'L',
    'iu': 'IU', 'unit': 'units', 'units': 'units', '%': '%',
}
_NAME = re.compile(r"[A-Z][A-Z0-9]*(?:[ '-][A-Z0-9]+)*", re.I)
_ALTERNATIVE = re.compile(r'\b(?:or|and)\b|[;,|+&\r\n]', re.I)


def _canonical_strength(value):
    parts = []
    for raw_part in value.split('/'):
        match = _COMPONENT.fullmatch(raw_part.strip())
        if match is None:
            return None
        unit = _UNIT_DISPLAY[match['unit'].lower()]
        number = match['number']
        if number is None:
            parts.append(unit)
            continue
        decimal = Decimal(number)
        if decimal <= 0:
            return None
        number = format(decimal, 'f')
        if '.' in number:
            number = number.rstrip('0').rstrip('.')
        parts.append(f'{number}{"" if unit == "%" else " "}{unit}')
    strength = '/'.join(parts)
    return strength if len(strength) <= 100 else None


def parse_prescription_drug_label(label):
    """Return ``(details, '')`` or ``(None, reason)`` for a free-text label.

    Accepted labels have one explicit allowlisted manufacturer at the start,
    or immediately before the terminal strength. A parenthesized manufacturer
    is also accepted in ``NAME (BRAND) STRENGTH``. Drug spelling and formulation
    words are preserved; only name/brand capitalization and strength formatting
    are normalized. A bare package volume is not a drug strength.
    """
    if not isinstance(label, str) or not label.strip() or len(label) > 400:
        return None, 'invalid_name'
    if _ALTERNATIVE.search(label):
        return None, 'ambiguous'
    if any(ord(character) < 32 and character != '\t' for character in label):
        return None, 'invalid_name'
    value = ' '.join(label.split())
    strength_match = _TERMINAL_STRENGTH.search(value)
    name_and_brand = value[:strength_match.start()].strip() if strength_match else value

    brands = _BRAND_TOKEN.findall(name_and_brand)
    if len(brands) > 1:
        return None, 'ambiguous'
    if not brands:
        return None, 'missing_brand'
    if strength_match is None:
        return None, 'ambiguous' if _ANY_STRENGTH.search(value) else 'missing_strength'

    # An earlier numeric dose or a separator before the terminal dose signals
    # multiple instructions/options, rather than one complete drug identity.
    if _ANY_STRENGTH.search(name_and_brand) or '/' in name_and_brand:
        return None, 'ambiguous'
    if name_and_brand.upper().strip(' -()') in _BRANDS:
        return None, 'invalid_name'
    brand_match = (
        _PAREN_BRAND.fullmatch(name_and_brand)
        or _PREFIX_BRAND.fullmatch(name_and_brand)
        or _SUFFIX_BRAND.fullmatch(name_and_brand)
    )
    if brand_match is None:
        return None, 'ambiguous'
    name = brand_match['name'].strip().upper()
    if not _NAME.fullmatch(name) or len(name) > 200:
        return None, 'invalid_name'
    if re.search(r'\s\d', name):
        # Do not absorb an earlier dose, quantity, or pack count into the name.
        return None, 'ambiguous'
    strength = _canonical_strength(strength_match['strength'])
    if strength is None:
        return None, 'ambiguous'
    return {
        'name': name,
        'brand': brand_match['brand'].upper(),
        'strength': strength,
    }, ''
