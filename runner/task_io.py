import re

OPENING_MESSAGE_PATTERN = re.compile(r"^>\s?(.*)$", re.MULTILINE)


def parse_opening_message(instruction_md: str) -> str:
    """The customer's first message is the blockquote in instruction.md."""
    lines = OPENING_MESSAGE_PATTERN.findall(instruction_md)
    if not lines:
        raise ValueError("instruction.md has no '> ' blockquote with the customer message")
    return " ".join(lines).strip()
