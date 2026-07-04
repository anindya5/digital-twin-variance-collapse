"""Interactive confirmation before costly API runs."""


def confirm_proceed() -> bool:
    reply = input("Proceed? [y/N] ").strip().lower()
    return reply == "y"
