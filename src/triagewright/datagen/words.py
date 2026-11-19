"""Fixed word lists for the generator. Every name, street, and town here is invented."""

from __future__ import annotations

FIRST_NAMES: tuple[str, ...] = (
    "Ambrel",
    "Brisco",
    "Calloway",
    "Dovetta",
    "Emberly",
    "Fennick",
    "Galora",
    "Hollis",
    "Isadore",
    "Jessamy",
    "Kestrel",
    "Lumen",
    "Marisol",
    "Nettle",
    "Orlaith",
    "Peregrin",
    "Quenby",
    "Rosalind",
    "Sorrel",
    "Tamsin",
    "Ulric",
    "Vesper",
    "Wren",
    "Yarrow",
    "Zephyrine",
)

LAST_NAMES: tuple[str, ...] = (
    "Ashgrove",
    "Brightwater",
    "Copperfield",
    "Dunmarrow",
    "Elderquist",
    "Fernhollow",
    "Glimmerton",
    "Hawthorne-Vale",
    "Inkwell",
    "Juniperson",
    "Kettlewick",
    "Larkspur",
    "Mossbridge",
    "Nightingale-Rue",
    "Oakenshaw",
    "Pennywhistle",
    "Quillfeather",
    "Rookwood",
    "Silverthistle",
    "Thistledown",
)

EMAIL_DOMAINS: tuple[str, ...] = ("mail.invalid", "post.invalid", "inbox.invalid")

# Area codes are ordinary; the 555-0100..555-0199 exchange is reserved for fiction.
AREA_CODES: tuple[str, ...] = ("212", "312", "415", "503", "617", "702", "808", "919")

STREETS: tuple[str, ...] = (
    "Lanternfall Lane",
    "Quillwick Row",
    "Marrowmere Road",
    "Thimbleberry Court",
    "Gloamwood Avenue",
    "Saltmarsh Crescent",
    "Cobblewhistle Street",
    "Ninefold Way",
)

TOWNS: tuple[str, ...] = (
    "Fableford",
    "Nowhere Falls",
    "Little Imaginary",
    "Port Pretend",
    "Mistlebrook",
    "Underhollow",
)

REGIONS: tuple[str, ...] = ("US", "US", "US", "GB", "DE", "FR", "CA", "BR")

CARRIERS: tuple[str, ...] = ("UPS", "FedEx", "USPS", "DHL")

# (sku, name, unit price in cents). Regular items sit under the default $500 hard limit.
REGULAR_ITEMS: tuple[tuple[str, str, int], ...] = (
    ("SYN-CBL-01", "Braided Cable 1m", 900),
    ("SYN-CHG-02", "Pocket Charger", 2400),
    ("SYN-MSE-03", "Feather Mouse", 3900),
    ("SYN-LMP-04", "Glowbud Desk Lamp", 5400),
    ("SYN-SPK-05", "Pebble Speaker", 7900),
    ("SYN-KBD-06", "Quietkey Keyboard", 11900),
    ("SYN-HDP-07", "Hush Headphones", 17900),
    ("SYN-MON-08", "Clearview Monitor", 27900),
    ("SYN-TAB-09", "Slate Tablet", 39900),
)

# Premium items each cost more than the default $500 refund hard limit.
PREMIUM_ITEMS: tuple[tuple[str, str, int], ...] = (
    ("SYN-LAP-10", "Nimbus Air Laptop", 89900),
    ("SYN-CAM-11", "Lumen Pro Camera", 64900),
    ("SYN-TV-12", "Horizon 55 TV", 129900),
)

REFUND_REASONS: tuple[str, ...] = (
    "It stopped working after a few days",
    "It arrived damaged",
    "It is not what I expected",
    "I ordered the wrong one",
)
