import re


PRODUCT_TYPES = {

    "laptop": [
        "gaming laptop",
        "laptop computer",
        "notebook computer",
        "laptop",
        "notebook"
    ],

    "speaker": [
        "bluetooth speaker",
        "wireless speaker",
        "portable speaker",
        "speaker"
    ],

    "headphones": [
        "wireless headphones",
        "bluetooth headphones",
        "gaming headset",
        "gaming headphones",
        "headphones",
        "earphones",
        "earbuds",
        "headset"
    ],

    "ssd": [
        "solid state drive",
        "ssd"
    ],

    "hard_drive": [
        "external hard drive",
        "internal hard drive",
        "hard disk drive",
        "hard drive",
        "hard disk",
        "hdd"
    ],

    "ram": [
        "memory module",
        "ddr3 ram",
        "ddr4 ram",
        "ddr5 ram",
        "ddr3",
        "ddr4",
        "ddr5",
        "ram"
    ],

    "tablet": [
        "android tablet",
        "tablet computer",
        "tablet pc",
        "ipad",
        "tablet"
    ],

    "smartphone": [
        "smartphone",
        "smart phone",
        "mobile phone",
        "iphone"
    ],

    "keyboard": [
        "mechanical keyboard",
        "gaming keyboard",
        "wireless keyboard",
        "computer keyboard",
        "keyboard"
    ],

    "mouse": [
        "gaming mouse",
        "wireless mouse",
        "computer mouse",
        "optical mouse",
        "mouse"
    ],

    "monitor": [
        "gaming monitor",
        "computer monitor",
        "desktop monitor",
        "lcd monitor",
        "led monitor",
        "monitor"
    ],

    "camera": [
        "digital camera",
        "dslr camera",
        "mirrorless camera",
        "webcam",
        "camera"
    ],

    "smartwatch": [
        "smartwatch",
        "smart watch"
    ],

    "graphics_card": [
        "graphics card",
        "video card",
        "gpu"
    ],

    "processor": [
        "computer processor",
        "desktop processor",
        "intel core",
        "amd ryzen",
        "processor",
        "cpu"
    ],

    "motherboard": [
        "motherboard",
        "mainboard"
    ],

    "router": [
        "wifi router",
        "wi-fi router",
        "wireless router",
        "network router",
        "router"
    ],

    "printer": [
        "laser printer",
        "inkjet printer",
        "multifunction printer",
        "printer"
    ],

    "projector": [
        "digital projector",
        "video projector",
        "lcd projector",
        "led projector",
        "projector"
    ],

    "power_bank": [
        "power bank",
        "portable power bank"
    ]
}


# Words that indicate the requested product is
# being mentioned as a target/compatible device,
# rather than being the actual product.
REFERENCE_PATTERNS = {

    "laptop": [
        r"\bfor\s+(?:the\s+)?laptop\b",
        r"\bfor\s+(?:the\s+)?laptop\s+computer\b",
        r"\bcompatible\s+with\s+(?:the\s+)?laptop\b",
        r"\bfor\s+laptop\b",
        r"\blaptop\s+(?:ram|memory|keyboard|mouse|stand|bag|case|cover|sleeve|charger|adapter)\b"
    ],

    "computer": [
        r"\bfor\s+(?:the\s+)?computer\b",
        r"\bcompatible\s+with\s+(?:the\s+)?computer\b"
    ]
}


def contains_keyword(text, keyword):
    return re.search(
        r"\b" + re.escape(keyword) + r"\b",
        text
    ) is not None


def detect_product_type(query):

    query = query.lower()

    matches = []

    for product_type, keywords in PRODUCT_TYPES.items():

        for keyword in keywords:

            if contains_keyword(query, keyword):

                matches.append(
                    (
                        len(keyword),
                        product_type
                    )
                )

    if not matches:
        return None

    # Prefer the longest / most specific phrase
    matches.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return matches[0][1]


def product_matches_type(title, product_type):

    title = title.lower()

    if product_type is None:
        return True

    keywords = PRODUCT_TYPES.get(
        product_type,
        []
    )

    # First determine whether this product
    # contains the requested product keyword.
    keyword_match = any(
        contains_keyword(title, keyword)
        for keyword in keywords
    )

    if not keyword_match:
        return False

    # -----------------------------------------
    # IMPORTANT:
    # Detect cases where the product is an
    # accessory/product FOR the requested item.
    # -----------------------------------------

    if product_type == "laptop":

        reference_patterns = [
            r"\bfor\s+(?:the\s+)?laptop\b",
            r"\bfor\s+laptop\b",
            r"\bcompatible\s+with\s+(?:the\s+)?laptop\b",
            r"\blaptop\s+(?:ram|memory|keyboard|mouse|stand|bag|case|cover|sleeve|charger|adapter)\b"
        ]

        for pattern in reference_patterns:

            if re.search(pattern, title):

                return False

    return True