"""Curated style prompt templates for interior design restyling."""

from __future__ import annotations

from typing import TypedDict


class StylePreset(TypedDict):
    """Structure for a single style preset."""

    name: str
    prompt: str
    negative_prompt: str
    description: str


STYLE_PRESETS: dict[str, StylePreset] = {
    "Minimalist": {
        "name": "Minimalist",
        "prompt": (
            "Turn this room into a minimalist style room with clean lines, "
            "neutral color palette of whites and soft grays, hidden storage, "
            "and only essential furniture pieces. Remove all clutter and "
            "decorative excess — every item should serve a clear purpose."
        ),
        "negative_prompt": "cluttered, ornate, colorful, busy, patterned wallpaper, heavy curtains",
        "description": "Clean lines, neutral tones, and purposeful simplicity.",
    },
    "Scandinavian": {
        "name": "Scandinavian",
        "prompt": (
            "Turn this room into a Scandinavian style room with light wood "
            "furniture, white walls, cozy textiles like wool throws and sheepskin "
            "rugs, and plenty of natural light. Embrace hygge with warm candle-like "
            "lighting and simple organic shapes."
        ),
        "negative_prompt": "dark, heavy, ornate, industrial, cold, sterile",
        "description": "Light wood, white walls, and cozy hygge warmth.",
    },
    "Industrial": {
        "name": "Industrial",
        "prompt": (
            "Turn this room into an industrial style room with exposed brick walls, "
            "raw steel fixtures, concrete floors, and visible ductwork. Add vintage "
            "factory lighting, reclaimed wood accents, and an unfinished aesthetic "
            "that celebrates raw materials."
        ),
        "negative_prompt": "floral, pastel, carpet, polished, suburban, cute",
        "description": "Exposed brick, raw steel, and unfinished urban character.",
    },
    "Japandi": {
        "name": "Japandi",
        "prompt": (
            "Turn this room into a Japandi style room blending Japanese wabi-sabi "
            "with Scandinavian functionality. Use light oak furniture, shoji-inspired "
            "screens, handmade ceramics, low-profile seating, and a muted earth-tone "
            "palette. Emphasize negative space and natural textures."
        ),
        "negative_prompt": "colorful, ornate, baroque, heavy, cluttered, synthetic",
        "description": "Japanese wabi-sabi meets Scandinavian warmth and simplicity.",
    },
    "Bohemian": {
        "name": "Bohemian",
        "prompt": (
            "Turn this room into a bohemian style room with layered textiles, "
            "vibrant patterned rugs and pillows, macramé wall hangings, lush indoor "
            "plants, and an eclectic mix of globally-sourced furniture. Create a "
            "relaxed, collected-over-time atmosphere rich with personality."
        ),
        "negative_prompt": "sterile, minimal, corporate, uniform, plain white",
        "description": "Layered textures, global accents, and free-spirited warmth.",
    },
    "Mid-Century Modern": {
        "name": "Mid-Century Modern",
        "prompt": (
            "Turn this room into a mid-century modern style room with iconic "
            "teak and walnut furniture, tapered legs, geometric patterns, and a "
            "warm color palette of mustard yellow, olive green, and burnt orange. "
            "Add starburst clocks, Eames-inspired chairs, and organic curves."
        ),
        "negative_prompt": "ornate, baroque, ultra-modern, cold, chrome, glossy",
        "description": "Retro elegance with teak, tapered legs, and warm retro tones.",
    },
    "Coastal": {
        "name": "Coastal",
        "prompt": (
            "Turn this room into a coastal style room with a breezy ocean-inspired "
            "palette of whites, soft blues, and sandy beiges. Add natural rope "
            "accents, driftwood pieces, linen upholstery, and nautical touches like "
            "striped pillows and sea glass decorations."
        ),
        "negative_prompt": "dark, heavy, industrial, urban, neon, gothic",
        "description": "Breezy ocean palette with natural textures and nautical charm.",
    },
    "Rustic": {
        "name": "Rustic",
        "prompt": (
            "Turn this room into a rustic style room with reclaimed wood beams, "
            "stone fireplaces, distressed leather furniture, and warm earthy tones. "
            "Add vintage farm accessories, wrought iron fixtures, and a cozy "
            "cabin-like atmosphere that celebrates natural imperfections."
        ),
        "negative_prompt": "modern, sleek, chrome, glass, minimalist, fluorescent",
        "description": "Reclaimed wood, stone, and cozy cabin warmth.",
    },
    "Art Deco": {
        "name": "Art Deco",
        "prompt": (
            "Turn this room into an art deco style room with bold geometric patterns, "
            "luxurious materials like velvet and brass, high-gloss surfaces, and a "
            "rich color palette of black, gold, and emerald green. Add sunburst "
            "motifs, mirrored surfaces, and streamlined symmetry."
        ),
        "negative_prompt": "rustic, casual, bohemian, minimal, plain, country",
        "description": "Bold geometry, brass, velvet, and 1920s glamour.",
    },
    "Modern Farmhouse": {
        "name": "Modern Farmhouse",
        "prompt": (
            "Turn this room into a modern farmhouse style room with shiplap walls, "
            "apron-front sinks, barn-light fixtures, and a mix of vintage and "
            "contemporary pieces. Use a neutral palette with black accents, natural "
            "wood, and comfortable oversized seating."
        ),
        "negative_prompt": "sleek, industrial, ornate, vibrant, neon, futuristic",
        "description": "Shiplap, vintage charm, and contemporary comfort combined.",
    },
}


def get_preset_names() -> list[str]:
    """Return ordered list of available style preset names."""
    return list(STYLE_PRESETS.keys())


def get_preset(name: str) -> StylePreset:
    """Look up a style preset by name.

    Args:
        name: Exact preset name (case-sensitive).

    Returns:
        The matching ``StylePreset`` dict.

    Raises:
        KeyError: If the preset name is not found.
    """
    if name not in STYLE_PRESETS:
        available = ", ".join(get_preset_names())
        raise KeyError(f"Unknown style preset '{name}'. Available: {available}")
    return STYLE_PRESETS[name]
