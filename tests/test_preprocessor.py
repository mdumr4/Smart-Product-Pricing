import pandas as pd
from src.preprocessor import CatalogPreprocessor


def test_clean_text():
    raw = "Apple iPhone 15 Pro Max 256GB - Blue Titanium! ✓ 🔥"
    cleaned = CatalogPreprocessor.clean_text(raw)
    assert "apple iphone 15 pro max" in cleaned
    assert "!" not in cleaned
    assert "✓" not in cleaned


def test_parse_catalog_content():
    sample_catalog = (
        "Item Name: Organic Peanut Butter Creamy\n"
        "Bullet Point 1: Made with 100% peanuts\n"
        "Product Description: Delicious spread for breakfast\n"
        "Value: 16.0\n"
        "Unit: Ounce"
    )
    parsed = CatalogPreprocessor.parse_catalog_content(sample_catalog)
    assert parsed["item_name"] == "Organic Peanut Butter Creamy"
    assert parsed["description"] == "Delicious spread for breakfast"
    assert parsed["value"] == "16.0"
    assert parsed["unit"] == "Ounce"


def test_normalize_unit():
    assert CatalogPreprocessor.normalize_unit("Fluid Ounce") == "fl_oz"
    assert CatalogPreprocessor.normalize_unit("Lbs") == "lb"
    assert CatalogPreprocessor.normalize_unit("Grams") == "g"
    assert CatalogPreprocessor.normalize_unit("Count") == "count"
    assert CatalogPreprocessor.normalize_unit("UNKNOWN") == "na"
