"""Approved example corpus (BF-29). The Chroma index build is `retrieval.index` (BF-30).

Importing this package loads the corpus only. The index imports Chroma, so it stays in its
own module.
"""

from brandforge.retrieval.corpus import ExampleLoadError, list_example_brand_ids, load_examples

__all__ = ["ExampleLoadError", "list_example_brand_ids", "load_examples"]
