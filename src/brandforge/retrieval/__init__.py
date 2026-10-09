"""Approved example corpus (BF-29).

Importing this package loads the corpus only. The Chroma index (`retrieval.index`, BF-30) and
the search (`retrieval.query`, BF-31) stay in their own modules, because both import Chroma.
"""

from brandforge.retrieval.corpus import ExampleLoadError, list_example_brand_ids, load_examples

__all__ = ["ExampleLoadError", "list_example_brand_ids", "load_examples"]
