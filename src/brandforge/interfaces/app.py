"""Streamlit demo (BF-40).

Pick a brand, paste a brief, and read the variants, scores and flags. Generate posts
the brief to ``POST /generate``. This page does not run the graph (ADR 0033). Start
the API with ``brandforge serve`` before generating.
"""

import streamlit as st

from brandforge import config
from brandforge.interfaces import demo
from brandforge.models import RunResult

_RESULT_KEY = "result"
_ERROR_KEY = "error"


def _show_result(result: RunResult) -> None:
    text = demo.status_text(result)
    if result.status == "complete":
        st.success(text)
    elif result.status == "partial":
        st.warning(text)
    else:
        st.error(text)
    st.caption(demo.run_caption(result))

    views = demo.variant_views(result)
    table = demo.format_score_table(demo.score_rows(views))
    if table:
        st.markdown(table)
    if not views:
        st.markdown("(no variants were produced)")
    for view in views:
        st.subheader(f"{view.variant_id} · {view.channel} · {view.verdict}")
        st.markdown(demo.format_variant(view))
    errors = demo.format_errors(result)
    if errors:
        st.markdown(errors)


def main() -> None:
    """Render the page. Streamlit reruns this on every interaction."""
    st.set_page_config(page_title="BrandForge", layout="centered")
    st.title("BrandForge")
    st.caption(
        "Pick a brand and paste a brief. The page posts them to the API "
        "and shows the variants, scores and flags."
    )

    brands = demo.brand_choices()
    if not brands:
        st.error("No brand profiles found.")
        return

    settings = config.get_settings()
    default = brands.index("voltride") if "voltride" in brands else 0
    brand_id = st.selectbox("Brand", brands, index=default)
    brief_text = st.text_area(
        "Brief",
        value=demo.EXAMPLE_BRIEF,
        height=220,
        help="YAML or JSON with product, audience, objective, channels and optional constraints.",
    )
    st.caption(f"Posts to {settings.api_base_url}/generate. Start the API with `brandforge serve`.")

    if not isinstance(brand_id, str):
        st.error("Pick a brand.")
        return

    if st.button("Generate", type="primary"):
        st.session_state[_ERROR_KEY] = None
        st.session_state[_RESULT_KEY] = None
        try:
            brief = demo.parse_brief(brief_text)
            with st.spinner("Generating. A run can take a couple of minutes."):
                st.session_state[_RESULT_KEY] = demo.request_generation(
                    brand_id,
                    brief,
                    api_base_url=settings.api_base_url,
                    timeout=demo.generation_timeout(settings),
                )
        except demo.DemoError as exc:
            st.session_state[_ERROR_KEY] = str(exc)

    error = st.session_state.get(_ERROR_KEY)
    stored = st.session_state.get(_RESULT_KEY)
    if isinstance(error, str) and error:
        st.error(error)
    elif isinstance(stored, RunResult):
        _show_result(stored)


main()
