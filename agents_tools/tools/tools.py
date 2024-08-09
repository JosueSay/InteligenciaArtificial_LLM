# tavily ya está integrado con langchain
from langchain_community.tools.tavily_search import TavilySearchResults

def get_profile_url_tavily(name: str):
    """Searches for linkedin Profile Page."""

    # Crear objeto tavily
    search = TavilySearchResults()
    res = search.run(f"{name}")

    return res[0]["url"]