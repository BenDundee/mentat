from googlesearch import search
from bs4 import BeautifulSoup
import concurrent.futures
import threading
from typing import List, Dict, Optional, Literal
import logging
import requests

from atomic_agents.lib.base.base_tool import BaseTool, BaseToolConfig

from src.interfaces import SearchResultItem, SearchToolInputSchema, SearchToolOutputSchema

# Lock for thread-safe operations if needed
lock = threading.Lock()


##############
# TOOL LOGIC #
##############
class SearchToolConfig(BaseToolConfig):
    base_url: str = ""
    max_results: int = 10


class SearchTool(BaseTool):
    """
    Tool for performing searches on SearxNG based on the provided queries and category.
    Attributes:
        input_schema (SearchToolInputSchema): The schema for the input data.
        output_schema (SearchToolOutputSchema): The schema for the output data.
        max_results (int): The maximum number of search results to return.
        base_url (str): The base URL for the SearxNG instance to use.
    """

    input_schema = SearchToolInputSchema
    output_schema = SearchToolOutputSchema

    def __init__(self, config: SearchToolConfig = SearchToolConfig()):
        """
        Initializes the SearxNGTool.

        Args:
            config (SearxNGSearchToolConfig):
                Configuration for the tool, including base URL, max results, and optional title and description overrides.
        """
        super().__init__(config)
        self.base_url = config.base_url
        self.max_results = config.max_results

    def extract_webpage_info(self, search_term: str, url: str) -> Dict[str, Optional[str]]:
        """
        Extract information from a webpage.
        Returns a dictionary with URL, title, published date, and content if available.
        """
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/91.0.4472.124 Safari/537.36'
        }
        result = {
            'query': search_term,
            'url': url,
            'title': None,
            'content': None
        }

        try:
            # NO SOUP FOR YOU
            soup = BeautifulSoup(requests.get(url).text, "html.parser")
            result["content"] = soup.get_text()

            # Extract title
            title_tag = soup.find('title')
            result['title'] = title_tag.text if title_tag else None

        except Exception as e:
            logging.error(f"Error processing {url}: {str(e)}")
        return result

    def get_google_search_results(self, search_term: str) -> List[Dict[str, Optional[str]]]:
        """
        Perform a Google search for the given term and return the top N results with extracted info.
        Uses parallel processing to fetch and parse webpages for a single search term.
        """
        results = []
        urls = []

        try:
            # Perform Google search and get top N URLs
            logging.info(f"Searching for: {search_term}")
            for i, url in enumerate(search(search_term)):
                if i >= self.max_results: break
                urls.append(url)
                logging.info(f"Found URL for {search_term}: {url}")

            # Process URLs in parallel using ThreadPoolExecutor
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_results) as executor:
                # Submit all tasks
                future_to_url = {executor.submit(self.extract_webpage_info, search_term, url): url for url in urls}
                for future in concurrent.futures.as_completed(future_to_url):
                    url = future_to_url[future]
                    try:
                        result = future.result()
                        results.append(result)
                        logging.info(f"Completed processing for {search_term}: {url}")
                    except Exception as e:
                        logging.error(f"Error processing {url} for {search_term}: {str(e)}")

        except Exception as e:
            logging.error(f"Error during Google search for {search_term}: {str(e)}")

        return results

    def process_multiple_searches(self, params: SearchToolInputSchema) -> SearchToolOutputSchema:
        """
        Process multiple search terms concurrently and return a dictionary mapping each term
        to its list of search results.
        """
        results = {}

        # Use ThreadPoolExecutor to process multiple search terms in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(params.queries)) as executor:
            # Submit all search tasks for different terms
            future_to_term = {
                executor.submit(self.get_google_search_results, term): term
                for term in params.queries
            }
            for future in concurrent.futures.as_completed(future_to_term):
                term = future_to_term[future]
                try:
                    result = future.result()
                    results[term] = result
                    logging.info(f"Completed search for term: {term}")
                except Exception as e:
                    logging.error(f"Error processing search term {term}: {str(e)}")
                    results[term] = []

        search_result_items = []
        for query in params.queries:
            search_result_items.extend([
                SearchResultItem(
                    url=_res["url"], title=_res["title"], content=_res.get("content"), query=_res["query"]
                ) for _res in results[query]
            ])

        return SearchToolOutputSchema(results=search_result_items, category=params.category)

    def run(self, params: SearchToolInputSchema) -> SearchToolOutputSchema:
        """
        Runs the SearxNGTool synchronously with the given parameters.

        This method creates an event loop in a separate thread to run the asynchronous operations.

        Args:
            params (SearchToolInputSchema): The input parameters for the tool, adhering to the input schema.

        Returns:
            SearchToolOutputSchema: The output of the tool, adhering to the output schema.

        Raises:
            ValueError: If the base URL is not provided.
            Exception: If the request to SearxNG fails.
        """
        return self.process_multiple_searches(params)


if __name__ == "__main__":
    search_terms = [
        "Python programming tutorial",
        "Machine learning basics",
        "Web scraping with Python"
    ]
    num_results = 10

    params = SearchToolInputSchema(queries=search_terms)
    tool = SearchTool(SearchToolConfig(max_results=num_results))
    results = tool.run(params)
    for r in results:
        print(r)