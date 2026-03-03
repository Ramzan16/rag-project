import arxiv
import requests
import logging
import re
from config.settings import Config, config
from schemas import PaperData

logger = logging.getLogger(__name__)

class ArxivService:
    def __init__(self, config: Config = config):
        self.config = config
        self.client = arxiv.Client()

    def search(self, query):
        # Map strings from config to enum values
        sort_by = arxiv.SortCriterion.__members__.get(self.config.arxiv.sort_by)
        sort_order = arxiv.SortOrder.__members__.get(self.config.arxiv.sort_order)
        logger.info(f"Searching arXiv with query: '{query}', max_results: {self.config.arxiv.max_results}")
        query_obj = arxiv.Search(
            query=query,
            max_results=self.config.arxiv.max_results,
            sort_by=sort_by,
            sort_order=sort_order
        )
        results = list(self.client.results(query_obj))
        logger.info(f"Found {len(results)} results on arXiv")
        return results

    def stream_papers(self, results):
        """
        Yields PaperData objects with a live stream from ArXiv.
        """
        for result in results:
            # stream=True prevents downloading the body immediately
            with requests.get(result.pdf_url, stream=True) as response:
                if response.status_code == 200:
                    # Content-Length is needed by MinIO for streaming uploads
                    content_length = int(response.headers.get('Content-Length', 0))
                    
                    # Sanitize filename
                    safe_title = re.sub(r'[^\w\s\.-]', '', result.title).replace(' ', '_')
                    
                    yield PaperData(
                        id=result.entry_id,
                        title=result.title,
                        filename=f"{safe_title}.pdf",
                        authors=[author.name for author in result.authors],
                        summary=result.summary,
                        stream=response.raw,  # The raw socket stream
                        content_length=content_length
                    )

    def run_service(self, query):
        results = self.search(query)
        return self.stream_papers(results)
