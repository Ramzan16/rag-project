from config.settings import Config, config
from arxiv import *
import urllib.request
from pathlib import Path
import re

class ArxivService:
    def __init__(self, config: Config = config):
        self.config = config
        self.client = Client()

    def search(self, query):
        #Map strings from config to enum values
        sort_by =SortCriterion.__members__.get(self.config.arxiv.sort_by)
        sort_order = SortOrder.__members__.get(self.config.arxiv.sort_order)
        query = Search(
            query=query,
            max_results=self.config.arxiv.max_results,
            sort_by=sort_by,
            sort_order=sort_order
        )
        results = self.client.results(query)
        return results

    def download_pdfs(self, results):
    # Ensure the directory exists
        download_path = Path(self.config.file_dir)
        download_path.mkdir(
            parents=True,
            exist_ok=True)

        for result in results:
            title = re.sub(r'[^a-zA-Z0-9\s_]', '', result.title).split('/')[-1]
            filename = f"{title}.pdf"
            filepath = download_path / filename
            print(f"Downloading {filename}")
            urllib.request.urlretrieve(result.pdf_url,
            filepath)

    def run_service(self, query):
        results = self.search(query)
        self.download_pdfs(results)