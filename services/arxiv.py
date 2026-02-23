from config.settings import Config, config
from arxiv import *


class ArxivService:
    def __init__(self, conifg: Config = config):
        self.config = config
        self.client = Client()

    def search(self, query):
        query = Search(
            query=query,
            max_results=2,
            sort_by=SortCriterion.Relevance,
            sort_order=SortOrder.Descending
        )
        results = self.client.results(query)

    def download_pdfs(self, results):
        for result in results:
            result.download_pdf(
                dirpath=config.file_dir
            )