import minio
import logging
from schemas import PaperData
from config.settings import config, Config

logger = logging.getLogger(__name__)

class FileUploadService:
    def __init__(self, config: Config = config):
        self.config = config
        self.client = minio.Minio(
            self.config.minio.endpoint,
            access_key=self.config.minio.access_key,
            secret_key=self.config.minio.secret_key,
            secure=False # Set to True if using HTTPS
        )
        self.bucket_name = self.config.minio.bucket_name
        self._ensure_bucket_exists()

    def _ensure_bucket_exists(self):
        """Checks if the bucket exists, creating it if it doesn't. (Non-dumb way)"""
        try:
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
                logger.info(f"Bucket {self.bucket_name} created")
        except Exception as e:
            logger.error(f"Error ensuring bucket exists: {e}")

    def upload_file(self, paper_data: PaperData):
        """Streams a single paper's PDF data directly into MinIO."""
        if not paper_data.stream or paper_data.content_length is None:
            logger.error(f"Paper {paper_data.title} is missing its stream or content length.")
            return

        try:
            self.client.put_object(
                bucket_name=self.bucket_name,
                object_name=paper_data.filename,
                data=paper_data.stream,        # Raw socket stream
                length=paper_data.content_length, # Required for streams
                content_type='application/pdf'
            )
            logger.info(f"Successfully uploaded: {paper_data.filename}")
        except Exception as e:
            logger.error(f"Failed to upload {paper_data.filename}: {e}")

    def run_service(self, paper_data: PaperData):
        self.upload_file(paper_data)
