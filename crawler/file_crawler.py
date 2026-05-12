# crawler/file_crawler.py
import os

class FileCrawler:
    def __init__(self, root_paths):
        self.root_paths = root_paths

    def crawl(self):
        files = []
        for root_path in self.root_paths:
            for root, dirs, filenames in os.walk(root_path):
                for f in filenames:
                    files.append(os.path.join(root, f))
        return files
