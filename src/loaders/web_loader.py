from typing import List

import bs4
from langchain_community.document_loaders import WebBaseLoader


class CustomWebLoader:
    def __init__(self, web_paths: List[str], css_classes: List[str]):
        """Initializes the CustomWebLoader with web paths and CSS classes.

        Args:
            web_paths (List[str]): List of URLs to load content from.
            css_classes (List[str]): List of CSS classes to filter the content.
        """
        self.loader = WebBaseLoader(
            web_paths=web_paths,
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(class_=css_classes)
            )
        )

    def load(self):
        """Loads the web content based on the initialized web paths and CSS classes.

        Returns:
            List[Document]: List of Document objects containing the loaded web content.
        """
        return self.loader.load()
