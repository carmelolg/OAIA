"""
Generic service module for interacting with APIs.

This module exposes `GenericHttpService` as a reusable base class.
Concrete services should inherit from it and use the `get` method defined
on the class.
"""
from typing import Any
import requests
import json


class GenericHttpService:
    """Base HTTP service providing a reusable GET implementation.

    Subclasses can reuse the concrete `get` implementation below.
    """

    @staticmethod
    def get(api_host: str, api_path: str, fallback_path: str) -> Any | None:
        """
        Makes a GET request to a specified API and falls back to a local file on failure.

        Args:
            api_host (str): The base URL of the API.
            api_path (str): The endpoint to make the request to.
            fallback_path (str): The path to the local JSON file to use as a fallback.

        Returns:
            dict: The JSON response from the API or the content of the fallback file.
        """
        if api_host and api_path:
            try:
                response = requests.get(f"{api_host}/{api_path}")
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException:
                # Fallback to local file if the API request fails
                pass

        try:
            with open(fallback_path) as json_file:
                return json.load(json_file)
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def post(self, api_host: str, api_path: str, fallback_path: str) -> Any:
        """Placeholder for POST method implementation."""
        raise NotImplementedError("POST method is not implemented yet.")

    def put(self, api_host: str, api_path: str, fallback_path: str) -> Any:
        """Placeholder for PUT method implementation."""
        raise NotImplementedError("PUT method is not implemented yet.")

    def delete(self, api_host: str, api_path: str, fallback_path: str) -> Any:
        """Placeholder for DELETE method implementation."""
        raise NotImplementedError("DELETE method is not implemented yet.")

    def patch(self, api_host: str, api_path: str, fallback_path: str) -> Any:
        """Placeholder for PATCH method implementation."""
        raise NotImplementedError("PATCH method is not implemented yet.")

    def options(self, api_host: str, api_path: str, fallback_path: str) -> Any:
        """Placeholder for OPTIONS method implementation."""
        raise NotImplementedError("OPTIONS method is not implemented yet.")
