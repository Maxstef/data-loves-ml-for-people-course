from pathlib import Path
from bs4 import BeautifulSoup
import urllib.parse
import requests
import logging

def download_file_public(file_id, output_path, chunk_size=65536):
    """
    Download a publicly accessible file from Google Drive, including large files
    that trigger the "Google Drive can't scan this file for viruses" warning.

    The function automatically detects the warning page, extracts the necessary
    hidden tokens (confirm, uuid, etc.), reconstructs the download URL, and
    streams the file to disk in chunks.

    Parameters
    ----------
    file_id : str
        Google Drive file ID of the public file to download.
    output_path : str or pathlib.Path
        Destination path to save the downloaded file. Can be relative or absolute.
    chunk_size : int, optional
        Number of bytes per chunk when streaming the file. Default is 65536 bytes.

    Raises
    ------
    requests.HTTPError
        If the HTTP request fails (e.g., invalid file ID, access denied).

    Notes
    -----
    - Only works for publicly accessible files.
    - Uses a persistent session to handle cookies and tokens.
    - Streams data in chunks to avoid loading large files into memory.
    - May fail if the file is private or restricted.
    """

    # Convert output_path to an absolute Path object
    output_path = Path(output_path).resolve()

    # Ensure the parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info(f"Starting download of Google Drive file ID: {file_id}")
    logging.info(f"Output will be saved to: {output_path}")

    # Base Google Drive public download endpoint
    BASE_URL = "https://drive.google.com/uc?export=download"
    CONFIRM_URL = "https://drive.usercontent.google.com/download?"

    # Use a session to persist cookies between requests
    session = requests.Session()

    # First request: may return the file directly or a warning page
    response = session.get(BASE_URL, params={"id": file_id}, stream=True)
    
    # Check for potential authorization issues
    if "accounts.google.com" in response.url:
        logging.warning(
            "The file may not be publicly accessible. Download might fail or produce an incomplete file."
        )

    # Handle the "file too large for virus scan" warning page
    elif "<input type=\"hidden\" name=\"confirm\"" in response.text:
        logging.warning(
            "Large file detected, handling Google Drive virus scan warning page and downloading file..."
        )
        soup = BeautifulSoup(response.text, "html.parser")
        inputs = soup.find_all("input", {"type": "hidden"})

        # Build a dictionary of all hidden inputs
        params = {i["name"]: i["value"] for i in inputs if i.get("name") and i.get("value")}

        # Always ensure mandatory parameters are present
        params.update({"id": file_id, "export": "download", "authuser": "0"})

        # Construct final download URL
        download_url = CONFIRM_URL + urllib.parse.urlencode(params)
        logging.info(f"Resolved confirmed download URL: {download_url}")

        # Send second request to actual file URL
        response = session.get(download_url, stream=True)

    logging.info("Downloading file in chunks...")

    # Write content to file
    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
    logging.info("Download completed successfully.")

