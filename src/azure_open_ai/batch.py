"""Azure OpenAI Batch Job Queueing Module"""
import io
import json
import time
from typing import Optional
from openai.types import Batch

from azure_open_ai.openai_client import OpenAIClient
from logger.logger import Logger
from utils.byte_utils import format_size


def queue_batch_job(
    jobs: list[dict],
) -> Optional[Batch]:
    """
    Queues a batch job using Azure OpenAI.

    Args:
        jobs (list[dict]): list of dictionaries containing job information

    Raises:
        RuntimeError: if the file upload fails

    Returns:
        Optional[Batch]: the batch job object if the job is queued successfully, None otherwise
    """
    if not isinstance(jobs, list) or len(jobs) <= 0:
        raise ValueError(
            "jobs must be a non-empty list of dictionaries.")

    jobs_jsonl = "\n".join(json.dumps(job) for job in jobs)
    jsonl_encoded = jobs_jsonl.encode("utf-8")

    Logger().info(f"batch file size: {format_size(len(jsonl_encoded))}")

    byte_stream = io.BytesIO(jsonl_encoded)

    Logger().info("Starting batch file upload ...")

    openai_client = OpenAIClient().get_client()

    # Upload jsonl file for batch processing
    batch_file = openai_client.files.create(
        file=(f'locomo-run-{Logger().get_run_id()}.jsonl',
              byte_stream, 'application/jsonl'),
        purpose="batch",
    )

    # Wait until the file is uploaded
    while True:
        file = openai_client.files.retrieve(batch_file.id)
        if file.status in ("processed", "error"):
            break
        Logger().info("Waiting for file to be uploaded...")
        time.sleep(10)

    if file.status == "error":
        # pylint: disable-next=broad-except
        raise RuntimeError(f"File upload failed: {file.error}")

    Logger().info("File upload succeeded.")
    Logger().info("Creating batch job ...")

    # Create a batch job
    batch_job = openai_client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )

    return batch_job
