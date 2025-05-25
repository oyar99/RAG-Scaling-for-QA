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
) -> Batch:
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

    if not openai_client:
        raise RuntimeError("Failed to create OpenAI client.")

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
        raise RuntimeError(f"File upload failed: {file.id}")

    Logger().info("File upload succeeded.")
    Logger().info("Creating batch job ...")

    # Create a batch job
    batch_job = openai_client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )

    return batch_job


def wait_for_batch_job_and_save_result(
    batch: Batch,
    output_file_path: str,
) -> None:
    """
    Waits for a batch job to complete and saves the result to a file.

    Args:
        batch_job (Batch): the batch job object
        output_file_path (str): the path to save the output file

    Raises:
        RuntimeError: if the batch job fails
    """
    # Polling for batch job completion
    while batch.status in ("in_progress", "validating", "finalizing"):
        Logger().info(
            f"Batch job status: {batch.status}. Waiting for completion...")
        time.sleep(120)  # Sleep for 2 minutes
        batch = retrieve_batch_job(batch.id)

    if batch.status != "completed":
        Logger().error(
            f"Batch job failed with status: {batch.status}. Please check the logs for more details.")
        raise RuntimeError(
            f"Batch job failed with status: {batch.status}. Please check the logs for more details.")

    Logger().info(
        f"Batch job completed with status: {batch.status}")
    Logger().info(
        f"Batch job output file ID: {batch.output_file_id}")

    result = retrieve_file(batch.output_file_id)

    with open(output_file_path, 'wb') as f:
        f.write(result)


def retrieve_batch_job(
    batch_job_id: str,
) -> Batch:
    """
    Retrieves a batch job using Azure OpenAI.

    Args:
        batch_job_id (str): the ID of the batch job to retrieve

    Returns:
        Batch: the batch job object if the job is retrieved successfully
    """
    openai_client = OpenAIClient().get_client()

    if not openai_client:
        raise RuntimeError("Failed to create OpenAI client.")

    return openai_client.batches.retrieve(batch_job_id)


def retrieve_file(
    file_id: Optional[str],
) -> bytes:
    """
    Retrieves a file using Azure OpenAI.

    Args:
        file_id (str): the ID of the file to retrieve

    Returns:
        bytes: the file object if the file is retrieved successfully
    """
    if not file_id:
        raise ValueError("file_id must be a non-empty string.")

    openai_client = OpenAIClient().get_client()

    if not openai_client:
        raise RuntimeError("Failed to create OpenAI client.")

    return openai_client.files.content(file_id).content
