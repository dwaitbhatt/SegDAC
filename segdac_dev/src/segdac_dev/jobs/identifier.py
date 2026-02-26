import uuid


def get_job_id() -> str:
    return str(uuid.uuid4())
