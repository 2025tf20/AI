"""AWS Lambda entrypoint for the contract-clause generator."""
from Summary import Summary


# Reuse a single client across invocations for performance.
summarizer = Summary()


def handler(event, context):
    """
    Lambda handler.

    Expects event like: {"clauses": "1. ...\n2. ..."}
    Returns: {"statusCode": int, "body": any}
    """
    clauses = event.get("clauses") if isinstance(event, dict) else None
    if not clauses or not str(clauses).strip():
        return {"statusCode": 400, "body": "clauses is required"}

    try:
        result = summarizer.ask(str(clauses))
        return {"statusCode": 200, "body": result}
    except Exception as exc:  # noqa: BLE001
        # Minimal error surface; expand logging as needed.
        return {"statusCode": 500, "body": f"error: {exc}"}


# For local smoke-testing: python lambda_handler.py
if __name__ == "__main__":
    sample_event = {
        "clauses": "1. 실내 흡연 가능\n2. 고양이 키우기 허용\n3. 주차 자리를 보장"
    }
    print(handler(sample_event, None))
