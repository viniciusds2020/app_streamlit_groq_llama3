"""Send a content-free Langfuse trace to verify local credentials and connectivity."""
import os
import sys


def main():
    missing = [
        name
        for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")
        if not os.getenv(name)
    ]
    if missing:
        print("Missing Langfuse environment variables: " + ", ".join(missing), file=sys.stderr)
        return 2

    try:
        from langfuse import get_client

        client = get_client()
        client.auth_check()
        with client.start_as_current_observation(
            as_type="span",
            name="rag.langfuse_smoke",
            metadata={"origin": "manual_smoke_test"},
        ) as span:
            span.update(metadata={"connectivity_check": "passed"})
        client.flush()
    except Exception as exc:
        print(f"Langfuse check failed ({type(exc).__name__}). Check URL and credentials.", file=sys.stderr)
        return 1

    print("Langfuse authenticated and accepted the smoke-test trace.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
