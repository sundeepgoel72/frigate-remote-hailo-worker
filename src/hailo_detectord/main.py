import argparse

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the remote Hailo detector worker.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=32168, type=int)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    uvicorn.run("hailo_detectord.app:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
