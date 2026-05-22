from pathlib import Path

from ruamel.yaml import YAML


def main() -> None:
    yaml = YAML()
    yaml.preserve_quotes = True

    source = Path("/config/config.yml")
    candidate = Path("/config/config.camera-detector-test.yml")
    with source.open() as source_file:
        config = yaml.load(source_file)

    camera_name = next(iter(config["cameras"]))
    config["cameras"][camera_name].setdefault("detect", {})["detector"] = "remote_hailo8"

    with candidate.open("w") as candidate_file:
        yaml.dump(config, candidate_file)

    print(f"{camera_name} -> {candidate}")


if __name__ == "__main__":
    main()
