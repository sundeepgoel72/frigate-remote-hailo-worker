from pathlib import Path

from ruamel.yaml import YAML


CONFIG_DIR = Path("/config")
SOURCE = CONFIG_DIR / "config.yml"
CANDIDATE = CONFIG_DIR / "config.remote-hailo.yml"
REMOTE_DIR = "/config/remote-hailo"


def main() -> None:
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)

    with SOURCE.open() as source_file:
        config = yaml.load(source_file)

    config["detectors"] = {
        "remote_hailo8": {
            "type": "deepstack",
            "api_url": "http://192.168.1.175:32168/v1/vision/detection",
            "api_timeout": 2.0,
        }
    }

    config["model"] = {
        "path": f"{REMOTE_DIR}/frigate-plus-hailo8.hef",
        "labelmap_path": f"{REMOTE_DIR}/labelmap.txt",
        "width": 640,
        "height": 640,
        "input_tensor": "nhwc",
        "input_pixel_format": "rgb",
        "input_dtype": "int",
        "model_type": "yolo-generic",
    }

    with CANDIDATE.open("w") as candidate_file:
        yaml.dump(config, candidate_file)

    print(CANDIDATE)


if __name__ == "__main__":
    main()
