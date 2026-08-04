import glob
import json
from pathlib import Path


def load_file_catalog(catalog_path):
    """Load catalog entries from a JSON-compatible YAML file."""
    path = Path(catalog_path)
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)

    if not isinstance(data, list):
        raise ValueError("Catalog root must be a list of file specifications")

    required_keys = {"id", "title", "key", "loader_ref", "builder_ref"}
    for spec in data:
        if not isinstance(spec, dict):
            raise ValueError("Each catalog item must be a dictionary")
        missing = required_keys - set(spec.keys())
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(f"Catalog item is missing required keys: {missing_list}")

    return data


def spec_exists(spec, output_path):
    """Return True when a file specification is available in output_path."""
    output_path = Path(output_path)

    for rel_path in spec.get("exists_files", []):
        if (output_path / rel_path).is_file():
            return True

    for rel_glob in spec.get("exists_globs", []):
        if glob.glob((output_path / rel_glob).as_posix()):
            return True

    point_prefix = spec.get("point_prefix")
    if point_prefix:
        if list(output_path.glob(f"{point_prefix}.*.nc")):
            return True
        if list((output_path / "run_001").glob(f"{point_prefix}.*.nc")):
            return True

    return False


def available_specs(specs, output_path):
    """Filter a list of specs down to those available in output_path."""
    return [spec for spec in specs if spec_exists(spec, output_path)]


def resolve_runtime_specs(specs, loader_registry, builder_registry):
    """Resolve loader_ref/builder_ref strings to callables from the registries.

    Raises ValueError for any unknown ref name so mismatches between the
    catalog and the registry are caught early.
    """
    runtime_specs = []
    for spec in specs:
        loader_ref = spec["loader_ref"]
        builder_ref = spec["builder_ref"]
        if loader_ref not in loader_registry:
            raise ValueError(
                f"Unknown loader_ref '{loader_ref}' for catalog entry '{spec['id']}'"
            )
        if builder_ref not in builder_registry:
            raise ValueError(
                f"Unknown builder_ref '{builder_ref}' for catalog entry '{spec['id']}'"
            )
        runtime_spec = spec.copy()
        runtime_spec["load_fn"] = loader_registry[loader_ref]
        runtime_spec["build_fn"] = builder_registry[builder_ref]
        runtime_specs.append(runtime_spec)
    return runtime_specs
