import pytest

from datus.utils.ontology_utils import is_ontology_file_path


@pytest.mark.parametrize(
    "path",
    [
        "subject/semantic_models/ontology/flights/ontology.yml",
        "/project/subject/semantic_models/ontology/flights/mapping.yaml",
        "subject/semantic_models/ontology/flights/nested/custom.YAML",
        "./subject/semantic_models/ontology/flights/./ontology.yml",
    ],
)
def test_ontology_files_include_new_and_custom_names(path):
    assert is_ontology_file_path(path)


@pytest.mark.parametrize(
    "path",
    [
        "subject/semantic_models/flights/flights.yml",
        "subject/semantic_models/ontology/flights/../../flights/model.yml",
        "subject/semantic_models/ontology/ontology.yml",
        "subject/semantic_models/ontology/flights/notes.txt",
        "other/ontology.yml",
    ],
)
def test_unrelated_or_escaped_paths_remain_ordinary_files(path):
    assert not is_ontology_file_path(path)
