import pytest

from openfoam.create_foam import create_foam


def make_meshes(folder, names):
    folder.mkdir(parents=True)
    for name in names:
        (folder / f"{name}.msh").write_text("mesh")


def test_template_sets(tmp_path):
    make_meshes(tmp_path / "mesh", ["case_a"])
    create_foam(tmp_path / "mesh", tmp_path / "first-order", "base", template_set="first-order")
    create_foam(tmp_path / "mesh", tmp_path / "second-order", "base")

    first = (tmp_path / "first-order" / "case_a" / "system" / "fvSchemes").read_text()
    second = (tmp_path / "second-order" / "case_a" / "system" / "fvSchemes").read_text()
    assert "div(phi,U)      bounded Gauss upwind;" in first and "linearUpwind grad(U)" in second
    assert (tmp_path / "second-order" / "case_a" / "mesh.msh").read_text() == "mesh"
    assert not (tmp_path / "second-order" / "case_a" / "init_from").exists()


def test_init_from_points_to_the_case_with_the_same_name(tmp_path):
    make_meshes(tmp_path / "mesh-base", ["case_a", "case_b"])
    make_meshes(tmp_path / "mesh-16x", ["case_a", "case_b"])
    create_foam(tmp_path / "mesh-base", tmp_path / "cases" / "base", "base")
    create_foam(tmp_path / "mesh-16x", tmp_path / "cases" / "16x", "16x", init_from=tmp_path / "cases" / "base")

    for name in ("case_a", "case_b"):
        source = (tmp_path / "cases" / "16x" / name / "init_from").read_text().strip()
        assert source == str((tmp_path / "cases" / "base" / name).resolve())


def test_init_from_needs_the_source_case(tmp_path):
    make_meshes(tmp_path / "mesh", ["case_a"])
    (tmp_path / "base").mkdir()
    with pytest.raises(FileNotFoundError):
        create_foam(tmp_path / "mesh", tmp_path / "16x", "16x", init_from=tmp_path / "base")
