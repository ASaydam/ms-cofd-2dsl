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


def test_relaxed_template_changes_only_the_solver_settings():
    # a case run with second-order-relaxed converges to the same solution: every file but system/fvSolution is equal
    from utils import REPO_ROOT

    templates = REPO_ROOT / 'openfoam' / 'case-templates'
    relaxed, plain = templates / 'second-order-relaxed' / '0.25x', templates / 'second-order' / '0.25x'
    files = sorted(p.relative_to(relaxed).as_posix() for p in relaxed.rglob('*') if p.is_file())
    assert files == sorted(p.relative_to(plain).as_posix() for p in plain.rglob('*') if p.is_file())
    different = [name for name in files if (relaxed / name).read_bytes() != (plain / name).read_bytes()]
    assert different == ['system/fvSolution']
    text = (relaxed / 'system' / 'fvSolution').read_text()
    assert 'consistent      no;' in text and 'p               0.3;' in text and 'U               0.7;' in text
