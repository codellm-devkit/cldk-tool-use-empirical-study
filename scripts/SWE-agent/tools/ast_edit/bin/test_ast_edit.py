import tempfile
from pathlib import Path
from ast_editor import CSTRewriter


def write_temp_file(content: str) -> Path:
    path = tempfile.NamedTemporaryFile(delete=False, suffix=".py").name
    Path(path).write_text(content)
    return Path(path)


def test_single_replace():
    original = """
def greet():
    print("hello")
    print("hello")
"""
    snippet = 'print("hello")'
    replacement = 'print("hi")'

    path = write_temp_file(original)
    CSTRewriter.apply_rw(
        file_path=str(path),
        snippet=snippet,
        mode="replace",
        replacement=replacement,
        match_strategy="exact",
        replace_all=False,
    )

    result = path.read_text()
    assert result.count('print("hi")') == 1
    assert result.count('print("hello")') == 1
    print("✅ test_single_replace passed.")


def test_multi_replace():
    original = """
def greet():
    print("hello")
    print("hello")
"""
    snippet = 'print("hello")'
    replacement = 'print("hi")'

    path = write_temp_file(original)
    CSTRewriter.apply_rw(
        file_path=str(path),
        snippet=snippet,
        mode="replace",
        replacement=replacement,
        match_strategy="exact",
        replace_all=True,
    )

    result = path.read_text()
    assert result.count('print("hi")') == 2
    assert 'print("hello")' not in result
    print("✅ test_multi_replace passed.")


def test_extract_all():
    original = """
def f():
    x = 1
    x = 1
"""
    snippet = "x = 1"
    path = write_temp_file(original)

    extracted = CSTRewriter.apply_rw(
        file_path=str(path),
        snippet=snippet,
        mode="extract",
        match_strategy="exact",
        replace_all=True,
    )

    assert extracted.count("x = 1") == 2
    print("✅ test_extract_all passed.")


if __name__ == "__main__":
    test_single_replace()
    test_multi_replace()
    test_extract_all()
