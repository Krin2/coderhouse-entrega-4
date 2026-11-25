import pytest

from src.ingestion.loaders import MarkdownLoader, inline_code, select_lines

CODE = "line1\nline2\nline3\nline4\nline5\n"


@pytest.fixture
def code_dir(tmp_path):
    d = tmp_path / "code"
    (d / "app").mkdir(parents=True)
    (d / "app" / "main.py").write_text(CODE, encoding="utf-8")
    (d / "app" / "page.html").write_text("<p>hola</p>", encoding="utf-8")
    return d


def test_select_lines_rangos():
    assert select_lines(CODE, "2:3") == "line2\nline3"
    assert select_lines(CODE, ":2") == "line1\nline2"
    assert select_lines(CODE, "4:") == "line4\nline5"
    assert select_lines(CODE, "3") == "line3"
    assert select_lines(CODE, "1:1,5:5") == "line1\n# ...\nline5"


def test_inline_sin_code_dir_elimina_el_include():
    assert inline_code("antes {* ../../docs_src/app/main.py hl[1] *} despues", None) == "antes  despues"


def test_inline_inserta_bloque_con_lenguaje(code_dir):
    out = inline_code("{* ../../docs_src/app/main.py hl[1] *}\n{* ../../docs_src/app/page.html *}", code_dir)
    assert "```python\nline1\nline2" in out and "```html\n<p>hola</p>\n```" in out


def test_inline_respeta_ln(code_dir):
    out = inline_code("{* ../../docs_src/app/main.py ln[2:3] hl[1] *}", code_dir)
    assert "line2\nline3" in out and "line1" not in out and "line4" not in out


def test_inline_no_repite_el_mismo_fragmento(code_dir):
    text = "{* ../../docs_src/app/main.py hl[1] *}\ntexto\n{* ../../docs_src/app/main.py hl[3] *}"
    assert inline_code(text, code_dir).count("line1") == 1
    distinto = "{* ../../docs_src/app/main.py ln[1:2] *}{* ../../docs_src/app/main.py ln[4:5] *}"
    assert inline_code(distinto, code_dir).count("```python") == 2         # distinto rango => distinto fragmento


def test_inline_archivo_inexistente_se_descarta(code_dir):
    assert inline_code("a {* ../../docs_src/app/no_existe.py *} b", code_dir) == "a  b"


def test_inline_no_sale_de_code_dir(code_dir, tmp_path):
    (tmp_path / "secreto.py").write_text("PASSWORD", encoding="utf-8")
    assert "PASSWORD" not in inline_code("{* ../../docs_src/../../secreto.py *}", code_dir)


def test_loader_con_codigo(tmp_path, code_dir):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "doc.md").write_text("# Doc\n{* ../../docs_src/app/main.py hl[1] *}", encoding="utf-8")
    sin = MarkdownLoader(raw).load()[0].page_content
    con = MarkdownLoader(raw, code_dir).load()[0].page_content
    assert "line1" not in sin and "line1" in con
