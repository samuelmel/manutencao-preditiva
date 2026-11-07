"""O README promete comandos `make X`: cada X tem de existir no Makefile."""
import re

from mp.config import ROOT

MAKEFILE = (ROOT / "Makefile").read_text(encoding="utf-8")
README = (ROOT / "README.md").read_text(encoding="utf-8")


def _alvos():
    return set(re.findall(r"^([a-z][a-z0-9-]*):", MAKEFILE, flags=re.MULTILINE))


def test_todo_make_citado_no_readme_existe():
    citados = set(re.findall(r"`make ([a-z][a-z0-9-]*)", README))
    assert citados, "o README deveria citar comandos make"
    assert citados <= _alvos(), citados - _alvos()


def test_reproduce_refaz_os_passos_a_partir_do_bronze():
    m = re.search(r"^reproduce:(.*)$", MAKEFILE, flags=re.MULTILINE)
    assert m, "falta o alvo reproduce (RNF06)"
    assert {"data", "train-rul", "text", "serving"} <= set(m.group(1).split())


def test_alvos_do_phony_existem():
    phony = set(re.search(r"^\.PHONY:(.*)$", MAKEFILE, flags=re.MULTILINE).group(1).split())
    assert phony <= _alvos(), phony - _alvos()


def test_makefile_sem_crlf():
    # CRLF fez uma edição automática falhar em silêncio; o make também reclama de ^M em receitas
    assert b"\r\n" not in (ROOT / "Makefile").read_bytes()
