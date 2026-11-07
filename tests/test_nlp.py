import pytest

spacy = pytest.importorskip("spacy")

from mp.nlp.ner import adicionar_ruler, extrair_entidades, normalizar_codigo
from mp.nlp.preprocess import limpar


@pytest.fixture(scope="module")
def nlp():
    return adicionar_ruler(spacy.blank("pt"))


def test_limpar_expande_abreviacoes_sem_quebrar_palavras_cheias():
    assert limpar("Rolam. com TEMP alta") == "rolamento com temperatura alta"
    assert limpar("temperatura   normal") == "temperatura normal"


@pytest.mark.parametrize("codigo", ["HPC-0412", "hpc-0412", "HPC0412", "brg0007"])
def test_ruler_acha_codigo_de_peca(nlp, codigo):
    ents = extrair_entidades(nlp(f"motor 3 apresenta ruído, peça {codigo}"))
    assert ents["codigo"] == normalizar_codigo(codigo)


def test_ruler_nao_confunde_numero_comum_com_codigo(nlp):
    assert extrair_entidades(nlp("motor 1234 ciclo 5678"))["codigo"] is None


def test_ruler_extrai_componente_e_sintoma(nlp):
    ents = extrair_entidades(nlp("compressor de alta apresenta vibração forte hpc-0412"))
    assert ents["componente"] == "compressor de alta"
    assert ents["sintoma"] == "vibração forte"


def test_normalizar_codigo():
    assert normalizar_codigo("hpc0412") == "HPC-0412"
    assert normalizar_codigo("HPC-0412") == "HPC-0412"
