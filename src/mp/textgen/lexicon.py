"""Léxico das OS sintéticas: componentes, prefixos de código de peça e sintomas por gravidade."""
from __future__ import annotations

GRAVIDADES = ["baixa", "média", "alta", "crítica"]

# componente -> (prefixo do código de peça, formas de escrita)
COMPONENTES: dict[str, tuple[str, list[str]]] = {
    "compressor de alta": ("HPC", ["compressor de alta", "compressor alta pressão", "compressor AP"]),
    "turbina de alta": ("HPT", ["turbina de alta", "turbina alta pressão", "turbina AP"]),
    "compressor de baixa": ("LPC", ["compressor de baixa", "compressor baixa pressão"]),
    "turbina de baixa": ("LPT", ["turbina de baixa", "turbina baixa pressão"]),
    "ventoinha": ("FAN", ["ventoinha", "fan", "ventilador principal"]),
    "câmara de combustão": ("CMB", ["câmara de combustão", "camara de combustao", "combustor"]),
    "rolamento": ("BRG", ["rolamento", "mancal", "rolamento do eixo"]),
    "bomba de óleo": ("OIL", ["bomba de óleo", "bomba de lubrificação", "bomba do óleo"]),
}

# sintomas por gravidade (índice em GRAVIDADES)
SINTOMAS: list[list[str]] = [
    [  # baixa
        "leve ruído", "pequena variação de temperatura", "inspeção de rotina sem anomalia",
        "pequeno desgaste visual", "vibração dentro do normal", "leitura estável",
        "sujeira superficial", "folga mínima",
    ],
    [  # média
        "vibração acima do normal", "aumento gradual de temperatura", "pequeno vazamento de óleo",
        "leitura de pressão oscilando", "ruído intermitente", "desgaste visível",
        "consumo de óleo elevado", "folga excessiva",
    ],
    [  # alta
        "vibração forte", "temperatura elevada constante", "vazamento de óleo contínuo",
        "queda de pressão", "ruído metálico", "desgaste avançado",
        "alarme de temperatura recorrente", "perda de rendimento",
    ],
    [  # crítica
        "falha total", "superaquecimento", "parada súbita", "fumaça visível",
        "trinca visível", "vibração severa com ruído metálico", "perda total de pressão",
        "travamento do eixo",
    ],
]

TEMPLATES = [
    "Motor {u}: {s} em {c} ({k}).",
    "{c} {k} apresenta {s}.",
    "Operador relata {s} em {c}, peça {k}.",
    "OS {k} - {s}. Verificar {c}.",
    "{s} / {c} / cód {k}",
    "Durante ronda, {s} em {c} do motor {u}. Peça {k}.",
    "Motor {u} ciclo {y}: {c} com {s}, trocar {k}?",
    "Chamado: {s} no {c}. Ref. {k}.",
]

ABREVIACOES = {
    "temperatura": "temp", "rolamento": "rolam.", "vibração": "vibr.", "compressor": "comp.",
    "vazamento": "vaz.", "pressão": "pres.", "turbina": "turb.", "inspeção": "insp.",
}
