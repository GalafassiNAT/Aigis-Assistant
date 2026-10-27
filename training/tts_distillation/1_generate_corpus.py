"""Gerador de Corpus Fonético Balanceado para Destilação do TTS da Aigis.

Gera frases variadas em português cobrindo:
- Saudações e respostas cotidianas da persona Aigis
- Comandos de sistema operacional, arquivos e automação
- Termos técnicos de programação, Python, Git e infraestrutura
- Números, datas, horários e fonemas variados
"""

import os
from pathlib import Path

BASE_PHRASES = [
    # Saudações e Persona Aigis
    "Olá, mestre! Como posso ajudar você hoje?",
    "Sistemas operacionais prontos e totalmente calibrados.",
    "Estou à sua disposição para qualquer tarefa técnica.",
    "Bom dia! Todos os módulos de vigilância e auxílio estão ativos.",
    "Boa tarde! Processando suas instruções em tempo real.",
    "Boa noite, mestre. Deseja realizar mais alguma automação antes de encerrar?",
    "Comando compreendido. Executando agora mesmo.",
    "Entendido perfeitamente. Iniciando a rotina solicitada.",
    "Tudo certo por aqui. Monitorando os recursos do sistema.",
    "Operação concluída com sucesso.",
    "Identifiquei um pequeno ajuste necessário, mas já resolvi.",
    "Prontidão confirmada. Aguardo seu próximo comando.",
    "Sim, estou te ouvindo claramente pelo microfone.",
    "Não hesite em me chamar caso precise de assistência no código.",
    "Aigis online. Todos os nós neurais respondendo adequadamente.",

    # Comandos de Sistema, Arquivos e Automação
    "Criando o arquivo de rascunho no diretório solicitado.",
    "Arquivo salvo com sucesso na pasta de projetos.",
    "Lendo o conteúdo do documento para análise estrutural.",
    "Apagando os arquivos temporários da compilação anterior.",
    "Listando os diretórios disponíveis no seu ambiente de trabalho.",
    "Executando script em segundo plano no PowerShell.",
    "O processo foi finalizado com código de retorno zero.",
    "A pasta de destino foi criada e configurada corretamente.",
    "Copiando os backups para a pasta segura no disco local.",
    "Verificando permissões de leitura e gravação no sistema de arquivos.",

    # Engenharia de Software, Git e Código
    "Executando os testes unitários da aplicação.",
    "Todas as asserções passaram sem nenhum erro reportado.",
    "Realizando o commit das alterações na ramificação principal.",
    "O repositório Git foi sincronizado com o servidor remoto.",
    "Identifiquei um conflito de mesclagem na linha quarenta e dois.",
    "Refatorando a função assíncrona para eliminar bloqueios de entrada e saída.",
    "Iniciando o servidor local na porta oitenta e oitenta.",
    "A requisição HTTP retornou código duzentos com sucesso.",
    "Carregando o modelo quantizado na memória da placa gráfica.",
    "O pipeline de inferência foi otimizado para latência mínima.",

    # Números, Cálculos e Horários
    "São exatamente quatorze horas e trinta minutos.",
    "A taxa de amostragem foi definida em vinte e dois mil e cinquenta hertz.",
    "O arquivo possui cento e vinte e oito kilobytes de tamanho.",
    "Foram processados quinhentos e doze tokens em trezentos milissegundos.",
    "A temperatura da unidade de processamento gráfico está em sessenta e dois graus Celsius.",
    "A porcentagem de uso de memória permaneceu em quarenta e cinco por cento.",
    "Temos três tarefas ativas na fila assíncrona de reprodução.",
    "O ciclo de processamento levou menos de cinquenta milissegundos por frase.",
    "Primeira verificação concluída, partindo para a segunda etapa.",
    "Totalizando dezessete mil linhas de código no projeto.",

    # Perguntas, Diálogos e Interações Gerais
    "Você gostaria que eu abrisse o editor de código agora?",
    "Deseja fazer o deploy dessas alterações no ambiente de teste?",
    "Posso resumir as anotações do dia para você se desejar.",
    "Tudo ótimo por aqui, obrigada por perguntar! E com você?",
    "Como vai a rotina de desenvolvimento hoje?",
    "Estou pronta para começar quando você quiser.",
    "Excelente trabalho! O código ficou limpo e direto ao ponto.",
    "Combinado! Fico no aguardo da sua próxima instrução.",
    "Pode falar, estou acompanhando com atenção.",
    "Sistemas operacionais! Olá mestre! Estou te ouvindo pelo microfone.",
]


def generate_corpus(output_path: str = "corpus_ptbr.txt", min_total: int = 200) -> list[str]:
    """Expande as frases base garantindo riqueza e número de amostras."""
    variations = [
        "Mestre, ",
        "Atenção: ",
        "Relatório: ",
        "Status: ",
        "Confirmado: ",
        "Notificação: ",
    ]

    corpus = list(BASE_PHRASES)

    for prefix in variations:
        for phrase in BASE_PHRASES:
            if len(corpus) >= min_total:
                break
            combined = f"{prefix}{phrase[0].lower()}{phrase[1:]}"
            corpus.append(combined)
        if len(corpus) >= min_total:
            break

    # Salva no arquivo
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for line in corpus:
            f.write(line.strip() + "\n")

    print(f"[Corpus] {len(corpus)} frases geradas e salvas em: {output_path}")
    return corpus


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    out_file = script_dir / "corpus_ptbr.txt"
    generate_corpus(str(out_file), min_total=250)
