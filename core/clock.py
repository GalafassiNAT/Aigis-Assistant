import datetime


def get_now_ptbr() -> str:
    now = datetime.datetime.now()
    dias = [
        "Segunda-feira",
        "Terça-feira",
        "Quarta-feira",
        "Quinta-feira",
        "Sexta-feira",
        "Sábado",
        "Domingo",
    ]
    meses = [
        "janeiro",
        "fevereiro",
        "março",
        "abril",
        "maio",
        "junho",
        "julho",
        "agosto",
        "setembro",
        "outubro",
        "novembro",
        "dezembro",
    ]
    dia_sem = dias[now.weekday()]
    mes_nome = meses[now.month - 1]
    hora = now.strftime("%H:%M")
    return f"{dia_sem}, {now.day} de {mes_nome} de {now.year} ({hora})"


if __name__ == "__main__":
    print(get_now_ptbr())
