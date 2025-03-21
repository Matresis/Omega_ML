with open("config.ini", "r", encoding="utf-8-sig") as f:
    content = f.read()

with open("config.ini", "w", encoding="utf-8") as f:
    f.write(content)