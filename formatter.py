import os
from pprint import pprint
from bs4 import BeautifulSoup


def ret_dirs(path):
    dirs = set()
    flag = False
    for file in os.listdir(path):
        if os.path.isdir(os.path.join(path, file)):
            dirs.update(ret_dirs(os.path.join(path, file)))
        else:
            flag = True
    if flag:
        dirs.update([path])
    return dirs

dirs = set()
for path in ['Оптика', 'Колебания и волны', 'Квантовая физика', 'Астрономия']:
    dirs.update(ret_dirs(path))

for dir in dirs:
    for file in os.listdir(dir):
        if os.path.isfile(os.path.join(dir, file)):
            if file[-1] == 't':
                with open(os.path.join(dir, file), 'r') as f_html, open(os.path.join(dir, file) + '.txt', 'w') as f_txt:
                    html_content = BeautifulSoup(f_html.read(), 'html.parser').text
                    f_txt.write(html_content)
                content = []
                with open(os.path.join(dir, file) + '.txt', 'r') as f:
                    for line in f:
                        if not line.startswith('Рис.') and len(line) > 5 and line != "Введение\n":
                            content.append(line.replace(' ', ''))
                    print(content)
                with open(os.path.join(dir, file) + '.txt', 'w') as f:
                    f.writelines(content)
