"""验证所有模块语法正确"""
import sys
import os
import ast

base = r'C:\Users\zhaor\.openclaw\workspace\credit-risk-optimized'
modules = [
    'src/__init__.py',
    'src/data_processor.py',
    'src/woe_iv.py',
    'src/pd_model.py',
    'src/scorecard.py',
    'src/ecl_calculator.py',
    'train.py',
    'app.py',
]

all_ok = True
for mod in modules:
    path = os.path.join(base, mod)
    try:
        with open(path, 'r', encoding='utf-8') as f:
            ast.parse(f.read())
        print(f'  OK  {mod}')
    except SyntaxError as e:
        print(f'  FAIL {mod}: {e}')
        all_ok = False

if all_ok:
    print('\n所有模块语法验证通过!')
else:
    print('\n存在语法错误，请检查!')
    sys.exit(1)
