from core.model import load_models

bi, be = load_models()
print(len(bi["nets"]), len(be["nets"]), bi["cfg"], be["cfg"])
print(bi["meta"]["sky_vals"], bi["meta"]["seeds"], len(bi["allc"]))