import fire
from evaluation.eval import single_eval, eval_DRAGBENCH

if __name__ == "__main__":
    fire.Fire({
        "single": single_eval,
        "benchmark": eval_DRAGBENCH,
    })