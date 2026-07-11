# -*- coding: utf-8 -*-

import os
import sys

#限制BLAS/LAPACK線程數為1，防止 Windows 多核心CPU線程競爭導致系統卡死
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# RBF切換：可設定為 "simple" (無 HPO)、"hpo" (快速 HPO)、"clustered_knn" (分群 KNN) 或 "mlp" (Optuna MLP)
# 若環境變數已設定則優先採用，否則預設使用 "simple"
if "ESA_RBF_MODE" not in os.environ:
    os.environ["ESA_RBF_MODE"] = "simple"


import csv
import time
import warnings
warnings.filterwarnings("ignore")

#自動檢測並安裝 rich 套件
try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
    from rich.live import Live
    from rich.panel import Panel
except ImportError:
    import subprocess
    print("偵測到未安裝 rich 套件，正在自動為您安裝...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "rich"])
        from rich.console import Console
        from rich.table import Table
        from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
        from rich.live import Live
        from rich.panel import Panel
    except Exception as e:
        print(f"自動安裝 rich 失敗，請手動執行 pip install rich。錯誤資訊: {e}")
        sys.exit(1)

import numpy as np
import brenchmarks.functions as bf
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
sys.stdout.reconfigure(encoding='utf-8')

from Qagent import Qlearning
from database import Database
from strategy.DE_strategy import DE_Strategy
from strategy.SLS_strategy import SLS_Strategy
from strategy.TRLS_Strategy import TRLS_Strategy
from strategy.crossover_strategy import crossover_strategy
from strategy.LBFGS_Strategy import LBFGS_Strategy

import RBF
#保存原有的 RBF 初始化函數
orig_init = RBF.RBF.__init__

def optimized_init(self, beta_candidates=None, lam_candidates=None, use_poly_tail=True, poly_tail_min_ratio=1.2, *args, **kwargs):
    if beta_candidates is None:
        #beta 候選倍率
        beta_candidates = np.array([0.1, 0.3, 0.7, 1.5])
    if lam_candidates is None:
        #正則化係數
        lam_candidates = np.array([1e-8, 1e-4])

    try:
        orig_init(self, beta_candidates=beta_candidates, lam_candidates=lam_candidates,
                  use_poly_tail=use_poly_tail, poly_tail_min_ratio=poly_tail_min_ratio, *args, **kwargs)
    except TypeError:
        orig_init(self, *args, **kwargs)

#在記憶體中替換 RBF 初始化方法
RBF.RBF.__init__ = optimized_init
# ==============================================================================

#實驗設定區
USE_LBFGS = False          #False: 使用論 4策略版; True: 使用 5策略版
#只跑指定 surrogate-mode 與 agent 的組合
RUN_CONFIGS = [
    ("hpo", ["qlearning", "ts", "ucb", "random", "seq", "alter"]),
    ("clustered_knn", ["qlearning"]),
    ("mlp", ["qlearning"]),
]

def make_agent(agent_type, n_actions, rng):
    """根據 agent_type 建立對應的 Agent 實例"""
    if agent_type == "qlearning":
        from Qagent import Qlearning
        agent = Qlearning(alpha=0.1, gamma=0.9, T=1.0, rng=rng)
        agent.qtable = np.full((8, n_actions), 1.0 / n_actions)
        return agent
    elif agent_type == "ts":
        from TSagent import DiscountedTS
        return DiscountedTS(n_actions=n_actions, decay=0.98, rng=rng)
    elif agent_type == "ucb":
        from UCBagent import UCB1Selector
        return UCB1Selector(n_actions=n_actions, c=0.5, rng=rng)
    elif agent_type in ["random", "seq", "alter"]:
        return None
    else:
        raise ValueError(f"未知的 AGENT_TYPE: '{agent_type}'")


def run_esa(dim, fit_func, func_name, agent_type="qlearning", max_fes=1000, seed=42):
    rng = np.random.default_rng(seed)

    bounds_dict = {
        "sphere":           [-100, 100],
        "ellipsoid":        [-5.12, 5.12],
        "simple_ellipsoid": [-5.12, 5.12],
        "rosenbrock":       [-2.048, 2.048],
        "ackley":           [-32.768, 32.768],
        "griewank":         [-600, 600],
        "srr":              [-5, 5],
        "rhc1":             [-5, 5],
        "rhc2":             [-5, 5],
    }
    lim = bounds_dict.get(func_name, [-100, 100])
    bounds = np.array([lim] * dim)
    init_samples = 150 if dim == 100 else 100

    DB = Database(dim, bounds)
    fes = DB.lhs(init_samples, fit_func)
    global_best_x, global_best_y = DB.getbest()

    #裝載策略
    if USE_LBFGS:
        strategies = [
            DE_Strategy(lb=bounds[:, 0], ub=bounds[:, 1], rng=rng),
            SLS_Strategy(lb=bounds[:, 0], ub=bounds[:, 1], rng=rng),
            crossover_strategy(lb=bounds[:, 0], ub=bounds[:, 1], rng=rng),
            TRLS_Strategy(lb=bounds[:, 0], ub=bounds[:, 1], rng=rng),
            LBFGS_Strategy(lb=bounds[:, 0], ub=bounds[:, 1], rng=rng),
        ]
        n_actions = 5
    else:
        strategies = [
            DE_Strategy(lb=bounds[:, 0], ub=bounds[:, 1], rng=rng),
            SLS_Strategy(lb=bounds[:, 0], ub=bounds[:, 1], rng=rng),
            crossover_strategy(lb=bounds[:, 0], ub=bounds[:, 1], rng=rng),
            TRLS_Strategy(lb=bounds[:, 0], ub=bounds[:, 1], rng=rng),
        ]
        n_actions = 4

    # 建立 Agent
    Agent = make_agent(agent_type, n_actions, rng)

    #取得初始狀態
    if Agent is not None and hasattr(Agent, "get_initial_qtable"):
        current_state = Agent.get_initial_qtable()
    else:
        current_state = None

    init_T = 1.0
    min_T = 0.1
    step_count = 0

    while fes < max_fes:
        # 溫度衰減（僅 Q-learning 且 Agent 存在時生效）
        if Agent is not None and hasattr(Agent, "update_T"):
            progress_ratio = fes / max_fes
            next_T = max(min_T, init_T * (1.0 - progress_ratio))
            Agent.update_T(next_T)

        # 動作選擇
        if agent_type in ["qlearning", "ts", "ucb"] and Agent is not None:
            action = Agent.select(current_state) if hasattr(Agent, "select") else int(Agent.select_action(current_state))
        elif agent_type == "random":
            action = rng.choice(n_actions)
        elif agent_type == "seq":
            action = step_count % n_actions
        elif agent_type == "alter":
            action = 0 if step_count % 2 == 0 else 1
        else:
            action = 0

        selected_strategy = strategies[action]
        D_new = selected_strategy.strategy(DB, fit_func)

        is_success = False
        n_evals = 0
        for x_new, y_new in D_new:
            if fes >= max_fes:
                break
            DB.add_sample(x_new, float(y_new))
            fes += 1
            n_evals += 1
            if float(y_new) < global_best_y:
                is_success = True
                global_best_y = float(y_new)

        reward = (1.0 / n_evals) if (is_success and n_evals > 0) else 0.0

        #Agent update
        if Agent is not None:
            if hasattr(Agent, "next_state"):
                next_state = Agent.next_state(action, is_success)
                Agent.q_update(current_state, action, reward, next_state)
                current_state = next_state
            else:
                Agent.update(current_state, action, reward, None)

        step_count += 1

    _, final_best_y = DB.getbest()
    return final_best_y


#任務執行器
def single_task_worker(args):
    func_name, dim, agent_type, run_idx, seed = args
    fit_func = getattr(bf, func_name)
    if func_name in ["srr", "rhc1", "rhc2"]:
        fit_func = fit_func(dim)

    t_start = time.perf_counter()
    best_y = run_esa(dim, fit_func, func_name, agent_type=agent_type, max_fes=1000, seed=seed)
    t_elapsed = time.perf_counter() - t_start
    return func_name, agent_type, run_idx, best_y, t_elapsed



def build_table(available_funcs, table_data, dim, runs, agent_type):
    table = Table(
        title=f"ESA {dim}D 實驗儀表板 | Agent: [{agent_type.upper()}] | {runs} 次",
        header_style="bold magenta"
    )
    table.add_column("測試函數", style="cyan", width=12)
    table.add_column("Mean ± Std", style="yellow", justify="center", width=30)
    table.add_column("單次均時", style="green", justify="center", width=18)
    table.add_column("運行狀態", style="bold", justify="center", width=16)

    for f in available_funcs:
        d = table_data[f]
        table.add_row(f, d["val"], d["time"], d["status"])
    return table


#平行實驗控制
def run_progressive_experiments(dims=[30], runs=5, agent_types=["qlearning"]):
    console = Console()
    max_cores = max(1, os.cpu_count() - 1)

    if not isinstance(dims, list):
        dims = [dims]
    if not isinstance(agent_types, list):
        agent_types = [agent_types]

    log_dir = "experiment_logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    agents_str = "_".join(agent_types)
    dims_str = "_".join(map(str, dims))
    mode_str = os.environ.get("ESA_RBF_MODE", "simple")
    history_csv = os.path.join(log_dir, f"results_{timestamp}_{agents_str}_{dims_str}D_LBFGS{int(USE_LBFGS)}_{mode_str}.csv")
    latest_csv  = "results.csv"

    for csv_file in [history_csv, latest_csv]:
        if not os.path.exists(csv_file) or os.path.getsize(csv_file) == 0:
            with open(csv_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Function", "Dimension", "Agent", "Run", "Best_Y", "Time_Taken", "Surrogate_Mode"])

    for agent_type in agent_types:
        for dim in dims:
            candidate_funcs = ["simple_ellipsoid", "ellipsoid", "rosenbrock", "ackley", "griewank", "srr", "rhc1", "rhc2"]
            available_funcs = []
            test_x = np.zeros(dim)

            for name in candidate_funcs:
                if hasattr(bf, name):
                    func = getattr(bf, name)
                    try:
                        if name in ["srr", "rhc1", "rhc2"]:
                            actual_func = func(dim)
                            actual_func(test_x)
                        else:
                            func(test_x)
                        available_funcs.append(name)
                    except Exception:
                        pass

            table_data = {}
            for f in available_funcs:
                table_data[f] = {
                    "val":    "[grey50]Waiting...[/grey50]",
                    "time":   "[grey50]Waiting...[/grey50]",
                    "status": "[grey50]Pending[/grey50]",
                }

            all_tasks = []
            results_by_func = {func: [] for func in available_funcs}

            for func_name in available_funcs:
                for r in range(1, runs + 1):
                    seed = 42 + r
                    all_tasks.append((func_name, dim, agent_type, r, seed))

            progress = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("已用時間:"),
                TimeElapsedColumn(),
                TextColumn("剩餘時間:"),
                TimeRemainingColumn()
            )
            task_id = progress.add_task("", total=len(all_tasks))

            def make_layout():
                from rich.console import Group
                t_render = build_table(available_funcs, table_data, dim, runs, agent_type)
                return Panel(Group(t_render, progress), title="", border_style="blue")

            with ProcessPoolExecutor(max_workers=max_cores) as executor:
                future_to_task = {executor.submit(single_task_worker, task): task for task in all_tasks}

                with Live(make_layout(), refresh_per_second=4, console=console) as live:
                    for future in as_completed(future_to_task):
                        try:
                            func_name, agent_name, run_idx, best_y, t_taken = future.result()

                            for csv_file in [history_csv, latest_csv]:
                                with open(csv_file, "a", newline="", encoding="utf-8") as f:
                                    writer = csv.writer(f)
                                    writer.writerow([func_name, dim, agent_name, run_idx, best_y, t_taken, os.environ.get("ESA_RBF_MODE", "simple")])

                            results_by_func[func_name].append({"best_y": best_y, "t_taken": t_taken})

                            total_completed = len(results_by_func[func_name])
                            if total_completed < runs:
                                table_data[func_name]["status"] = f"[bold yellow]Running ({total_completed}/{runs})[/bold yellow]"
                            else:
                                ys = [r["best_y"] for r in results_by_func[func_name]]
                                ts = [r["t_taken"] for r in results_by_func[func_name]]
                                table_data[func_name]["val"]    = f"{np.mean(ys):.3e} ± {np.std(ys):.3e}"
                                table_data[func_name]["time"]   = f"{np.mean(ts):.2f}s ± {np.std(ts):.2f}s"
                                table_data[func_name]["status"] = "[bold green]Done[/bold green]"

                            progress.advance(task_id)
                            live.update(make_layout())

                        except Exception as exc:
                            task_args = future_to_task[future]
                            console.print(f"[bold red]任務 {task_args} 產生異常: {exc}[/bold red]")

    console.print("\n[bold green]所有選定維度與 Agent 的實驗全部完畢！[/bold green]", style="bold green")
    console.print(f"- 最新副本已寫入: [cyan]{os.path.abspath(latest_csv)}[/cyan]")
    console.print(f"- 歷史紀錄存檔於: [cyan]{os.path.abspath(history_csv)}[/cyan]")


if __name__ == "__main__":

    RUN_CONFIGS = [
        ("hpo", ["qlearning"]),
    ]
    
    for mode, agent_types in RUN_CONFIGS:
        print(f"\n" + "="*50)
        print(f"開始執行實驗，Surrogate-Mode: [{mode.upper()}]  Agents: {agent_types}")
        print("="*50)
        
        #設定當前與子進程繼承的代理模型模式
        os.environ["ESA_RBF_MODE"] = mode
        

        run_progressive_experiments(dims=[50, 100], runs=20, agent_types=agent_types)
