# -*- coding: utf-8 -*-

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'code')))

#限制 BLAS/LAPACK 線程數為 1，防止 Windows 多核心 CPU 線程競爭導致系統卡死
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

#RBF代理模型切換
#可設定為 "simple"（無 HPO）、"hpo"、"clustered_knn"、"mlp"
#若環境變數已設定則優先採用，否則預設為 "simple"
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
sys.stdout.reconfigure(encoding='utf-8')

from Qagent import Qlearning
from database import Database
from strategy.DE_strategy import DE_Strategy
from strategy.SLS_strategy import SLS_Strategy
from strategy.TRLS_Strategy import TRLS_Strategy
from strategy.crossover_strategy import crossover_strategy
from strategy.LBFGS_Strategy import LBFGS_Strategy

import RBF
#保存原有的 RBF 函數
orig_init = RBF.RBF.__init__

def optimized_init(self, beta_candidates=None, lam_candidates=None, use_poly_tail=True, poly_tail_min_ratio=1.2, *args, **kwargs):
    if beta_candidates is None:
        #beta 候選倍率（控制 RBF kernel 的寬度）
        beta_candidates = np.array([0.1, 0.3, 0.7, 1.5])
    if lam_candidates is None:

        lam_candidates = np.array([1e-8, 1e-4])

    try:
        orig_init(self, beta_candidates=beta_candidates, lam_candidates=lam_candidates,
                  use_poly_tail=use_poly_tail, poly_tail_min_ratio=poly_tail_min_ratio, *args, **kwargs)
    except TypeError:
        #若不支援這些參數（GP 代理模型），則退而使用預設初始化
        orig_init(self, *args, **kwargs)

#以 Monkey-patch 方式在記憶體中替換 RBF 的初始化方法，注入最佳化預設參數
RBF.RBF.__init__ = optimized_init
# ================================這是分隔線=========================================

#ESA_USE_LBFGS=True  → 5 策略（新增 L-BFGS 作為第 5 個動作）
#ESA_USE_LBFGS=False → 4 策略（預DE / SLS / crossover / TRLS）
USE_LBFGS = os.environ.get("ESA_USE_LBFGS", "False").lower() == "true"


def make_agent(agent_type, n_actions, rng):
    """根據 agent_type 建立對應的 Agent 實例"""
    if agent_type == "qlearning":
        from Qagent import Qlearning
        #n_actions 動態決定 Q-table 大小，狀態數自動設為 n_actions * 2
        agent = Qlearning(alpha=0.1, gamma=0.9, T=1.0, rng=rng, n_actions=n_actions)
        return agent
    elif agent_type == "ts":
        from TSagent import DiscountedTS
        return DiscountedTS(n_actions=n_actions, decay=0.98, rng=rng)
    elif agent_type == "ucb":
        from UCBagent import UCB1Selector
        return UCB1Selector(n_actions=n_actions, c=0.5, rng=rng)
    elif agent_type in ["random", "seq", "alter"]:
        #無學習能力的基準 Agent，不需要建立實例
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
    #100D 問題使用較多初始樣本，確保代理模型品質
    init_samples = 150 if dim == 100 else 100

    DB = Database(dim, bounds)
    fes = DB.lhs(init_samples, fit_func)
    global_best_x, global_best_y = DB.getbest()

    #依環境變數決定策略組合
    #ESA_REPLACE_A2_WITH_A5=True → a5（L-BFGS）取代 a2（SLS），維持 4 策略
    #ESA_USE_LBFGS=True          → 加入 a5（L-BFGS）成為第 5 個策略
    #預設                         → 標準 4 策略（DE / SLS / crossover / TRLS）
    if os.environ.get("ESA_REPLACE_A2_WITH_A5", "False").lower() == "true":
        strategies = [
            DE_Strategy(lb=bounds[:, 0], ub=bounds[:, 1], rng=rng),
            LBFGS_Strategy(lb=bounds[:, 0], ub=bounds[:, 1], rng=rng),  #a2 改為 L-BFGS
            crossover_strategy(lb=bounds[:, 0], ub=bounds[:, 1], rng=rng),
            TRLS_Strategy(lb=bounds[:, 0], ub=bounds[:, 1], rng=rng),
        ]
    elif USE_LBFGS:
        strategies = [
            DE_Strategy(lb=bounds[:, 0], ub=bounds[:, 1], rng=rng),
            SLS_Strategy(lb=bounds[:, 0], ub=bounds[:, 1], rng=rng),
            crossover_strategy(lb=bounds[:, 0], ub=bounds[:, 1], rng=rng),
            TRLS_Strategy(lb=bounds[:, 0], ub=bounds[:, 1], rng=rng),
            LBFGS_Strategy(lb=bounds[:, 0], ub=bounds[:, 1], rng=rng),  #a5 新增 L-BFGS
        ]
    else:
        strategies = [
            DE_Strategy(lb=bounds[:, 0], ub=bounds[:, 1], rng=rng),
            SLS_Strategy(lb=bounds[:, 0], ub=bounds[:, 1], rng=rng),
            crossover_strategy(lb=bounds[:, 0], ub=bounds[:, 1], rng=rng),
            TRLS_Strategy(lb=bounds[:, 0], ub=bounds[:, 1], rng=rng),
        ]
    n_actions = len(strategies)

    #建立決策 Agent（Q-learning / TS / UCB / 無學習）
    Agent = make_agent(agent_type, n_actions, rng)

    #取得初始狀態（Q-learning 專用，其他 Agent 回傳 None）
    if Agent is not None and hasattr(Agent, "get_initial_qtable"):
        current_state = Agent.get_initial_qtable()
    else:
        current_state = None

    init_T = 1.0   #初始探索溫度
    min_T = 0.1    #最低探索溫度
    step_count = 0

    #Q-table 快照旗標，用於記錄 NFE=300 時的中間狀態
    captured_300 = False
    qtable_300 = None

    while fes < max_fes:
        #溫度線性衰減（僅Q-learning Agent生效）
        if Agent is not None and hasattr(Agent, "update_T"):
            progress_ratio = fes / max_fes
            next_T = max(min_T, init_T * (1.0 - progress_ratio))
            Agent.update_T(next_T)

        
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

        #reward
        reward = (1.0 / n_evals) if (is_success and n_evals > 0) else 0.0

        #Agent更新（Q-learning 執行 Q-update，TS/UCB 執行各自的更新方法）
        if Agent is not None:
            if hasattr(Agent, "next_state"):
                next_state = Agent.next_state(action, is_success)
                Agent.q_update(current_state, action, reward, next_state)
                current_state = next_state
            else:
                Agent.update(current_state, action, reward, None)


        if fes >= 300 and not captured_300 and Agent is not None and hasattr(Agent, "qtable"):
            qtable_300 = np.copy(Agent.qtable)
            captured_300 = True

        step_count += 1

    _, final_best_y = DB.getbest()
    #擷取NFE=1000結束時的最終Q-table
    qtable_1000 = np.copy(Agent.qtable) if (Agent is not None and hasattr(Agent, "qtable")) else None
    return final_best_y, qtable_300, qtable_1000


#任務執行器：接收任務參數，執行單次 ESA 並回傳結果
def single_task_worker(args):
    func_name, dim, agent_type, run_idx, seed = args
    fit_func = getattr(bf, func_name)
    if func_name in ["srr", "rhc1", "rhc2"]:
        fit_func = fit_func(dim)

    t_start = time.perf_counter()
    best_y, qtable_300, qtable_1000 = run_esa(dim, fit_func, func_name, agent_type=agent_type, max_fes=1000, seed=seed)
    t_elapsed = time.perf_counter() - t_start
    return func_name, agent_type, run_idx, best_y, t_elapsed, qtable_300, qtable_1000


def build_table(available_funcs, table_data, dim, runs, agent_type):
    """建立 rich 進度儀表板表格"""
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


#平行實驗控制：依序對各 Agent/維度組合啟動多核心平行實驗
def run_progressive_experiments(dims=[30], runs=5, agent_types=["qlearning"]):
    
    global USE_LBFGS
    USE_LBFGS = os.environ.get("ESA_USE_LBFGS", "False").lower() == "true"
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
    #歷史CSV
    history_csv = os.path.join(log_dir, f"results_{timestamp}_{agents_str}_{dims_str}D_LBFGS{int(USE_LBFGS)}_{mode_str}.csv")
    latest_csv  = "results.csv"

    for csv_file in [history_csv, latest_csv]:
        if not os.path.exists(csv_file) or os.path.getsize(csv_file) == 0:
            with open(csv_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Function", "Dimension", "Agent", "Run", "Best_Y", "Time_Taken", "Surrogate_Mode"])

    for agent_type in agent_types:
        for dim in dims:
            #動態偵測當前維度支援的測試函數
            # （排除因維度過小導致崩潰的函數）
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

            #是否啟用 Q-table 分析統計
            enable_qtable = os.environ.get("ESA_QTABLE_ANALYSIS", "False").lower() == "true"
            #各函數的 Q-table 快照清單
            qtable_300_dict = {f: [] for f in available_funcs}
            qtable_1000_dict = {f: [] for f in available_funcs}

            with ProcessPoolExecutor(max_workers=max_cores) as executor:
                future_to_task = {executor.submit(single_task_worker, task): task for task in all_tasks}

                with Live(make_layout(), refresh_per_second=4, console=console) as live:
                    for future in as_completed(future_to_task):
                        try:
                            func_name, agent_name, run_idx, best_y, t_taken, q300, q1000 = future.result()

                            for csv_file in [history_csv, latest_csv]:
                                with open(csv_file, "a", newline="", encoding="utf-8") as f:
                                    writer = csv.writer(f)
                                    writer.writerow([func_name, dim, agent_name, run_idx, best_y, t_taken, os.environ.get("ESA_RBF_MODE", "simple")])

                            results_by_func[func_name].append({"best_y": best_y, "t_taken": t_taken})
                            if enable_qtable:
                                if q300 is not None:
                                    qtable_300_dict[func_name].append(q300)
                                if q1000 is not None:
                                    qtable_1000_dict[func_name].append(q1000)

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

            #若有收集到 Q-table 且功能啟用，則輸出分析表格
            has_qtable = enable_qtable and any(len(qtable_300_dict[f]) > 0 for f in available_funcs)
            if has_qtable:
                mode_str = os.environ.get("ESA_RBF_MODE", "simple")
                if os.environ.get("ESA_REPLACE_A2_WITH_A5", "False").lower() == "true":
                    mode_str += "_replace_a2_with_a5"
                qtable_log_file = os.path.join(log_dir, f"qtable_analysis_{timestamp}_{agent_type}_{dim}D_{mode_str}.txt")
                with open(qtable_log_file, "w", encoding="utf-8") as qf:
                    title_line = f"=== Q-Table Analysis (Dim: {dim}, Agent: {agent_type}, Mode: {mode_str}) ===\n\n"
                    qf.write(title_line)
                    print("\n" + "="*50)
                    print(title_line.strip())
                    print("="*50)

                    for func_name in available_funcs:
                        q300_list = qtable_300_dict[func_name]
                        q1000_list = qtable_1000_dict[func_name]

                        if len(q300_list) > 0 and len(q1000_list) > 0:
                            avg_q300 = np.mean(q300_list, axis=0)
                            avg_q1000 = np.mean(q1000_list, axis=0)

                            for step_name, avg_q in [("NFE=300", avg_q300), ("NFE=1000", avg_q1000)]:
                                header = f"\n--- {func_name}-{dim}D-{step_name} ---\n"
                                qf.write(header)
                                print(header, end="")

                                n_act = avg_q.shape[1]
                                action_cols = " ".join([f"   a{i+1:<2} " for i in range(n_act)])
                                col_line = f"State | {action_cols}\n"
                                qf.write(col_line)
                                print(col_line, end="")

                                separator = "-" * (8 + 10 * n_act) + "\n"
                                qf.write(separator)
                                print(separator, end="")

                                for row_idx in range(avg_q.shape[0]):
                                    row_vals = " ".join([f"{avg_q[row_idx, col_idx]:8.3f}" for col_idx in range(n_act)])
                                    row_line = f"  s{row_idx+1:<2} | {row_vals}\n"
                                    qf.write(row_line)
                                    print(row_line, end="")
                                qf.write("\n")
                                print("")
                console.print(f"\n[bold green]Q-Table 分析結果已寫入至: [cyan]{qtable_log_file}[/cyan][/bold green]\n")

    console.print("\n[bold green]所有選定維度與 Agent 的實驗全部完畢！[/bold green]", style="bold green")
    console.print(f"- 最新副本已寫入: [cyan]{os.path.abspath(latest_csv)}[/cyan]")
    console.print(f"- 歷史紀錄存檔於: [cyan]{os.path.abspath(history_csv)}[/cyan]")


if __name__ == "__main__":

    #實驗設定區（每次執行前在此調整參數）
    #ESA_RBF_MODE           → "simple" / "hpo" / "clustered_knn" / "mlp"
    #ESA_USE_LBFGS          → "True"（5 策略）/ "False"（4 策略）
    #ESA_REPLACE_A2_WITH_A5 → "True"（L-BFGS 取代 SLS，維持 4 策略）/ "False"
    #ESA_QTABLE_ANALYSIS    → "True"（輸出 Q-Table 平均分析表格）/ "False"（跳過）

    os.environ["ESA_RBF_MODE"] = "hpo"
    os.environ["ESA_USE_LBFGS"] = "False"
    os.environ["ESA_REPLACE_A2_WITH_A5"] = "False"
    os.environ["ESA_QTABLE_ANALYSIS"] = "False"

    run_progressive_experiments(dims=[30, 50, 100], runs=10, agent_types=["qlearning"])
