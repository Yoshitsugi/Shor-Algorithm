# shor_full_animation.py
# 必須：matplotlib のインポート（順序自由）
import matplotlib
# shor_algorithm_animation_final.py

# shor_full_animation.py
# shor_full_animation.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from qutip import Bloch, basis, Qobj, tensor
from sympy import gcd
import math
import random
import matplotlib
import time

matplotlib.rcParams['animation.ffmpeg_path'] = r'C:\ffmpeg\bin\ffmpeg.exe'

MAX_M = 100000

def hadamard_transform(n):
    H = Qobj([[1, 1], [1, -1]], dims=[[2], [2]]) / np.sqrt(2)
    return tensor([H] * n)

def oracle_gate(n_qubits, f_type="balanced"):
    N = 2 ** n_qubits
    U = np.eye(N, dtype=complex)
    if f_type == "balanced":
        for i in range(N):
            if bin(i).count('1') % 2 == 1:
                U[i, i] *= -1
    U_op = Qobj(U)
    U_op.dims = [[2] * n_qubits, [2] * n_qubits]
    return U_op

def run_animation(states_list, title, filename="shor_animation.mp4", view=(60, 30), fps=15):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    bloch = Bloch(fig=fig, axes=ax)
    bloch.vector_color = ['b']
    bloch.point_color = ['b']
    bloch.point_marker = ['o']
    bloch.point_size = [20]
    bloch.show_axes_label = True
    bloch.frame_alpha = 0.1

    frame_save_indices = [0, 299, 449, 599, 719, 899]

    def update(i):
        bloch.clear()
        state_one_qubit = states_list[i].ptrace(0)
        bloch.add_states(state_one_qubit)
        bloch.make_sphere()
        bloch.set_label_convention("xyz")
        bloch.xlabel = ['$x$', '']
        bloch.ylabel = ['$y$', '']
        bloch.zlabel = ['$z$', '']
        bloch.axes.view_init(view[0], view[1])
        bloch.fig.suptitle(f"{title} - Frame {i}", y=0.95)

        if i in frame_save_indices:
            frame_file = f"shor_frame_{i}.png"
            plt.savefig(frame_file, dpi=200)
            print(f"🖼️ フレーム {i} を '{frame_file}' に保存しました。")

    ani = FuncAnimation(fig, update, len(states_list), interval=100, repeat=False)

    try:
        ani.save(filename, fps=fps, dpi=200, writer="ffmpeg")
        print(f"\n🎥 アニメーションファイル '{filename}' が正常に保存されました。")
    except Exception as e:
        print(f"❌ アニメーション保存中にエラーが発生しました: {e}")

    plt.close(fig)

def interpolate_states(start, end, num_steps):
    states = []
    start_vec = start.full().flatten()
    end_vec = end.full().flatten()

    for i in range(num_steps):
        ratio = i / num_steps
        vec = (1 - ratio) * start_vec + ratio * end_vec
        psi = Qobj(vec.reshape((2, 1)), dims=[[2], [1]])
        psi = psi.unit()
        states.append(psi)

    return states

def find_period(a, N, max_tries=100000):
    a, N = int(a), int(N)
    if gcd(a, N) != 1:
        return None
    seen = {}
    x = 1
    count = 0
    while count < max_tries:
        x = (x * a) % N
        if x in seen:
            return count - seen[x]
        seen[x] = count
        count += 1
    return None

def shor(N, max_attempts=20, max_period_tries=100000):
    N = int(N)
    if N <= 2:
        return None, None
    if N % 2 == 0:
        return 2, N // 2

    for _ in range(max_attempts):
        a = random.randint(2, N - 1)
        g = gcd(int(a), int(N))
        if g != 1 and g != N:
            return int(g), int(N // g)
        r = find_period(a, N, max_period_tries)
        if r and r % 2 == 0:
            ar2 = pow(int(a), int(r) // 2, int(N))
            if ar2 != N - 1:
                p = gcd(int(ar2) - 1, int(N))
                q = N // int(p)
                if int(p) not in (0, 1, N) and int(q) not in (0, 1, N):
                    return int(p), int(q)
    return None, None

def full_prime_factors(n):
    factors = []

    def _factor(n):
        n = int(n)
        if n <= 1:
            return
        if is_prime(n):
            factors.append(n)
            return
        p, q = shor(n)
        if p and q:
            _factor(p)
            _factor(q)
        else:
            factors.append(n)

    _factor(n)
    return factors

def is_prime(n):
    n = int(n)
    if n < 2:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

def show_final_probability(states_list, filename="probability_distribution.png"):
    final_probs = np.abs(states_list[-1].full()) ** 2
    final_probs = final_probs.flatten().tolist()
    plt.figure()
    plt.bar(["|0⟩", "|1⟩"], final_probs, color='skyblue')
    plt.title("Final Probability Distribution")
    plt.ylabel("Probability")
    plt.ylim(0, 1)
    plt.savefig(filename)
    print(f"📊 確率分布画像 '{filename}' を保存しました。")
    plt.show()

def main():
    start_time = time.time()

    print("=== Shor のアルゴリズムによる素因数分解 ===")
    try:
        M = int(input(f"正の整数 M を入力してください（2 ≦ M ≦ {MAX_M}）> "))
    except ValueError:
        print("❌ 整数を入力してください。")
        return

    if M < 2 or M > MAX_M:
        print(f"❌ 2以上かつ {MAX_M} 以下の整数を入力してください。")
        return

    n_qubits = 1
    psi0 = basis(2, 0)
    psi0.dims = [[2], [1]]

    H = hadamard_transform(n_qubits)
    psi_plus = H * psi0
    oracle = oracle_gate(n_qubits, f_type="balanced")
    psi_marked = oracle * psi_plus
    psi_final = H * psi_marked

    num_transition_steps = 300
    states_list = []
    states_list += interpolate_states(psi0, psi_plus, num_transition_steps)
    states_list += interpolate_states(psi_plus, psi_marked, num_transition_steps)
    states_list += interpolate_states(psi_marked, psi_final, num_transition_steps)

    print(f"\n{M} の素因数分解を開始します...")
    print("アニメーションを生成中...\n")

    filename = "shor_animation.mp4"
    run_animation(states_list, f"Factorizing {M}", filename=filename, view=(60, 30), fps=15)

    all_factors = full_prime_factors(M)
    if all_factors:
        print(f"\n✅ {M} の素因数: {all_factors}")
    else:
        print(f"\n❌ {M} の因数分解に失敗しました。")

    show_final_probability(states_list)

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\n✅ No. Errorで終了（実行時間: {elapsed:.2f} 秒）")

if __name__ == "__main__":
    main()




